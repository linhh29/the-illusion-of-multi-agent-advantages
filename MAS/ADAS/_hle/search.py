import os
import ast
import sys
import copy
import json
import glob
import argparse
import requests
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

import backoff
import numpy as np
import openai
from tqdm import tqdm

import datetime

import threading

from hle_prompt import get_init_archive, get_prompt, get_reflexion_prompt
from utils import random_id, load_questions, score_hle, bootstrap_confidence_interval

load_dotenv()  # Load environment variables from .env file


##### API SETUP ######
# OpenAI API setup
client = openai.OpenAI(
    # api_key="sk-xxxx"
    # api_key="sk-xxxx"
    api_key=os.getenv("OPENAI_API_KEY")
)

# IBM API setup
# 1/ Get IBM Cloud IAM token
url = 'https://iam.cloud.ibm.com/identity/token'
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
data = {
    'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
    'apikey': 'xxxx'
}
response = requests.post(url, headers=headers, data=data)
ibm_token = response.json().get('access_token')
# ibm_token = "<jwt_token_redacted>"
url = "https://xx.ml.cloud.ibm.com/ml/v1/text/chat?version=xxxx"
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {ibm_token}'
}
##### /API SETUP END ######

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

# current_time = datetime.datetime.now().strftime("%m%d-%H%M%S-%f")
CURRENT_DAY = datetime.datetime.now().strftime("%m%d-%H%M%S")
EXPR_NAME = ''

class GLOBALS:
    num_api_calls = 0
    num_complete_tokens = 0
    num_prompt_tokens = 0
    num_total_tokens = 0
    lock = threading.Lock()


def to_dict(content):
    # try to parse str content to dict
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # back to safe Python literal eval (handles single quotes)
            return ast.literal_eval(content)
        except Exception as e:
            raise ValueError(f"Failed to parse content: {e}")
        

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5,
        seed=42,
        max_new_tokens=4096, # 4096 for gpt-3.5/4, 32768 for gpt-5
        gpt5_reasoning_effort="medium", # "minimal", "low", "medium", "high"
        gpt5_verbosity="low",
):
    token_usage = []
    if model.startswith('gpt-3.5') or model.startswith('gpt-4'):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ],
            temperature=temperature, seed=seed,
            max_tokens=max_new_tokens, stop=None, response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        token_usage = [response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens]
        # == Cost info ==
        # cost_4o = response.usage.completion_tokens/1000000*15 + response.usage.prompt_tokens/1000000*5
        # cost_3.5 = response.usage.completion_tokens/1000000*2 + response.usage.prompt_tokens/1000000*1.5
        # cost_5nano = response.usage.completion_tokens/1000000*0.4 + response.usage.prompt_tokens/1000000*0.05
    elif model.startswith('gpt-5'): 
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ],
            seed=seed,
            max_completion_tokens=max_new_tokens, #16384, 32768, 8192
            stop=None, 
            response_format={"type": "json_object"},
            reasoning_effort=gpt5_reasoning_effort,
            verbosity=gpt5_verbosity,
        ) # <- temperature is no longer needed in gpt5; needs longer completion length otherwise risk empty response
        content = response.choices[0].message.content
        if content == '':
            print('Empty response from gpt-5!')
        token_usage = [response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens]
    elif model == "openai/gpt-oss-120b":
        payload = {
            "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": msg},
                    ],
            "project_id": "xxxx-xxxx-xxxx-xxxx-xxxx",
            "model_id": "openai/gpt-oss-120b",
            "frequency_penalty": 0,
            "max_tokens": max_new_tokens,
            "presence_penalty": 0,
            "temperature": 0.5,
            "top_p": 1,
            "seed": seed,
            "stop": []
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload)).json()
        if 'errors' in response:
            print(f"IBM API Error: {response['errors'][0]['message']}")
            return {}
        else:
            content = response['choices'][0]['message']['content']
    elif model == 'gemini-2.5-pro':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from vertex_helper import get_openai_client, VERTEX_MODEL_NAME
        client = get_openai_client()
        response = client.chat.completions.create(
            model=VERTEX_MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=int(os.environ.get("MAX_NEW_TOKENS", "4096")),
            temperature=0.0)
        content = response.choices[0].message.content
        token_usage = [response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens]
    else:
        raise NotImplementedError(f"model {model} not implemented")
    
    json_dict = to_dict(content)
    return json_dict, token_usage


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8,
        seed=42,
        max_new_tokens=4096, # 4096 for gpt-3.5/4, 32768 for gpt-5
        gpt5_reasoning_effort="medium", # "minimal", "low", "medium", "high"
        gpt5_verbosity="low",
):
    token_usage = [] # complete_token, prompt_token, total_token
    if model.startswith('gpt-3.5') or model.startswith('gpt-4'):
        response = client.chat.completions.create(
            model=model,
            messages=msg_list,
            temperature=temperature, 
            seed=seed,
            max_tokens=max_new_tokens, stop=None, response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        token_usage = [response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens]
    elif model.startswith('gpt-5'):
        response = client.chat.completions.create(
                model=model,
                messages=msg_list,
                max_completion_tokens=max_new_tokens,
                seed=seed,
                reasoning_effort=gpt5_reasoning_effort,
                verbosity=gpt5_verbosity,
                stop=None, 
                response_format={"type": "json_object"}
            )
        content = response.choices[0].message.content
        token_usage = [response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens]
    elif model == "openai/gpt-oss-120b":
        payload = {
            "messages": msg_list,
            "project_id": "xxxx-xxxx-xxxx-xxxx-xxxx",
            "model_id": "openai/gpt-oss-120b",
            "frequency_penalty": 0,
            "max_tokens": max_new_tokens,
            "presence_penalty": 0,
            "temperature": 0.5,
            "top_p": 1,
            "seed": seed,
            "stop": []
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload)).json()
        if 'errors' in response:
            print(f"IBM API Error: {response['errors'][0]['message']}")
            return {}
        else:
            content = response['choices'][0]['message']['content']
    elif model == 'gemini-2.5-pro':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from vertex_helper import get_openai_client, VERTEX_MODEL_NAME
        client = get_openai_client()
        response = client.chat.completions.create(
            model=VERTEX_MODEL_NAME, 
            messages=msg_list,
            response_format={"type": "json_object"},
            max_tokens=int(os.environ.get("MAX_NEW_TOKENS", "4096")),
            temperature=0.0)
        content = response.choices[0].message.content
        token_usage = [response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens]
    else:
        raise NotImplementedError(f"model {model} not implemented")

    json_dict = to_dict(content)
    return json_dict, token_usage


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', 
                #  model='gpt-3.5-turbo-0125',
                #  model='gpt-5-nano',
                #  model="openai/gpt-oss-120b",
                #  model="gpt-4o-2024-05-13",
                #   model="gpt-4o",
                #   model="gpt-5",
                model=os.getenv("AGENT_BASE_MODEL", "gpt-4o"),
                temperature=0.5,
                seed=int(os.getenv("SEED", 42)),
                max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 4096)),
                gpt5_reasoning_effort=os.getenv("GPT5_REASONING_EFFORT", "none"),
                gpt5_verbosity=os.getenv("GPT5_VERBOSITY", "none"),
                 ) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id() # give each instance a unique id
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.gpt5_reasoning_effort = gpt5_reasoning_effort
        self.gpt5_verbosity = gpt5_verbosity

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY an integer. DO NOT return anything other than the integer answer." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json, token_usage = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature, self.seed,
                                                                    self.max_new_tokens, self.gpt5_reasoning_effort, self.gpt5_verbosity)            
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
            token_usage = []

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        if token_usage != []:
            assert len(token_usage) == 3, "token_usage should be a list of 3 elements"
            for _tok_name, _tok_value in zip(['completion_tokens', 'prompt_tokens', 'total_tokens'], token_usage):
                info = Info(_tok_name, self.__repr__(), _tok_value, iteration_idx)
                output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        infos = self.query(input_infos, instruction, iteration_idx=iteration_idx)
        with GLOBALS.lock: # lock secures one update in multi-threading
            GLOBALS.num_api_calls += 1
        token_counters = {
            'completion_tokens': 'num_complete_tokens',
            'prompt_tokens': 'num_prompt_tokens',
            'total_tokens': 'num_total_tokens',
        }
        filtered_infos = []
        for _info in infos:
            counter_name = token_counters.get(_info.name)
            if counter_name is None:
                filtered_infos.append(_info)
                continue
            with GLOBALS.lock:
                setattr(
                    GLOBALS,
                    counter_name,
                    getattr(GLOBALS, counter_name) + _info.content,
                )
        return filtered_infos


class AgentSystem():
    def __init__(self) -> None:
        pass


def search(args):
    global EXPR_NAME
    EXPR_NAME = f"{args.expr_name}_{CURRENT_DAY}"
    file_path = os.path.join(args.save_dir, f"{EXPR_NAME}_seed-0-{args.seed}_run_search.json") # seed-subsetseed-modelseed
    print("Saving research in file_path:", file_path)

    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    # 1/ Run methods in archive
    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list, error_list, api_call, complete_token, prompt_token, total_token = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['acc_list'] = ', '.join([str(acc) for acc in acc_list])
        solution['error_list'] = error_list
        solution['fitness'] = fitness_str
        solution['API_call'] = api_call # new
        solution['API_call_per_question'] = api_call / len(acc_list) # new
        solution['complete_token'] = complete_token # new
        solution['prompt_token'] = prompt_token # new
        solution['total_token'] = total_token # new

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, ensure_ascii=False, indent=4)
    print('Finished evaluating initial archive.\n')
    # return

    # 2/ Generate new solutions
    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        if args.model == 'gpt-5':
            agent_model = 'gpt-5' # gpt-5-nano
        elif args.model == 'openai/gpt-oss-120b':
            agent_model = 'openai/gpt-oss-120b'
        elif args.model == 'gemini-2.5-pro':
            agent_model = 'gemini-2.5-pro'
        elif args.model in ['gpt-4o-2024-05-13', 'gpt-4o']:
            agent_model = 'gpt-4o'
        else:
            raise NotImplementedError
        
        meta_api_call = 0
        meta_complete_token = 0
        meta_prompt_token = 0
        meta_total_token = 0

        system_prompt, prompt = get_prompt(archive, agent_model, args.seed, args.max_new_tokens, args.gpt5_reasoning_effort, args.gpt5_verbosity)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution, meta_token_usage = get_json_response_from_gpt_reflect(msg_list, args.model, 
                                                               seed=int(args.seed), 
                                                               max_new_tokens=int(args.max_new_tokens),
                                                               gpt5_reasoning_effort=args.gpt5_reasoning_effort, 
                                                               gpt5_verbosity=args.gpt5_verbosity)
            meta_api_call += 1
            if meta_token_usage != []: # in case of error in LLM response parsing, skip token usage counting
                meta_complete_token += meta_token_usage[0]
                meta_prompt_token += meta_token_usage[1]
                meta_total_token += meta_token_usage[2]

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            # Reflexion 1
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution, meta_token_usage = get_json_response_from_gpt_reflect(msg_list, args.model,
                                                                                 seed=int(args.seed), 
                                                                                 max_new_tokens=int(args.max_new_tokens),
                                                                                 gpt5_reasoning_effort=args.gpt5_reasoning_effort, 
                                                                                 gpt5_verbosity=args.gpt5_verbosity)
            meta_api_call += 1
            if meta_token_usage != []:
                meta_complete_token += meta_token_usage[0]
                meta_prompt_token += meta_token_usage[1]
                meta_total_token += meta_token_usage[2]
            
            # Reflexion 2
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution, meta_token_usage = get_json_response_from_gpt_reflect(msg_list, args.model,
                                                                                 seed=int(args.seed), 
                                                                                 max_new_tokens=int(args.max_new_tokens),
                                                                                 gpt5_reasoning_effort=args.gpt5_reasoning_effort, 
                                                                                 gpt5_verbosity=args.gpt5_verbosity)
            meta_api_call += 1
            if meta_token_usage != []:
                meta_complete_token += meta_token_usage[0]
                meta_prompt_token += meta_token_usage[1]
                meta_total_token += meta_token_usage[2]

        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            n -= 1
            continue

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list, error_list, api_call, complete_token, prompt_token, total_token = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During new solution evaluation (debugging):", end=' ')
                print(e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution, meta_token_usage = get_json_response_from_gpt_reflect(msg_list, args.model, 
                                                                       seed=int(args.seed), 
                                                                       max_new_tokens=int(args.max_new_tokens),
                                                                       gpt5_reasoning_effort=args.gpt5_reasoning_effort, 
                                                                       gpt5_verbosity=args.gpt5_verbosity)
                    meta_api_call += 1
                    if meta_token_usage != []:
                        meta_complete_token += meta_token_usage[0]
                        meta_prompt_token += meta_token_usage[1]
                        meta_total_token += meta_token_usage[2]
                except Exception as e:
                    print("During LLM generate new solution (debugging):")
                    print(e)
                    continue
                continue
        if not acc_list:
            n -= 1
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['acc_list'] = ', '.join([str(acc) for acc in acc_list])
        next_solution['error_list'] = error_list
        next_solution['generation'] = n + 1
        next_solution['API_call'] = api_call # Sub-agent level API usage (validation set)
        next_solution['API_call_per_question'] = api_call / len(acc_list)
        next_solution['complete_token'] = complete_token
        next_solution['prompt_token'] = prompt_token
        next_solution['total_token'] = total_token
        next_solution['meta_API_call'] = meta_api_call # meta-level API usage
        next_solution['meta_complete_token'] = meta_complete_token
        next_solution['meta_prompt_token'] = meta_prompt_token
        next_solution['meta_total_token'] = meta_total_token

        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate2(args):
    # li: only evaluate the best performing generated agent in validation set, save time and cost
    # file_path = os.path.join(args.save_dir, f"{EXPR_NAME}_seed-0-{args.seed}_run_search.json")
    file_path = glob.glob(args.save_dir + f"/{EXPR_NAME}_*_run_search.json")[0]
    assert os.path.exists(file_path), f"{file_path} does not exist."

    eval_file_path = str(os.path.join(args.save_dir, f"{EXPR_NAME}_seed-0-{args.seed}_run_search.json")).strip(".json") + f"_evaluate2_{args.model}.json"
    if os.path.exists(eval_file_path):
        print(f"Evaluation file {eval_file_path} already exists. Skipping evaluation.")
        return

    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    
    # extract inital archive and the best solution
    new_archive = []
    best_sol = None
    best_median = -1

    for sol in archive:
        if sol['generation'] == "initial":
            if sol['name'] in ["Chain-of-Thought", "Self-Consistency with Chain-of-Thought"]:
                new_archive.append(sol)
        else:
            # pass
            try:
                median_str = sol['fitness'].split('Median:')[-1].strip().strip('%').strip()
                median = float(median_str)
            except Exception as e:
                median_str = sol['fitness'].split('Median:')[-1].strip().split('|')[0].strip().strip('%').strip()
                median = float(median_str)
            if median > best_median:
                best_median = median
                best_sol = sol
    new_archive.append(best_sol)
    print(f"best_sol: {best_sol['name']}, generation: {best_sol['generation']}, fitness: {best_sol['fitness']}")
    assert len(new_archive) == 3 # CoT, CoT-SC + 1 best

    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while (current_idx < len(new_archive)):
        sol = new_archive[current_idx]
        print(f"current_gen: {sol['generation']}, name: {sol['name']}, current_idx: {current_idx+1}/{len(new_archive)}")
        current_idx += 1
        try:
            acc_list, test_error_list, api_call, complete_token, prompt_token, total_token = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        sol['test_error_list'] = test_error_list
        sol['test_acc_list'] = ', '.join([str(acc) for acc in acc_list])
        sol['test_API_call'] = api_call # new, average per question
        sol['test_API_call_per_question'] = api_call / len(acc_list) # new, average per question
        sol['test_complete_token'] = complete_token # new
        sol['test_prompt_token'] = prompt_token # new
        sol['test_total_token'] = total_token # new
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)


def evaluate_forward_fn(args, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    # set seed 0 for valid set
    questions = load_questions(args.subdata_name, seed=0)

    if SEARCHING_MODE:
        val_questions = questions[:args.valid_size] * args.n_repreat
    else:
        val_questions = questions[args.valid_size:] * args.n_repreat

    questions = [example['inputs'] for example in val_questions]
    answers = [example['targets'] for example in val_questions]

    print(f"problem length: {len(val_questions)}")
    max_workers = min(len(val_questions), args.max_workers) if args.multiprocessing else 1

    task_queue = []
    for q in questions:
        task_content = f"Solve this math problem:\n{q}\n"
        taskInfo = Info('task', 'User', task_content, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()

    acc_list = []
    error_list = []
    api_call = 0 # new
    complete_token = 0
    prompt_token = 0
    total_token = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))
    # li: single thread debug
    # results = []
    # for _task in tqdm(task_queue):
    #     _result = agentSystem.forward(_task)
    #     results.append(_result)

    # Evaluate results
    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, Info):
                extracted_answer = res.content
            else:
                extracted_answer = res
            if extracted_answer == '':
                print(f"error in q {q_idx}: empty answer")
                acc_list.append(0)
                error_list.append(f"q {q_idx}: empty answer")
                continue
            correct_answer = answers[q_idx]
            correct = score_hle(correct_answer, extracted_answer)
        except Exception as e:
            print(f"error in q {q_idx}: {e}")
            acc_list.append(0)
            error_list.append(f"q {q_idx}: {e}")
            continue
        acc_list.append(1 if correct else 0)

    # Report API calls and token usage
    print(f"Number of API calls: {GLOBALS.num_api_calls}")
    print(f"Number of API calls / question: {GLOBALS.num_api_calls / len(results)}")
    api_call = GLOBALS.num_api_calls
    GLOBALS.num_api_calls = 0
    print(f"Number of complete tokens: {GLOBALS.num_complete_tokens}")
    complete_token = GLOBALS.num_complete_tokens
    GLOBALS.num_complete_tokens = 0
    print(f"Number of prompt tokens: {GLOBALS.num_prompt_tokens}")
    prompt_token = GLOBALS.num_prompt_tokens
    GLOBALS.num_prompt_tokens = 0
    print(f"Number of total tokens: {GLOBALS.num_total_tokens}")
    total_token = GLOBALS.num_total_tokens
    GLOBALS.num_total_tokens = 0
    
    return acc_list, error_list, api_call, complete_token, prompt_token, total_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdata_name', type=str, default='hle_math200int',
                        choices=['hle_math200', 'hle_math200int'])
    parser.add_argument('--valid_size', type=int, default=32)
    parser.add_argument('--n_repreat', type=int, default=3)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='li_results/')
    parser.add_argument('--expr_name', type=str, default="hle_gpt5")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model', #li: means meta-agent, sub-agent is hard-coded in LLMAgentBase
                        type=str,
                        default='gpt-5',
                        choices=['gpt-5', 'openai/gpt-oss-120b', 'gemini-2.5-pro',
                                 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 
                                 'gpt-4o-2024-05-13', 'gpt-4o', 'gpt-4o-mini'])
    parser.add_argument('--seed', type=int, default=0, choices=[0, 1, 42])
    parser.add_argument('--max_new_tokens', type=int, default=4096) #4096 for gpt-3.5/4, 32768 for gpt-5
    parser.add_argument('--gpt5_reasoning_effort', type=str, default='medium', choices=['minimal', 'low', 'medium', 'high'])
    parser.add_argument('--gpt5_verbosity', type=str, default='low', choices=['low', 'medium', 'high'])
    parser.add_argument('--search_only', action='store_true', default=False) #li: newly added
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--search_exprname', type=str, default='') #li: newly added, only for eval_only to specify the 
    
    args = parser.parse_args()

    # only gpt-5
    if args.model == 'gpt-5':
        args.expr_name = args.expr_name + f'-{args.gpt5_reasoning_effort}-{args.gpt5_verbosity}'
    
    # search only
    if args.search_only:
        SEARCHING_MODE = True
        search(args)
        exit(0)
    
    # evaluate only
    if args.eval_only:
        SEARCHING_MODE = False
        # EXPR_NAME = 'gpqa_gpt-4o_0915-230502' #li: manually set the expr_name
        if args.search_exprname != '':
            EXPR_NAME = args.search_exprname
        else:
            raise ValueError("Please provide --search_exprname for eval_only mode.")
        evaluate2(args) # li: only evaluate initial and the best performing generated agent in validation set
        exit(0)
  
    # run both search and evaluate
    print(args)
    SEARCHING_MODE = True
    search(args)
    SEARCHING_MODE = False
    evaluate2(args)

    # Usage example [old]:
    # python _hle/search.py --subdata_name hle_math200int --save_dir li_results/ --n_repreat 3 --expr_name hle200int_gpt5 --n_generation 30 --model gpt-5 --seed 0
    # python _hle/search.py --subdata_name hle_math200int --save_dir li_results/ --n_repreat 3 --expr_name hle200int_gpt5 --n_generation 30 --model gpt-5 --eval_only --eval_expr hle200int_gpt5_0902-170221 --seed 0
    # python _hle/search.py --subdata_name hle_math200int --save_dir li_results/ --n_repreat 3 --expr_name hle200int_gpt5 --n_generation 30 --model gpt-5 --eval_only --eval_expr hle200int_gpt5_0902-172812 --seed 1
    # python _hle/search.py --subdata_name hle_math200int --save_dir li_results/ --n_repreat 3 --expr_name hle200int_gpt5 --n_generation 30 --model gpt-5 --eval_only --eval_expr hle200int_gpt5_0902-173123 --seed 2


    # 1018
    # USE THE FOLLOWING COMMANDS IN TERMINAL:
    # [gpt-4o]
    # export OPENAI_API_KEY=sk-xxxx
    # export AGENT_BASE_MODEL=gpt-4o
    # export SEED=0/1/42
    # export MAX_NEW_TOKENS=8192
    # export SEARCH_EXPRNAME=hle_gpt-4o_1019-161616
    # export SEARCH_EXPRNAME=hle_gpt-4o_1019-162203
    # export SEARCH_EXPRNAME=hle_gpt-4o_1019-162207
    # echo python _hle/search.py --save_dir li_results/ --n_repreat 2 --expr_name hle_${AGENT_BASE_MODEL} --n_generation 30 --model ${AGENT_BASE_MODEL} --seed ${SEED} --max_new_tokens ${MAX_NEW_TOKENS} --eval_only --search_exprname ${SEARCH_EXPRNAME}


    # [gpt-5]
    # export OPENAI_API_KEY=sk-xxxx
    # export AGENT_BASE_MODEL=gpt-5
    # export SEED=0
    # export MAX_NEW_TOKENS=32768
    # export GPT5_REASONING_EFFORT=medium
    # export GPT5_VERBOSITY=low
    # echo python _hle/search.py --save_dir li_results/ --n_repreat 2 --expr_name hle_${AGENT_BASE_MODEL} --n_generation 15/30 --model ${AGENT_BASE_MODEL} --seed ${SEED} --max_new_tokens ${MAX_NEW_TOKENS} --gpt5_reasoning_effort ${GPT5_REASONING_EFFORT} --gpt5_verbosity ${GPT5_VERBOSITY}

    # /eval_only/
    # export SEED=1/42
    # export SEARCH_EXPRNAME=hle_gpt-5-medium-low_1019-225414
    # echo python _hle/search.py --save_dir li_results/ --n_repreat 2 --expr_name hle_${AGENT_BASE_MODEL} --n_generation 15/30 --model ${AGENT_BASE_MODEL} --seed ${SEED} --max_new_tokens ${MAX_NEW_TOKENS} --gpt5_reasoning_effort ${GPT5_REASONING_EFFORT} --gpt5_verbosity ${GPT5_VERBOSITY} --eval_only --search_exprname ${SEARCH_EXPRNAME}
