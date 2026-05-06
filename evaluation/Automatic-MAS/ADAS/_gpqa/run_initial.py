import argparse
import copy
import json
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm

import datetime

client = openai.OpenAI(
    api_key="sk-xxxx"
)

from utils import load_questions, random_id, bootstrap_confidence_interval

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
# SEARCHING_MODE = True

# current_time = datetime.datetime.now().strftime("%m%d-%H%M%S-%f")
CURRENT_DAY = datetime.datetime.now().strftime("%m%d-%H%M%S")
EXPR_NAME = ""

# global BASE_MODEL
# BASE_MODEL = 'gpt-3.5-turbo-0125'
# # BASE_MODEL = 'gpt-4.1-nano'
# # BASE_MODEL = 'gpt-5-nano'
BASE_MODEL = ""

COT = {
    "thought": "By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.",
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Instruction for the Chain-of-Thought (CoT) approach
    # It is an important practice that allows the LLM to think step by step before solving the task.
    cot_instruction = "Please think step by step and then solve the task."

    # Instantiate a new LLM agent specifically for CoT
    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # Prepare the inputs for the CoT agent
    # The input should be a list of Info, and the first one is often the taskInfo
    cot_agent_inputs = [taskInfo]

    # Get the response from the CoT agent
    thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)

    # Return only the final answer
    return answer
"""
}

COT_SC = {"thought": "While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling.",
          "name": "Self-Consistency with Chain-of-Thought",
          "code": """def forward(self, taskInfo):
    # Instruction for step-by-step reasoning
    cot_instruction = "Please think step by step and then solve the task."
    N = 5 # Number of CoT agents

    # Initialize multiple CoT agents with a higher temperature for varied reasoning
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]

    # Majority voting function to select the most common answer
    from collections import Counter
    def majority_voting(answers):
        return Counter(answers).most_common(1)[0][0]
    
    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.append(answer.content)

    # Ensembling the answers from multiple CoT agents
    answer = majority_voting(possible_answers)
    return answer  
"""
          }


def get_init_archive():
    return [COT, COT_SC]


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    if model.startswith('gpt-3.5') or model.startswith('gpt-4'):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ],
            temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
        )
        # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    
    elif model.startswith('gpt-5'): 
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": msg},
                ],
                max_completion_tokens=25000, stop=None, response_format={"type": "json_object"}
            ) # <- temperature is no longer needed in gpt5
        except Exception as e:
            print(f"\nError in gpt response: {e}")
            response = {}
    else:
        raise NotImplementedError(f"model {model} not implemented")
    
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', 
                #  model='gpt-3.5-turbo-0125',
                 model=None,
                #  model='gpt-5-nano',
                 temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model if model is not None else BASE_MODEL
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
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
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            print(e)
            # if "maximum context length" in str(e) and SEARCHING_MODE:
            #     raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self) -> None:
        pass


def search(args):
    global EXPR_NAME, BASE_MODEL
    EXPR_NAME = f"{args.expr_name}_{CURRENT_DAY}"
    BASE_MODEL = args.base_model
    file_path = os.path.join(args.save_dir, f"{EXPR_NAME}_seed-{args.seed}_run_archive_{BASE_MODEL}.json")
    print("Saving initial archieve in file_path:", file_path)

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
        print(f"============Initial Archive (valid set): {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"], SEARCHING_MODE=True)
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str

        print(f"============Initial Archive (test set): {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"], SEARCHING_MODE=False)
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['test_fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)
    print('Finished evaluating initial archive.')


def evaluate_forward_fn(args, forward_str, SEARCHING_MODE=False):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace) # exec. places the forward str func in the provided namespace dictionary
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func) #li: assign the function as a method named "forward" to the class AgentSystem

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # set seed 0 for valid set
    questions = load_questions(args.data_filename, seed=args.seed) #li: changed to args.seed
    if SEARCHING_MODE:
        val_questions = questions[:args.valid_size] * args.n_repreat
    else:
        val_questions = questions[args.valid_size:] * args.n_repreat

    print(f"problem length: {len(val_questions)}")
    max_workers = min(len(val_questions), args.max_workers) if args.multiprocessing else 1

    task_queue = []
    for q in val_questions:
        task_content = f"What is the correct answer to this question: {q.question}" \
                       + f"\n\nChoices:\n(A) {q.choice1}\n(B) {q.choice2}\n(C) {q.choice3}\n(D) {q.choice4}"
        taskInfo = Info('task', 'User', task_content, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()

    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))
    # li: single thread debug
    # results = []
    # for _task in tqdm(task_queue):
    #     results.append(agentSystem.forward(_task))
    
    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, str) and res in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[res]
            elif 'A)' in res:
                predicted_idx = 0
            elif 'B)' in res:
                predicted_idx = 1
            elif 'C)' in res:
                predicted_idx = 2
            elif 'D)' in res:
                predicted_idx = 3
            elif isinstance(res, list):
                try_res = res[1]
                predicted_idx = LETTER_TO_INDEX[try_res.content]
            elif res.content in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[res.content]
            elif 'A)' in res.content:
                predicted_idx = 0
            elif 'B)' in res.content:
                predicted_idx = 1
            elif 'C)' in res.content:
                predicted_idx = 2
            elif 'D)' in res.content:
                predicted_idx = 3
            else:
                print(f"error in q {q_idx}: ", end='')
                if res == '':
                    print("empty response")
                else:
                    print("response:", res)
                acc_list.append(0)
                continue
        except Exception as e:
            acc_list.append(0)
            continue

        if predicted_idx == val_questions[q_idx].correct_index:
            acc_list.append(1)
        else:
            acc_list.append(0)
    print(f"acc: {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default='dataset/gpqa_diamond.csv')
    parser.add_argument('--valid_size', type=int, default=32)
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--save_dir', type=str, default='li_results/') #-> changed to li_results/
    parser.add_argument('--expr_name', type=str, default="gpqa_init")
    parser.add_argument('--seed', type=int, default=0) #li: newly added
    parser.add_argument('--base_model', type=str, default="gpt-3.5-turbo-0125",
                        choices=['gpt-3.5-turbo-0125', 'gpt-4.1-nano', 'gpt-5-nano']) #li: newly added

    args = parser.parse_args()
    
    search(args)

    # Usage example:
    # python _gpqa/run_initial.py --save_dir li_results/_results_initial --expr_name gpqa_init --n_repreat 3 --seed 0 --base_model gpt-3.5-turbo-0125
    