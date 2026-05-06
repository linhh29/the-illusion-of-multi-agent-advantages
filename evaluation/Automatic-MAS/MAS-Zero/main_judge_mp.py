import argparse
import ast
import copy
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common import EQUALITY_TEMPLATE, MCQ_EQUALITY_TEMPLATE, ANSWER_PATTERN, BROWSECOMP_PLUS_TEMPLATE
from llm_judge import self_verifier_list_wise
from sampler.chat_completion_sampler import (
    ChatCompletionSampler,
)
from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.together_completion_sampler import ChatCompletionSampler as ToChatCompletionSampler
from sampler.vllm_completion_sampler import ChatCompletionSampler as VllmChatCompletionSampler
from swe_utils import run_swebench_evaluation

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def is_smfr_dataset(dataset_name: str) -> bool:
    if not dataset_name:
        return False
    name = dataset_name.lower().strip()
    return name == "workflow_search/smfr" or "smfr_synthetic" in name or "smfr" in name


_SMFR_EVAL_DIR = Path(__file__).resolve().parent / "smfr_synthetic_dataset" / "evaluate"
if _SMFR_EVAL_DIR.exists():
    sys.path.append(str(_SMFR_EVAL_DIR))
try:
    from safe_code_executor import SafeCodeExecutor
except Exception:
    SafeCodeExecutor = None


def _extract_smfr_answer_blob(response_text: str) -> str:
    if not response_text:
        return ""
    lowered = response_text.lower()
    idx = lowered.rfind("answer:")
    if idx == -1:
        return response_text.strip()
    return response_text[idx + len("answer:"):].strip()


def _try_parse_mapping(text: str):
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_smfr_model_output(response_text: str) -> dict:
    answer_blob = _extract_smfr_answer_blob(response_text)
    parsed = _try_parse_mapping(answer_blob)
    if parsed is None:
        start = answer_blob.find("{")
        end = answer_blob.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = _try_parse_mapping(answer_blob[start:end + 1])

    if isinstance(parsed, dict):
        if isinstance(parsed.get("output"), dict):
            output_block = parsed["output"]
            return {
                "answer": output_block.get("answer"),
                "code": output_block.get("code"),
                "raw_answer": answer_blob,
            }
        return {
            "answer": parsed.get("answer", parsed.get("final_answer")),
            "code": parsed.get("code"),
            "raw_answer": answer_blob,
        }

    code = None
    code_match = re.search(r"```(?:python)?\n(.*?)```", answer_blob, re.DOTALL | re.IGNORECASE)
    if code_match:
        code = code_match.group(1).strip()
        answer_blob = (answer_blob[:code_match.start()] + answer_blob[code_match.end():]).strip()

    if code is None:
        code_marker = re.search(r"(?is)\bcode\s*:\s*", answer_blob)
        if code_marker:
            code = answer_blob[code_marker.end():].strip()
            answer_blob = answer_blob[:code_marker.start()].strip()

    return {
        "answer": None,
        "code": code,
        "raw_answer": answer_blob,
    }


def _extract_reference_answer(reference):
    if isinstance(reference, dict):
        ref = reference.get("answer", [])
        if isinstance(ref, dict):
            ref = ref.get("answer", [])
        return ref or []
    return reference or []


def _evaluate_direct_answer(model_answer, reference_answer):
    if not reference_answer:
        return False, 0

    partial_count = 0
    for name in reference_answer:
        if isinstance(model_answer, list):
            if name in model_answer:
                partial_count += 1
        else:
            if name in str(model_answer):
                partial_count += 1

    return partial_count == len(reference_answer), partial_count


def _evaluate_code_output(code, reference_answer, executor):
    if not code or executor is None:
        return False, False, True

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        exec_result = executor.execute(code, inputs={})
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    if not exec_result.get("success", False):
        return False, False, True

    result = exec_result.get("result")
    code_answer = None
    if isinstance(result, dict):
        code_answer = result.get("answer")
    else:
        code_answer = result

    if code_answer is None:
        return False, False, True

    if isinstance(code_answer, str):
        if code_answer in reference_answer:
            is_full = len(reference_answer) == 1
            return is_full, True, False
        return False, False, False

    if isinstance(code_answer, list):
        if set(code_answer) == set(reference_answer):
            return True, False, False

    return False, False, False


def _evaluate_smfr_candidate(correct_answer, candidate, executor):
    reference_answer = _extract_reference_answer(correct_answer)

    model_answer = candidate.get("answer")
    if isinstance(model_answer, dict) and "answer" in model_answer:
        model_answer = model_answer["answer"]
    if model_answer is None:
        model_answer = candidate.get("raw_answer", "")

    direct_full, partial_count = _evaluate_direct_answer(model_answer, reference_answer)
    direct_partial = partial_count > 0 and not direct_full

    code_full, code_partial, code_failed = _evaluate_code_output(candidate.get("code"), reference_answer, executor)

    metrics = {
        "direct_full": int(direct_full),
        "direct_partial": int(direct_partial),
        "code_full": int(code_full),
        "code_partial": int(code_partial),
        "code_failed": int(code_failed),
    }

    return (direct_full or code_full), metrics


def rule_equality(correct, candidate):
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    res = candidate
    answer = correct

    is_early_stop = False
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
            print(f"error in q")
            score = 0
            is_early_stop = True
    except Exception as e:
        score = 0
        is_early_stop = True

    if not is_early_stop:  # if cannot find predicted_idx, then done
        if predicted_idx == answer:
            score = 1
        else:
            score = 0
    print(f'rule_based: extracted_answer: {predicted_idx}; answer: {answer}; score: {score}')
    return score


def rule_equality_folio(correct, candidate):
    res = candidate
    answer = correct
    if 'True' in res:
        pred = 'True'
    elif 'False' in res:
        pred = 'False'
    elif 'Uncertain' in res:
        pred = 'Uncertain'
    else:
        pred = ''

    score = pred == answer
    print(f'rule_based: extracted_answer: {pred}; answer: {answer}; score: {score}')
    return float(score)


def check_equality(dataset, question, correct, candidate, cfg):
    if is_smfr_dataset(dataset):
        if isinstance(candidate, dict):
            parsed_candidate = candidate
        else:
            parsed_candidate = _parse_smfr_model_output(candidate)

        smfr_executor = cfg.get("_smfr_executor")
        if smfr_executor is None and SafeCodeExecutor:
            smfr_executor = SafeCodeExecutor(timeout=30)

        score, _ = _evaluate_smfr_candidate(correct, parsed_candidate, smfr_executor)
        return float(score)

    FORMAT_INST = lambda \
            request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n\n"""

    output_description = "Return ONLY 'yes' or 'no' and DO NOT return anything other than these two."
    thinking_description = "Give your detailed thinking, Specifically, what is expression 1 and what is expression 2."

    output_fields_and_description = {key: f"Your {key}. {thinking_description}" if 'thinking' in key else f"Your {key}. {output_description}" for key in
                                     ['thinking', 'equal']}

    system_prompt = 'You are a helpful assistant. ' + FORMAT_INST(output_fields_and_description)

    if dataset == 'aime24' or dataset == 'hle_math':

        prompt = EQUALITY_TEMPLATE % {"expression1": correct, "expression2": candidate}

        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        cnt = 0
        while True:
            if cnt > 3:
                json_dict = {'equal': 'no'}
                break
            try:
                response, _ = equality_checker(msg)
                json_dict = json.loads(response)
                break
            except Exception as e:
                print(f"Entering here ========================= {cnt}")
                print(f'Error: {e}')
                cnt += 1

        print(f'json_dict: {json_dict}')
        score = json_dict['equal'].lower().strip() == "yes"

    elif dataset == 'browsecomp-plus':
        prompt = BROWSECOMP_PLUS_TEMPLATE.format(question=question, response=candidate, correct_answer=correct)

        msg = [
            {"role": "user", "content": prompt}
        ]

        while True:
            try:
                response, _ = equality_checker(msg)
                json_dict = json.loads(response)
                break
            except Exception as e:
                print(f'Error: {e}')

        print(f'json_dict: {json_dict}')
        score = json_dict['correct'].lower().strip() == "yes"

    elif dataset == 'gpqa_diamond':

        score = rule_equality(correct, candidate)

        if score == 1:
            print('Take rule based as it directly matched')
            return score  # we are done
        else:
            return score  # we are done; TODO: Dp not use LLM

            # use LLM to decide, sometimes it gives the value instead of the option
            INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

            prompt = MCQ_EQUALITY_TEMPLATE % {"question": question, "expression1": INDEX_TO_LETTER[correct], "expression2": candidate}

            msg = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            while True:
                try:
                    response, _ = equality_checker(msg)
                    json_dict = json.loads(response)
                    break
                except Exception as e:
                    print(f'Error: {e}')

            print(f'json_dict: {json_dict}')
            score = json_dict['equal'].lower().strip() == "yes"
    elif dataset == 'folio':
        score = rule_equality_folio(correct, candidate)
    elif dataset == 'knights-and-knaves':
        _judge_prompt = ("I will show you a model's response as well as the ground-truth answer towards a knights-and-knaves puzzle. "
                         "Please determine if the model's response is consistent with the ground truth."
                         "\n\nMode's Response:\n"
                         "{response}\n\n"
                         "Ground Truth:\n"
                         "{answer}\n\n"
                         "Your response should only contains `Yes` or `No`.").format(response=candidate, answer=correct)

        res = equality_checker([{"role": "user", "content": _judge_prompt}], response_format="normal")
        res = res[0]
        if "yes" in res.lower():
            score = 1.0
        else:
            score = 0.0
    elif dataset == 'hanoi':
        from hanoi import judge_prompt

        _judge_prompt = ("Here is a move sequence of Hanoi Game:\n\n{response}\n\n"
                         "Please evaluate the it according to the following criteria:\n\n").format(response=candidate) + judge_prompt
        # _judge_prompt = ("Here is a move sequence of Hanoi Game:\n\n{response}\n\n"
        #                  "Here is the ground-truth move of Hanoi Game:\n\n{answer}\n\n"
        #                  "Please evaluate the it by comparing the predicted move sequence and the ground truth one."
        #                  ).format(response=candidate, answer=correct) + judge_prompt

        _judge_prompt = _judge_prompt + "\n\nYou can first think step by step, and put your final decision in <decision> True or False </decision>."

        res = equality_checker([{"role": "user", "content": _judge_prompt}], response_format="normal")
        res = res[0]
        m = re.search(r"<decision>\s*(true|false)\s*</decision>", res, re.IGNORECASE)
        # return None if not m else (m.group(1).lower() == "true")
        if not m:
            score = 0.0
        else:
            pred = m.group(1).lower()
            score = float(pred == "true")

    elif dataset == "swe":
        example_id2instance_id = json.load(open("dataset/swe_exp_id2ins_id.json"))

        instance_id = example_id2instance_id[str(cfg["example_id"])]
        judge_path = f'{root_dir}/{dataset}/{cfg["example_id"]}/{model}_{model}_{cfg["orig_verifier_model"]}_0_{cfg["option"]}_judge'
        score = run_swebench_evaluation(judge_path, instance_id, candidate, cfg["judge_method"], cfg["option"])

    return score


model_sampler_map = {
    "o3-mini": OChatCompletionSampler(
        model="o3-mini",
    ),
    "gpt-4o_chatgpt": ChatCompletionSampler(
        model="gpt-4o", max_tokens=16384
    ),
    "gpt-5-nano": ChatCompletionSampler(model="gpt-5-nano-2025-08-07", temperature=1.0),
    "gpt-5": ChatCompletionSampler(model="gpt-5-2025-08-07", temperature=1.0, max_tokens=128000),
    "qwen-2.5-32b-instr": VllmChatCompletionSampler(
        model="qwen-2.5-32b-instr",
    ),
    "qwen3-30b-a3b": VllmChatCompletionSampler(
        model="qwen3-30b-a3b",
    ),
    "qwq-32b": ToChatCompletionSampler(
        model="Qwen/Qwen2.5-32B-Instruct",
    ),
    "llama-3.3-70b-instr": ToChatCompletionSampler(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    "deepseek-v3": ToChatCompletionSampler(
        model="deepseek-ai/DeepSeek-V3"
    ),
    "qwen3-next-80b-reasoning": VllmChatCompletionSampler(model="qwen3-next-80b-reasoning", response_format="xml", max_tokens=16384),
    "qwen3-30b-a3b-reasoning": VllmChatCompletionSampler(model="qwen3-30b-a3b-reasoning", response_format="xml", max_tokens=65536),
    "gpt-oss-120b": VllmChatCompletionSampler(model="gpt-oss-120b", response_format="json", max_tokens=131072),
    # "gpt-oss-120b": ChatCompletionSampler(model="gpt-oss-120b", max_tokens=131072, response_format="json"),
    "gemini-2.5-pro": ChatCompletionSampler(model="gemini-2.5-pro", max_tokens=131072)
}


def _process_one_example(example_id: int, cfg: dict):
    """
    子进程执行：处理单个 example_id，返回 (example_id, is_correct:int, special_msgs:list[str], lines_to_write:list[str])
    """
    dataset = cfg['dataset']
    judge_method = cfg['judge_method']
    max_response_per_sample = cfg['max_response_per_sample']
    model = cfg['model']
    majority_vote = cfg['majority_vote']
    root_dir = cfg['root_dir']
    option = cfg['option']
    prm_model_path = cfg.get('prm_model_path', None)
    skip_eval = cfg.get('skip_eval', False)
    pipeline_stage = cfg.get('pipeline_stage', 'end_to_end')

    exp_cfg = copy.deepcopy(cfg)
    exp_cfg['example_id'] = example_id

    # —— 每个进程内独立初始化 sampler，避免 pickling 问题 ——
    # model_sampler_map = {
    #     "o3-mini": OChatCompletionSampler(model="o3-mini"),
    #     "gpt-4o_chatgpt": ChatCompletionSampler(model="gpt-4o"),
    #     "qwen-2.5-32b-instr": VllmChatCompletionSampler(model="qwen-2.5-32b-instr"),
    #     "qwen3-30b-a3b": VllmChatCompletionSampler(model="qwen3-30b-a3b"),
    #     "qwq-32b": ToChatCompletionSampler(model="Qwen/Qwen2.5-32B-Instruct"),
    #     "llama-3.3-70b-instr": ToChatCompletionSampler(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    #     "deepseek-v3": ToChatCompletionSampler(model="deepseek-ai/DeepSeek-V3"),
    # }
    post_processer = model_sampler_map['gpt-4o_chatgpt']
    equality_checker = model_sampler_map['gpt-4o_chatgpt']
    sampler = model_sampler_map[model]

    is_smfr = is_smfr_dataset(dataset)
    smfr_executor = SafeCodeExecutor(timeout=30) if is_smfr and SafeCodeExecutor else None
    smfr_metrics = {
        "direct_full": 0,
        "direct_partial": 0,
        "code_full": 0,
        "code_partial": 0,
        "code_failed": 0,
    }

    special_msgs = []
    lines_to_write = []
    is_correct = False

    try:
        response_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{cfg["orig_verifier_model"]}_0_{option}_response'
        try:
            with open(response_path, 'r') as json_file:
                responses = json.load(json_file)
        except Exception:
            special_msgs.append(f'example_id {example_id} response file {response_path} does not exisit')
            if is_smfr:
                return example_id, 0, special_msgs, lines_to_write, smfr_metrics
            return example_id, 0, special_msgs, lines_to_write

        if len(responses) < max_response_per_sample:
            special_msgs.append(f'example_id {example_id}: responses length {len(responses)} is lower than {max_response_per_sample}')
            # 仅警告，不中断

        if len(responses) == 0:
            raise ValueError(f"No response found for example id {example_id}")

        question = responses[0]['problem']

        extracted_answers, correct_answers = [], []
        for resp in responses:
            filter_response = resp['response']
            if '<TOO_HARD>' in filter_response:
                filter_response = filter_response[:filter_response.index('<TOO_HARD>')]

            if is_smfr:
                extracted = _parse_smfr_model_output(filter_response)
            elif 'swe' not in dataset:
                match = re.search(ANSWER_PATTERN, filter_response)
                extracted = match.group(1) if match else None

            else:
                if "Answer:" in filter_response:
                    extracted = filter_response.rsplit("Answer:", 1)[1]
                else:
                    extracted = None
            if is_smfr:
                extracted_answers.append(extracted)
            else:
                extracted_answers.append(extracted.strip() if extracted is not None else extracted)

            correct_answers.append(resp['correct_answer'])

        # —— 三种 judge_method 的处理 ——
        if judge_method == 'oracle':
            for round_id, (ca, ea) in enumerate(zip(correct_answers, extracted_answers)):
                if round_id == max_response_per_sample:
                    break
                # 兼容 rule_equality* 中的打印（使用到全局 extracted_answer）
                globals()['extracted_answer'] = ea

                if is_smfr:
                    score, metrics = _evaluate_smfr_candidate(ca, ea, smfr_executor)
                    smfr_metrics = metrics
                else:
                    score = check_equality(dataset, question, ca, ea, exp_cfg)

                if score == 1:
                    if is_smfr:
                        lines_to_write.append(
                            f'experiemnt {example_id}: 1 ({responses[round_id]["n"]}); '
                            f'direct_full={smfr_metrics["direct_full"]}; code_full={smfr_metrics["code_full"]}; '
                            f'correct_answer: {ca} vs. extracted_answer: {ea}\n'
                        )
                    else:
                        lines_to_write.append(
                            f'experiemnt {example_id}: 1 ({responses[round_id]["n"]}); correct_answer: {ca} vs. extracted_answer: {ea}\n'
                        )
                    is_correct = True
                    break

        elif judge_method == 'external':
            from llm_judge import prm
            if not prm_model_path:
                prm_model_path = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"

            post_process_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{cfg["orig_verifier_model"]}_0_{option}_post_process.json'
            try:
                chosen_id = prm.run_judge(
                    prm_model_path, None, post_process_path, responses, sampler, post_processer, extracted_answers, dataset
                )
            except Exception as e:
                special_msgs.append(f'Error: {e}; skip')
                if is_smfr:
                    return example_id, 0, special_msgs, lines_to_write, smfr_metrics
                return example_id, 0, special_msgs, lines_to_write

            ca = correct_answers[chosen_id]
            ea = extracted_answers[chosen_id]
            globals()['extracted_answer'] = ea
            if is_smfr:
                score, metrics = _evaluate_smfr_candidate(ca, ea, smfr_executor)
                smfr_metrics = metrics
            else:
                score = check_equality(dataset, question, ca, ea, exp_cfg)
            if score == 1:
                if is_smfr:
                    lines_to_write.append(
                        f'experiemnt {example_id}: 1 ({responses[chosen_id]["n"]}); '
                        f'direct_full={smfr_metrics["direct_full"]}; code_full={smfr_metrics["code_full"]}; '
                        f'correct_answer: {ca} vs. extracted_answer: {ea}\n'
                    )
                else:
                    lines_to_write.append(
                        f'experiemnt {example_id}: 1 ({responses[chosen_id]["n"]}); correct_answer: {ca} vs. extracted_answer: {ea}\n'
                    )
                is_correct = True

        elif judge_method == 'self':
            post_process_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_{option}_sub_task_post_process.json'
            log_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_{option}_sub_self_verifier_log'
            score_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_{option}_score.json'

            if pipeline_stage == 'select':
                chosen_id = self_verifier_list_wise.run_self_verifier(
                    post_process_path, log_path, score_path, responses, sampler, post_processer,
                    extracted_answers, dataset, max_response_per_sample, majority_vote
                )
                if is_smfr:
                    return example_id, 0, special_msgs, lines_to_write, smfr_metrics
                return example_id, 0, special_msgs, lines_to_write
            elif pipeline_stage == 'eval':
                try:
                    chosen_id = self_verifier_list_wise.load_selection(score_path)
                except Exception as e:
                    special_msgs.append(f'example_id {example_id} failed to load score: {e}')
                    if is_smfr:
                        return example_id, 0, special_msgs, lines_to_write, smfr_metrics
                    return example_id, 0, special_msgs, lines_to_write
            elif skip_eval:
                chosen_id = self_verifier_list_wise.run_self_verifier(
                    post_process_path, log_path, score_path, responses, sampler, post_processer,
                    extracted_answers, dataset, max_response_per_sample, majority_vote=True
                )
            else:
                chosen_id = self_verifier_list_wise.run_self_verifier(
                    post_process_path, log_path, score_path, responses, sampler, post_processer,
                    extracted_answers, dataset, max_response_per_sample, majority_vote
                )
            ca = correct_answers[chosen_id]
            ea = extracted_answers[chosen_id]
            globals()['extracted_answer'] = ea
            if is_smfr:
                score, metrics = _evaluate_smfr_candidate(ca, ea, smfr_executor)
                smfr_metrics = metrics
            else:
                score = check_equality(dataset, question, ca, ea, exp_cfg)
            if score == 1:
                if is_smfr:
                    lines_to_write.append(
                        f'experiemnt {example_id}: 1 ({responses[chosen_id]["n"]}); '
                        f'direct_full={smfr_metrics["direct_full"]}; code_full={smfr_metrics["code_full"]}; '
                        f'correct_answer: {ca} vs. extracted_answer: {ea}\n'
                    )
                else:
                    lines_to_write.append(
                        f'experiemnt {example_id}: 1 ({responses[chosen_id]["n"]}); correct_answer: {ca} vs. extracted_answer: {ea}\n'
                    )
                is_correct = True

        elif judge_method in ["cot", "cot-sc", "debate", "reflexion"]:
            if judge_method == "cot":
                target = "Chain-of-Thought"
            elif judge_method == "cot-sc":
                target = "Self-Consistency with Chain-of-Thought"
            elif judge_method == "debate":
                target = "LLM Debate"
            elif judge_method == "reflexion":
                target = "Self-Refine (Reflexion)"
            else:
                raise ValueError

            ca = correct_answers[0]
            ea = extracted_answers[0]
            resp_id = 0
            for resp_id, resp in enumerate(responses):
                if resp["n"] == target:
                    ca = correct_answers[resp_id]
                    ea = extracted_answers[resp_id]
                    break

            globals()['extracted_answer'] = ea
            if is_smfr:
                score, metrics = _evaluate_smfr_candidate(ca, ea, smfr_executor)
                smfr_metrics = metrics
            else:
                score = check_equality(dataset, question, ca, ea, exp_cfg)
            if score == 1:
                if is_smfr:
                    lines_to_write.append(
                        f'experiemnt {example_id}: 1 ({responses[resp_id]["n"]}); '
                        f'direct_full={smfr_metrics["direct_full"]}; code_full={smfr_metrics["code_full"]}; '
                        f'correct_answer: {ca} vs. extracted_answer: {ea}\n'
                    )
                else:
                    lines_to_write.append(
                        f'experiemnt {example_id}: 1 ({responses[resp_id]["n"]}); correct_answer: {ca} vs. extracted_answer: {ea}\n'
                    )
                is_correct = True

        if not is_correct:
            special_msgs.append(f'Cannot Find Correct Answer acorss reponses for example_id: {example_id}')

        if is_smfr:
            return example_id, int(is_correct), special_msgs, lines_to_write, smfr_metrics
        return example_id, int(is_correct), special_msgs, lines_to_write

    except Exception as e:
        import traceback
        traceback.print_exc()
        special_msgs.append(f'example_id {example_id} crashed: {repr(e)}')
        if is_smfr:
            return example_id, 0, special_msgs, lines_to_write, smfr_metrics
        return example_id, 0, special_msgs, lines_to_write


parser = argparse.ArgumentParser()
parser.add_argument('--judge_method', type=str)
parser.add_argument('--baseline', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--max_sample', type=int)
parser.add_argument('--min_sample', type=int, default=0)
parser.add_argument('--max_response_per_sample', type=int)
parser.add_argument('--model', type=str, default="gpt-4o_chatgpt")
parser.add_argument('--majority_vote', action='store_true')
parser.add_argument("--save_dir", type=str, default="results")
parser.add_argument("--option", type=str, default="plan")
parser.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
parser.add_argument("--orig_verifier_model", type=str, default="gpt-4o_chatgpt")
parser.add_argument("--skip_eval", default=False, action="store_true")
parser.add_argument("--pipeline_stage", type=str, default="end_to_end")
args = parser.parse_args()

if __name__ == "__main__":

    dataset = args.dataset
    judge_method = args.judge_method
    max_sample = args.max_sample
    min_sample = args.min_sample
    max_response_per_sample = args.max_response_per_sample
    model = args.model
    majority_vote = args.majority_vote

    special_ids = []
    root_dir = f'{args.save_dir}/question/meta_agent/{args.baseline}'
    is_smfr = is_smfr_dataset(dataset)

    # all results
    if judge_method == 'external':
        prm_model_path = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
        result_path = f'{root_dir}/{dataset}/{model}_{model}_Skywork-o1-Open-PRM-Qwen-2.5-7B.results'
    else:
        result_path = f'{root_dir}/{dataset}/{model}_{model}_{judge_method}.results_{max_response_per_sample}'
    if os.path.exists(result_path):
        os.remove(result_path)  # remove the file, do not repeat

    print('result_path: ', result_path)

    # we always use gpt-4o for post-process and equilty check
    post_processer = model_sampler_map['gpt-4o_chatgpt']
    equality_checker = model_sampler_map['gpt-4o_chatgpt']
    sampler = model_sampler_map[model]

    correct_example = []
    special_ids = []
    smfr_metrics_list = []

    if 'gpqa' in dataset:
        assert min_sample == 32
        assert max_sample == 197

    example_ids = list(range(min_sample, max_sample + 1))

    # 组装子进程需要的配置
    cfg = {
        'dataset': dataset,
        'judge_method': judge_method,
        'max_response_per_sample': max_response_per_sample,
        'model': model,
        'majority_vote': majority_vote,
        'root_dir': root_dir,
        'option': args.option,
        'prm_model_path': "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B" if judge_method == 'external' else None,
        'orig_verifier_model': args.orig_verifier_model,
        'skip_eval': args.skip_eval,
        'pipeline_stage': args.pipeline_stage
    }

    lines_buffer = []

    print(f'Launching multiprocessing with num_workers={args.num_workers}, total examples={len(example_ids)}')
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_process_one_example, eid, cfg): eid for eid in example_ids}
        for fut in as_completed(futures):
            if is_smfr:
                eid, is_correct, specials, lines, smfr_metrics = fut.result()
                smfr_metrics_list.append(smfr_metrics)
            else:
                eid, is_correct, specials, lines = fut.result()
            if specials:
                special_ids.extend(specials)
            if lines:
                lines_buffer.extend(lines)
            correct_example.append(is_correct)
    # for eid in example_ids:
    #     res = _process_one_example(eid, cfg)
    # raise Exception

    # 输出 special 消息
    for sid in special_ids:
        print('special_id: ', sid)

    # 统一写入 result_path（避免多进程并发写文件）
    if lines_buffer:
        with open(result_path, "w") as fh:
            for line in lines_buffer:
                fh.write(line)

    if is_smfr and smfr_metrics_list:
        total = len(smfr_metrics_list)
        direct_full = sum(m["direct_full"] for m in smfr_metrics_list)
        direct_partial = sum(m["direct_partial"] for m in smfr_metrics_list)
        code_full = sum(m["code_full"] for m in smfr_metrics_list)
        code_partial = sum(m["code_partial"] for m in smfr_metrics_list)
        code_failed = sum(m["code_failed"] for m in smfr_metrics_list)

        print("\n" + "=" * 60)
        print("SMFR DATASET EVALUATION")
        print("=" * 60)
        print(f"Total samples evaluated: {total}")
        print("\nDirect Answer Metrics:")
        print(f"  Full Match:    {direct_full}/{total} ({direct_full/total:.2%})")
        print(f"  Partial Match: {direct_partial}/{total} ({direct_partial/total:.2%})")
        print("\nCode Output Metrics:")
        print(f"  Full Match:    {code_full}/{total} ({code_full/total:.2%})")
        print(f"  Partial Match: {code_partial}/{total} ({code_partial/total:.2%})")
        print(f"  Execution Failures:      {code_failed}/{total} ({code_failed/total:.2%})")
        print("=" * 60)

        with open(result_path, "a+") as fh:
            fh.write("\n" + "=" * 60 + "\n")
            fh.write("SMFR DATASET EVALUATION\n")
            fh.write("=" * 60 + "\n")
            fh.write(f"Total samples evaluated: {total}\n\n")
            fh.write("Direct Answer Metrics:\n")
            fh.write(f"  Full Match:    {direct_full}/{total} ({direct_full/total:.2%})\n")
            fh.write(f"  Partial Match: {direct_partial}/{total} ({direct_partial/total:.2%})\n\n")
            fh.write("Code Output Metrics:\n")
            fh.write(f"  Full Match:    {code_full}/{total} ({code_full/total:.2%})\n")
            fh.write(f"  Partial Match: {code_partial}/{total} ({code_partial/total:.2%})\n")
            fh.write(f"  Execution Failures:      {code_failed}/{total} ({code_failed/total:.2%})\n")
            fh.write("=" * 60 + "\n")

    if not (args.pipeline_stage == "select" and judge_method == "self"):
        acc = (sum(correct_example) / len(correct_example)) if correct_example else 0.0
        print(f'correct {sum(correct_example)}; Total: {len(correct_example)}; Acc: {acc}')

        with open(result_path, "a+") as fh:
            fh.write(f'correct {sum(correct_example)}; Total: {len(correct_example)}; Acc: {acc}\n')
