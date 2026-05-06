import argparse
import asyncio
import copy
import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

import async_search as search
from prompts.swe.patch_oracle import AGENTLESS_REPAIR
from sampler import init_model
from utils import extract_xml
from utils import load_questions

SMFR_DATASET_NAME = "workflow_search/smfr"
SMFR_MERGED_DATASET_FILE = os.environ.get("SMFR_MERGED_DATASET_FILE", "balanced_dataset_merged_depth_fixed.jsonl")


def determine_format(model_name):
    json_model = ['gpt', 'gemini']
    xml_model = ['qwen', 'llama-3.3', 'deepseek']

    if any(kw in model_name for kw in json_model):
        format_inst_template = ("Reply EXACTLY with the following JSON format.\n{request_keys}\n"
                                "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n\n")
        return format_inst_template, "json"

    elif any(kw in model_name for kw in xml_model):
        format_inst_template = ("Reply EXACTLY with the following XML format.\n{request_keys}\n"
                                "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\n\n")
        return format_inst_template, 'xml'

    else:
        raise NotImplementedError


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str)
    parser.add_argument('--n_generation', type=int, default=10)
    parser.add_argument('--max_round', type=int, default=5)
    parser.add_argument('--max_sc', type=int, default=5)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--option', type=str, default='')
    parser.add_argument('--meta_model',
                        type=str)
    parser.add_argument('--node_model',
                        type=str)
    parser.add_argument('--verifier_model',
                        type=str,
                        default="o3-mini")
    # gpt-4o
    parser.add_argument('--shorten_context', action='store_true')
    parser.add_argument('--merge_context', action='store_true')

    parser.add_argument(
        "--blocks", type=str, nargs="*", help="Number of examples to use (overrides default)"
    )
    parser.add_argument('--dataset', type=str)
    parser.add_argument(
        "--given_examples", type=int, nargs="*", help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--use_oracle_verifier", action='store_true', default=False
    )
    parser.add_argument(
        "--defer_verifier", action='store_true'
    )
    parser.add_argument(
        "--no_decompose", action='store_true'
    )
    parser.add_argument(
        "--no_meta_reward", action='store_true'
    )
    parser.add_argument("--early_stop", action='store_true', default=False)
    parser.add_argument("--no_history", action='store_true', default=False)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    return args


def is_smfr_dataset(dataset_name: str) -> bool:
    if not dataset_name:
        return False
    name = dataset_name.lower().strip()
    return name == SMFR_DATASET_NAME or "smfr_synthetic" in name


def load_smfr_examples():
    dataset_path = Path(__file__).resolve().parent / "smfr_synthetic_dataset" / SMFR_MERGED_DATASET_FILE
    if not dataset_path.exists():
        raise FileNotFoundError(f"Smfr data file not found at {dataset_path}")

    examples = []
    with dataset_path.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "problem": data["problem"],
                "answer": data["answer"],
                "depth": data.get("depth"),
                "instance_id": data.get("id", idx),
            })
    return examples, dataset_path


async def run_aime_search(example, example_id, meta_model, node_model, verifier_model, n, dataset, extra_info,
                          blocks, n_generation, save_dir, option, defer_verifier, debug_max):
    expr_name = f'question/meta_agent/{dataset}/{example_id}/{meta_model}_{node_model}_{verifier_model}_{n}'
    print('args.expr_name: ', expr_name)

    questions = [example['problem']]
    answers = [example['answer']]
    instance_identifier = example.get('instance_id', example_id)

    task_queue = []
    for q in questions:
        taskInfo = ('task', 'User', q, None, None, None, -1)
        task_queue.append(taskInfo)

    extra_info["answers"] = answers
    extra_info["questions"] = questions
    extra_info["example_id"] = example_id
    extra_info["instance_id"] = instance_identifier
    extra_info["response_dict"] = []

    # search
    await search.search(extra_info, task_queue, meta_model, blocks, verifier_model, n_generation,
                        save_dir, expr_name, option, dataset, defer_verifier, debug_max)


async def run_gpqa_search(example, example_id, meta_model, node_model, verifier_model, n, dataset, extra_info,
                          blocks, n_generation, save_dir, option, defer_verifier, debug_max):
    expr_name = f'question/meta_agent/{dataset}/{example_id}/{meta_model}_{node_model}_{verifier_model}_{n}'
    # print('args.expr_name: ', args.expr_name)

    questions = [example['problem']]
    answers = [example['answer']]

    final_question = []
    task_queue = []
    for q in questions:
        task_content = f"What is the correct answer to this question: {q.question}" \
                       + f"\n\nChoices:\n(A) {q.choice1}\n(B) {q.choice2}\n(C) {q.choice3}\n(D) {q.choice4}"
        taskInfo = ('task', 'User', task_content, None, None, None, -1)
        task_queue.append(taskInfo)
        final_question.append(task_content)

    # extra_info["score_compute"] = data_scorer.score
    extra_info["answers"] = answers
    extra_info["questions"] = final_question  # 注意：此处原始代码使用了 final_question 变量
    extra_info["example_id"] = example_id
    extra_info["response_dict"] = []
    extra_info["instance_id"] = example.get('instance_id', example_id)

    # search
    await search.search(extra_info, task_queue, meta_model, blocks, verifier_model, n_generation,
                        save_dir, expr_name, option, dataset, defer_verifier, debug_max)


async def run_swe_search(example, example_id, meta_model, node_model, verifier_model, n, dataset, extra_info,
                         blocks, n_generation, save_dir, option, defer_verifier, debug_max):
    expr_name = f'question/meta_agent/{dataset}/{example_id}/{meta_model}_{node_model}_{verifier_model}_{n}'
    print('args.expr_name: ', expr_name)

    instance_id = example['instance_id']
    example_text = example['text']

    cot_instruction = extra_info["cot_instruction"]
    if extra_info["format_choice"] == 'xml':
        example_text = example_text.replace('<patch>', '<answer>').replace('</patch>', '</answer>')
        example_text = example_text.replace('Please respond with a single patch file in the following format.',
                                            'If asked for <answer> field, the <answer> field should be a single patch file in the following format')
        cot_instruction = "Put your thinking process in the <thinking> field and the final patch in the <answer> field."

    code_snippet = extract_xml(example_text, 'code').strip()
    print('code_snippet: ', code_snippet)

    questions = [example_text + '\n\n' + AGENTLESS_REPAIR]
    answers = [None]

    print('instance_id: ', instance_id)

    task_queue = []
    for q in questions:
        taskInfo = ('task', 'User', q, None, None, None, -1)
        task_queue.append(taskInfo)

    extra_info["cot_instruction"] = cot_instruction
    extra_info["answers"] = answers
    extra_info["questions"] = questions
    extra_info["example_id"] = example_id
    extra_info["response_dict"] = []
    extra_info["dataset"] = dataset
    extra_info["instance_id"] = instance_id
    extra_info["code_snippet"] = code_snippet

    await search.search(extra_info, task_queue, meta_model, blocks, verifier_model, n_generation,
                        save_dir, expr_name, option, dataset, defer_verifier, debug_max)


async def main(args):
    blocks = args.blocks
    meta_model = args.meta_model
    node_model = args.node_model
    verifier_model = args.verifier_model
    use_oracle_verifier = args.use_oracle_verifier
    max_round = args.max_round
    max_sc = args.max_sc

    # SEARCHING_MODE = True
    # technique = args.dataset.split('/')[0]
    # data_scorer = DataScorer(args.dataset, technique)

    print('verifier_model: ', verifier_model)
    # print('technique: ', technique)
    print('node_model: ', node_model)

    extra_info = {}
    format_inst_template, format_choice = determine_format(node_model)
    extra_info["format_choice"] = format_choice

    init_model(verifier_model, response_format=determine_format(verifier_model)[1], max_tokens=args.max_tokens)
    init_model(node_model, response_format=format_choice, max_tokens=args.max_tokens)
    init_model(meta_model, response_format=format_choice, max_tokens=args.max_tokens)

    extra_info["FORMAT_INST"] = format_inst_template
    extra_info["shorten_context"] = args.shorten_context
    extra_info["merge_context"] = args.merge_context
    extra_info["COST_TOTAL"] = 0.0
    extra_info["USAGE_TOTAL"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "calls": 0,
    }
    extra_info["USAGE_BY_MODEL"] = {}
    extra_info["no_decompose"] = args.no_decompose
    extra_info["no_meta_reward"] = args.no_meta_reward

    print('shorten_context: ', args.shorten_context)
    print('merge_context: ', args.merge_context)
    print('global_no_meta_reward: ', args.no_meta_reward)
    print('global_no_decompose: ', args.no_decompose)

    code_snippet = None
    for n in range(args.n_repeats):
        if ('swe_bench' in args.dataset) or ('workflow_search/swe' in args.dataset) or ('swe_test' in args.dataset):

            cot_instruction = "Put your thinking process in the 'thinking' entry and the final patch in the 'answer' entry."  # TODO: may need something for xml

            debate_role = ['Computer Science Professor', 'Software Engineer']

            # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
            output_description = (
                "If the question is asked for a patch to fix an issue, Return ONLY the solution and DO NOT return anything other than the patch;"
                " If the question is asked for more than a patch, Return what the question asked and make sure the answer is complete.")

            # Load SWE dataset
            if 'swe_bench' in args.dataset:
                examples = load_dataset("princeton-nlp/SWE-bench_Lite_oracle", split="test")
            else:
                dataset_path = Path("dataset/swe_test.jsonl")
                if not dataset_path.exists():
                    raise FileNotFoundError(f"SWE data file not found at {dataset_path}")
                examples = []
                with dataset_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        examples.append(json.loads(line))

            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["node_model"] = node_model
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["early_stop"] = args.early_stop
            extra_info["verifier_model"] = verifier_model
            extra_info["no_history"] = args.no_history

            semaphore = asyncio.Semaphore(args.max_workers)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_swe_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            print(len(tasks))
            await tqdm_asyncio.gather(*tasks)
        elif 'aime24' in args.dataset:
            cot_instruction = "Please think step by step and then solve the task."
            # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
            output_description = (
                "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; "
                "If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete.")

            debate_role = ['Math Professor', 'Grade School Teacher']

            dataset = load_dataset("simplescaling/aime24_nofigures")
            df = pd.DataFrame(dataset['train'])
            examples = [row.to_dict() for _, row in df.iterrows()]

            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet
            extra_info["early_stop"] = args.early_stop
            extra_info["no_history"] = args.no_history

            # 控制并发数量的信号量，最多同时运行5个任务
            semaphore = asyncio.Semaphore(args.max_workers)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_aime_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

                # if len(tasks) >= 1:
                #     break

            print(len(tasks))
            await tqdm_asyncio.gather(*tasks)

        elif 'gpqa_diamond' in args.dataset:
            cot_instruction = "Please think step by step and then solve the task."
            # output_description = "Return ONLY the alphabet choice, i.e. A or B or C or D."
            output_description = ("If the question is asked for a multiple-choice result, Return ONLY the alphabet choice, i.e. A or B or C or D; "
                                  "If the question is asked for more than multiple-choice results, "
                                  "return what the question asked and make sure the answer is complete.")
            # need to consider sub-task output as well (no fixed form for sub-tasks)
            debate_role = ['Biology Expert', 'Physics Expert', 'Chemistry Expert', 'Science Generalist']

            # set seed 0 for valid set
            questions = load_questions('dataset/gpqa_diamond.csv', seed=0)
            answers = [question.correct_index for question in questions]

            examples = [{'problem': questions[i], 'answer': answers[i]} for i in range(len(questions))]
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["output_description"] = output_description
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet
            extra_info["early_stop"] = args.early_stop
            extra_info["no_history"] = args.no_history

            # 控制并发数量的信号量，最多同时运行5个任务
            semaphore = asyncio.Semaphore(args.max_workers)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_gpqa_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):
                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            print(len(tasks))
            await tqdm_asyncio.gather(*tasks)
        elif 'folio' in args.dataset:
            cot_instruction = "Please think step by step and then solve the task."
            output_description = "Your final answer should be one of {True, False, Uncertain} to indicate the given conclusion is correct, incorrect, or cannot be inferred from the given premises, respectively."
            debate_role = ['Philosopher 1', 'Philosopher 2', 'Philosopher 3']
            dataset = load_dataset('yale-nlp/FOLIO', split="validation")

            _template = (f"I will show you a series of premises, and one conclusion. "
                         f"Please decide if the conclusion can be inferred from the premises based on logic relations."
                         f"\n\n"
                         f"Premises:\n{{premises}}\n\nConclusion:\n{{conclusion}}")

            examples = []
            for item in dataset:
                question = _template.format(premises=item["premises"], conclusion=item["conclusion"])
                answer = item["label"]
                examples.append({'problem': question, 'answer': answer})

            examples = examples[:12]

            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["output_description"] = output_description
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet
            extra_info["early_stop"] = args.early_stop
            extra_info["no_history"] = args.no_history

            semaphore = asyncio.Semaphore(args.max_workers)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_aime_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):
                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            print(len(tasks))
            await tqdm_asyncio.gather(*tasks)
        elif 'hle_math' in args.dataset:  # I simply copy the instruction from AIME
            cot_instruction = "Please think step by step and then solve the task."
            # Multiple choice or exact match.
            output_description = (
                "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; "
                "If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete.;"
                "If the question is in multiple-choice format, Return ONLY the alphabet choice, i.e. A or B or C or D or E.")

            debate_role = ['Math Professor', 'Grade School Teacher']

            # dataset = load_dataset("cais/hle", split="test")
            dataset = json.load(open("dataset/hle_math200int_seed0.json"))
            # dataset = [item for item in dataset if item["category"] == "Math"]
            examples = []
            for item in dataset:
                examples.append({'problem': item['question'], 'answer': item['answer']})

            # examples = examples[:24]

            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet
            extra_info["early_stop"] = args.early_stop
            extra_info["no_history"] = args.no_history

            # 控制并发数量的信号量，最多同时运行5个任务
            semaphore = asyncio.Semaphore(args.max_workers)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_aime_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            print(len(tasks))
            await tqdm_asyncio.gather(*tasks)
        elif 'browsecomp-plus' in args.dataset:
            cot_instruction = ("Thoroughly read the question and provided documents, reason step by step, "
                               "and cite document evidence in your explanations whenever possible.")
            output_description = ("Return ONLY the final answer requested by the question. "
                                  "Keep it concise and grounded in the supplied documents.")
            debate_role = ['Research Analyst 1', 'Research Analyst 2', 'Research Analyst 3']

            dataset_path = Path("dataset/bcp_test.jsonl")
            if not dataset_path.exists():
                raise FileNotFoundError(f"browsecomp-plus data file not found at {dataset_path}")

            examples = []
            with dataset_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    question = data["question"].strip()
                    answer = data["answer"].strip()
                    instance_identifier = data.get("query_id", len(examples))
                    examples.append({
                        'problem': question,
                        'answer': answer,
                        'instance_id': instance_identifier
                    })

            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet
            extra_info["early_stop"] = args.early_stop
            extra_info["no_history"] = args.no_history

            semaphore = asyncio.Semaphore(args.max_workers)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_aime_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):
                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            print(len(tasks))
            await tqdm_asyncio.gather(*tasks)

        elif is_smfr_dataset(args.dataset):
            cot_instruction = "Please think step by step and then solve the task."
            output_description = (
                "Contain single-line JSON **string** with keys: "
                "\"answer\" and \"code\". "
                "\"answer\" must be the final winner name (string), a list of names for ties, or null. "
                "\"code\" must be a JSON string with escaped newlines (use \\\\n, no markdown fences). "
                "The code must define solve() and return a dict with keys "
                "\"investor_dates\", \"comparison\", and \"answer\". "
                "Include all required input data in the code. Do not include extra text."
            )
            debate_role = ["Financial Analyst", "Quant Researcher", "Risk Manager"]

            examples, dataset_path = load_smfr_examples()
            print(f"Loaded smfr dataset from {dataset_path} with {len(examples)} examples")

            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet
            extra_info["early_stop"] = args.early_stop
            extra_info["no_history"] = args.no_history

            semaphore = asyncio.Semaphore(args.max_workers)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_aime_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):
                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            print(len(tasks))
            await tqdm_asyncio.gather(*tasks)

        elif 'knights-and-knaves' in args.dataset:
            cot_instruction = "Let's think step by step, by considering whether each person is lying and if that leads to contradiction"
            debate_role = ["Expert Player 1", "Expert Player 2", "Expert Player 3"]
            output_description = "Please illustrate each one's identity clearly after enough reasoning."

            dataset = load_dataset("K-and-K/knights-and-knaves", "test", split="2ppl")

            examples = []
            for item in dataset:
                examples.append({
                    "problem": item["quiz"],
                    "answer": item["solution_text_format"]
                })

            examples = examples[:24]

            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.datasets
            extra_info["code_snippet"] = code_snippet
            extra_info["early_stop"] = args.early_stop
            extra_info["no_history"] = args.no_history

            # 控制并发数量的信号量，最多同时运行5个任务
            semaphore = asyncio.Semaphore(args.max_workers)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_aime_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            print(len(tasks))
            await tqdm_asyncio.gather(*tasks)
        elif "hanoi" in args.dataset:
            from hanoi import load_hanoi, problem_template as cot_instruction

            debate_role = ["Expert Player 1", "Expert Player 2", "Expert Player 3"]
            output_description = ""

            examples = load_hanoi(3, 5)

            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet
            extra_info["early_stop"] = args.early_stop
            extra_info["no_history"] = args.no_history

            # 控制并发数量的信号量，最多同时运行5个任务
            semaphore = asyncio.Semaphore(args.max_workers)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_aime_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            print(len(tasks))
            await tqdm_asyncio.gather(*tasks)

        else:
            raise NotImplementedError


if __name__ == '__main__':
    args = parse_arguments()

    asyncio.run(main(args))
