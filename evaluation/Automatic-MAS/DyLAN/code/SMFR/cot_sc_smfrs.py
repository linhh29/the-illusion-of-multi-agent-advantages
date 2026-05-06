# -*- coding: utf-8 -*-
"""
SMFR CoT-SC: multiple CoT samples per question + voting on extracted answer.
Direct: most_frequent on parsed model_answer after dropping None, empty strings, and literal 'None'.
Code: execute each completion's code, then majority vote on meaningful executed answers (same filters);
metrics and saved JSON use this ensemble (not only the first completion).
"""
import ast
import asyncio
import glob
import json
import os
import random
import re
import sys
from utils import (
    get_smfr_qa_pairs,
    extract_reference_answer,
    parse_model_output,
    evaluate_direct_answer,
    generate_answer_with_mode,
    most_frequent,
    is_meaningless_direct_vote_value,
    run_code_ensemble_eval,
)
from safe_code_executor import SafeCodeExecutor

QUERY_JSONL = sys.argv[1]
EXP_NAME = sys.argv[2]
MODEL = sys.argv[3]
DIR_NAME = sys.argv[4]
ROLES = ast.literal_eval(sys.argv[5])
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)

RUN_ID = sys.argv[6] if len(sys.argv) > 6 else None
if RUN_ID:
    DIR_NAME = DIR_NAME + '_run' + str(RUN_ID)
CODE_EVAL_MODE = int(sys.argv[7]) if len(sys.argv) > 7 else 1
REQUIRE_CODE = CODE_EVAL_MODE == 1


def parse_ensemble_override(arg_index):
    """
    Optional argv[arg_index]: strict ensemble size for CoT-SC.
    If missing or empty, return None (use DyLAN-derived size or default 5).
    Placed after argv[7] CODE_EVAL_MODE for SMFR.
    """
    if len(sys.argv) <= arg_index:
        return None
    s = sys.argv[arg_index].strip()
    if not s:
        return None
    try:
        n = int(s)
        if n < 1:
            print(f"Warning: ensemble number must be >= 1, got {n}; using auto-detect from DyLAN.")
            return None
        return n
    except ValueError:
        print(f"Warning: invalid ensemble number {sys.argv[arg_index]!r}; using auto-detect from DyLAN.")
        return None

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "5"))
file_lock = asyncio.Lock()

MODEL_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-5": {"input": 0.00125, "output": 0.01},
    "gpt-5-mini": {"input": 0.00025, "output": 0.002},
}

def calculate_cost(model_name, prompt_tokens, completion_tokens):
    model_key = "gpt-4o"
    for key in MODEL_PRICING:
        if key in model_name.lower():
            model_key = key
            break
    p = MODEL_PRICING[model_key]
    return {
        "input_cost": (prompt_tokens / 1000.0) * p["input"],
        "output_cost": (completion_tokens / 1000.0) * p["output"],
        "total_cost": (prompt_tokens / 1000.0) * p["input"] + (completion_tokens / 1000.0) * p["output"],
    }

def set_rd_seed(seed):
    random.seed(seed)

def get_cot_sc_ensemble_size(base_dir_name):
    """Infer ensemble size from LLMLP run txt (avg API calls per question). Default 5."""
    dir_name = os.path.basename(base_dir_name) if os.path.dirname(base_dir_name) else base_dir_name
    if '_run' in dir_name:
        base_pattern = dir_name.rsplit('_run', 1)[0]
    else:
        base_pattern = dir_name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.join(script_dir, base_dir_name)) if os.path.dirname(base_dir_name) else script_dir
    pattern = os.path.join(parent_dir, base_pattern + '_run*', '*_43.txt')
    txt_files = glob.glob(pattern)
    if not txt_files:
        pattern = os.path.join(parent_dir, '*_run*', '*_43.txt')
        all_txt = glob.glob(pattern)
        txt_files = [f for f in all_txt if base_pattern in os.path.basename(os.path.dirname(f))]
    avg_values = []
    for txt_file in sorted(txt_files):
        try:
            with open(txt_file, 'r') as f:
                content = f.read()
                match = re.search(r'avg:\s*([\d.]+)\s*per question', content)
                if match:
                    avg_values.append(float(match.group(1)))
        except Exception:
            continue
    if not avg_values:
        return 5
    final_avg = sum(avg_values) / len(avg_values)
    return max(1, int(round(final_avg)))

_executor = SafeCodeExecutor(timeout=30)


async def generate_cot_response(question, model, require_code):
    user_message = {
        "role": "user",
        "content": (
            f"Here is the question:\n{question}\n\n"
            + (
                "Solve step by step. Provide your analysis, final answer (investor name or names), and Python code with a solve() function that returns a dict with investor_dates, comparison, and answer."
                if require_code
                else "Solve step by step and provide your analysis and final answer (investor name or names). Do not include code."
            )
        ),
    }
    reply, prompt_tokens, completion_tokens = await asyncio.to_thread(
        generate_answer_with_mode, [user_message], model, require_code
    )
    return reply, prompt_tokens, completion_tokens


async def process_single_question(idx, input_text, problem, model, ensemble_size, dir_name, exp_name):
    # Depth indicator from merged smfr_test.jsonl (2,3,4,5,6)
    depth = problem.get("source_split")
    try:
        completions = []
        answers = []  # list of parsed model_answer (string)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for _ in range(ensemble_size):
            reply, pt, ct = await generate_cot_response(input_text, model, REQUIRE_CODE)
            completions.append(reply)
            _, model_answer, _ = parse_model_output(reply)
            answers.append(model_answer if model_answer is not None else "")
            total_prompt_tokens += pt
            total_completion_tokens += ct

        # Direct vote: only among non-None, non-empty, non-'None' strings
        nonempty_answers = [a for a in answers if not is_meaningless_direct_vote_value(a)]
        if not nonempty_answers:
            final_answer, vote_count = "", 0
        else:
            final_answer, vote_count = most_frequent(
                nonempty_answers, lambda x, y: (x or "").strip() == (y or "").strip()
            )

        reference_answer = extract_reference_answer(problem)
        direct_full, direct_partial_count = evaluate_direct_answer(final_answer, reference_answer)
        score = 1.0 if direct_full else 0.0
        acc = score == 1.0

        # Code: run every completion's code, then ensemble-vote on meaningful executed answers
        # (thread pool: avoid blocking the asyncio event loop during sandbox execution)
        if REQUIRE_CODE:
            ce_result = await asyncio.to_thread(
                run_code_ensemble_eval, completions, reference_answer, _executor
            )
            code_full = ce_result["code_full"]
            code_partial = ce_result["code_partial"]
            code_failed = ce_result["code_failed"]
            code_eval_per_completion = ce_result["code_eval_per_completion"]
            code_ensemble = ce_result["code_ensemble"]
            code_exec_info = json.dumps({"code_ensemble": code_ensemble}, ensure_ascii=False)
        else:
            code_full, code_partial, code_failed = False, False, False
            code_exec_info = "skipped_by_mode"
            code_eval_per_completion = []
            code_ensemble = {}
        eval_details = {
            "input_text": input_text,
            "direct_full": direct_full,
            "direct_partial_count": direct_partial_count,
            "code_full": code_full,
            "code_partial": code_partial,
            "code_exec_info": code_exec_info,
        }
        data_to_save = {
            "completions": completions,
            "answers": answers,
            "final_result": final_answer,
            "vote_count": vote_count,
            "ground_truth": reference_answer,
            "correct": acc,
            "score": score,
            "eval_details": eval_details,
            "source_split": depth,
            "code_eval_per_completion": code_eval_per_completion,
            "code_ensemble": code_ensemble,
        }
        async with file_lock:
            await asyncio.to_thread(
                lambda: open(os.path.join(dir_name, exp_name + '_cot_sc.json'), 'a').write(json.dumps(data_to_save, ensure_ascii=False) + '\n')
            )
        cost_info = calculate_cost(model, total_prompt_tokens, total_completion_tokens)
        status = "✓ CORRECT" if acc else "✗ WRONG"
        print(f"Question {idx+1}: {status} | Predicted: {final_answer} | Ref: {reference_answer} | Votes: {vote_count}/{ensemble_size} | Cost: ${cost_info['total_cost']:.6f}")
        return {
            "idx": idx,
            "acc": acc,
            "score": score,
            "resp_cnt": ensemble_size,
            "completions": completions,
            "final_answer": final_answer,
            "vote_count": vote_count,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "input_cost": cost_info["input_cost"],
            "output_cost": cost_info["output_cost"],
            "total_cost": cost_info["total_cost"],
            "input_text": input_text,
            "direct_full": direct_full,
            "direct_partial_count": direct_partial_count,
            "code_full": code_full,
            "code_partial": code_partial,
            "code_failed": code_failed,
            "source_split": depth,
        }
    except Exception as e:
        print(f"Error processing question {idx+1}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "idx": idx,
            "acc": False,
            "score": 0.0,
            "resp_cnt": ensemble_size,
            "completions": None,
            "final_answer": None,
            "vote_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "input_text": input_text,
            "direct_full": False,
            "direct_partial_count": 0,
            "code_full": False,
            "code_partial": False,
            "code_failed": True,
            "source_split": depth,
        }


async def main():
    set_rd_seed(0)
    os.makedirs(DIR_NAME, exist_ok=True)
    # Ensemble size: optional argv[8] overrides (after CODE_EVAL_MODE); else DyLAN / default 5
    ensemble_override = parse_ensemble_override(8)
    if ensemble_override is not None:
        ensemble_size = ensemble_override
        print(f"Using CoT-SC ensemble size (from argument): {ensemble_size}")
    else:
        ensemble_size = get_cot_sc_ensemble_size(DIR_NAME)
        print(f"Using CoT-SC ensemble size: {ensemble_size}")

    qa_pairs = get_smfr_qa_pairs(QUERY_JSONL, require_code=REQUIRE_CODE)
    with open(os.path.join(DIR_NAME, EXP_NAME + '_cot_sc.json'), 'w') as f:
        f.write("")

    print(
        f"Processing {len(qa_pairs)} SMFR questions with CoT-SC (ensemble={ensemble_size}, max {MAX_CONCURRENT} concurrent, code_eval_mode={CODE_EVAL_MODE})"
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    running_correct = 0
    running_total = 0
    stats_lock = asyncio.Lock()

    async def process_with_semaphore(idx, input_text, problem):
        async with semaphore:
            result = await process_single_question(idx, input_text, problem, MODEL, ensemble_size, DIR_NAME, EXP_NAME)
            async with stats_lock:
                nonlocal running_correct, running_total
                running_total += 1
                if result["acc"]:
                    running_correct += 1
                acc_so_far = running_correct / running_total if running_total > 0 else 0.0
                print(f"  → Running Accuracy: {running_correct}/{running_total} = {acc_so_far:.4f}")
            return result

    tasks = [process_with_semaphore(idx, input_text, problem) for idx, (input_text, problem) in enumerate(qa_pairs)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x["idx"])

    total_count = len(qa_pairs)
    correct_count = sum(r["acc"] for r in results)
    final_accuracy = correct_count / total_count if total_count > 0 else 0.0
    total_cost = sum(r["total_cost"] for r in results)
    resp_cnts = sum(r["resp_cnt"] for r in results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)

    direct_full_cnt = sum(1 for r in results if r["direct_full"])
    direct_partial_cnt = sum(
        1 for r in results if (not r["direct_full"] and r["direct_partial_count"] > 0)
    )
    code_full_cnt = sum(1 for r in results if r["code_full"])
    code_partial_cnt = sum(1 for r in results if r["code_partial"])
    code_fail_cnt = sum(1 for r in results if r["code_failed"])

    from collections import defaultdict

    depth_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "direct_full": 0,
            "direct_partial": 0,
            "code_full": 0,
            "code_partial": 0,
            "code_failed": 0,
        }
    )

    for r in results:
        depth = r.get("source_split")
        if depth is None:
            continue
        s = depth_stats[depth]
        s["total"] += 1
        if r["acc"]:
            s["correct"] += 1
        if r["direct_full"]:
            s["direct_full"] += 1
        elif r["direct_partial_count"] > 0:
            s["direct_partial"] += 1
        if r["code_full"]:
            s["code_full"] += 1
        if r["code_partial"]:
            s["code_partial"] += 1
        if r["code_failed"]:
            s["code_failed"] += 1

    print(f"\n{'='*60}")
    print("FINAL EVALUATION RESULTS (SMFR CoT-SC)")
    print(f"{'='*60}")
    print(f"Total Questions: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy (AFlow metric - direct full only): {correct_count}/{total_count} = {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print("\nDirect Answer Metrics:")
    print(f"  Full Match:    {direct_full_cnt}/{total_count} ({(direct_full_cnt/total_count):.2%})")
    print(f"  Partial Match: {direct_partial_cnt}/{total_count} ({(direct_partial_cnt/total_count):.2%})")
    print("\nCode Output Metrics:")
    if REQUIRE_CODE:
        print(f"  Full Match:    {code_full_cnt}/{total_count} ({(code_full_cnt/total_count):.2%})")
        print(f"  Partial Match: {code_partial_cnt}/{total_count} ({(code_partial_cnt/total_count):.2%})")
        print(f"  Execution Failures: {code_fail_cnt}/{total_count} ({(code_fail_cnt/total_count):.2%})")
    else:
        print("  Skipped by CODE_EVAL_MODE=0")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Ensemble size: {ensemble_size}")
    print(f"{'='*60}")

    if depth_stats:
        print("\nPer-depth metrics (source_split):")
        for depth in sorted(depth_stats.keys()):
            s = depth_stats[depth]
            d_total = s["total"]
            if d_total == 0:
                continue
            d_correct = s["correct"]
            print(f"\n--- Depth {depth} ---")
            print(f"Total Questions: {d_total}")
            print(
                f"Accuracy (AFlow metric - direct full only): {d_correct}/{d_total} = {d_correct/d_total:.4f} ({(d_correct/d_total)*100:.2f}%)"
            )
            print("Direct Answer Metrics:")
            print(
                f"  Full Match:    {s['direct_full']}/{d_total} ({(s['direct_full']/d_total):.2%})"
            )
            print(
                f"  Partial Match: {s['direct_partial']}/{d_total} ({(s['direct_partial']/d_total):.2%})"
            )
            print("Code Output Metrics:")
            if REQUIRE_CODE:
                print(
                    f"  Full Match:    {s['code_full']}/{d_total} ({(s['code_full']/d_total):.2%})"
                )
                print(
                    f"  Partial Match: {s['code_partial']}/{d_total} ({(s['code_partial']/d_total):.2%})"
                )
                print(
                    f"  Execution Failures: {s['code_failed']}/{d_total} ({(s['code_failed']/d_total):.2%})"
                )
            else:
                print("  Skipped by CODE_EVAL_MODE=0")

    txt_path = os.path.join(DIR_NAME, EXP_NAME + '_cot_sc.txt')
    with open(txt_path, 'w') as f:
        f.write(str([r["acc"] for r in results]) + ' ' + str(final_accuracy) + '\n')
        f.write(str(resp_cnts) + ' ' + str(resp_cnts / total_count) + '\n')
        f.write(str(total_prompt_tokens) + '\n')
        f.write(str(total_completion_tokens) + '\n')
        f.write(f"Total cost: ${total_cost:.6f}\n")
        f.write(f"Accuracy (AFlow metric - direct full only): {correct_count}/{total_count} = {final_accuracy:.4f}\n")
        f.write(f"Direct Full Match: {direct_full_cnt}/{total_count}\n")
        f.write(f"Direct Partial Match: {direct_partial_cnt}/{total_count}\n")
        if REQUIRE_CODE:
            f.write(f"Code Full Match: {code_full_cnt}/{total_count}\n")
            f.write(f"Code Partial Match: {code_partial_cnt}/{total_count}\n")
            f.write(f"Code Execution Failures: {code_fail_cnt}/{total_count}\n")
        else:
            f.write("Code metrics: skipped by CODE_EVAL_MODE=0\n")
        if depth_stats:
            f.write("\nPer-depth metrics (source_split):\n")
            for depth in sorted(depth_stats.keys()):
                s = depth_stats[depth]
                d_total = s["total"]
                if d_total == 0:
                    continue
                d_correct = s["correct"]
                f.write(f"Depth {depth}:\n")
                f.write(
                    f"  Accuracy (AFlow metric - direct full only): {d_correct}/{d_total} = {d_correct/d_total:.4f}\n"
                )
                f.write(
                    f"  Direct Full Match: {s['direct_full']}/{d_total}\n"
                )
                f.write(
                    f"  Direct Partial Match: {s['direct_partial']}/{d_total}\n"
                )
                if REQUIRE_CODE:
                    f.write(
                        f"  Code Full Match: {s['code_full']}/{d_total}\n"
                    )
                    f.write(
                        f"  Code Partial Match: {s['code_partial']}/{d_total}\n"
                    )
                    f.write(
                        f"  Code Execution Failures: {s['code_failed']}/{d_total}\n"
                    )
                else:
                    f.write("  Code metrics: skipped by CODE_EVAL_MODE=0\n")
        f.write(f"Ensemble size used: {ensemble_size}\n")


if __name__ == "__main__":
    asyncio.run(main())
