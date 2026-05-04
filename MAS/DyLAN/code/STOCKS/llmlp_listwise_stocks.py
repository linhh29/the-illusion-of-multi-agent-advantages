# -*- coding: utf-8 -*-
"""
STOCKS evaluation in DyLAN. Aligned with AFlow benchmarks/stocks.py:
- Same data format: stocks_validate.jsonl / stocks_test.jsonl (problem, answer)
- Same Instruction appended to problem
- Same scoring: 1.0 if direct answer full match (all reference names in model answer), else 0.0
- Same code execution via SafeCodeExecutor (eval_details logged)
"""
import ast
import asyncio
import json
import os
import random
import sys
from LLMLP import LLMLP
from utils import (
    get_stocks_qa_pairs,
    extract_reference_answer,
    parse_model_output,
    evaluate_direct_answer,
    evaluate_code_output,
    Instruction,
)
from safe_code_executor import SafeCodeExecutor

QUERY_JSONL = sys.argv[1]
EXP_NAME = sys.argv[2]
MODEL = sys.argv[3]
ACTIVATION = "listwise"
TYPE = "stocks"
DIR_NAME = sys.argv[4]
ROLES = ast.literal_eval(sys.argv[5])
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)

RUN_ID = sys.argv[6] if len(sys.argv) > 6 else None
if RUN_ID:
    DIR_NAME = DIR_NAME + '_run' + str(RUN_ID)
CODE_EVAL_MODE = int(sys.argv[7]) if len(sys.argv) > 7 else 1
REQUIRE_CODE = CODE_EVAL_MODE == 1

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "5"))
file_lock = asyncio.Lock()

MODEL_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
    "gpt-5": {"input": 0.00125, "output": 0.01},
    "gpt-5-mini": {"input": 0.00025, "output": 0.002},
}


def calculate_cost(model_name, prompt_tokens, completion_tokens):
    model_key = None
    for key in MODEL_PRICING:
        if key in model_name.lower():
            model_key = key
            break
    if model_key is None:
        model_key = "gpt-4o"
    p = MODEL_PRICING[model_key]
    return {
        "input_cost": (prompt_tokens / 1000.0) * p["input"],
        "output_cost": (completion_tokens / 1000.0) * p["output"],
        "total_cost": (prompt_tokens / 1000.0) * p["input"] + (completion_tokens / 1000.0) * p["output"],
    }


def set_rd_seed(seed):
    random.seed(seed)


# Executor for code evaluation (same timeout as AFlow)
_executor = SafeCodeExecutor(timeout=30)


async def process_single_question(idx, input_text, problem, model, roles, dir_name, exp_name):
    # Depth indicator from merged stocks_test.jsonl (2,3,4,5,6)
    depth = problem.get("source_split")
    try:
        def run_llmlp():
            llmlp = LLMLP(model, len(roles), roles, 3, ACTIVATION, TYPE, model)
            llmlp.zero_grad()
            res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(input_text)
            imp_score = llmlp.backward(res)
            return res, resp_cnt, completions, prompt_tokens, completion_tokens, imp_score

        res, resp_cnt, completions, prompt_tokens, completion_tokens, imp_score = await asyncio.to_thread(run_llmlp)

        reference_answer = extract_reference_answer(problem)
        raw_str, model_answer, code = parse_model_output(res)

        direct_full, direct_partial_count = evaluate_direct_answer(model_answer, reference_answer)
        if REQUIRE_CODE:
            code_full, code_partial, code_failed, code_exec_info = evaluate_code_output(code, reference_answer, _executor)
        else:
            code_full, code_partial, code_failed, code_exec_info = False, False, False, "skipped_by_mode"

        # Same scoring as AFlow: 1.0 if direct_full else 0.0
        score = 1.0 if direct_full else 0.0
        acc = score == 1.0

        eval_details = {
            "input_text": input_text,
            "direct_full": direct_full,
            "direct_partial_count": direct_partial_count,
            "code_full": code_full,
            "code_partial": code_partial,
            "code_exec_info": code_exec_info,
        }

        data_to_save = {
            "input_text": input_text,
            "completions": completions,
            "final_result": res,
            "ground_truth": reference_answer,
            "correct": acc,
            "score": score,
            "eval_details": eval_details,
            "source_split": depth,
            "exception": False,
        }

        async with file_lock:
            await asyncio.to_thread(
                lambda: open(os.path.join(dir_name, exp_name + '_' + str(len(roles)) + '3.json'), 'a').write(json.dumps(data_to_save, ensure_ascii=False) + '\n')
            )

        cost_info = calculate_cost(model, prompt_tokens, completion_tokens)
        status = "✓ CORRECT" if acc else "✗ WRONG"
        print(f"Question {idx+1}: {status} | Score: {score} | Ref: {reference_answer} | Cost: ${cost_info['total_cost']:.6f}")

        return {
            "idx": idx,
            "acc": acc,
            "score": score,
            "resp_cnt": resp_cnt,
            "importance": imp_score,
            "completions": completions,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
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
            "exception": False,
        }
    except Exception as e:
        print(f"Error processing question {idx+1}: {e}")
        import traceback

        traceback.print_exc()
        err_type = type(e).__name__
        err_msg = str(e)
        try:
            ground_truth_err = extract_reference_answer(problem)
        except Exception:
            ground_truth_err = None
        data_to_save = {
            "input_text": input_text,
            "completions": None,
            "final_result": None,
            "ground_truth": ground_truth_err,
            "correct": False,
            "score": 0.0,
            "eval_details": None,
            "source_split": depth,
            "exception": True,
            "error": err_msg,
            "error_type": err_type,
        }
        async with file_lock:
            await asyncio.to_thread(
                lambda: open(
                    os.path.join(dir_name, exp_name + '_' + str(len(roles)) + '3.json'), 'a'
                ).write(json.dumps(data_to_save, ensure_ascii=False) + '\n')
            )
        return {
            "idx": idx,
            "acc": False,
            "score": 0.0,
            "resp_cnt": 0,
            "importance": [0] * (len(roles) * 3),
            "completions": None,
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
            "code_failed": False,
            "source_split": depth,
            "exception": True,
        }


async def main():
    set_rd_seed(0)
    assert len(ROLES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)

    qa_pairs = get_stocks_qa_pairs(QUERY_JSONL, require_code=REQUIRE_CODE)

    with open(os.path.join(DIR_NAME, EXP_NAME + '_' + str(len(ROLES)) + '3.json'), 'w') as f:
        f.write("")

    print(f"Processing {len(qa_pairs)} STOCKS questions (max {MAX_CONCURRENT} concurrent, code_eval_mode={CODE_EVAL_MODE})")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    running_correct = 0
    running_total = 0
    stats_lock = asyncio.Lock()

    async def process_with_semaphore(idx, input_text, problem):
        async with semaphore:
            result = await process_single_question(idx, input_text, problem, MODEL, ROLES, DIR_NAME, EXP_NAME)
            async with stats_lock:
                nonlocal running_correct, running_total
                running_total += 1
                if result["acc"]:
                    running_correct += 1
                acc_so_far = running_correct / running_total if running_total > 0 else 0.0
                print(f"  → Running Accuracy: {running_correct}/{running_total} = {acc_so_far:.4f}")
            return result

    tasks = [
        process_with_semaphore(idx, input_text, problem)
        for idx, (input_text, problem) in enumerate(qa_pairs)
    ]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x["idx"])

    total_count = len(qa_pairs)
    correct_count = sum(r["acc"] for r in results)
    final_accuracy = correct_count / total_count if total_count > 0 else 0.0
    total_cost = sum(r["total_cost"] for r in results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    resp_cnts = sum(r["resp_cnt"] for r in results)

    # Aggregate stocks-specific metrics (direct/code full/partial/execution failure)
    direct_full_cnt = sum(1 for r in results if r["direct_full"])
    direct_partial_cnt = sum(
        1 for r in results if (not r["direct_full"] and r["direct_partial_count"] > 0)
    )
    code_full_cnt = sum(1 for r in results if r["code_full"])
    code_partial_cnt = sum(1 for r in results if r["code_partial"])
    code_fail_cnt = sum(1 for r in results if r["code_failed"])
    exception_cnt = sum(1 for r in results if r.get("exception"))

    # Per-depth (source_split) aggregation
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
            "exception": 0,
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
        if r.get("exception"):
            s["exception"] += 1

    print(f"\n{'='*60}")
    print("FINAL EVALUATION RESULTS (STOCKS, same metric as AFlow)")
    print(f"{'='*60}")
    print(f"Total Questions: {total_count}")
    print(f"Pipeline exceptions (LLM/parse/etc., not scored): {exception_cnt}/{total_count}")
    print(f"Correct (direct full match): {correct_count}")
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
    print(f"{'='*60}")

    # Per-depth metrics (by source_split)
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
            print(
                f"  Pipeline exceptions: {s['exception']}/{d_total} ({(s['exception']/d_total):.2%})"
            )

    summary_path = os.path.join(DIR_NAME, EXP_NAME + '_' + str(len(ROLES)) + '3.txt')
    with open(summary_path, 'w') as f:
        f.write(str([r["acc"] for r in results]) + ' ' + str(final_accuracy) + '\n')
        f.write(str(resp_cnts) + ' ' + str(resp_cnts / total_count) + '\n')
        f.write(json.dumps([r["importance"] for r in results]) + '\n')
        f.write(str(total_prompt_tokens) + '\n')
        f.write(str(total_completion_tokens) + '\n')
        f.write(f"Total cost: ${total_cost:.6f}\n")
        f.write(f"Accuracy (AFlow metric - direct full only): {correct_count}/{total_count} = {final_accuracy:.4f}\n")
        f.write(f"Pipeline exceptions: {exception_cnt}/{total_count}\n")
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
                f.write(
                    f"  Pipeline exceptions: {s['exception']}/{d_total}\n"
                )


if __name__ == "__main__":
    asyncio.run(main())
