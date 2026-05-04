import ast
import asyncio
import json
import os
import openai
import random
import sys
from LLMLP import LLMLP
from utils import *
from swe_utils import run_swebench_evaluation, extract_xml

# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

QUERY_JSONL = sys.argv[1]
EXP_NAME = sys.argv[2]
MODEL = sys.argv[3]

ACTIVATION = "listwise"
TYPE = "code_patch"
DIR_NAME = sys.argv[4]
ROLES = ast.literal_eval(sys.argv[5])
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)

# Optional run_id parameter for multiple runs (to avoid file conflicts)
RUN_ID = sys.argv[6] if len(sys.argv) > 6 else None
if RUN_ID:
    DIR_NAME = DIR_NAME + '_run' + str(RUN_ID)

# Optional evaluation parameters
JUDGE_PATH = sys.argv[7] if len(sys.argv) > 7 else None
FILE_PATH = sys.argv[8] if len(sys.argv) > 8 else None
ENABLE_EVALUATION = JUDGE_PATH is not None and FILE_PATH is not None

# Maximum number of concurrent async tasks (adjust based on API rate limits)
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "50"))

# Lock for thread-safe file writing
file_lock = asyncio.Lock()

# OpenAI pricing per 1K tokens (as of 2024)
# Prices are in USD per 1000 tokens
MODEL_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
    "gpt-5": {"input": 0.00125, "output": 0.01},  # Assuming same as gpt-4o, update if needed
}

def calculate_cost(model_name, prompt_tokens, completion_tokens):
    """Calculate API cost based on model and token usage."""
    # Normalize model name to find pricing
    model_key = None
    for key in MODEL_PRICING.keys():
        if key in model_name.lower():
            model_key = key
            break
    
    # Default to gpt-4o pricing if model not found
    if model_key is None:
        model_key = "gpt-4o"
    pricing = MODEL_PRICING[model_key]
    input_cost = (prompt_tokens / 1000.0) * pricing["input"]
    output_cost = (completion_tokens / 1000.0) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model_key
    }

def set_rd_seed(seed):
    random.seed(seed)

async def process_single_question(idx, instance_id, text, expected_patch, model, roles, activation, qtype, dir_name, exp_name,
                                  judge_path=None, file_path=None, enable_evaluation=False):
    """Process a single question asynchronously."""
    try:
        # Run synchronous LLMLP operations in a thread pool to avoid blocking
        def run_llmlp():
            llmlp = LLMLP(model, len(roles), roles, 3, activation, qtype, model)
            llmlp.zero_grad()
            res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(text)
            imp_score = llmlp.backward(res)
            return res, resp_cnt, completions, prompt_tokens, completion_tokens, imp_score
        
        # Initialize variables for retry loop
        res = None
        resp_cnt = 0
        completions = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        imp_score = 0.0
        evaluation_score = 0.0
        max_retries = 10
        
        # Retry loop for evaluation with docker error handling
        if enable_evaluation and judge_path and file_path:
            code_snippet = extract_xml(text, 'code').strip()
            solution_name = f"llmlp_{idx}"
            
            for attempt in range(max_retries):
                # Generate new result if this is a retry (attempt > 0) or first attempt
                if attempt == 0 or (res is None):
                    # Execute synchronous code in thread pool
                    res, resp_cnt, completions, prompt_tokens, completion_tokens, imp_score = await asyncio.to_thread(run_llmlp)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                
                if not res:
                    print(f"Instance {idx+1} ({instance_id}): No result generated, skipping evaluation")
                    break
                
                try:
                    # Run evaluation
                    eval_result = await run_swebench_evaluation(
                        judge_path, instance_id, res, "llmlp", solution_name, 
                        code_snippet, file_path
                    )
                    evaluation_score = eval_result['score']
                    error_instances = eval_result['error_instances']
                    
                    print(f"Instance {idx+1} ({instance_id}): Attempt {attempt + 1}/{max_retries} - Evaluation score: {evaluation_score}, Error instances: {error_instances}")
                    
                    # If error_instances is 1, it's a docker problem, retry with new result
                    if error_instances == 1:
                        if attempt < max_retries - 1:
                            print(f"Instance {idx+1} ({instance_id}): Docker error detected, regenerating result and retrying...")
                            res = None  # Force regeneration on next iteration
                            await asyncio.sleep(1)  # Brief delay before retry
                            continue
                        else:
                            print(f"Instance {idx+1} ({instance_id}): Max retries reached, keeping last result")
                            break
                    else:
                        # Success (error_instances != 1), exit retry loop
                        print(f"Instance {idx+1} ({instance_id}): Evaluation completed successfully")
                        break
                        
                except Exception as e:
                    print(f"Instance {idx+1} ({instance_id}): Evaluation error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        res = None  # Force regeneration on next iteration
                        await asyncio.sleep(1)
                        continue
                    else:
                        evaluation_score = 0.0
                        break
        else:
            # No evaluation enabled, just run LLMLP once
            res, resp_cnt, completions, prompt_tokens, completion_tokens, imp_score = await asyncio.to_thread(run_llmlp)
            total_prompt_tokens = prompt_tokens
            total_completion_tokens = completion_tokens
        
        # Prepare data to save (include completions and final result)
        data_to_save = {
            'completions': completions,
            'final_result': res if res else "",
            'expected_patch': expected_patch,
            'instance_id': instance_id,
            'evaluation_score': evaluation_score
        }
        
        # Thread-safe file writing
        async with file_lock:
            # Use asyncio.to_thread for file I/O
            await asyncio.to_thread(
                lambda: open(dir_name+'/'+exp_name+'_'+str(len(roles))+'3.json', 'a').write(json.dumps(data_to_save) + '\n')
            )
        
        # Calculate cost for this question
        cost_info = calculate_cost(model, total_prompt_tokens, total_completion_tokens)
        
        # Print evaluation result immediately
        eval_info = f" | Eval: {evaluation_score:.2f}" if enable_evaluation else ""
        print(f"Instance {idx+1} ({instance_id}): Generated patch{eval_info} | Cost: ${cost_info['total_cost']:.6f}")
        
        return {
            'idx': idx,
            'instance_id': instance_id,
            'resp_cnt': resp_cnt,
            'importance': imp_score,
            'completions': completions,
            'patch': res if res else "",
            'expected_patch': expected_patch,
            'evaluation_score': evaluation_score,
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'input_cost': cost_info['input_cost'],
            'output_cost': cost_info['output_cost'],
            'total_cost': cost_info['total_cost']
        }
    except Exception as e:
        print(f"Error processing instance {idx+1}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'idx': idx,
            'instance_id': instance_id,
            'resp_cnt': 0,
            'importance': [0] * (len(roles) * 3),
            'completions': None,
            'patch': "",
            'expected_patch': expected_patch,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0
        }

async def main():
    set_rd_seed(0)
    assert len(ROLES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)
    
    # Setup evaluation directories if enabled
    if ENABLE_EVALUATION:
        os.makedirs(f'{JUDGE_PATH}/results', exist_ok=True)
        os.makedirs(f'{JUDGE_PATH}/reports', exist_ok=True)
        print(f"Docker evaluation enabled. Judge path: {JUDGE_PATH}, File path: {FILE_PATH}")
    else:
        print("Docker evaluation disabled. Set JUDGE_PATH and FILE_PATH to enable.")

    qa_pairs = get_swe_qa_pairs(QUERY_JSONL)
    # Remove the limit for full dataset processing
    # qa_pairs = qa_pairs[:5]

    # Initialize JSON file
    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.json', 'w') as f:
        f.write("")

    print(f"Processing {len(qa_pairs)} instances with async (max {MAX_CONCURRENT} concurrent)")

    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    stats_lock = asyncio.Lock()
    
    async def process_with_semaphore(idx, instance_id, text, expected_patch):
        async with semaphore:
            result = await process_single_question(
                idx, instance_id, text, expected_patch, MODEL, ROLES, ACTIVATION, TYPE, DIR_NAME, EXP_NAME,
                JUDGE_PATH, FILE_PATH, ENABLE_EVALUATION
            )
            return result
    
    # Create all tasks
    tasks = [
        process_with_semaphore(idx, instance_id, text, expected_patch)
        for idx, (instance_id, text, expected_patch) in enumerate(qa_pairs)
    ]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Sort results by index to maintain order
    results.sort(key=lambda x: x['idx'])

    # Aggregate results
    resp_cnts = sum(r['resp_cnt'] for r in results)
    importances = [r['importance'] for r in results]
    completion_list = [r['completions'] for r in results]
    total_prompt_tokens = sum(r['prompt_tokens'] for r in results)
    total_completion_tokens = sum(r['completion_tokens'] for r in results)
    total_input_cost = sum(r['input_cost'] for r in results)
    total_output_cost = sum(r['output_cost'] for r in results)
    total_cost = sum(r['total_cost'] for r in results)
    
    # Aggregate evaluation scores if enabled
    total_evaluation_score = 0.0
    if ENABLE_EVALUATION:
        evaluation_scores = [r.get('evaluation_score', 0.0) for r in results]
        total_evaluation_score = sum(evaluation_scores)
        passed_count = sum(1 for s in evaluation_scores if s >= 1.0)
    
    total_count = len(qa_pairs)

    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Instances: {total_count}")
    print(f"\nAPI Usage:")
    print(f"  Total API calls: {resp_cnts} (avg: {resp_cnts/total_count:.2f} per instance)")
    print(f"  Total tokens: {total_prompt_tokens + total_completion_tokens:,} (prompt: {total_prompt_tokens:,}, completion: {total_completion_tokens:,})")
    print(f"\nCost Summary:")
    print(f"  Total cost: ${total_cost:.4f} (input: ${total_input_cost:.4f}, output: ${total_output_cost:.4f})")
    print(f"  Cost per instance: ${total_cost/total_count:.6f}")
    if ENABLE_EVALUATION:
        print(f"\nDocker Evaluation Results:")
        print(f"  Passed instances: {passed_count}/{total_count} ({passed_count/total_count*100:.2f}%)")
        print(f"  Total score: {total_evaluation_score:.2f}")
        print(f"  Average score: {total_evaluation_score/total_count:.4f}")
    print(f"{'='*60}")

    # Write final results
    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.txt', 'w') as f:
        f.write(str(resp_cnts) + " " + str(resp_cnts/len(qa_pairs)) + '\n')
        f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos)/len(qa_pairs) for pos in zip(*importances)]) + '\n')
        f.write(str(total_prompt_tokens) + '\n')
        f.write(str(total_completion_tokens) + '\n')
        # Add cost information
        f.write(f"Total cost: ${total_cost:.6f}\n")
        f.write(f"Input cost: ${total_input_cost:.6f}\n")
        f.write(f"Output cost: ${total_output_cost:.6f}\n")
        f.write(f"Cost per instance: ${total_cost/len(qa_pairs):.6f}\n")
        # Add detailed evaluation results
        f.write(f"\n{'='*60}\n")
        f.write(f"FINAL EVALUATION RESULTS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total Instances: {total_count}\n")
        f.write(f"\nAPI Usage:\n")
        f.write(f"  Total API calls: {resp_cnts} (avg: {resp_cnts/total_count:.2f} per instance)\n")
        f.write(f"  Total tokens: {total_prompt_tokens + total_completion_tokens:,} (prompt: {total_prompt_tokens:,}, completion: {total_completion_tokens:,})\n")
        f.write(f"\nCost Summary:\n")
        f.write(f"  Total cost: ${total_cost:.4f} (input: ${total_input_cost:.4f}, output: ${total_output_cost:.4f})\n")
        f.write(f"  Cost per instance: ${total_cost/total_count:.6f}\n")
        if ENABLE_EVALUATION:
            f.write(f"\nDocker Evaluation Results:\n")
            f.write(f"  Passed instances: {passed_count}/{total_count} ({passed_count/total_count*100:.2f}%)\n")
            f.write(f"  Total score: {total_evaluation_score:.2f}\n")
            f.write(f"  Average score: {total_evaluation_score/total_count:.4f}\n")
        f.write(f"{'='*60}\n")
    
    # Write results in JSONL format for evaluation
    results_jsonl = [
        {
            "instance_id": r['instance_id'],
            "patch": r['patch'],
            "expected_patch": r['expected_patch']
        }
        for r in results
    ]
    write_jsonl(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.jsonl', results_jsonl)

if __name__ == "__main__":
    asyncio.run(main())
