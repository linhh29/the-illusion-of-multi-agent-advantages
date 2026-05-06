import ast
import asyncio
import json
import os
import openai
import random
import sys
import re
import glob
from utils import *
from prompt_lib import SYSTEM_PROMPT_SWE, construct_message, ROLE_MAP, construct_ensemble_message
from swe_utils import run_swebench_evaluation, extract_xml

# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

QUERY_JSONL = sys.argv[1]
EXP_NAME = sys.argv[2]
MODEL = sys.argv[3]
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


def parse_ensemble_override(arg_index):
    """
    Optional argv[arg_index]: strict ensemble size for CoT-SC.
    If missing or empty, return None (use MAS-derived size or default 5).
    Placed after JUDGE_PATH / FILE_PATH for SWE.
    """
    if len(sys.argv) <= arg_index:
        return None
    s = sys.argv[arg_index].strip()
    if not s:
        return None
    try:
        n = int(s)
        if n < 1:
            print(f"Warning: ensemble number must be >= 1, got {n}; using default ensemble size.")
            return None
        return n
    except ValueError:
        print(f"Warning: invalid ensemble number {sys.argv[arg_index]!r}; using default ensemble size.")
        return None

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

def get_cot_sc_ensemble_size(base_dir_name):
    """
    Read avg values from all run directories and calculate the average.
    The avg value is from lines like: "Total API calls: 842 (avg: 5.07 per question)"
    """
    # Get the directory name (without path)
    dir_name = os.path.basename(base_dir_name) if os.path.dirname(base_dir_name) else base_dir_name
    
    # Extract the base pattern (without run number)
    if '_run' in dir_name:
        base_pattern = dir_name.rsplit('_run', 1)[0]
    else:
        base_pattern = dir_name
    
    # Find the parent directory (where all run directories are located)
    if os.path.isabs(base_dir_name):
        parent_dir = os.path.dirname(base_dir_name)
    else:
        # If relative path, assume it's in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.dirname(base_dir_name):
            parent_dir = os.path.dirname(os.path.join(script_dir, base_dir_name))
        else:
            parent_dir = script_dir
    
    # Search for all run directories matching the pattern
    # Try the exact pattern first
    pattern = os.path.join(parent_dir, base_pattern + '_run*', EXP_NAME + '_*3.txt')
    txt_files = glob.glob(pattern)
    
    # If no files found, try a broader search
    if len(txt_files) == 0:
        # Try searching in the parent directory directly for any matching pattern
        pattern = os.path.join(parent_dir, '*_run*', EXP_NAME + '_*3.txt')
        all_txt_files = glob.glob(pattern)
        # Filter to only include files that match the base pattern
        txt_files = [f for f in all_txt_files if base_pattern in os.path.basename(os.path.dirname(f))]
    
    avg_values = []
    for txt_file in sorted(txt_files):  # Sort to ensure consistent ordering
        try:
            with open(txt_file, 'r') as f:
                content = f.read()
                # Look for pattern: "avg: X.XX per question" or "X.XX per question"
                match = re.search(r'avg:\s*([\d.]+)\s*per|([\d.]+)\s+[\d.]+\s+([\d.]+)', content)
                if match:
                    # Try to get avg value from different patterns
                    avg_val = None
                    if match.group(1):
                        avg_val = float(match.group(1))
                    elif match.group(3):
                        # Second number in line like "842 5.07"
                        lines = content.split('\n')
                        for line in lines:
                            if re.match(r'^\d+\s+[\d.]+', line):
                                parts = line.split()
                                if len(parts) >= 2:
                                    avg_val = float(parts[1])
                                    break
                    
                    if avg_val:
                        avg_values.append(avg_val)
                        print(f"Found avg value {avg_val:.2f} from {txt_file}")
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
            continue
    
    if len(avg_values) == 0:
        print(f"Warning: No avg values found in run directories. Using default ensemble size of 5.")
        print(f"Searched in: {parent_dir}")
        print(f"Pattern used: {base_pattern}_run*/{EXP_NAME}_*3.txt")
        return 5
    
    final_avg = sum(avg_values) / len(avg_values)
    ensemble_size = int(round(final_avg))
    print(f"Average ensemble size from {len(avg_values)} run(s): {final_avg:.2f} -> {ensemble_size}")
    return ensemble_size

async def generate_cot_response(text, model):
    """Generate a single CoT response for the problem."""
    # For CoT-SC, we use a simple system prompt without specific roles
    # sys_prompt = SYSTEM_PROMPT_SWE
    
    # Construct the user message with step-by-step instruction
    user_message = {
        "role": "user",
        "content": f"{text}\n\nPlease solve this problem step by step, showing your reasoning process. Provide a patch in unified diff format, wrapped in <patch> tags or a code block."
    }
    
    contexts = []
    contexts.append(user_message)
    
    # Generate answer using asyncio.to_thread for better async performance
    reply, prompt_tokens, completion_tokens = await asyncio.to_thread(generate_answer, contexts, model)
    patch = extract_patch(reply)
    
    return reply, patch, prompt_tokens, completion_tokens

async def generate_llm_ensemble(patches, question, model):
    """Use LLM to ensemble multiple patches into a final best patch."""
    sys_prompt = SYSTEM_PROMPT_SWE
    ensemble_message = construct_ensemble_message(patches, question)
    
    contexts = [{"role": "system", "content": sys_prompt}]
    contexts.append(ensemble_message)
    
    # Generate ensemble answer
    reply, prompt_tokens, completion_tokens = await asyncio.to_thread(generate_answer, contexts, model)
    final_patch = extract_patch(reply)
    
    return final_patch, reply, prompt_tokens, completion_tokens

async def process_single_question(idx, instance_id, text, expected_patch, model, ensemble_size, dir_name, exp_name, 
                                  judge_path=None, file_path=None, enable_evaluation=False):
    """Process a single question with CoT-SC (multiple CoT samples + LLM ensemble)."""
    try:
        # Initialize variables for retry loop
        completions = []
        patches = []
        final_patch = None
        ensemble_reply = ""
        vote_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        evaluation_score = 0.0
        max_retries = 10
        valid_patches = []  # Initialize for use in return statement
        
        # Retry loop for evaluation with docker error handling
        if enable_evaluation and judge_path and file_path:
            code_snippet = extract_xml(text, 'code').strip()
            solution_name = f"cot_sc_{idx}"
            
            for attempt in range(max_retries):
                # Generate new CoT responses and ensemble if this is a retry (attempt > 0) or first attempt
                if attempt == 0 or (final_patch is None):
                    # Generate multiple CoT responses
                    completions = []
                    patches = []
                    attempt_prompt_tokens = 0
                    attempt_completion_tokens = 0
                    
                    for i in range(ensemble_size):
                        reply, patch, prompt_tokens, completion_tokens = await generate_cot_response(text, model)
                        completions.append(reply)
                        patches.append(patch if patch else "")
                        attempt_prompt_tokens += prompt_tokens
                        attempt_completion_tokens += completion_tokens
                    
                    # Use LLM ensemble instead of simple voting
                    # Filter out empty patches
                    valid_patches = [p for p in patches if p and p.strip()]  # Update valid_patches
                    if len(valid_patches) == 0:
                        final_patch = ""
                        ensemble_reply = "No valid patches generated"
                        ensemble_prompt_tokens = 0
                        ensemble_completion_tokens = 0
                    elif len(valid_patches) == 1:
                        final_patch = valid_patches[0]
                        ensemble_reply = "Only one valid patch generated"
                        ensemble_prompt_tokens = 0
                        ensemble_completion_tokens = 0
                    else:
                        # Use LLM to ensemble multiple patches
                        final_patch, ensemble_reply, ensemble_prompt_tokens, ensemble_completion_tokens = await generate_llm_ensemble(
                            valid_patches, text, model
                        )
                        attempt_prompt_tokens += ensemble_prompt_tokens
                        attempt_completion_tokens += ensemble_completion_tokens
                    
                    total_prompt_tokens += attempt_prompt_tokens
                    total_completion_tokens += attempt_completion_tokens
                    
                    # Count votes for statistics (how many patches were similar to final)
                    vote_count = sum(1 for p in patches if p and p.strip() == final_patch.strip()) if final_patch else 0
                
                if not final_patch:
                    print(f"Instance {idx+1} ({instance_id}): No patch generated, skipping evaluation")
                    break
                
                try:
                    # Run evaluation
                    eval_result = await run_swebench_evaluation(
                        judge_path, instance_id, final_patch, "cot_sc", solution_name, 
                        code_snippet, file_path
                    )
                    evaluation_score = eval_result['score']
                    error_instances = eval_result['error_instances']
                    
                    print(f"Instance {idx+1} ({instance_id}): Attempt {attempt + 1}/{max_retries} - Evaluation score: {evaluation_score}, Error instances: {error_instances}")
                    
                    # If error_instances is 1, it's a docker problem, retry with new result
                    if error_instances == 1:
                        if attempt < max_retries - 1:
                            print(f"Instance {idx+1} ({instance_id}): Docker error detected, regenerating CoT responses and ensemble, retrying...")
                            final_patch = None  # Force regeneration on next iteration
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
                        final_patch = None  # Force regeneration on next iteration
                        await asyncio.sleep(1)
                        continue
                    else:
                        evaluation_score = 0.0
                        break
        else:
            # No evaluation enabled, just generate CoT responses and ensemble once
            completions = []
            patches = []
            
            for i in range(ensemble_size):
                reply, patch, prompt_tokens, completion_tokens = await generate_cot_response(text, model)
                completions.append(reply)
                patches.append(patch if patch else "")
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
            
            # Use LLM ensemble instead of simple voting
            # Filter out empty patches
            valid_patches = [p for p in patches if p and p.strip()]
            if len(valid_patches) == 0:
                final_patch = ""
                ensemble_reply = "No valid patches generated"
                ensemble_prompt_tokens = 0
                ensemble_completion_tokens = 0
            elif len(valid_patches) == 1:
                final_patch = valid_patches[0]
                ensemble_reply = "Only one valid patch generated"
                ensemble_prompt_tokens = 0
                ensemble_completion_tokens = 0
            else:
                # Use LLM to ensemble multiple patches
                final_patch, ensemble_reply, ensemble_prompt_tokens, ensemble_completion_tokens = await generate_llm_ensemble(
                    valid_patches, text, model
                )
                total_prompt_tokens += ensemble_prompt_tokens
                total_completion_tokens += ensemble_completion_tokens
            
            # Count votes for statistics (how many patches were similar to final)
            vote_count = sum(1 for p in patches if p and p.strip() == final_patch.strip()) if final_patch else 0
        
        # Prepare data to save
        data_to_save = {
            'instance_id': instance_id,
            'completions': completions,
            'patches': patches,
            'final_patch': final_patch,
            'ensemble_reply': ensemble_reply,
            'vote_count': vote_count,
            'expected_patch': expected_patch,
            'evaluation_score': evaluation_score
        }
        
        # Thread-safe file writing
        async with file_lock:
            await asyncio.to_thread(
                lambda: open(dir_name+'/'+exp_name+'_cot_sc.json', 'a').write(json.dumps(data_to_save) + '\n')
            )
        
        # Calculate cost for this question
        cost_info = calculate_cost(model, total_prompt_tokens, total_completion_tokens)
        
        # Print evaluation result immediately
        eval_info = f" | Eval: {evaluation_score:.2f}" if enable_evaluation else ""
        print(f"Instance {idx+1} ({instance_id}): Generated patch | Votes: {vote_count}/{ensemble_size}{eval_info} | Cost: ${cost_info['total_cost']:.6f}")
        
        return {
            'idx': idx,
            'instance_id': instance_id,
            'resp_cnt': ensemble_size + (1 if len(valid_patches) > 1 else 0),  # Include ensemble call
            'completions': completions,
            'patches': patches,
            'final_patch': final_patch,
            'vote_count': vote_count,
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
            'resp_cnt': ensemble_size,
            'completions': None,
            'patches': None,
            'final_patch': None,
            'vote_count': 0,
            'expected_patch': expected_patch,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0
        }

async def main():
    set_rd_seed(0)
    os.makedirs(DIR_NAME, exist_ok=True)
    
    # Setup evaluation directories if enabled
    if ENABLE_EVALUATION:
        os.makedirs(f'{JUDGE_PATH}/results', exist_ok=True)
        os.makedirs(f'{JUDGE_PATH}/reports', exist_ok=True)
        print(f"Docker evaluation enabled. Judge path: {JUDGE_PATH}, File path: {FILE_PATH}")
    else:
        print("Docker evaluation disabled. Set JUDGE_PATH and FILE_PATH to enable.")

    # Ensemble size: optional argv[9] overrides; else use default ensemble size of 5
    ensemble_override = parse_ensemble_override(9)
    if ensemble_override is not None:
        ensemble_size = ensemble_override
        print(f"Using CoT-SC ensemble size (from argument): {ensemble_size}")
    else:
        ensemble_size = get_cot_sc_ensemble_size(DIR_NAME)
        print(f"Using CoT-SC ensemble size: {ensemble_size}")

    qa_pairs = get_swe_qa_pairs(QUERY_JSONL)
    # Remove the limit for full dataset processing
    # qa_pairs = qa_pairs[:5]

    # Initialize JSON file
    with open(DIR_NAME+'/'+EXP_NAME+'_cot_sc.json', 'w') as f:
        f.write("")

    print(f"Processing {len(qa_pairs)} instances with CoT-SC (ensemble size: {ensemble_size}, max {MAX_CONCURRENT} concurrent)")

    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    stats_lock = asyncio.Lock()
    
    async def process_with_semaphore(idx, instance_id, text, expected_patch):
        async with semaphore:
            result = await process_single_question(
                idx, instance_id, text, expected_patch, MODEL, ensemble_size, DIR_NAME, EXP_NAME,
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
    with open(DIR_NAME+'/'+EXP_NAME+'_cot_sc.txt', 'w') as f:
        f.write(str(resp_cnts) + " " + str(resp_cnts/total_count) + '\n')
        f.write(str(total_prompt_tokens) + '\n')
        f.write(str(total_completion_tokens) + '\n')
        # Add cost information
        f.write(f"Total cost: ${total_cost:.6f}\n")
        f.write(f"Input cost: ${total_input_cost:.6f}\n")
        f.write(f"Output cost: ${total_output_cost:.6f}\n")
        f.write(f"Cost per instance: ${total_cost/total_count:.6f}\n")
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
        f.write(f"Ensemble size used: {ensemble_size}\n")
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
            "patch": r['final_patch'] if r['final_patch'] else "",
            "expected_patch": r['expected_patch']
        }
        for r in results
    ]
    write_jsonl(DIR_NAME+'/'+EXP_NAME+'_cot_sc.jsonl', results_jsonl)

if __name__ == "__main__":
    asyncio.run(main())

