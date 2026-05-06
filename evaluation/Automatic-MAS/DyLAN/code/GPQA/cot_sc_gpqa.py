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
from prompt_lib import SYSTEM_PROMPT_GPQA, construct_message, ROLE_MAP

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


def parse_ensemble_override(arg_index):
    """
    Optional argv[arg_index]: strict ensemble size for CoT-SC.
    If missing or empty, return None (use DyLAN-derived size or default 5).
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

# Maximum number of concurrent async tasks (adjust based on API rate limits)
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "5"))

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

def evaluate_answer(predicted, ground_truth):
    """Evaluate if the predicted answer matches the ground truth."""
    if predicted is None:
        return False
    # Normalize both to uppercase strings for comparison
    pred_str = str(predicted).strip().upper()
    truth_str = str(ground_truth).strip().upper()
    return pred_str == truth_str

def get_cot_sc_ensemble_size(base_dir_name):
    """
    Read avg values from all run directories and calculate the average.
    The avg value is from lines like: "Total API calls: 842 (avg: 5.07 per question)"
    """
    # Get the directory name (without path)
    dir_name = os.path.basename(base_dir_name) if os.path.dirname(base_dir_name) else base_dir_name
    
    # Extract the base pattern (without run number)
    # e.g., "gpqa_downsampled_gpt-4o_Theoretical Physicist_Molecular Chemist_Cellular Biologist_Assistant"
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
    pattern = os.path.join(parent_dir, base_pattern + '_run*', 'gpqa_test_43.txt')
    txt_files = glob.glob(pattern)
    
    # If no files found, try a broader search
    if len(txt_files) == 0:
        # Try searching in the parent directory directly for any matching pattern
        pattern = os.path.join(parent_dir, '*_run*', 'gpqa_test_43.txt')
        all_txt_files = glob.glob(pattern)
        # Filter to only include files that match the base pattern
        txt_files = [f for f in all_txt_files if base_pattern in os.path.basename(os.path.dirname(f))]
    
    avg_values = []
    for txt_file in sorted(txt_files):  # Sort to ensure consistent ordering
        try:
            with open(txt_file, 'r') as f:
                content = f.read()
                # Look for pattern: "avg: X.XX per question"
                match = re.search(r'avg:\s*([\d.]+)\s*per question', content)
                if match:
                    avg_val = float(match.group(1))
                    avg_values.append(avg_val)
                    print(f"Found avg value {avg_val:.2f} from {txt_file}")
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
            continue
    
    if len(avg_values) == 0:
        print(f"Warning: No avg values found in run directories. Using default ensemble size of 5.")
        print(f"Searched in: {parent_dir}")
        print(f"Pattern used: {base_pattern}_run*/gpqa_test_43.txt")
        return 5
    
    final_avg = sum(avg_values) / len(avg_values)
    ensemble_size = int(round(final_avg))
    print(f"Average ensemble size from {len(avg_values)} run(s): {final_avg:.2f} -> {ensemble_size}")
    return ensemble_size

async def generate_cot_response(question, model):
    """Generate a single CoT response for the question."""
    # For CoT-SC, we use a simple system prompt without specific roles
    # The model will solve the problem step by step
    # sys_prompt = "Please solve the problem step by step."
    
    # Construct the user message with step-by-step instruction and boxed format requirement
    user_message = {
        "role": "user",
        "content": f"Here is the question:\n{question}\n\nPlease solve this problem step by step. At the end of your response, put your final answer in the form \\boxed{{X}}, where X represents choice (A), (B), (C), or (D)."
    }
    
    contexts = []
    contexts.append(user_message)
    
    # Generate answer
    reply, prompt_tokens, completion_tokens = await asyncio.to_thread(generate_answer, contexts, model)
    answer = parse_single_choice(reply)
    
    return reply, answer, prompt_tokens, completion_tokens

async def process_single_question(idx, que, ans, model, ensemble_size, dir_name, exp_name):
    """Process a single question with CoT-SC (multiple CoT samples + voting)."""
    try:
        # Generate multiple CoT responses
        completions = []
        answers = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i in range(ensemble_size):
            reply, answer, prompt_tokens, completion_tokens = await generate_cot_response(que, model)
            completions.append(reply)
            answers.append(answer)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
        
        # Use voting to get the final answer
        final_answer, vote_count = most_frequent(answers, lambda x, y: x == y)
        
        # Extract answer from final_answer (in case it's not already extracted)
        # The answers list already contains extracted answers, but ensure final_answer is properly formatted
        if final_answer is not None:
            final_answer = str(final_answer).strip().upper()
        
        # Evaluate the answer
        acc = evaluate_answer(final_answer, ans)
        
        # Prepare data to save
        data_to_save = {
            'completions': completions,
            'answers': answers,
            'final_result': final_answer,
            'vote_count': vote_count,
            'ground_truth': ans,
            'correct': acc
        }
        
        # Thread-safe file writing
        async with file_lock:
            await asyncio.to_thread(
                lambda: open(dir_name+'/'+exp_name+'_cot_sc.json', 'a').write(json.dumps(data_to_save) + '\n')
            )
        
        # Calculate cost for this question
        cost_info = calculate_cost(model, total_prompt_tokens, total_completion_tokens)
        
        # Print evaluation result immediately
        status = "✓ CORRECT" if acc else "✗ WRONG"
        pred_str = str(final_answer) if final_answer else "None"
        print(f"Question {idx+1}: {status} | Predicted: {pred_str} | Ground Truth: {ans} | Votes: {vote_count}/{ensemble_size} | Cost: ${cost_info['total_cost']:.6f}")
        
        return {
            'idx': idx,
            'acc': acc,
            'resp_cnt': ensemble_size,
            'completions': completions,
            'answers': answers,
            'final_answer': final_answer,
            'vote_count': vote_count,
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'input_cost': cost_info['input_cost'],
            'output_cost': cost_info['output_cost'],
            'total_cost': cost_info['total_cost']
        }
    except Exception as e:
        print(f"Error processing question {idx+1}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'idx': idx,
            'acc': False,
            'resp_cnt': ensemble_size,
            'completions': None,
            'answers': None,
            'final_answer': None,
            'vote_count': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0
        }

async def main():
    set_rd_seed(0)
    os.makedirs(DIR_NAME, exist_ok=True)

    # Ensemble size: optional argv[7] overrides; else infer from DyLAN run logs (default 5)
    ensemble_override = parse_ensemble_override(7)
    if ensemble_override is not None:
        ensemble_size = ensemble_override
        print(f"Using CoT-SC ensemble size (from argument): {ensemble_size}")
    else:
        ensemble_size = get_cot_sc_ensemble_size(DIR_NAME)
        print(f"Using CoT-SC ensemble size: {ensemble_size}")

    qa_pairs = get_gpqa_qa_pairs(QUERY_JSONL)
    # Remove the limit for full dataset processing
    # qa_pairs = qa_pairs[:5]

    # Initialize JSON file
    with open(DIR_NAME+'/'+EXP_NAME+'_cot_sc.json', 'w') as f:
        f.write("")

    print(f"Processing {len(qa_pairs)} questions with CoT-SC (ensemble size: {ensemble_size}, max {MAX_CONCURRENT} concurrent)")

    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Track running statistics for real-time accuracy
    running_correct = 0
    running_total = 0
    stats_lock = asyncio.Lock()
    
    async def process_with_semaphore(idx, que, ans):
        async with semaphore:
            result = await process_single_question(idx, que, ans, MODEL, ensemble_size, DIR_NAME, EXP_NAME)
            # Update running statistics
            async with stats_lock:
                nonlocal running_correct, running_total
                running_total += 1
                if result['acc']:
                    running_correct += 1
                current_acc = running_correct / running_total if running_total > 0 else 0.0
                print(f"  → Running Accuracy: {running_correct}/{running_total} = {current_acc:.4f}")
            return result
    
    # Create all tasks
    tasks = [
        process_with_semaphore(idx, que, ans)
        for idx, (que, ans) in enumerate(qa_pairs)
    ]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Sort results by index to maintain order
    results.sort(key=lambda x: x['idx'])

    # Aggregate results
    accs = [r['acc'] for r in results]
    resp_cnts = sum(r['resp_cnt'] for r in results)
    completion_list = [r['completions'] for r in results]
    total_prompt_tokens = sum(r['prompt_tokens'] for r in results)
    total_completion_tokens = sum(r['completion_tokens'] for r in results)
    total_input_cost = sum(r['input_cost'] for r in results)
    total_output_cost = sum(r['output_cost'] for r in results)
    total_cost = sum(r['total_cost'] for r in results)
    
    # Calculate final accuracy
    correct_count = sum(accs)
    total_count = len(qa_pairs)
    final_accuracy = correct_count / total_count if total_count > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Questions: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Wrong Answers: {total_count - correct_count}")
    print(f"Accuracy: {correct_count}/{total_count} = {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"\nAPI Usage:")
    print(f"  Total API calls: {resp_cnts} (avg: {resp_cnts/total_count:.2f} per question)")
    print(f"  Total tokens: {total_prompt_tokens + total_completion_tokens:,} (prompt: {total_prompt_tokens:,}, completion: {total_completion_tokens:,})")
    print(f"\nCost Summary:")
    print(f"  Total cost: ${total_cost:.4f} (input: ${total_input_cost:.4f}, output: ${total_output_cost:.4f})")
    print(f"  Cost per question: ${total_cost/total_count:.6f}")
    print(f"{'='*60}")

    # Write final results
    with open(DIR_NAME+'/'+EXP_NAME+'_cot_sc.txt', 'w') as f:
        # Original format (for compatibility)
        f.write(str(accs) + ' ' + str(sum(accs)/len(qa_pairs)) + '\n')
        f.write(str(resp_cnts) + " " + str(resp_cnts/len(qa_pairs)) + '\n')
        f.write(str(total_prompt_tokens) + '\n')
        f.write(str(total_completion_tokens) + '\n')
        # Add cost information
        f.write(f"Total cost: ${total_cost:.6f}\n")
        f.write(f"Input cost: ${total_input_cost:.6f}\n")
        f.write(f"Output cost: ${total_output_cost:.6f}\n")
        f.write(f"Cost per question: ${total_cost/len(qa_pairs):.6f}\n")
        # Add detailed evaluation results
        f.write(f"\n{'='*60}\n")
        f.write(f"FINAL EVALUATION RESULTS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total Questions: {total_count}\n")
        f.write(f"Correct Answers: {correct_count}\n")
        f.write(f"Wrong Answers: {total_count - correct_count}\n")
        f.write(f"Accuracy: {correct_count}/{total_count} = {final_accuracy:.4f} ({final_accuracy*100:.2f}%)\n")
        f.write(f"\nAPI Usage:\n")
        f.write(f"  Total API calls: {resp_cnts} (avg: {resp_cnts/total_count:.2f} per question)\n")
        f.write(f"  Total tokens: {total_prompt_tokens + total_completion_tokens:,} (prompt: {total_prompt_tokens:,}, completion: {total_completion_tokens:,})\n")
        f.write(f"\nCost Summary:\n")
        f.write(f"  Total cost: ${total_cost:.4f} (input: ${total_input_cost:.4f}, output: ${total_output_cost:.4f})\n")
        f.write(f"  Cost per question: ${total_cost/total_count:.6f}\n")
        f.write(f"Ensemble size used: {ensemble_size}\n")
        f.write(f"{'='*60}\n")
    

if __name__ == "__main__":
    asyncio.run(main())

