import ast
import asyncio
import json
import os
import openai
import random
import sys
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

async def generate_cot_response(question, model):
    """Generate a single CoT response for the question."""
    # Construct the user message with step-by-step instruction and boxed format requirement
    user_message = {
        "role": "user",
        "content": f"Here is the question:\n{question}\n\nPlease solve this problem step by step. At the end of your response, put your final answer in the form \\boxed{{X}}, where X represents choice (A), (B), (C), or (D)."
    }
    
    contexts = []
    contexts.append(user_message)
    
    # Generate answer using asyncio.to_thread for better async performance
    reply, prompt_tokens, completion_tokens = await asyncio.to_thread(generate_answer, contexts, model)
    answer = parse_single_choice(reply)
    
    return reply, answer, prompt_tokens, completion_tokens

async def process_single_question(idx, que, ans, model, dir_name, exp_name):
    """Process a single question with CoT (single CoT response, no ensemble)."""
    try:
        # Generate single CoT response
        reply, answer, prompt_tokens, completion_tokens = await generate_cot_response(que, model)
        
        # Extract answer from final_answer (in case it's not already extracted)
        if answer is not None:
            final_answer = str(answer).strip().upper()
        else:
            print(f"Error: Could not extract final answer for question {idx+1}")
            final_answer = None
        
        # Evaluate the answer
        acc = evaluate_answer(final_answer, ans)
        
        # Prepare data to save
        data_to_save = {
            'completion': reply,
            'final_result': final_answer,
            'ground_truth': ans,
            'correct': acc
        }
        
        # Thread-safe file writing
        async with file_lock:
            await asyncio.to_thread(
                lambda: open(dir_name+'/'+exp_name+'_cot.json', 'a').write(json.dumps(data_to_save) + '\n')
            )
        
        # Calculate cost for this question
        cost_info = calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Print evaluation result immediately
        status = "✓ CORRECT" if acc else "✗ WRONG"
        pred_str = str(final_answer) if final_answer else "None"
        print(f"Question {idx+1}: {status} | Predicted: {pred_str} | Ground Truth: {ans} | Cost: ${cost_info['total_cost']:.6f}")
        
        return {
            'idx': idx,
            'acc': acc,
            'resp_cnt': 1,  # Single CoT call
            'completion': reply,
            'final_answer': final_answer,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
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
            'resp_cnt': 1,
            'completion': None,
            'final_answer': None,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0
        }

async def main():
    set_rd_seed(0)
    os.makedirs(DIR_NAME, exist_ok=True)

    qa_pairs = get_gpqa_qa_pairs(QUERY_JSONL)
    # Remove the limit for full dataset processing
    # qa_pairs = qa_pairs[:5]

    # Initialize JSON file
    with open(DIR_NAME+'/'+EXP_NAME+'_cot.json', 'w') as f:
        f.write("")

    print(f"Processing {len(qa_pairs)} questions with CoT (max {MAX_CONCURRENT} concurrent)")

    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Track running statistics for real-time accuracy
    running_correct = 0
    running_total = 0
    stats_lock = asyncio.Lock()
    
    async def process_with_semaphore(idx, que, ans):
        async with semaphore:
            result = await process_single_question(idx, que, ans, MODEL, DIR_NAME, EXP_NAME)
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
    completion_list = [r['completion'] for r in results]
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
    with open(DIR_NAME+'/'+EXP_NAME+'_cot.txt', 'w') as f:
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
        f.write(f"{'='*60}\n")
    

if __name__ == "__main__":
    asyncio.run(main())

