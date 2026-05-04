import os
import ast
import json
import random
import string

from pathlib import Path
import evaluate_stock_answer_code as eval_stock


def to_dict(content):
    # try to parse str content to dict
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # back to safe Python literal eval (handles single quotes)
            return ast.literal_eval(content)
        except Exception as e:
            raise ValueError(f"Failed to parse model chat output content: {e}")
        

def score_stocks(pred_file, ref_file, mode="test") -> bool: # TODO  
    timeout = 30 # seconds
    reference = str(ref_file)
    model_output = str(pred_file)

    # Load data
    print(f"Loading reference data from: {reference}")
    references = eval_stock.load_reference_data(reference)

    print(f"Loading model outputs from: {model_output}")
    model_outputs = eval_stock.load_model_outputs(model_output)

    # Initialize metrics
    full = 0  # Direct answer full matches
    cfull = 0  # Code output full matches
    partial_match = 0  # Direct answer partial matches
    cpartial_match = 0  # Code output partial matches
    code_failures = 0  # Code execution failures
    total = 0

    # if mode is 'test', evaluate all depths individually and report results per depth
    res_per_depth = {}
    if mode == 'test':
        for depth in list(range(2, 7)):
            res_per_depth[depth] = {
                "full": 0,
                "partial_match": 0,
                "cfull": 0,
                "cpartial_match": 0,
                "code_failures": 0,
                "total": 0,
            }

    # Initialize code executor
    executor = eval_stock.SafeCodeExecutor(timeout=timeout)

    # Process each sample
    print("\nEvaluating samples...")
    for idx, model_output in enumerate(model_outputs):
        depth = int(model_output['depth'])
        # Match model output with reference
        sample_id = model_output.get('id', idx)
        if sample_id not in references:
            print(f"Warning: No reference found for sample {sample_id}")
            continue

        reference = references[sample_id]

        # Extract the actual answer list from the reference
        # Handle both nested and flat answer structures
        if isinstance(reference.get('answer'), dict):
            reference_answer = reference['answer'].get('answer', [])
        else:
            reference_answer = reference.get('answer', [])

        # Skip if no reference answer
        if not reference_answer:
            continue

        # Evaluate direct answer
        try:
            answer = to_dict(model_output['model_answer'])['answer']
        except Exception as e:
            print(f"Error parsing model answer for sample {sample_id}: {e}")
            answer = []
        is_full, partial_count = eval_stock.evaluate_direct_answer(
            #model_output['answer'],
            # model_output['output']['answer'],
            answer,
            reference_answer
        )

        if is_full:
            full += 1
            if mode == 'test':
                res_per_depth[depth]["full"] += 1
        elif partial_count > 0:
            partial_match += 1
            if mode == 'test':
                res_per_depth[depth]["partial_match"] += 1

        # Evaluate code output
        try:
            code = to_dict(model_output['model_answer'])['code']
            code = code.encode().decode("unicode_escape") #li: handle escaped \\n and \\" in code string
        except Exception as e:
            print(f"Error parsing model code for sample {sample_id}: {e}")
            code = ""
        code_full, code_partial, code_failed = eval_stock.evaluate_code_output(
            #model_output['code'],
            code,
            reference_answer,
            executor
        )

        if code_full:
            cfull += 1
            if mode == 'test':
                res_per_depth[depth]["cfull"] += 1
        if code_partial:
            cpartial_match += 1
            if mode == 'test':
                res_per_depth[depth]["cpartial_match"] += 1
        if code_failed:
            code_failures += 1
            if mode == 'test':
                res_per_depth[depth]["code_failures"] += 1

        total += 1
        if mode == 'test':
            res_per_depth[depth]["total"] += 1

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples evaluated: {total}")
    print(f"\nDirect Answer Metrics:")
    print(f"  Full Match:    {full}/{total} ({full/total:.2%})")
    print(f"  Partial Match: {partial_match}/{total} ({partial_match/total:.2%})")
    print(f"\nCode Output Metrics:")
    print(f"  Full Match:    {cfull}/{total} ({cfull/total:.2%})")
    print(f"  Partial Match: {cpartial_match}/{total} ({cpartial_match/total:.2%})")
    print(f"  Execution Failures:      {code_failures}/{total} ({code_failures/total:.2%})")
    print("="*60)
    # Print results per depth if in test mode, in the order of depth 2-6
    if mode == 'test':
        print("\nResults per depth:")
        for depth in range(2, 7):
            metrics = res_per_depth.get(depth, {})
            d_total = metrics.get("total", 0)
            if d_total == 0:
                continue
            print(f"\nDepth: {depth}")
            print(f"  Total: {d_total}")
            print(f"  Direct Answer Full Match:    {metrics.get('full', 0)}/{d_total} ({metrics.get('full', 0)/d_total:.2%})")
            print(f"  Direct Answer Partial Match: {metrics.get('partial_match', 0)}/{d_total} ({metrics.get('partial_match', 0)/d_total:.2%})")
            print(f"  Code Output Full Match:      {metrics.get('cfull', 0)}/{d_total} ({metrics.get('cfull', 0)/d_total:.2%})")
            print(f"  Code Output Partial Match:   {metrics.get('cpartial_match', 0)}/{d_total} ({metrics.get('cpartial_match', 0)/d_total:.2%})")
            print(f"  Code Execution Failures:    {metrics.get('code_failures', 0)}/{d_total} ({metrics.get('code_failures', 0)/d_total:.2%})")
            print("-"*40)

    return full, partial_match, cfull, cpartial_match, code_failures, total, res_per_depth


def load_stock_examples(split: str):
    # Load STOCKS examples from the corresponding jsonl file in AFlow/data/datasets/stocks_{split}.jsonl
    dataset_path = Path(__file__).resolve().parent.parent.parent/"AFlow"/"data"/"datasets"/"stocks_{}.jsonl".format(split)
    assert os.path.exists(dataset_path), f"{dataset_path} does not exist."
    assert split in ['validate', 'test'], f"Invalid split: {split}. Must be 'validate' or 'test'."
    
    examples = []
    with dataset_path.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            
            if split == 'test':
                depth = data["source_split"]
            else:
                depth = data["generation_params"].get("depth")
            # seed = data["generation_params"].get("seed")
            # comb_ind = data["generation_params"].get("combination_index")

            examples.append({
                "problem": data["problem"],
                "answer": data["answer"],
                "id": idx,  # Use index as ID
                "depth": depth,
            })
    return examples  # 588 test, 16 validate



def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id



if __name__ == "__main__":
    # Example usage
    # load_stock_examples(split='validate')
    # load_stock_examples(split='test')
    
    pass