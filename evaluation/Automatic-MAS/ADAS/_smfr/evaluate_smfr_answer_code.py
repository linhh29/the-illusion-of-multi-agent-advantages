"""
Evaluation script for smfr trading analysis tasks.

This script evaluates model outputs against reference answers by:
1. Comparing direct text answers (Direct Match)
2. Executing generated code and comparing outputs (Code Match)
"""

import argparse
import json
import jsonlines
import os
import sys
import traceback
from safe_code_executor import SafeCodeExecutor


def load_reference_data(reference_file):
    """
    Load reference dataset from a JSONL file.

    Args:
        reference_file: Path to JSONL file containing reference data

    Returns:
        Dictionary mapping sample IDs to reference data
    """
    references = {}
    with jsonlines.open(reference_file, mode='r') as reader:
        for idx, sample in enumerate(reader):
            # Use index as ID if no explicit ID field exists
            sample_id = sample.get('id', idx)
            references[sample_id] = sample
    return references


def load_model_outputs(model_output_file):
    """
    Load model outputs from a JSON or JSONL file.
    Expected format: each entry should have 'answer' and 'code' keys.

    Args:
        model_output_file: Path to JSON/JSONL file with model outputs

    Returns:
        Dictionary or list of model outputs
    """
    if model_output_file.endswith('.jsonl'):
        outputs = []
        with jsonlines.open(model_output_file, mode='r') as reader:
            for sample in reader:
                outputs.append(sample)
        return outputs
    else:
        with open(model_output_file, 'r') as f:
            return json.load(f)


def evaluate_direct_answer(model_answer, reference_answer):
    """
    Evaluate direct text answer comparison.

    Args:
        model_answer: Model's text answer (string or list)
        reference_answer: Reference answer (list of correct names/entities)

    Returns:
        Tuple of (is_full_match, partial_match_count)
    """
    if not reference_answer:
        return False, 0

    # Count how many reference answers appear in model answer
    partial_count = 0
    for name in reference_answer:
        if name in model_answer:
            partial_count += 1

    # Full match if all reference answers found
    is_full = (partial_count == len(reference_answer))
    return is_full, partial_count


def evaluate_code_output(code, reference_answer, executor):
    """
    Execute code and compare output against reference answer.

    Args:
        code: Python code string to execute
        reference_answer: Reference answer (list of correct names/entities)
        executor: SafeCodeExecutor instance

    Returns:
        Tuple of (is_full_match, has_any_match, execution_failed)
        - is_full_match: True if exact full match
        - has_any_match: True if any match found (includes full matches for strings)
        - execution_failed: True if code execution failed
    """
    try:
        # Suppress stdout during code execution to avoid cluttering output
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            # Execute the code with a timeout
            exec_result = executor.execute(code, inputs={})
        finally:
            # Restore stdout
            sys.stdout.close()
            sys.stdout = old_stdout

        # Check if execution was successful
        if not exec_result.get('success', False):
            return False, False, True  # Execution failed

        # Extract answer from execution result
        code_answer = exec_result['result'].get('answer')
        if code_answer is None:
            return False, False, True  # No answer returned

        # Compare based on answer type
        if isinstance(code_answer, str):
            # String answer: check if it's in reference list
            if code_answer in reference_answer:
                # Full match only if reference has exactly one answer
                is_full = (len(reference_answer) == 1)
                # Original behavior: string matches always count as "partial"
                return is_full, True, False
        elif isinstance(code_answer, list):
            # List answer: compare sets
            code_set = set(code_answer)
            ref_set = set(reference_answer)

            if code_set == ref_set:
                return True, False, False  # Full match only, no partial tracking for lists

        return False, False, False  # No match, but execution succeeded
    except Exception:
        # Code execution failed - silently continue
        return False, False, True


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model outputs for smfr trading analysis tasks.'
    )
    parser.add_argument(
        '--reference',
        required=True,
        help='Path to reference dataset (JSONL format)'
    )
    parser.add_argument(
        '--model-output',
        required=True,
        help='Path to model output file (JSON or JSONL format with "answer" and "code" keys)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout in seconds for code execution (default: 30)'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading reference data from: {args.reference}")
    references = load_reference_data(args.reference)

    print(f"Loading model outputs from: {args.model_output}")
    model_outputs = load_model_outputs(args.model_output)

    # Initialize metrics
    full = 0  # Direct answer full matches
    cfull = 0  # Code output full matches
    partial_match = 0  # Direct answer partial matches
    cpartial_match = 0  # Code output partial matches
    code_failures = 0  # Code execution failures
    total = 0

    # Initialize code executor
    executor = SafeCodeExecutor(timeout=args.timeout)

    # Process each sample
    print("\nEvaluating samples...")
    for idx, model_output in enumerate(model_outputs):
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
        is_full, partial_count = evaluate_direct_answer(
            #model_output['answer'],
            model_output['output']['answer'],
            reference_answer
        )

        if is_full:
            full += 1
        elif partial_count > 0:
            partial_match += 1

        # Evaluate code output
        code_full, code_partial, code_failed = evaluate_code_output(
            #model_output['code'],
            model_output['output']['code'],
            reference_answer,
            executor
        )

        if code_full:
            cfull += 1
        if code_partial:
            cpartial_match += 1
        if code_failed:
            code_failures += 1

        total += 1

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


if __name__ == "__main__":
    main()


