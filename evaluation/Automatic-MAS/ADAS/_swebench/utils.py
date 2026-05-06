import os
import json
import random
import string
import subprocess

import numpy as np



# python -m swebench.harness.run_evaluation \
#     --dataset_name SWE-bench/SWE-bench_Lite \
#     --predictions_path gold \
#     --max_workers 4 \
#     --instance_ids sympy__sympy-20590 \
#     --run_id search-valid \
#     --split valid/test

def score_swe(pred_file, question_ids, model, mode, solution_name) -> bool: # TODO    
    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "SWE-bench/SWE-bench_Lite",
        "--predictions_path", pred_file,
        "--max_workers", "12", # TODO: fewer than min(0.75 * os.cpu_count(), 24)
        "--run_id", f"{mode}-{solution_name.replace(' ', '_')}",
        "--split", 'test',
        "--cache_level", "instance", #new
        "--clean", "False", # new, do not clean up docker images/containers, else hit docker pull quota
        "--instance_ids",
    ] + question_ids
    # subprocess.run(cmd)

    # # [OLD] read evaluation result and return acc and err lists
    # acc_list = []
    # err_list = []
    # swe_output_path = f"{model}.{mode}-{solution_name.replace(' ', '_')}.json"
    # with open(swe_output_path, 'r') as f:
    #     results = json.load(f)
    
    # total_instances = results['total_instances']
    # submitted = results['submitted_ids']
    # completed = results['completed_ids']
    # resolved = results['resolved_ids']
    # unresolved = results['unresolved_ids']
    # errors = results['error_ids']
    # for _sub in submitted:
    #     if _sub in completed: # either resolved or unresolved
    #         if _sub in resolved:
    #             acc_list.append(1)
    #         else:
    #             acc_list.append(0)
    #     else: # not completed
    #         err_list.append(f"q {_sub}: error during evaluation")
    #         acc_list.append(0)
    
    # return acc_list, err_list

    # [NEW] run evaluation with retry mechanism
    acc_list = []
    err_list = []
    cnt = 0
    score = 0
    while cnt < 10: # if error, let's rerun
        try:
            # Run the command and capture output    
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
                )
            print('SWE score: Evaluation completed successfully.')
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print('SWE score: Error during evaluation:')
            print(e.stderr)
            print('Rerun')
            cnt += 1
        except Exception as e:
            print(f'SWE score: Unexpected error during evaluation: {e}')
            cnt += 1

        swe_output_path = f"{model}.{mode}-{solution_name.replace(' ', '_')}.json"
        if os.path.exists(swe_output_path):
            with open(swe_output_path, 'r') as f:
                results = json.load(f)
            score = float(results['resolved_instances'])
            if results['error_instances'] > 0:
                cnt += 1
                continue
            else:
                pass# TODO
        else:
            print(f"Report file not found at {swe_output_path}")
            cnt += 1
            continue
    # fill in acc_list and err_list
    for _sub in question_ids:
        if _sub in results['completed_ids']: # either resolved or unresolved
            if _sub in results['resolved_ids']:
                acc_list.append(1)
            else:
                acc_list.append(0)
        else: # not completed
            err_list.append(f"q {_sub}: error during evaluation")
            acc_list.append(0)
    
    return acc_list, err_list



def load_questions(data_name: str, seed: str) -> list[dict[str, str]]:
    # data_filename = f'dataset/{data_name}.jsonl'
    data_filename = data_name
    assert os.path.exists(data_filename), f"{data_filename} does not exist."

    examples = []
    with open(data_filename, mode='r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            examples.append({"inputs": ex['text'], "targets": ex['patch'], "instance_id": ex['instance_id']})
    return examples


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"
