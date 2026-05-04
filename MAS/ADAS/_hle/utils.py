import os
import json
import random
import string

import numpy as np


# def score_mgsm(target: str, prediction: str) -> bool:
#     if "." in prediction:
#         prediction = prediction.rstrip("0").rstrip(".")
#     target = target.replace(",", "")
#     prediction = prediction.replace(",", "")
#     return target == prediction

def score_hle(target, prediction) -> bool: # TODO
    # Simple version, exact match    
    def are_equal_as_int(a, b):
        try:
            return int(a) == int(b)
        except (ValueError, TypeError):
            return False

    return are_equal_as_int(target, prediction)
    # TODO: advanced compare, use LLM judge


def load_questions(subdata_name: str, seed: str) -> list[dict[str, str]]:
    data_filename = f'dataset/{subdata_name}_seed{str(seed)}.json'
    assert os.path.exists(data_filename), f"{data_filename} does not exist."

    examples = []
    with open(data_filename, mode='r', encoding='utf-8') as f:
        content = json.load(f)
        for ex in content:
            examples.append({"inputs": ex['question'], "targets": ex['answer']})
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
