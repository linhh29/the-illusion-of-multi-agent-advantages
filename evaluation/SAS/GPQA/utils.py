import json
import os
import re
import time
import pandas as pd
from prompt_lib import TEMPERATURE, MAX_TOKENS
from openai import OpenAI
import backoff
from openai import RateLimitError, APIError, APIConnectionError, Timeout


class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

# Initialize OpenAI client
_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

@backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError, Timeout), max_tries=20)
def generate_answer(answer_context, model):
    client = get_client()
    try:
        # For newer models like gpt-5, use max_completion_tokens instead of max_tokens
        if model.startswith('gpt-5') or 'gpt-5' in model.lower():
            completion = client.chat.completions.create(
                model=model,
                messages=answer_context,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
                n=1)
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=answer_context,
                temperature=TEMPERATURE,
                # max_tokens=MAX_TOKENS,
                n=1)
    except RateLimitError as e:
        error_message = str(e)
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if "You exceeded your current quota" in error_message or "quota" in error_message.lower():
            raise OutOfQuotaException(api_key)
        elif "access was terminated" in error_message or "terminated" in error_message.lower():
            raise AccessTerminatedException(api_key)
        else:
            raise e

    return completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens

def parse_single_choice(reply):
    """Parse single choice answer from reply. GPQA uses boxed format."""
    # First try to extract from boxed format: \boxed{A}
    pattern = r'\\boxed\{([ABCDabcd])\}'
    matches = re.findall(pattern, reply)
    if matches:
        return matches[-1].upper()
    
    # Then try standard format: (A)
    pattern = r'\(([ABCDabcd])\)'
    matches = re.findall(pattern, reply)
    solution = None
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is None:
        alter_pattern = r'([ABCDabcd])\)'
        alter_matches = re.findall(alter_pattern, reply)
        for match_str in alter_matches[::-1]:
            solution = match_str.upper()
            if solution:
                break

    return solution

def get_gpqa_qa_pairs(jsonl_path):
    """Load GPQA questions from JSONL file."""
    qa_pairs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question = data.get("question", "")
                answer = data.get("answer", "").upper()
                qa_pairs.append((question, answer))
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line[:100]}... Error: {e}")
                continue
    return qa_pairs

def most_frequent(clist, cmp_func):
    counter = 0
    num = clist[0]

    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter

