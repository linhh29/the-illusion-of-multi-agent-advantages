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

def get_bcp_qa_pairs(jsonl_path):
    """Load BCP questions from JSONL file."""
    qa_pairs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question = data.get("question", "")
                answer = data.get("answer", "")
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

def extract_math_answer(pred_str):
    """Extract answer from prediction string. Supports various formats including \boxed{} for math and general answers."""
    if not pred_str:
        return None
    
    # Method 1: Extract from \boxed{...} format using a more robust approach
    # Find all occurrences of \boxed{ and match the corresponding closing }
    # This handles nested braces correctly
    boxed_matches = []
    start_idx = 0
    while True:
        # Find the next \boxed{ occurrence
        boxed_start = pred_str.find('\\boxed{', start_idx)
        if boxed_start == -1:
            break
        
        # Find the content start (after \boxed{)
        content_start = boxed_start + 7  # len('\\boxed{')
        
        # Find the matching closing brace
        brace_count = 1
        content_end = content_start
        while content_end < len(pred_str) and brace_count > 0:
            if pred_str[content_end] == '{':
                brace_count += 1
            elif pred_str[content_end] == '}':
                brace_count -= 1
            content_end += 1
        
        if brace_count == 0:
            # Found a complete \boxed{...}
            answer_content = pred_str[content_start:content_end-1]
            boxed_matches.append(answer_content)
        
        start_idx = content_end
    
    if boxed_matches:
        # Take the last match (most likely the final answer)
        answer = boxed_matches[-1].strip()
        # Remove any LaTeX commands like \text{}
        answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
        # Remove other common LaTeX commands
        answer = re.sub(r'\\(?:textbf|textit|emph)\{([^}]+)\}', r'\1', answer)
        answer = answer.strip()
        if answer:
            return answer
    
    # Method 2: Try regex pattern as fallback (for simpler cases)
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, pred_str)
    if matches:
        answer = matches[-1].strip()
        answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
        answer = answer.strip()
        if answer:
            return answer
    
    # Method 3: Look for explicit answer patterns
    answer_patterns = [
        r'(?:final\s+)?answer[:\s]+(.+?)(?:\.|$|\n)',
        r'answer[:\s]+(.+?)(?:\.|$|\n)',
        r'the\s+answer\s+is[:\s]+(.+?)(?:\.|$|\n)',
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, pred_str, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1].strip()
            # Clean up the answer
            answer = re.sub(r'^["\']|["\']$', '', answer)  # Remove quotes
            if answer:
                return answer
    
    # Method 4: If the response is very short and looks like an answer, return it
    lines = pred_str.strip().split('\n')
    last_line = lines[-1].strip() if lines else ""
    # If last line is short and doesn't contain common question words, it might be the answer
    if last_line and len(last_line) < 100 and not any(word in last_line.lower() for word in ['therefore', 'thus', 'hence', 'so', 'because', 'since']):
        # Remove common prefixes
        last_line = re.sub(r'^(?:answer|final answer|result)[:\s]+', '', last_line, flags=re.IGNORECASE)
        last_line = last_line.strip().rstrip('.')
        if last_line:
            return last_line
    
    # If nothing found, return None to indicate extraction failure
    return None


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

import re

def grade_bcp_answer(question, response, correct_answer):
    """Grade BCP answer."""
    prompt = GRADER_TEMPLATE.format(question=question, response=response, correct_answer=correct_answer)
    messages = [
        {"role": "user", "content": prompt}
    ]
    final_response, _, _ = generate_answer(messages, "gpt-4o")
    # Extract correct (yes/no)
    correct_match = re.search(
        r"\*\*correct:\*\*\s*(yes|no)", final_response, re.IGNORECASE
    )
    if not correct_match:
        correct_match = re.search(
            r"\*\*correct\*\*:\s*(yes|no)", final_response, re.IGNORECASE
        )
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", final_response, re.IGNORECASE)
    if correct_match:
        correctness = correct_match.group(1).lower() == "yes"
    else:
        correctness = False
    return correctness

EQUIVALENCE_GRADER_TEMPLATE = """
Answer1: {answer1}
Answer2: {answer2}

Judge whether the two answers are equivalent or not.

Your judgement must be in the format and criteria specified below:

equivalent: Answer 'yes' if the two answers are equivalent, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the answers are different.
Please just output 'yes' or 'no' in a single line.
""".strip()

def is_equivalent(answer1, answer2):
    """Check if two answers are equivalent."""
    prompt = EQUIVALENCE_GRADER_TEMPLATE.format(answer1=answer1, answer2=answer2)
    messages = [
        {"role": "user", "content": prompt}
    ]
    response, _, _ = generate_answer(messages, "gpt-4o")
    if "yes" in response.lower():
        return True
    else:
        return False