import json
import os
import re
import time
from prompt_lib import TEMPERATURE, MAX_TOKENS
from openai import OpenAI, AsyncOpenAI
import backoff
from openai import RateLimitError, APIError, APIConnectionError, Timeout
from math import isclose
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
import regex


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
_async_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def get_async_client():
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI()
    return _async_client


# @backoff.on_exception(
#     backoff.expo,
#     (RateLimitError, APIError, APIConnectionError, Timeout),
#     max_tries=5,          # 避免单次调用因为频繁重试卡太久
#     max_time=300          # 单次调用最长重试时间约 5 分钟
# )
def generate_answer(answer_context, model):
    client = get_client()
    try:
        # For newer models like gpt-5, use max_completion_tokens instead of max_tokens
        print(answer_context)
        if model.startswith('gpt-5') or 'gpt-5' in model.lower():
            completion = client.chat.completions.create(
                model=model,
                messages=answer_context,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
                n=1,
            )
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=answer_context,
                temperature=TEMPERATURE,
                # max_tokens=MAX_TOKENS,
                n=1,
            )
        print(completion.choices[0].message.content)
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

async def generate_answer_async(answer_context, model):
    """Async version of generate_answer for concurrent execution."""
    client = get_async_client()
    try:
        # For newer models like gpt-5, use max_completion_tokens instead of max_tokens
        print(answer_context)
        if model.startswith('gpt-5') or 'gpt-5' in model.lower():
            completion = await client.chat.completions.create(
                model=model,
                messages=answer_context,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
                n=1,
            )
        else:
            completion = await client.chat.completions.create(
                model=model,
                messages=answer_context,
                temperature=TEMPERATURE,
                # max_tokens=MAX_TOKENS,
                n=1,
            )
        print(completion.choices[0].message.content)
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


def extract_math_answer(pred_str):
    """Extract math answer from prediction string."""
    # Try to extract from boxed format
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    boxed_matches = re.findall(pattern, pred_str, re.DOTALL)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # Try "The answer is" format
    if 'The answer is ' in pred_str:
        pred = pred_str.split('The answer is ')[-1].strip()
    elif 'the answer is ' in pred_str:
        pred = pred_str.split('the answer is ')[-1].strip()
    else:
        # Try to extract last sentence
        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, pred_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        pred = sentences[-1] if sentences else ""
    
    return pred

def is_equiv(str1, str2, verbose=False):
    """Check if two math expressions are equivalent."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        # Try numeric comparison
        if is_digit(str1) and is_digit(str2):
            val1 = parse_digits(str1)
            val2 = parse_digits(str2)
            if val1 is not None and val2 is not None:
                return isclose(val1, val2, abs_tol=1e-3)
    except:
        pass

    try:
        # Try symbolic comparison
        return symbolic_equal(str1, str2)
    except:
        pass

    # Fallback to string comparison
    return str(str1).strip() == str(str2).strip()

def is_digit(num):
    return parse_digits(num) is not None

def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None

def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), abs_tol=1e-3):
            return True
    except:
        pass
    return False

def get_hlemath_qa_pairs(jsonl_path):
    """Load HLEMATH questions from JSONL file."""
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

