import json
import os
import re
import time
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

def extract_patch(text: str) -> str:
    """Extract patch from model response."""
    # Try to extract from XML tag
    pattern = r'<patch>(.*?)</patch>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    # Try to extract from code blocks
    pattern = r'```(?:diff|patch)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    # Try to find diff --git pattern
    if 'diff --git' in text:
        idx = text.find('diff --git')
        return text[idx:].strip()
    
    # Try to find --- a/ pattern
    if '--- a/' in text:
        idx = text.find('--- a/')
        return text[idx:].strip()
    
    return text.strip()

def get_swe_qa_pairs(jsonl_path):
    """Load SWE questions from JSONL file."""
    qa_pairs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                instance_id = data.get("instance_id", "")
                text = data.get("text", "")
                patch = data.get("patch", "")
                qa_pairs.append((instance_id, text, patch))
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

def write_jsonl(filename, data):
    """Write data to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text.
    
    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.
    
    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1) if match else ""

