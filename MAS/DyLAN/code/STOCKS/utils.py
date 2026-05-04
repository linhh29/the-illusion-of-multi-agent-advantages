# -*- coding: utf-8 -*-
"""
STOCKS dataset utils: data loading and evaluation logic.
Evaluation logic is aligned with AFlow benchmarks/stocks.py.
"""
import ast
import json
import os
import re
import sys
from collections import Counter
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from prompt_lib import TEMPERATURE, MAX_TOKENS
from openai import OpenAI
import backoff
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError

from safe_code_executor import SafeCodeExecutor


def json_dumps_safe(obj: Any, **kwargs: Any) -> str:
    """Serialize dicts from SafeCodeExecutor (datetime, set, etc. in nested result)."""

    def _default(o: Any) -> Any:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, (set, frozenset)):
            return list(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        return str(o)

    kw = {"ensure_ascii": False, "default": _default}
    kw.update(kwargs)
    return json.dumps(obj, **kw)


def multiset_equal_answer_lists(a: List[Any], b: List[str]) -> bool:
    """Multiset equality (order ignored); safe when elements are unhashable (e.g. nested lists)."""
    if len(a) != len(b):
        return False

    def _canon(x: Any) -> str:
        return json.dumps(x, sort_keys=True, default=str)

    return sorted(_canon(x) for x in a) == sorted(_canon(x) for x in b)


# ---------- Instruction templates ----------
InstructionWithCode = """
First, solve the problem and explain your reasoning step-by-step. Provide your reasoning and final answer as `analysis` and `answer`. Then write code to solve this problem based on the following instructions.  
To solve this problem, write Python code with a solve() function that returns a dictionary. When your code is executed, solve() should return, for example:{
    "investor_dates": {
        "Alice": ["November 19, 2025", "November 21, 2025"],
        "Bob": ["November 25, 2025"],
        "Charlie": []
    },      
    "comparison": {
        "Alice": "November 19, 2025",
        "Bob": "November 25, 2025",
        "Charlie": None
    },      
    "answer": "Alice"
}

Format:
- "investor_dates": dict mapping each investor name to list of valid dates (strings in "Month Day, Year" format)
- "comparison": dict mapping each investor name to their first valid date or None
- "answer": winning investor's name (string), or list of names if tied, or None if no one achieves target

For ties, return a list:
{"answer": ["Alice", "Bob"]}

Ensure that all required input data is included within the code as needed.
"""

InstructionDirectOnly = """
First, solve the problem and explain your reasoning step-by-step.  
Provide your reasoning and final answer using JSON fields:
- analysis: concise step-by-step reasoning
- answer: investor name, or list of names for ties

Do not include code.
"""

# Keep backward-compatible alias for existing imports.
Instruction = InstructionWithCode


def get_instruction(require_code: bool = True) -> str:
    return InstructionWithCode if require_code else InstructionDirectOnly


# ---------- Data loading (AFlow format: problem, answer) ----------
def get_stocks_qa_pairs(jsonl_path: str, require_code: bool = True) -> List[Tuple[str, Dict]]:
    """
    Load STOCKS problems from JSONL. Each line: {"problem": "...", "answer": ...}.
    Returns list of (input_text, problem_dict) where input_text = problem["problem"] + selected instruction.
    """
    pairs = []
    instruction = get_instruction(require_code=require_code)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                problem_text = data.get("problem", "")
                input_text = problem_text + instruction
                pairs.append((input_text, data))
            except json.JSONDecodeError:
                continue
    return pairs


def extract_reference_answer(problem: Dict[str, Any]) -> List[str]:
    """Extract reference answer list from one problem (same as AFlow)."""
    answer = problem.get("answer")
    if isinstance(answer, dict):
        ref = answer.get("answer", [])
    else:
        ref = answer
    if ref is None:
        return []
    if isinstance(ref, list):
        return [str(x) for x in ref]
    return [str(ref)]


# ---------- Parse model output (same as AFlow StocksBenchmark._parse_model_output) ----------
def parse_model_output(raw_output: Any) -> Tuple[str, str, str]:
    """Parse graph/LLM output into (raw_str, answer, code)."""
    raw_str = None
    answer = ""
    code = ""
    obj = raw_output
    if isinstance(raw_output, str):
        raw_str = raw_output
        try:
            obj = json.loads(raw_output)
        except Exception:
            return raw_str, raw_output, ""
    if isinstance(obj, dict):
        payload = obj.get("output", obj.get("response", obj))
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        if isinstance(payload, dict):
            answer = payload.get("answer", "") or ""
            code = payload.get("code", "") or ""
        if raw_str is None:
            raw_str = json.dumps(obj, ensure_ascii=False)
    else:
        return raw_str if raw_str else str(obj), str(obj), ""
    return raw_str if raw_str else "", str(answer), str(code)


# ---------- Direct answer evaluation ----------
def parse_investor_answer_names(model_answer: Any) -> List[str]:
    """
    Normalize model output to a list of investor names (order not significant for multiset).
    Handles JSON lists, 'A and B' ties, comma-separated names, and single strings.
    """
    if model_answer is None:
        return []
    if isinstance(model_answer, list):
        return [str(x).strip() for x in model_answer if str(x).strip()]
    s = str(model_answer).strip()
    if not s or s.lower() == "none":
        return []
    if s.startswith("["):
        try:
            v = json.loads(s)
        except Exception:
            try:
                v = ast.literal_eval(s)
            except Exception:
                v = None
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    if re.search(r"\s+and\s+", s):
        parts = re.split(r"\s+and\s+", s, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def evaluate_direct_answer(model_answer: Any, reference_answer: List[str]) -> Tuple[bool, int]:
    """
    Full match iff predicted investor multiset equals reference (exact names, no substring hacks).
    Partial count: how many reference names appear in the parsed prediction set.
    """
    if not reference_answer:
        return False, 0
    names = parse_investor_answer_names(model_answer)
    if not names:
        return False, 0
    ref_c = Counter(reference_answer)
    pred_c = Counter(names)
    is_full = ref_c == pred_c
    partial_count = sum(1 for name in reference_answer if name in set(names))
    return is_full, partial_count


# ---------- Code evaluation (same as AFlow StocksBenchmark._evaluate_code_output) ----------
def evaluate_code_output(
    code: str, reference_answer: List[str], executor: SafeCodeExecutor
) -> Tuple[bool, bool, bool, str]:
    if not reference_answer:
        return False, False, False, ""
    try:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            exec_result = executor.execute(code, inputs={})
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        exec_info = json_dumps_safe(exec_result)
        if not exec_result.get("success", False):
            return False, False, True, exec_info
        code_answer = exec_result["result"].get("answer")
        if code_answer is None:
            return False, False, True, exec_info
        if isinstance(code_answer, str):
            if code_answer in reference_answer:
                is_full = len(reference_answer) == 1
                return is_full, True, False, exec_info
        elif isinstance(code_answer, list):
            if multiset_equal_answer_lists(code_answer, reference_answer):
                return True, False, False, exec_info
        return False, False, False, exec_info
    except Exception as e:
        return False, False, True, json.dumps({"exception": str(e)}, ensure_ascii=False)


def most_frequent(clist, cmp_func):
    """Return most frequent element and its count (for LLMLP / CoT-SC consensus)."""
    if not clist:
        raise ValueError("most_frequent: empty list")
    counter = 0
    num = clist[0]
    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = i
    return num, counter


# ---------- CoT-SC: direct vote filter + code ensemble (all completions) ----------
VoteKey = Tuple[Any, ...]


def is_meaningless_direct_vote_value(a: Any) -> bool:
    """Exclude from direct-answer ensemble: None, empty/whitespace, string 'None'."""
    if a is None:
        return True
    s = a.strip() if isinstance(a, str) else str(a).strip()
    if not s:
        return True
    if s.lower() == "none":
        return True
    return False


def is_meaningless_exec_answer(ans: Any) -> bool:
    """Exclude from code-result ensemble: None, '', whitespace, string 'None', empty list."""
    if ans is None:
        return True
    if isinstance(ans, str):
        s = ans.strip()
        if not s:
            return True
        if s.lower() == "none":
            return True
        return False
    if isinstance(ans, list) and len(ans) == 0:
        return True
    return False


def vote_key_code_answer(ans: Any) -> Optional[VoteKey]:
    """Canonical key for voting on executed code answers (list order ignored)."""
    if is_meaningless_exec_answer(ans):
        return None
    if isinstance(ans, str):
        return ("str", ans.strip())
    if isinstance(ans, list):
        return ("list", tuple(sorted(str(x) for x in ans)))
    return ("json", json.dumps(ans, sort_keys=True, default=str))


def score_execution_result_vs_ref(
    exec_result: Dict[str, Any], reference_answer: List[str]
) -> Tuple[bool, bool, bool]:
    """Same as evaluate_code_output scoring, given raw executor return dict (no execution)."""
    if not reference_answer:
        return False, False, False
    if not exec_result.get("success", False):
        return False, False, True
    result = exec_result.get("result") or {}
    code_answer = result.get("answer") if isinstance(result, dict) else None
    if code_answer is None:
        return False, False, True
    if isinstance(code_answer, str):
        if code_answer in reference_answer:
            is_full = len(reference_answer) == 1
            return is_full, True, False
    elif isinstance(code_answer, list):
        if multiset_equal_answer_lists(code_answer, reference_answer):
            return True, False, False
    return False, False, False


def ensemble_code_answers_from_exec_results(exec_answers: List[Any]) -> Tuple[Any, int, Optional[VoteKey]]:
    """Majority vote over meaningful executed `answer` values (after filtering meaningless)."""
    meaningful: List[Tuple[VoteKey, Any]] = []
    for a in exec_answers:
        k = vote_key_code_answer(a)
        if k is not None:
            meaningful.append((k, a))
    if not meaningful:
        return None, 0, None
    keys = [k for k, _ in meaningful]
    winner_key, vote_count = most_frequent(keys, lambda x, y: x == y)
    ensemble_answer = next(v for k, v in meaningful if k == winner_key)
    return ensemble_answer, vote_count, winner_key


def score_ensemble_code_answer_vs_ref(
    ensemble_answer: Any, reference_answer: List[str]
) -> Tuple[bool, bool]:
    """Full / partial match for voted code answer vs reference."""
    if not reference_answer or ensemble_answer is None:
        return False, False
    if isinstance(ensemble_answer, str):
        if ensemble_answer in reference_answer:
            is_full = len(reference_answer) == 1
            return is_full, True
    elif isinstance(ensemble_answer, list):
        if multiset_equal_answer_lists(ensemble_answer, reference_answer):
            return True, False
    return False, False


def infer_ensemble_code_failed(
    ensemble_answer: Any, code_full: bool, code_partial: bool
) -> bool:
    """True only when no usable voted answer (exec failures / all meaningless); wrong answer => False."""
    if code_full or code_partial:
        return False
    if ensemble_answer is not None:
        return False
    return True


def run_code_ensemble_eval(
    completions: List[Any],
    reference_answer: List[str],
    executor: SafeCodeExecutor,
) -> Dict[str, Any]:
    """
    Run code from every completion, record per-index metrics, then majority-vote on meaningful
    executed `answer` values (exclude None, '', string 'None', empty list).
    Returns code_full, code_partial, code_failed for the ensemble, plus structured details.
    """
    per: List[Dict[str, Any]] = []
    exec_answers: List[Any] = []

    for idx, comp in enumerate(completions):
        _, _, code = parse_model_output(comp)
        code = code or ""
        entry: Dict[str, Any] = {
            "index": idx,
            "code_char_len": len(code),
            "code_empty": not code.strip(),
        }
        if not code.strip():
            entry["exec_success"] = False
            entry["code_full"] = False
            entry["code_partial"] = False
            entry["code_failed"] = True
            entry["result_answer"] = None
            entry["code_exec_info"] = json.dumps({"skipped": "empty_code"})
            exec_answers.append(None)
        else:
            try:
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
                try:
                    exec_result = executor.execute(code, inputs={})
                finally:
                    sys.stdout.close()
                    sys.stdout = old_stdout
            except Exception as e:
                exec_result = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            entry["code_exec_info"] = json_dumps_safe(exec_result)
            entry["exec_success"] = bool(exec_result.get("success"))
            ra = None
            if entry["exec_success"] and isinstance(exec_result.get("result"), dict):
                ra = exec_result["result"].get("answer")
            entry["result_answer"] = ra
            exec_answers.append(ra)
            cf, cp, failed = score_execution_result_vs_ref(exec_result, reference_answer)
            entry["code_full"] = cf
            entry["code_partial"] = cp
            entry["code_failed"] = failed
        per.append(entry)

    ens_ans, ens_votes, ens_key = ensemble_code_answers_from_exec_results(exec_answers)
    ens_full, ens_partial = score_ensemble_code_answer_vs_ref(ens_ans, reference_answer)
    code_failed = infer_ensemble_code_failed(ens_ans, ens_full, ens_partial)
    meaningful_cnt = sum(1 for a in exec_answers if not is_meaningless_exec_answer(a))

    code_ensemble = {
        "ensemble_result_answer": ens_ans,
        "ensemble_vote_count": ens_votes,
        "ensemble_vote_key": list(ens_key) if isinstance(ens_key, tuple) else ens_key,
        "ensemble_code_full": ens_full,
        "ensemble_code_partial": ens_partial,
        "meaningful_exec_answer_count": meaningful_cnt,
    }

    return {
        "code_full": ens_full,
        "code_partial": ens_partial,
        "code_failed": code_failed,
        "code_eval_per_completion": per,
        "code_ensemble": code_ensemble,
    }


# ---------- OpenAI client and generate_answer (for LLM_Neuron) ----------
_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _debug_print_chat_completion_kwargs(common_kwargs: Dict[str, Any], max_content: int = 4000) -> None:
    """Log a JSON preview of kwargs passed to chat.completions.create (truncate long message text)."""
    trimmed: Dict[str, Any] = dict(common_kwargs)
    msgs_out: List[Any] = []
    for m in common_kwargs.get("messages") or []:
        if isinstance(m, dict):
            d = dict(m)
            c = d.get("content")
            if isinstance(c, str) and len(c) > max_content:
                d["content"] = (
                    c[:max_content] + f"\n... [truncated {len(c) - max_content} chars, total {len(c)}]"
                )
            msgs_out.append(d)
        else:
            msgs_out.append(m)
    trimmed["messages"] = msgs_out
    try:
        preview = json.dumps(trimmed, ensure_ascii=False, indent=2)
        print("[STOCKS utils] chat.completions.create kwargs (preview):\n", preview, sep="", flush=True)
    except Exception as e:
        print("[STOCKS utils] json.dumps(trimmed kwargs) failed:", e, flush=True)
    try:
        raw = json.dumps(common_kwargs, ensure_ascii=False)
    except Exception as e:
        print("[STOCKS utils] strict json.dumps(full kwargs) failed:", e, flush=True)
    else:
        print(
            "[STOCKS utils] strict json.dumps(full kwargs) OK, byte length:",
            len(raw.encode("utf-8")),
            flush=True,
        )


@backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError, APITimeoutError), max_tries=20)
def generate_answer(answer_context, model):
    """
    Call OpenAI chat completion with JSON schema response_format for STOCKS.

    The schema matches AFlow's ReverseAnswerCodeResponse:
    {analysis: str, answer: str, code: str}
    so that downstream parse_model_output can reliably extract answer/code.
    """
    require_code = True
    for msg in answer_context:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str) and "Do not include code." in content:
            require_code = False
            break
    return _generate_answer_with_schema(answer_context, model, require_code=require_code)


def generate_answer_with_mode(answer_context, model, require_code: bool = True):
    """Call OpenAI completion with schema toggled by require_code."""
    return _generate_answer_with_schema(answer_context, model, require_code=require_code)


def _generate_answer_with_schema(answer_context, model, require_code: bool = True):
    # Define schema locally to avoid global dependency
    from pydantic import BaseModel, Field

    if require_code:
        class ResponseSchema(BaseModel):
            analysis: str = Field(..., description="Step by Step reasoning")
            answer: str = Field(..., description="Final answer, ONLY NAME (or list of names for ties)")
            code: str = Field(..., description="Code with solve() function and all required input data")
    else:
        class ResponseSchema(BaseModel):
            analysis: str = Field(..., description="Step by Step reasoning")
            answer: str = Field(..., description="Final answer, ONLY NAME (or list of names for ties)")

    schema = ResponseSchema.model_json_schema()

    client = get_client()

    common_kwargs = {
        "model": model,
        "messages": answer_context,
        "temperature": TEMPERATURE,
        "n": 1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ReverseAnswerCodeResponse",
                "schema": schema,
            },
        },
    }

    if model.startswith("gpt-5") or "gpt-5" in model.lower():
        common_kwargs["max_completion_tokens"] = MAX_TOKENS

    _debug_print_chat_completion_kwargs(common_kwargs)

    completion = client.chat.completions.create(**common_kwargs)
    return (
        completion.choices[0].message.content,
        completion.usage.prompt_tokens,
        completion.usage.completion_tokens,
    )


# ---------- Parser used by LLMLP: return raw reply for stocks ----------
def parse_stocks_raw(reply):
    """Identity parser: return raw reply so we can parse answer/code later."""
    return reply if reply is not None else ""
