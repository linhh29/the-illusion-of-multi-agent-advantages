"""
Evaluation script for stock trading analysis tasks.

Evaluates model outputs against the fixed reference dataset by:
  1. Comparing direct text answers  (Direct Match)
  2. Executing generated code and comparing outputs  (Code Match)

Ground truth is always sourced from the fixed reference file (balanced_dataset_single_N_fixed.jsonl),
matched to model outputs by problem text.  Validation samples (n=2, n=3 only) are excluded
automatically when a validation file is provided.

Answer normalisation
--------------------
Model answers are compared using investor-name-aware extraction:
  - Plain list/string answers are handled directly.
  - Free-form strings such as "Alice and Bob" or '["Alice", "Bob"]' are parsed by
    scanning for known investor names (taken from the GT's investor_dates keys).
  - "None" / "null" strings are treated as no answer.
  - False ties are penalised: "Alice and Bob" when only "Alice" is expected
    extracts {Alice, Bob} which does NOT match {Alice}.

Usage
-----
  python evaluate_stock_answer_code.py \\
      --reference  balanced_dataset_single_2_fixed.jsonl \\
      --model-output  my_model_outputs.jsonl

  python evaluate_stock_answer_code.py \\
      --reference  balanced_dataset_single_2_fixed.jsonl \\
      --model-output  my_model_outputs.jsonl \\
      --validation  balanced_dataset_single_validate.jsonl \\
      --timeout 30

Input formats
-------------
Reference JSONL (one sample per line):
  {"problem": "...", "answer": {"investor_dates": {...}, "comparison": {...}, "answer": [...]}}

Model output JSONL (one sample per line):
  {"input": {"problem": "..."}, "output": {"answer": "...", "code": "..."}}

The "answer" field inside "output" may be:
  - A string:  "Alice"  or  "Alice and Bob"  or  '["Alice", "Bob"]'
  - A list:    ["Alice", "Bob"]
  - A dict:    {"answer": ["Alice"], ...}
  - null / "None"

The "code" field inside "output" is optional Python code whose solve() function
should return a dict with an "answer" key.
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from safe_code_executor import SafeCodeExecutor
    _HAS_EXECUTOR = True
except ImportError:
    _HAS_EXECUTOR = False


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

def _extract_final_answer(answer: Any) -> Optional[List[str]]:
    """Unwrap nested dicts and normalise to a flat list of name strings."""
    if answer is None:
        return None
    if isinstance(answer, dict):
        answer = answer.get("answer")
    if answer is None:
        return None
    if isinstance(answer, list):
        names = [str(a) for a in answer if a is not None]
        return names if names else None
    if isinstance(answer, str):
        return [answer]
    return [str(answer)]


def _extract_answer_with_context(answer: Any, known_names: List[str]) -> Optional[List[str]]:
    """Extract answer names using known investor names as a vocabulary.

    `answer` may be a full output dict (MAS pipeline), a bare answer value (CoT/SC), or None.

    Returns None (valid "no winner") when:
      - The model explicitly outputs the string "None"/"null" or an empty list
      - MAS pipeline ran to completion, has investor_dates, and answer is null

    Returns "__no_answer__" (API failure, counts as wrong) when:
      - Output is completely missing (parsed as {}) — no answer key at all
      - Bare Python None at top level (no output dict present)

    Penalises false ties: "Alice and Bob" when only "Alice" is correct → {Alice, Bob} ≠ {Alice}.
    """
    # Full output dict — check for reasoned None (handles single and double nesting)
    if isinstance(answer, dict):
        has_investor_dates = "investor_dates" in answer
        inner = answer.get("answer")
        if inner is None:
            return None if has_investor_dates else "__no_answer__"
        # Double-nesting: inner is itself a structured pipeline output dict
        if isinstance(inner, dict) and "investor_dates" in inner:
            nested_answer = inner.get("answer")
            if nested_answer is None:
                return None  # reasoned None from nested structure
            answer = nested_answer
        else:
            answer = inner

    if answer is None:
        return "__no_answer__"

    if isinstance(answer, list):
        names = [str(a) for a in answer if a is not None]
        return names if names else None  # empty list = explicit no-winner

    if not isinstance(answer, str):
        return [str(answer)]

    if answer.strip().lower() in ("none", "null", ""):
        return None  # explicit "no winner" string from model

    if known_names:
        found = [
            name for name in known_names
            if re.search(rf'\b{re.escape(name)}\b', answer, re.IGNORECASE)
        ]
        if found:
            return found

    # Fallback: try JSON parse (handles '["Alice"]')
    stripped = answer.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                names = [str(x) for x in parsed if x is not None]
                return names if names else "__no_answer__"
            if isinstance(parsed, str):
                return "__no_answer__" if parsed.strip().lower() in ("none", "null") else [parsed]
        except (json.JSONDecodeError, TypeError):
            pass

    return [answer]


def compare_answers(expected: Any, actual: Any) -> bool:
    """Return True iff the final answer names in `actual` exactly match `expected`.

    A None answer is correct only when GT is also None AND the model output is a
    structured dict containing "investor_dates" (reasoned None, not API failure).
    """
    known_names: List[str] = []
    if isinstance(expected, dict):
        known_names = list(expected.get("investor_dates", {}).keys())

    exp = _extract_final_answer(expected)
    act = _extract_answer_with_context(actual, known_names)

    if exp is None and act is None:
        return True
    if exp is None or act is None or act == "__no_answer__":
        return False
    return set(exp) == set(act)


# ---------------------------------------------------------------------------
# Code evaluation
# ---------------------------------------------------------------------------

def _evaluate_code(code: str, expected: Any, executor) -> Tuple[bool, bool]:
    """
    Execute `code` and compare its answer against `expected`.

    Returns (is_correct, execution_failed).
    """
    if not code or not _HAS_EXECUTOR:
        return False, True
    try:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            result = executor.execute(code, inputs={})
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        if not result.get("success"):
            return False, True

        code_answer = result.get("result", {})
        return compare_answers(expected, code_answer), False

    except Exception:
        return False, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs for stock trading analysis tasks."
    )
    parser.add_argument(
        "--reference", required=True,
        help="Path to fixed reference dataset (JSONL). "
             "Each line: {\"problem\": \"...\", \"answer\": {...}}",
    )
    parser.add_argument(
        "--model-output", required=True,
        help="Path to model output JSONL. "
             "Each line: {\"input\": {\"problem\": \"...\"}, \"output\": {\"answer\": ..., \"code\": ...}}",
    )
    parser.add_argument(
        "--validation", default=None,
        help="Path to validation JSONL whose problems should be excluded (n=2/n=3 only).",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Timeout in seconds for code execution (default: 30).",
    )
    args = parser.parse_args()

    # Load ground truth keyed by problem text
    print(f"Loading reference data from: {args.reference}")
    gt_by_problem: Dict[str, Any] = {}
    for s in _load_jsonl(args.reference):
        gt_by_problem[s["problem"]] = s["answer"]
    print(f"  {len(gt_by_problem)} reference samples loaded")

    # Load optional validation exclusion set
    val_problems: frozenset = frozenset()
    if args.validation:
        val_problems = frozenset(
            s["problem"] for s in _load_jsonl(args.validation) if "problem" in s
        )
        print(f"  {len(val_problems)} validation problems will be excluded")

    # Load model outputs
    print(f"\nLoading model outputs from: {args.model_output}")
    model_outputs = _load_jsonl(args.model_output)
    print(f"  {len(model_outputs)} records loaded")

    executor = SafeCodeExecutor(timeout=args.timeout) if _HAS_EXECUTOR else None

    # Metrics
    direct_correct = 0
    code_correct   = 0
    code_failures  = 0
    skipped        = 0
    total          = 0

    for record in model_outputs:
        inp  = record.get("input", {})
        prob = inp.get("problem", "")

        # Skip validation samples
        if prob in val_problems:
            skipped += 1
            continue

        # Look up ground truth
        gt = gt_by_problem.get(prob)
        if gt is None:
            skipped += 1
            continue

        # Unwrap output
        out = record.get("output", {})
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except (json.JSONDecodeError, TypeError):
                out = {}
        if not isinstance(out, dict):
            out = {}

        direct_ans = out.get("answer")
        code_str   = out.get("code")

        # Direct answer evaluation
        if compare_answers(gt, direct_ans):
            direct_correct += 1

        # Code evaluation
        if code_str:
            ok, failed = _evaluate_code(code_str, gt, executor)
            if ok:
                code_correct += 1
            if failed:
                code_failures += 1

        total += 1

    # Results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total evaluated : {total}")
    if skipped:
        print(f"Skipped         : {skipped}  (not in reference / validation set)")

    def pct(n): return f"{n/total:.1%}" if total else "N/A"

    print(f"\nDirect Answer:")
    print(f"  Correct : {direct_correct}/{total} ({pct(direct_correct)})")
    print(f"  Wrong   : {total - direct_correct}/{total} ({pct(total - direct_correct)})")

    if _HAS_EXECUTOR:
        print(f"\nCode Execution:")
        print(f"  Correct          : {code_correct}/{total} ({pct(code_correct)})")
        print(f"  Execution errors : {code_failures}/{total} ({pct(code_failures)})")
    else:
        print("\nCode Execution: skipped (safe_code_executor not available)")

    print("=" * 60)


if __name__ == "__main__":
    main()
