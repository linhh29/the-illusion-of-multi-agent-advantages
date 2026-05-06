import json
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from benchmarks.safe_code_executor import SafeCodeExecutor
from scripts.logs import logger

Instruction = """
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

class SmfrsBenchmark(BaseBenchmark):
    """
    Benchmark for the synthetic smfr dataset.

    Expected reference format (per line in the jsonl file, consistent with
    `smfr_synthetic_dataset/evaluate/README.md`):

        {
            "problem": "...",
            "answer": {
                "answer": ["Alice", "Bob"]
            }
        }

    We extract:
    - input_text  := problem["problem"]
    - ref_answer  := list of correct entities (names) from problem["answer"]

    Expected model output format (per sample), following
    `evaluate_smfr_answer_code.py`:

        {
            "answer": "<model's final text answer>",
            "code": "<python code string>"
        }

    or wrapped as:

        {
            "output": {
                "answer": "...",
                "code": "..."
            }
        }

    The graph is allowed to return:
    - a Python dict with the above shape, or
    - a JSON string that parses into such a dict.

    Scoring:
    - Direct answer full match: all reference names appear in model_answer.
    - Code full match: executing the code returns a dict with an "answer"
      field whose content is in the reference list (string) or whose list
      matches the reference list (set equality).
    - The final score is:
        1.0 if either direct or code evaluation yields a full match;
        0.5 if there is any partial match (direct or code);
        0.0 otherwise.
    """

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        # Use the same sandboxed executor as in
        # tmp/mas_eval/smfr_synthetic_dataset/evaluate/safe_code_executor.py
        self.executor = SafeCodeExecutor(timeout=30)

    # ---------------------- helpers for reference ---------------------- #

    def _extract_reference_answer(self, problem: Dict[str, Any]) -> List[str]:
        """
        Extract the reference answer list from one problem item.
        """
        answer = problem.get("answer")
        if isinstance(answer, dict):
            ref = answer.get("answer", [])
        else:
            ref = answer
        if ref is None:
            return []
        if isinstance(ref, list):
            return [str(x) for x in ref]
        # Single value
        return [str(ref)]

    # ---------------------- helpers for model output ---------------------- #

    def _parse_model_output(self, raw_output: Any) -> Tuple[str, str, str]:
        """
        Parse the graph output into (raw_str, answer, code).

        raw_output:
            - dict with keys "answer"/"code" or nested under "output", or
            - JSON string, or
            - plain text string (treated as direct answer only).
        """
        raw_str = None
        answer = ""
        code = ""

        obj: Any = raw_output

        if isinstance(raw_output, str):
            raw_str = raw_output
            # try to parse JSON
            try:
                obj = json.loads(raw_output)
            except Exception:
                # treat as plain text answer only
                return raw_str, raw_output, ""

        if isinstance(obj, dict):
            # optional wrapper: "output" or "response" (SMFR workflow returns {"response": json_string})
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
                if raw_str is None:
                    raw_str = json.dumps(obj, ensure_ascii=False)
        else:
            # fallback: treat as plain text
            return raw_str if raw_str is not None else str(obj), str(obj), ""

        return raw_str if raw_str is not None else "", str(answer), str(code)

    # ---------------------- direct answer evaluation ---------------------- #

    def _evaluate_direct_answer(
        self, model_answer: Any, reference_answer: List[str]
    ) -> Tuple[bool, int]:
        """
        Replicates the logic from evaluate_smfr_answer_code.evaluate_direct_answer.
        """
        if not reference_answer:
            return False, 0

        # Treat model_answer as-is (string or list); membership works for both.
        partial_count = 0
        for name in reference_answer:
            if name in model_answer:
                partial_count += 1

        is_full = partial_count == len(reference_answer)
        return is_full, partial_count

    # ---------------------- code evaluation ---------------------- #

    def _evaluate_code_output(
        self, code: str, reference_answer: List[str]
    ) -> Tuple[bool, bool, bool, str]:
        """
        Execute code and compare output against reference answer.

        This implementation is intentionally kept identical in logic to
        tmp/mas_eval/smfr_synthetic_dataset/evaluate/evaluate_smfr_answer_code.py,
        and uses the sandboxed SafeCodeExecutor from safe_code_executor.py.

        Returns:
            (is_full_match, has_any_match, execution_failed)
        """
        if not reference_answer:
            return False, False, False, ""

        try:
            # Suppress stdout during code execution to avoid cluttering output
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

            try:
                # Execute the code with a timeout in a sandbox
                exec_result = self.executor.execute(code, inputs={})
            finally:
                # Restore stdout
                sys.stdout.close()
                sys.stdout = old_stdout

            # Serialize exec_result for logging / CSV
            exec_info = json.dumps(exec_result, ensure_ascii=False)

            # Check if execution was successful
            if not exec_result.get("success", False):
                return False, False, True, exec_info  # Execution failed

            # Extract answer from execution result
            code_answer = exec_result["result"].get("answer")
            if code_answer is None:
                return False, False, True, exec_info  # No answer returned

            # Compare based on answer type
            if isinstance(code_answer, str):
                # String answer: check if it's in reference list
                if code_answer in reference_answer:
                    # Full match only if reference has exactly one answer
                    is_full = len(reference_answer) == 1
                    # String matches always count as "partial" in addition
                    return is_full, True, False, exec_info
            elif isinstance(code_answer, list):
                # List answer: compare sets
                code_set = set(code_answer)
                ref_set = set(reference_answer)

                if code_set == ref_set:
                    # Full match only, no partial tracking for lists
                    return True, False, False, exec_info

            return False, False, False, exec_info  # No match, but execution succeeded
        except Exception as e:
            # Code execution failed - record the exception message
            exec_info = json.dumps({"exception": str(e)}, ensure_ascii=False)
            return False, False, True, exec_info

    # ---------------------- BaseBenchmark interface ---------------------- #

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        """
        This benchmark computes scores inside `evaluate_problem`, so this
        method is only required to satisfy the BaseBenchmark interface.
        We simply return 0.0 and the prediction; it is never used by the
        current evaluation pipeline.
        """
        return 0.0, prediction

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_output(self, graph: Callable, input_text: str):
        return await graph(input_text)

    async def evaluate_problem(
        self, problem: dict, graph: Callable
    ) -> Tuple[str, Any, Any, str, float, float]:
        input_text = problem.get("problem", "") + Instruction
        print(f"input_text: {input_text}")
        reference_answer = self._extract_reference_answer(problem)

        try:
            output, cost = await self._generate_output(graph, input_text)

            raw_output_str, model_answer, code = self._parse_model_output(output)

            # Direct answer evaluation
            direct_full, direct_partial_count = self._evaluate_direct_answer(
                model_answer, reference_answer
            )

            # Code evaluation
            code_full, code_partial, code_failed, code_exec_info = self._evaluate_code_output(
                code, reference_answer
            )

            # Aggregate score
            # if direct_full and code_full:
            if direct_full:
                score = 1.0
            else:
                score = 0.0

            # Single dict for CSV and logs: direct_full, direct_partial_count, code_full, code_partial, code_exec_info
            eval_details = {
                "direct_full": direct_full,
                "direct_partial_count": direct_partial_count,
                "code_full": code_full,
                "code_partial": code_partial,
                "code_exec_info": code_exec_info,
            }
            eval_details_str = json.dumps(eval_details, ensure_ascii=False)

            if score == 0.0:
                # Log detailed mismatch information for later inspection
                self.log_mismatch(
                    problem=input_text,
                    expected_output=reference_answer,
                    prediction=raw_output_str if raw_output_str is not None else str(output),
                    extracted_output={
                        "direct_answer": model_answer,
                        "code": code,
                        "eval_details": eval_details_str,
                    },
                    extract_answer_code="smfr_benchmark",
                )

            # For CSV: store eval_details (five fields) as one JSON string column
            return (
                input_text,
                raw_output_str if raw_output_str is not None else str(output),
                json.dumps(reference_answer, ensure_ascii=False),
                eval_details_str,
                score,
                cost,
            )

        except Exception as e:
            logger.info(
                f"Maximum retries reached for a smfr sample. "
                f"Skipping this sample. Error: {e}"
            )
            # When we fail before code execution, fill eval_details with exception only
            code_exec_info = json.dumps({"exception": str(e)}, ensure_ascii=False)
            eval_details = {
                "direct_full": False,
                "direct_partial_count": 0,
                "code_full": False,
                "code_partial": False,
                "code_exec_info": code_exec_info,
            }
            eval_details_str = json.dumps(eval_details, ensure_ascii=False)
            return (
                input_text,
                str(e),
                json.dumps(reference_answer, ensure_ascii=False),
                eval_details_str,
                0.0,
                0.0,
            )

    def get_result_columns(self) -> List[str]:
        # Add an explicit column for the sandboxed code execution result
        # so it can be inspected directly from the CSV.
        return ["inputs", "prediction", "expected_output", "eval_details", "score", "cost"]

