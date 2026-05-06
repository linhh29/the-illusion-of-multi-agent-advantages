import asyncio
import json
import os
import sys
import torch
from typing import Any, Callable, Dict, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.ext.maas.benchmark.safe_code_executor import SafeCodeExecutor
from maas.logs import logger

# Verbatim copy of AFlow benchmarks/smfr.py `Instruction` (lines 12–37). Keep in sync with
# /data/qin/lhh/MAS-Eval-Foundation/AFlow/benchmarks/smfr.py — do not edit unless upstream changes.
SMFR_AFLOW_INSTRUCTION = """
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

# AFlow ``backup_workspace/.../SMFR/workflows/template/op_prompt.py`` — ``SC_ENSEMBLE_PROMPT``
# (used with ``call_reverse_answer_code`` / ``acall_reverse_answer_code``).
SMFR_AFLOW_SC_ENSEMBLE_PROMPT = """
Several answers have been generated to the same question. They are as follows:
{solutions}

Synthesize the best solution from the above: pick the most consistent and correct one, or merge the best parts. You must output a single structured response with exactly these three fields (same format as each solution):

1. "analysis": Step-by-step reasoning explaining why you chose or merged this answer.
2. "answer": The final answer - ONLY the name(s), e.g. "Alice" or ["Alice", "Bob"] if tied. No extra text.
3. "code": Python code that defines a solve() function returning a dict with an "answer" key (and any other keys the problem needs). All input data must be included in the code.

Output only valid JSON with keys: analysis, answer, code.
"""

# AFlow ``op_prompt.py`` — ``ANSWER_GENERATION_PROMPT`` (``AnswerGenerate`` operator).
SMFR_AFLOW_ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem. You must output a single structured response with exactly these three fields:

1. "analysis": Step-by-step reasoning.
2. "answer": The final answer - ONLY the name(s), e.g. "Alice" or ["Alice", "Bob"] if tied. No extra text.
3. "code": Python code with a solve() function that returns a dict containing "answer" (and any other keys the problem needs). Include all required input data in the code.

Output only valid JSON with keys: analysis, answer, code.

Your task: {input}
"""


def extract_smfr_reference_answer(problem: Dict[str, Any]) -> List[str]:
    """
    Same as AFlow ``benchmarks/smfr.SmfrsBenchmark._extract_reference_answer``.
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
    return [str(ref)]


def parse_smfr_model_output(raw_output: Any) -> Tuple[str, str, str]:
    """
    Same as AFlow ``benchmarks/smfr.SmfrsBenchmark._parse_model_output`` (verbatim logic).
    """
    raw_str = None
    answer = ""
    code = ""

    obj: Any = raw_output

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
            if raw_str is None:
                raw_str = json.dumps(obj, ensure_ascii=False)
    else:
        return raw_str if raw_str is not None else str(obj), str(obj), ""

    return raw_str if raw_str is not None else "", str(answer), str(code)


def smfr_evaluate_direct_answer(
    model_answer: Any, reference_answer: List[str]
) -> Tuple[bool, int]:
    """
    Same as AFlow ``benchmarks/smfr.SmfrsBenchmark._evaluate_direct_answer``:
    ``name in model_answer`` for each reference name (model_answer string or list, not coerced).
    """
    if not reference_answer:
        return False, 0

    partial_count = 0
    for name in reference_answer:
        if name in model_answer:
            partial_count += 1

    is_full = partial_count == len(reference_answer)
    return is_full, partial_count


class SmfrBenchmark(BaseBenchmark):
    """
    Benchmark for smfr trading analysis tasks.

    Evaluation follows the same logic as AFlow's SmfrsBenchmark:
      - Direct answer evaluation: check whether each reference name appears
        in the model's answer text.
      - Code evaluation: execute model-generated solve() and compare the
        structured output against the reference answer.
      - Final score = 1.0 if direct_full match, else 0.0.
      - Both evaluation paths are logged in *eval_details* for analysis.
    """

    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        batch_size: int,
        controller: torch.nn.Module,
        operator_embeddings,
        optimizer: torch.optim.Optimizer,
        eval_mode: str = "answer",
    ):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer)
        self.eval_mode = eval_mode
        self.executor = SafeCodeExecutor(timeout=30)
        logger.info(f"SmfrBenchmark initialised with eval_mode='{self.eval_mode}'")

    # ------------------------------------------------------------------
    # Direct answer evaluation  (aligned with AFlow)
    # ------------------------------------------------------------------

    def _evaluate_direct_answer(
        self, model_answer: Any, reference_answer: List[str]
    ) -> Tuple[bool, int]:
        return smfr_evaluate_direct_answer(model_answer, reference_answer)

    # ------------------------------------------------------------------
    # Code evaluation  (aligned with AFlow — SafeCodeExecutor sandbox)
    # ------------------------------------------------------------------

    def _evaluate_code_output(
        self, code: str, reference_answer: List[str]
    ) -> Tuple[bool, bool, bool, str]:
        """Execute *code* via SafeCodeExecutor and compare output against
        *reference_answer*.

        Returns (is_full_match, has_any_match, execution_failed, info_str).
        """
        if not reference_answer:
            return False, False, False, ""

        try:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            try:
                exec_result = self.executor.execute(code, inputs={})
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout

            exec_info = json.dumps(exec_result, ensure_ascii=False, default=str)

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
                code_set = set(code_answer)
                ref_set = set(reference_answer)
                if code_set == ref_set:
                    return True, False, False, exec_info

            return False, False, False, exec_info

        except Exception as e:
            exec_info = json.dumps({"exception": str(e)}, ensure_ascii=False)
            return False, False, True, exec_info

    # ------------------------------------------------------------------
    # Scoring  (satisfies BaseBenchmark interface; not used directly)
    # ------------------------------------------------------------------

    def calculate_score(self, expected_output: List[str], prediction: Any) -> Tuple[float, Any]:
        return 0.0, prediction

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_output(self, graph, input_text):
        return await asyncio.wait_for(graph(input_text), timeout=1800)

    async def evaluate_problem(self, problem: dict, graph: Callable):
        # AFlow: problem.get("problem", "") + Instruction
        input_text = problem.get("problem", "") + SMFR_AFLOW_INSTRUCTION
        expected_names: List[str] = extract_smfr_reference_answer(problem)
        logger.info(f"input_text: {input_text}")

        try:
            output, cost, logprob = await self._generate_output(graph, input_text)
            if not output:
                raise ValueError("empty output from graph")

            raw_output_str, model_answer, code_str = parse_smfr_model_output(output)

            direct_full, direct_partial = self._evaluate_direct_answer(
                model_answer, expected_names
            )

            # AFlow always runs _evaluate_code_output(code, ...) even when code is empty.
            code_full, code_partial, code_failed, code_info = self._evaluate_code_output(
                code_str, expected_names
            )

            # --- Aggregate score (same rule as AFlow) ---
            score = 1.0 if direct_full else 0.0

            eval_details = json.dumps({
                "direct_full": direct_full,
                "direct_partial_count": direct_partial,
                "code_full": code_full,
                "code_partial": code_partial,
                "code_exec_info": code_info,
            }, ensure_ascii=False)

            if score == 0.0:
                self.log_mismatch(
                    problem=input_text,
                    expected_output=expected_names,
                    prediction=raw_output_str if raw_output_str is not None else str(output),
                    extracted_output={
                        "direct_answer": model_answer,
                        "code": code_str,
                        "eval_details": eval_details,
                    },
                    extract_answer_code="smfr_benchmark",
                )

            pred_for_return = raw_output_str if raw_output_str is not None else str(output)
            return (
                input_text,
                pred_for_return,
                json.dumps(expected_names, ensure_ascii=False),
                score,
                cost,
                logprob,
                eval_details,
            )

        except Exception as e:
            logger.info(
                f"Maximum retries reached for a smfr sample. "
                f"Skipping this sample. Error: {e}"
            )
            eval_details = json.dumps({
                "direct_full": False,
                "direct_partial_count": 0,
                "code_full": False,
                "code_partial": False,
                "code_exec_info": json.dumps({"exception": str(e)}, ensure_ascii=False),
            }, ensure_ascii=False)
            return (
                input_text,
                str(e),
                json.dumps(expected_names, ensure_ascii=False),
                0.0,
                0.0,
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
                eval_details,
            )

    def get_result_columns(self) -> List[str]:
        return [
            "question", "prediction", "expected_output",
            "score", "cost", "logprob", "eval_details",
        ]