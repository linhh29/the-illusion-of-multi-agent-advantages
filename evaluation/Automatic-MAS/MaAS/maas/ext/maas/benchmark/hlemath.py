import asyncio
import re
import torch
from math import isclose
from typing import Any, Callable, List, Optional, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.logs import logger


class HLEMATHBenchmark(BaseBenchmark):
    """
    HLEMath (High-Level Math) Benchmark
    Mathematical questions with numerical or expression answers
    Supports multiple equivalent forms of answers
    """
    
    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        batch_size: int,
        controller: torch.nn.Module,
        operator_embeddings: List[List[float]],
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer)

    def extract_model_answer(self, text: str) -> str:
        """
        Extract answer from model output:
        1. First try to extract from \boxed{} format
        2. If not found, extract the last sentence
        """
        # Try to find \boxed{} pattern
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # If no boxed found, split by sentence and return last one
        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        """
        Calculate score by comparing mathematical equivalence
        """
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1.0, predicted_answer
        else:
            return 0.0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        """
        Check if two mathematical expressions are equal
        Tries three methods:
        1. String equality
        2. Numerical equality (for digits)
        3. Symbolic equality (using sympy)
        """
        # Method 1: Direct string comparison
        if str(prediction) == str(reference):
            return True

        # Method 2: Numerical comparison
        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction_num = self.parse_digits(prediction)
                reference_num = self.parse_digits(reference)
                if prediction_num is not None and reference_num is not None:
                    return isclose(prediction_num, reference_num, abs_tol=1e-3)
        except Exception:
            pass

        # Method 3: Symbolic comparison
        try:
            return self.symbolic_equal(prediction, reference)
        except Exception:
            pass

        return False

    def is_digit(self, num) -> bool:
        """Check if the input can be parsed as a number"""
        return self.parse_digits(num) is not None

    def parse_digits(self, num) -> Optional[float]:
        """
        Parse a string into a float number
        Supports regular numbers and percentages
        """
        num_str = regex.sub(",", "", str(num))
        try:
            return float(num_str)
        except Exception:
            # Try to parse percentage
            if num_str.endswith("%"):
                num_str = num_str[:-1]
                if num_str.endswith("\\"):
                    num_str = num_str[:-1]
                try:
                    return float(num_str) / 100
                except Exception:
                    pass
        return None

    def symbolic_equal(self, a, b) -> bool:
        """
        Check symbolic equality using sympy
        """
        def _parse(s):
            """Try to parse string as mathematical expression"""
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except Exception:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        # Try symbolic simplification
        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass

        # Try numerical evaluation
        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except Exception:
            pass

        return False

    @retry(stop=stop_after_attempt(20), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await asyncio.wait_for(graph(input_text), timeout=1500)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float, torch.Tensor]:
        input_text = problem["question"]
        expected_output = problem["answer"]

        try:
            output, cost, logprob = await self._generate_output(graph, input_text)
            
            if not output:
                raise ValueError("output is empty")

            score, extracted_output = self.calculate_score(expected_output, output)

            if score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                )

            return input_text, output, expected_output, score, cost, logprob

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob"]

