import asyncio
import re
import torch
from typing import Any, Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.logs import logger


class GPQABenchmark(BaseBenchmark):
    """
    GPQA (Graduate-Level Google-Proof Q&A) Benchmark
    Multiple choice questions with options A, B, C, D
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

    def normalize_answer(self, s: str) -> str:
        """
        Normalize answer for evaluation by:
        1. Extracting content from \boxed{} if present
        2. Converting to lowercase
        3. Removing whitespace
        """
        try:
            # Try to extract from \boxed{} format
            if '\\boxed{' in s:
                s = s.replace('\\text{', '').split('boxed{')[-1]
                # Get the first character after boxed{
                if len(s) > 0:
                    s = s[0]
            # If no boxed format, try to find single letter A-D
            else:
                # Look for standalone option letters
                matches = re.findall(r'\b([A-Da-d])\b', s)
                if matches:
                    s = matches[-1]  # Take the last match
        except Exception as e:
            logger.warning(f"Error normalizing answer '{s}': {e}")
        
        return s.lower().strip()

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute exact match score between prediction and ground truth answers.
        Score is 1.0 if strings match exactly after normalization, 0.0 otherwise.
        """
        normalized_gt = self.normalize_answer(ground_truth)
        normalized_pred = self.normalize_answer(prediction)
        score = 1.0 if normalized_pred == normalized_gt else 0.0
        return score, normalized_pred

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
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost, logprob

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob"]

