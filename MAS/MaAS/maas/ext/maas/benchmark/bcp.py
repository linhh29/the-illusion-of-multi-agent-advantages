import asyncio
import re
import torch
from typing import Any, Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.configs.models_config import ModelsConfig
from maas.provider.llm_provider_registry import create_llm_instance
from maas.logs import logger


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()


class BCPBenchmark(BaseBenchmark):
    """
    BrowseCompPlus Benchmark
    Uses an LLM grader (default gpt-4o) to judge free-form answers.
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
        grader_model_name: str = "gpt-4o",
    ):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer)

        models_config = ModelsConfig.default()
        grader_llm_config = models_config.get(grader_model_name)
        if grader_llm_config is None:
            raise ValueError(f"Grader model '{grader_model_name}' not found in models config.")
        self.grader_model = create_llm_instance(grader_llm_config)

    async def grade_sample(self, question: str, correct_answer: str, response: str) -> bool:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            response=response,
            correct_answer=correct_answer,
        )
        grading_response = await self.grader_model.aask(grader_prompt)

        # Extract correct (yes/no)
        correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", grading_response, re.IGNORECASE)
        if not correct_match:
            correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", grading_response, re.IGNORECASE)
        if not correct_match:
            correct_match = re.search(r"correct:\s*(yes|no)", grading_response, re.IGNORECASE)

        if correct_match:
            return correct_match.group(1).lower() == "yes"
        return False

    def normalize_answer(self, text: str) -> str:
        """Lightweight normalization for free-form answers."""
        if text is None:
            return ""
        text = str(text).strip()
        text = text.strip('"').strip("'")
        text = text.rstrip(".,;!?")
        text = text.lower().strip()
        return text

    async def calculate_score(self, inputs, correct_answer: str, prediction: str) -> Tuple[float, Any]:
        """
        Primary: LLM grader. Fallback: relaxed string match (case/quote/punct stripped).
        """
        grader_correct = await self.grade_sample(inputs, correct_answer, prediction)
        if grader_correct:
            return 1.0, prediction

        norm_gt = self.normalize_answer(correct_answer)
        norm_pred = self.normalize_answer(prediction)
        score = 1.0 if norm_gt and (norm_gt == norm_pred) else 0.0
        return score, prediction

    @retry(stop=stop_after_attempt(20), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await asyncio.wait_for(graph(input_text), timeout=1500)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float, torch.Tensor]:
        input_text = problem["question"]
        expected_output = problem["answer"]

        try:
            output, cost, logprob = await self._generate_output(graph, input_text)

            extracted_output = output
            score, _ = await self.calculate_score(input_text, expected_output, output)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            # Ensure logprob tensor type
            if not isinstance(logprob, torch.Tensor):
                logprob = torch.tensor(logprob if logprob is not None else 0.0, dtype=torch.float32, device=self.device)

            return input_text, output, expected_output, score, cost, logprob

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob"]

