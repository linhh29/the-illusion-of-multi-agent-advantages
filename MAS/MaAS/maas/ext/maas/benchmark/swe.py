import asyncio
import torch
from typing import Any, Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.ext.maas.benchmark.swe_utils import (
    extract_xml,
    run_swebench_evaluation,
)
from maas.logs import logger


AGENTLESS_REPAIR = """
You must make sure 
(1) the patch is correct and can be applied to the code. 
(2) Please note that the patch REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
(3) Wrap each patch in a code block as shown in the example above. If you have multiple patchs, use a separate code block for each one. For example,
(5) Your patch must be significant enough to change the PASS or FAIL status of potential test cases. DO NOT include trivial patch like change the doc string, add empty lines, add comments or change the vairable names as these trivial patches cannot change a failed test cases to passed.
(6) The patch must be COMPLETE CODE and without any syntax error. Please implement complete, reliable, reusable code snippets.
(7) A user will run unix's patch program directly to apply the patch, so please make sure the patch is correct and directly runnable by the unix's patch program.
"""


class SWEBenchmark(BaseBenchmark):
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
        super().__init__(
            name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer
        )

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        # Harness score is already boolean-ish (1/0 resolved_instances)
        return float(prediction), prediction

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_output(self, graph, input_text):
        return await asyncio.wait_for(graph(input_text), timeout=60)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float, torch.Tensor]:
        instance_id = problem["instance_id"]
        input_text = problem["text"] + AGENTLESS_REPAIR
        expected_output = problem["patch"]
        code_snippet = extract_xml(input_text, "code").strip()

        judge_path = self.log_path

        try:
            output, cost, logprob = await self._generate_output(graph, input_text)

            scores: List[float] = []
            extracted_output = ""

            if isinstance(output, list):
                for idx, pred in enumerate(output):
                    extracted_output = pred
                    tmp_score = await run_swebench_evaluation(
                        judge_path,
                        instance_id,
                        pred,
                        technique="maas",
                        solution_name=str(idx),
                        code_snippet=code_snippet,
                        file_path=self.file_path,
                    )
                    scores.append(tmp_score)
            else:
                extracted_output = output
                tmp_score = await run_swebench_evaluation(
                    judge_path,
                    instance_id,
                    output,
                    technique="maas",
                    solution_name="0",
                    code_snippet=code_snippet,
                    file_path=self.file_path,
                )
                scores.append(tmp_score)

            score = 1.0 if sum(scores) >= 1 else 0.0

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            if not isinstance(logprob, torch.Tensor):
                logprob = torch.tensor(
                    logprob if logprob is not None else 0.0,
                    dtype=torch.float32,
                    device=self.device,
                )

            return input_text, output, expected_output, score, cost, logprob

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return (
                input_text,
                str(e),
                expected_output,
                0.0,
                0.0,
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
            )

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob"]

