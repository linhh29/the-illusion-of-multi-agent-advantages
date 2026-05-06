import re
import string
from typing import Callable, List, Tuple
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger
from scripts.async_llm import AsyncLLM


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


class BCPBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.grader_model = AsyncLLM(config="gpt-4o")

    async def grade_sample(self, question: str, correct_answer: str, response: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            response=response,
            correct_answer=correct_answer,
        )
    
        # response = await self.grader_model(grader_prompt)
        grading_response = await self.grader_model(grader_prompt)

        # Extract correct (yes/no)
        correct_match = re.search(
            r"\*\*correct:\*\*\s*(yes|no)", grading_response, re.IGNORECASE
        )
        if not correct_match:
            correct_match = re.search(
                r"\*\*correct\*\*:\s*(yes|no)", grading_response, re.IGNORECASE
            )
        if not correct_match:
            correct_match = re.search(r"correct:\s*(yes|no)", grading_response, re.IGNORECASE)
        if correct_match:
            correctness = correct_match.group(1).lower() == "yes"
        else:
            correctness = False

        return correctness
    
    async def calculate_score(self, inputs, correct_answer: str, response: str) -> Tuple[float, str]:
        """
        Compute exact match score between prediction and ground truth answers.
        Score is 1.0 if strings match exactly after normalization, 0.0 otherwise.
        """
        if 'boxed{' in response:
            response = response.replace('\\text{', '').split('boxed{')[-1].split('}')[0]
        return (float(await self.grade_sample(inputs, correct_answer, response)), response)
    
    # def calculate_score(self, inputs, correct_answer: str, response: str) -> Tuple[float, str]:
    #     """
    #     Compute exact match score between prediction and ground truth answers.
    #     Score is 1.0 if strings match exactly after normalization, 0.0 otherwise.
    #     """
    #     if 'boxed{' in response:
    #         response = response.replace('\\text{', '').split('boxed{')[-1].split('}')[0]
    #     return (1.0 if correct_answer == response else 0.0, response)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["question"]
        expected_output = problem["answer"]
        inputs = input_text
        # print('inputs:', len(inputs))
        # query = inputs.split('Here are some related documents:\n')[0].split('The query is: ')[1].strip()
        # print('query:', query)
        # inputs = query
        # input_text = query
        try:
            output, cost = await self._generate_output(graph, inputs)

            scores = []
            # pass@K
            if isinstance(output, list):
                print('Pass@K', len(output))
                for pred in output:
                    tmp_score, extracted_output = await self.calculate_score(inputs, expected_output, pred)
                    scores.append(tmp_score)
            # pass@1
            else:
                tmp_score, extracted_output = await self.calculate_score(inputs, expected_output, output)
                scores.append(tmp_score) 

            if sum(scores) >= 1:
                score = 1
            else:
                score = 0
            # score, extracted_output = await self.calculate_score(inputs, expected_output, output)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]