from typing import Literal
import workspace.HLEMATH.workflows.template.operator as operator
import workspace.HLEMATH.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        N = 3
        
        from collections import Counter
        def majority_voting(answers):
            return Counter(answers).most_common(1)[0][0]
        
        import re
        def extract_model_answer(text):
            pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
            boxed_matches = re.findall(pattern, text, re.DOTALL)
            if boxed_matches:
                return boxed_matches[-1].strip()

            sentence_end_pattern = r"(?<!\d)[.!?]\s+"
            sentences = re.split(sentence_end_pattern, text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences[-1] if sentences else ""

        possible_answers = []
        for i in range(N):
            solution = await self.custom(input=problem, instruction='Can you solve the following question? Please reasoning step-by-step.')
            
            if self.dataset == 'GPQA':
                solution = solution['response'].replace('\\text{', '').split('boxed{')[-1][0]
            elif self.dataset == 'HLEMATH':
                solution = extract_model_answer(solution['response'])
            elif self.dataset == 'BCP':
                solution = solution['response'].replace('\\text{', '').split('boxed{')[-1].split('}')[0]
            else:
                solution = solution['response']
            
            possible_answers.append(solution)
        answer = majority_voting(possible_answers)
        return answer, self.llm.get_usage_summary()["total_cost"]