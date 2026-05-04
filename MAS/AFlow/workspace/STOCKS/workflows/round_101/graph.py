from typing import Literal
import workspace.STOCKS.workflows.template.operator as operator
import workspace.STOCKS.workflows.round_1.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        N = 3

        possible_answers = []
        for i in range(N):
            solution = await self.custom(input=problem, instruction='Can you solve the following question? Please reasoning step-by-step.')
            possible_answers.append(solution)
        
        answer = await self.sc_ensemble(solutions=possible_answers)
        return answer['response'], self.llm.get_usage_summary()["total_cost"]