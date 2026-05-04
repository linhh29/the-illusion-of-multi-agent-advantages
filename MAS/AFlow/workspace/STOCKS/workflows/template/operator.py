import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.STOCKS.workflows.template.operator_an import *
from workspace.STOCKS.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import re


from scripts.operators import Operator


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        # call_reverse_answer_code returns a JSON string; graph expects solution["response"]
        response_text = await self.llm.call_reverse_answer_code(prompt)
        return {"response": response_text}


class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str, mode: str = None) -> Dict[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response_text = await self.llm.call_reverse_answer_code(prompt)
        return {"response": response_text}


class ScEnsemble(Operator):
    """
    For STOCKS: LLM synthesizes best answer in same format (analysis, answer, code).
    call_reverse_answer_code returns a JSON string, not a dict - do not call .get() on it.
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str]):
        solution_text = ""
        for index, solution in enumerate(solutions):
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        response_text = await self.llm.call_reverse_answer_code(prompt)
        return {"response": response_text}