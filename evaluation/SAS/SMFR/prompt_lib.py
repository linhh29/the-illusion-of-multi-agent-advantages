# -*- coding: utf-8 -*-
"""Prompt and role config for SMFR (aligned with AFlow task format)."""
TEMPERATURE = 1.0
MAX_TOKENS = 32768

SYSTEM_PROMPT_SMFR = (
    "You are solving a smfr investment analysis problem. "
    "Provide your reasoning (analysis), final answer (investor name or names), and Python code with a solve() function that returns a dict containing investor_dates, comparison, and answer."
)

ROLE_MAP = {
    "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
    "FinancialAnalyst": "You are a Financial Analyst expert in interpreting smfr price data and investment timelines.",
    "DataScientist": "You are a Data Scientist skilled in parsing tables and dates and deriving correct conclusions.",
    "Programmer": "You are an expert Programmer who writes correct Python code to implement solutions.",
}


def construct_ranking_message(responses, question, qtype):
    if qtype == "smfr":
        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions from other agents: "
        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response
        prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
        return {"role": "user", "content": prefix_string}
    raise ValueError("Question type is incorrect.", qtype)


def construct_message(responses, question, qtype):
    if qtype == "smfr":
        if len(responses) == 0:
            prefix_string = (
                "Here is the question:\n" + question + "\n\n"
                "Provide your analysis, your final answer (investor name or names), and Python code with a solve() function that returns a dictionary with keys: investor_dates, comparison, answer."
            )
            return {"role": "user", "content": prefix_string}
        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions from other agents: "
        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response
        prefix_string = prefix_string + "\n\nUsing the reasoning from other agents, give your updated analysis, final answer, and code. Also give a score from 1 to 5 for each other agent's solution. Put all scores in the form [[1, 5, 2, ...]]."
        return {"role": "user", "content": prefix_string}
    raise ValueError("Question type is incorrect.", qtype)
