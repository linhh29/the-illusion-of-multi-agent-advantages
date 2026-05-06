# Output format for SMFR must match ReverseAnswerCodeResponse (analysis, answer, code)
# so the benchmark can parse it and run code evaluation.

SC_ENSEMBLE_PROMPT = """
Several answers have been generated to the same question. They are as follows:
{solutions}

Synthesize the best solution from the above: pick the most consistent and correct one, or merge the best parts. You must output a single structured response with exactly these three fields (same format as each solution):

1. "analysis": Step-by-step reasoning explaining why you chose or merged this answer.
2. "answer": The final answer - ONLY the name(s), e.g. "Alice" or ["Alice", "Bob"] if tied. No extra text.
3. "code": Python code that defines a solve() function returning a dict with an "answer" key (and any other keys the problem needs). All input data must be included in the code.

Output only valid JSON with keys: analysis, answer, code.
"""

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem. You must output a single structured response with exactly these three fields:

1. "analysis": Step-by-step reasoning.
2. "answer": The final answer - ONLY the name(s), e.g. "Alice" or ["Alice", "Bob"] if tied. No extra text.
3. "code": Python code with a solve() function that returns a dict containing "answer" (and any other keys the problem needs). Include all required input data in the code.

Output only valid JSON with keys: analysis, answer, code.

Your task: {input}
"""