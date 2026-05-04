import inspect

from async_search import LLMAgentBase


async def forward(self, taskInfo, extra_info):
    # Instruction for initial reasoning
    cot_initial_instruction = self.cot_instruction
    #  Instantiate a new LLM agent specifically for CoT
    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)

    # Instruction to construct a reverse question for verification.
    cot_reverse_instruction = "Construct a reverse question that can verify the correctness of the answer above. The reverse question should be clear and concise, allowing for a straightforward verification process."
    reverse_agent = LLMAgentBase(['thinking', 'answer'], 'Reverse Question Agent', model=self.node_model, temperature=0.0)

    # Instruction for answering this reversed question.
    cot_answer_instruction = "Please think step by step to solve the problem."
    answer_agent = LLMAgentBase(['thinking', 'answer'], 'Answer Agent', model=self.node_model, temperature=0.0)

    # Instruction for reviewing the answer to the reverse question as well as the original answer. Decide the correctness.
    critic_instruction = "Please review the answer to the reverse question above and the answer to the original question, and determine if original answer is correct or not. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    review_agent = LLMAgentBase(['thinking', 'correct'], 'Review Agent', model=self.node_model, temperature=0.0)

    # Prepare the inputs for the CoT agent
    # The input should be a list of Info, and the first one is often the taskInfo
    cot_agent_inputs = [taskInfo]

    # Get the response from the CoT agent
    thinking0, answer0 = await cot_agent(cot_agent_inputs, extra_info, cot_initial_instruction)

    # Generate the reverse question
    thinking1, answer1 = await reverse_agent([taskInfo], extra_info, cot_reverse_instruction, 0)

    # Answering the reverse question
    thinking2, answer2 = await answer_agent([taskInfo, answer1], extra_info, cot_answer_instruction, 0)

    # Reviewing the answer to the reverse question and the original answer
    thinking3, correct = await review_agent([taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2], extra_info, critic_instruction, 0)

    if correct.content == 'True':
        final_answer = self.make_final_answer(thinking0, answer0)
        return final_answer

    cot_instruction_again = "Given the verification process above, please try to reason again by finding the potential flaw."
    thinking4, answer4 = await cot_agent([taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2, thinking3, correct], extra_info, cot_instruction_again, 1)
    final_answer = self.make_final_answer(thinking4, answer4)

    return final_answer


func_string = inspect.getsource(forward)

verification = {
    "thought": "To ensure the correctness of one reasoning process, an LLM is employed to construct some reverse questions that can verify the correctness of the original answer. The LLM then answers these reverse questions and reviews the answers to determine if the original answer is correct. If not, it will refine its reasoning process and try again.",
    "name": "Self-Verification through Reverse Question Construction",
    "code": """{func_string}""".format(func_string=func_string)
}
