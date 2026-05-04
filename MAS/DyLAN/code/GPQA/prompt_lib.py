TEMPERATURE = 1.0
MAX_TOKENS = 32768

SYSTEM_PROMPT_GPQA = "Here's a debate. Explain your reasons at each round thoroughly.\nAll questions are single choice."

ROLE_MAP = {
    "Theoretical Physicist": "You are a Theoretical Physicist. Your expertise spans the entire landscape of physics, from the foundational principles of classical mechanics to the complexities of quantum mechanics, special and general relativity, and electromagnetism. You solve graduate-level problems by applying fundamental laws and rigorous mathematical derivations from first principles.",
    "Molecular Chemist": "You are a Molecular Chemist. You are an expert in the structure, properties, synthesis, and reactions of molecules. Your knowledge integrates the core disciplines of organic, inorganic, and physical chemistry. You integrate principles from across chemistry to provide precise, theory-grounded solutions to graduate-level problems.",
    "Cellular Biologist": "You are a Molecular and Cellular Biologist. You are an expert in the intricate mechanisms of life at the molecular level. Your expertise covers genetics, biochemistry, cell signaling, and molecular biology techniques. Your primary function is to analyze biological systems and solve graduate-level problems by detailing the underlying molecular interactions.",
    "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans."
}

def construct_ranking_message(responses, question, qtype):
    if qtype == "single_choice":
        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
    else:
        raise ValueError("Question type is incorrect.", qtype)

    return {"role": "user", "content": prefix_string}

def construct_message(responses, question, qtype):
    if qtype == "single_choice":
        if len(responses) == 0:
            prefix_string = "Here is the question:\n" + question + "\n\nPut your answer in the form \\boxed{{X}} at the end of your response, where X represents choice (A), (B), (C), or (D)."
            return {"role": "user", "content": prefix_string}

        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that their answers might be all wrong. Put your answer in the form \\boxed{{X}} at the end of your response, where X represents choice (A), (B), (C), or (D). Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))
    else:
        raise ValueError("Question type is incorrect.", qtype)

    return {"role": "user", "content": prefix_string}

