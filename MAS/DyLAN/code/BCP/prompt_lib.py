TEMPERATURE = 1.0
MAX_TOKENS = 32768

SYSTEM_PROMPT_BCP = "Here's a debate. Explain your reasons at each round thoroughly."

ROLE_MAP = {
    "Knowledge Researcher": "You are a Knowledge Researcher. Your expertise spans multiple domains including history, culture, entertainment, sports, and current events. You excel at synthesizing information from diverse sources, cross-referencing facts, and identifying precise answers to complex questions. Your approach involves systematic information gathering, critical evaluation of sources, and connecting disparate pieces of information to solve graduate-level knowledge problems.",
    "Cultural Historian": "You are a Cultural Historian. You specialize in understanding historical events, cultural movements, biographical information, and temporal relationships across different eras and regions. Your knowledge encompasses political history, social history, and the interconnected narratives that shape human civilization. You solve problems by placing information in historical context, identifying chronological patterns, and drawing connections between events, people, and cultural phenomena.",
    "Information Analyst": "You are an Information Analyst. Your expertise lies in extracting, verifying, and synthesizing information from complex textual sources. You excel at understanding nuanced queries, identifying key information requirements, and systematically searching through knowledge to find precise answers. Your approach combines logical reasoning, pattern recognition, and meticulous attention to detail to solve graduate-level information retrieval and analysis problems.",
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
            prefix_string = "Here is the question:\n" + question + "\n\nPut your answer in the form \\boxed{{final answer}} at the end of your response."
            return {"role": "user", "content": prefix_string}

        prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that their answers might be all wrong. Put your answer in the form \\boxed{{final answer}} at the end of your response. Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))
    else:
        raise ValueError("Question type is incorrect.", qtype)

    return {"role": "user", "content": prefix_string}

