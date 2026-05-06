TEMPERATURE = 1.0
MAX_TOKENS = 32768

SYSTEM_PROMPT_HLEMATH = "It's a debate. Explain your reasons at each round thoroughly.\nFollow the given examples and answer the mathematics problem."

ROLE_MAP_MATH = {
    "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
    "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
    "AlgebraExpert": "You are an expert in the field of algebra, skilled at solving equations, understanding variables and adept at the logical manipulation of symbols.",
    "CountingProbabilitySpecialist": "You specialize in the realm of counting and probability, able to calculate complex events with accuracy, analyze data and predict outcomes.",
    "GeometryWizard": "You are a wizard of geometry, deeply familiar with shapes, dimensions, and properties, and capable of theorizing spatial relationships and understanding geometric proofs.",
    "IntermediateAlgebraMaestro": "You are a maestro of intermediate algebra, adept at handling polynomials, quadratic equations, and dealing with complex numerical relationships.",
    "NumberTheoryScholar": "As a scholar in number theory, you excel in studying properties and relationships of numbers. Prime numbers, divisibility, and mathematical patterns are within your area of expertise.",
    "PrealgebraProdigy": "You are a prodigy in prealgebra, skillful at understanding mathematical principles and fundamentals like operations, fractions, and basic equations.",
    "PrecalculusGuru": "You are a guru in precalculus, proficient at handling functions, limits, rates of change, and confidently preparing for the concepts of calculus."
}

def construct_ranking_message(responses, question, qtype):
    if qtype == "math_exp":
        prefix_string = "Follow the given examples and answer the mathematics problem.\n\n" + question +  "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
    else:
        raise ValueError("Question type is incorrect.", qtype)

    return {"role": "user", "content": prefix_string}

def construct_message(responses, question, qtype):
    if qtype == "math_exp":
        if len(responses) == 0:
            return {"role": "user", "content": question}

        prefix_string = "Follow the given examples and answer the mathematics problem.\n\n" + question +  "\n\nThese are the solutions to the problem from other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nUsing the reasoning from other agents as additional advice with critical thinking, can you give an updated answer? Examine your solution and that other agents step by step. Notice that their answers might be all wrong. Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))
    else:
        raise ValueError("Question type is incorrect.", qtype)

    return {"role": "user", "content": prefix_string}

