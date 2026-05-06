TEMPERATURE = 1.0
MAX_TOKENS = 32768

SYSTEM_PROMPT_SWE = """You are an expert software engineer specializing in fixing bugs and implementing patches.
You must make sure:
(1) the patch is correct and can be applied to the code.
(2) Please note that the patch REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
(3) Wrap each patch in a code block or <patch> tags. If you have multiple patches, use a separate code block for each one.
(4) Your patch must be significant enough to change the PASS or FAIL status of potential test cases. DO NOT include trivial patches like changing the doc string, adding empty lines, adding comments or changing variable names as these trivial patches cannot change a failed test case to passed.
(5) The patch must be COMPLETE CODE and without any syntax error. Please implement complete, reliable, reusable code snippets.
(6) A user will run unix's patch program directly to apply the patch, so please make sure the patch is correct and directly runnable by the unix's patch program.

The patch format should be a unified diff format, starting with 'diff --git' or '--- a/...'.
IMPORTANT: Remember to contain line information like @@ -333,7 +333,7 @@ in your patch!!!"""

ROLE_MAP = {
    "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
    "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
    "CodeReviewer": "You are a code reviewer with extensive experience in software engineering. You excel at identifying bugs, understanding code structure, and proposing fixes.",
    "SoftwareEngineer": "You are a software engineer specializing in debugging and fixing complex software issues. You have deep knowledge of various programming languages and software architectures.",
    "DebuggingExpert": "You are an expert in debugging software. You can quickly identify root causes of bugs and implement effective fixes.",
}

AGENTLESS_REPAIR = """
You must make sure 
(1) the patch is correct and can be applied to the code. 
(2) Please note that the patch REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
(3) Wrap each patch in a code block as shown in the example above. If you have multiple patches, use a separate code block for each one. For example,
(4) Your patch must be significant enough to change the PASS or FAIL status of potential test cases. DO NOT include trivial patch like change the doc string, add empty lines, add comments or change the variable names as these trivial patches cannot change a failed test cases to passed.
(5) The patch must be COMPLETE CODE and without any syntax error. Please implement complete, reliable, reusable code snippets.
(6) A user will run unix's patch program directly to apply the patch, so please make sure the patch is correct and directly runnable by the unix's patch program.

Examples:

This is CORRECT patch:
 "diff --git a/src/_pytest/python_api.py b/src/_pytest/python_api.py\nindex a3d0b90..b1a7c6a 100644\n--- a/src/_pytest/python_api.py\n+++ b/src/_pytest/python_api.py\n@@ -711,8 +711,15 @@ def raises(  # noqa: F811\n         except expected_exception as e:\n             # We just caught the exception - there is a traceback.\n             assert e.__traceback__ is not None\n-            return _pytest._code.ExceptionInfo.from_exc_info(\n-                (type(e), e, e.__traceback__)\n+            exc_info = (type(e), e, e.__traceback__)\n+            \n+            # Walk the traceback chain to get the full exception chain\n+            while exc_info[2].tb_next is not None:\n+                exc_info = (\n+                    type(exc_info[1]), \n+                    exc_info[1], \n+                    exc_info[2].tb_next\n+                )\n+            return _pytest._code.ExceptionInfo.from_exc_info(exc_info\n             )\n     fail(message)\n"

This is CORRECT patch:
 "--- a/django/db/models/deletion.py\n+++ b/django/db/models/deletion.py\n@@ -329,7 +329,13 @@\n             for model, instances in self.data.items():\n                 query = sql.DeleteQuery(model)\n                 pk_list = [obj.pk for obj in instances]\n-                count = query.delete_batch(pk_list, self.using)\n+                # Combine delete queries by table\n+                by_table = {}\n+                for pk in pk_list:\n+                    by_table.setdefault(model._meta.db_table, []).append(pk)\n+                for table, pks in by_table.items():\n+                    query.table = table\n+                    count = query.delete_batch(pks, self.using)\n                 deleted_counter[model._meta.label] += count\n \n                 if not model._meta.auto_created:\n"

IMPORTANT:
Remember to contain line information like @@ -333,7 +333,7 @@ in your patch!!!
"""

def construct_ranking_message(responses, question, qtype):
    if qtype == "code_patch":
        prefix_string = "Here is the problem:\n" + question + "\n\nThese are the patches proposed by other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent patch " + str(aid) + ": ```\n{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nPlease choose the best 2 patches and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."
    else:
        raise ValueError("Question type is incorrect.", qtype)

    return {"role": "user", "content": prefix_string}

def construct_message(responses, question, qtype):
    if qtype == "code_patch":
        if len(responses) == 0:
            prefix_string = question + "\n\n" + AGENTLESS_REPAIR + "\n\nPlease provide a patch in unified diff format. Wrap your patch in <patch> tags or a code block."
            return {"role": "user", "content": prefix_string}

        prefix_string = "Here is the problem:\n" + question + "\n\nThese are the patches proposed by other agents: "

        for aid, aresponse in enumerate(responses, 1):
            response = "\n\nAgent patch " + str(aid) + ": ```\n{}```".format(aresponse)
            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\nUsing the patches from other agents as additional advice with critical thinking, can you provide an improved patch? Examine the patches step by step. Notice that their patches might be all wrong. Please provide your patch in unified diff format, wrapped in <patch> tags or a code block. Along with the patch, give a score ranged from 1 to 5 to the patches of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))
    else:
        raise ValueError("Question type is incorrect.", qtype)

    return {"role": "user", "content": prefix_string}

def construct_ensemble_message(patches, question):
    """
    Construct a prompt for LLM to ensemble multiple patches.
    The LLM should analyze all patches and select/merge the best one.
    """
    prefix_string = "Here is the problem:\n" + question + "\n\n"
    prefix_string += "Below are multiple patches proposed by different attempts. Your task is to:\n"
    prefix_string += "1. Analyze each patch carefully\n"
    prefix_string += "2. Identify which patch is most likely to be correct\n"
    prefix_string += "3. If multiple patches have correct parts, you may merge them intelligently\n"
    prefix_string += "4. Provide the final best patch in unified diff format\n\n"
    prefix_string += "These are the patches:\n\n"
    
    for idx, patch in enumerate(patches, 1):
        prefix_string += f"Patch {idx}:\n```\n{patch}\n```\n\n"
    
    prefix_string += "\n" + AGENTLESS_REPAIR + "\n\n"
    prefix_string += "Please analyze all patches above and provide the best final patch. "
    prefix_string += "Wrap your final patch in <patch> tags or a code block. "
    prefix_string += "If you think one patch is clearly correct, use that one. "
    prefix_string += "If multiple patches have correct elements, you may combine them intelligently. "
    prefix_string += "Make sure the final patch is syntactically correct and can be applied by the unix patch program."
    
    return {"role": "user", "content": prefix_string}

