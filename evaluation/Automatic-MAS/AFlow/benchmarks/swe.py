import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger
from scripts.utils.sanitize import sanitize
from swe_utils import run_swebench_evaluation
import re
import os
import subprocess

AGENTLESS_REPAIR = """
You must make sure 
(1) the patch is correct and can be applied to the code. 
(2) Please note that the patch REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
(3) Wrap each patch in a code block as shown in the example above. If you have multiple patchs, use a separate code block for each one. For example,
(5) Your patch must be significant enough to change the PASS or FAIL status of potential test cases. DO NOT include trivial patch like change the doc string, add empty lines, add comments or change the vairable names as these trivial patches cannot change a failed test cases to passed.
(6) The patch must be COMPLETE CODE and without any syntax error. Please implement complete, reliable, reusable code snippets.
(7) A user will run unix's path program directly to apply the patch, so please make sure the patch is correct and directly runnable by the unix's path program.

Examples:

This is CORRECT patch:
 "diff --git a/src/_pytest/python_api.py b/src/_pytest/python_api.py\nindex a3d0b90..b1a7c6a 100644\n--- a/src/_pytest/python_api.py\n+++ b/src/_pytest/python_api.py\n@@ -711,8 +711,15 @@ def raises(  # noqa: F811\n         except expected_exception as e:\n             # We just caught the exception - there is a traceback.\n             assert e.__traceback__ is not None\n-            return _pytest._code.ExceptionInfo.from_exc_info(\n-                (type(e), e, e.__traceback__)\n+            exc_info = (type(e), e, e.__traceback__)\n+            \n+            # Walk the traceback chain to get the full exception chain\n+            while exc_info[2].tb_next is not None:\n+                exc_info = (\n+                    type(exc_info[1]), \n+                    exc_info[1], \n+                    exc_info[2].tb_next\n+                )\n+            return _pytest._code.ExceptionInfo.from_exc_info(exc_info\n             )\n     fail(message)\n"

This is CORRECT patch:
 "--- a/django/db/models/deletion.py\n+++ b/django/db/models/deletion.py\n@@ -329,7 +329,13 @@\n             for model, instances in self.data.items():\n                 query = sql.DeleteQuery(model)\n                 pk_list = [obj.pk for obj in instances]\n-                count = query.delete_batch(pk_list, self.using)\n+                # Combine delete queries by table\n+                by_table = {}\n+                for pk in pk_list:\n+                    by_table.setdefault(model._meta.db_table, []).append(pk)\n+                for table, pks in by_table.items():\n+                    query.table = table\n+                    count = query.delete_batch(pks, self.using)\n                 deleted_counter[model._meta.label] += count\n \n                 if not model._meta.auto_created:\n"}

This is CORRECT patch:
"--- a/src/_pytest/python.py\n+++ b/src/_pytest/python.py\n@@ -271,7 +271,7 @@ class Function(PyobjMixin, Node):\n         except KeyboardInterrupt:\n             raise\n         except: # noqa\n-            return s.replace(\".[\", \"[\")\n+            return s\n \n         if name == \"__init__\":\n             cls = getattr(self.obj, \"__qualname__\", None)\n"


This is CORRECT patch:
"--- a/sympy/combinatorics/homomorphisms.py\n+++ b/sympy/combinatorics/homomorphisms.py\n@@ -333,7 +333,7 @@\n             if r[i] in gens:\n                 s = domain.generators[gens.index(r[i])]\n             else:\n-                s = r[i]\n+                s = r[i] ** -1\n             if s in images:\n                 w = w*images[s]**power\n             elif s**-1 in images:\n"


This is CORRECT patch:
"diff --git a/src/_pytest/logging.py b/src/_pytest/logging.py\nindex 4d7c65e..2b60f1d 100644\n--- a/src/_pytest/logging.py\n+++ b/src/_pytest/logging.py\n@@ -458,6 +458,7 @@ class LogCaptureFixture:\n             self.handler.setLevel(handler_orig_level)\n \n     def _finalize(self) -> None:\n+        self.handler.reset()\n         \"\"\"Finalizes the fixture.\n \n         This restores the log levels changed by :meth:`set_level`.\n"

"""


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses 

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1) if match else ""


class SWEBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, args, timeout):
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func(*args))
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def check_solution(self, solution, test, entry_point):
        solution = sanitize(code=solution, entrypoint=entry_point)
        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            # Add handling for special cases
            if entry_point == "decode_cyclic":
                solution = (
                    '\n\ndef encode_cyclic(s: str):\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n    # split string to groups. Each of length 3.\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    # cycle elements in each group. Unless group has fewer elements than 3.\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)'
                    + "\n\n"
                    + solution
                )
            elif entry_point == "decode_shift":
                solution = (
                    '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                    + solution
                )
            elif entry_point == "find_zero":
                solution = (
                    "\n\ndef poly(xs: list, x: float):\n    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))\n\n"
                    + solution
                )

            exec(solution, global_dict)

            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")

            exec(test, global_dict)

            check = global_dict["check"]

            result = self.run_with_timeout(check, (global_dict[entry_point],), 15)

            if result is None:
                result = (self.PASS, "The solution passed all test cases.")

        except self.TimeoutError:
            result = (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
        except Exception as e:
            error_message = f"Error: {str(e)}.\n Solution: {solution}.\n Test: {test}"
            result = (self.FAIL, error_message)

            with open("error.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")

        return result

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, prompt):
        # Generate output with a timeout of 60 seconds
        return await asyncio.wait_for(graph(prompt), timeout=60)

    async def evaluate_problem(self, data: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        instance_id = data['instance_id']
        input_text = data["text"] + AGENTLESS_REPAIR
        # input_text = data["problem_statement"]
        expected_output = data['patch']
        code_snippet = extract_xml(input_text, 'code').strip()

        # print('111111')
        # print(self.file_path)
        # print(self.log_path)
        judge_path = self.log_path
        # for i in range(1, 6):
        os.makedirs(f'{judge_path}/results', exist_ok=True)
        os.makedirs(f'{judge_path}/reports', exist_ok=True)

        try:
            # Generate prediction using the graph function
            prediction, cost = await self._generate_output(graph, input_text)

            scores = []
            # pass@K
            if isinstance(prediction, list):
                print('Pass@K', len(prediction))
                for pred in prediction:
                    tmp_score = await run_swebench_evaluation(judge_path, instance_id, pred, '', '', code_snippet, self.file_path)
                    scores.append(tmp_score)
                    # remove docker container to avoid lock
                    container_name = 'sweb.eval.' + instance_id + '.' + instance_id.replace('-', '_') + '__'
                    result = subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
                
                    if result.returncode == 0:
                        print("Remove Container ----- Directory listing:")
                        print(result.stdout)
                    else:
                        print("Remove Container ----- Error:", result.stderr)
            # pass@1
            else:
                tmp_score = await run_swebench_evaluation(judge_path, instance_id, prediction, '', '', code_snippet, self.file_path)
                scores.append(tmp_score)
                # remove docker container to avoid lock
                container_name = 'sweb.eval.' + instance_id + '.' + instance_id.replace('-', '_') + '__'
                result = subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
            
                if result.returncode == 0:
                    print("Remove Container ----- Directory listing:")
                    print(result.stdout)
                else:
                    print("Remove Container ----- Error:", result.stderr)

            if sum(scores) >= 1:
                score = 1
            else:
                score = 0

            # Log mismatch if the score is 0
            if score == 0:
                self.log_mismatch(input_text, expected_output, prediction, score)

            return input_text, prediction, expected_output, score, cost

        except asyncio.TimeoutError:
            logger.info("Timeout error. Skipping this sample.")
            return input_text, "Timeout", expected_output, 0.0, 0.0

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0
        
    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        # The scoring logic for HumanEval is already implemented in evaluate_problem, this is just to conform to the interface
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
