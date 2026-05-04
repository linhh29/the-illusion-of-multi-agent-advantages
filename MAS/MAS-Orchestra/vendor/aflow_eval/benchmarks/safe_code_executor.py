"""
Safe code executor for running model-generated solutions.

This module provides a sandboxed environment for executing model-generated
code to solve problems. It includes safety features like timeouts, restricted
imports, and isolated namespaces.
"""

import ast
import concurrent.futures
import json
import signal
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional


class SafeCodeExecutor:
    """
    Executes model-generated code in a sandboxed environment.

    Safety features:
    - Timeout limits (default 10 seconds)
    - Restricted imports (whitelist only)
    - No file system access
    - No network access
    - Isolated namespace
    """

    # Blacklist of disallowed imports (security risks)
    DISALLOWED_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
        'http', 'ftplib', 'smtplib', 'telnetlib', 'asyncio',
        '__builtin__', '__builtins__', 'builtins',
        'importlib', 'imp', 'pkgutil', 'multiprocessing', 'threading',
        'ctypes', 'cffi', 'pty', 'fcntl', 'resource', 'mmap',
        'pickle', 'shelve', 'dbm', 'sqlite3', 'marshal', 'code',
        'codeop', 'pdb', 'bdb', 'inspect', 'dis', 'gc'
    }

    # Restricted built-ins (removed for security)
    RESTRICTED_BUILTINS = {
        'open', 'eval', 'exec', 'compile', '__import__',
        'exit', 'quit', 'help', 'input', 'breakpoint'
    }

    def __init__(self, timeout: int = 10, max_output_length: int = 10000):
        """
        Initialize the safe code executor.

        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum length of captured output
        """
        self.timeout = timeout
        self.max_output_length = max_output_length

    def execute(self, code: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model-generated code with given inputs.

        Args:
            code: Python code to execute (should define a function)
            inputs: Dictionary of input parameters

        Returns:
            Dictionary with:
                - 'success': bool
                - 'result': Any (if success)
                - 'error': str (if failure)
                - 'error_type': str (if failure)
                - 'execution_time': float (seconds)
        """
        start_time = datetime.now()

        # Validate code first
        validation_result = self._validate_code(code)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['error'],
                'error_type': 'ValidationError',
                'execution_time': 0.0
            }

        # Execute with timeout
        try:
            result = self._execute_with_timeout(code, inputs)
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'result': result,
                'execution_time': execution_time
            }

        except TimeoutError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                'success': False,
                'error': f'Execution timed out after {self.timeout} seconds',
                'error_type': 'TimeoutError',
                'execution_time': execution_time
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
                'execution_time': execution_time
            }

    def _validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate code for safety before execution.

        Args:
            code: Python code to validate

        Returns:
            Dictionary with 'valid' (bool) and 'error' (str) if invalid
        """
        try:
            # Parse code to AST
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f'Syntax error: {str(e)}'
            }

        # Check for restricted operations
        for node in ast.walk(tree):
            # Check for disallowed imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_base = alias.name.split('.')[0]  # Get base module name
                    if module_base in self.DISALLOWED_MODULES:
                        return {
                            'valid': False,
                            'error': f'Import "{alias.name}" is not allowed for security reasons. Disallowed modules: {sorted(self.DISALLOWED_MODULES)}'
                        }

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_base = node.module.split('.')[0]  # Get base module name
                    if module_base in self.DISALLOWED_MODULES:
                        return {
                            'valid': False,
                            'error': f'Import from "{node.module}" is not allowed for security reasons. Disallowed modules: {sorted(self.DISALLOWED_MODULES)}'
                        }

            # Check for file operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.RESTRICTED_BUILTINS:
                        return {
                            'valid': False,
                            'error': f'Use of "{node.func.id}" is not allowed for security reasons'
                        }

        # Check that code defines at least one function
        has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        if not has_function:
            return {
                'valid': False,
                'error': 'Code must define at least one function to execute'
            }

        return {'valid': True}

    @staticmethod
    def _can_use_sigalrm_timeout() -> bool:
        """``signal.signal`` / ``SIGALRM`` only work on the main thread (Unix)."""
        if sys.platform == "win32":
            return False
        return threading.current_thread() is threading.main_thread()

    def _run_user_code(self, code: str, inputs: Dict[str, Any]) -> Any:
        """Load ``code``, find ``solve``/``main``/first function, call with ``inputs``."""
        namespace = self._create_safe_namespace()
        exec(code, namespace)
        func_name = self._find_main_function(code, namespace)
        if func_name not in namespace:
            raise ValueError(f'Could not find function "{func_name}" in code')
        func = namespace[func_name]
        return func(**inputs)

    def _execute_with_timeout_pool(self, code: str, inputs: Dict[str, Any]) -> Any:
        """
        Enforce timeout without signals (safe from worker threads, Windows, asyncio workers).
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self._run_user_code, code, inputs)
            try:
                return fut.result(timeout=float(self.timeout))
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Execution timed out after {self.timeout} seconds"
                ) from None

    def _execute_with_timeout(self, code: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute code with timeout and safety restrictions.

        Args:
            code: Python code to execute
            inputs: Input parameters

        Returns:
            Result from function execution

        Raises:
            TimeoutError: If execution exceeds timeout
            Exception: Any exception from code execution
        """
        if not self._can_use_sigalrm_timeout():
            return self._execute_with_timeout_pool(code, inputs)

        @contextmanager
        def timeout_handler(seconds):
            def timeout_signal_handler(signum, frame):
                raise TimeoutError(f'Execution exceeded {seconds} second timeout')

            old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        with timeout_handler(self.timeout):
            return self._run_user_code(code, inputs)

    def _create_safe_namespace(self) -> Dict[str, Any]:
        """
        Create a safe namespace with restricted built-ins.

        Returns:
            Dictionary namespace for code execution
        """
        # Get __builtins__ as dict (it can be a dict or module)
        if isinstance(__builtins__, dict):
            builtins_dict = __builtins__
        else:
            builtins_dict = __builtins__.__dict__

        # Start with safe built-ins
        safe_builtins = {
            name: builtins_dict[name]
            for name in builtins_dict
            if name not in self.RESTRICTED_BUILTINS and not name.startswith('_')
        }

        # Add a safe __import__ that blocks disallowed modules
        original_import = builtins_dict['__import__']

        def safe_import(name, *args, **kwargs):
            module_base = name.split('.')[0]  # Get base module name
            if module_base in self.DISALLOWED_MODULES:
                raise ImportError(f"Import of '{name}' is not allowed for security reasons. Disallowed modules: {sorted(self.DISALLOWED_MODULES)}")
            return original_import(name, *args, **kwargs)

        safe_builtins['__import__'] = safe_import

        # Add namespace with common dunder variables
        namespace = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
        }

        return namespace

    def _find_main_function(self, code: str, namespace: Dict[str, Any]) -> str:
        """
        Find the main function to call.

        Strategy:
        1. Look for 'solve' function
        2. Look for 'main' function
        3. Take the first function defined

        Args:
            code: Source code
            namespace: Execution namespace

        Returns:
            Name of function to call
        """
        # Parse code to find function definitions
        tree = ast.parse(code)
        function_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]

        # Priority order
        if 'solve' in function_names:
            return 'solve'
        elif 'main' in function_names:
            return 'main'
        elif function_names:
            return function_names[0]
        else:
            raise ValueError('No functions found in code')

    def batch_execute(self, code: str, inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute code with multiple input sets.

        Args:
            code: Python code to execute
            inputs_list: List of input dictionaries

        Returns:
            List of result dictionaries
        """
        results = []
        for inputs in inputs_list:
            result = self.execute(code, inputs)
            results.append(result)
        return results


class CodeExecutionEvaluator:
    """
    Evaluates model-generated code by comparing execution results with expected answers.
    """

    def __init__(self, executor: Optional[SafeCodeExecutor] = None):
        """
        Initialize evaluator.

        Args:
            executor: SafeCodeExecutor instance (creates default if None)
        """
        self.executor = executor or SafeCodeExecutor()

    def evaluate(self, code: str, inputs: Dict[str, Any], expected_answer: Any,
                tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Evaluate code by comparing result with expected answer.

        Args:
            code: Model-generated code
            inputs: Input parameters
            expected_answer: Expected correct answer
            tolerance: Tolerance for floating-point comparisons

        Returns:
            Dictionary with evaluation results
        """
        # Execute code
        execution_result = self.executor.execute(code, inputs)

        if not execution_result['success']:
            return {
                'correct': False,
                'execution_success': False,
                'error': execution_result['error'],
                'error_type': execution_result['error_type'],
                'execution_time': execution_result['execution_time']
            }

        # Compare result with expected answer
        result = execution_result['result']
        correct = self._compare_results(result, expected_answer, tolerance)

        return {
            'correct': correct,
            'execution_success': True,
            'result': result,
            'expected': expected_answer,
            'execution_time': execution_result['execution_time']
        }

    def _compare_results(self, result: Any, expected: Any, tolerance: float) -> bool:
        """
        Compare execution result with expected answer.

        Handles different types: numbers, strings, lists, dates, and structured dicts.

        Args:
            result: Actual result from code execution
            expected: Expected answer
            tolerance: Tolerance for float comparison

        Returns:
            True if results match
        """
        # Handle None
        if result is None and expected is None:
            return True
        if result is None or expected is None:
            return False

        # Handle structured answers (for comparative questions)
        if isinstance(result, dict) and isinstance(expected, dict):
            return self._compare_structured_answer(result, expected, tolerance)

        # Handle floats with tolerance
        if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
            return abs(float(result) - float(expected)) <= tolerance

        # Handle lists
        if isinstance(result, list) and isinstance(expected, list):
            if len(result) != len(expected):
                return False
            return all(self._compare_results(r, e, tolerance) for r, e in zip(result, expected))

        # Handle strings (with special date handling)
        if isinstance(result, str) and isinstance(expected, str):
            # Try to parse as dates first
            result_date = self._try_parse_date(result)
            expected_date = self._try_parse_date(expected)

            if result_date and expected_date:
                # Both are dates, compare them
                return result_date == expected_date

            # Not dates or parsing failed, do string comparison
            return result.strip().lower() == expected.strip().lower()

        # Direct comparison for other types
        return result == expected

    def _compare_structured_answer(self, result: Dict[str, Any], expected: Dict[str, Any],
                                   tolerance: float) -> bool:
        """
        Compare structured answers for comparative questions.

        Expected structure:
        {
            "investor_dates": {"Alice": [...], "Bob": [...]},
            "comparison": {"Alice": "date", "Bob": "date"},
            "answer": "Alice"
        }

        Args:
            result: Result from code execution
            expected: Expected structured answer
            tolerance: Tolerance for comparisons

        Returns:
            True if all components match
        """
        # Check all required keys present
        required_keys = {"investor_dates", "comparison", "answer"}
        if not required_keys.issubset(result.keys()) or not required_keys.issubset(expected.keys()):
            return False

        # 1. Compare investor_dates (dict of lists)
        if not isinstance(result["investor_dates"], dict) or not isinstance(expected["investor_dates"], dict):
            return False

        if set(result["investor_dates"].keys()) != set(expected["investor_dates"].keys()):
            return False

        for investor in expected["investor_dates"]:
            if not self._compare_results(result["investor_dates"][investor],
                                        expected["investor_dates"][investor],
                                        tolerance):
                return False

        # 2. Compare comparison (dict of first dates or None)
        if not isinstance(result["comparison"], dict) or not isinstance(expected["comparison"], dict):
            return False

        if set(result["comparison"].keys()) != set(expected["comparison"].keys()):
            return False

        for investor in expected["comparison"]:
            if not self._compare_results(result["comparison"][investor],
                                        expected["comparison"][investor],
                                        tolerance):
                return False

        # 3. Compare final answer
        if not self._compare_results(result["answer"], expected["answer"], tolerance):
            return False

        return True

    def _try_parse_date(self, date_str: str) -> Optional[Any]:
        """
        Try to parse a string as a date.

        Supports multiple formats:
        - "November 15, 2025"
        - "Nov 15, 2025"
        - "2025-11-15"
        - "11/15/2025"
        - "15-11-2025"

        Args:
            date_str: String that might be a date

        Returns:
            datetime object if parsing succeeds, None otherwise
        """
        from datetime import datetime

        # List of date formats to try
        formats = [
            "%B %d, %Y",      # November 15, 2025
            "%b %d, %Y",      # Nov 15, 2025
            "%Y-%m-%d",       # 2025-11-15
            "%m/%d/%Y",       # 11/15/2025
            "%d/%m/%Y",       # 15/11/2025
            "%d-%m-%Y",       # 15-11-2025
            "%Y/%m/%d",       # 2025/11/15
            "%B %d %Y",       # November 15 2025 (no comma)
            "%b %d %Y",       # Nov 15 2025 (no comma)
        ]

        date_str = date_str.strip()

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None


# Convenience functions

def execute_code(code: str, inputs: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
    """
    Convenience function to execute code.

    Args:
        code: Python code to execute
        inputs: Input parameters
        timeout: Timeout in seconds

    Returns:
        Execution result dictionary
    """
    executor = SafeCodeExecutor(timeout=timeout)
    return executor.execute(code, inputs)


def evaluate_code(code: str, inputs: Dict[str, Any], expected_answer: Any,
                 tolerance: float = 1e-6, timeout: int = 10) -> Dict[str, Any]:
    """
    Convenience function to evaluate code.

    Args:
        code: Python code to execute
        inputs: Input parameters
        expected_answer: Expected answer
        tolerance: Float comparison tolerance
        timeout: Timeout in seconds

    Returns:
        Evaluation result dictionary
    """
    executor = SafeCodeExecutor(timeout=timeout)
    evaluator = CodeExecutionEvaluator(executor)
    return evaluator.evaluate(code, inputs, expected_answer, tolerance)
