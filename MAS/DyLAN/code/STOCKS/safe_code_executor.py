"""
Safe code executor for running model-generated solutions.
(Copied from AFlow benchmarks/safe_code_executor.py for STOCKS evaluation.)
"""

import ast
import signal
import sys
import threading
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
import json


class SafeCodeExecutor:
    """
    Executes model-generated code in a sandboxed environment.
    """

    DISALLOWED_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
        'http', 'ftplib', 'smtplib', 'telnetlib', 'asyncio',
        '__builtin__', '__builtins__', 'builtins',
        'importlib', 'imp', 'pkgutil', 'multiprocessing', 'threading',
        'ctypes', 'cffi', 'pty', 'fcntl', 'resource', 'mmap',
        'pickle', 'shelve', 'dbm', 'sqlite3', 'marshal', 'code',
        'codeop', 'pdb', 'bdb', 'inspect', 'dis', 'gc'
    }

    RESTRICTED_BUILTINS = {
        'open', 'eval', 'exec', 'compile', '__import__',
        'exit', 'quit', 'help', 'input', 'breakpoint'
    }

    def __init__(self, timeout: int = 30, max_output_length: int = 10000):
        self.timeout = timeout
        self.max_output_length = max_output_length

    def execute(self, code: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(code, str):
            code = str(code)
        # ast.parse / compile reject NUL; model output may contain stray \x00
        code = code.replace("\x00", "")
        start_time = datetime.now()
        validation_result = self._validate_code(code)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['error'],
                'error_type': 'ValidationError',
                'execution_time': 0.0
            }
        try:
            result = self._execute_with_timeout(code, inputs)
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time
            }
        except TimeoutError:
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
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {'valid': False, 'error': f'Syntax error: {str(e)}'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_base = alias.name.split('.')[0]
                    if module_base in self.DISALLOWED_MODULES:
                        return {'valid': False, 'error': f'Import "{alias.name}" is not allowed'}
            if isinstance(node, ast.ImportFrom) and node.module:
                module_base = node.module.split('.')[0]
                if module_base in self.DISALLOWED_MODULES:
                    return {'valid': False, 'error': f'Import from "{node.module}" is not allowed'}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.RESTRICTED_BUILTINS:
                    return {'valid': False, 'error': f'Use of "{node.func.id}" is not allowed'}
        if not any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree)):
            return {'valid': False, 'error': 'Code must define at least one function'}
        return {'valid': True}

    def _can_use_sigalrm_timeout(self) -> bool:
        """SIGALRM only works on the main thread; asyncio.to_thread runs in a worker pool."""
        return sys.platform != "win32" and threading.current_thread() is threading.main_thread()

    def _execute_with_timeout(self, code: str, inputs: Dict[str, Any]) -> Any:
        namespace = self._create_safe_namespace()

        @contextmanager
        def timeout_handler(seconds):
            def handler(signum, frame):
                raise TimeoutError(f'Execution exceeded {seconds} second timeout')
            if self._can_use_sigalrm_timeout():
                old = signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
            try:
                yield
            finally:
                if self._can_use_sigalrm_timeout():
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)

        with timeout_handler(self.timeout):
            exec(code, namespace)
        func_name = self._find_main_function(code, namespace)
        if func_name not in namespace:
            raise ValueError(f'Could not find function "{func_name}"')
        return namespace[func_name](**inputs)

    def _create_safe_namespace(self) -> Dict[str, Any]:
        builtins_dict = __builtins__.__dict__ if not isinstance(__builtins__, dict) else __builtins__
        safe_builtins = {
            n: builtins_dict[n] for n in builtins_dict
            if n not in self.RESTRICTED_BUILTINS and not n.startswith('_')
        }
        orig_import = builtins_dict['__import__']

        def safe_import(name, *args, **kwargs):
            if name.split('.')[0] in self.DISALLOWED_MODULES:
                raise ImportError(f"Import of '{name}' is not allowed")
            return orig_import(name, *args, **kwargs)

        safe_builtins['__import__'] = safe_import
        return {'__builtins__': safe_builtins, '__name__': '__main__', '__doc__': None}

    def _find_main_function(self, code: str, namespace: Dict[str, Any]) -> str:
        tree = ast.parse(code)
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if 'solve' in names:
            return 'solve'
        if 'main' in names:
            return 'main'
        if names:
            return names[0]
        raise ValueError('No functions found in code')
