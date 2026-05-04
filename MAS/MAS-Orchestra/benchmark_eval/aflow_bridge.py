"""
Resolve vendored AFlow-eval paths (``MAS-Orchestra/vendor/aflow_eval``) and load constants
that must match AFlow benchmarks exactly.
"""
from __future__ import annotations

import ast
import functools
import sys
from pathlib import Path


def get_aflow_root() -> Path:
    """Embedded AFlow eval snapshot: ``MAS-Orchestra/vendor/aflow_eval``."""
    return Path(__file__).resolve().parent.parent / "vendor" / "aflow_eval"


@functools.lru_cache(maxsize=1)
def load_agentless_repair() -> str:
    """
    Same suffix as AFlow SWEBenchmark model input: data['text'] + AGENTLESS_REPAIR.
    Parsed from vendored ``benchmarks/swe.py`` (AST) so we do not import swe.py.
    """
    root = get_aflow_root()
    path = root / "benchmarks" / "swe.py"
    if not path.is_file():
        raise FileNotFoundError(f"Expected vendored AFlow file: {path}")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "AGENTLESS_REPAIR":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
                if hasattr(ast, "Str") and isinstance(node.value, ast.Str):
                    return node.value.s
    raise ValueError(f"Could not find AGENTLESS_REPAIR string assignment in {path}")


def ensure_aflow_on_syspath() -> Path:
    """Insert vendored AFlow root on sys.path for ``from benchmarks.*`` / ``from scripts.*`` imports."""
    root = get_aflow_root()
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    return root
