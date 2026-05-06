"""Same XML extraction as ``AFlow/benchmarks/swe.extract_xml`` (no full ``swe`` import)."""
from __future__ import annotations

import re


def extract_xml(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""
