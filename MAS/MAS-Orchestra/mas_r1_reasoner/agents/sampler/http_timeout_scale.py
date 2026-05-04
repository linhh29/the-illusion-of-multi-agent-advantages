"""
Scale sub-agent HTTP read timeouts for slower models relative to the gpt-4o baseline.

Baseline seconds come from the same env vars as always (e.g. ``MAS_OPENAI_CHAT_TIMEOUT_SEC`` /
``MAS_TOGETHER_CHAT_TIMEOUT_SEC``). For ``gpt-5*`` and ``gpt-oss*`` models, the effective timeout is
**3×** that baseline so runs match ``run_gpt4o.sh`` budgets without per-script duplication.
"""
from __future__ import annotations

from typing import Optional


def scaled_subagent_read_timeout_sec(base_seconds: float, model: Optional[str]) -> float:
    """
    Return ``base_seconds`` for gpt-4o-style models; **3×** for ``gpt-5*`` and ``gpt-oss*``.

    ``base_seconds`` should already be the configured baseline (e.g. from env), not a hardcoded guess.
    """
    try:
        base = float(base_seconds)
    except (TypeError, ValueError):
        base = 120.0
    if model is None or not str(model).strip():
        return max(30.0, base)
    ml = str(model).strip().lower()
    if ml.startswith("gpt-5") or "gpt-oss" in ml:
        return max(30.0, base * 3.0)
    return max(30.0, base)
