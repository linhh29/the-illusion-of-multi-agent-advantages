"""
Global async concurrency limit and per-call token usage recording for LLM samplers.
Used by benchmark_eval and any run that sets MAS_API_MAX_CONCURRENCY (default 50).
"""
from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

# Max concurrent sub-agent API requests (OpenAI / Together-compatible).
# MAS_API_MAX_CONCURRENCY takes precedence; MAX_CONCURRENT is an alias (e.g. run_gpqa.sh).
def _read_default_concurrency() -> int:
    for key in ("MAS_API_MAX_CONCURRENCY", "MAX_CONCURRENT"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            try:
                return int(str(raw).strip())
            except ValueError:
                pass
    return 50


_api_semaphore: asyncio.Semaphore | None = None
# ``asyncio.Semaphore`` is bound to the loop that was running at creation time. Benchmark / eval often
# call ``asyncio.run()`` once per sample (or use worker threads), each with a new loop — re-use would
# raise "bound to a different event loop" or behave incorrectly.
_semaphore_loop_id: int | None = None


def reset_api_semaphore() -> None:
    """Call after changing ``MAS_API_MAX_CONCURRENCY`` / ``MAX_CONCURRENT`` so the limit is re-read."""
    global _api_semaphore, _semaphore_loop_id
    _api_semaphore = None
    _semaphore_loop_id = None


def get_api_semaphore() -> asyncio.Semaphore:
    global _api_semaphore, _semaphore_loop_id
    try:
        loop = asyncio.get_running_loop()
        lid = id(loop)
    except RuntimeError:
        lid = None
    if _api_semaphore is not None and lid == _semaphore_loop_id:
        return _api_semaphore
    # New loop (or first use): create a semaphore for this loop; read env at creation time.
    _api_semaphore = asyncio.Semaphore(_read_default_concurrency())
    _semaphore_loop_id = lid
    return _api_semaphore


# List of dicts appended per completion: model, input_tokens, output_tokens, provider
_usage_records: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "mas_sampler_usage_records", default=None
)


def set_usage_buffer(buf: Optional[List[Dict[str, Any]]]) -> object:
    """Returns a token for reset()."""
    return _usage_records.set(buf)


def reset_usage_buffer(token: object) -> None:
    _usage_records.reset(token)


def get_usage_buffer() -> Optional[List[Dict[str, Any]]]:
    return _usage_records.get()


# Per-task log path (asyncio + asyncio.to_thread): avoids ``os.environ`` races when multiple samples run in parallel.
_usage_log_path_ctx: ContextVar[Optional[str]] = ContextVar("mas_api_usage_log_path", default=None)


def effective_usage_log_path() -> str:
    ctx = _usage_log_path_ctx.get()
    if ctx is not None and str(ctx).strip():
        return str(ctx).strip()
    return os.environ.get("MAS_API_USAGE_LOG", "").strip()


def set_usage_log_path_context(path: Optional[str]) -> object:
    """Prefer over mutating ``MAS_API_USAGE_LOG`` when running concurrent eval tasks."""
    return _usage_log_path_ctx.set(path)


def reset_usage_log_path_context(token: object) -> None:
    _usage_log_path_ctx.reset(token)


def record_completion(
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "openai",
) -> None:
    rec = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "provider": provider,
    }
    buf = get_usage_buffer()
    if buf is not None:
        buf.append(rec)
    log_path = effective_usage_log_path()
    if log_path:
        import json

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)


def build_openai_client_kwargs() -> dict:
    """AsyncOpenAI kwargs from environment (public OpenAI API by default)."""
    try:
        _t = float(os.environ.get("MAS_OPENAI_CHAT_TIMEOUT_SEC", "120"))
    except ValueError:
        _t = 120.0
    _t = max(30.0, _t)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        # Backward compatibility: legacy API gateway
        if os.environ.get("X_API_KEY"):
            return {
                "api_key": "dummy",
                "base_url": "https://api.openai.com/v1",
                "default_headers": {"X-Api-Key": os.environ.get("X_API_KEY", "")},
                "timeout": _t,
            }
    # Default HTTP timeout for OpenAI-compatible calls; BrowseComp+ / long prompts: set MAS_OPENAI_CHAT_TIMEOUT_SEC=300–600.
    return {"api_key": api_key or "dummy", "base_url": base_url.rstrip("/"), "timeout": _t}


def build_together_openai_client_kwargs() -> dict:
    """Together AI via OpenAI-compatible HTTP API."""
    try:
        _t = float(os.environ.get("MAS_TOGETHER_CHAT_TIMEOUT_SEC", "300"))
    except ValueError:
        _t = 300.0
    _t = max(30.0, _t)
    api_key = (
        os.environ.get("TOGETHER_API_KEY", "").strip()
        or os.environ.get("OPENAI_TOGETHER_KEY", "").strip()
    )
    base_url = (
        os.environ.get("TOGETHER_BASE_URL", "").strip()
        or os.environ.get("OPENAI_TOGETHER_API_BASE", "").strip()
        or "https://api.together.xyz/v1"
    )
    return {"api_key": api_key or "dummy", "base_url": base_url.rstrip("/"), "timeout": _t}
