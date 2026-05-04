"""
STOCKS: second-pass structured completion aligned with AFlow ``call_reverse_answer_code``.

After the Harmony MAS graph returns a draft (often portfolio JSON without ``analysis`` / ``code``),
call the same agent model with ``response_format`` **json_schema** (Pydantic ``ReverseAnswerCodeResponse``) so ``prediction`` under ``results_*`` includes
``analysis``, ``answer`` (string), and ``code`` for ``StocksBenchmark._parse_model_output``.

Disable with env ``MAS_STOCKS_STRUCTURED_OUTPUT=0`` (or ``false`` / ``no``).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, Optional

from mas_r1_reasoner.agents.sampler.grpo_model_sampler_params import (
    openai_chat_completion_create_kwargs,
    resolve_sampler_map_entry,
    subagent_use_together_completion,
)

# All sub-agent models: primary path is json_schema; if the API rejects it (e.g. some gateways), retry with strict JSON prompt only.

_STRICT_JSON_TAIL = (
    "\n\nRespond with valid JSON only, exactly one object with keys "
    '"analysis", "answer", "code" (no markdown fences).'
)


def _structured_output_disabled() -> bool:
    raw = (os.environ.get("MAS_STOCKS_STRUCTURED_OUTPUT") or "1").strip().lower()
    return raw in ("0", "false", "no", "off")


def _use_together_client(model: str) -> bool:
    """Together HTTP client vs OpenAI: driven by ``model_sampler_map`` ``type`` and YAML ``subagent_together_*`` lists."""
    e = resolve_sampler_map_entry(model)
    if e and str(e.get("type")) == "TogetherCompletionSampler":
        return True
    return subagent_use_together_completion(model)


def _openai_client_kwargs(model: str) -> dict:
    from mas_r1_reasoner.agents.sampler.api_usage_tracker import (
        build_openai_client_kwargs,
        build_together_openai_client_kwargs,
    )

    if _use_together_client(model):
        return build_together_openai_client_kwargs()
    return build_openai_client_kwargs()


def _parse_response_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse model output: raw JSON, fenced ```json```, or outermost {...} slice."""
    s = (text or "").strip()
    if not s:
        return None
    try:
        o = json.loads(s)
        if isinstance(o, dict):
            return o
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if m:
        try:
            o = json.loads(m.group(1).strip())
            if isinstance(o, dict):
                return o
        except json.JSONDecodeError:
            pass
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j > i:
        try:
            o = json.loads(s[i : j + 1])
            if isinstance(o, dict):
                return o
        except json.JSONDecodeError:
            pass
    return None


def _merge_draft_portfolio(structured: Dict[str, Any], mas_draft: str) -> Dict[str, Any]:
    """Attach non-conflicting portfolio keys from the MAS draft for inspection."""
    out = dict(structured)
    try:
        draft = json.loads(mas_draft.strip())
    except Exception:
        return out
    if not isinstance(draft, dict):
        return out
    if "investor_dates" in draft:
        out["investor_dates"] = draft["investor_dates"]
    if "comparison" in draft:
        out["comparison"] = draft["comparison"]
    ans = draft.get("answer")
    if isinstance(ans, list):
        out["portfolio_answer"] = ans
    return out


async def _call_structured_async(
    *,
    full_prompt: str,
    mas_draft: str,
    model: str,
) -> Optional[str]:
    from openai import AsyncOpenAI
    from pydantic import BaseModel, Field

    from mas_r1_reasoner.agents.sampler.api_usage_tracker import (
        get_api_semaphore,
        record_completion,
    )
    from mas_r1_reasoner.agents.sampler.http_timeout_scale import (
        scaled_subagent_read_timeout_sec,
    )

    class ReverseAnswerCodeResponse(BaseModel):
        analysis: str = Field(..., description="Step by Step reasoning")
        answer: str = Field(..., description="Final answer, ONLY NAME")
        code: str = Field(..., description="Code with solve() function and all required input data")

    schema = ReverseAnswerCodeResponse.model_json_schema()

    user_body = (
        f"{full_prompt}\n\n---\n\n"
        "Draft output from the multi-agent run (may be incomplete or only portfolio JSON). "
        "Refine it into the required structured response.\n\n"
        f"Draft:\n{mas_draft}\n\n"
        'Your response must be one JSON object with keys "analysis" (string), '
        '"answer" (string: winning investor name only, or comma-separated names if tied), '
        'and "code" (string: full Python with def solve() returning the portfolio dict per the task).'
    )

    user_body_strict = user_body + _STRICT_JSON_TAIL
    together = _use_together_client(model)
    kwargs = _openai_client_kwargs(model)
    client = AsyncOpenAI(**kwargs)

    base_t = float(kwargs.get("timeout") or (300.0 if together else 120.0))
    _htt = scaled_subagent_read_timeout_sec(base_t, model)

    completion_base = openai_chat_completion_create_kwargs(model)
    _prov = "together" if together else "openai"

    async with get_api_semaphore():
        create_kwargs: Dict[str, Any] = {
            **completion_base,
            "messages": [{"role": "user", "content": user_body}],
            "timeout": _htt,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ReverseAnswerCodeResponse",
                    "schema": schema,
                },
            },
        }
        try:
            response = await client.chat.completions.create(**create_kwargs)
        except Exception:
            create_kwargs.pop("response_format", None)
            create_kwargs["messages"] = [{"role": "user", "content": user_body_strict}]
            response = await client.chat.completions.create(**create_kwargs)

    usage = getattr(response, "usage", None)
    if usage is not None:
        pt = getattr(usage, "prompt_tokens", None) or 0
        ct = getattr(usage, "completion_tokens", None) or 0
        record_completion(str(completion_base.get("model", model)), int(pt), int(ct), provider=_prov)

    raw = response.choices[0].message.content
    if not raw or not str(raw).strip():
        return None
    text = str(raw).strip()
    obj = _parse_response_object(text)
    if not isinstance(obj, dict):
        return None
    # Minimal validation
    for k in ("analysis", "answer", "code"):
        if k not in obj:
            return None
    merged = _merge_draft_portfolio(obj, mas_draft)
    return json.dumps(merged, ensure_ascii=False, indent=4)


def maybe_augment_stocks_prediction(
    pred_text: str,
    *,
    question: str,
    agent_model: str,
) -> str:
    """
    If enabled, returns JSON string with ``analysis`` / ``answer`` / ``code`` (+ optional portfolio keys).
    On failure or when disabled, returns ``pred_text`` unchanged.
    """
    if _structured_output_disabled():
        return pred_text
    if not (pred_text or "").strip():
        return pred_text
    try:
        out = asyncio.run(
            _call_structured_async(
                full_prompt=question,
                mas_draft=pred_text,
                model=agent_model,
            )
        )
        if out:
            return out
    except Exception as e:
        print(
            f"benchmark_eval.stocks_structured_output: keeping MAS draft ({e})",
            file=sys.stderr,
            flush=True,
        )
    return pred_text
