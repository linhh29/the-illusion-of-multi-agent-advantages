"""
Read sub-agent sampling policy only from ``mas_r1_reasoner/configs/grpo_trainer.yaml``:

- ``azr.mas_r1.agent.sampler_defaults`` — per-``type`` defaults (temperature, max_tokens, …)
- ``azr.mas_r1.agent.model_sampler_map`` — per-model overrides

No hardcoded temperature/max token values here; add new models (e.g. Gemini) entirely in YAML.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

_GRPO_YAML = Path(__file__).resolve().parent.parent.parent / "configs" / "grpo_trainer.yaml"


@lru_cache(maxsize=1)
def _agent_block() -> Dict[str, Any]:
    if not _GRPO_YAML.is_file():
        return {}
    cfg = OmegaConf.load(_GRPO_YAML)
    try:
        agent = cfg.azr.mas_r1.agent
    except (AttributeError, KeyError):
        return {}
    out = OmegaConf.to_container(agent, resolve=True)
    return out if isinstance(out, dict) else {}


def clear_grpo_sampler_config_cache() -> None:
    """Call after editing ``grpo_trainer.yaml`` in a long-lived process."""
    _agent_block.cache_clear()


def _model_sampler_map() -> Dict[str, Any]:
    m = _agent_block().get("model_sampler_map")
    return m if isinstance(m, dict) else {}


def _sampler_defaults() -> Dict[str, Any]:
    d = _agent_block().get("sampler_defaults")
    return d if isinstance(d, dict) else {}


def get_sampler_defaults(sampler_type: str) -> Dict[str, Any]:
    """
    Defaults for a sampler ``type`` (e.g. ``ChatCompletionSampler``) from YAML ``sampler_defaults``.
    Raises if the type is missing so misconfiguration fails fast.
    """
    d = _sampler_defaults().get(sampler_type)
    if not isinstance(d, dict) or not d:
        raise KeyError(
            f"sampler_defaults.{sampler_type} missing or empty in {_GRPO_YAML}. "
            "Define it under azr.mas_r1.agent.sampler_defaults in grpo_trainer.yaml."
        )
    return copy.deepcopy(d)


def merge_sampler_entry(sampler_type: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Merge per-model ``entry`` over ``sampler_defaults[sampler_type]`` (entry wins)."""
    base = get_sampler_defaults(sampler_type)
    for k, v in entry.items():
        if v is None and k != "model":
            continue
        base[k] = v
    return base


def build_minimal_sampler_config(
    *,
    model_id: str,
    sampler_type: str,
) -> Dict[str, Any]:
    """Used when injecting a dynamic benchmark entry: defaults + ``type`` + ``model``."""
    return merge_sampler_entry(sampler_type, {"type": sampler_type, "model": model_id})


def subagent_use_together_completion(agent_model: str) -> bool:
    """Whether an unknown ``--agent-model`` should use TogetherCompletionSampler (from YAML lists only)."""
    m = (agent_model or "").strip()
    agent = _agent_block()
    for p in agent.get("subagent_together_model_prefixes") or []:
        if m.startswith(str(p)):
            return True
    names = agent.get("subagent_together_model_names") or []
    if isinstance(names, (list, tuple)):
        return m in set(names)
    return False


def resolve_sampler_map_entry(agent_model: str) -> Optional[Dict[str, Any]]:
    """Resolve ``model_sampler_map`` entry, including optional ``model_sampler_lookup_aliases``."""
    m = (agent_model or "").strip()
    if not m:
        return None
    msm = _model_sampler_map()
    if not msm:
        return None
    if m in msm:
        e = msm[m]
        return e if isinstance(e, dict) else None
    aliases = _agent_block().get("model_sampler_lookup_aliases") or {}
    if isinstance(aliases, dict) and m in aliases:
        canon = aliases[m]
        if isinstance(canon, str) and canon in msm:
            e = msm[canon]
            return e if isinstance(e, dict) else None
    return None


def resolve_api_model_id(agent_model: str, entry: Optional[Dict[str, Any]] = None) -> str:
    """API ``model`` id: always from the map entry when present, else the requested id."""
    if entry and entry.get("model") is not None:
        return str(entry["model"])
    return (agent_model or "").strip()


def openai_chat_completion_create_kwargs(agent_model: str) -> Dict[str, Any]:
    """
    Kwargs for ``AsyncOpenAI.chat.completions.create`` besides ``messages`` / ``timeout``.
    Requires a ``model_sampler_map`` entry (or alias); numeric fields come from defaults + overrides in YAML only.
    """
    raw = resolve_sampler_map_entry(agent_model)
    if not raw:
        raise KeyError(
            f"No model_sampler_map entry for {agent_model!r}. "
            f"Add it under azr.mas_r1.agent.model_sampler_map in {_GRPO_YAML} "
            "(optionally use model_sampler_lookup_aliases)."
        )
    sampler_type = str(raw.get("type", "ChatCompletionSampler"))
    merged = merge_sampler_entry(sampler_type, raw)
    api_model = resolve_api_model_id(agent_model, merged)

    if sampler_type == "TogetherCompletionSampler":
        out: Dict[str, Any] = {
            "model": api_model,
            "max_tokens": int(merged["max_tokens"]),
            "reasoning_effort": str(merged["reasoning_effort"]),
        }
        if not merged.get("omit_temperature"):
            out["temperature"] = float(merged["temperature"])
        return out

    if sampler_type == "ChatCompletionSampler":
        out = {"model": api_model}
        mct = merged.get("max_completion_tokens")
        mt = merged.get("max_tokens")
        omit = bool(merged.get("omit_temperature"))
        if omit:
            if mct is not None:
                out["max_completion_tokens"] = int(mct)
            elif mt is not None:
                out["max_tokens"] = int(mt)
            else:
                raise KeyError(
                    "ChatCompletionSampler with omit_temperature needs max_completion_tokens or max_tokens "
                    f"in merged config for {agent_model!r} (YAML defaults + model_sampler_map)."
                )
            return out
        out["temperature"] = float(merged["temperature"])
        if mct is not None:
            out["max_completion_tokens"] = int(mct)
        elif mt is not None:
            out["max_tokens"] = int(mt)
        else:
            raise KeyError(
                f"Merged ChatCompletionSampler config for {agent_model!r} missing max_tokens / max_completion_tokens."
            )
        return out

    raise ValueError(
        f"Unsupported sampler type {sampler_type!r} for openai_chat_completion_create_kwargs "
        f"({agent_model!r}). Add handling or use ChatCompletionSampler / TogetherCompletionSampler."
    )


def chat_completion_sampler_init_kwargs_from_merged(
    merged: Dict[str, Any], *, mock_output: bool
) -> Dict[str, Any]:
    """Build kwargs for ``ChatCompletionSampler(**...)`` from a merged map entry."""
    return {
        "model": merged.get("model"),
        "system_message": merged.get("system_message"),
        "temperature": merged.get("temperature"),
        "mock_output": mock_output,
        "max_tokens": merged.get("max_tokens"),
        "max_completion_tokens": merged.get("max_completion_tokens"),
        "omit_temperature": bool(merged.get("omit_temperature")),
    }


def together_completion_sampler_init_kwargs_from_merged(
    merged: Dict[str, Any], *, mock_output: bool
) -> Dict[str, Any]:
    """Build kwargs for ``TogetherCompletionSampler(**...)`` from a merged map entry."""
    return {
        "model": merged.get("model"),
        "system_message": merged.get("system_message"),
        "temperature": merged.get("temperature"),
        "mock_output": mock_output,
        "max_tokens": merged.get("max_tokens"),
        "reasoning_effort": merged.get("reasoning_effort"),
        "omit_temperature": bool(merged.get("omit_temperature")),
    }


# Backwards-compatible name
clear_model_sampler_map_cache = clear_grpo_sampler_config_cache
