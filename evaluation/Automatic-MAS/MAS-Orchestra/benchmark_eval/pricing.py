"""
USD per 1K tokens (input / output). Extend AFlow ModelPricing pattern.
Prices are approximate; update from provider pages when billing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Reference: OpenAI / Together public pricing pages (2025–2026); verify before publishing.
MODEL_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-5": {"input": 0.00125, "output": 0.01},  # placeholder; override with env if needed
    "gpt-5-nano": {"input": 0.00015, "output": 0.0006},
    "openai/gpt-oss-120b": {"input": 0.000, "output": 0.000},  # Together: adjust to live price
    "gpt-oss-120b": {"input": 0.000, "output": 0.000},
}


def get_price(model: str, token_type: str) -> float:
    if model in MODEL_PRICES_PER_1K:
        return MODEL_PRICES_PER_1K[model][token_type]
    for key, v in MODEL_PRICES_PER_1K.items():
        if key in model or model in key:
            return v[token_type]
    return 0.0


def cost_from_usage(model: str, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000.0) * get_price(model, "input") + (output_tokens / 1000.0) * get_price(
        model, "output"
    )


@dataclass
class UsageRollup:
    records: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        c = cost_from_usage(model, input_tokens, output_tokens)
        r = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": c,
        }
        self.records.append(r)
        return r

    def totals(self) -> Dict[str, Any]:
        inp = sum(x.get("input_tokens", 0) for x in self.records)
        out = sum(x.get("output_tokens", 0) for x in self.records)
        cost = sum(float(x.get("cost_usd", 0)) for x in self.records)
        return {
            "total_input_tokens": inp,
            "total_output_tokens": out,
            "total_tokens": inp + out,
            "total_cost_usd": cost,
            "num_api_calls": len(self.records),
        }


def load_usage_log(path: str) -> UsageRollup:
    import json

    rollup = UsageRollup()
    if not path or not __import__("os").path.isfile(path):
        return rollup
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = rec.get("model", "")
            rollup.add(m, int(rec.get("input_tokens", 0)), int(rec.get("output_tokens", 0)))
    return rollup
