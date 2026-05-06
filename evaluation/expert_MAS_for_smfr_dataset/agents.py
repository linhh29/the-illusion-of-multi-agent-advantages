"""
Agent classes for the manual MAS pipeline.

Three agents:
  MetaAgent      — parses problem text into structured parameters
  ExtractAgent   — generic information retrieval from text
  CalculateAgent — generic numerical computation

All agents share a single ModelClient that accepts a response_format per call.
Each call returns (response, CallUsage) so the pipeline can accumulate cost stats.
"""

import os
import re
import json
import asyncio
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
from pydantic import BaseModel
from openai import OpenAI
import openai

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

PROMPTS_DIR = Path(__file__).parent / "prompts"

# ---------------------------------------------------------------------------
# Model pricing  (USD per 1M tokens, input / output)
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":              {"input": 2.50,  "output": 10.00},
    "gpt-4o-2024-08-06":   {"input": 2.50,  "output": 10.00},
    "gpt-4.1":             {"input": 2.00,  "output":  8.00},
    "gpt-4.1-2025-04-14":  {"input": 2.00,  "output":  8.00},
    "o3":                  {"input": 10.00, "output": 40.00},
    "o3-2025-04-16":       {"input": 10.00, "output": 40.00},
    "o4-mini":             {"input": 1.10,  "output":  4.40},
    "o4-mini-2025-04-16":  {"input": 1.10,  "output":  4.40},
    "gpt-5":               {"input": 1.25,  "output": 10.00},
    "gpt-5-2025-08-07":    {"input": 1.25,  "output": 10.00},
    # Google Gemini
    "gemini-2.0-flash":               {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro":                 {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash":               {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite-preview-06-17": {"input": 0.10, "output": 0.40},
    "gemini-3-pro-preview":           {"input": 2.00, "output": 12.00},
    # Open-source models (via Together.ai-compatible gateway)
    "gpt-oss-120b":                   {"input": 0.15,  "output":  0.60},  # Together.ai pricing
    # Anthropic Claude
    "claude-haiku-4-5":               {"input": 0.80,  "output":  4.00},
    "claude-haiku-4-5-20251001":      {"input": 0.80,  "output":  4.00},
    "claude-sonnet-4-6":              {"input": 3.00,  "output": 15.00},
    "claude-opus-4-6":                {"input": 15.00, "output": 75.00},
}


def get_pricing(model_name: str) -> Optional[Dict[str, float]]:
    """Look up pricing by exact match, then by prefix (handles versioned IDs)."""
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    for key in MODEL_PRICING:
        if model_name.startswith(key):
            return MODEL_PRICING[key]
    return None


# ---------------------------------------------------------------------------
# Token / cost tracking
# ---------------------------------------------------------------------------

@dataclass
class CallUsage:
    """Token usage for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class PipelineStats:
    """Accumulated token usage and cost across all LLM calls in one pipeline run."""
    model_name: str = ""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # Per-phase breakdowns (phase name → CallUsage)
    phases: Dict[str, CallUsage] = field(default_factory=dict)

    def add(self, usage: CallUsage, phase: str = ""):
        """Record usage from one LLM call."""
        self.llm_calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        if phase:
            if phase not in self.phases:
                self.phases[phase] = CallUsage()
            self.phases[phase].input_tokens += usage.input_tokens
            self.phases[phase].output_tokens += usage.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self) -> Dict[str, float]:
        """Return {"input": $, "output": $, "total": $} for this run."""
        pricing = get_pricing(self.model_name)
        if pricing is None:
            return {"input": 0.0, "output": 0.0, "total": 0.0, "pricing_found": False}
        input_cost = self.input_tokens / 1_000_000 * pricing["input"]
        output_cost = self.output_tokens / 1_000_000 * pricing["output"]
        return {
            "input": round(input_cost, 6),
            "output": round(output_cost, 6),
            "total": round(input_cost + output_cost, 6),
            "pricing_found": True,
        }

    def to_dict(self) -> Dict:
        cost = self.cost()
        return {
            "model": self.model_name,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": cost,
            "phases": {
                phase: {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                }
                for phase, u in self.phases.items()
            },
        }


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class ProblemMetaResponse(BaseModel):
    investors: List[str]
    question_type: str        # "sell" or "buy"
    target_percentage: float  # e.g. 1.3
    aggregation: str          # "earliest" or "latest"
    price_type: str           # "Open" or "Close"


class ExtractResponse(BaseModel):
    result: str    # JSON-encoded value — parse with json.loads()
    reasoning: str


class CalculateResponse(BaseModel):
    result: str    # JSON-encoded value — parse with json.loads()
    reasoning: str


# ---------------------------------------------------------------------------
# Model client — abstract base + provider implementations
# ---------------------------------------------------------------------------

class ModelClient(ABC):
    """Abstract base for all model backends.

    generate() returns (parsed_response, CallUsage).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def generate(
        self, prompt: str, response_format: Type[BaseModel]
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        ...


_CUSTOM_BASE_URLS: Dict[str, str] = {
    # Add custom base URLs here if using a non-OpenAI-compatible gateway.
    # e.g. "gpt-oss-120b": "https://your-gateway/v1"
}
_DEFAULT_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
_DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Models whose gateway returns delta instead of message (streaming-style response)
_DELTA_RESPONSE_MODELS = {"gpt-oss-120b"}


class OpenAIModelClient(ModelClient):
    """Wraps the OpenAI API (or a compatible gateway) with structured-output parsing."""

    def __init__(self, model_name: str = "gpt-4.1-2025-04-14"):
        super().__init__(model_name)
        base_url = _CUSTOM_BASE_URLS.get(model_name, _DEFAULT_OPENAI_BASE_URL)
        self.client = OpenAI(
            base_url=base_url,
            api_key=_DEFAULT_OPENAI_API_KEY,
        )

    async def generate(
        self, prompt: str, response_format: Type[BaseModel]
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_generate, prompt, response_format
        )

    def _sync_generate(
        self, prompt: str, response_format: Type[BaseModel]
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        usage = CallUsage()
        try:
            if self.model_name in _DELTA_RESPONSE_MODELS:
                return self._sync_generate_delta(prompt, response_format, usage)
            return self._sync_generate_parsed(prompt, response_format, usage)
        except openai.APITimeoutError:
            return None, usage
        except Exception:
            traceback.print_exc()
            return None, usage

    def _sync_generate_parsed(
        self, prompt: str, response_format: Type[BaseModel], usage: CallUsage
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        """Standard path: beta.parse with structured output."""
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
            timeout=600,
        )
        if not any(x in self.model_name for x in ("o3", "o4", "gpt-5", "gpt-oss")):
            kwargs["temperature"] = 0.0

        completion = self.client.beta.chat.completions.parse(**kwargs)

        if completion.usage:
            usage.input_tokens = completion.usage.prompt_tokens or 0
            usage.output_tokens = completion.usage.completion_tokens or 0

        msg = completion.choices[0].message
        if msg.parsed:
            return response_format.model_validate(msg.parsed), usage
        return None, usage

    def _sync_generate_delta(
        self, prompt: str, response_format: Type[BaseModel], usage: CallUsage
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        """Fallback for gateways that return delta instead of message.

        Appends a JSON schema instruction and parses the raw text response.
        """
        schema = response_format.model_json_schema()
        fields = list(schema.get("properties", {}).keys())
        json_prompt = (
            prompt
            + f"\n\nRespond with a JSON object containing exactly these fields: {fields}. "
            "Output only valid JSON, no markdown fences."
        )
        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": json_prompt}],
                    timeout=1200,
                )
                break
            except openai.InternalServerError as e:
                if attempt == 2:
                    raise
                import time; time.sleep(10 * (attempt + 1))
        choice = completion.choices[0]
        raw = completion.model_dump()["choices"][0]
        text = (
            (choice.message.content if choice.message else None)
            or (raw.get("delta") or {}).get("content")
            or ""
        )
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
        try:
            data = json.loads(text)
            return response_format.model_validate(data), usage
        except Exception:
            return None, usage


class GeminiModelClient(ModelClient):
    """Wraps the Google GenAI SDK (Vertex AI) with JSON-schema constrained output."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        if not _GENAI_AVAILABLE:
            raise ImportError("google-genai package is required for Gemini models")
        super().__init__(model_name)
        gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.client = _genai.Client(
            vertexai=True,
            project=gcp_project,
            location="global",
        )

    async def generate(
        self, prompt: str, response_format: Type[BaseModel]
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_generate, prompt, response_format
        )

    def _sync_generate(
        self, prompt: str, response_format: Type[BaseModel]
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        usage = CallUsage()
        try:
            # Use thinking budget only for models that don't support it natively
            config_kwargs = dict(
                response_mime_type="application/json",
                response_schema=response_format,
            )
            if "none" in self.model_name:
                config_kwargs["thinking_config"] = _genai_types.ThinkingConfig(
                    thinking_budget=1024
                )
            else:
                config_kwargs["temperature"] = 0.0

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=_genai_types.GenerateContentConfig(**config_kwargs),
            )

            if response.usage_metadata:
                usage.input_tokens = response.usage_metadata.prompt_token_count or 0
                usage.output_tokens = response.usage_metadata.candidates_token_count or 0

            parsed = json.loads(response.text)
            return response_format.model_validate(parsed), usage

        except Exception:
            traceback.print_exc()
            return None, usage


def create_model_client(model_name: str) -> ModelClient:
    """Factory: route to the right backend based on model name prefix."""
    if model_name.startswith("gemini"):
        return GeminiModelClient(model_name)
    # Default: OpenAI-compatible (GPT, o-series, etc.)
    return OpenAIModelClient(model_name)


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------

class MetaAgent:
    """Parses raw problem text into structured parameters."""

    def __init__(self, model_client: ModelClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "meta_agent.txt").read_text()

    async def parse(
        self, problem_text: str
    ) -> Tuple[Optional[ProblemMetaResponse], CallUsage]:
        prompt = self._template.replace("{problem_text}", problem_text)
        return await self.model.generate(prompt, ProblemMetaResponse)


class ExtractAgent:
    """Generic information retrieval: extract(context, query) → (result, usage)."""

    def __init__(self, model_client: ModelClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "extract.txt").read_text()

    async def extract(
        self, context: str, query: str
    ) -> Tuple[Optional[ExtractResponse], CallUsage]:
        prompt = self._template.replace("{context}", context).replace("{query}", query)
        return await self.model.generate(prompt, ExtractResponse)


class CalculateAgent:
    """Generic numerical computation: calculate(data, query) → (result, usage)."""

    def __init__(self, model_client: ModelClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "calculate.txt").read_text()

    async def calculate(
        self, data: str, query: str
    ) -> Tuple[Optional[CalculateResponse], CallUsage]:
        prompt = self._template.replace("{data}", data).replace("{query}", query)
        return await self.model.generate(prompt, CalculateResponse)
