"""
Agents for the Simple MAS pipeline.

Imports shared infrastructure from expert_MAS_for_smfr_dataset to avoid duplication.
Adds SimpleMASClient (duck-type compatible with ModelClient, adds generate_text()),
RetrievalAgent, and CalculationAgent — both making free-form JSON calls with no
Pydantic schema enforcement.
"""

import os
import re
import json
import asyncio
import traceback
import sys
import importlib.util
from pathlib import Path
from typing import Optional, Tuple, Type

# Public entry points read configuration from this repository, not a sibling
# workspace directory. Already running jobs keep their in-memory configuration.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)

# Load expert_MAS agents module directly by file path to avoid circular import
# (this file is also named agents.py)
_EXPERT_MAS_DIR = Path(__file__).parent.parent / "expert_MAS_for_smfr_dataset"
_spec = importlib.util.spec_from_file_location(
    "expert_mas_agents", _EXPERT_MAS_DIR / "agents.py"
)
_expert_agents = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_expert_agents)

ModelClient = _expert_agents.ModelClient
MetaAgent = _expert_agents.MetaAgent
PipelineStats = _expert_agents.PipelineStats
CallUsage = _expert_agents.CallUsage
ProblemMetaResponse = _expert_agents.ProblemMetaResponse
create_model_client = _expert_agents.create_model_client
get_pricing = _expert_agents.get_pricing
_DELTA_RESPONSE_MODELS = _expert_agents._DELTA_RESPONSE_MODELS

from openai import OpenAI
import openai
from pydantic import BaseModel

_NO_TEMP_MODELS = ("o3", "o4", "gpt-5", "gpt-oss")

PROMPTS_DIR = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# SimpleMASClient
# Duck-type compatible with ModelClient so MetaAgent can be used unchanged.
# Also provides generate_text() for raw (no-schema) calls used by the two
# new agents.
# ---------------------------------------------------------------------------

def _make_openai_client(model_name: str) -> OpenAI:
    """Build a client for an explicitly configured public or research endpoint."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        return OpenAI(api_key=api_key, base_url=os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1')
    research_key = os.environ.get('X_API_KEY')
    research_url = os.environ.get('RESEARCH_BASE_URL')
    if research_key and research_url:
        return OpenAI(api_key='dummy', base_url=research_url, default_headers={'X-Api-Key': research_key})
    raise EnvironmentError('Set OPENAI_API_KEY, or X_API_KEY with RESEARCH_BASE_URL, in repo .env')


class SimpleMASClient:
    """
    Model client for Simple MAS using the configured API endpoint.

    - generate(prompt, response_format): structured call via beta.parse
      (used by MetaAgent which requires Pydantic output).
    - generate_text(prompt): plain chat.completions.create(), no schema
      (used by RetrievalAgent and CalculationAgent).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._openai = _make_openai_client(model_name)

    async def generate(
        self, prompt: str, response_format: Type[BaseModel]
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        """Structured call — used by MetaAgent."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_generate_structured, prompt, response_format
        )

    def _sync_generate_structured(
        self, prompt: str, response_format: Type[BaseModel]
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        usage = CallUsage()
        try:
            if self.model_name in _DELTA_RESPONSE_MODELS:
                # Delta gateway: append schema instructions, parse raw text
                schema = response_format.model_json_schema()
                fields = list(schema.get("properties", {}).keys())
                json_prompt = (
                    prompt
                    + f"\n\nRespond with a JSON object containing exactly these fields: {fields}. "
                    "Output only valid JSON, no markdown fences."
                )
                for attempt in range(6):
                    try:
                        completion = self._openai.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": json_prompt}],
                            timeout=1200,
                        )
                        break
                    except (openai.InternalServerError, openai.RateLimitError, openai.AuthenticationError) as e:
                        if attempt == 5:
                            raise
                        import time; time.sleep(30 * (attempt + 1))
                choice = completion.choices[0]
                raw = completion.model_dump()["choices"][0]
                text = (
                    (choice.message.content if choice.message else None)
                    or (raw.get("delta") or {}).get("content") or ""
                )
                text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
                text = re.sub(r"\s*```$", "", text.strip())
                if completion.usage:
                    usage.input_tokens = completion.usage.prompt_tokens or 0
                    usage.output_tokens = completion.usage.completion_tokens or 0
                data = json.loads(text)
                return response_format.model_validate(data), usage
            else:
                kwargs = dict(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_format,
                    timeout=600,
                )
                if not any(x in self.model_name for x in _NO_TEMP_MODELS):
                    kwargs["temperature"] = 0.0
                completion = self._openai.beta.chat.completions.parse(**kwargs)
                if completion.usage:
                    usage.input_tokens = completion.usage.prompt_tokens or 0
                    usage.output_tokens = completion.usage.completion_tokens or 0
                msg = completion.choices[0].message
                if msg.parsed:
                    return response_format.model_validate(msg.parsed), usage
                return None, usage
        except openai.APITimeoutError:
            return None, usage
        except Exception:
            traceback.print_exc()
            return None, usage

    async def generate_text(self, prompt: str) -> Tuple[Optional[str], CallUsage]:
        """Raw text call — no schema. Returns (text | None, CallUsage)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_text, prompt)

    def _sync_text(self, prompt: str) -> Tuple[Optional[str], CallUsage]:
        usage = CallUsage()
        try:
            if self.model_name in _DELTA_RESPONSE_MODELS:
                return self._sync_text_delta(prompt, usage)
            return self._sync_text_standard(prompt, usage)
        except openai.APITimeoutError:
            return None, usage
        except Exception:
            traceback.print_exc()
            return None, usage

    def _sync_text_standard(self, prompt: str, usage: CallUsage) -> Tuple[Optional[str], CallUsage]:
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=600,
        )
        if not any(x in self.model_name for x in _NO_TEMP_MODELS):
            kwargs["temperature"] = 0.0
        completion = self._openai.chat.completions.create(**kwargs)
        if completion.usage:
            usage.input_tokens = completion.usage.prompt_tokens or 0
            usage.output_tokens = completion.usage.completion_tokens or 0
        text = completion.choices[0].message.content or ""
        return text, usage

    def _sync_text_delta(self, prompt: str, usage: CallUsage) -> Tuple[Optional[str], CallUsage]:
        """Fallback for gateways returning delta instead of message (e.g. gpt-oss-120b)."""
        for attempt in range(6):
            try:
                completion = self._openai.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=1200,
                )
                break
            except (openai.InternalServerError, openai.RateLimitError, openai.AuthenticationError) as e:
                if attempt == 5:
                    raise
                import time
                time.sleep(30 * (attempt + 1))
        choice = completion.choices[0]
        raw = completion.model_dump()["choices"][0]
        text = (
            (choice.message.content if choice.message else None)
            or (raw.get("delta") or {}).get("content") or ""
        )
        if completion.usage:
            usage.input_tokens = completion.usage.prompt_tokens or 0
            usage.output_tokens = completion.usage.completion_tokens or 0
        return text, usage


def create_simple_mas_client(model_name: str) -> SimpleMASClient:
    return SimpleMASClient(model_name)


# ---------------------------------------------------------------------------
# RetrievalAgent
# Extracts all investors' transactions with transaction-date prices in one call.
# ---------------------------------------------------------------------------

class RetrievalAgent:
    """
    One LLM call: given the full problem text and investor names, extract every
    transaction for every investor along with the exact transaction-date price.
    Returns raw text (the LLM's response) and CallUsage.
    No Pydantic enforcement — output is parsed from free-form JSON.
    """

    def __init__(self, model_client: SimpleMASClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "retrieval.txt").read_text()

    async def extract(
        self,
        problem_text: str,
        investors: list,
        price_type: str,
    ) -> Tuple[Optional[str], CallUsage]:
        investors_str = ", ".join(investors)
        prompt = (
            self._template
            .replace("{problem_text}", problem_text)
            .replace("{investors}", investors_str)
            .replace("{price_type}", price_type)
        )
        return await self.model.generate_text(prompt)


# ---------------------------------------------------------------------------
# CalculationAgent
# Given extracted transaction data, identifies open positions, computes P&L,
# finds required prices, scans valid dates, and picks the winner.
# All reasoning is done in-context — no Python scaffolding.
# ---------------------------------------------------------------------------

class CalculationAgent:
    """
    One LLM call: receives extracted transaction data + full problem text (for
    price table scanning) + meta parameters. Returns the final answer dict as
    free-form JSON: {"investor_dates": {...}, "answer": [...]}.
    """

    def __init__(self, model_client: SimpleMASClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "calculation.txt").read_text()

    async def calculate(
        self,
        problem_text: str,
        extracted_data: str,
        meta: ProblemMetaResponse,
    ) -> Tuple[Optional[str], CallUsage]:
        prompt = (
            self._template
            .replace("{problem_text}", problem_text)
            .replace("{extracted_data}", extracted_data)
            .replace("{question_type}", meta.question_type)
            .replace("{target_pct}", str(meta.target_percentage))
            .replace("{price_type}", meta.price_type)
            .replace("{aggregation}", meta.aggregation)
        )
        return await self.model.generate_text(prompt)
