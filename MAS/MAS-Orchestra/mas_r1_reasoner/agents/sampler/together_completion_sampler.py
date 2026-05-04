import base64
import os
import time
import json
import asyncio
from typing import Any

import openai
from openai import APITimeoutError, AsyncOpenAI

from dataclasses import dataclass, field
from mas_r1_reasoner.agents.sampler.chat_common import SamplerBase, EvalResult, SingleEvalResult, Eval
from mas_r1_reasoner.agents.sampler.api_usage_tracker import (
    get_api_semaphore,
    record_completion,
    build_together_openai_client_kwargs,
)
from mas_r1_reasoner.agents.sampler.http_timeout_scale import scaled_subagent_read_timeout_sec
from mas_r1_reasoner.agents.sampler.grpo_model_sampler_params import get_sampler_defaults

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


def _together_http_timeout_sec(model: str | None = None) -> float:
    """Per-request read timeout; env ``MAS_TOGETHER_CHAT_TIMEOUT_SEC`` (default 300)."""
    try:
        t = float(os.environ.get("MAS_TOGETHER_CHAT_TIMEOUT_SEC", "300"))
    except ValueError:
        t = 300.0
    return scaled_subagent_read_timeout_sec(t, model)


class TogetherCompletionSampler(SamplerBase):
    """
    Sample from Together AI's chat completion API
    
    Args:
        model: The model to use for completion
        system_message: Optional system message to prepend
        temperature: Sampling temperature (0.0 to 2.0)
        mock_output: If True, returns mock responses instead of calling the API
    """

    def __init__(
        self,
        model: str | None = None,
        system_message: str | None = None,
        temperature: float | None = None,
        mock_output: bool = False,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        omit_temperature: bool | None = None,
    ):
        self.api_key_name = "TOGETHER_API_KEY"
        # OpenAI-compatible Together HTTP API (same client as OpenAI SDK)
        self.client = AsyncOpenAI(**build_together_openai_client_kwargs())
        # Note: Concurrency is now controlled at the RayAgentWorker level
        _d = get_sampler_defaults("TogetherCompletionSampler")

        # Convert OmegaConf objects to basic Python types (these are loaded fron config, so we need to convert them to basic types)
        self.model = self._convert_to_basic_type(model)
        self.system_message = self._convert_to_basic_type(system_message)
        _temp = self._convert_to_basic_type(temperature if temperature is not None else _d["temperature"])
        self.temperature = float(_temp)
        _mt = self._convert_to_basic_type(max_tokens if max_tokens is not None else _d["max_tokens"])
        self.max_tokens = int(_mt)
        _re = reasoning_effort if reasoning_effort is not None else _d["reasoning_effort"]
        self.reasoning_effort = str(self._convert_to_basic_type(_re))
        _omit = omit_temperature if omit_temperature is not None else _d.get("omit_temperature")
        self.omit_temperature = bool(self._convert_to_basic_type(_omit))
        self.mock_output = mock_output

    def _convert_to_basic_type(self, value: Any) -> Any:
        """Convert OmegaConf objects to basic Python types."""
        if value is None:
            return None
        
        # If it's already a basic type, return as-is
        if isinstance(value, (str, int, float, bool)):
            return value
        
        # If it's an OmegaConf object, convert to string
        
        try:
            from omegaconf import OmegaConf
            if hasattr(value, '__class__') and 'omegaconf' in str(value.__class__).lower():
                return str(value)
        except ImportError:
            pass
        
        # For other types, convert to string
        return str(value)

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


    async def __call__(self, message_list: MessageList, temperature=None, output_fields=None) -> str:
            
        # print(f"\n=== ChatCompletionSampler.__call__ Debug ===")
        # print(f"Model: {self.model}")
        # print(f"Temperature: {temperature if temperature is not None else self.temperature}")
        # print(f"System message: {self.system_message}")
        # print(f"Input message count: {len(message_list)}")
        
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
            print(f"Added system message, total messages: {len(message_list)}")
        
        trial = 0
        while True:
            print(f"\n--- Async Trial {trial + 1} ---")
            try:
                # Convert non-string content to strings
                # print("Converting message content to strings...")
                for message_id, message in enumerate(message_list):
                    if type(message['content']) != str:
                        original_type = type(message['content']).__name__
                        # print(f"  Converting message {message_id} content from {original_type} to string...")
                        message_list[message_id]['content'] = str(message['content'])
            
                # Ensure all parameters are JSON serializable
                safe_model = self._convert_to_basic_type(self.model)
                safe_temperature = self._convert_to_basic_type(temperature if temperature is not None else self.temperature)
                
                # Ensure temperature is a float
                if safe_temperature is not None:
                    try:
                        safe_temperature = float(safe_temperature)
                        # print(f"✓ Temperature converted to float: {safe_temperature} (type: {type(safe_temperature).__name__})")
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Failed to convert temperature '{safe_temperature}' to float: {e}")
                
                print(f"  - Safe model: {safe_model} (type: {type(safe_model).__name__})")
                print(f"  - Safe temperature: {safe_temperature} (type: {type(safe_temperature).__name__})")
                print(f"  - Msg: {message_list}")

                if self.mock_output:
                    content = '<thinking>This is a mock output</thinking><answer>This is a mock answer</answer><correct>True</correct><feedback>This is a mock feedback</feedback>'
                else:
                    if self.omit_temperature:
                        safe_temperature = None

                    _htt = _together_http_timeout_sec(self.model)
                    api_params = {
                        "model": safe_model,
                        "messages": message_list,
                        "timeout": _htt,
                        "reasoning_effort": self.reasoning_effort,
                    }
                    if safe_temperature is not None:
                        api_params["temperature"] = safe_temperature
                    api_params["max_tokens"] = self.max_tokens

                    print(f"  - Max tokens: {self.max_tokens}, reasoning_effort: {self.reasoning_effort}")

                    async with get_api_semaphore():
                        response = await self.client.chat.completions.create(**api_params)

                    print(f"✓ Async API request successful")
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        pt = getattr(usage, "prompt_tokens", None) or 0
                        ct = getattr(usage, "completion_tokens", None) or 0
                        record_completion(str(safe_model), int(pt), int(ct), provider="together")
                    msg = response.choices[0].message
                    content = msg.content
                    if content is None and getattr(msg, "reasoning_content", None):
                        content = getattr(msg, "reasoning_content", "") or ""

                if not self.mock_output:
                    if content is None:
                        content = ""
                    if not str(content).strip():
                        print(
                            f"\n✗ Together: empty completion after HTTP 200 (Trial {trial + 1}) — retrying"
                        )
                        exception_backoff = min(2**trial, 60.0)
                        await asyncio.sleep(exception_backoff)
                        trial += 1
                        if trial >= 5:
                            print(f"\n✗ Max trials reached (5) — empty completion persisted")
                            return ""
                        continue

                return content if content is not None else ""

            except APITimeoutError as e:
                print(f"\n✗ Together AI Async API Timeout Error (Trial {trial + 1})")
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Error message: {e}")
                print(f"  - Model used: {self.model}")
                print(f"  - Message count: {len(message_list)}")
                print(f"  - Temperature: {temperature if temperature is not None else self.temperature}")
                print(f"  - Hint: raise MAS_TOGETHER_CHAT_TIMEOUT_SEC (e.g. 600) for long BrowseComp+ prompts")
                
                # For timeout errors, retry with exponential backoff
                exception_backoff = min(2**trial, 120.0)
                print(f"  - Backoff time: {exception_backoff} seconds")
                print(f"  - Waiting {exception_backoff} seconds before retry...")
                await asyncio.sleep(exception_backoff)
                trial += 1
                
                if trial == 5:  # Max retries reached
                    print(f"\n✗ Max trials reached (5) - API Timeout persisted")
                    print(f"  - Final error type: {type(e).__name__}")
                    print(f"  - Final error message: {e}")
                    print(f"  - Returning empty string")
                    return ""

            except openai.BadRequestError as e:
                print(f"\n✗ Together AI Async Bad Request Error (Trial {trial + 1})")
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Error message: {e}")
                print(f"  - Error code: {getattr(e, 'code', 'N/A')}")
                print(f"  - Error status: {getattr(e, 'status', 'N/A')}")
                print(f"  - Error response: {getattr(e, 'response', 'N/A')}")
                print(f"  - Model used: {self.model}")
                print(f"  - Message count: {len(message_list)}")
                print(f"  - Temperature: {temperature if temperature is not None else self.temperature}")
                exception_backoff = min(2**trial, 60.0)
                print(f"  - Backoff: {exception_backoff}s then retry")
                await asyncio.sleep(exception_backoff)
                trial += 1
                if trial >= 5:
                    print(f"  - Returning empty string after {trial} trials")
                    return ""
                continue
                
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(f"\n✗ Together AI Exception (Trial {trial + 1})")
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Error message: {e}")
                print(f"  - Error args: {e.args}")
                print(f"  - Backoff time: {exception_backoff} seconds")
                print(f"  - Model used: {self.model}")
                print(f"  - Message count: {len(message_list)}")
                print(f"  - Temperature: {temperature if temperature is not None else self.temperature}")
                
                # Check if it's a rate limit error
                if "rate limit" in str(e).lower() or "429" in str(e):
                    print(f"  - Detected rate limit error")
                elif "timeout" in str(e).lower() or "APITimeoutError" in str(type(e).__name__):
                    print(f"  - Detected timeout error")
                elif "connection" in str(e).lower():
                    print(f"  - Detected connection error")
                else:
                    print(f"  - Unknown error type")
                
                print(f"  - Waiting {exception_backoff} seconds before retry...")
                await asyncio.sleep(exception_backoff)
                trial += 1
                
                if trial == 5: # basically mean it is bad request after 5 trials
                    print(f"\n✗ Max trials reached (5)")
                    print(f"  - Final error type: {type(e).__name__}")
                    print(f"  - Final error message: {e}")
                    print(f"  - Returning empty string")
                    return ""                    
            # unknown error shall throw exception
