import asyncio
import json
import os
import re
import time
from collections import OrderedDict
from time import process_time_ns
from typing import Any
import requests

import aiohttp

from utils import extract_xml
from .sampler_base import SamplerBase, MessageList


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
            self,
            system_message: str | None = None,
            temperature: float = 0.5,
            model: str | None = None,
            max_tokens: int = 4096,
            response_format: str = "json"
    ):

        # model_api_map = {
        #     'qwen-2.5-32b-instr': '8082',
        #     'qwen3-30b-a3b': '8000',
        # }
        self.api_key_name = "API_KEY"
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.getenv(self.api_key_name, "")

        url_base = os.getenv("BASE_URL", "http://localhost:8000/v1/chat/completions")
        self.url_base = url_base

        self.model = model

        self.response_format = response_format
        # Prevent unbounded waiting on flaky backend responses.
        self.max_retries = int(os.getenv("VLLM_MAX_RETRIES", "3"))
        self.request_timeout = int(os.getenv("VLLM_REQUEST_TIMEOUT", "1200"))
        # Keep last two user/assistant rounds by default (4 non-system messages).
        self.keep_last_non_system = max(0, int(os.getenv("VLLM_KEEP_LAST_NON_SYSTEM", "4")))

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def xml_to_json(self, ori_answer):
        output_dict = OrderedDict()  # <-- keep insertion order
        tag_names = re.findall(r"</?(\w+)>", ori_answer)
        ordered_unique_tags = list(OrderedDict.fromkeys(tag_names))
        print('tag_names: ', tag_names)

        for tag in ordered_unique_tags:
            if all(t not in tag for t in ['A', 'B', 'C', 'D', 'sub', 'S_y', 'TOO_HARD', 'command', 'new', 'data', 'comment']):
                tag_text = extract_xml(ori_answer, tag)
                output_dict[tag] = tag_text
        json_string = json.dumps(output_dict, indent=4)
        return json_string

    @staticmethod
    def _extract_answer_and_usage(result: dict):
        choices = result.get("choices")
        if not choices or not isinstance(choices, list):
            raise KeyError("choices")
        message = choices[0].get("message", {})
        ori_answer = message.get("content")
        if ori_answer is None:
            raise KeyError("choices[0].message.content")
        usage = result.get("usage", {})
        return ori_answer, usage

    @staticmethod
    def _is_max_length_error(error: Exception) -> bool:
        error_text = str(error).lower()
        max_length_patterns = [
            "max_tokens must be at least 1",
            "maximum context length",
            "max model len",
            "context length",
            "prompt is too long",
        ]
        return any(pattern in error_text for pattern in max_length_patterns)

    @staticmethod
    def _drop_earliest_non_system_turn(
        message_list: MessageList,
        keep_last_non_system: int = 0,
    ) -> bool:
        non_system_idxs = [
            idx
            for idx, message in enumerate(message_list)
            if str(message.get("role", "")).lower() != "system"
        ]
        if len(non_system_idxs) <= keep_last_non_system:
            return False
        message_list.pop(non_system_idxs[0])
        return True

    def _shrink_context_after_overflow(self, message_list: MessageList) -> bool:
        # First preserve recent turns, then relax this if context still cannot fit.
        if self._drop_earliest_non_system_turn(
            message_list, keep_last_non_system=self.keep_last_non_system
        ):
            print(
                "VLLM: max model length reached, dropped earliest non-system turn and retrying.",
                flush=True,
            )
            return True
        if self.keep_last_non_system > 0 and self._drop_earliest_non_system_turn(
            message_list, keep_last_non_system=0
        ):
            print(
                "VLLM: context still too long, overriding keep-last policy and dropping more history.",
                flush=True,
            )
            return True
        return False

    def __call__(self, message_list: MessageList, temperature=None, response_format=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        message_list = [dict(message) for message in message_list]
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        last_exception = None
        while trial < self.max_retries:
            try:
                for message_id, message in enumerate(message_list):
                    if not isinstance(message['content'], str):
                        message_list[message_id]['content'] = str(message['content'])

                payload = {
                    "model": self.model,
                    "messages": message_list,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "stream": False,
                }

                # 发送同步请求
                response = requests.post(
                    self.url_base,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.request_timeout,
                )
                # print(response.headers)

                if response.status_code == 200:
                    response = response.json()
                else:
                    response.raise_for_status()
                # print('response: ', response, flush=True)
                ori_answer, usage = self._extract_answer_and_usage(response)
                # print('ori_answer: ',ori_answer)

                # json_string = self.xml_to_json(ori_answer)
                if self.response_format == "xml":
                    json_string = self.xml_to_json(ori_answer)
                else:
                    json_string = ori_answer

                return json_string, usage
            except Exception as e:
                import traceback
                traceback.print_exc()
                last_exception = e
                if self._is_max_length_error(e):
                    if self._shrink_context_after_overflow(message_list):
                        continue
                    raise RuntimeError(
                        "VLLM sampler context exceeds model length and cannot be reduced further "
                        "(current retained turns/system prompt are too large)."
                    ) from e
                exception_backoff = 2 ** trial  # expontial back off
                print(
                    f"VLLM: Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                    flush=True
                )
                time.sleep(exception_backoff)
                trial += 1
        raise RuntimeError(
            f"VLLM sampler failed after {self.max_retries} retries (model={self.model}, url={self.url_base})"
        ) from last_exception


class AsyncChatCompletionSampler(ChatCompletionSampler):
    async def __call__(self, message_list: MessageList, temperature=None, response_format=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        message_list = [dict(message) for message in message_list]
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        last_exception = None
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)

        while trial < self.max_retries:
            try:
                for message_id, message in enumerate(message_list):
                    if not isinstance(message['content'], str):
                        message_list[message_id]['content'] = str(message['content'])

                payload = {
                    "model": self.model,
                    "messages": message_list,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "stream": False,
                }
                # 异步请求
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.url_base, headers=headers, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(
                                f"Request failed: HTTP {response.status}, {error_text}"
                            )

                # print('response: ',response)
                ori_answer, usage = self._extract_answer_and_usage(result)
                # print('ori_answer: ',ori_answer)
                if self.response_format == "xml":
                    json_string = self.xml_to_json(ori_answer)
                else:
                    json_string = ori_answer

                # json_string = ori_answer

                # print(json_string)
                return json_string, usage
            except Exception as e:
                import traceback
                traceback.print_exc()
                last_exception = e
                if self._is_max_length_error(e):
                    if self._shrink_context_after_overflow(message_list):
                        continue
                    raise RuntimeError(
                        "VLLM async sampler context exceeds model length and cannot be reduced further "
                        "(current retained turns/system prompt are too large)."
                    ) from e
                exception_backoff = 2 * trial  # exponential back off
                print(
                    f"VLLM: Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                await asyncio.sleep(exception_backoff)
                trial += 1
        raise RuntimeError(
            f"VLLM async sampler failed after {self.max_retries} retries (model={self.model}, url={self.url_base})"
        ) from last_exception


if __name__ == '__main__':
    client = AsyncChatCompletionSampler(model="gpt-oss-120b", response_format="json")

    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Output a json string with single field: `capital`."},
    ]

    results = asyncio.run(client(history))
    print(results)
