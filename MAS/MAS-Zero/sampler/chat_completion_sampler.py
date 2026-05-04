import json
import os
import time
from typing import Any

import openai
from openai import OpenAI, AsyncOpenAI

from .sampler_base import SamplerBase, MessageList


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
            self,
            model: str = "gpt-3.5-turbo",
            system_message: str | None = None,
            temperature: float = 0.5,
            max_tokens: int = 1024,
            response_format: str = "json"
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI(
            api_key=os.getenv(self.api_key_name, ""),
            base_url=os.getenv("OPENAI_BASE_URL", None)
        )
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.response_format = response_format

    def _handle_image(
            self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList, temperature=None, response_format=None) -> str:
        if temperature != 1.0 and "gpt-5" in self.model:
            temperature = 1.0
        if response_format is None:
            response_format = self.response_format
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                for message_id, message in enumerate(message_list):
                    if type(message['content']) != str:
                        message_list[message_id]['content'] = str(message['content'])
                # print('message_list: ',message_list)

                kwargs = {"stream": False}
                if "gpt-5" in self.model:
                    kwargs["max_completion_tokens"] = self.max_tokens
                else:
                    kwargs["max_tokens"] = self.max_tokens

                if response_format == 'normal':
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        **kwargs
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        response_format={"type": "json_object"},
                        **kwargs
                    )
                # print('response: ', response)
                return response.choices[0].message.content, response.usage
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                import traceback
                traceback.print_exc()
                print("Bad Request Error", e)
                return json.dumps({}), 0
            except Exception as e:
                exception_backoff = 2 ** trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial == 3:  # basically mean it is bad request after 3 trials
                    print("Bad Request Error", e)
                    return json.dumps({}), 0
            # unknown error shall throw exception


class AsyncChatCompletionSampler(ChatCompletionSampler):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
            self,
            model: str = "gpt-3.5-turbo",
            system_message: str | None = None,
            temperature: float = 0.5,
            max_tokens: int = 1024,
            response_format: str = "json",
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = AsyncOpenAI(
            api_key=os.getenv(self.api_key_name, ""),
            base_url=os.getenv("OPENAI_BASE_URL", None)
        )
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        # print(os.getenv("OPENAI_BASE_URL", None))
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.response_format = response_format

    async def __call__(self, message_list: MessageList, temperature=None, response_format=None) -> str:
        if temperature != 1.0 and "gpt-5" in self.model:
            temperature = 1.0
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                for message_id, message in enumerate(message_list):
                    if type(message['content']) != str:
                        message_list[message_id]['content'] = str(message['content'])
                # print('message_list: ',message_list)

                if response_format == 'normal':
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        max_tokens=self.max_tokens
                    )
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        # max_tokens=self.max_tokens,
                        response_format={"type": "json_object"}
                    )
                # print('response: ',response)
                return response.choices[0].message.content, response.usage
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2 ** trial  # expontial back off
                import traceback
                traceback.print_exc()
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial == 3:  # basically mean it is bad request after 3 trials
                    print("Bad Request Error", e)
                    return ""
            # unknown error shall throw exception


if __name__ == "__main__":
    import asyncio

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY is not set; skipping sampler tests.")
    else:
        messages = [{"role": "user", "content": "Say 'Hello World' in a creative way."}]

        print("Testing ChatCompletionSampler (Sync)...")
        try:
            sync_sampler = ChatCompletionSampler(model="gemini-2.5-pro")
            text, usage = sync_sampler(messages, response_format="normal")
            print(f"Response: {text}")
            print(f"Usage: {usage}\n")
        except Exception as e:
            print("Sync sampler test failed:", e)


        async def test_async():
            print("Testing AsyncChatCompletionSampler (Async)...")
            sampler = AsyncChatCompletionSampler(model="gemini-2.5-pro", response_format="normal")
            try:
                text, usage = await sampler(messages, response_format="normal")
                print(f"Response: {text}")
                print(f"Usage: {usage}")
            except Exception as e:
                print("Async sampler test failed:", e)


        asyncio.run(test_async())
