from functools import partial

from sampler.chat_completion_sampler import ChatCompletionSampler, AsyncChatCompletionSampler
from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.together_completion_sampler import ChatCompletionSampler as ToChatCompletionSampler
from sampler.vllm_completion_sampler import AsyncChatCompletionSampler as VllmChatCompletionSampler

model_init_map = {
    "o3-mini": partial(OChatCompletionSampler, model="o3-mini"),
    "gpt-4o_chatgpt": partial(AsyncChatCompletionSampler, model="gpt-4o"),
    "gpt-5-nano": partial(AsyncChatCompletionSampler, model="gpt-5-nano-2025-08-07", temperature=1.0),
    "gpt-5": partial(AsyncChatCompletionSampler, model="gpt-5-2025-08-07", temperature=1.0),
    "qwen-2.5-32b-instr": partial(VllmChatCompletionSampler, model="qwen-2.5-32b-instr"),
    "qwen3-30b-a3b": partial(VllmChatCompletionSampler, model="qwen3-30b-a3b"),
    "qwen3-30b-a3b-reasoning": partial(VllmChatCompletionSampler, model="qwen3-30b-a3b-reasoning"),
    "qwq-32b": partial(ToChatCompletionSampler, model="Qwen/Qwen2.5-32B-Instruct"),
    "llama-3.3-70b-instr": partial(ToChatCompletionSampler, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "qwen3-235b": partial(VllmChatCompletionSampler, model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"),
    "deepseek-v3": partial(ToChatCompletionSampler, model="deepseek-ai/DeepSeek-V3"),
    "qwen3-next-80b-reasoning": partial(VllmChatCompletionSampler, model="qwen3-next-80b-reasoning"),
    "qwen3-235b-reasoning": partial(VllmChatCompletionSampler, model="Qwen/Qwen3-235B-A22B-Thinking-2507"),
    "gpt-oss-120b": partial(VllmChatCompletionSampler, model="gpt-oss-120b"),
    "gemini-2.5-pro": partial(AsyncChatCompletionSampler, model="gemini-2.5-pro")
}

AVAILABLE_MODELS = {}


def init_model(name: str, response_format: str = "xml", max_tokens: int = 4096):
    global AVAILABLE_MODELS
    if name in AVAILABLE_MODELS:
        return
    AVAILABLE_MODELS[name] = model_init_map[name](max_tokens=max_tokens, response_format=response_format)


def get_model(name):
    global AVAILABLE_MODELS
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {name} is not initialized. Please call init_model('{name}') first.")
    return AVAILABLE_MODELS[name]
