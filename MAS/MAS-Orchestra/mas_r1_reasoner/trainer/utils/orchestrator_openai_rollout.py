"""
Orchestrator generation via an OpenAI-compatible HTTP API (e.g. standalone ``vllm serve``).

Set env ``MAS_ORCHESTRATOR_OPENAI_BASE`` (e.g. ``http://127.0.0.1:8000/v1``) so the training driver
calls the remote server with ``AsyncOpenAI`` + ``asyncio.gather`` instead of in-process vLLM.

Optional: ``MAS_ORCHESTRATOR_MODEL`` (served model id), ``MAS_ORCHESTRATOR_HTTP_CONCURRENCY`` (default 32),
``OPENAI_API_KEY`` (use ``EMPTY`` for local vLLM).
"""
from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import numpy as np
import torch
from openai import AsyncOpenAI
from tensordict import TensorDict
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

_resolved_model_id: Optional[str] = None


def _normalize_openai_base(url: str) -> str:
    u = url.strip().rstrip("/")
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


def _strip_left_pad(pad_token_id: int, prompt_token_ids: torch.Tensor) -> List[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:].tolist()


async def _resolve_model_id(client: AsyncOpenAI) -> str:
    global _resolved_model_id
    env_m = os.environ.get("MAS_ORCHESTRATOR_MODEL", "").strip()
    if env_m:
        return env_m
    if _resolved_model_id:
        return _resolved_model_id
    models = await client.models.list()
    if not models.data:
        raise RuntimeError("OpenAI-compatible server returned no models (GET /v1/models).")
    _resolved_model_id = models.data[0].id
    return _resolved_model_id


async def orchestrator_openai_generate_sequences_async(
    trainer_instance,
    gen_batch: DataProto,
    *,
    is_validation: bool,
) -> DataProto:
    tokenizer = trainer_instance.tokenizer
    cfg = trainer_instance.config.actor_rollout_ref.rollout
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    response_length = int(cfg.response_length)

    base = os.environ.get("MAS_ORCHESTRATOR_OPENAI_BASE", "").strip()
    if not base:
        raise RuntimeError("MAS_ORCHESTRATOR_OPENAI_BASE is not set")

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = AsyncOpenAI(api_key=api_key, base_url=_normalize_openai_base(base))
    model_id = await _resolve_model_id(client)

    if is_validation:
        temperature = float(cfg.val_kwargs.temperature)
        do_sample = bool(cfg.val_kwargs.do_sample)
    else:
        temperature = float(cfg.temperature)
        do_sample = True
    temp = temperature if do_sample else 0.0

    idx = gen_batch.batch["input_ids"]
    attention_mask = gen_batch.batch["attention_mask"]
    position_ids = gen_batch.batch["position_ids"]
    non_tensor_batch = gen_batch.non_tensor_batch
    batch_size = idx.size(0)

    raw_lists: List[List[int]] = []
    if non_tensor_batch is not None and "raw_prompt_ids" in non_tensor_batch:
        rpi = non_tensor_batch["raw_prompt_ids"]
        for i in range(batch_size):
            p = rpi[i]
            if isinstance(p, np.ndarray):
                raw_lists.append(p.tolist())
            else:
                raw_lists.append(list(p))
    else:
        for i in range(batch_size):
            raw_lists.append(_strip_left_pad(pad_token_id, idx[i]))

    conc = int(os.environ.get("MAS_ORCHESTRATOR_HTTP_CONCURRENCY", "32"))
    sem = asyncio.Semaphore(max(1, conc))

    async def one_completion(i: int) -> str:
        prompt_ids = raw_lists[i]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        async with sem:
            comp = await client.completions.create(
                model=model_id,
                prompt=prompt_text,
                max_tokens=response_length,
                temperature=temp,
            )
        return comp.choices[0].text or ""

    texts = await asyncio.gather(*[one_completion(i) for i in range(batch_size)])

    response_rows: List[List[int]] = []
    rollout_log_probs: List[List[float]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > response_length:
            ids = ids[:response_length]
        response_rows.append(ids)
        rollout_log_probs.append([-1.0] * len(ids))

    device = idx.device
    response = pad_2d_list_to_length(response_rows, pad_token_id, max_length=response_length).to(device)
    rlp = pad_2d_list_to_length(rollout_log_probs, -1, max_length=response_length).to(device).to(torch.float32)

    batch_size = idx.size(0)
    seq = torch.cat([idx, response], dim=-1)
    resp_len = response.size(1)
    delta_position_id = torch.arange(1, resp_len + 1, device=position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
    if position_ids.dim() == 3:
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
    response_position_ids = position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
    response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
    attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

    batch = TensorDict(
        {
            "prompts": idx,
            "responses": response,
            "input_ids": seq,
            "rollout_log_probs": rlp,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def orchestrator_openai_generate_sequences(
    trainer_instance,
    gen_batch: DataProto,
    *,
    is_validation: bool,
) -> DataProto:
    """Sync entrypoint (benchmark / Ray driver runs asyncio in a fresh loop)."""
    return asyncio.run(
        orchestrator_openai_generate_sequences_async(trainer_instance, gen_batch, is_validation=is_validation)
    )
