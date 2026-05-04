"""
Phase 1: Generate Harmony MAS forward code per sample via local vLLM (OpenAI Completions API).

Prompt construction matches ``helper._prepare_raw_data_batch_for_generation`` + ``prepare_batch_for_generation``
(the same path as the benchmark subprocess). Concurrency is capped with ``asyncio.Semaphore`` (``--jobs`` in
flight): when one sample finishes, the next pending one starts immediately. AsyncOpenAI only. No Ray / VERL.

Extracted ``forward`` code uses ``__MAS_SUB_AGENT_MODEL__`` instead of a fixed LLM name; phase 2 substitutes
the configured sub-agent model at execution (see ``execution.execute_code``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import mas_r1_reasoner.torch_tensordict_compat  # noqa: F401
except ImportError:
    pass
import numpy as np
import torch
from openai import APITimeoutError, AsyncOpenAI
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from mas_r1_reasoner.agents.code_sanity import validate_python_code
from mas_r1_reasoner.data_precessor.MathDatasetProcessor import MathDatasetProcessor
from mas_r1_reasoner.rewards.utils.harmony_parser import extract_harmony_code_from_response
from mas_r1_reasoner.trainer.utils.helper import prepare_batch_for_generation
from mas_r1_reasoner.trainer.utils.orchestrator_openai_rollout import _normalize_openai_base
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

from .bench_hydra import load_grpo_config_for_benchmark
from .cli import default_aflow_data_dir, orchestrator_openai_timeout_seconds, sanitize_agent_dir
from .datasets_aflow import (
    DATASETS,
    DatasetSpec,
    load_jsonl,
    orchestrator_served_model_id,
    row_to_parquet_record,
    stable_id_for_row,
)
from .runner import orchestrator_root, shared_mas_eval_read_root


def _is_real_harmony_forward_code(code: Optional[str]) -> bool:
    """
    True when extraction yielded executable Harmony ``forward`` code, not the
    ``direct_answer`` sentinel and not empty / parse-error placeholders.
    """
    if not isinstance(code, str) or not code:
        return False
    s = code.strip()
    if not s or s == "direct_answer":
        return False
    if "def forward" not in s:
        return False
    return True


class _BenchTrainerStub:
    """Minimal trainer object for ``prepare_batch_for_generation`` (needs ``config``, ``tokenizer``)."""

    def __init__(self, config: Any, tokenizer: Any) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.mas_r1_config = config.azr.mas_r1
        self.processor = MathDatasetProcessor(self)


def _shared_mas_dataset_dir(mas_base: Path, spec: DatasetSpec) -> Path:
    """``mas_base`` is typically ``{repo}/shared_mas`` (see ``shared_mas_eval_read_root``)."""
    return mas_base / spec.result_dir_name


def _placeholder_dataproto(
    *,
    question: str,
    raw_prompt: str,
    reward_model: Dict[str, Any],
    max_prompt_length: int,
    pad_token_id: int,
) -> DataProto:
    input_ids = torch.zeros(1, max_prompt_length, dtype=torch.long)
    attention_mask = torch.zeros(1, max_prompt_length, dtype=torch.long)
    position_ids = compute_position_id_with_mask(attention_mask)
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=[1],
    )
    non_tensor_batch = {
        "question": np.array([question], dtype=object),
        "raw_prompt": np.array([raw_prompt], dtype=object),
        "reward_model": np.array([reward_model], dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def _prompt_text_from_gen_batch(tokenizer, gen_batch: DataProto) -> str:
    """Match ``orchestrator_openai_rollout`` decoding (strip left pad, ``skip_special_tokens=False``)."""
    pad_token_id = tokenizer.pad_token_id
    row = gen_batch.batch["input_ids"][0]
    non_pad_index = torch.nonzero(row != pad_token_id, as_tuple=False)[0][0]
    prompt_ids = row[non_pad_index:].tolist()
    return tokenizer.decode(prompt_ids, skip_special_tokens=False)


async def _orchestrator_completion(
    *,
    client: AsyncOpenAI,
    model_id: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
) -> str:
    comp = await client.completions.create(
        model=model_id,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return comp.choices[0].text or ""


def _should_skip_existing_shared_mas(
    prev: Dict[str, Any],
    *,
    resume: bool,
    resume_harmony: bool,
) -> bool:
    """Whether to skip regenerating an existing JSON (see ``--resume`` / ``--resume-harmony``)."""
    if resume_harmony:
        if prev.get("status") != "completed":
            return False
        return _is_real_harmony_forward_code(prev.get("extracted_code"))
    if resume:
        if prev.get("status") == "completed" and prev.get("extracted_code"):
            return True
    return False


def _pending_indices_for_shared_mas(
    *,
    rows: List[Dict[str, Any]],
    spec: DatasetSpec,
    samples_dir: Path,
    resume: bool,
    resume_harmony: bool,
) -> List[int]:
    """
    Indices that still need a generation pass. When ``--resume`` / ``--resume-harmony`` is off,
    every row is pending (full pass). When on, skips indices that would be no-ops in
    ``_generate_one_sample``, so the semaphore pool stays full of real API work instead of
    mixing many fast skips with few concurrent requests.
    """
    if not resume and not resume_harmony:
        return list(range(len(rows)))
    pending: List[int] = []
    for idx in range(len(rows)):
        out_path = samples_dir / f"{stable_id_for_row(spec.key, rows[idx], idx)}.json"
        if not out_path.is_file():
            pending.append(idx)
            continue
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pending.append(idx)
            continue
        if not _should_skip_existing_shared_mas(prev, resume=resume, resume_harmony=resume_harmony):
            pending.append(idx)
    return pending


async def _generate_one_sample(
    *,
    record: Dict[str, Any],
    raw_row: Dict[str, Any],
    idx: int,
    spec: DatasetSpec,
    cfg: Any,
    tokenizer: Any,
    trainer: _BenchTrainerStub,
    client: AsyncOpenAI,
    orch_model_id: str,
    out_path: Path,
    resume: bool,
    resume_harmony: bool,
    harmony_retry_max: int,
) -> None:
    sid = stable_id_for_row(spec.key, raw_row, idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if (resume or resume_harmony) and out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if _should_skip_existing_shared_mas(prev, resume=resume, resume_harmony=resume_harmony):
                return
        except (json.JSONDecodeError, OSError):
            pass

    question = str(record["prompt"])
    raw_prompt = question
    reward_model = record.get("reward_model") or {}
    gt = reward_model.get("ground_truth", "")

    rollout = cfg.actor_rollout_ref.rollout
    val_kwargs = rollout.val_kwargs
    temperature = float(val_kwargs.temperature)
    do_sample = bool(val_kwargs.do_sample)
    temp = temperature if do_sample else 0.0
    response_length = int(rollout.response_length)
    max_prompt_length = int(cfg.data.max_prompt_length)

    batch = _placeholder_dataproto(
        question=question,
        raw_prompt=raw_prompt,
        reward_model=reward_model if isinstance(reward_model, dict) else {},
        max_prompt_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    gen_batch = prepare_batch_for_generation(trainer, batch)
    prompt_text = _prompt_text_from_gen_batch(tokenizer, gen_batch)

    retries = max(1, int(harmony_retry_max))
    completion_text = ""
    code: str = ""
    name = ""
    thought = ""
    extraction_error: Optional[str] = None
    attempts_used = 0
    ok = False

    for attempt in range(1, retries + 1):
        attempts_used = attempt
        try:
            completion_text = await _orchestrator_completion(
                client=client,
                model_id=orch_model_id,
                prompt_text=prompt_text,
                max_tokens=response_length,
                temperature=temp,
            )
        except APITimeoutError:
            # One attempt failed on the wire; retry within harmony_retry_max without killing other samples.
            completion_text = ""
            code = ""
            name = ""
            thought = ""
            if attempt >= retries:
                extraction_error = "orchestrator_apitimeout"
                break
            extraction_error = None
            continue

        code = ""
        name = ""
        thought = ""
        extraction_error = None
        try:
            if "harmony" in str(cfg.azr.problem_type):
                code, name, thought = extract_harmony_code_from_response(
                    completion_text, validate_python_code, None
                )
            else:
                extraction_error = "non_harmony_problem_type_not_supported_in_shared_generator"
        except Exception as e:
            extraction_error = str(e)

        if extraction_error == "non_harmony_problem_type_not_supported_in_shared_generator":
            break

        if extraction_error is None and _is_real_harmony_forward_code(code):
            ok = True
            break
        # discard this attempt and retry unless last attempt
        if attempt >= retries:
            break

    if not ok and extraction_error is None:
        extraction_error = "harmony_retry_no_valid_forward_code"
    ok = ok and extraction_error is None

    # Omit orchestrator checkpoint paths from exported JSON (anonymous + portable); use
    # ``DatasetSpec.orchestrator_hf`` in ``datasets_aflow.py`` at runtime.
    doc: Dict[str, Any] = {
        "status": "completed" if ok else "failed",
        "sample_id": sid,
        "dataset_key": spec.key,
        "benchmark_index": idx,
        "question": question,
        "label_ground_truth": gt,
        "prompt_text": prompt_text,
        "orchestrator_completion_text": completion_text,
        "extracted_code": code,
        "extracted_name": name,
        "extracted_thought": thought,
        "code_extraction_success": ok,
        "extraction_error": extraction_error,
        "problem_type": spec.problem_type,
        "hydra_agent_model_name": OmegaConf.to_container(cfg.azr.mas_r1.agent, resolve=True).get("model_name"),
        "harmony_retry_max": retries,
        "harmony_retry_attempts_used": attempts_used,
        "harmony_retry_exhausted": bool(not ok and attempts_used >= retries),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate shared MAS forward code (phase 1)")
    p.add_argument(
        "--run-suffix",
        default=os.environ.get("RUN_SUFFIX", "run1"),
        help="Legacy label for logs/env only; output goes to --mas-root / shared_mas (not shared_mas_{suffix}).",
    )
    p.add_argument(
        "--mas-root",
        type=Path,
        default=None,
        help="Directory for <Dataset>/samples/*.json (default: {repo}/shared_mas; env MAS_SHARED_MAS_DIR).",
    )
    p.add_argument("--datasets", nargs="*", default=["GPQA"])
    p.add_argument("--aflow-data-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--orchestrator-openai-base",
        default=os.environ.get("MAS_ORCHESTRATOR_OPENAI_BASE", "").strip() or None,
        help="Required. E.g. http://127.0.0.1:8000/v1",
    )
    p.add_argument("--orchestrator-model", default=os.environ.get("MAS_ORCHESTRATOR_MODEL", "").strip() or None)
    p.add_argument(
        "--orchestrator-agent-model-name",
        default=os.environ.get("MAS_ORCHESTRATOR_AGENT_MODEL_NAME", "gpt-4o"),
        help="Sets azr.mas_r1.agent.model_name for [AGENT_MODEL]/[MODEL] in orchestrator prompts (default gpt-4o).",
    )
    p.add_argument("--n-gpus", type=int, default=int(os.environ.get("N_GPUS", "1")))
    p.add_argument("--tp-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    p.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("MAS_ORCHESTRATOR_HTTP_CONCURRENCY", "8")),
        help="Max concurrent orchestrator HTTP calls (asyncio.Semaphore; next sample starts when a slot frees).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples whose JSON exists with status completed and non-empty extracted_code (legacy; includes direct_answer).",
    )
    p.add_argument(
        "--resume-harmony",
        action="store_true",
        help="Skip only when the JSON has status completed and real Harmony forward code (not direct_answer). "
        "Re-runs missing, failed, and direct_answer-only samples.",
    )
    p.add_argument(
        "--harmony-retry-max",
        type=int,
        default=1,
        metavar="N",
        help="Per sample: up to N orchestrator calls until extracted_code is real forward() (not direct_answer). "
        "Default 1. Use e.g. 10 with --resume-harmony to retry direct-answer generations.",
    )
    args = p.parse_args(argv)
    if args.resume and args.resume_harmony:
        print("ERROR: use only one of --resume or --resume-harmony", file=sys.stderr)
        raise SystemExit(2)
    if int(args.harmony_retry_max) < 1:
        print("ERROR: --harmony-retry-max must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    return args


async def _run_async(args: argparse.Namespace) -> int:
    if not args.orchestrator_openai_base:
        print("ERROR: --orchestrator-openai-base or MAS_ORCHESTRATOR_OPENAI_BASE is required", file=sys.stderr)
        return 2

    aflow = args.aflow_data_dir or default_aflow_data_dir()
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_normalize_openai_base(args.orchestrator_openai_base),
        timeout=orchestrator_openai_timeout_seconds(),
    )
    concurrency = max(1, int(args.jobs))

    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"Unknown dataset {ds_name}, skip", file=sys.stderr)
            continue
        spec = DATASETS[ds_name]
        jsonl_path = aflow / spec.jsonl_name
        if not jsonl_path.is_file():
            print(f"Missing data file: {jsonl_path}", file=sys.stderr)
            continue

        rows = load_jsonl(jsonl_path)
        if args.limit is not None:
            rows = rows[: args.limit]

        cfg = load_grpo_config_for_benchmark(
            spec,
            args.orchestrator_agent_model_name,
            n_gpus=args.n_gpus,
            tp_size=args.tp_size,
        )
        orch_model = args.orchestrator_model or orchestrator_served_model_id(spec.orchestrator_hf)

        local_model = copy_to_local(
            cfg.actor_rollout_ref.model.path,
            use_shm=cfg.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = cfg.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_model, trust_remote_code=trust_remote_code)

        trainer = _BenchTrainerStub(cfg, tokenizer)
        agent_cfg = OmegaConf.to_container(cfg.azr.mas_r1.agent, resolve=True)
        if not isinstance(agent_cfg, dict):
            agent_cfg = dict(agent_cfg)
        mas_r1_cfg = OmegaConf.to_container(cfg.azr.mas_r1, resolve=True)
        if not isinstance(mas_r1_cfg, dict):
            mas_r1_cfg = dict(mas_r1_cfg)
        trainer.processor.setup_global_variables(agent_cfg, mas_r1_cfg, cfg)

        mas_base = args.mas_root if args.mas_root is not None else shared_mas_eval_read_root()
        out_dir = _shared_mas_dataset_dir(mas_base, spec)
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        pending = _pending_indices_for_shared_mas(
            rows=rows,
            spec=spec,
            samples_dir=samples_dir,
            resume=args.resume,
            resume_harmony=args.resume_harmony,
        )
        if args.resume or args.resume_harmony:
            n_skip = len(rows) - len(pending)
            print(
                f"{ds_name}: resume: {n_skip} sample(s) already done, {len(pending)} to run "
                f"(max in-flight --jobs={concurrency})",
                flush=True,
            )

        if not pending:
            continue

        sem = asyncio.Semaphore(concurrency)

        async def _run_one_index(idx: int) -> None:
            async with sem:
                await _generate_one_sample(
                    record=row_to_parquet_record(spec, rows[idx], idx),
                    raw_row=rows[idx],
                    idx=idx,
                    spec=spec,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    trainer=trainer,
                    client=client,
                    orch_model_id=orch_model,
                    out_path=samples_dir / f"{stable_id_for_row(spec.key, rows[idx], idx)}.json",
                    resume=args.resume,
                    resume_harmony=args.resume_harmony,
                    harmony_retry_max=args.harmony_retry_max,
                )

        await asyncio.gather(*[_run_one_index(idx) for idx in pending])

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rr = orchestrator_root()
    os.environ["PYTHONPATH"] = str(rr) + os.pathsep + os.environ.get("PYTHONPATH", "")
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
