"""
Merge phase-1 shared_mas jobs across multiple ``--run-suffix`` values.

Scans ``shared_mas_{suffix}/<dataset>/samples/*.json`` for every listed suffix and dataset.
Queues samples that still need real Harmony ``forward`` code (same criterion as ``--resume-harmony``:
re-run if missing file, not completed, or ``extracted_code`` is not valid forward code).

Runs the queued work under one ``asyncio.Semaphore`` (``--jobs`` in flight): when a sample finishes,
the next queued task starts immediately. Each result is written only to its own
``shared_mas_{suffix}/.../samples/{idx}.json``.

Before regenerating, re-reads the JSON; if it already has valid forward code (e.g. another process
finished), skips without overwriting — **never overwrites already-good samples**.

Assumptions (same as phase-1 index-based layout): the JSONL used for ``--aflow-data-dir`` / default
aflow dir is the **same ordering** as when ``shared_mas_*`` was produced; ``idx`` matches
``stable_id_for_row`` (linear index). ``--limit`` only scans the first N indices — do not expect a
full-dataset sweep. Concurrent **processes** writing the same ``out_path`` are not file-locked (last
writer wins); use a single cross-run worker or accept rare races.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI
from omegaconf import OmegaConf

from .bench_hydra import load_grpo_config_for_benchmark
from .cli import default_aflow_data_dir, orchestrator_openai_timeout_seconds
from .datasets_aflow import DATASETS, DatasetSpec, load_jsonl, orchestrator_served_model_id, row_to_parquet_record
from .runner import orchestrator_root, shared_mas_root

# Set by lazy import inside ``_run_async`` (must be module globals so ``_guarded_generate`` can see them).
_BenchTrainerStub: Any = None
_generate_one_sample: Optional[Callable[..., Any]] = None


def _normalize_openai_base(url: str) -> str:
    """Same normalization as ``orchestrator_openai_rollout`` without importing torch/verl."""
    u = url.strip().rstrip("/")
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


def _is_real_harmony_forward_code(code: Optional[str]) -> bool:
    """Keep in sync with ``generate_mas_shared._is_real_harmony_forward_code`` (lightweight copy for import-free scan)."""
    if not isinstance(code, str) or not code:
        return False
    s = code.strip()
    if not s or s == "direct_answer":
        return False
    if "def forward" not in s:
        return False
    return True


@dataclass(frozen=True)
class CrossRunTask:
    run_suffix: str
    dataset_name: str
    benchmark_index: int
    out_path: Path


def _load_sample_doc(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _needs_regeneration_for_resume_harmony(prev: Optional[Dict[str, Any]], path_exists: bool) -> bool:
    """Same logical negation as ``--resume-harmony`` skip: queue if we would NOT skip."""
    if not path_exists or prev is None:
        return True
    if prev.get("status") != "completed":
        return True
    return not _is_real_harmony_forward_code(prev.get("extracted_code"))


async def _guarded_generate(
    *,
    task: CrossRunTask,
    spec: DatasetSpec,
    rows: List[Dict[str, Any]],
    cfg: Any,
    tokenizer: Any,
    trainer: Any,
    client: AsyncOpenAI,
    orch_model_id: str,
    harmony_retry_max: int,
) -> None:
    """Re-read on disk; skip if sample already has valid Harmony forward code."""
    if task.benchmark_index < 0 or task.benchmark_index >= len(rows):
        print(
            f"[skip] {task.run_suffix} {task.dataset_name} idx={task.benchmark_index} "
            f"(out of range for jsonl n={len(rows)} — check --limit / aflow data)",
            file=sys.stderr,
            flush=True,
        )
        return

    if task.out_path.is_file():
        prev = _load_sample_doc(task.out_path)
        if prev and prev.get("status") == "completed" and _is_real_harmony_forward_code(prev.get("extracted_code")):
            print(
                f"[skip] {task.run_suffix} {task.dataset_name} idx={task.benchmark_index} "
                f"(already has forward code)",
                flush=True,
            )
            return

    raw_row = rows[task.benchmark_index]
    record = row_to_parquet_record(spec, raw_row, task.benchmark_index)

    await _generate_one_sample(
        record=record,
        raw_row=raw_row,
        idx=task.benchmark_index,
        spec=spec,
        cfg=cfg,
        tokenizer=tokenizer,
        trainer=trainer,
        client=client,
        orch_model_id=orch_model_id,
        out_path=task.out_path,
        resume=False,
        resume_harmony=False,
        harmony_retry_max=harmony_retry_max,
    )
    print(
        f"[done] {task.run_suffix} {task.dataset_name} idx={task.benchmark_index} -> {task.out_path}",
        flush=True,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge shared_mas phase-1 regeneration across multiple run suffixes (one concurrent pool)."
    )
    p.add_argument(
        "--run-suffixes",
        nargs="+",
        default=["run1", "run2", "run3"],
        metavar="SUFFIX",
        help="E.g. run1 run2 run3 -> shared_mas_run1, shared_mas_run2, ...",
    )
    p.add_argument("--datasets", nargs="+", default=["SWE-Bench-Lite"], help="Keys in benchmark_eval.datasets_aflow.DATASETS")
    p.add_argument("--aflow-data-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Truncate jsonl rows (debug).")
    p.add_argument(
        "--orchestrator-openai-base",
        default=os.environ.get("MAS_ORCHESTRATOR_OPENAI_BASE", "").strip() or None,
    )
    p.add_argument("--orchestrator-model", default=os.environ.get("MAS_ORCHESTRATOR_MODEL", "").strip() or None)
    p.add_argument(
        "--orchestrator-agent-model-name",
        default=os.environ.get("MAS_ORCHESTRATOR_AGENT_MODEL_NAME", "gpt-4o"),
    )
    p.add_argument("--n-gpus", type=int, default=int(os.environ.get("N_GPUS", "1")))
    p.add_argument("--tp-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    p.add_argument("--jobs", type=int, default=int(os.environ.get("MAS_ORCHESTRATOR_HTTP_CONCURRENCY", "8")))
    p.add_argument(
        "--harmony-retry-max",
        type=int,
        default=10,
        metavar="N",
        help="Per sample orchestrator retries until valid forward() (default 10).",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Print queued task counts per (run_suffix, dataset) and exit without calling the API.",
    )
    args = p.parse_args(argv)
    if int(args.harmony_retry_max) < 1:
        print("ERROR: --harmony-retry-max must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    # Stable order; avoids duplicate work if the same suffix is passed twice.
    args.run_suffixes = list(dict.fromkeys(args.run_suffixes))
    return args


async def _run_async(args: argparse.Namespace) -> int:
    aflow = args.aflow_data_dir or default_aflow_data_dir()
    tasks, rows_by_dataset = _discover_tasks_with_aflow(
        run_suffixes=args.run_suffixes,
        dataset_names=args.datasets,
        limit_per_jsonl=args.limit,
        aflow=aflow,
    )

    if not tasks:
        print("No samples need regeneration (all have valid Harmony forward code, or missing jsonl).")
        return 0

    if args.list_only:
        _print_task_breakdown(tasks)
        print(f"Total queued tasks: {len(tasks)}")
        return 0

    if not args.orchestrator_openai_base:
        print("ERROR: --orchestrator-openai-base or MAS_ORCHESTRATOR_OPENAI_BASE is required (unless --list-only)", file=sys.stderr)
        return 2

    # Heavy stack (torch / verl / MAS generation) only when we actually call the API.
    # Bind on the module so module-level ``_guarded_generate`` resolves ``_generate_one_sample``.
    global _BenchTrainerStub, _generate_one_sample
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local

    from .generate_mas_shared import _BenchTrainerStub as _BTS, _generate_one_sample as _gos

    _BenchTrainerStub = _BTS
    _generate_one_sample = _gos

    print(f"Cross-run regeneration: {len(tasks)} task(s) across run_suffixes={list(args.run_suffixes)} datasets={list(args.datasets)}", flush=True)

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_normalize_openai_base(args.orchestrator_openai_base),
        timeout=orchestrator_openai_timeout_seconds(),
    )
    concurrency = max(1, int(args.jobs))

    # Group tasks by dataset_name for shared cfg/tokenizer/trainer
    by_ds: Dict[str, List[CrossRunTask]] = {}
    for t in tasks:
        by_ds.setdefault(t.dataset_name, []).append(t)

    for ds_name in args.datasets:
        if ds_name not in by_ds:
            continue
        spec = DATASETS[ds_name]
        rows = rows_by_dataset.get(ds_name) or []
        if not rows:
            continue

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

        ds_tasks = by_ds[ds_name]
        if not ds_tasks:
            continue

        sem = asyncio.Semaphore(concurrency)

        async def _run_task(t: CrossRunTask) -> None:
            async with sem:
                await _guarded_generate(
                    task=t,
                    spec=spec,
                    rows=rows,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    trainer=trainer,
                    client=client,
                    orch_model_id=orch_model,
                    harmony_retry_max=args.harmony_retry_max,
                )

        await asyncio.gather(*[_run_task(t) for t in ds_tasks])

    return 0


def _discover_tasks_with_aflow(
    *,
    run_suffixes: Sequence[str],
    dataset_names: Sequence[str],
    limit_per_jsonl: Optional[int],
    aflow: Path,
) -> Tuple[List[CrossRunTask], Dict[str, List[Dict[str, Any]]]]:
    """Scan shared_mas_{suffix} trees; queue indices that ``--resume-harmony`` would still re-run."""
    tasks: List[CrossRunTask] = []
    rows_by_dataset: Dict[str, List[Dict[str, Any]]] = {}

    for ds_name in dataset_names:
        if ds_name not in DATASETS:
            print(f"Unknown dataset {ds_name}, skip", file=sys.stderr)
            continue
        spec = DATASETS[ds_name]
        jsonl_path = aflow / spec.jsonl_name
        if not jsonl_path.is_file():
            print(f"Missing data file: {jsonl_path}", file=sys.stderr)
            continue
        rows = load_jsonl(jsonl_path)
        if limit_per_jsonl is not None:
            rows = rows[: int(limit_per_jsonl)]
        rows_by_dataset[ds_name] = rows
        n = len(rows)

        for run_suffix in run_suffixes:
            samples_dir = shared_mas_root(run_suffix) / spec.result_dir_name / "samples"
            for idx in range(n):
                out_path = samples_dir / f"{idx}.json"
                path_exists = out_path.is_file()
                prev = _load_sample_doc(out_path) if path_exists else None
                if _needs_regeneration_for_resume_harmony(prev, path_exists):
                    tasks.append(
                        CrossRunTask(
                            run_suffix=run_suffix,
                            dataset_name=ds_name,
                            benchmark_index=idx,
                            out_path=out_path,
                        )
                    )

    return tasks, rows_by_dataset


def _print_task_breakdown(tasks: Sequence[CrossRunTask]) -> None:
    key_counts: Dict[Tuple[str, str], int] = {}
    for t in tasks:
        k = (t.run_suffix, t.dataset_name)
        key_counts[k] = key_counts.get(k, 0) + 1
    print("Queued regeneration (would call API unless --list-only):\n")
    for (rs, dn) in sorted(key_counts.keys()):
        print(f"  {rs}  {dn}: {key_counts[(rs, dn)]}")
    print(
        "\nThese counts are samples that still need a valid Harmony forward (same as --resume-harmony). "
        "They are NOT the same as 'non-empty extracted_code %' — many rows can have text in extracted_code "
        "but fail the stricter forward check (`def forward`, not `direct_answer`, etc.).\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rr = orchestrator_root()
    os.environ["PYTHONPATH"] = str(rr) + os.pathsep + os.environ.get("PYTHONPATH", "")
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
