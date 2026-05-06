"""
Phase 2: Load cached MAS forward code from ``shared_mas/<dataset>/samples/`` (same tree for every
``--run-suffix``; override with ``--mas-root`` or ``MAS_SHARED_MAS_DIR``) and execute with
chosen agent models (OpenAI-compatible APIs via existing samplers). Writes the same sample JSON
layout as ``benchmark_eval.cli`` (per agent / dataset / sample_id).

Execution uses in-process ``AgentSystem.forward`` (``AsyncAgentSystem(..., local_exec=True)``): no Ray
remote workers and no Phase-1 VERL trainer stack; config is ``grpo_trainer.yaml`` via OmegaConf only.

By default, sub-agent API concurrency is capped at 1 (sequential calls along the forward code path).
Override with ``--api-max-concurrency`` or ``MAS_API_MAX_CONCURRENCY`` / ``MAX_CONCURRENT``.

Harmony ``forward`` code uses the placeholder ``__MAS_SUB_AGENT_MODEL__``; ``execute_code`` replaces it with
``global_node_model`` from the current config (set from ``--agent-model`` before execution).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf, open_dict

from mas_r1_reasoner.agents.sampler.grpo_model_sampler_params import (
    build_minimal_sampler_config,
    resolve_sampler_map_entry,
    subagent_use_together_completion,
)
from mas_r1_reasoner.agents.sampler.api_usage_tracker import (
    reset_api_semaphore,
    reset_usage_log_path_context,
    set_usage_log_path_context,
)
from mas_r1_reasoner.data_precessor.BaseDatasetProcessor import BaseDatasetProcessor
from mas_r1_reasoner.rewards.utils.execution import execute_code
from mas_r1_reasoner.trainer.phase2_agent_init import initialize_mas_r1_agent_system_phase2

from .aflow_bridge import ensure_aflow_on_syspath
from .phase2_config import load_grpo_config_phase2
from .cli import (
    dataset_result_dir,
    default_aflow_data_dir,
    final_sample_path,
    load_cached,
    sanitize_agent_dir,
)
from .datasets_aflow import DATASETS, load_jsonl, row_to_parquet_record, stable_id_for_row
from .phase2_summary import write_phase2_dataset_summary
from .pricing import load_usage_log
from .runner import _resolve_mas_api_max_concurrency, orchestrator_root, shared_mas_eval_read_root
from .scoring import compute_accuracy, smfr_eval_breakdown
from .smfr_structured_output import maybe_augment_smfr_prediction


class _EvalTrainerStub:
    """Minimal trainer for Phase-2 agent init + ``execute_code`` (no tokenizer / no actor weights)."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.tokenizer = None
        self.mas_r1_config = config.azr.mas_r1
        self.processor = BaseDatasetProcessor(self)
        self.agent_system = None
        self.code_execution_timeout = config.azr.mas_r1.get("code_execution_timeout")


def _init_eval_stub(cfg: Any) -> _EvalTrainerStub:
    """One isolated stub + AgentSystem (safe for concurrent workers; do not share across tasks)."""
    stub = _EvalTrainerStub(cfg)
    agent_cfg = OmegaConf.to_container(cfg.azr.mas_r1.agent, resolve=True)
    if not isinstance(agent_cfg, dict):
        agent_cfg = dict(agent_cfg)
    mas_r1_cfg = OmegaConf.to_container(cfg.azr.mas_r1, resolve=True)
    if not isinstance(mas_r1_cfg, dict):
        mas_r1_cfg = dict(mas_r1_cfg)
    stub.processor.setup_global_variables(agent_cfg, mas_r1_cfg, cfg)
    # In-process forward only; avoids RayAgentWorker pool (important for multi-sample --jobs).
    stub._mas_eval_local_exec = True
    initialize_mas_r1_agent_system_phase2(stub, cfg)
    return stub


def _ensure_sampler_for_agent(cfg: Any, agent_model: str) -> None:
    msm = cfg.azr.mas_r1.agent.model_sampler_map
    keys = list(msm.keys()) if msm is not None else []
    if agent_model in keys:
        return
    yaml_entry = resolve_sampler_map_entry(agent_model)
    if yaml_entry:
        with open_dict(cfg.azr.mas_r1.agent.model_sampler_map):
            cfg.azr.mas_r1.agent.model_sampler_map[agent_model] = OmegaConf.create(dict(yaml_entry))
        return
    use_together = subagent_use_together_completion(agent_model)
    st = "TogetherCompletionSampler" if use_together else "ChatCompletionSampler"
    merged = build_minimal_sampler_config(model_id=agent_model, sampler_type=st)
    with open_dict(cfg.azr.mas_r1.agent.model_sampler_map):
        cfg.azr.mas_r1.agent.model_sampler_map[agent_model] = OmegaConf.create(merged)


def _results_base(run_suffix: str) -> Path:
    return orchestrator_root() / f"results_{run_suffix}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate cached shared MAS with agent LLMs (phase 2)")
    p.add_argument("--run-suffix", default=os.environ.get("RUN_SUFFIX", "run1"))
    p.add_argument(
        "--agent-model",
        nargs="+",
        default=[os.environ.get("AGENT_MODEL", "gpt-4o")],
        help="One or more agent models (missing map entries get ChatCompletionSampler)",
    )
    p.add_argument("--datasets", nargs="*", default=["GPQA"])
    p.add_argument("--aflow-data-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--n-gpus",
        type=int,
        default=int(os.environ.get("N_GPUS", "1")),
        help="Ignored in Phase 2 (no orchestrator rollout). Kept for CLI compatibility.",
    )
    p.add_argument(
        "--tp-size",
        type=int,
        default=int(os.environ.get("TP_SIZE", "1")),
        help="Ignored in Phase 2. Kept for CLI compatibility.",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=int(os.environ.get("BENCHMARK_JOBS", "1")),
        metavar="N",
        help="Max concurrent samples per dataset (asyncio). Each slot uses its own AgentSystem. "
        "Default 1 (sequential). Env: BENCHMARK_JOBS.",
    )
    p.add_argument(
        "--api-max-concurrency",
        type=int,
        default=None,
        metavar="N",
        help="Max concurrent sub-agent API calls inside one forward run (global semaphore in samplers). "
        "If omitted and MAS_API_MAX_CONCURRENCY / MAX_CONCURRENT are unset, defaults to 1 "
        "(sub-agents run one after another). Set higher to allow parallel sub-agent calls.",
    )
    p.add_argument(
        "--mas-root",
        type=Path,
        default=None,
        help="Override MAS cache directory for loading (default: {repo}/shared_mas; env MAS_SHARED_MAS_DIR)",
    )
    p.add_argument(
        "--swe-dataset-name",
        default=os.environ.get("SWE_DATASET_NAME") or None,
    )
    p.add_argument("--skip-bcp-grader", action="store_true")
    return p.parse_args(argv)


def _append_eval_samples_jsonl(out_json: Path, summary_row: Dict[str, Any]) -> None:
    """Append-only DyLAN-style JSONL (one object per line) under ``<dataset_dir>/logs/samples.jsonl``."""
    logd = out_json.parent.parent / "logs"
    logd.mkdir(parents=True, exist_ok=True)
    path = logd / "samples.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_row, ensure_ascii=False) + "\n")


def _run_one(
    *,
    idx: int,
    raw_row: Dict[str, Any],
    spec,
    ds_name: str,
    agent: str,
    shared_path: Path,
    out_json: Path,
    stub: _EvalTrainerStub,
    usage_log_path: Path,
    swe_judge: Optional[Path],
    swe_harness_name: Optional[str],
    skip_bcp_grader: bool,
) -> Tuple[int, Optional[float], Dict[str, Any]]:
    sid = stable_id_for_row(spec.key, raw_row, idx)
    cache_file = shared_path / f"{sid}.json"
    record = row_to_parquet_record(spec, raw_row, idx)

    usage_log_path.parent.mkdir(parents=True, exist_ok=True)
    if usage_log_path.exists():
        usage_log_path.unlink()

    log_ctx_tok = set_usage_log_path_context(str(usage_log_path))
    old_max = os.environ.get("MAS_API_MAX_CONCURRENCY")
    os.environ["MAS_API_MAX_CONCURRENCY"] = _resolve_mas_api_max_concurrency(dict(os.environ))

    rc = 0
    pred_text = ""
    mas_payload: Dict[str, Any] = {}
    acc_val: Optional[float] = None
    acc_extra: Optional[Dict[str, Any]] = None
    harmony_mode: str = "error"

    try:
        if not cache_file.is_file():
            raise FileNotFoundError(f"missing shared_mas cache: {cache_file}")
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        code = (cached.get("extracted_code") or "").strip()
        extracted_name = (cached.get("extracted_name") or "").strip()
        extracted_thought = cached.get("extracted_thought")
        # Harmony minimal/medium: orchestrator may emit no Python ``forward`` graph and instead take the
        # direct-answer path (sentinel ``extracted_code`` / ``extracted_name`` == "direct_answer").
        # ``execute_code`` then bypasses AgentSystem.exec and passes ``extracted_thought`` through as the
        # sample result (same as training). SMFR / GPQA / etc. grading still sees that string as ``pred``.
        harmony_direct: Optional[List[Optional[str]]] = None
        harmony_mode = "forward_code"
        if extracted_name == "direct_answer" or code == "direct_answer":
            harmony_mode = "direct_answer"
            harmony_direct = [extracted_thought if extracted_thought is not None else ""]
        elif not code:
            raise RuntimeError("empty extracted_code in shared_mas JSON")

        question = str(record["prompt"])
        gt = (record.get("reward_model") or {}).get("ground_truth", "")
        task_info = stub.processor.build_task_info(question)

        rows = execute_code(stub, [code], [task_info], harmony_direct_answer_texts=harmony_direct)
        r, ok, err, traces = rows[0] if rows else ("", False, "no_result", [])
        pred_text = str(r) if r is not None else ""
        if not ok:
            rc = 1
        if rc == 0 and spec.key == "SMFR" and pred_text.strip():
            pred_text = maybe_augment_smfr_prediction(
                pred_text, question=question, agent_model=agent
            )
        exec_dict = {
            "result": pred_text,
            "success": bool(ok),
            "error": str(err) if err is not None else "",
            "question": question,
            "ground_truth": gt,
            "agent_traces": traces if isinstance(traces, list) else list(traces) if traces else [],
        }
        mas_payload = {
            "predicted_output_text": pred_text,
            "execution_results": [exec_dict],
            "reward_extra_info": {"predicted_answer": [pred_text]},
            "label_ground_truth": gt,
            "shared_mas_source": str(cache_file),
            "orchestrator_completion_text": cached.get("orchestrator_completion_text"),
        }

        if rc == 0:
            try:
                if spec.key == "BCP" and skip_bcp_grader:
                    acc_val = None
                else:
                    acc_val, pred_text, acc_extra = compute_accuracy(
                        ds_name,
                        spec.key,
                        raw_row,
                        mas_payload,
                        swe_judge if spec.key == "SWE" else None,
                        swe_harness_name if spec.key == "SWE" else None,
                        problem_type=spec.problem_type,
                        no_decompose=spec.no_decompose,
                    )
            except Exception as ge:
                acc_val = None
                pred_text = f"{pred_text}\n[grading_error] {ge}"
    except Exception:
        rc = 1
        pred_text = traceback.format_exc()
        mas_payload = {"error": pred_text}

    rollup = load_usage_log(str(usage_log_path))
    usage_totals = rollup.totals()
    usage_detail = {"records": rollup.records, "totals": usage_totals}

    gt_str = str((record.get("reward_model") or {}).get("ground_truth", ""))
    doc: Dict[str, Any] = {
        "status": "completed" if rc == 0 else "failed",
        "returncode": rc,
        "sample_id": sid,
        "dataset": ds_name,
        "benchmark_key": spec.key,
        "index": idx,
        "agent_model": agent,
        "input": record["prompt"],
        "label": raw_row.get("answer") or raw_row.get("patch", ""),
        "ground_truth": gt_str,
        "prediction": pred_text,
        "prompt_tokens": int(usage_totals.get("total_input_tokens", 0) or 0),
        "completion_tokens": int(usage_totals.get("total_output_tokens", 0) or 0),
        "total_cost_usd_sub_agents": float(usage_totals.get("total_cost_usd", 0) or 0),
        "raw_row": raw_row,
        "mas_intermediate": mas_payload,
        "api_usage": usage_detail,
        "orchestrator": {
            "hf_model": spec.orchestrator_hf,
            "problem_type": spec.problem_type,
            "note": "MAS architecture from shared_mas cache; sub-agent token/cost in api_usage.",
        },
        "harmony_execution_mode": harmony_mode if rc == 0 else f"failed:{harmony_mode}",
        "final_output": pred_text,
        "accuracy": acc_val,
    }
    if acc_extra:
        doc.update(acc_extra)
    if spec.key == "SMFR":
        doc["source_split"] = raw_row.get("source_split")
        try:
            doc["smfr_eval"] = smfr_eval_breakdown(pred_text, raw_row)
        except Exception as e:
            doc["smfr_eval"] = {"breakdown_error": str(e)[:500]}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    jsonl_row = {
        "sample_id": sid,
        "dataset": ds_name,
        "benchmark_key": spec.key,
        "index": idx,
        "agent_model": agent,
        "status": doc["status"],
        "accuracy": acc_val,
        "prediction": pred_text,
        "ground_truth": gt_str,
        "prompt_tokens": doc["prompt_tokens"],
        "completion_tokens": doc["completion_tokens"],
        "total_cost_usd_sub_agents": doc["total_cost_usd_sub_agents"],
    }
    if acc_extra:
        jsonl_row.update(acc_extra)
    if spec.key == "SMFR":
        jsonl_row["source_split"] = raw_row.get("source_split")
    _append_eval_samples_jsonl(out_json, jsonl_row)

    reset_usage_log_path_context(log_ctx_tok)
    if old_max is not None:
        os.environ["MAS_API_MAX_CONCURRENCY"] = old_max
    else:
        os.environ.pop("MAS_API_MAX_CONCURRENCY", None)

    return rc, acc_val, usage_totals


async def _eval_samples_parallel(
    *,
    jobs: int,
    row_tasks: List[Tuple[int, Dict[str, Any]]],
    cfg: Any,
    spec,
    ds_name: str,
    agent: str,
    base: Path,
    mas_base: Path,
    swe_harness_name: Optional[str],
    skip_bcp_grader: bool,
) -> None:
    """
    Concurrent sample evaluation: one ``AgentSystem`` per worker (no shared mutable agent state).
    ``asyncio.to_thread`` runs sync ``_run_one``; usage log path uses ``ContextVar`` in ``record_completion``.
    ``row_tasks`` is a list of ``(jsonl_index, raw_row)`` (only samples not already cached).
    """
    if not row_tasks:
        return
    n_workers = max(1, min(int(jobs), len(row_tasks)))
    stubs = [_init_eval_stub(cfg) for _ in range(n_workers)]

    queue: asyncio.Queue = asyncio.Queue()
    for idx, raw_row in row_tasks:
        queue.put_nowait((idx, raw_row))
    for _ in range(n_workers):
        queue.put_nowait(None)

    out_dir = dataset_result_dir(base, agent, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    swe_judge = out_dir / "swe_judge_workspace"
    shared_samples = mas_base / spec.result_dir_name / "samples"

    async def worker(wid: int) -> None:
        stub = stubs[wid]
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            idx, raw_row = item
            sid = stable_id_for_row(spec.key, raw_row, idx)
            print(
                f"eval_mas_shared: start parallel sample idx={idx} sample_id={sid} agent={agent}",
                file=sys.stderr,
                flush=True,
            )
            out_json = final_sample_path(out_dir, sid)
            fd, usage_path_str = tempfile.mkstemp(prefix=f"mas_usage_{sid}_", suffix=".logl")
            os.close(fd)
            usage_path = Path(usage_path_str)
            try:
                try:
                    await asyncio.to_thread(
                        _run_one,
                        idx=idx,
                        raw_row=raw_row,
                        spec=spec,
                        ds_name=ds_name,
                        agent=agent,
                        shared_path=shared_samples,
                        out_json=out_json,
                        stub=stub,
                        usage_log_path=usage_path,
                        swe_judge=swe_judge,
                        swe_harness_name=swe_harness_name,
                        skip_bcp_grader=skip_bcp_grader,
                    )
                finally:
                    try:
                        usage_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            finally:
                queue.task_done()

    await asyncio.gather(*[worker(i) for i in range(n_workers)])


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ensure_aflow_on_syspath()
    # Sub-agent calls share one process-wide asyncio.Semaphore (see get_api_semaphore). Default 1 =
    # strictly sequential sub-agents within each sample unless user sets env or --api-max-concurrency.
    if args.api_max_concurrency is not None:
        os.environ["MAS_API_MAX_CONCURRENCY"] = str(max(1, int(args.api_max_concurrency)))
    elif not any(str(os.environ.get(k, "") or "").strip() for k in ("MAS_API_MAX_CONCURRENCY", "MAX_CONCURRENT")):
        os.environ["MAS_API_MAX_CONCURRENCY"] = "1"
    reset_api_semaphore()

    aflow = args.aflow_data_dir or default_aflow_data_dir()
    base = _results_base(args.run_suffix)
    mas_base = args.mas_root if args.mas_root is not None else shared_mas_eval_read_root()

    rr = orchestrator_root()
    os.environ["PYTHONPATH"] = str(rr) + os.pathsep + os.environ.get("PYTHONPATH", "")

    for agent in args.agent_model:
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
            if not rows:
                continue

            t0 = time.time()
            out_dir = dataset_result_dir(base, agent, spec)
            out_dir.mkdir(parents=True, exist_ok=True)

            pending_rows: List[Tuple[int, Dict[str, Any]]] = []
            cached_n = 0
            for idx, raw_row in enumerate(rows):
                sid = stable_id_for_row(spec.key, raw_row, idx)
                out_json = final_sample_path(out_dir, sid)
                if load_cached(out_json) is not None:
                    cached_n += 1
                    continue
                pending_rows.append((idx, raw_row))
            if cached_n:
                print(
                    f"eval_mas_shared: skip {cached_n} sample(s) with status=completed under {out_dir / 'samples'}",
                    file=sys.stderr,
                )
            print(
                f"eval_mas_shared: run_suffix={args.run_suffix} agent={agent} dataset={ds_name} "
                f"rows={len(rows)} pending={len(pending_rows)} -> {out_dir}",
                file=sys.stderr,
            )

            swe_harness_name = args.swe_dataset_name if args.swe_dataset_name else str(jsonl_path.resolve())
            shared_samples = mas_base / spec.result_dir_name / "samples"

            if pending_rows:
                cfg = load_grpo_config_phase2(spec, agent)
                _ensure_sampler_for_agent(cfg, agent)
                with open_dict(cfg):
                    cfg.azr.mas_r1.agent.model_name = agent

                if int(args.jobs) <= 1 or len(pending_rows) <= 1:
                    stub = _init_eval_stub(cfg)
                    swe_judge = out_dir / "swe_judge_workspace"
                    for ni, (idx, raw_row) in enumerate(pending_rows):
                        sid = stable_id_for_row(spec.key, raw_row, idx)
                        print(
                            f"eval_mas_shared: [{ni + 1}/{len(pending_rows)}] idx={idx} sample_id={sid} "
                            f"agent={agent} (sub-agent calls may take minutes per sample)",
                            file=sys.stderr,
                            flush=True,
                        )
                        out_json = final_sample_path(out_dir, sid)
                        with tempfile.NamedTemporaryFile(prefix=f"mas_usage_{sid}_", suffix=".logl", delete=False) as tf:
                            usage_path = Path(tf.name)
                        try:
                            _run_one(
                                idx=idx,
                                raw_row=raw_row,
                                spec=spec,
                                ds_name=ds_name,
                                agent=agent,
                                shared_path=shared_samples,
                                out_json=out_json,
                                stub=stub,
                                usage_log_path=usage_path,
                                swe_judge=swe_judge,
                                swe_harness_name=swe_harness_name,
                                skip_bcp_grader=args.skip_bcp_grader,
                            )
                        finally:
                            try:
                                usage_path.unlink(missing_ok=True)
                            except OSError:
                                pass
                else:
                    asyncio.run(
                        _eval_samples_parallel(
                            jobs=int(args.jobs),
                            row_tasks=pending_rows,
                            cfg=cfg,
                            spec=spec,
                            ds_name=ds_name,
                            agent=agent,
                            base=base,
                            mas_base=mas_base,
                            swe_harness_name=swe_harness_name,
                            skip_bcp_grader=args.skip_bcp_grader,
                        )
                    )
            else:
                print(
                    f"eval_mas_shared: all {len(rows)} sample(s) already completed; no execution.",
                    file=sys.stderr,
                )
            sp = write_phase2_dataset_summary(
                out_dir=out_dir,
                spec=spec,
                ds_name=ds_name,
                agent=agent,
                run_suffix=args.run_suffix,
                rows=rows,
                jsonl_path=jsonl_path,
                aflow_data_resolved=aflow.resolve(),
                elapsed_sec=time.time() - t0,
                jobs=int(args.jobs),
            )
            print(
                f"eval_mas_shared: done agent={agent} dataset={ds_name} run_suffix={args.run_suffix} summary={sp}",
                file=sys.stderr,
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
