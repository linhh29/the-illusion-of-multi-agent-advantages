"""
CLI for MAS-Orchestra benchmark_eval: iterate datasets/samples, cache, summarize.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .aflow_bridge import ensure_aflow_on_syspath
from .datasets_aflow import DATASETS, DatasetSpec, load_jsonl, row_to_parquet_record, stable_id_for_row
from .phase2_summary import write_phase2_dataset_summary
from .pricing import UsageRollup, get_price, load_usage_log
from .runner import run_one_sample
from .scoring import compute_accuracy, stocks_eval_breakdown
from .stocks_structured_output import maybe_augment_stocks_prediction


def sanitize_agent_dir(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def default_aflow_data_dir() -> Path:
    """
    Directory of AFlow-style ``*_test.jsonl`` files.

    Prefer ``MAS-Orchestra/data/datasets`` when it contains at least one ``*.jsonl``;
    otherwise use sibling ``AFlow/data/datasets`` (same layout as this repo).

    Override with env ``AFLOW_DATASETS_DIR`` (absolute path to the folder that holds ``bcp_test.jsonl``, etc.).
    """
    env = (os.environ.get("AFLOW_DATASETS_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    orch = Path(__file__).resolve().parent.parent
    primary = orch / "data" / "datasets"
    fallback = orch.parent / "AFlow" / "data" / "datasets"
    try:
        if primary.is_dir() and any(primary.glob("*.jsonl")):
            return primary
    except OSError:
        pass
    if fallback.is_dir():
        return fallback
    return primary


def orchestrator_openai_timeout_seconds() -> float:
    """
    HTTP timeout (seconds) for ``AsyncOpenAI`` calls to the orchestrator (vLLM OpenAI-compatible API).

    The OpenAI Python SDK defaults to about 600s read timeout; long BrowseComp+ prompts often need more.
    Default here is **1800** (3× that typical read budget). Override with env ``MAS_ORCHESTRATOR_HTTP_TIMEOUT_SEC``.
    """
    return float(os.environ.get("MAS_ORCHESTRATOR_HTTP_TIMEOUT_SEC", "1800"))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAS-Orchestra benchmark harness")
    p.add_argument(
        "--run-suffix",
        default=os.environ.get("RUN_SUFFIX", "run1"),
        help="e.g. run1 -> results_run1",
    )
    p.add_argument(
        "--agent-model",
        default=os.environ.get("AGENT_MODEL", "gpt-4o"),
        choices=["gpt-4o", "gpt-5", "openai/gpt-oss-120b", "gpt-oss-120b"],
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=["GPQA", "HLEMATH", "SWE-Bench-Lite", "BrowseComp+", "STOCKS"],
        help="Subset of dataset keys",
    )
    p.add_argument("--aflow-data-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Max samples per dataset (debug)")
    p.add_argument("--n-gpus", type=int, default=int(os.environ.get("N_GPUS", "1")))
    p.add_argument("--tp-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    p.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        metavar="N",
        help="Physical GPU index for the orchestrator subprocess: sets CUDA_VISIBLE_DEVICES=N. "
        "If unset, uses MAS_GPU_ID / BENCHMARK_GPU_ID when CUDA_VISIBLE_DEVICES is not already set. "
        "For tensor parallel across multiple GPUs, set CUDA_VISIBLE_DEVICES=0,1 (etc.) yourself and omit --gpu-id.",
    )
    p.add_argument(
        "--swe-dataset-name",
        default=os.environ.get("SWE_DATASET_NAME") or None,
        help="Passed to swebench harness as dataset_name (default: absolute path to swe_test.jsonl under --aflow-data-dir, same file as row source)",
    )
    p.add_argument("--skip-bcp-grader", action="store_true", help="Skip LLM-as-judge for BrowseComp+")
    p.add_argument("--dry-run", action="store_true", help="Print actions only")
    p.add_argument(
        "--orchestrator-openai-base",
        default=os.environ.get("MAS_ORCHESTRATOR_OPENAI_BASE"),
        metavar="URL",
        help="OpenAI-compatible API base for the Harmony orchestrator (e.g. http://127.0.0.1:8000/v1). "
        "If set, skips in-process vLLM; run `vllm serve ...` separately.",
    )
    p.add_argument(
        "--orchestrator-model",
        default=os.environ.get("MAS_ORCHESTRATOR_MODEL"),
        metavar="ID",
        help="Model id on the orchestrator server (default: first model from GET /v1/models).",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=int(os.environ.get("BENCHMARK_JOBS", "1")),
        metavar="N",
        help="Max concurrent fresh samples (ThreadPoolExecutor). Default 1 (sequential). "
        "Use >1 with external orchestrator HTTP (MAS_ORCHESTRATOR_OPENAI_BASE); "
        "keep 1 if each subprocess loads vLLM on GPU (OOM / Ray port risk). "
        "SWE-Bench-Lite: prefer 1 (shared harness). Env: BENCHMARK_JOBS.",
    )
    return p.parse_args(argv)


def results_base(run_suffix: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / f"results_{run_suffix}"


def dataset_result_dir(base: Path, agent: str, ds: DatasetSpec) -> Path:
    return base / sanitize_agent_dir(agent) / ds.result_dir_name


def final_sample_path(out_dir: Path, sample_id: str) -> Path:
    return out_dir / "samples" / f"{sample_id}.json"


def _fmt_gpqa_style_summary(
    *,
    row_correct: List[Optional[bool]],
    mean_acc: float,
    tok_in: int,
    tok_out: int,
    n_api_calls: int,
    cost_sum: float,
    agent: str,
) -> List[str]:
    """DyLAN gpqa_test_cot.txt-style evaluation + API/cost block (human-readable)."""
    flags = [x for x in row_correct if x is not None]
    line1 = f"{flags} {mean_acc:.16f}" if flags else f"[] {mean_acc:.16f}"
    n_graded = len(flags)
    correct_n = sum(1 for x in flags if x)
    wrong_n = n_graded - correct_n
    total_tokens = tok_in + tok_out
    avg_calls = (n_api_calls / n_graded) if n_graded else 0.0
    avg_tok = (total_tokens / n_graded) if n_graded else 0.0
    cost_in = (tok_in / 1000.0) * get_price(agent, "input")
    cost_out = (tok_out / 1000.0) * get_price(agent, "output")
    cost_per_q = (cost_sum / n_graded) if n_graded else 0.0
    lines = [
        line1,
        f"{n_graded} {mean_acc:.6f}" if n_graded else f"0 {mean_acc:.6f}",
        str(tok_in),
        str(tok_out),
        f"Total cost: ${cost_sum:.6f}",
        f"Input cost (est. from tokens×{agent} price): ${cost_in:.6f}",
        f"Output cost (est. from tokens×{agent} price): ${cost_out:.6f}",
        f"Cost per graded question: ${cost_per_q:.6f}",
        "",
        "============================================================",
        "FINAL EVALUATION RESULTS",
        "============================================================",
        f"Total rows (jsonl): {len(row_correct)}",
        f"Graded (accuracy defined): {n_graded}",
        f"Correct answers: {correct_n}",
        f"Wrong answers: {wrong_n}",
        f"Accuracy: {correct_n}/{n_graded} = {mean_acc:.4f} ({100.0 * mean_acc:.2f}%)" if n_graded else "Accuracy: n/a (no graded samples)",
        "",
        "API Usage (sub-agent / OpenAI usage log only; orchestrator local not included):",
        f"  Total API calls: {n_api_calls} (avg: {avg_calls:.2f} per graded question)",
        f"  Total tokens: {total_tokens} (prompt: {tok_in}, completion: {tok_out})",
        f"  Avg tokens per graded question: {avg_tok:.1f}",
        "",
        "Cost Summary:",
        f"  Total cost (logged): ${cost_sum:.4f}",
        f"  Input / output split (estimated from token totals): ${cost_in:.4f} / ${cost_out:.4f}",
        f"  Cost per graded question: ${cost_per_q:.6f}",
        "============================================================",
    ]
    return lines


def load_cached(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("status") == "completed":
            return d
    except (json.JSONDecodeError, OSError):
        return None
    return None


def _run_one_fresh_sample(
    *,
    idx: int,
    raw_row: Dict[str, Any],
    spec: DatasetSpec,
    ds_name: str,
    agent: str,
    out_dir: Path,
    jsonl_path: Path,
    args: argparse.Namespace,
    swe_harness_name: str,
    swe_judge: Path,
) -> Dict[str, Any]:
    """
    Run subprocess benchmark for one row, write samples/{id}.json.
    Returns dict with idx, acc_val, usage_totals for aggregation (usage_totals always present).
    """
    sid = stable_id_for_row(spec.key, raw_row, idx)
    out_json = final_sample_path(out_dir, sid)
    record = row_to_parquet_record(spec, raw_row, idx)
    try:
        with tempfile.TemporaryDirectory(prefix=f"masbench_{sid}_") as td:
            tmp = Path(td)
            exp = tmp / "export"
            usage_log = tmp / "usage.logl"
            rc = run_one_sample(
                record=record,
                dataset_spec=spec,
                agent_model=agent,
                tmp_dir=tmp,
                export_dir=exp,
                usage_log_path=usage_log,
                sample_id=sid,
                n_gpus=args.n_gpus,
                tp_size=args.tp_size,
                gpu_id=args.gpu_id,
                orchestrator_openai_base=args.orchestrator_openai_base,
                orchestrator_model=args.orchestrator_model,
            )

            export_file = exp / f"{sid}.json"
            mas_payload: Dict[str, Any] = {}
            if export_file.is_file():
                with open(export_file, "r", encoding="utf-8") as f:
                    mas_payload = json.load(f)
            else:
                mas_payload = {"error": "missing_mas_export", "returncode": rc}

            rollup = load_usage_log(str(usage_log))
            usage_totals = rollup.totals()
            usage_detail = {"records": rollup.records, "totals": usage_totals}

            acc_val: Optional[float] = None
            acc_extra: Optional[Dict[str, Any]] = None
            pred_text = mas_payload.get("predicted_output_text") or ""
            if rc == 0 and spec.key == "STOCKS" and pred_text.strip():
                pred_text = maybe_augment_stocks_prediction(
                    pred_text,
                    question=str(record["prompt"]),
                    agent_model=agent,
                )
                mas_payload["predicted_output_text"] = pred_text
                mas_payload.setdefault("reward_extra_info", {})["predicted_answer"] = [pred_text]
            try:
                if spec.key == "BCP" and args.skip_bcp_grader:
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
            except Exception as ex:
                acc_val = None
                pred_text = pred_text or str(ex)

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
                    "note": "Sub-agent token/cost in api_usage; orchestrator 7B local cost not estimated here.",
                },
                "final_output": pred_text,
                "accuracy": acc_val,
            }
            if acc_extra:
                doc.update(acc_extra)
            if spec.key == "STOCKS":
                doc["source_split"] = raw_row.get("source_split")
                try:
                    doc["stocks_eval"] = stocks_eval_breakdown(pred_text, raw_row)
                except Exception as e:
                    doc["stocks_eval"] = {"breakdown_error": str(e)[:500]}
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

            return {
                "idx": idx,
                "acc_val": acc_val,
                "usage_totals": usage_totals,
            }
    except Exception:
        err = traceback.format_exc()
        doc = {
            "status": "failed",
            "returncode": -1,
            "sample_id": sid,
            "dataset": ds_name,
            "benchmark_key": spec.key,
            "index": idx,
            "agent_model": agent,
            "error": err,
            "accuracy": None,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        return {
            "idx": idx,
            "acc_val": None,
            "usage_totals": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "num_api_calls": 0,
            },
        }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ensure_aflow_on_syspath()
    aflow_data = args.aflow_data_dir or default_aflow_data_dir()
    base = results_base(args.run_suffix)
    agent = args.agent_model

    summary_lines: List[str] = []

    for ds_name in args.datasets:
        t0 = time.time()
        if ds_name not in DATASETS:
            print(f"Unknown dataset {ds_name}, skip", file=sys.stderr)
            continue
        spec = DATASETS[ds_name]
        jsonl_path = aflow_data / spec.jsonl_name
        if not jsonl_path.is_file():
            print(f"Missing data file: {jsonl_path}", file=sys.stderr)
            continue

        rows = load_jsonl(jsonl_path)
        if args.limit is not None:
            rows = rows[: args.limit]

        out_dir = dataset_result_dir(base, agent, spec)
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        swe_judge = out_dir / "swe_judge_workspace"

        row_correct: List[Optional[bool]] = [None] * len(rows)
        acc_sum = 0.0
        acc_cnt = 0
        cost_sum = 0.0
        tok_in_sum = 0
        tok_out_sum = 0
        n_api_calls_sum = 0
        cached_cnt = 0
        fresh_cnt = 0

        swe_harness_name = args.swe_dataset_name if args.swe_dataset_name else str(jsonl_path.resolve())
        pending_fresh: List[Tuple[int, Dict[str, Any]]] = []

        for idx, raw_row in enumerate(rows):
            sid = stable_id_for_row(spec.key, raw_row, idx)
            out_json = final_sample_path(out_dir, sid)
            cached = load_cached(out_json)
            if cached is not None:
                cached_cnt += 1
                ca = cached.get("accuracy")
                if ca is not None:
                    acc_sum += float(ca)
                    acc_cnt += 1
                    row_correct[idx] = float(ca) >= 0.5
                tot = cached.get("api_usage", {}).get("totals", {})
                cost_sum += float(tot.get("total_cost_usd", 0) or 0)
                tok_in_sum += int(tot.get("total_input_tokens", 0) or 0)
                tok_out_sum += int(tot.get("total_output_tokens", 0) or 0)
                n_api_calls_sum += int(tot.get("num_api_calls", 0) or 0)
                continue

            if args.dry_run:
                print(f"Would run {ds_name} idx={idx} id={sid}")
                continue

            pending_fresh.append((idx, raw_row))

        def _merge_fresh(fr: Dict[str, Any]) -> None:
            nonlocal acc_sum, acc_cnt, cost_sum, tok_in_sum, tok_out_sum, n_api_calls_sum, fresh_cnt
            idx_i = int(fr["idx"])
            acc_v = fr.get("acc_val")
            ut = fr.get("usage_totals") or {}
            if acc_v is not None:
                acc_sum += float(acc_v)
                acc_cnt += 1
                row_correct[idx_i] = float(acc_v) >= 0.5
            cost_sum += float(ut.get("total_cost_usd", 0) or 0)
            tok_in_sum += int(ut.get("total_input_tokens", 0) or 0)
            tok_out_sum += int(ut.get("total_output_tokens", 0) or 0)
            n_api_calls_sum += int(ut.get("num_api_calls", 0) or 0)
            fresh_cnt += 1

        jobs = max(1, int(args.jobs))
        if pending_fresh:
            if jobs <= 1:
                for idx, raw_row in pending_fresh:
                    fr = _run_one_fresh_sample(
                        idx=idx,
                        raw_row=raw_row,
                        spec=spec,
                        ds_name=ds_name,
                        agent=agent,
                        out_dir=out_dir,
                        jsonl_path=jsonl_path,
                        args=args,
                        swe_harness_name=swe_harness_name,
                        swe_judge=swe_judge,
                    )
                    _merge_fresh(fr)
            else:
                n_workers = min(jobs, len(pending_fresh))

                def _task(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
                    idx, raw_row = item
                    return _run_one_fresh_sample(
                        idx=idx,
                        raw_row=raw_row,
                        spec=spec,
                        ds_name=ds_name,
                        agent=agent,
                        out_dir=out_dir,
                        jsonl_path=jsonl_path,
                        args=args,
                        swe_harness_name=swe_harness_name,
                        swe_judge=swe_judge,
                    )

                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [ex.submit(_task, item) for item in pending_fresh]
                    for fut in as_completed(futures):
                        _merge_fresh(fut.result())

        elapsed = time.time() - t0
        mean_acc = acc_sum / acc_cnt if acc_cnt else 0.0
        extra_h = [
            f"samples_cached: {cached_cnt}",
            f"samples_fresh: {fresh_cnt}",
            f"mean_accuracy (where defined, pre-scan): {mean_acc:.6f}",
        ]
        if spec.key == "SWE":
            extra_h.append(f"swe_harness_dataset_name: {args.swe_dataset_name or str(jsonl_path.resolve())}")
        summary_path = write_phase2_dataset_summary(
            out_dir=out_dir,
            spec=spec,
            ds_name=ds_name,
            agent=agent,
            run_suffix=args.run_suffix,
            rows=rows,
            jsonl_path=jsonl_path,
            aflow_data_resolved=aflow_data.resolve(),
            elapsed_sec=elapsed,
            jobs=int(args.jobs),
            phase_tag="benchmark_eval.cli (subprocess + Ray)",
            extra_header_lines=extra_h,
        )
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_lines.extend(f.read().splitlines() + [""])

    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
