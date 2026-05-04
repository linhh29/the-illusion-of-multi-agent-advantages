"""
Phase-2 (eval_mas_shared) dataset summaries: human-readable ``summary.txt`` per agent/dataset.

STOCKS: DyLAN-compatible metrics + per-``source_split`` (depth) cost and token totals.
Other benchmarks: same block as ``benchmark_eval.cli`` (GPQA-style accuracy + API/cost).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .datasets_aflow import DatasetSpec, stable_id_for_row


def _load_sample_doc(out_dir: Path, sid: str) -> Optional[Dict[str, Any]]:
    from .cli import final_sample_path
    p = final_sample_path(out_dir, sid)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _pipeline_exception(doc: Dict[str, Any]) -> bool:
    if doc.get("status") != "completed":
        return True
    if int(doc.get("returncode", 0) or 0) != 0:
        return True
    fo = str(doc.get("final_output") or "")
    if "[grading_error]" in fo or "Traceback (most recent call last)" in fo:
        return True
    return False


def write_phase2_dataset_summary(
    *,
    out_dir: Path,
    spec: DatasetSpec,
    ds_name: str,
    agent: str,
    run_suffix: str,
    rows: List[Dict[str, Any]],
    jsonl_path: Path,
    aflow_data_resolved: Path,
    elapsed_sec: float,
    jobs: int,
    phase_tag: str = "eval_mas_shared",
    extra_header_lines: Optional[List[str]] = None,
) -> Path:
    """
    Writes ``out_dir / summary.txt``. Always includes header + evaluation block.
    STOCKS appends DyLAN-style extended metrics and per-depth (``source_split``) cost.
    """
    n = len(rows)
    row_correct: List[Optional[bool]] = [None] * n
    acc_sum = 0.0
    acc_cnt = 0
    cost_sum = 0.0
    bcp_grader_cost_sum = 0.0
    tok_in_sum = 0
    tok_out_sum = 0
    n_api_calls_sum = 0
    completed_cnt = 0
    failed_cnt = 0

    # STOCKS aggregates (filled when spec.key == STOCKS)
    stocks_docs: List[Tuple[int, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]] = []

    for idx, raw_row in enumerate(rows):
        sid = stable_id_for_row(spec.key, raw_row, idx)
        doc = _load_sample_doc(out_dir, sid)
        if doc is None:
            failed_cnt += 1
            row_correct[idx] = None
            if spec.key == "STOCKS":
                stocks_docs.append((idx, raw_row, None, None))
            continue
        completed_cnt += 1
        ca = doc.get("accuracy")
        if ca is not None:
            acc_sum += float(ca)
            acc_cnt += 1
            row_correct[idx] = float(ca) >= 0.5
        tot = (doc.get("api_usage") or {}).get("totals") or {}
        cost_sum += float(tot.get("total_cost_usd", 0) or 0)
        tok_in_sum += int(tot.get("total_input_tokens", 0) or 0)
        tok_out_sum += int(tot.get("total_output_tokens", 0) or 0)
        n_api_calls_sum += int(tot.get("num_api_calls", 0) or 0)
        bgu = doc.get("bcp_grader_usage")
        if isinstance(bgu, dict):
            bcp_grader_cost_sum += float(bgu.get("total_cost", 0) or 0)
        if doc.get("status") != "completed":
            failed_cnt += 1

        if spec.key == "STOCKS":
            sm = doc.get("stocks_eval") if isinstance(doc.get("stocks_eval"), dict) else None
            stocks_docs.append((idx, raw_row, doc, sm))

    mean_acc = acc_sum / acc_cnt if acc_cnt else 0.0

    header = [
        f"dataset: {ds_name}",
        f"benchmark_key: {spec.key}",
        f"agent_model: {agent}",
        f"run_suffix: {run_suffix}",
        f"phase: {phase_tag}",
        f"benchmark_jobs: {jobs}",
        f"aflow_data_dir: {aflow_data_resolved}",
        f"samples_total: {n}",
        f"samples_with_json: {completed_cnt}",
        f"samples_missing_or_failed: {failed_cnt}",
        f"mean_accuracy (where defined): {mean_acc:.6f}",
        f"count_with_accuracy: {acc_cnt}",
        f"approx_sub_agent_cost_usd_sum: {cost_sum:.6f}",
        f"approx_bcp_grader_cost_usd_sum: {bcp_grader_cost_sum:.6f}",
        f"wall_time_sec_dataset: {elapsed_sec:.1f}",
        f"data_jsonl: {jsonl_path}",
    ]
    if spec.key == "SWE":
        header.append("note: SWE harness dataset name is passed separately to eval_mas_shared")
    if extra_header_lines:
        header.extend(extra_header_lines)

    from .cli import _fmt_gpqa_style_summary

    eval_block = _fmt_gpqa_style_summary(
        row_correct=row_correct,
        mean_acc=mean_acc,
        tok_in=tok_in_sum,
        tok_out=tok_out_sum,
        n_api_calls=n_api_calls_sum,
        cost_sum=cost_sum,
        agent=agent,
    )

    lines: List[str] = header + [""] + eval_block

    if spec.key == "STOCKS":
        lines.extend(_stocks_dylan_style_extension(rows, stocks_docs))

    summary_txt = out_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summary_txt


def _stocks_dylan_style_extension(
    rows: List[Dict[str, Any]],
    stocks_docs: List[Tuple[int, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]],
) -> List[str]:
    """DyLAN ``stocks_test_*.txt``-style tail: booleans line, token lines, detailed match counts, per-depth + cost."""
    flags: List[bool] = []
    for idx, raw_row, doc, sm in stocks_docs:
        ca = doc.get("accuracy") if doc else None
        if ca is None:
            flags.append(False)
        else:
            flags.append(float(ca) >= 0.5)

    total_count = len(rows)
    exception_cnt = sum(
        1
        for idx, raw_row, doc, sm in stocks_docs
        if doc is None or _pipeline_exception(doc)
    )

    direct_full_cnt = 0
    direct_partial_cnt = 0
    code_full_cnt = 0
    code_partial_cnt = 0
    code_fail_cnt = 0

    depth_stats: Dict[Any, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "direct_full": 0,
            "direct_partial": 0,
            "code_full": 0,
            "code_partial": 0,
            "code_failed": 0,
            "exception": 0,
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "api_calls": 0,
        }
    )

    for idx, raw_row, doc, sm in stocks_docs:
        depth = raw_row.get("source_split")
        s = depth_stats[depth]
        s["total"] += 1
        if doc is None:
            s["exception"] += 1
            continue
        if _pipeline_exception(doc):
            s["exception"] += 1
        tot = (doc.get("api_usage") or {}).get("totals") or {}
        s["cost_usd"] += float(tot.get("total_cost_usd", 0) or 0)
        s["prompt_tokens"] += int(tot.get("total_input_tokens", 0) or 0)
        s["completion_tokens"] += int(tot.get("total_output_tokens", 0) or 0)
        s["api_calls"] += int(tot.get("num_api_calls", 0) or 0)

        if sm:
            if sm.get("direct_full"):
                direct_full_cnt += 1
                s["direct_full"] += 1
            elif int(sm.get("direct_partial_count") or 0) > 0:
                direct_partial_cnt += 1
                s["direct_partial"] += 1
            if sm.get("code_full"):
                code_full_cnt += 1
                s["code_full"] += 1
            if sm.get("code_partial"):
                code_partial_cnt += 1
                s["code_partial"] += 1
            if sm.get("code_failed"):
                code_fail_cnt += 1
                s["code_failed"] += 1
            ca = doc.get("accuracy")
            if ca is not None and float(ca) >= 0.5:
                s["correct"] += 1
        else:
            # Missing breakdown (older runs): infer minimal stats from accuracy only
            ca = doc.get("accuracy")
            if ca is not None and float(ca) >= 0.5:
                direct_full_cnt += 1
                s["direct_full"] += 1
                s["correct"] += 1

    correct_count = sum(1 for f in flags if f)
    final_accuracy = correct_count / total_count if total_count else 0.0
    direct_full_rate = direct_full_cnt / total_count if total_count else 0.0
    code_full_rate = code_full_cnt / total_count if total_count else 0.0

    t_in = 0
    t_out = 0
    n_api_total = 0
    for idx, raw_row, doc, sm in stocks_docs:
        if not doc:
            continue
        tot = (doc.get("api_usage") or {}).get("totals") or {}
        t_in += int(tot.get("total_input_tokens", 0) or 0)
        t_out += int(tot.get("total_output_tokens", 0) or 0)
        n_api_total += int(tot.get("num_api_calls", 0) or 0)
    total_cost = sum(
        float((doc.get("api_usage") or {}).get("totals", {}).get("total_cost_usd", 0) or 0)
        for idx, raw_row, doc, sm in stocks_docs
        if doc
    )

    out: List[str] = [
        "",
        "============================================================",
        "STOCKS extended (DyLAN / AFlow-aligned)",
        "============================================================",
        f"{flags} {final_accuracy}",
        f"{n_api_total} {n_api_total / total_count if total_count else 0.0}",
        "# importance_matrix: N/A for MAS-Orchestra (DyLAN listwise only; not applicable)",
        str(t_in),
        str(t_out),
    ]
    out.append(f"Total cost: ${total_cost:.6f}")
    out.append(
        f"Accuracy (AFlow metric - direct full only): {correct_count}/{total_count} = {final_accuracy:.4f}"
    )
    out.append(f"Pipeline exceptions: {exception_cnt}/{total_count}")
    out.append(
        f"Direct Full Match: {direct_full_cnt}/{total_count} = {direct_full_rate:.4f}"
    )
    out.append(f"Direct Partial Match: {direct_partial_cnt}/{total_count}")
    out.append(
        f"Code Full Match: {code_full_cnt}/{total_count} = {code_full_rate:.4f}"
    )
    out.append(f"Code Partial Match: {code_partial_cnt}/{total_count}")
    out.append(f"Code Execution Failures: {code_fail_cnt}/{total_count}")

    if depth_stats:
        out.append("")
        out.append("Per-depth metrics (source_split):")
        for depth in sorted(depth_stats.keys(), key=lambda x: (x is None, x)):
            s = depth_stats[depth]
            d_total = s["total"]
            if d_total == 0:
                continue
            d_correct = s["correct"]
            d_cost = s["cost_usd"]
            out.append(f"Depth {depth}:")
            out.append(
                f"  Accuracy (AFlow metric - direct full only): {d_correct}/{d_total} = {d_correct/d_total:.4f}"
            )
            df_r = s["direct_full"] / d_total if d_total else 0.0
            cf_r = s["code_full"] / d_total if d_total else 0.0
            out.append(
                f"  Direct Full Match: {s['direct_full']}/{d_total} = {df_r:.4f}"
            )
            out.append(f"  Direct Partial Match: {s['direct_partial']}/{d_total}")
            out.append(f"  Code Full Match: {s['code_full']}/{d_total} = {cf_r:.4f}")
            out.append(f"  Code Partial Match: {s['code_partial']}/{d_total}")
            out.append(f"  Code Execution Failures: {s['code_failed']}/{d_total}")
            out.append(f"  Pipeline exceptions: {s['exception']}/{d_total}")
            out.append(f"  Sub-agent cost (USD, logged): ${d_cost:.6f}")
            out.append(f"  Prompt tokens (sub-agent): {s['prompt_tokens']}")
            out.append(f"  Completion tokens (sub-agent): {s['completion_tokens']}")
            out.append(f"  API calls: {s['api_calls']}")
            avg_c = d_cost / d_total if d_total else 0.0
            out.append(f"  Avg cost per sample (depth): ${avg_c:.6f}")

    out.append("============================================================")
    return out
