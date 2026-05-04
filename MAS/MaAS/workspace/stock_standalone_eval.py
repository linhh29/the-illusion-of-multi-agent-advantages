# -*- coding: utf-8 -*-
"""
Multi-level STOCKS standalone evaluation (AFlow-aligned scoring).

Saves each result with the full dataset ``problem`` string in the ``input`` field
(no truncation).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from maas.context import Context
from maas.configs.llm_config import LLMConfig
from maas.ext.maas.benchmark.safe_code_executor import SafeCodeExecutor
from maas.ext.maas.benchmark.stock import (
    STOCKS_AFLOW_INSTRUCTION,
    extract_stock_reference_answer,
    parse_stock_model_output,
    stocks_evaluate_direct_answer,
)
from maas.logs import logger

# Avoid interleaving when many STOCKS requests print full inputs concurrently.
_stocks_llm_input_print_lock = threading.Lock()


def _maas_repo_root() -> Path:
    """``workspace/`` 的上一级 → MaAS 仓库根目录（含 ``maas/`` 包）。"""
    return Path(__file__).resolve().parent.parent


def _resolve_under_maas(path_str: str) -> Path:
    """相对路径相对 MaAS 根目录解析，避免依赖当前 shell 的 cwd。"""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return _maas_repo_root() / p


def _merged_buckets_from_jsonl(merged_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """AFlow ``stocks_test.jsonl``：按 ``source_split``（难度 2–6）分桶。"""
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ss = d.get("source_split")
            if ss is None:
                continue
            try:
                lv = int(ss)
            except (TypeError, ValueError):
                continue
            buckets.setdefault(lv, []).append(d)
    return buckets


def _resolve_stock_merged_jsonl(explicit: Optional[str]) -> Optional[Path]:
    """合并版 jsonl：优先参数 / 环境变量，其次 ``mas_eval/AFlow/data/datasets/stocks_test.jsonl``。"""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        for root in (_maas_repo_root(), _maas_repo_root().parent, Path.cwd()):
            c = (root / explicit).resolve()
            if c.is_file():
                return c
        return p if p.exists() else None

    env = os.environ.get("STOCKS_MERGED_JSONL")
    if env:
        pe = Path(env).expanduser()
        if pe.is_file():
            return pe

    cand = _maas_repo_root().parent / "AFlow" / "data" / "datasets" / "stocks_test.jsonl"
    if cand.is_file():
        return cand
    return None


def _evaluate_code_output(
    executor: SafeCodeExecutor, code: str, reference_answer: List[str]
) -> Tuple[bool, bool, bool, str]:
    """Same logic as ``StockBenchmark._evaluate_code_output``."""
    if not reference_answer:
        return False, False, False, ""

    try:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            exec_result = executor.execute(code, inputs={})
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        exec_info = json.dumps(exec_result, ensure_ascii=False, default=str)

        if not exec_result.get("success", False):
            return False, False, True, exec_info

        code_answer = exec_result["result"].get("answer")
        if code_answer is None:
            return False, False, True, exec_info

        if isinstance(code_answer, str):
            if code_answer in reference_answer:
                is_full = len(reference_answer) == 1
                return is_full, True, False, exec_info

        elif isinstance(code_answer, list):
            code_set = set(code_answer)
            ref_set = set(reference_answer)
            if code_set == ref_set:
                return True, False, False, exec_info

        return False, False, False, exec_info

    except Exception as e:
        exec_info = json.dumps({"exception": str(e)}, ensure_ascii=False)
        return False, False, True, exec_info


def _stock_aggregate_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    missing = sum(1 for r in rows if "stock_eval_metrics" not in r)
    direct_full = sum(
        1 for r in rows if r.get("stock_eval_metrics", {}).get("direct_full")
    )
    code_full = sum(1 for r in rows if r.get("stock_eval_metrics", {}).get("code_full"))
    code_failed = sum(
        1 for r in rows if r.get("stock_eval_metrics", {}).get("code_exec_failed")
    )
    code_ok_no_match = sum(
        1
        for r in rows
        if r.get("stock_eval_metrics")
        and not r["stock_eval_metrics"].get("code_full")
        and not r["stock_eval_metrics"].get("code_exec_failed")
    )
    both = sum(
        1
        for r in rows
        if r.get("stock_eval_metrics", {}).get("direct_full")
        and r.get("stock_eval_metrics", {}).get("code_full")
    )
    denom = total - missing if total else 0
    return {
        "total_samples": total,
        "samples_missing_stock_eval_metrics": missing,
        "rates_denominator": denom,
        "count_direct_full": direct_full,
        "count_code_full": code_full,
        "count_code_exec_failed": code_failed,
        "count_code_exec_success_but_answer_not_full": code_ok_no_match,
        "count_samples_direct_and_code_full": both,
        "rate_direct_full": direct_full / denom if denom else 0.0,
        "rate_code_full": code_full / denom if denom else 0.0,
        "rate_code_exec_failed": code_failed / denom if denom else 0.0,
        "scoring_note": "score==1.0 iff direct_full (AFlow StocksBenchmark); code_full is logged separately.",
        "interpretation": (
            "count_direct_full and count_code_full are independent booleans per sample; they do NOT sum to total_samples. "
            "Along the code path, count_code_full + count_code_exec_failed + count_code_exec_success_but_answer_not_full "
            "should equal total_samples when each row has stock_eval_metrics (and typical non-empty reference answers): "
            "sandbox match, sandbox failure, or sandbox OK but structured answer did not match the reference."
        ),
    }


async def _call_stocks_llm(llm: Any, input_text: str) -> str:
    if hasattr(llm, "acall_reverse_answer_code"):
        return await llm.acall_reverse_answer_code(input_text)
    return await llm.aask(input_text)


def _stocks_row_llm_failed(
    full_problem: str,
    ref: List[str],
    level: int,
    method_label: str,
    exc: BaseException,
) -> Dict[str, Any]:
    """One sample failed (e.g. API 400); score 0 for direct and code; others keep running."""
    err_s = f"{type(exc).__name__}: {exc}"
    logger.warning(f"STOCKS sample failed, scored 0 (level={level}): {err_s}")
    logger.opt(exception=exc).debug("Full traceback for failed STOCKS sample")
    ed: Dict[str, Any] = {
        "llm_or_parse_error": err_s,
        "direct_full": False,
        "direct_partial_count": 0,
        "code_full": False,
        "code_partial": False,
        "code_exec_info": "",
    }
    return {
        "input": full_problem,
        "prediction": f"[LLM/pipeline error] {err_s}",
        "ground_truth": json.dumps(ref, ensure_ascii=False),
        "score": 0.0,
        "cost": 0.0,
        "method": method_label,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "eval_detail": json.dumps(ed, ensure_ascii=False),
        "parsed_model_answer": "",
        "parsed_code": "",
        "stock_eval_metrics": {
            "direct_full": False,
            "direct_partial_count": 0,
            "code_full": False,
            "code_partial": False,
            "code_exec_failed": False,
            "code_exec_info": err_s,
        },
        "level": level,
    }


async def _evaluate_one(
    llm: Any,
    problem: Dict[str, Any],
    executor: SafeCodeExecutor,
    level: int,
    method_label: str,
) -> Dict[str, Any]:
    full_problem = problem.get("problem", "") or ""
    input_text = full_problem + STOCKS_AFLOW_INSTRUCTION
    ref = extract_stock_reference_answer(problem)

    cm = getattr(llm, "cost_manager", None)
    c0 = cm.get_costs() if cm else None
    pt0 = c0.total_prompt_tokens if c0 else 0
    ct0 = c0.total_completion_tokens if c0 else 0

    with _stocks_llm_input_print_lock:
        print(
            f"[STOCKS LLM] level={level} method={method_label} input_chars={len(input_text)} ref={ref}",
            flush=True,
        )

    try:
        output = await _call_stocks_llm(llm, input_text)

        c1 = cm.get_costs() if cm else None
        d_pt = (c1.total_prompt_tokens - pt0) if c1 else 0
        d_ct = (c1.total_completion_tokens - ct0) if c1 else 0
        row_cost = (c1.total_cost - c0.total_cost) if (c0 and c1) else 0.0

        raw_str, model_answer, code_str = parse_stock_model_output(output)
        direct_full, direct_partial = stocks_evaluate_direct_answer(model_answer, ref)
        code_full, code_partial, code_failed, code_info = _evaluate_code_output(
            executor, code_str, ref
        )
        score = 1.0 if direct_full else 0.0

        eval_details = {
            "direct_full": direct_full,
            "direct_partial_count": direct_partial,
            "code_full": code_full,
            "code_partial": code_partial,
            "code_exec_info": code_info,
        }

        pred_for_log = raw_str if raw_str is not None else str(output)

        return {
            "input": full_problem,
            "prediction": pred_for_log,
            "ground_truth": json.dumps(ref, ensure_ascii=False),
            "score": score,
            "cost": row_cost,
            "method": method_label,
            "usage": {
                "prompt_tokens": d_pt,
                "completion_tokens": d_ct,
                "total_tokens": d_pt + d_ct,
            },
            "eval_detail": json.dumps(eval_details, ensure_ascii=False),
            "parsed_model_answer": str(model_answer),
            "parsed_code": code_str,
            "stock_eval_metrics": {
                "direct_full": direct_full,
                "direct_partial_count": direct_partial,
                "code_full": code_full,
                "code_partial": code_partial,
                "code_exec_failed": code_failed,
                "code_exec_info": code_info,
            },
            "level": level,
        }
    except Exception as exc:
        return _stocks_row_llm_failed(full_problem, ref, level, method_label, exc)


async def _evaluate_one_cot_sc(
    llm: Any,
    problem: Dict[str, Any],
    executor: SafeCodeExecutor,
    level: int,
    num_samples: int,
) -> Dict[str, Any]:
    """Multiple STOCKS calls; self-consistency by majority on ``parsed_model_answer``."""
    full_problem = problem.get("problem", "") or ""
    ref = extract_stock_reference_answer(problem)
    partial: List[Dict[str, Any]] = []
    for _ in range(max(1, num_samples)):
        partial.append(await _evaluate_one(llm, problem, executor, level, "CoT"))

    names = [x.get("parsed_model_answer") or "" for x in partial]
    votes = Counter(names)
    winner = votes.most_common(1)[0][0] if votes else (names[-1] if names else "")
    direct_full, _ = stocks_evaluate_direct_answer(winner, ref)
    last = partial[-1]
    score = 1.0 if direct_full else 0.0

    return {
        "input": full_problem,
        "prediction": json.dumps(
            {"ensemble_samples": [x["prediction"] for x in partial]}, ensure_ascii=False
        ),
        "ground_truth": json.dumps(ref, ensure_ascii=False),
        "score": score,
        "cost": sum(x["cost"] for x in partial),
        "method": "CoT-SC",
        "usage": {
            "prompt_tokens": sum(x["usage"]["prompt_tokens"] for x in partial),
            "completion_tokens": sum(x["usage"]["completion_tokens"] for x in partial),
            "total_tokens": sum(x["usage"]["total_tokens"] for x in partial),
        },
        "eval_detail": last["eval_detail"],
        "parsed_model_answer": winner,
        "parsed_code": last.get("parsed_code", ""),
        "stock_eval_metrics": last["stock_eval_metrics"],
        "level": level,
    }


async def run_stock_levels_async(
    *,
    stock_levels: List[int],
    stock_data_dir: str,
    output_path: Optional[str],
    method: str,
    model: Optional[str],
    config_path: str,
    max_concurrent: int,
    num_samples: int,
    limit_per_level: Optional[int],
    stock_merged_jsonl: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    os.environ["MAAS_CONFIG_PATH"] = config_path

    # Relative paths must not depend on shell CWD: always under MaAS repo root (same as stock_data_dir).
    out_file: Optional[Path] = None
    if output_path:
        out_file = _resolve_under_maas(output_path)
        logger.info(f"STOCKS results will be written to (resolved): {out_file}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    llm_config: Optional[LLMConfig] = None
    if model:
        if "models" in config_dict and model in config_dict["models"]:
            llm_config = LLMConfig(**config_dict["models"][model])
        elif "llm" in config_dict:
            llm_config = LLMConfig(**config_dict["llm"])
            llm_config.model = model
        else:
            raise ValueError(f"No LLM config for model {model}")

    if max_tokens is not None and llm_config is not None:
        llm_config.max_token = max_tokens

    context = Context()
    llm = (
        context.llm_with_cost_manager_from_llm_config(llm_config)
        if llm_config
        else context.llm()
    )

    executor = SafeCodeExecutor(timeout=30)

    base = _resolve_under_maas(stock_data_dir)
    logger.info(f"STOCKS data directory (resolved): {base}")
    all_results: List[Dict[str, Any]] = []
    per_level: Dict[str, Any] = {}
    sem = asyncio.Semaphore(max_concurrent)

    for level in stock_levels:
        data_file = base / f"stock_level_{level}.jsonl"
        if not data_file.is_file():
            logger.warning(f"Skip missing file: {data_file}")
            continue

        problems: List[Dict[str, Any]] = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        if limit_per_level is not None:
            problems = problems[:limit_per_level]

        n_problems = len(problems)
        done_count: List[int] = [0]

        async def run_one(p: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                if method == "cot-sc":
                    row = await _evaluate_one_cot_sc(
                        llm, p, executor, level, num_samples
                    )
                else:
                    row = await _evaluate_one(llm, p, executor, level, "CoT")
            done_count[0] += 1
            metrics = row.get("stock_eval_metrics") or {}
            code_score = 1.0 if metrics.get("code_full") else 0.0
            print(
                f"[STOCKS done] level={level} {done_count[0]}/{n_problems} "
                f"direct_score={row.get('score')} code_score={code_score} method={method}",
                flush=True,
            )
            return row

        level_rows: List[Dict[str, Any]] = list(
            await asyncio.gather(*(run_one(p) for p in problems))
        )
        all_results.extend(level_rows)

        total = len(problems)
        correct = sum(1 for r in level_rows if r["score"] >= 1.0)
        acc = correct / total if total else 0.0
        tot_cost = sum(r["cost"] for r in level_rows)
        tot_pt = sum(r["usage"]["prompt_tokens"] for r in level_rows)
        tot_ct = sum(r["usage"]["completion_tokens"] for r in level_rows)
        per_level[f"level_{level}"] = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "max_concurrent": max_concurrent,
            "total_cost": tot_cost,
            "average_cost": tot_cost / total if total else 0.0,
            "total_usage": {
                "prompt_tokens": tot_pt,
                "completion_tokens": tot_ct,
                "total_tokens": tot_pt + tot_ct,
            },
            "average_usage_per_problem": {
                "prompt_tokens": tot_pt / total if total else 0.0,
                "completion_tokens": tot_ct / total if total else 0.0,
                "total_tokens": (tot_pt + tot_ct) / total if total else 0.0,
            },
            "stock_aggregate": _stock_aggregate_from_rows(level_rows),
        }

    overall_total = len(all_results)
    overall_correct = sum(1 for r in all_results if r["score"] >= 1.0)
    otot_cost = sum(r["cost"] for r in all_results)
    o_pt = sum(r["usage"]["prompt_tokens"] for r in all_results)
    o_ct = sum(r["usage"]["completion_tokens"] for r in all_results)

    out: Dict[str, Any] = {
        "method": method,
        "stock_eval": "aflow_stocks_benchmark",
        "task_type": "stock",
        "max_concurrent": max_concurrent,
        "levels": stock_levels,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "per_level": per_level,
        "overall": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_correct / overall_total if overall_total else 0.0,
            "total_cost": otot_cost,
            "average_cost": otot_cost / overall_total if overall_total else 0.0,
            "total_usage": {
                "prompt_tokens": o_pt,
                "completion_tokens": o_ct,
                "total_tokens": o_pt + o_ct,
            },
            "average_usage_per_problem": {
                "prompt_tokens": o_pt / overall_total if overall_total else 0.0,
                "completion_tokens": o_ct / overall_total if overall_total else 0.0,
                "total_tokens": (o_pt + o_ct) / overall_total if overall_total else 0.0,
            },
            "stock_aggregate": _stock_aggregate_from_rows(all_results),
            "cost_attribution": "batch_equal_split",
        },
        "results": all_results,
    }

    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_file.with_suffix(out_file.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        tmp_path.replace(out_file)
        logger.info(f"Stock results saved to {out_file}")

    return out


def run_stock_levels(**kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(run_stock_levels_async(**kwargs))
