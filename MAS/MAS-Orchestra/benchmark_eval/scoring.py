"""
Post-hoc accuracy using AFlow benchmark helpers where possible.

Harmony orchestrator outputs are XML-structured. Post-hoc parsing imports the
same Python modules as training (paths from repo root ``MAS-Orchestra/``):
GPQA / HLEMATH / SWE-Bench / BrowseComp+ / STOCKS →
``mas_r1_reasoner/rewards/utils/harmony_parser/minimal.py``; IGSM mode uses
``mas_r1_reasoner/rewards/utils/harmony_parser/medium_igsm.py`` when
``global_use_igsm_prompt`` is set. Routing matches
``mas_r1_reasoner/rewards/utils/harmony_parser/__init__.py``. When
``predicted_answer`` is missing, prefer ``execution_results``, then parse raw
``predicted_output_text`` for a direct ``<answer>`` (agent graphs still require
execution).
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .aflow_bridge import ensure_aflow_on_syspath, get_aflow_root
from .datasets_aflow import _stocks_reference_names
from .swe_xml import extract_xml


def score_gpqa(prediction: str, ground_truth: str) -> Tuple[float, str]:
    ensure_aflow_on_syspath()
    from benchmarks.gpqa import GPQABenchmark

    b = GPQABenchmark("GPQA", "", "")
    return b.calculate_score(ground_truth, prediction)


def score_stocks(prediction: str, reference_names: List[str]) -> Tuple[float, str]:
    """
    Aligns with ``AFlow/benchmarks/stocks.py`` ``evaluate_problem`` aggregate score:
    1.0 if direct-answer evaluation is a full match, else 0.0.
    Uses ``StocksBenchmark._parse_model_output`` and ``_evaluate_direct_answer``.
    """
    ensure_aflow_on_syspath()
    from benchmarks.stocks import StocksBenchmark

    b = StocksBenchmark("STOCKS", "", "")
    ref = [str(x) for x in (reference_names or [])]
    _, model_answer, _ = b._parse_model_output(prediction)
    direct_full, _ = b._evaluate_direct_answer(model_answer, ref)
    return (1.0 if direct_full else 0.0), prediction


def stocks_eval_breakdown(prediction: str, raw_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per-sample STOCKS metrics aligned with DyLAN ``cot_stocks.py`` / AFlow ``StocksBenchmark``:
    direct_full, direct_partial_count, code_full, code_partial, code_failed, score (direct-full only).
    """
    ensure_aflow_on_syspath()
    from benchmarks.stocks import StocksBenchmark

    ref = _stocks_reference_names(raw_row)
    try:
        b = StocksBenchmark("STOCKS", "", "")
        _, model_answer, code = b._parse_model_output(prediction)
        direct_full, direct_partial_count = b._evaluate_direct_answer(model_answer, ref)
        code_full, code_partial, code_failed, exec_info = b._evaluate_code_output(code, ref)
        info = str(exec_info) if len(str(exec_info)) <= 4000 else str(exec_info)[:4000] + "…"
        score = 1.0 if direct_full else 0.0
        return {
            "direct_full": bool(direct_full),
            "direct_partial_count": int(direct_partial_count),
            "code_full": bool(code_full),
            "code_partial": bool(code_partial),
            "code_failed": bool(code_failed),
            "code_exec_info_truncated": info,
            "score": float(score),
        }
    except Exception as e:
        return {
            "direct_full": False,
            "direct_partial_count": 0,
            "code_full": False,
            "code_partial": False,
            "code_failed": True,
            "code_exec_info_truncated": str(e)[:2000],
            "score": 0.0,
            "breakdown_error": str(e),
        }


def score_hlemath(prediction: str, ground_truth: str) -> Tuple[int, str]:
    ensure_aflow_on_syspath()
    from benchmarks.hlemath import HLEMATHBenchmark

    b = HLEMATHBenchmark("HLEMATH", "", "")
    return b.calculate_score(ground_truth, prediction)


async def score_bcp_async(
    question: str,
    prediction: str,
    ground_truth: str,
) -> Tuple[float, str, Dict[str, Any]]:
    """
    AsyncLLM loads config relative to cwd; vendored AFlow uses ``vendor/aflow_eval`` as cwd.
    On grader failure, returns ``(0.0, prediction, {"error": ...})``.
    Third element is ``AsyncLLM.get_usage_summary()`` when grading succeeds (tokens/cost).
    """
    ensure_aflow_on_syspath()
    aflow_root = get_aflow_root()
    old_cwd = os.getcwd()
    try:
        os.chdir(aflow_root)
        from benchmarks.bcp import BCPBenchmark

        logp = Path(__file__).resolve().parent / "_bcp_grader_logs"
        logp.mkdir(parents=True, exist_ok=True)
        b = BCPBenchmark("BCP", "", str(logp))
        s, pred_out = await b.calculate_score(question, ground_truth, prediction)
        grader_usage: Dict[str, Any] = {}
        if getattr(b, "grader_model", None) is not None and hasattr(
            b.grader_model, "get_usage_summary"
        ):
            grader_usage = b.grader_model.get_usage_summary()
        return float(s), str(pred_out), grader_usage
    except Exception as e:
        print(f"benchmark_eval: BCP grader failed: {e}", file=sys.stderr)
        return 0.0, prediction, {"error": str(e)}
    finally:
        try:
            os.chdir(old_cwd)
        except OSError:
            pass


async def score_swe_async(
    *,
    instance_id: str,
    prediction: str,
    judge_path: Path,
    dataset_name: str,
    code_snippet: str = "",
) -> float:
    """
    Aligns with ``AFlow/benchmarks/swe.py`` ``SWEBenchmark.evaluate_problem`` harness loop:
    ``max_retries = 10``, ``max_time_limit = 600`` wall-clock seconds, ``docker rm`` after each
    ``run_swebench_evaluation`` call, retry on ``asyncio.TimeoutError`` or other exceptions.
    (MAS has no in-process graph generation between retries; the same prediction is re-evaluated.)
    """
    ensure_aflow_on_syspath()
    from benchmarks.swe_utils import run_swebench_evaluation

    judge_path.mkdir(parents=True, exist_ok=True)
    (judge_path / "results").mkdir(parents=True, exist_ok=True)
    (judge_path / "reports").mkdir(parents=True, exist_ok=True)

    max_retries = 10
    max_time_limit = 600
    start_time = time.time()
    last_score = 0.0

    for attempt in range(max_retries):
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time_limit:
            print(
                f"benchmark_eval: SWE evaluation exceeded time limit of {max_time_limit} seconds; "
                f"returning last score {last_score}.",
                file=sys.stderr,
            )
            return float(last_score)
        try:
            last_score = await run_swebench_evaluation(
                str(judge_path) + "/",
                instance_id,
                prediction,
                "",
                "mas",
                code_snippet,
                dataset_name,
            )
            # Mirror benchmarks/swe.py: remove docker container to avoid lock
            container_name = "sweb.eval." + instance_id + "." + instance_id.replace("-", "_") + "__"
            result = subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("Remove Container ----- Directory listing:")
                print(result.stdout)
            else:
                print("Remove Container ----- Error:", result.stderr)

            if float(last_score) >= 1.0:
                return float(last_score)
        except asyncio.TimeoutError:
            print(f"Timeout error on attempt {attempt + 1}/{max_retries}. Retrying...")
            if attempt == max_retries - 1:
                return float(last_score)
            continue
        except Exception as e:
            print(f"Error on attempt {attempt + 1}/{max_retries}. Error: {e}. Retrying...")
            if attempt == max_retries - 1:
                return float(last_score)
            continue

    print("benchmark_eval: All SWE harness retries exhausted.", file=sys.stderr)
    return float(last_score)


def _execution_results_to_text(dataset_key: str, er: Any) -> str:
    """Best-effort final string from ``execution_results`` (same shape as training logs)."""
    if er is None:
        return ""
    if isinstance(er, list) and er:
        return _execution_results_to_text(dataset_key, er[0])
    if not isinstance(er, dict):
        return ""
    if dataset_key == "SWE":
        r = er.get("result") or er.get("code") or ""
        return str(r).strip()
    if er.get("error") and er.get("result") in (None, ""):
        return str(er.get("error", "")).strip()
    r = er.get("result")
    return str(r).strip() if r is not None else ""


# GPQA, HLEMATH, SWE-Bench, BrowseComp+, STOCKS → harmony_parser/minimal.py (DoM Low).
_HARMONY_PROBLEM_TYPE_BY_DATASET_KEY: Dict[str, str] = {
    "GPQA": "harmony_minimal",
    "HLEMATH": "harmony_minimal",
    "SWE": "harmony_minimal",
    "BCP": "harmony_minimal",
    "STOCKS": "harmony_minimal",
}

_NO_DECOMPOSE_BY_DATASET_KEY: Dict[str, bool] = {
    "GPQA": True,
    "HLEMATH": True,
    "SWE": True,
    "BCP": False,
    "STOCKS": True,
}


def _harmony_parse_globals_for_benchmark(
    dataset_key: str,
    problem_type_fallback: str,
    no_decompose_fallback: bool,
) -> Tuple[str, bool]:
    """Parser globals for post-hoc Harmony parse; keys above override fallbacks."""
    pt = _HARMONY_PROBLEM_TYPE_BY_DATASET_KEY.get(dataset_key, problem_type_fallback)
    nd = _NO_DECOMPOSE_BY_DATASET_KEY.get(dataset_key, no_decompose_fallback)
    return pt, nd


def _extract_harmony_code_from_response_training(
    response_text: str,
    validate_python_code,
    logger,
) -> Tuple[str, str, str]:
    """
    Same routing as
    ``mas_r1_reasoner/rewards/utils/harmony_parser/__init__.py``:
    ``extract_harmony_code_from_response`` dispatches to
    ``mas_r1_reasoner/rewards/utils/harmony_parser/minimal.py`` (DoM Low),
    ``mas_r1_reasoner/rewards/utils/harmony_parser/medium.py`` (DoM High), or
    ``mas_r1_reasoner/rewards/utils/harmony_parser/medium_igsm.py`` when
    ``global_use_igsm_prompt`` is true. Caller must set globals (e.g.
    ``global_problem_type``).
    """
    from mas_r1_reasoner.agents.shared_vars import get_global

    problem_type = get_global("global_problem_type")
    use_igsm_prompt = get_global("global_use_igsm_prompt")
    if use_igsm_prompt is None:
        use_igsm_prompt = False

    if problem_type == "harmony_minimal":
        from mas_r1_reasoner.rewards.utils.harmony_parser.minimal import (
            extract_harmony_code_from_response as minimal_extract,
        )

        return minimal_extract(response_text, validate_python_code, logger)
    if problem_type == "harmony_medium":
        if use_igsm_prompt:
            from mas_r1_reasoner.rewards.utils.harmony_parser.medium_igsm import (
                extract_harmony_code_from_response as medium_igsm_extract,
            )

            print("Using IGSM parser")
            return medium_igsm_extract(response_text, validate_python_code, logger)
        from mas_r1_reasoner.rewards.utils.harmony_parser.medium import (
            extract_harmony_code_from_response as medium_extract,
        )

        return medium_extract(response_text, validate_python_code, logger)
    from mas_r1_reasoner.rewards.utils.harmony_parser.minimal import (
        extract_harmony_code_from_response as minimal_extract,
    )

    return minimal_extract(response_text, validate_python_code, logger)


def _harmony_direct_answer_from_raw(
    raw: str,
    problem_type: str,
    no_decompose: bool,
) -> str:
    """
    Run training-identical harmony parsing on raw orchestrator text.
    Returns a non-empty string only when the chosen parser yields
    ``code == 'direct_answer'`` (see ``minimal.py`` / ``medium.py`` in
    ``mas_r1_reasoner/rewards/utils/harmony_parser/``).
    """
    if not raw or not str(raw).strip():
        return ""
    if not problem_type.startswith("harmony_"):
        return ""
    from mas_r1_reasoner.agents.code_sanity import validate_python_code
    from mas_r1_reasoner.agents.shared_vars import get_global, set_global

    keys = ("global_problem_type", "global_no_decompose", "global_use_igsm_prompt")
    saved = {k: get_global(k) for k in keys}
    try:
        set_global("global_problem_type", problem_type)
        set_global("global_no_decompose", bool(no_decompose))
        set_global("global_use_igsm_prompt", False)
        code, _name, thought = _extract_harmony_code_from_response_training(
            str(raw), validate_python_code, None
        )
        if code == "direct_answer":
            return str(thought or "").strip()
    except Exception:
        pass
    finally:
        for k in keys:
            set_global(k, saved[k])
    return ""


def extract_prediction_for_dataset(
    dataset_key: str,
    export_record: Dict[str, Any],
    *,
    problem_type: str = "harmony_minimal",
    no_decompose: bool = True,
) -> str:
    """Best-effort final text from MAS export JSON (reward answer > execution > harmony parse > raw)."""
    extra = export_record.get("reward_extra_info") or {}
    pred = extra.get("predicted_answer")
    if isinstance(pred, list) and pred:
        pred = pred[0]
    if pred is not None and str(pred).strip():
        return str(pred).strip()

    er = export_record.get("execution_results")
    ex = _execution_results_to_text(dataset_key, er)
    if ex:
        return ex

    po = export_record.get("predicted_output_text")
    raw = str(po).strip() if po else ""
    pt, nd = _harmony_parse_globals_for_benchmark(
        dataset_key, problem_type, no_decompose
    )
    parsed = _harmony_direct_answer_from_raw(raw, pt, nd)
    if parsed:
        return parsed
    return raw


def compute_accuracy(
    dataset_name: str,
    dataset_key: str,
    raw_row: Dict[str, Any],
    export_record: Dict[str, Any],
    swe_judge_path: Optional[Path],
    swe_dataset_name: Optional[str],
    problem_type: str = "harmony_minimal",
    no_decompose: bool = True,
) -> Tuple[Optional[float], str, Optional[Dict[str, Any]]]:
    """
    Returns ``(accuracy, prediction_for_scoring, extra)``.
    ``extra`` is set only for BCP: ``{"bcp_grader_usage": ...}`` from the LLM judge.
    """
    pred = extract_prediction_for_dataset(
        dataset_key,
        export_record,
        problem_type=problem_type,
        no_decompose=no_decompose,
    )
    if dataset_key == "GPQA":
        gt = raw_row.get("answer", "")
        s, _ = score_gpqa(pred, gt)
        return float(s), pred, None
    if dataset_key == "HLEMATH":
        gt = raw_row.get("answer", "")
        s, _ = score_hlemath(pred, raw_row.get("answer", ""))
        return float(s), pred, None
    if dataset_key == "STOCKS":
        ref = _stocks_reference_names(raw_row)
        s, p2 = score_stocks(pred, ref)
        return float(s), p2, None
    if dataset_key == "BCP":
        q = raw_row.get("question", "")
        gt = raw_row.get("answer", "")
        s, pred2, gr_usage = asyncio.run(score_bcp_async(q, pred, gt))
        return float(s), pred2, {"bcp_grader_usage": gr_usage}
    if dataset_key == "SWE":
        if swe_judge_path is None or not swe_dataset_name:
            return None, pred, None
        inst = raw_row["instance_id"]
        text = raw_row.get("text", "")
        code_snippet = extract_xml(text, "code").strip()

        async def _run():
            return await score_swe_async(
                instance_id=inst,
                prediction=pred,
                judge_path=swe_judge_path,
                dataset_name=str(swe_dataset_name),
                code_snippet=code_snippet,
            )

        score = asyncio.run(_run())
        return float(score), pred, None
    return None, pred, None
