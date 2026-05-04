"""
Optional per-sample JSON export during MAS-R1 validation (benchmark_eval harness).

Set trainer.mas_export_dir to a writable directory. Optionally set env MAS_BENCHMARK_SAMPLE_ID
for a single-file name {id}.json; otherwise files are batch{batch_idx}_idx{i}.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def export_mas_validation_samples(
    *,
    export_dir: str,
    tokenizer,
    test_batch,
    reward_extra_info: Dict[str, Any],
    sample_outputs: Optional[List[str]],
    batch_idx: int,
    global_steps: int,
) -> None:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = len(test_batch)
    single_id = os.environ.get("MAS_BENCHMARK_SAMPLE_ID", "").strip()

    for i in range(n):
        row = test_batch[i]
        input_ids = row.batch["input_ids"]
        if hasattr(input_ids, "tolist"):
            ids = input_ids.tolist()
        else:
            ids = list(input_ids)
        prompt_text = tokenizer.decode(ids, skip_special_tokens=True)

        rm = row.non_tensor_batch.get("reward_model", {}) or {}
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", None)
        else:
            gt = None

        exec_results = row.non_tensor_batch.get("execution_results", None)
        exec_stats = row.non_tensor_batch.get("execution_stats", None)
        question = row.non_tensor_batch.get("question", None)
        extra = {}
        for key in reward_extra_info:
            vals = reward_extra_info[key]
            if isinstance(vals, list) and i < len(vals):
                extra[key] = vals[i]
            else:
                extra[key] = vals

        pred_out = None
        if sample_outputs is not None and i < len(sample_outputs):
            pred_out = sample_outputs[i]

        record = {
            "global_steps": global_steps,
            "batch_idx": batch_idx,
            "index_in_batch": i,
            "prompt_text": prompt_text,
            "question": _json_safe(question),
            "label_ground_truth": _json_safe(gt),
            "predicted_output_text": pred_out,
            "reward_extra_info": _json_safe(extra),
            "execution_results": _json_safe(exec_results),
            "execution_stats": _json_safe(exec_stats),
        }

        if single_id:
            fname = f"{single_id}.json"
        else:
            bench_id = row.non_tensor_batch.get("mas_benchmark_id", None)
            if bench_id is not None:
                fname = f"{str(bench_id)}.json"
            else:
                fname = f"batch{batch_idx}_idx{i}.json"

        with open(out / fname, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
