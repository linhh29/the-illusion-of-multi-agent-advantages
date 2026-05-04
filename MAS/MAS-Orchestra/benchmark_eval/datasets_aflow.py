"""
Load AFlow-style JSONL and build VERL raw parquet rows for MAS-Orchestra.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .aflow_bridge import load_agentless_repair

# Orchestrator checkpoints from the Hub are VERL FSDP layouts under ``actor/`` (sharded
# ``.pt``), not a single ``model.safetensors``. Merge first:
#   python verl/scripts/model_merger.py merge --backend fsdp --local_dir .../actor --target_dir ...-merged
# Then point ``orchestrator_hf`` at the merged folder (standard HF layout).
_MAS_ORCHESTRA_ROOT = Path(__file__).resolve().parents[1]


def _default_checkpoint_dir(dirname: str) -> str:
    return str(_MAS_ORCHESTRA_ROOT / "checkpoints" / dirname)


_HARMONY_LOW = os.environ.get(
    "MAS_ORCHESTRA_HARMONY_LOW",
    _default_checkpoint_dir("harmony-grpo-7b-global-step-180-merged"),
)
_HARMONY_HIGH = os.environ.get(
    "MAS_ORCHESTRA_HARMONY_HIGH",
    _default_checkpoint_dir("harmony-medium-grpo-7b-browse-comp-plus-global-step-140-merged"),
)


def _resolve_orchestrator_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (_MAS_ORCHESTRA_ROOT / p).resolve()

# Names for OpenAI-compatible ``/v1/completions`` ``model`` field — must match ``vllm serve`` (use ``--served-model-name``).
SERVED_MODEL_NAME_HARMONY_LOW = "harmony-grpo-7b-global-step-180-merged"
SERVED_MODEL_NAME_HARMONY_HIGH = "harmony-medium-grpo-7b-browse-comp-plus-global-step-140-merged"

# Must match ``AFlow/benchmarks/stocks.py`` ``Instruction`` (appended to ``problem`` in ``StocksBenchmark.evaluate_problem``).
STOCKS_INSTRUCTION = """
First, solve the problem and explain your reasoning step-by-step. Provide your reasoning and final answer as `analysis` and `answer`. Then write code to solve this problem based on the following instructions.  
To solve this problem, write Python code with a solve() function that returns a dictionary. When your code is executed, solve() should return, for example:{
    "investor_dates": {
        "Alice": ["November 19, 2025", "November 21, 2025"],
        "Bob": ["November 25, 2025"],
        "Charlie": []
    },      
    "comparison": {
        "Alice": "November 19, 2025",
        "Bob": "November 25, 2025",
        "Charlie": None
    },      
    "answer": "Alice"
}

Format:
- "investor_dates": dict mapping each investor name to list of valid dates (strings in "Month Day, Year" format)
- "comparison": dict mapping each investor name to their first valid date or None
- "answer": winning investor's name (string), or list of names if tied, or None if no one achieves target

For ties, return a list:
{"answer": ["Alice", "Bob"]}

Ensure that all required input data is included within the code as needed.
"""


def _stocks_reference_names(row: Dict[str, Any]) -> List[str]:
    """Same reference extraction as ``benchmarks.stocks.StocksBenchmark._extract_reference_answer``."""
    answer = row.get("answer")
    if isinstance(answer, dict):
        ref = answer.get("answer", [])
    else:
        ref = answer
    if ref is None:
        return []
    if isinstance(ref, list):
        return [str(x) for x in ref]
    return [str(ref)]


def orchestrator_served_model_id(orchestrator_hf_path: str) -> str:
    """
    Map ``DatasetSpec.orchestrator_hf`` to the model id clients should pass to the orchestrator server.

    Aligns with local ``vllm serve`` defaults (directory basename) or an explicit ``--served-model-name``
    equal to one of the constants above.
    """
    resolved = str(_resolve_orchestrator_path(orchestrator_hf_path))
    if resolved == str(_resolve_orchestrator_path(_HARMONY_LOW)):
        return SERVED_MODEL_NAME_HARMONY_LOW
    if resolved == str(_resolve_orchestrator_path(_HARMONY_HIGH)):
        return SERVED_MODEL_NAME_HARMONY_HIGH
    return Path(orchestrator_hf_path).name


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # internal key
    result_dir_name: str  # folder under results_runK/model/
    jsonl_name: str  # e.g. gpqa_test.jsonl
    orchestrator_hf: str
    problem_type: str  # harmony_minimal | harmony_medium
    init_archive: List[str]
    no_decompose: bool


DATASETS: Dict[str, DatasetSpec] = {
    "GPQA": DatasetSpec(
        key="GPQA",
        result_dir_name="GPQA",
        jsonl_name="gpqa_test.jsonl",
        orchestrator_hf=_HARMONY_LOW,
        problem_type="harmony_minimal",
        # harmony_minimal prompts index archive[0:4] (COT, COT_SC, Reflexion, LLM_debate); see common.get_prompt
        init_archive=["COT", "COT_SC", "Reflexion", "LLM_debate"],
        no_decompose=False,
    ),
    "HLEMATH": DatasetSpec(
        key="HLEMATH",
        result_dir_name="HLEMATH",
        jsonl_name="hlemath_test.jsonl",
        orchestrator_hf=_HARMONY_LOW,
        problem_type="harmony_minimal",
        init_archive=["COT", "COT_SC", "Reflexion", "LLM_debate"],
        no_decompose=False,
    ),
    "SWE-Bench-Lite": DatasetSpec(
        key="SWE",
        result_dir_name="SWE-Bench-Lite",
        jsonl_name="swe_test.jsonl",
        orchestrator_hf=_HARMONY_LOW,
        problem_type="harmony_minimal",
        init_archive=["COT", "COT_SC", "Reflexion", "LLM_debate"],
        no_decompose=False,
    ),
    "BrowseComp+": DatasetSpec(
        key="BCP",
        result_dir_name="BrowseComp+",
        jsonl_name="bcp_test.jsonl",
        orchestrator_hf=_HARMONY_LOW,
        problem_type="harmony_minimal",
        init_archive=["COT", "COT_SC", "Reflexion", "LLM_debate"],
        no_decompose=False,
    ),
    "STOCKS": DatasetSpec(
        key="STOCKS",
        result_dir_name="STOCKS",
        jsonl_name="stocks_test.jsonl",
        orchestrator_hf=_HARMONY_LOW,
        problem_type="harmony_minimal",
        init_archive=["COT", "COT_SC", "Reflexion", "LLM_debate"],
        no_decompose=False,
    ),
}


def stable_id_for_row(_dataset_key: str, _row: Dict[str, Any], index: int) -> str:
    """
    Sample filename stem under ``shared_mas_*/<dataset>/samples/`` and ``results_*/.../samples/``.

    All registered benchmarks use a fixed-order JSONL; use the **linear index** (same as GPQA:
    ``0``, ``1``, ``2``, …) so paths stay human-readable. Row-specific ids (``instance_id``,
    ``query_id``, etc.) remain in each sample's ``raw_row`` / export JSON for harness use.
    """
    return str(index)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def row_to_parquet_record(
    dataset_spec: DatasetSpec,
    row: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """Single RLHF-style row for raw_data mode."""
    sid = stable_id_for_row(dataset_spec.key, row, index)
    if dataset_spec.key == "SWE":
        # AFlow SWEBenchmark.evaluate_problem: input_text = data["text"] + AGENTLESS_REPAIR
        question = row.get("text", "") + load_agentless_repair()
        gt = row.get("patch", "")
    elif dataset_spec.key == "STOCKS":
        question = row.get("problem", "") + STOCKS_INSTRUCTION
        gt = json.dumps(_stocks_reference_names(row), ensure_ascii=False)
    else:
        question = row.get("question", "")
        gt = row.get("answer", "")
    reward_model: Dict[str, Any] = {"style": "rule", "ground_truth": gt}
    return {
        "prompt": question,
        "reward_model": reward_model,
        "data_source": f"benchmark_{dataset_spec.key.lower()}",
        "mas_benchmark_id": sid,
        "benchmark_index": index,
    }


def write_pair_parquets(tmp_dir: Path, record: Dict[str, Any]) -> tuple[str, str]:
    """Train and val parquet paths (same single-row logic; train needs >=1 batch)."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Writing parquet requires pyarrow (or fastparquet). Install with: pip install pyarrow"
        ) from e
    df = pd.DataFrame([record])
    train_p = tmp_dir / "train.parquet"
    val_p = tmp_dir / "val.parquet"
    df.to_parquet(train_p, index=False)
    df.to_parquet(val_p, index=False)
    return str(train_p), str(val_p)
