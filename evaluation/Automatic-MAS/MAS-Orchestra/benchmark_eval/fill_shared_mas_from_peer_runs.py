"""
Copy ``samples/{idx}.json`` so every run shares the best available Harmony forward per index.

For each ``idx``, the **source** run is the **first** entry in ``--source-priority`` whose JSON
already passes the quality bar (any of run1/run2/run3 can be the donor—whichever appears first
in the list and qualifies). Every **other** listed run that does **not** yet qualify gets that
source document copied over (full JSON overwrite).

Quality bar matches ``generate_mas_shared_cross_run`` / ``--resume-harmony``:
``status == "completed"`` and real Harmony ``forward`` (not ``direct_answer``).

Runs that already qualify are left unchanged. If no run has a good sample for an index, nothing
is copied for that index.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .datasets_aflow import DATASETS
from .runner import orchestrator_root, shared_mas_root


def _is_real_harmony_forward_code(code: Optional[str]) -> bool:
    if not isinstance(code, str) or not code:
        return False
    s = code.strip()
    if not s or s == "direct_answer":
        return False
    if "def forward" not in s:
        return False
    return True


def _sample_ok_resume_harmony_style(doc: Optional[Dict[str, Any]]) -> bool:
    """Same as a row that ``generate_mas_shared_cross_run`` would skip (no regeneration needed)."""
    if not doc:
        return False
    if doc.get("status") != "completed":
        return False
    return _is_real_harmony_forward_code(doc.get("extracted_code"))


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _sample_path(run_suffix: str, dataset_key: str, idx: int) -> Path:
    spec = DATASETS[dataset_key]
    return shared_mas_root(run_suffix) / spec.result_dir_name / "samples" / f"{idx}.json"


def _collect_indices(run_suffixes: List[str], dataset_key: str) -> Set[int]:
    spec = DATASETS[dataset_key]
    out: Set[int] = set()
    for rs in run_suffixes:
        d = shared_mas_root(rs) / spec.result_dir_name / "samples"
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            if p.stem.isdigit():
                out.add(int(p.stem))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fill missing Harmony code from peer shared_mas runs.")
    p.add_argument("--dataset", default="SWE-Bench-Lite", help="Key in DATASETS")
    p.add_argument(
        "--run-suffixes",
        nargs="+",
        default=["run1", "run2", "run3"],
        metavar="SUFFIX",
        help="E.g. run1 run2 run3",
    )
    p.add_argument(
        "--source-priority",
        nargs="+",
        default=["run1", "run2", "run3"],
        metavar="SUFFIX",
        help="Pick source per idx: first listed run that has a completed valid forward wins; "
        "that file is copied onto other --run-suffixes that do not. Reorder to prefer e.g. run2 first.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies only; do not write files.",
    )
    args = p.parse_args(argv)

    if args.dataset not in DATASETS:
        print(f"Unknown dataset {args.dataset}", file=sys.stderr)
        return 2

    run_suffixes = list(dict.fromkeys(args.run_suffixes))
    source_priority = list(dict.fromkeys(args.source_priority))

    indices = sorted(_collect_indices(run_suffixes, args.dataset))
    if not indices:
        print("No sample JSON files found under given runs.")
        return 0

    copies: List[str] = []
    skipped_no_source = 0
    skipped_all_good = 0

    for idx in indices:
        per_run: Dict[str, Optional[Dict[str, Any]]] = {}
        paths: Dict[str, Path] = {}
        for rs in run_suffixes:
            paths[rs] = _sample_path(rs, args.dataset, idx)
            if paths[rs].is_file():
                per_run[rs] = _load(paths[rs])
            else:
                per_run[rs] = None

        source: Optional[str] = None
        for rs in source_priority:
            if rs not in per_run:
                continue
            doc = per_run[rs]
            if _sample_ok_resume_harmony_style(doc):
                source = rs
                break

        if source is None:
            skipped_no_source += 1
            continue

        doc_src = per_run[source]
        assert doc_src is not None

        need_copy = False
        for rs in run_suffixes:
            if rs == source:
                continue
            doc_t = per_run.get(rs)
            if _sample_ok_resume_harmony_style(doc_t):
                continue
            need_copy = True
            line = f"idx={idx}  {source} -> {rs}  ({paths[rs]})"
            copies.append(line)
            if not args.dry_run:
                paths[rs].parent.mkdir(parents=True, exist_ok=True)
                with open(paths[rs], "w", encoding="utf-8") as f:
                    json.dump(doc_src, f, ensure_ascii=False, indent=2)

        if not need_copy:
            skipped_all_good += 1

    print(f"Dataset: {args.dataset}  runs: {run_suffixes}")
    print(f"Indices scanned: {len(indices)}  (no good source anywhere: {skipped_no_source}, all runs already good: {skipped_all_good})")
    print(f"Copy operations: {len(copies)}" + (" (dry-run)" if args.dry_run else ""))
    for line in copies:
        print(line)
    return 0


if __name__ == "__main__":
    rr = orchestrator_root()
    os.environ["PYTHONPATH"] = str(rr) + os.pathsep + os.environ.get("PYTHONPATH", "")
    raise SystemExit(main())
