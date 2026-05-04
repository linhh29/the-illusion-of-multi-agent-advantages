#!/usr/bin/env python3
"""Preview MaAS stock eval JSON: first N lines or parsed per-level summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def print_head(path: Path, n: int) -> None:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i > n:
                break
            sys.stdout.write(line)


def print_summary(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    meta_keys = ("method", "stock_eval", "task_type", "max_concurrent", "levels", "timestamp")
    print("=== meta ===")
    for k in meta_keys:
        if k in data:
            print(f"  {k}: {data[k]!r}")

    per = data.get("per_level")
    if not isinstance(per, dict):
        print("No per_level object found.", file=sys.stderr)
        return

    print("\n=== per_level ===")
    for key in sorted(per.keys(), key=lambda x: (len(x), x)):
        block = per[key]
        if not isinstance(block, dict):
            continue
        acc = block.get("accuracy")
        tot = block.get("total")
        cor = block.get("correct")
        cost = block.get("total_cost")
        line = f"  {key}: correct={cor}/{tot} accuracy={acc}"
        if cost is not None:
            line += f" total_cost={cost}"
        print(line)
        sa = block.get("stock_aggregate")
        if isinstance(sa, dict):
            df = sa.get("count_direct_full")
            cf = sa.get("count_code_full")
            ef = sa.get("count_code_exec_failed")
            if any(x is not None for x in (df, cf, ef)):
                print(
                    f"    stock_aggregate: direct_full={df} code_full={cf} "
                    f"code_exec_failed={ef}"
                )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        default="results_cot_stock_run3.json",
        help="Path to results JSON (default: results_cot_stock_run2.json in cwd)",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--head",
        type=int,
        metavar="N",
        help="Print first N lines of the file (raw text)",
    )
    g.add_argument(
        "--summary",
        action="store_true",
        help="Parse JSON and print meta + per_level summary",
    )
    args = p.parse_args()
    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    if args.summary:
        print_summary(path)
    else:
        print_head(path, args.head)


if __name__ == "__main__":
    main()
