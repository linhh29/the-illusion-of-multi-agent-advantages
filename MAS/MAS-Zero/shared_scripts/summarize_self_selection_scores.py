#!/usr/bin/env python3
"""Summarize self-judge selection frequencies from *_score.json files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LABELS = {
    0: "Chain-of-Thought",
    1: "Self-Consistency with Chain-of-Thought",
    2: "Self-Refine (Reflexion)",
    3: "LLM Debate",
    4: "generated n=0",
    5: "generated n=1",
    6: "generated n=2",
    7: "generated n=3",
    8: "generated n=4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively read self-verifier *_score.json files and summarize "
            "selection frequencies for response indices 0..8."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing per-example *_score.json files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_score.json",
        help="Glob pattern to find score files under root. Default: *_score.json",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=8,
        help="Highest valid response index to report. Default: 8",
    )
    parser.add_argument(
        "--group-by",
        choices=["none", "dataset", "parent"],
        default="none",
        help=(
            "Grouping mode. 'dataset' uses the path segment after workflow_search; "
            "'parent' uses the score file's parent directory. Default: none"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help=(
            "Optional override path for one combined summary CSV. By default, "
            "the script writes self_selection_summary.csv under each detected dataset directory."
        ),
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        help=(
            "Optional override path for one combined details CSV. By default, "
            "the script writes self_selection_details.csv under each detected dataset directory."
        ),
    )
    parser.add_argument(
        "--summary-name",
        default="self_selection_summary.csv",
        help="Default summary CSV filename when --csv is not set.",
    )
    parser.add_argument(
        "--details-name",
        default="self_selection_details.csv",
        help="Default details CSV filename when --details-csv is not set.",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=10,
        help="Number of parse/selection errors to print. Default: 10",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_selection(data: Any) -> int:
    if isinstance(data, dict):
        for key in ("selection", "chosen_id", "selected_id", "choice", "chosen"):
            if key in data:
                return parse_selection_value(data[key])
        raise ValueError("missing selection-like key")

    return parse_selection_value(data)


def parse_selection_value(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"boolean selection is not valid: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        match = re.search(r"(?<!\d)[+-]?\d+(?!\d)", text)
        if match:
            return int(match.group(0))
    raise ValueError(f"cannot parse selection value: {value!r}")


def group_name(path: Path, root: Path, mode: str) -> str:
    if mode == "none":
        return "all"
    if mode == "parent":
        return path.parent.name

    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "workflow_search" and idx + 1 < len(parts):
            return parts[idx + 1]

    try:
        return path.relative_to(root).parts[0]
    except Exception:
        return path.parent.name


def dataset_dir(path: Path, root: Path) -> Path:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "workflow_search" and idx + 1 < len(parts):
            return Path(*parts[: idx + 2])
    return root


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(
    counts_by_group: dict[str, Counter[int]],
    parsed_by_group: Counter[str],
    out_of_range_by_group: dict[str, Counter[int]],
    max_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in sorted(parsed_by_group):
        total = parsed_by_group[group]
        row: dict[str, Any] = {"group": group, "total": total}
        for idx in range(max_index + 1):
            count = counts_by_group[group][idx]
            pct = (count / total * 100.0) if total else 0.0
            row[f"idx{idx}_count"] = count
            row[f"idx{idx}_pct"] = f"{pct:.2f}"
        row["out_of_range"] = dict(sorted(out_of_range_by_group[group].items()))
        rows.append(row)
    return rows


def print_summary(rows: list[dict[str, Any]], max_index: int) -> None:
    headers = ["group", "total"] + [f"idx{i}" for i in range(max_index + 1)] + ["out_of_range"]
    print("\t".join(headers))
    for row in rows:
        values = [str(row["group"]), str(row["total"])]
        for idx in range(max_index + 1):
            values.append(f'{row[f"idx{idx}_count"]} ({row[f"idx{idx}_pct"]}%)')
        values.append(str(row["out_of_range"]))
        print("\t".join(values))


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")
    if args.max_index < 0:
        raise SystemExit("--max-index must be non-negative")

    paths = sorted(root.rglob(args.pattern))
    counts_by_group: dict[str, Counter[int]] = defaultdict(Counter)
    out_of_range_by_group: dict[str, Counter[int]] = defaultdict(Counter)
    parsed_by_group: Counter[str] = Counter()
    counts_by_dir_group: dict[Path, dict[str, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    out_of_range_by_dir_group: dict[Path, dict[str, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    parsed_by_dir_group: dict[Path, Counter[str]] = defaultdict(Counter)
    details_by_dir: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    errors: list[tuple[Path, str]] = []
    details: list[dict[str, Any]] = []

    for path in paths:
        try:
            data = load_json(path)
            selection = extract_selection(data)
        except Exception as exc:
            errors.append((path, str(exc)))
            continue

        group = group_name(path, root, args.group_by)
        out_dir = dataset_dir(path, root)
        parsed_by_group[group] += 1
        parsed_by_dir_group[out_dir][group] += 1
        if 0 <= selection <= args.max_index:
            counts_by_group[group][selection] += 1
            counts_by_dir_group[out_dir][group][selection] += 1
        else:
            out_of_range_by_group[group][selection] += 1
            out_of_range_by_dir_group[out_dir][group][selection] += 1

        detail = {
            "group": group,
            "selection": selection,
            "example_id": path.parent.name,
            "path": str(path),
        }
        details.append(detail)
        details_by_dir[out_dir].append(detail)

    summary_rows = build_summary_rows(
        counts_by_group,
        parsed_by_group,
        out_of_range_by_group,
        args.max_index,
    )

    print(f"root: {root}")
    print(f"score_files_found: {len(paths)}")
    print(f"score_files_parsed: {sum(parsed_by_group.values())}")
    print(f"score_files_failed: {len(errors)}")
    print()
    print("index_map:")
    for idx in range(args.max_index + 1):
        print(f"  {idx}: {DEFAULT_LABELS.get(idx, f'index {idx}')}")
    print()
    print_summary(summary_rows, args.max_index)

    if errors and args.show_errors:
        print()
        print(f"errors (showing up to {args.show_errors}):")
        for path, message in errors[: args.show_errors]:
            print(f"  {path}: {message}")

    summary_fieldnames = ["group", "total"]
    for idx in range(args.max_index + 1):
        summary_fieldnames.extend([f"idx{idx}_count", f"idx{idx}_pct"])
    summary_fieldnames.append("out_of_range")

    if args.csv:
        write_csv(args.csv, summary_rows, summary_fieldnames)
        print(f"\nwrote summary CSV: {args.csv}")

    if args.details_csv:
        write_csv(args.details_csv, details, ["group", "selection", "example_id", "path"])
        print(f"wrote details CSV: {args.details_csv}")

    if not args.csv or not args.details_csv:
        for out_dir in sorted(parsed_by_dir_group):
            if not args.csv:
                dir_summary_rows = build_summary_rows(
                    counts_by_dir_group[out_dir],
                    parsed_by_dir_group[out_dir],
                    out_of_range_by_dir_group[out_dir],
                    args.max_index,
                )
                summary_path = out_dir / args.summary_name
                write_csv(summary_path, dir_summary_rows, summary_fieldnames)
                print(f"wrote summary CSV: {summary_path}")
            if not args.details_csv:
                details_path = out_dir / args.details_name
                write_csv(details_path, details_by_dir[out_dir], ["group", "selection", "example_id", "path"])
                print(f"wrote details CSV: {details_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
