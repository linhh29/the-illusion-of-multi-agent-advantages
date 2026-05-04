#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SUCCESS_RE = re.compile(r"^experi?emnt\s+(\d+):\s+1\b.*correct_answer:", re.IGNORECASE)
SUMMARY_RE = re.compile(r"^correct\s+(\d+);\s+Total:\s+(\d+);\s+Acc:\s+([0-9.]+)")
RESULT_NAME_RE = re.compile(r"_(self|cot|cot-sc|debate|reflexion)\.results(?:_\d+)?$")


def load_depth_map(dataset_path: Path) -> dict[int, int]:
    depth_map = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            depth = item.get("depth")
            if depth is None:
                raise ValueError(f"Missing `depth` in dataset row {idx}")
            depth_map[idx] = int(depth)
    return depth_map


def collect_result_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            for candidate in sorted(path.rglob("*.results*")):
                if RESULT_NAME_RE.search(candidate.name):
                    files.append(candidate)
    return sorted(set(files))


def parse_result_file(path: Path) -> tuple[set[int], dict | None]:
    correct_ids = set()
    summary = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            m = SUCCESS_RE.match(line)
            if m:
                correct_ids.add(int(m.group(1)))
                continue
            m = SUMMARY_RE.match(line)
            if m:
                summary = {
                    "correct": int(m.group(1)),
                    "total": int(m.group(2)),
                    "acc": float(m.group(3)),
                }
    return correct_ids, summary


def infer_method(path: Path) -> str:
    m = RESULT_NAME_RE.search(path.name)
    return m.group(1) if m else "unknown"


def summarize_file(path: Path, depth_map: dict[int, int]) -> dict:
    correct_ids, summary = parse_result_file(path)
    total_by_depth = Counter(depth_map.values())
    correct_by_depth = Counter()
    unknown_ids = []

    for example_id in sorted(correct_ids):
        depth = depth_map.get(example_id)
        if depth is None:
            unknown_ids.append(example_id)
            continue
        correct_by_depth[depth] += 1

    per_depth = {}
    for depth in sorted(total_by_depth):
        total = total_by_depth[depth]
        correct = correct_by_depth[depth]
        per_depth[str(depth)] = {
            "correct": correct,
            "total": total,
            "acc": (correct / total) if total else 0.0,
        }

    return {
        "file": str(path),
        "method": infer_method(path),
        "overall_summary": summary,
        "parsed_correct": len(correct_ids),
        "unknown_example_ids": unknown_ids,
        "per_depth": per_depth,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result files or directories containing main_judge_mp stock result files.",
    )
    parser.add_argument(
        "--dataset-path",
        default="stocks_synthetic_dataset/balanced_dataset_merged_depth.jsonl",
        help="Merged stock dataset jsonl with top-level `depth` field.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    depth_map = load_depth_map(dataset_path)
    result_files = collect_result_files([Path(p) for p in args.paths])
    if not result_files:
        raise FileNotFoundError("No stock judge result files found.")

    summaries = [summarize_file(path, depth_map) for path in result_files]

    for summary in summaries:
        print(summary["file"])
        if summary["overall_summary"] is not None:
            overall = summary["overall_summary"]
            print(
                f"  overall: correct={overall['correct']} total={overall['total']} acc={overall['acc']:.6f}"
            )
        else:
            print("  overall: summary line not found")
        print(f"  parsed_correct={summary['parsed_correct']}")
        if summary["unknown_example_ids"]:
            print(f"  unknown_example_ids={summary['unknown_example_ids']}")
        for depth, metrics in summary["per_depth"].items():
            print(
                f"  depth={depth}: correct={metrics['correct']} total={metrics['total']} acc={metrics['acc']:.6f}"
            )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
