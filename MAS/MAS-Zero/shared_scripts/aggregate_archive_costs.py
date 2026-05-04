#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ARCHIVE_SUFFIX = "_plan_archive.json"


def normalize_dataset(dataset: str) -> str:
    return dataset.strip("/")


def archive_model_prefix(path: Path) -> str | None:
    name = path.name
    if not name.endswith(ARCHIVE_SUFFIX):
        return None
    stem = name[: -len(ARCHIVE_SUFFIX)]
    if "_" not in stem:
        return None
    return stem.rsplit("_", 1)[0]


def model_matches(prefix: str, model: str, role: str) -> bool:
    if role == "any":
        return (
            prefix == model
            or prefix.startswith(f"{model}_")
            or prefix.endswith(f"_{model}")
            or f"_{model}_" in prefix
        )
    if role == "meta":
        return prefix == model or prefix.startswith(f"{model}_")
    if role == "verifier":
        return prefix == model or prefix.endswith(f"_{model}")
    if role == "node":
        return f"_{model}_" in f"_{prefix}_"
    raise ValueError(f"Unsupported role: {role}")


def infer_example_id(path: Path, root: Path, dataset: str) -> str:
    dataset_parts = normalize_dataset(dataset).split("/")
    rel_parts = path.resolve().relative_to(root.resolve()).parts
    for idx in range(0, len(rel_parts) - len(dataset_parts)):
        if list(rel_parts[idx : idx + len(dataset_parts)]) == dataset_parts:
            if idx + len(dataset_parts) < len(rel_parts):
                return rel_parts[idx + len(dataset_parts)]
    return path.parent.name


def extract_generation_label(item: dict[str, Any], fallback: int) -> Any:
    return item.get("generation", fallback)


def load_archive(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item #{idx} is not a JSON object")
    return data


def cost_values(items: list[dict[str, Any]]) -> list[float]:
    values = []
    for item in items:
        if "total_cost" in item and item["total_cost"] is not None:
            values.append(float(item["total_cost"]))
    return values


def analyze_archive(path: Path, root: Path, dataset: str) -> dict[str, Any]:
    items = load_archive(path)
    costs = cost_values(items)
    if not costs:
        raise ValueError(f"{path} has no total_cost values")

    first_item = None
    final_item = None
    for item in items:
        if "total_cost" in item and item["total_cost"] is not None:
            first_item = item
            break
    for item in reversed(items):
        if "total_cost" in item and item["total_cost"] is not None:
            final_item = item
            break

    first_trajectory_cost = float(first_item["total_cost"])
    if len(items) > 1 and "total_cost" in items[1] and items[1]["total_cost"] is not None:
        cot_sc_cost = float(items[1]["total_cost"]) - first_trajectory_cost
    else:
        cot_sc_cost = 0
    final_total_cost = float(final_item["total_cost"])
    return {
        "path": str(path),
        "example_id": infer_example_id(path, root, dataset),
        "archive_items": len(items),
        "cost_entries": len(costs),
        "sample_total_cost": final_total_cost,
        "first_trajectory_cost": first_trajectory_cost,
        "cot_cost": first_trajectory_cost,
        "cot_sc_cost": cot_sc_cost,
        "final_total_cost": final_total_cost,
        "max_total_cost": max(costs),
        "first_generation": extract_generation_label(first_item, 0),
        "final_generation": extract_generation_label(final_item, len(items) - 1),
        "model_prefix": archive_model_prefix(path),
    }


def find_archives(root: Path, dataset: str, model: str, role: str) -> list[Path]:
    dataset_path = Path(*normalize_dataset(dataset).split("/"))
    candidates = sorted(root.glob(f"**/{dataset_path}/**/*{ARCHIVE_SUFFIX}"))
    files = []
    for path in candidates:
        prefix = archive_model_prefix(path)
        if prefix is None:
            continue
        if model_matches(prefix, model, role):
            files.append(path)
    return files


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(record["sample_total_cost"] for record in records)
    first_total = sum(record["first_trajectory_cost"] for record in records)
    cot_sc_cost_total = sum(record["cot_sc_cost"] for record in records)
    return {
        "samples": len(records),
        "matched_archives": len(records),
        "sum_sample_cost": total,
        "cot_sc_cost": cot_sc_cost_total,
        "avg_cost_per_sample": total / len(records) if records else 0.0,
        "min_sample_cost": min((record["sample_total_cost"] for record in records), default=0.0),
        "max_sample_cost": max((record["sample_total_cost"] for record in records), default=0.0),
        "sum_first_trajectory_cost": first_total,
        "avg_first_trajectory_cost_per_sample": first_total / len(records) if records else 0.0,
        "min_first_trajectory_cost": min((record["first_trajectory_cost"] for record in records), default=0.0),
        "max_first_trajectory_cost": max((record["first_trajectory_cost"] for record in records), default=0.0),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example_id",
                "sample_total_cost",
                "first_trajectory_cost",
                "final_total_cost",
                "max_total_cost",
                "archive_items",
                "cost_entries",
                "first_generation",
                "final_generation",
                "path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate total_cost from *_plan_archive.json files by model and dataset."
    )
    parser.add_argument(
        "--root",
        default="outputs",
        help="Root directory to search under. Default: outputs",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset path fragment, for example workflow_search/stock",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to filter, for example gpt-5",
    )
    parser.add_argument(
        "--role",
        choices=["any", "meta", "node", "verifier"],
        default="any",
        help="Which filename model slot to match. Default: any",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to save the full JSON summary.",
    )
    parser.add_argument(
        "--csv-output",
        help="Optional path to save per-sample CSV rows.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per matched sample.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    archives = find_archives(root, args.dataset, args.model, args.role)
    if not archives:
        raise SystemExit(
            f"No archive files matched root={root}, dataset={args.dataset}, "
            f"model={args.model}, role={args.role}"
        )

    records = [analyze_archive(path, root, args.dataset) for path in archives]
    records.sort(
        key=lambda item: (
            0 if re.fullmatch(r"\d+", item["example_id"]) else 1,
            int(item["example_id"]) if re.fullmatch(r"\d+", item["example_id"]) else item["example_id"],
            item["path"],
        )
    )
    summary = summarize(records)

    print(f"Root: {root}")
    print(f"Dataset: {args.dataset}")
    print(f"Model filter: {args.model} ({args.role})")
    print(f"Matched samples: {summary['samples']}")
    print(f"Matched archives: {summary['matched_archives']}")
    print(f"Sum sample cost: {summary['sum_sample_cost']:.6f}")
    print(f"Sum CoT-SC cost: {summary['cot_sc_cost']:.6f}")
    # print(f"Average cost per sample: {summary['avg_cost_per_sample']:.6f}")
    # print(f"Min sample cost: {summary['min_sample_cost']:.6f}")
    # print(f"Max sample cost: {summary['max_sample_cost']:.6f}")
    print(f"Sum first trajectory cost: {summary['sum_first_trajectory_cost']:.6f}")
    # print(f"Average first trajectory cost per sample: {summary['avg_first_trajectory_cost_per_sample']:.6f}")
    # print(f"Min first trajectory cost: {summary['min_first_trajectory_cost']:.6f}")
    # print(f"Max first trajectory cost: {summary['max_first_trajectory_cost']:.6f}")

    if args.verbose:
        print("\nPer sample:")
        for record in records:
            print(
                f"  example_id={record['example_id']}, "
                f"sample_total_cost={record['sample_total_cost']:.6f}, "
                f"first_trajectory_cost={record['first_trajectory_cost']:.6f}, "
                f"trajectories={record['cost_entries']}, "
                f"path={record['path']}"
            )

    payload = {
        "root": str(root),
        "dataset": args.dataset,
        "model": args.model,
        "role": args.role,
        "summary": summary,
        "samples": records,
    }

    if args.json_output:
        write_json(Path(args.json_output), payload)
        print(f"\nSaved JSON summary to: {args.json_output}")

    if args.csv_output:
        write_csv(Path(args.csv_output), records)
        print(f"Saved CSV rows to: {args.csv_output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise SystemExit(str(exc))
