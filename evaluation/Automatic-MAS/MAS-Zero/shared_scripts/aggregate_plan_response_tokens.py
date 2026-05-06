#!/usr/bin/env python3

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import tiktoken


def find_plan_response_files(patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        for matched in glob.glob(pattern, recursive=True):
            path = Path(matched).resolve()
            if path.is_file() and path.name.endswith("_plan_response"):
                files.add(path)
    return sorted(files)


def get_encoding(model_name: str | None, encoding_name: str | None):
    try:
        if encoding_name:
            return tiktoken.get_encoding(encoding_name)
        if model_name:
            try:
                return tiktoken.encoding_for_model(model_name)
            except KeyError:
                pass
        return tiktoken.get_encoding("o200k_base")
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize a tiktoken encoding. "
            "This environment does not appear to have the required encoding cached locally. "
            "Provide --encoding/--model with an available cached tokenizer, or run once in an "
            "environment where tiktoken can download and cache the encoding files."
        ) from exc


def count_tokens(encoding, value: Any) -> int:
    text = "" if value is None else str(value)
    return len(encoding.encode(text))


def normalize_round_label(label: Any) -> str:
    if isinstance(label, str):
        return label
    return json.dumps(label, ensure_ascii=False)


def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item #{idx} is not a JSON object")
    return data


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
                # raise ValueError(f"Missing `depth` in dataset row {idx}")
                depth = 0
            depth_map[idx] = int(depth)
    return depth_map


def analyze_file(path: Path, encoding) -> dict[str, Any]:
    items = load_items(path)
    rounds: dict[str, dict[str, int]] = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "items": 0})

    total_input_tokens = 0
    total_output_tokens = 0
    example_ids = set()

    idx2input_tokens = {}
    idx2output_tokens = {}
    for idx, item in enumerate(items):
        if "problem" not in item:
            raise KeyError(f"{path} item #{idx} missing 'problem'")
        if "response" not in item:
            raise KeyError(f"{path} item #{idx} missing 'response'")
        if "example_id" not in item:
            raise KeyError(f"{path} item #{idx} missing 'example_id'")

        round_label = normalize_round_label(item.get("n"))
        input_tokens = count_tokens(encoding, item["problem"])
        output_tokens = count_tokens(encoding, item["response"])
        example_ids.add(int(item["example_id"]))

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        rounds[round_label]["input_tokens"] += input_tokens
        rounds[round_label]["output_tokens"] += output_tokens
        rounds[round_label]["items"] += 1
        idx2input_tokens[idx] = input_tokens
        idx2output_tokens[idx] = output_tokens

    if len(example_ids) != 1:
        raise ValueError(f"{path} contains multiple example_id values: {sorted(example_ids)}")

    return {
        "path": path,
        "example_id": next(iter(example_ids)),
        "items": len(items),
        "rounds": dict(rounds),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cot_input_tokens": idx2input_tokens[0] if len(idx2input_tokens) > 0 else 0,
        "cot_output_tokens": idx2output_tokens[0] if len(idx2output_tokens) > 0 else 0,
        "cot_sc_input_tokens": idx2input_tokens[1] if len(idx2input_tokens) > 1 else 0,
        "cot_sc_output_tokens": idx2output_tokens[1] if len(idx2output_tokens) > 1 else 0,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate problem/response token counts from *_plan_response files."
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["outputs/**/*_plan_response"],
        help="Glob patterns for *_plan_response files. Defaults to outputs/**/*_plan_response",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Tokenizer model name for tiktoken.encoding_for_model (default: gpt-5).",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Explicit tiktoken encoding name. If set, overrides --model.",
    )
    parser.add_argument(
        "--dataset-path",
        default="smfr_synthetic_dataset/balanced_dataset_merged_depth.jsonl",
        help="Merged smfr dataset jsonl with top-level `depth` field.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file totals and per-round breakdowns.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path to save aggregated usage.",
    )
    args = parser.parse_args()

    encoding = get_encoding(args.model, args.encoding)
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        # raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        print("Depth map is not provided")
        depth_map = {}
    else:
        depth_map = load_depth_map(dataset_path)

    files = find_plan_response_files(args.patterns)
    if not files:
        raise SystemExit("No *_plan_response files matched the given patterns.")

    analyses = [analyze_file(path, encoding) for path in files]

    depth_groups: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "files": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "cot_input_tokens": 0,
            "cot_output_tokens": 0,
            "cot_sc_input_tokens": 0,
            "cot_sc_output_tokens": 0,
            "example_ids": [],
        }
    )
    for analysis in analyses:
        depth = depth_map.get(analysis["example_id"])
        if depth is None:
            print(
                f"example_id {analysis['example_id']} from {analysis['path']} not found in depth dataset. Set to zero."
            )
            depth = 0
        analysis["depth"] = depth
        depth_groups[depth]["files"] += 1
        depth_groups[depth]["total_input_tokens"] += analysis["total_input_tokens"]
        depth_groups[depth]["total_output_tokens"] += analysis["total_output_tokens"]
        depth_groups[depth]["cot_input_tokens"] += analysis["cot_input_tokens"]
        depth_groups[depth]["cot_output_tokens"] += analysis["cot_output_tokens"]
        depth_groups[depth]["cot_sc_input_tokens"] += analysis["cot_sc_input_tokens"]
        depth_groups[depth]["cot_sc_output_tokens"] += analysis["cot_sc_output_tokens"]
        depth_groups[depth]["example_ids"].append(analysis["example_id"])

    print(f"Tokenizer encoding: {encoding.name}")
    print(f"Matched sample files: {len(analyses)}")
    print("\nAverage tokens per depth:")
    depth_summaries = []
    for depth in sorted(depth_groups):
        stats = depth_groups[depth]
        avg_input = stats["total_input_tokens"] / stats["files"]
        avg_output = stats["total_output_tokens"] / stats["files"]
        cot_input = stats["cot_input_tokens"]
        cot_output = stats["cot_output_tokens"]
        cot_sc_input = stats["cot_sc_input_tokens"]
        cot_sc_output = stats["cot_sc_output_tokens"]
        depth_summary = {
            "depth": depth,
            "files": stats["files"],
            "example_ids": sorted(stats["example_ids"]),
            "usage": {
                "avg_input_tokens": avg_input,
                "avg_output_tokens": avg_output,
                "avg_total_tokens": avg_input + avg_output,
                "cot_avg_input_tokens": cot_input,
                "cot_avg_output_tokens": cot_output,
                "cot_estimated_price": cot_input * 0.15 / 1e6 + cot_output * 0.6 / 1e6,
                "cot_sc_avg_input_tokens": cot_sc_input,
                "cot_sc_avg_output_tokens": cot_sc_output,
                "cot_sc_estimated_price": cot_sc_input * 0.15 / 1e6 + cot_sc_output * 0.6 / 1e6,
                "sum_input_tokens": stats["total_input_tokens"],
                "sum_output_tokens": stats["total_output_tokens"],
                "sum_estimated_price": stats["total_input_tokens"] * 0.15 / 1e6 + stats["total_output_tokens"] * 0.6 / 1e6,
                "sum_total_tokens": stats["total_input_tokens"] + stats["total_output_tokens"],
            },
        }
        depth_summaries.append(depth_summary)
        print(
            f"  depth={depth}: "
            f"files={stats['files']}, "
            f"avg_input_tokens={avg_input:.2f}, "
            f"avg_output_tokens={avg_output:.2f}"
        )

    if args.verbose:
        print("\nPer-file totals:")
        for analysis in sorted(analyses, key=lambda item: (item["depth"], item["example_id"], str(item["path"]))):
            print(
                f"  {analysis['path']}: "
                f"depth={analysis['depth']}, "
                f"example_id={analysis['example_id']}, "
                f"items={analysis['items']}, "
                f"input_tokens={analysis['total_input_tokens']}, "
                f"output_tokens={analysis['total_output_tokens']}"
            )
            for round_label, stats in analysis["rounds"].items():
                print(
                    f"    round={round_label}, "
                    f"items={stats['items']}, "
                    f"input_tokens={stats['input_tokens']}, "
                    f"output_tokens={stats['output_tokens']}"
                )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable_files = []
        for analysis in sorted(analyses, key=lambda item: (item["depth"], item["example_id"], str(item["path"]))):
            rounds = {}
            for round_label, stats in analysis["rounds"].items():
                rounds[round_label] = {
                    "items": stats["items"],
                    "usage": {
                        "input_tokens": stats["input_tokens"],
                        "output_tokens": stats["output_tokens"],
                        "total_tokens": stats["input_tokens"] + stats["output_tokens"],
                    },
                }
            serializable_files.append(
                {
                    "path": str(analysis["path"]),
                    "depth": analysis["depth"],
                    "example_id": analysis["example_id"],
                    "items": analysis["items"],
                    "usage": analysis["usage"],
                    "rounds": rounds,
                }
            )

        payload = {
            "tokenizer_encoding": encoding.name,
            "dataset_path": str(dataset_path),
            "matched_files": len(analyses),
            "per_depth": depth_summaries,
            "per_file": serializable_files,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved usage JSON to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise SystemExit(str(exc))
