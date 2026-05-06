#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PLAN_RESPONSE_SUFFIX = "_plan_response"


def plan_response_model_prefix(path: Path) -> str | None:
    name = path.name
    if not name.endswith(PLAN_RESPONSE_SUFFIX):
        return None
    stem = name[: -len(PLAN_RESPONSE_SUFFIX)]
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


def find_plan_response_files(root: Path, dataset: str, model: str | None = None, role: str = "any") -> list[Path]:
    dataset_path = Path(*dataset.strip("/").split("/"))
    files = sorted(root.glob(f"**/{dataset_path}/**/*{PLAN_RESPONSE_SUFFIX}"))
    if not model:
        return files
    matched = []
    for path in files:
        prefix = plan_response_model_prefix(path)
        if prefix and model_matches(prefix, model, role):
            matched.append(path)
    return matched


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_evaluator(evaluator_path: Path):
    sys.path.insert(0, str(evaluator_path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("smfr_answer_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_answer_payload(response: Any) -> dict[str, Any]:
    if not isinstance(response, str):
        return {"answer": response}

    marker = "Answer:"
    idx = response.rfind(marker)
    if idx < 0:
        return {"answer": response}

    payload = response[idx + len(marker):].strip()
    if not payload:
        return {"answer": ""}

    decoder = json.JSONDecoder()
    for start in range(len(payload)):
        if payload[start] not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(payload[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        return {"answer": parsed}

    return {"answer": payload}


def load_plan_response(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def convert_file(path: Path) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(load_plan_response(path)):
        if not isinstance(item, dict):
            continue
        problem = item.get("problem")
        if not problem:
            continue
        rows.append(
            {
                "input": {"problem": problem},
                "output": parse_answer_payload(item.get("response")),
                "metadata": {
                    "source_file": str(path),
                    "source_index": idx,
                    "example_id": item.get("example_id"),
                    "trajectory": item.get("n"),
                },
            }
        )
    return rows


def evaluate_rows(
    rows: list[dict[str, Any]],
    reference_path: Path,
    validation_path: Path | None,
    evaluator_path: Path,
    timeout: int,
) -> None:
    evaluator = load_evaluator(evaluator_path)
    gt_by_problem = {}
    depth_by_problem = {}
    for item in load_jsonl(reference_path):
        problem = item.get("problem")
        if not problem:
            continue
        gt_by_problem[problem] = item.get("answer")
        depth_by_problem[problem] = item.get("depth", "unknown")

    validation_problems = set()
    if validation_path:
        validation_problems = {item["problem"] for item in load_jsonl(validation_path) if "problem" in item}

    executor = evaluator.SafeCodeExecutor(timeout=timeout) if evaluator._HAS_EXECUTOR else None
    stats = defaultdict(lambda: {"total": 0, "answer_correct": 0, "code_total": 0, "code_correct": 0, "code_failures": 0})
    skipped = 0

    for row in rows:
        problem = row.get("input", {}).get("problem", "")
        if problem in validation_problems:
            skipped += 1
            continue
        if problem not in gt_by_problem:
            skipped += 1
            continue

        depth = depth_by_problem.get(problem, "unknown")
        gt = gt_by_problem[problem]
        output = row.get("output", {})
        if not isinstance(output, dict):
            output = {}

        stats[depth]["total"] += 1
        if evaluator.compare_answers(gt, output.get("answer")):
            stats[depth]["answer_correct"] += 1

        code = output.get("code")
        if code:
            stats[depth]["code_total"] += 1
            ok, failed = evaluator._evaluate_code(code, gt, executor)
            if ok:
                stats[depth]["code_correct"] += 1
            if failed:
                stats[depth]["code_failures"] += 1

    def pct(num: int, den: int) -> str:
        return f"{num / den:.2%}" if den else "N/A"

    print("\nExact match accuracy by depth")
    print("depth\ttotal\tanswer_correct\tanswer_acc\tcode_correct\tcode_total\tcode_acc\tcode_errors")
    for depth in sorted(stats, key=lambda value: (value == "unknown", value)):
        item = stats[depth]
        print(
            f"{depth}\t"
            f"{item['total']}\t"
            f"{item['answer_correct']}\t"
            f"{pct(item['answer_correct'], item['total'])}\t"
            f"{item['code_correct']}\t"
            f"{item['code_total']}\t"
            f"{pct(item['code_correct'], item['code_total'])}\t"
            f"{item['code_failures']}"
        )

    total = sum(item["total"] for item in stats.values())
    answer_correct = sum(item["answer_correct"] for item in stats.values())
    code_total = sum(item["code_total"] for item in stats.values())
    code_correct = sum(item["code_correct"] for item in stats.values())
    code_failures = sum(item["code_failures"] for item in stats.values())
    print(
        f"all\t{total}\t{answer_correct}\t{pct(answer_correct, total)}\t"
        f"{code_correct}\t{code_total}\t{pct(code_correct, code_total)}\t{code_failures}"
    )
    if skipped:
        print(f"Skipped rows: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert old smfr *_plan_response files and optionally evaluate them by depth."
    )
    parser.add_argument(
        "--root",
        default="outputs",
        help="Root directory containing previous outputs. Default: outputs",
    )
    parser.add_argument(
        "--dataset",
        default="workflow_search/smfr",
        help="Dataset path fragment under outputs. Default: workflow_search/smfr",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path for evaluate_smfr_answer_code.py --model-output",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model-name filter applied to *_plan_response filenames, e.g. gpt-5.",
    )
    parser.add_argument(
        "--role",
        choices=["any", "meta", "node", "verifier"],
        default="any",
        help="Which filename model slot to match when --model is set. Default: any.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Fixed reference JSONL. If set, print exact match accuracy by depth.",
    )
    parser.add_argument(
        "--validation",
        default=None,
        help="Optional validation JSONL whose problems should be excluded.",
    )
    parser.add_argument(
        "--evaluator",
        default="smfr_synthetic_dataset/evaluate/evaluate_smfr_answer_code.py",
        help="Path to evaluate_smfr_answer_code.py.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for code execution. Default: 30.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = find_plan_response_files(root, args.dataset, args.model, args.role)
    if not files:
        raise SystemExit(f"No *_plan_response files found under {root} for dataset {args.dataset}")

    rows = []
    for path in files:
        rows.extend(convert_file(path))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Matched plan_response files: {len(files)}")
    print(f"Converted trajectories: {len(rows)}")
    if args.output:
        print(f"Saved model-output JSONL to: {output_path}")

    if args.reference:
        evaluate_rows(
            rows,
            reference_path=Path(args.reference),
            validation_path=Path(args.validation) if args.validation else None,
            evaluator_path=Path(args.evaluator),
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
