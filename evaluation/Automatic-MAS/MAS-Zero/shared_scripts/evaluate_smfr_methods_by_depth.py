#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_ROWS = {
    "cot": "Chain-of-Thought",
    "cot_sc": "Self-Consistency with Chain-of-Thought",
    "reflexion": "Self-Refine (Reflexion)",
    "debate": "LLM Debate",
}
PRICE_INPUT_PER_TOKEN = 0.15 / 1e6
PRICE_OUTPUT_PER_TOKEN = 0.6 / 1e6


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_evaluator(path: Path):
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("smfr_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_prefix(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)].rsplit("_", 1)[0]


def model_matches(prefix: str, model: str | None) -> bool:
    if not model:
        return True
    return prefix == model or prefix.startswith(f"{model}_") or prefix.endswith(f"_{model}") or f"_{model}_" in prefix


def find_response_files(root: Path, dataset: str, model: str | None) -> list[Path]:
    dataset_path = Path(*dataset.strip("/").split("/"))
    files = sorted(root.glob(f"**/{dataset_path}/**/*_plan_response"))
    return [p for p in files if model_matches(model_prefix(p, "_plan_response"), model)]


def parse_output(response: Any) -> dict[str, Any]:
    if not isinstance(response, str):
        return {"answer": response}
    idx = response.rfind("Answer:")
    payload = response[idx + len("Answer:"):].strip() if idx >= 0 else response.strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(payload):
        if ch not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(payload[i:])
            return parsed if isinstance(parsed, dict) else {"answer": parsed}
        except json.JSONDecodeError:
            pass
    return {"answer": payload}


def score_path_for(response_path: Path) -> Path:
    return response_path.with_name(response_path.name[: -len("_response")] + "_score.json")


def archive_path_for(response_path: Path) -> Path:
    return response_path.with_name(response_path.name[: -len("_response")] + "_archive.json")


def token_usage(value: Any) -> tuple[int, int]:
    if not isinstance(value, dict):
        return 0, 0
    input_tokens = int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0)
    output_tokens = int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0)
    return input_tokens, output_tokens


def usage_price(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * PRICE_INPUT_PER_TOKEN + output_tokens * PRICE_OUTPUT_PER_TOKEN


def usage_by_method(response_path: Path) -> dict[str, tuple[int, int]]:
    path = archive_path_for(response_path)
    if not path.exists():
        return {}
    archive = load_json(path)
    if not isinstance(archive, list) or not archive:
        return {}

    usage = {}
    if len(archive) >= 1:
        usage["cot"] = token_usage(archive[0].get("round_usage"))
    if len(archive) >= 2:
        usage["cot_sc"] = token_usage(archive[1].get("round_usage"))

    for item in reversed(archive):
        if isinstance(item, dict) and "usage" in item:
            usage["self"] = token_usage(item.get("usage"))
            break
    return usage


def selected_self_row(rows: list[dict[str, Any]], response_path: Path) -> dict[str, Any] | None:
    path = score_path_for(response_path)
    if not path.exists():
        return None
    try:
        selection = load_json(path).get("selection")
    except Exception as e:
        print(f"Failed to load selection from {path}: {e}")
        return None
    for row in rows:
        if row.get("n") == selection or str(row.get("n")) == str(selection):
            return row
    if isinstance(selection, int) and 0 <= selection < len(rows):
        return rows[selection]
    return None


def iter_method_outputs(response_path: Path):
    rows = load_json(response_path)
    if not isinstance(rows, list):
        return
    by_name = {row.get("n"): row for row in rows if isinstance(row, dict)}
    for method, row_name in METHOD_ROWS.items():
        row = by_name.get(row_name)
        if row:
            yield method, row
    row = selected_self_row(rows, response_path)
    if row:
        yield "self", row
    elif rows:
        yield "self", {"problem": rows[0].get("problem"), "response": ""}


def update_stats(
    stats,
    method: str,
    depth: Any,
    row: dict[str, Any],
    gt: Any,
    evaluator,
    executor,
    usage: tuple[int, int] = (0, 0),
) -> None:
    out = parse_output(row.get("response"))
    item = stats[method][depth]
    item["total"] += 1
    input_tokens, output_tokens = usage
    item["input_tokens"] += input_tokens
    item["output_tokens"] += output_tokens
    item["price"] += usage_price(input_tokens, output_tokens)
    if evaluator.compare_answers(gt, out.get("answer")):
        item["answer_correct"] += 1
    code = out.get("code")
    if code:
        ok, failed = evaluator._evaluate_code(code, gt, executor)
        item["code_errors"] += int(failed)
        item["code_correct"] += int(ok)


def pct(num: int, den: int) -> str:
    return f"{num / den:.2%}" if den else "N/A"


def print_stats(stats) -> None:
    print("method\tdepth\ttotal\tanswer_correct\tanswer_acc\tcode_correct\tcode_acc\tcode_errors\tinput_tokens\toutput_tokens\tprice\tavg_price")
    for method in ["cot", "cot_sc", "reflexion", "debate", "self"]:
        if method not in stats:
            continue
        totals = defaultdict(int)
        for depth in sorted(stats[method], key=lambda x: (x == "unknown", x)):
            s = stats[method][depth]
            print(
                f"{method}\t{depth}\t{s['total']}\t{s['answer_correct']}\t"
                f"{pct(s['answer_correct'], s['total'])}\t{s['code_correct']}\t"
                f"{pct(s['code_correct'], s['total'])}\t{s['code_errors']}\t"
                f"{s['input_tokens']}\t{s['output_tokens']}\t{s['price']:.6f}\t"
                f"{(s['price'] / s['total']) if s['total'] else 0:.6f}"
            )
            for key, value in s.items():
                totals[key] += value
        print(
            f"{method}\tall\t{totals['total']}\t{totals['answer_correct']}\t"
            f"{pct(totals['answer_correct'], totals['total'])}\t{totals['code_correct']}\t"
            f"{pct(totals['code_correct'], totals['total'])}\t{totals['code_errors']}\t"
            f"{totals['input_tokens']}\t{totals['output_tokens']}\t{totals['price']:.6f}\t"
            f"{(totals['price'] / totals['total']) if totals['total'] else 0:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate smfr methods by depth from old *_plan_response files.")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--dataset", default="workflow_search/smfr")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--evaluator", default="smfr_synthetic_dataset/evaluate/evaluate_smfr_answer_code.py")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    evaluator = load_evaluator(Path(args.evaluator))
    executor = evaluator.SafeCodeExecutor(timeout=args.timeout) if evaluator._HAS_EXECUTOR else None
    reference = {r["problem"]: r for r in load_jsonl(Path(args.reference))}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    files = find_response_files(Path(args.root), args.dataset, args.model)
    for path in files:
        usage_map = usage_by_method(path)
        for method, row in iter_method_outputs(path):
            problem = row.get("problem")
            ref = reference.get(problem)
            if not ref:
                continue
            update_stats(
                stats,
                method,
                ref.get("depth", "unknown"),
                row,
                ref["answer"],
                evaluator,
                executor,
                usage_map.get(method, (0, 0)),
            )

    print(f"Matched response files: {len(files)}")
    print_stats(stats)


if __name__ == "__main__":
    main()
