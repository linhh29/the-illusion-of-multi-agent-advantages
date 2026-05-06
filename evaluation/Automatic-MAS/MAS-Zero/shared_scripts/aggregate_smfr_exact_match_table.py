#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LINE_RE = re.compile(
    r"^experi?emnt\s+(\d+):\s+1\b.*?direct_full=(\d+);\s*code_full=(\d+);",
    re.IGNORECASE,
)
METHOD_RE = re.compile(r"_(self|cot|cot-sc|debate|reflexion)\.results(?:_\d+)?$")


def load_depth_map(dataset_path: Path) -> dict[int, int]:
    depth_map = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            depth_map[idx] = int(item["depth"])
    return depth_map


def infer_method(file_path: Path) -> str | None:
    match = METHOD_RE.search(file_path.name)
    return match.group(1) if match else None


def summarize_result_file(file_path: Path, depth_map: dict[int, int]) -> dict:
    method = infer_method(file_path)
    if method is None:
        raise ValueError(f"Unsupported result filename: {file_path}")

    direct_by_depth = Counter()
    code_by_depth = Counter()
    seen_examples = set()

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            match = LINE_RE.match(line)
            if not match:
                continue
            example_id = int(match.group(1))
            direct_full = int(match.group(2))
            code_full = int(match.group(3))
            depth = depth_map.get(example_id)
            if depth is None:
                continue
            seen_examples.add(example_id)
            direct_by_depth[depth] += direct_full
            code_by_depth[depth] += code_full

    return {
        "method": method,
        "direct_by_depth": direct_by_depth,
        "code_by_depth": code_by_depth,
        "matched_examples": len(seen_examples),
    }


def collect_result_files(result_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in result_dir.iterdir()
        if p.is_file() and METHOD_RE.search(p.name)
    )


def build_rows(result_dirs: list[Path], depth_map: dict[int, int]) -> list[dict]:
    total_by_depth = Counter(depth_map.values())
    rows = []

    for result_dir in result_dirs:
        run_name = result_dir.parts[-5] if len(result_dir.parts) >= 5 else result_dir.name
        for result_file in collect_result_files(result_dir):
            summary = summarize_result_file(result_file, depth_map)
            row = {
                "run": run_name,
                "method": summary["method"],
            }
            total_direct = 0
            total_code = 0
            total_all = sum(total_by_depth.values())

            for depth in sorted(total_by_depth):
                total = total_by_depth[depth]
                direct = summary["direct_by_depth"][depth]
                code = summary["code_by_depth"][depth]
                total_direct += direct
                total_code += code

                row[f"d{depth}_total"] = total
                row[f"d{depth}_direct_count"] = direct
                row[f"d{depth}_direct_acc"] = f"{direct / total:.6f}"
                row[f"d{depth}_code_count"] = code
                row[f"d{depth}_code_acc"] = f"{code / total:.6f}"

            row["overall_total"] = total_all
            row["overall_direct_count"] = total_direct
            row["overall_direct_acc"] = f"{total_direct / total_all:.6f}"
            row["overall_code_count"] = total_code
            row["overall_code_acc"] = f"{total_code / total_all:.6f}"
            rows.append(row)

    rows.sort(key=lambda x: (x["run"], x["method"]))
    return rows


def write_csv(rows: list[dict], output_path: Path):
    fieldnames = ["run", "method"]
    for depth in [2, 3, 4, 5, 6]:
        fieldnames.extend(
            [
                f"d{depth}_total",
                f"d{depth}_direct_count",
                f"d{depth}_direct_acc",
                f"d{depth}_code_count",
                f"d{depth}_code_acc",
            ]
        )
    fieldnames.extend(
        [
            "overall_total",
            "overall_direct_count",
            "overall_direct_acc",
            "overall_code_count",
            "overall_code_acc",
        ]
    )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path):
    headers = [
        "run",
        "method",
        "d2 direct",
        "d2 code",
        "d3 direct",
        "d3 code",
        "d4 direct",
        "d4 code",
        "d5 direct",
        "d5 code",
        "d6 direct",
        "d6 code",
        "overall direct",
        "overall code",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        values = [row["run"], row["method"]]
        for depth in [2, 3, 4, 5, 6]:
            values.append(f'{row[f"d{depth}_direct_count"]}/{row[f"d{depth}_total"]} ({row[f"d{depth}_direct_acc"]})')
            values.append(f'{row[f"d{depth}_code_count"]}/{row[f"d{depth}_total"]} ({row[f"d{depth}_code_acc"]})')
        values.append(f'{row["overall_direct_count"]}/{row["overall_total"]} ({row["overall_direct_acc"]})')
        values.append(f'{row["overall_code_count"]}/{row["overall_total"]} ({row["overall_code_acc"]})')
        lines.append("| " + " | ".join(values) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", help="Smfr result directories.")
    parser.add_argument(
        "--dataset-path",
        default="smfr_synthetic_dataset/balanced_dataset_merged_depth.jsonl",
    )
    parser.add_argument(
        "--csv-output",
        default="outputs/smfr_exact_match_by_depth.csv",
    )
    parser.add_argument(
        "--md-output",
        default="outputs/smfr_exact_match_by_depth.md",
    )
    args = parser.parse_args()

    depth_map = load_depth_map(Path(args.dataset_path))
    result_dirs = [Path(p) for p in args.result_dirs]
    rows = build_rows(result_dirs, depth_map)

    csv_output = Path(args.csv_output)
    md_output = Path(args.md_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, csv_output)
    write_markdown(rows, md_output)

    print(f"wrote csv: {csv_output}")
    print(f"wrote md: {md_output}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
