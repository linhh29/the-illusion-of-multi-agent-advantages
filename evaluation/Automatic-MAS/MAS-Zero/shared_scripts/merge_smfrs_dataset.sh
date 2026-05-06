#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/smfr_synthetic_dataset}"
OUT_FILE="${2:-${DATA_DIR}/balanced_dataset_merged_depth_fixed.jsonl}"

python - "${DATA_DIR}" "${OUT_FILE}" <<'PY'
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])
depths = [2, 3, 4, 5, 6]
#depths = [4, 5]

out_file.parent.mkdir(parents=True, exist_ok=True)
counts = {}
total = 0

with out_file.open("w", encoding="utf-8") as fout:
    for depth in depths:
        src = data_dir / f"balanced_dataset_single_{depth}_fixed.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Missing input file: {src}")

        count = 0
        with src.open("r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {src}:{line_no}: {exc}") from exc

                item["depth"] = depth
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                total += 1

        counts[depth] = count

print(f"merged -> {out_file}")
print(f"total records: {total}")
print("per-depth counts:", counts)
PY

