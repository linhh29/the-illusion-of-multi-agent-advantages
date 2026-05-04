#!/bin/bash
# STOCKS evaluation (data and metric aligned with AFlow benchmarks/stocks.py)
export OPENAI_API_KEY=
export MAX_CONCURRENT=50
CODE_EVAL_MODE=${CODE_EVAL_MODE:-1}
MODEL=gpt-4o

# Data path: use AFlow datasets (stocks_validate.jsonl / stocks_test.jsonl)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAS_EVAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
dir="$MAS_EVAL_ROOT/AFlow/data/datasets"
exp_name=stocks_${MODEL}

ROLES="['Assistant', 'FinancialAnalyst', 'DataScientist', 'Programmer']"

for run_num in 1 2 3
do
    echo "=========================================="
    echo "STOCKS Run $run_num of 3"
    echo "=========================================="
    OUTPUT_DIR="${exp_name}_Assistant_FinancialAnalyst_DataScientist_Programmer_run${run_num}"
    mkdir -p "$OUTPUT_DIR"

    for file in "$dir"/stocks_test.jsonl
    do
        if [ ! -f "$file" ]; then
            echo "File not found: $file"
            continue
        fi
        filename=$(basename -- "$file" .jsonl)
        echo "Processing $filename (Run $run_num)..."
        python llmlp_listwise_stocks.py "$file" "$filename" "$MODEL" "$exp_name" "$ROLES" "$run_num" "$CODE_EVAL_MODE" > "$OUTPUT_DIR/${filename}.log" 2>&1
    done
    echo "Run $run_num completed!"
done
echo "All STOCKS runs completed!"
