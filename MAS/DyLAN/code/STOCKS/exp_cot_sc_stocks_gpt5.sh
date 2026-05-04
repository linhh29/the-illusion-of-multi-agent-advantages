#!/bin/bash
export OPENAI_API_KEY=
export MAX_CONCURRENT=50
CODE_EVAL_MODE=${CODE_EVAL_MODE:-1}
number=5
MODEL=gpt-5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAS_EVAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
dir="$MAS_EVAL_ROOT/AFlow/data/datasets"
exp_name=stocks_${MODEL}
ROLES="['Assistant', 'FinancialAnalyst', 'DataScientist', 'Programmer']"

for run_num in 1 2 3
do
    echo "=========================================="
    echo "STOCKS CoT-SC Run $run_num of 3"
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
        echo "Processing $filename (CoT-SC Run $run_num)..."
        python cot_sc_stocks.py "$file" "$filename" "$MODEL" "$exp_name" "$ROLES" "$run_num" "$CODE_EVAL_MODE" "$number" > "$OUTPUT_DIR/${filename}_cot_sc.log" 2>&1
    done
    echo "Run $run_num completed!"
done
echo "All STOCKS CoT-SC runs completed!"
