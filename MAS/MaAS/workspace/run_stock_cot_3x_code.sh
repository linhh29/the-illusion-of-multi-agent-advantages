#!/bin/bash
# Same as run_stock_cot_3x.sh with larger max_tokens for long outputs.

export TMPDIR="${TMPDIR:-$HOME/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
mkdir -p "$TMPDIR"

cd /home/qin/wsd/mas_eval/MaAS

for i in 1 2 3; do
    echo "============================================"
    echo "  CoT Run $i / 3 (max_tokens=16384)  —  $(date)"
    echo "============================================"

    python -m workspace.standalone_cot_eval \
        --stock_levels 2 3 4 5 6 \
        --method cot \
        --model gpt-4o \
        --max_tokens 16384 \
        --output_path "workspace/results_cot_stock_code_run${i}.json"

    echo ""
    echo "Run $i finished at $(date), exit code: $?"
    echo ""
done

echo "All 3 runs done."
