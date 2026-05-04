#!/bin/bash
# Stock CoT-SC evaluation. Levels 2–6, three runs.
# Up to 30 concurrent async STOCKS tasks per run (asyncio.Semaphore in stock_standalone_eval).
# Lower than CoT's 50 because each task fires num_samples LLM calls internally.
#
# Smoke test (one problem per level):   STOCK_LIMIT=1 bash workspace/run_stock_cot_sc_3x.sh

export TMPDIR="${TMPDIR:-$HOME/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
mkdir -p "$TMPDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAAS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MAAS_ROOT" || exit 1
export PYTHONPATH="${MAAS_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

#MODEL="gpt-5"
MODEL="openai/gpt-oss-120b"

#for i in 1 2 3; do
for i in 1 2; do
    echo "============================================"
    echo "  CoT-SC Run $i / 3  ($MODEL)  —  $(date)"
    echo "============================================"

    python -m workspace.standalone_cot_eval \
        --stock_levels 2 3 4 5 6 \
        --method cot-sc \
        --model "$MODEL" \
        --num_samples 5 \
        --max_concurrent 50 \
        --max_tokens 32768 \
        --output_path "workspace/results_cot_sc_stock_${MODEL}_run${i+1}.json"
    run_ec=$?

    echo ""
    echo "Run $i finished at $(date), exit code: ${run_ec}"
    echo ""
    if [ "${run_ec}" -ne 0 ]; then
        echo "Python exited with ${run_ec}; results JSON is only written after a successful full run."
        exit "${run_ec}"
    fi
done

echo "All 3 runs done."
