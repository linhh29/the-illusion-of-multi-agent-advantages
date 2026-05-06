#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export HF_HOME=/data/qin/wsd/.cache/huggingface
export TMPDIR=/data/qin/wsd/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

MODEL="gpt-4o"

# === Step 1: Train (round 5) ===
echo "============================================"
echo "  Training round 5 ($MODEL)  —  $(date)"
echo "============================================"

python -m examples.maas.optimize \
    --dataset Smfr \
    --round 5 \
    --sample 4 \
    --exec_model_name "$MODEL"

echo ""
echo "Training finished at $(date), exit code: $?"
echo ""

# === Step 2: Test x3 ===
for i in 2; do
    echo "============================================"
    echo "  Test $i / 3 ($MODEL)  —  $(date)"
    echo "============================================"

    python -m examples.maas.optimize \
        --dataset Smfr \
        --round 5 \
        --sample 4 \
        --exec_model_name "$MODEL" \
        --is_test True

    echo ""
    echo "Test $i finished at $(date), exit code: $?"
    echo ""
done

echo "All done (round 4)."
