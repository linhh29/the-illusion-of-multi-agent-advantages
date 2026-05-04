#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

MODEL="openai/gpt-oss-120b"

# === Step 1: Train (round 7) ===
echo "============================================"
echo "  Training round 7 ($MODEL)  —  $(date)"
echo "============================================"

python -m examples.maas.optimize \
    --dataset Stock \
    --round 7 \
    --sample 4 \
    --exec_model_name "$MODEL"

echo ""
echo "Training finished at $(date), exit code: $?"
echo ""

# === Step 2: Test x3 ===
for i in 1 2 3; do
    echo "============================================"
    echo "  Test $i / 3 ($MODEL)  —  $(date)"
    echo "============================================"

    python -m examples.maas.optimize \
        --dataset Stock \
        --round 7 \
        --sample 4 \
        --exec_model_name "$MODEL" \
        --is_test True

    echo ""
    echo "Test $i finished at $(date), exit code: $?"
    echo ""
done

echo "All done (round 4)."
