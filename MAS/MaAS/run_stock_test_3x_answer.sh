#!/bin/bash

# === Step 1: Train (round 3, answer mode) ===
echo "============================================"
echo "  Training round 3 (answer)  —  $(date)"
echo "============================================"

python -m examples.maas.optimize \
    --dataset Stock \
    --round 5 \
    --sample 4 \
    --exec_model_name "gpt-5" \
    --eval_mode answer

echo ""
echo "Training finished at $(date), exit code: $?"
echo ""

# === Step 2: Test x3 (round 3, answer mode) ===
for i in 1 2 3; do
    echo "============================================"
    echo "  Test $i / 3 (answer)  —  $(date)"
    echo "============================================"

    python -m examples.maas.optimize \
        --dataset Stock \
        --round 3 \
        --sample 4 \
        --exec_model_name "gpt-4o" \
        --is_test True \
        --eval_mode answer

    echo ""
    echo "Test $i finished at $(date), exit code: $?"
    echo ""
done

echo "All done (answer, round 3)."
