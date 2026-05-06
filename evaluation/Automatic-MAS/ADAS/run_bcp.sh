#!/bin/bash

export OPENAI_API_KEY="sk-xxx"
export AGENT_BASE_MODEL=gemini-2.5-pro
export MAX_NEW_TOKENS=32768

for seed in 0 1 42; do
    time python _bcp/search.py \
        --data_filename dataset/bcp.jsonl \
        --save_dir li_results/ \
        --expr_name bcp_${AGENT_BASE_MODEL} \
        --n_repreat 1 \
        --n_generation 15 \
        --model ${AGENT_BASE_MODEL} \
        --seed ${seed} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --valid_size 32 \
        --debug_max 3
done