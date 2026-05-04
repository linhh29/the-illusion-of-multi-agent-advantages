#!/bin/bash


###########
# GPT-5 GPQA
###########
config=configs/exp_gpt5.env
echo "Running experiment with config: $config"

# Load the configuration file
set -a # export automatically
source "$config"
set +a # stop automatic export

# print loaded variables for verification
echo "AGENT_BASE_MODEL: $AGENT_BASE_MODEL"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "SEED: $SEED"
echo "GPT5_REASONING_EFFORT: $GPT5_REASONING_EFFORT"
echo "GPT5_VERBOSITY: $GPT5_VERBOSITY"  

# Run the experiment
# seeds=(0 1 42)
# for SEED in "${seeds[@]}"; do
# echo "Running experiment with SEED: $SEED"
python _gpqa/search2.py --save_dir li_results/ --expr_name gpqa_${AGENT_BASE_MODEL} \
--n_generation 30 --model ${AGENT_BASE_MODEL} --seed ${SEED} --max_new_tokens ${MAX_NEW_TOKENS} \
--gpt5_reasoning_effort ${GPT5_REASONING_EFFORT} --gpt5_verbosity ${GPT5_VERBOSITY}
# done