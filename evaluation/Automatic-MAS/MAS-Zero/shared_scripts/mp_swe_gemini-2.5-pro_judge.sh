#!/usr/bin/env bash
set -euo pipefail   # 关键：出错就退出；用到未定义变量也报错


export OPENAI_API_KEY=""

export OPENAI_BASE_URL="https://aihubmix.com/v1"

#python main_judge_mp.py  \
#    --dataset swe \
#    --judge_method reflexion \
#    --baseline workflow_search --model gemini-2.5-pro --min_sample 0 --max_sample 267 --max_response_per_sample 9 \
#    --save_dir outputs/async-gemini-2.5-pro --num_workers 64


for output_dir in "outputs/async-gemini-2.5-pro" "outputs/async-gemini-2.5-pro-run1" "outputs/async-gemini-2.5-pro-run2"; do
#output_dir="outputs/async-gemini-2.5-pro"
  python main_judge_mp.py  \
    --dataset swe \
    --judge_method self \
    --baseline workflow_search --model gemini-2.5-pro --min_sample 0 --max_sample 267 --max_response_per_sample 9 \
    --save_dir $output_dir --num_workers 16

  for judge_method in "cot" "cot-sc" "debate" "reflexion"; do
#    judge_method="reflexion"
    python main_judge_mp.py  \
    --dataset swe \
    --judge_method $judge_method \
    --baseline workflow_search --model gemini-2.5-pro --min_sample 0 --max_sample 267 --max_response_per_sample 9 \
    --save_dir $output_dir  --num_workers 16
  done
done