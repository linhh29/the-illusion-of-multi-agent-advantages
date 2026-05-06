
export OPENAI_API_KEY=""

export OPENAI_BASE_URL="https://aihubmix.com/v1"


for output_dir in "outputs/async-gemini-2.5-pro-run1" "outputs/async-gemini-2.5-pro-run2"; do
python main_judge_mp.py  \
  --dataset gpqa_diamond \
  --judge_method self \
  --baseline workflow_search --model gemini-2.5-pro --min_sample 32 --max_sample 197 --max_response_per_sample 9 \
  --save_dir $output_dir --num_workers 8

for judge_method in "cot" "cot-sc" "debate" "reflexion"; do
  python main_judge_mp.py  \
  --dataset gpqa_diamond \
  --judge_method $judge_method \
  --baseline workflow_search --model gemini-2.5-pro --min_sample 32 --max_sample 197 --max_response_per_sample 9 \
  --save_dir $output_dir --num_workers 8
done
done