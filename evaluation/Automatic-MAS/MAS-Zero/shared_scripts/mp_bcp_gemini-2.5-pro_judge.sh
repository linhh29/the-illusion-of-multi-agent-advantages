
export OPENAI_API_KEY=""

export OPENAI_BASE_URL="https://aihubmix.com/v1"

#for output_dir in "outputs/async-gemini-2.5-pro-run1" "outputs/async-gemini-2.5-pro-run2"; do
output_dir="outputs/async-gemini-2.5-pro-run1"
python main_judge_mp.py  \
  --dataset browsecomp-plus \
  --judge_method self \
  --baseline workflow_search --model gemini-2.5-pro --min_sample 0 --max_sample 167 --max_response_per_sample 9 \
  --save_dir $output_dir --num_workers 64

for judge_method in "cot" "cot-sc" "debate" "reflexion"; do
  python main_judge_mp.py  \
  --dataset browsecomp-plus \
  --judge_method $judge_method \
  --baseline workflow_search --model gemini-2.5-pro --min_sample 0 --max_sample 167 --max_response_per_sample 9 \
  --save_dir $output_dir --num_workers 64
done
#done