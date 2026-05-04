
export OPENAI_API_KEY=""

export OPENAI_BASE_URL="https://aihubmix.com/v1"

#for output_dir in "outputs/async-gemini-2.5-pro-run1" "outputs/async-gemini-2.5-pro-run2"; do
output_dir=outputs/async-gemini-2.5-pro-run0
dataset="stock"
model="gpt-5"
min_sample=0
max_sample=603
max_response_per_sample=9
num_workers=32

python main_judge_mp.py \
  --dataset "${dataset}" \
  --judge_method self \
  --baseline workflow_search \
  --model "${model}" \
  --min_sample "${min_sample}" \
  --max_sample "${max_sample}" \
  --max_response_per_sample "${max_response_per_sample}" \
  --save_dir "${output_dir}" \
  --num_workers "${num_workers}"

for judge_method in "cot" "cot-sc" "debate" "reflexion"; do
  python main_judge_mp.py \
    --dataset "${dataset}" \
    --judge_method "${judge_method}" \
    --baseline workflow_search \
    --model "${model}" \
    --min_sample "${min_sample}" \
    --max_sample "${max_sample}" \
    --max_response_per_sample "${max_response_per_sample}" \
    --save_dir "${output_dir}" \
    --num_workers "${num_workers}"
done


# After evaluation, run `shared_scripts/summarize_stock_results_by_depth.py` to obtain the results of each depth.
