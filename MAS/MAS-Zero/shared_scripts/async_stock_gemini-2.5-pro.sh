
export OPENAI_API_KEY=""
export OPENAI_BASE_URL="https://aihubmix.com/v1"
export BASE_PORT=8000


# Before running this script, run `shared_scripts/merge_stocks_dataset.sh` to merge the dataset. You need to put the stock dataset at `stocks_synthetic_dataset/balanced_dataset_single_*.jsonl`

python async_main_question.py  \
  --dataset workflow_search/stock \
  --option plan \
  --meta_model gemini-2.5-pro \
  --node_model gemini-2.5-pro \
  --verifier_model gpt-4o_chatgpt --blocks COT COT_SC Reflexion LLM_debate \
  --use_oracle_verifier --defer_verifier --n_generation 5 --save_dir outputs/async-gemini-2.5-pro-run0 --max_workers 32 --max_tokens 131072