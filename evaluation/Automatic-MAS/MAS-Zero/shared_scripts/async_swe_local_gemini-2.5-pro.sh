export OPENAI_API_KEY=""
export OPENAI_BASE_URL="https://aihubmix.com/v1"
export BASE_PORT=8000

#python async_main_question.py  \
#  --dataset workflow_search/swe \
#  --option plan \
#  --meta_model gemini-2.5-pro \
#  --node_model gemini-2.5-pro \
#  --verifier_model gpt-4o_chatgpt --blocks COT COT_SC Reflexion LLM_debate \
#  --use_oracle_verifier --defer_verifier --n_generation 5 --save_dir outputs/async-gemini-2.5-pro --max_workers 64 --max_tokens 32768


#python async_main_question.py  \
#  --dataset workflow_search/swe \
#  --option plan \
#  --meta_model gemini-2.5-pro \
#  --node_model gemini-2.5-pro \
#  --verifier_model gpt-4o_chatgpt --blocks COT COT_SC Reflexion LLM_debate \
#  --use_oracle_verifier --defer_verifier --n_generation 5 --save_dir outputs/async-gemini-2.5-pro-run1 --max_workers 64 --max_tokens 32768


python async_main_question.py  \
  --dataset workflow_search/swe \
  --option plan \
  --meta_model gemini-2.5-pro \
  --node_model gemini-2.5-pro \
  --verifier_model gpt-4o_chatgpt --blocks COT COT_SC Reflexion LLM_debate \
  --use_oracle_verifier --defer_verifier --n_generation 5 --save_dir outputs/async-gemini-2.5-pro-run2 --max_workers 64 --max_tokens 32768