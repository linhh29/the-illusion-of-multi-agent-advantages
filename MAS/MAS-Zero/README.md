# MAS-Zero: Multi-Agent System Design with Zero Supervision

MAS-Zero automatically designs multi-agent systems at inference time without validation data.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

Place benchmark datasets (gpqa, hlemath, swe) in `dataset/`.

### Search (Design MAS)

```bash
python main_question.py \
    --dataset workflow_search/gpqa_diamond \
    --option plan \
    --meta_model gpt-4o_chatgpt \
    --node_model gpt-4o_chatgpt \
    --verifier_model gpt-4o_chatgpt \
    --blocks COT COT_SC Reflexion LLM_debate \
    --n_generation 5
```

### Verification (Select Best Answer)

```bash
python main_judge.py \
    --dataset gpqa_diamond \
    --judge_method self \
    --baseline workflow_search \
    --model gpt-4o_chatgpt \
    --max_response_per_sample 9
```

Supported datasets: `workflow_search/gpqa_diamond`, `workflow_search/hlemath`, `workflow_search/swe_bench`
