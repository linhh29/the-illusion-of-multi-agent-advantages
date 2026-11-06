# Simple MAS for Stock Dataset

Ablation study designed to disentangle the contribution of **task-type context separation** from the Python scaffolding present in Expert MAS.

## Architecture

```
MetaAgent (1 call)
    ↓ {investors, question_type, target_pct, aggregation, price_type}
RetrievalAgent (1 call)
    ↓ all investors' transactions with transaction-date prices
CalculationAgent (1 call)
    ↓ final answer {"investor_dates": {...}, "answer": [...]}
```

**3 LLM calls per problem, regardless of number of investors.**

## What is removed from Expert MAS

| Feature removed | Where it goes |
|---|---|
| Python open-position identification (set difference) | LLM reasoning in CalculationAgent |
| Python formula embedding (pre-computed values in prompt) | LLM derives from symbolic formula |
| Pydantic schema enforcement | Free-form JSON parsing |
| Python argmin/argmax aggregation | LLM reasoning in CalculationAgent |

What is kept: MetaAgent (identical), per-task context isolation, robust JSON parser.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # if using a gateway
```

Configuration can also be placed in the repository-root `.env`; see
[`../.env.example`](../.env.example). For Together inference with per-phase
checkpoints and expense tracking, use the [final experiment runners](../experiments/final/README.md).
MetaAgent uses a schema; retrieval/calculation are free-form JSON calls.

## Running inference

```bash
# Quick test (2 samples)
python run_inference.py --input ../stocks_synthetic_dataset/balanced_dataset_single_2_fixed_v2.jsonl --model gpt-5 --test

# First N samples
python run_inference.py --input ../stocks_synthetic_dataset/balanced_dataset_single_2_fixed_v2.jsonl --model gpt-5 --slice 20

# Full run, 2 repetitions
python run_inference.py --input ../stocks_synthetic_dataset/balanced_dataset_single_2_fixed_v2.jsonl --model gpt-5 --reps 2
```

Outputs are written to `outputs/{model}_run-{N}__{dataset}.jsonl`.

## Evaluation

Run from `stocks_synthetic_dataset/evaluate/`:

```bash
python gen_stocks_v6.py
```

This regenerates `stocks_accuracy_comparison_v6.csv` using canonical v2 labels,
including all available Simple MAS runs. Current controlled-baseline snapshots
have a separate offline scorer documented in the final experiment guide.
