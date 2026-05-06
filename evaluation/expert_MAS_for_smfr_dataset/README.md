# Expert MAS for Smfr Dataset

A manually designed multi-agent system (MAS) for the smfr trading analysis benchmark. The pipeline decomposes each problem into structured sub-tasks handled by three specialized agents.

## Pipeline overview

```
Phase 1  MetaAgent        — parse problem into structured parameters (1 LLM call)
Phase 2  ExtractAgent     — extract each investor's transactions + prices (parallel)
Phase 3  CalculateAgent   — compute P&L and required target price per investor (parallel)
Phase 4  ExtractAgent     — find valid dates where the target price is met (parallel)
Phase 5  (pure Python)    — aggregate across investors and return final answer
```

Each agent calls the same underlying model. All LLM calls return structured JSON via Pydantic.

## Setup

```bash
pip install -r requirements.txt
```

For Google Gemini models, also install:
```bash
pip install google-genai>=1.0.0
```

## Configuration

Set credentials via environment variables before running:

```bash
# OpenAI (or any OpenAI-compatible gateway)
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional; defaults to OpenAI

# Google Gemini via Vertex AI
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
# Authenticate: gcloud auth application-default login
```

## Running inference

```bash
# Quick test (2 samples)
python run_inference.py \
    --input <data_dir>/smfr_test.jsonl \
    --model gpt-4.1 \
    --test

# First 20 samples
python run_inference.py \
    --input <data_dir>/smfr_test.jsonl \
    --model gpt-4.1 \
    --slice 20

# Full run with 10 parallel pipelines
python run_inference.py \
    --input <data_dir>/smfr_test.jsonl \
    --model gpt-4.1 \
    --concurrency 10
```

### Supported model aliases

| Alias | Full model ID |
|---|---|
| `gpt-4o` | `gpt-4o-2024-08-06` |
| `gpt-4.1` | `gpt-4.1-2025-04-14` |
| `o3` | `o3-2025-04-16` |
| `o4-mini` | `o4-mini-2025-04-16` |
| `gpt-5` | `gpt-5-2025-08-07` |
| `gemini-flash` | `gemini-2.5-flash` |
| `gemini-pro` | `gemini-2.5-pro` |
| `gemini-flash-lite` | `gemini-2.5-flash-lite-preview-06-17` |

Any full model ID not in this table is passed through unchanged.

### Output format

Each run produces two files:

- `{model}_run-0__{dataset_name}.jsonl` — one record per sample:
  ```json
  {
    "input":  { "<original sample fields>" },
    "model":  "gpt-4.1",
    "output": {
      "answer": {
        "investor_dates": {"Alice": ["January 5, 2026", ...], ...},
        "comparison":     {"Alice": "January 5, 2026", ...},
        "answer":         ["Alice"]
      }
    },
    "stats":  { "llm_calls": 12, "total_tokens": 48200, "cost_usd": {...}, ... },
    "trace":  { "<intermediate sub-agent outputs>" }
  }
  ```
- `{model}_run-0__{dataset_name}_stats.json` — aggregated token and cost summary
- `{model}_run-0__{dataset_name}_eval.json` — accuracy results

## Evaluating existing outputs

Evaluation uses the shared script in `../evaluate/`:

```bash
python ../evaluate/evaluate_smfr_answer_code.py \
    --reference <data_dir>/smfr_test.jsonl \
    --model-output gpt-4.1_run-0__smfr_test.jsonl
```

## Debugging traces

To inspect intermediate sub-agent outputs for a run:

```bash
# First produce a short trace
python run_inference.py \
    --input <data_dir>/smfr_test.jsonl \
    --model gpt-4.1 --test --output debug_out.jsonl

# Then inspect
python debug_trace.py debug_out.jsonl
```

## File structure

```
expert_MAS_for_smfr_dataset/
├── agents.py           # ModelClient, MetaAgent, ExtractAgent, CalculateAgent
├── pipeline.py         # Orchestration logic (phases 1–5)
├── run_inference.py    # CLI entry point
├── debug_trace.py      # Trace inspection tool
├── prompts/
│   ├── meta_agent.txt  # Prompt for MetaAgent
│   ├── extract.txt     # Prompt for ExtractAgent
│   └── calculate.txt   # Prompt for CalculateAgent
└── requirements.txt
```
