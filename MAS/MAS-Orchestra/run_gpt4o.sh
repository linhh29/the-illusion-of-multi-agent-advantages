#!/usr/bin/env bash
# Phase 1: generate shared MAS (orchestrator / vLLM) — use generate_mas.sh / generate_mas_shared separately.
# Phase 2: evaluate cached MAS with sub-agent LLMs (OpenAI-compatible API); writes results_{RUN_SUFFIX}/.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export OPENAI_API_KEY=
# Together (sub-agent gpt-oss-120b): eval_mas_shared uses TogetherCompletionSampler; credentials via
# TOGETHER_API_KEY + TOGETHER_BASE_URL, or aliases OPENAI_TOGETHER_KEY + OPENAI_TOGETHER_API_BASE (see
# mas_r1_reasoner/agents/sampler/api_usage_tracker.build_together_openai_client_kwargs).
export OPENAI_TOGETHER_KEY=
export OPENAI_TOGETHER_API_BASE=
# In-flight sub-agent HTTP calls (shared semaphore across all samples in one process).
# Only speeds things up when the MAS ``forward`` schedules multiple LLM calls concurrently;
# if the graph is strictly sequential, raising this alone will not help much.
export MAS_API_MAX_CONCURRENCY="${MAS_API_MAX_CONCURRENCY:-50}"
# Sub-agent OpenAI HTTP read timeout (seconds). BrowseComp+ prompts are huge — 120s often yields empty completions; use 600 when running BCP.
# Baseline for gpt-4o; ``gpt-5*`` / ``gpt-oss*`` sub-agents multiply this by 3 in code (see http_timeout_scale.py).
export MAS_OPENAI_CHAT_TIMEOUT_SEC="${MAS_OPENAI_CHAT_TIMEOUT_SEC:-600}"
# SWE-Bench Lite: ``swebench.harness.run_evaluation`` (Docker) wall-clock cap per sample (vendor swe_utils).
export MAS_SWEBENCH_EVAL_TIMEOUT_SEC="${MAS_SWEBENCH_EVAL_TIMEOUT_SEC:-600}"

# --- Phase 2: eval_mas_shared (reads shared_mas/<dataset>/samples/ for all RUN_SUFFIX; writes results_${RUN_SUFFIX}/...) ---
# ``--jobs`` = how many *samples* run in parallel (asyncio + thread pool). This is usually what you
# want to increase when the run feels slow. Default below is >1; override with e.g. BENCHMARK_JOBS=1.
BENCHMARK_JOBS="${BENCHMARK_JOBS:-50}"
# JSONL default: MAS-Orchestra/data/datasets/gpqa_test.jsonl ; override with AFLOW_DATA_DIR below
# AFLOW_DATA_DIR="${AFLOW_DATA_DIR:-}"

# Order: for each RUN_SUFFIX, outer loop runs each dataset (one python process per dataset); the next
# dataset does not start until the previous ``python -m benchmark_eval.eval_mas_shared`` exits.
AGENT_MODEL=gpt-4o
for RUN_SUFFIX in run1 run2 run3; do
  for dataset in GPQA HLEMATH STOCKS SWE-Bench-Lite BrowseComp+; do
    docker ps -aq --filter "name=sweb.eval" | xargs -r docker rm -f
    echo "========== dataset=${dataset}  RUN_SUFFIX=${RUN_SUFFIX}  AGENT_MODEL=${AGENT_MODEL} =========="
    ARGS=(
      --run-suffix "$RUN_SUFFIX"
      --datasets "$dataset"
      --agent-model "$AGENT_MODEL"
      --jobs "$BENCHMARK_JOBS"
    )
    # Optional: uncomment to cap samples (debug)
    # ARGS+=( --limit 10 )
    # Optional: uncomment if JSONL is not under data/datasets/
    # ARGS+=( --aflow-data-dir "$AFLOW_DATA_DIR" )

    python -m benchmark_eval.eval_mas_shared "${ARGS[@]}"
  done
done
