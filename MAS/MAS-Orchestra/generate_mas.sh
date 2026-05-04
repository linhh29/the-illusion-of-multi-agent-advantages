#!/usr/bin/env bash
# Two-phase MAS (shared_mas cache + sub-agent evaluation) — independent of benchmark_eval/cli subprocess+Ray path.
# Phase1 writes: {repo}/shared_mas/<Dataset>/samples/ (consistent with eval_mas_shared read path; override with MAS_SHARED_MAS_DIR).
#
# Phase 1 generate_mas_shared --jobs:
#   Max concurrent AsyncOpenAI completions per batch (batched asyncio.gather), avoids pre-creating massive coroutines.
#   Single process, no Ray, no VERL, no separate worker processes.
#   If RayPPOTrainer / benchmark_eval/cli subprocess appears, you're not running this script.
#
# Phase 2 eval_mas_shared --jobs:
#   Concurrent sample evaluation (asyncio + thread pool for sync execution); still no Ray worker (local_exec).

# Phase 1: requires `vllm serve` Orchestra and MAS_ORCHESTRATOR_OPENAI_BASE set
# Default JSONL: MAS-Orchestra/data/datasets/ (copy *_test.jsonl from AFlow) or specify via --aflow-data-dir
# Dataset names match benchmark_eval.datasets_aflow.DATASETS keys: GPQA, HLEMATH, SWE-Bench-Lite, BrowseComp+, STOCKS
# "BrowseComp+"
for dataset in "BrowseComp+"
do
  for run_num in 1 2 3
  do
    python -m benchmark_eval.generate_mas_shared \
      --run-suffix run${run_num} --datasets ${dataset} \
      --jobs 32 \
      --orchestrator-model harmony-grpo-7b-global-step-180-merged \
      --orchestrator-openai-base http://127.0.0.1:8000/v1 \
      --resume-harmony \
      --harmony-retry-max 1000
  done
done



# # 阶段二：需 OPENAI_API_KEY 等（子 agent 调 API）
# # --jobs：同时跑几条「样本」；单条样本内 sub-agent 默认顺序执行（并发=1），若要并行 sub-agent 可设 MAS_API_MAX_CONCURRENCY 或 --api-max-concurrency
# python -m benchmark_eval.eval_mas_shared \
#   --run-suffix run1 --datasets GPQA --jobs 4 \
#   --agent-model gpt-4o gpt-5
