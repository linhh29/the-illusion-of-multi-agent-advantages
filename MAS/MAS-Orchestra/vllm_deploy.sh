# Max concurrent sequences processed by the engine (batching / scheduling cap).
# --served-model-name must match benchmark_eval.datasets_aflow (SERVED_MODEL_NAME_HARMONY_*); benchmark sets MAS_ORCHESTRATOR_MODEL accordingly.
#
# BrowseComp+ uses the same DoM-low orchestrator as GPQA/HLEMATH/SWE/STOCKS (see benchmark_eval.datasets_aflow).

export CUDA_VISIBLE_DEVICES=3,4,5,6
export VLLM_USE_V1=0
# Avoid appending to ~/.config/vllm/usage_stats.json (fails with Errno 28 when / is full).
export VLLM_NO_USAGE_STATS=1

# Triton/torch.inductor may run gcc under repo dir (e.g. tmp*/main.c). If gcc exits 1 (disk, libcudart,
# headers), vLLM workers crash with BackendCompilerFailed. These reduce JIT/compile paths:
export TORCH_COMPILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TMPDIR:=$SCRIPT_DIR/.tmp_vllm}"
mkdir -p "${TMPDIR}"

# Override with env VLLM_MODEL_PATH if your merged checkpoint lives elsewhere.
: "${VLLM_MODEL_PATH:=$SCRIPT_DIR/checkpoints/harmony-grpo-7b-global-step-180-merged}"

# Checkpoint uses YaRN (rope_scaling) 2x over pretrained 32k -> max_position_embeddings 65536.
# Total context = prompt + completion; keep data.max_prompt_length + data.max_response_length <= 65536.
vllm serve "$VLLM_MODEL_PATH" \
  --tensor-parallel-size 4 --host 0.0.0.0 --port 8000 \
  --max-num-seqs 32 \
  --max-model-len 65536 \
  --enforce-eager \
  --served-model-name harmony-grpo-7b-global-step-180-merged
