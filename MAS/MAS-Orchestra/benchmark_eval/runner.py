"""
Subprocess runner: invokes mas_r1_reasoner.main_mas_r1 with Hydra overrides for one sample.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .datasets_aflow import DatasetSpec, orchestrator_served_model_id, write_pair_parquets


def _resolve_mas_api_max_concurrency(env: Dict[str, str]) -> str:
    """Prefer MAS_API_MAX_CONCURRENCY; else MAX_CONCURRENT (e.g. run_gpqa.sh); default 50."""
    for key in ("MAS_API_MAX_CONCURRENCY", "MAX_CONCURRENT"):
        v = env.get(key)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return "50"


def orchestrator_root() -> Path:
    return Path(__file__).resolve().parent.parent


def shared_mas_root(run_suffix: str) -> Path:
    """
    Phase-1 Harmony MAS code cache: ``{repo}/shared_mas_{suffix}/`` (sibling to ``results_{suffix}/``).
    """
    return orchestrator_root() / f"shared_mas_{run_suffix}"


def shared_mas_eval_read_root() -> Path:
    """
    Unified Harmony MAS JSON tree: ``{repo}/shared_mas/<Dataset>/samples/*.json``.

    Phase-1 ``generate_mas_shared`` writes here by default; phase-2 ``eval_mas_shared`` loads from
    here for every ``--run-suffix`` and every ``--agent-model`` (only ``results_{suffix}/`` is per-run).

    Override with env ``MAS_SHARED_MAS_DIR`` or ``SHARED_MAS_ROOT`` (path to the parent of
    ``<Dataset>/``). Per-run trees ``shared_mas_{suffix}/`` remain only for legacy helpers
    (``fill_shared_mas_from_peer_runs``, ``generate_mas_shared_cross_run``).
    """
    env = (os.environ.get("MAS_SHARED_MAS_DIR") or os.environ.get("SHARED_MAS_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return orchestrator_root() / "shared_mas"


def _rollout_gpu_memory_utilization() -> str:
    """vLLM fraction of per-GPU memory. Hybrid VERL keeps FSDP weights on the same GPUs, so 0.85 often OOMs."""
    for key in ("MAS_ROLLOUT_GPU_MEMORY_UTIL", "VLLM_GPU_MEMORY_UTILIZATION"):
        v = os.environ.get(key)
        if v is not None and str(v).strip() != "":
            return str(float(v))
    return "0.5"


def _apply_short_ray_tmpdir(merged_env: Dict[str, str], sample_id: str) -> None:
    """
    Ray session + AF_UNIX sockets must stay under ~107 bytes. Do not follow TMPDIR (often under a long
    project path). Pin each benchmark subprocess to /tmp/masr<8 hex> (unique per sample_id).
    """
    token = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8]
    p = Path("/tmp") / f"masr{token}"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    s = str(p)
    merged_env["RAY_TMPDIR"] = s
    merged_env["TMPDIR"] = s


def _apply_cuda_visible_for_mas_subprocess(merged_env: Dict[str, str], gpu_id: Optional[int]) -> None:
    """
    Pin VERL/Ray/vLLM to specific device(s) for parallel benchmark shells.

    Priority:
    1. Explicit gpu_id from CLI → CUDA_VISIBLE_DEVICES=<gpu_id> (single GPU index).
    2. Else if CUDA_VISIBLE_DEVICES is already set → unchanged (user override, e.g. "0,1" for TP).
    3. Else MAS_GPU_ID or BENCHMARK_GPU_ID → CUDA_VISIBLE_DEVICES=<value> (comma-separated allowed).
    """
    if gpu_id is not None:
        merged_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return
    cur = (merged_env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cur:
        return
    for key in ("MAS_GPU_ID", "BENCHMARK_GPU_ID"):
        raw = merged_env.get(key)
        if raw is not None and str(raw).strip() != "":
            merged_env["CUDA_VISIBLE_DEVICES"] = str(raw).strip()
            return


def build_hydra_cli(
    *,
    dataset_spec: DatasetSpec,
    train_parquet: str,
    val_parquet: str,
    export_dir: str,
    agent_model: str,
    usage_log: str,
    n_gpus: int = 1,
    tp_size: int = 1,
) -> List[str]:
    init_list = ",".join(dataset_spec.init_archive)
    rr = orchestrator_root()
    cfg_dir = rr / "mas_r1_reasoner" / "configs"
    return [
        sys.executable,
        "-u",
        "-m",
        "mas_r1_reasoner.main_mas_r1",
        f"--config-path={cfg_dir}",
        "--config-name=grpo_trainer",
        "data.raw_data=True",
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        "data.train_batch_size=1",
        "data.shuffle=False",
        "trainer.total_epochs=0",
        "trainer.resume_mode=disable",
        "trainer.val_before_train=True",
        "trainer.logger=[console]",
        "trainer.project_name=bench_eval",
        "trainer.experiment_name=mas_benchmark",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        f"trainer.mas_export_dir={export_dir}",
        f"actor_rollout_ref.model.path={dataset_spec.orchestrator_hf}",
        # Harmony / custom HF checkpoints need remote modeling code + tokenizer
        "actor_rollout_ref.model.trust_remote_code=true",
        "data.trust_remote_code=true",
        f"azr.problem_type={dataset_spec.problem_type}",
        f"azr.mas_r1.multiply_processes=0",
        f"azr.mas_r1.no_decompose={str(dataset_spec.no_decompose).lower()}",  # hydra bool
        f"azr.mas_r1.agent.model_name={agent_model}",
        f"azr.mas_r1.agent.init_archive=[{init_list}]",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={_rollout_gpu_memory_utilization()}",
        # One completion per prompt for benchmark (default grpo_trainer.yaml has n=8 for GRPO).
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "reward_model.enable=False",
        # No PPO training in benchmark: omit reference policy so HF weights are not loaded twice for KL.
        "actor_rollout_ref.actor.use_kl_loss=false",
        "algorithm.use_kl_in_reward=false",
        # No FSDP actor; orchestrator inference runs on vLLM only (see verl ActorRolloutRefWorker + RayPPOTrainer.init_workers).
        "actor_rollout_ref.vllm_inference_only=true",
    ]


def run_one_sample(
    *,
    record: Dict,
    dataset_spec: DatasetSpec,
    agent_model: str,
    tmp_dir: Path,
    export_dir: Path,
    usage_log_path: Path,
    sample_id: str,
    env: Optional[Dict[str, str]] = None,
    n_gpus: int = 1,
    tp_size: int = 1,
    gpu_id: Optional[int] = None,
    orchestrator_openai_base: Optional[str] = None,
    orchestrator_model: Optional[str] = None,
) -> int:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    train_p, val_p = write_pair_parquets(tmp_dir, record)
    usage_log_path.parent.mkdir(parents=True, exist_ok=True)
    if usage_log_path.exists():
        usage_log_path.unlink()

    merged_env = os.environ.copy()
    merged_env["PYTHONPATH"] = str(orchestrator_root()) + os.pathsep + merged_env.get("PYTHONPATH", "")
    merged_env["MAS_BENCHMARK_SAMPLE_ID"] = sample_id
    merged_env["MAS_API_USAGE_LOG"] = str(usage_log_path)
    merged_env["MAS_API_MAX_CONCURRENCY"] = _resolve_mas_api_max_concurrency(merged_env)
    if orchestrator_openai_base:
        merged_env["MAS_ORCHESTRATOR_OPENAI_BASE"] = orchestrator_openai_base
    if orchestrator_model:
        merged_env["MAS_ORCHESTRATOR_MODEL"] = orchestrator_model
    # External orchestrator: pin model id to the dataset checkpoint (see datasets_aflow.SERVED_MODEL_NAME_*).
    _orch_base = (merged_env.get("MAS_ORCHESTRATOR_OPENAI_BASE") or "").strip()
    _orch_model = (merged_env.get("MAS_ORCHESTRATOR_MODEL") or "").strip()
    if _orch_base and not _orch_model:
        merged_env["MAS_ORCHESTRATOR_MODEL"] = orchestrator_served_model_id(dataset_spec.orchestrator_hf)
    if env:
        merged_env.update(env)
    _apply_short_ray_tmpdir(merged_env, sample_id)
    merged_env.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    merged_env.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    merged_env.setdefault("RAY_DEDUP_LOGS", "1")
    merged_env.setdefault("RAY_BACKEND_LOG_LEVEL", "error")
    _apply_cuda_visible_for_mas_subprocess(merged_env, gpu_id)

    cmd = build_hydra_cli(
        dataset_spec=dataset_spec,
        train_parquet=train_p,
        val_parquet=val_p,
        export_dir=str(export_dir),
        agent_model=agent_model,
        usage_log=str(usage_log_path),
        n_gpus=n_gpus,
        tp_size=tp_size,
    )

    rr = orchestrator_root()
    proc = subprocess.run(
        cmd,
        cwd=str(rr),
        env=merged_env,
        text=True,
    )
    return proc.returncode
