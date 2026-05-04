"""
Load ``grpo_trainer`` with the same Hydra overrides as ``benchmark_eval.runner.build_hydra_cli``.
Used by two-phase MAS scripts (no subprocess / Ray).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .datasets_aflow import DatasetSpec
from .runner import _rollout_gpu_memory_utilization, orchestrator_root


def _dummy_parquet_pair() -> Tuple[str, str]:
    """Minimal parquet files so Hydra ``data.train_files`` / ``val_files`` resolve like the benchmark harness."""
    td = Path(tempfile.mkdtemp(prefix="mas_bench_hydra_"))
    train_p = td / "train.parquet"
    val_p = td / "val.parquet"
    df = pd.DataFrame(
        [
            {
                "prompt": "placeholder",
                "reward_model": {"ground_truth": "y"},
            }
        ]
    )
    df.to_parquet(train_p, index=False)
    df.to_parquet(val_p, index=False)
    return str(train_p), str(val_p)


def build_benchmark_hydra_overrides(
    *,
    dataset_spec: "DatasetSpec",
    agent_model: str,
    train_parquet: str,
    val_parquet: str,
    mas_export_dir: str,
    n_gpus: int = 1,
    tp_size: int = 1,
) -> List[str]:
    """CLI override list matching ``runner.build_hydra_cli`` (excluding the executable / config-path / config-name)."""
    init_list = ",".join(dataset_spec.init_archive)
    return [
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
        f"trainer.mas_export_dir={mas_export_dir}",
        f"actor_rollout_ref.model.path={dataset_spec.orchestrator_hf}",
        "actor_rollout_ref.model.trust_remote_code=true",
        "data.trust_remote_code=true",
        f"azr.problem_type={dataset_spec.problem_type}",
        "azr.mas_r1.multiply_processes=0",
        f"azr.mas_r1.no_decompose={str(dataset_spec.no_decompose).lower()}",
        f"azr.mas_r1.agent.model_name={agent_model}",
        f"azr.mas_r1.agent.init_archive=[{init_list}]",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={_rollout_gpu_memory_utilization()}",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "reward_model.enable=False",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "algorithm.use_kl_in_reward=false",
        "actor_rollout_ref.vllm_inference_only=true",
    ]


def load_grpo_config_for_benchmark(
    dataset_spec: "DatasetSpec",
    agent_model: str,
    *,
    n_gpus: int = 1,
    tp_size: int = 1,
    mas_export_dir: str = "/tmp/mas_bench_export_unused",
    train_parquet: Optional[str] = None,
    val_parquet: Optional[str] = None,
) -> OmegaConf:
    """
    Compose ``grpo_trainer`` with benchmark-equivalent overrides.
    If ``train_parquet`` / ``val_parquet`` are omitted, creates a temporary dummy pair.
    """
    if train_parquet is None or val_parquet is None:
        train_parquet, val_parquet = _dummy_parquet_pair()

    cfg_dir = orchestrator_root() / "mas_r1_reasoner" / "configs"
    overrides = build_benchmark_hydra_overrides(
        dataset_spec=dataset_spec,
        agent_model=agent_model,
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        mas_export_dir=mas_export_dir,
        n_gpus=n_gpus,
        tp_size=tp_size,
    )
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir.resolve())):
        cfg = compose(config_name="grpo_trainer", overrides=overrides)
    return cfg
