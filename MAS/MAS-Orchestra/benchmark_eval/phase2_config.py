"""
Phase 2 (``eval_mas_shared``) configuration only.

Loads ``grpo_trainer.yaml`` with OmegaConf — **no Hydra compose**, no dummy parquet, no
``actor_rollout_ref.model.path`` pointing at Phase-1 orchestrator checkpoints. Phase 2 only needs
``azr.mas_r1.agent`` (and related flags) for ``setup_global_variables`` / AgentSystem.
"""
from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf, open_dict

from .datasets_aflow import DatasetSpec
from .runner import orchestrator_root


def load_grpo_config_phase2(spec: DatasetSpec, agent_model: str) -> Any:
    """
    Minimal runtime config for benchmark Phase 2: sub-agent MAS execution over ``shared_mas`` cache.

    Does not configure VERL rollout, actor weights, or tokenizer loading.
    """
    cfg_path = orchestrator_root() / "mas_r1_reasoner" / "configs" / "grpo_trainer.yaml"
    cfg = OmegaConf.load(str(cfg_path.resolve()))
    with open_dict(cfg):
        cfg.azr.problem_type = spec.problem_type
        cfg.azr.mas_r1.no_decompose = spec.no_decompose
        cfg.azr.mas_r1.multiply_processes = 0
        cfg.azr.mas_r1.agent.model_name = agent_model
        cfg.azr.mas_r1.agent.init_archive = list(spec.init_archive)
        # Satisfies ``BaseDatasetProcessor`` ``config.actor_rollout_ref.model.get`` — never load weights in Phase 2.
        if "actor_rollout_ref" not in cfg or cfg.actor_rollout_ref is None:
            cfg.actor_rollout_ref = OmegaConf.create({})
        if "model" not in cfg.actor_rollout_ref or cfg.actor_rollout_ref.model is None:
            cfg.actor_rollout_ref.model = OmegaConf.create({})
        cfg.actor_rollout_ref.model.path = "phase2-no-orchestrator-weights"
        cfg.actor_rollout_ref.model.lora_rank = 0
        # IGSM detection uses ``str(config.data.train_files)`` — avoid substring ``igsm``.
        if "data" not in cfg or cfg.data is None:
            cfg.data = OmegaConf.create({})
        cfg.data.train_files = "phase2-unused.parquet"
        cfg.data.val_files = cfg.data.get("val_files") or "phase2-unused-val.parquet"
    return cfg
