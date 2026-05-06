"""
AgentSystem initialization for benchmark Phase 2 only (no VERL / PPO imports).

``mas_r1_reasoner.trainer.utils.helper`` pulls in the full training stack; Phase 2 eval must not depend on it.
"""
from __future__ import annotations

from typing import Any

from mas_r1_reasoner.agents.common import main_rank_print
from mas_r1_reasoner.agents.agent_system_async import AsyncAgentSystem
from mas_r1_reasoner.agents.shared_vars import get_global


def initialize_mas_r1_agent_system_phase2(trainer_instance: Any, config: Any) -> None:
    """Same behavior as ``initialize_mas_r1_agent_system`` in helper.py, without importing VERL."""
    if trainer_instance.agent_system is None:
        agent_config = trainer_instance.mas_r1_config.get("agent", {})
        main_rank_print(f"MAS agent config: {agent_config}")
        multiply_processes = get_global("global_multiply_processes")
        if multiply_processes == 0:
            main_rank_print("Initializing ASYNC AgentSystem (Phase 2)...")
            local_exec = getattr(trainer_instance, "_mas_eval_local_exec", False)
            if local_exec:
                main_rank_print("Benchmark/local mode: in-process MAS execution (no Ray workers).")
            trainer_instance.agent_system = AsyncAgentSystem(agent_config, local_exec=local_exec)
            main_rank_print("✓ Async AgentSystem initialized with agent configuration")
        else:
            raise ValueError("Process execution is not supported")
    else:
        main_rank_print("AgentSystem already initialized")
