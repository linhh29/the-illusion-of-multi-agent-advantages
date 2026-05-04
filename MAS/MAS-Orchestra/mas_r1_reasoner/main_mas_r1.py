# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import mas_r1_reasoner.torch_tensordict_compat  # noqa: F401 — before verl/tensordict

import logging
import os

import hydra
import ray

from mas_r1_reasoner.trainer.mas_r1_trainer import MASR1Trainer
from mas_r1_reasoner.rewards.mas_r1_reward_manager import setup_reward_manager as setup_reward_manager_mas_r1
from mas_r1_reasoner.rewards.direct_reward_manager import setup_reward_manager as setup_reward_manager_direct
from mas_r1_reasoner.rewards.harmony_reward_manager import setup_reward_manager as setup_reward_manager_harmony


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    benchmark = bool(os.environ.get("MAS_BENCHMARK_SAMPLE_ID"))
    if not ray.is_initialized():
        init_kw = dict(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )
        if benchmark:
            init_kw["include_dashboard"] = False
            init_kw["logging_level"] = logging.ERROR
            init_kw["log_to_driver"] = False
        ray.init(**init_kw)

    # Benchmark is inference-only (total_epochs=0); VERL still needs Ray for vLLM workers, but we skip
    # the extra TaskRunner Ray actor and run the trainer driver in-process.
    if benchmark:
        _mas_r1_task_run(config)
    else:
        runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))


def _mas_r1_task_run(config) -> None:
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_to_local

    if not os.environ.get("MAS_BENCHMARK_SAMPLE_ID"):
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get('use_shm', False))

    # instantiate tokenizer
    from verl.utils import hf_processor, hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    # vLLM LoRA version gate: do not import verl.utils.vllm_utils here — it eagerly
    # imports all of vLLM and can fail (e.g. transformers/vLLM skew) before workers start.
    if config.actor_rollout_ref.rollout.name in ["vllm"]:
        if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
            from importlib.metadata import PackageNotFoundError, version as pkg_version

            from packaging.version import Version

            try:
                vllm_v = Version(pkg_version("vllm"))
            except PackageNotFoundError:
                vllm_v = Version("0")
            if vllm_v < Version("0.7.3"):
                raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

    # define worker classes
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.critic.strategy in ["fsdp", "fsdp2"]
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # use reference model
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    # Choose reward manager based on problem_type
    problem_type = config.azr.get('problem_type')
    if problem_type == 'direct' or problem_type == 'mcp':
        reward_fn = setup_reward_manager_direct(tokenizer=tokenizer, num_examine=0, config=config)
        # Note that we always use function-based RM for validation
        val_reward_fn = setup_reward_manager_direct(tokenizer=tokenizer, num_examine=1, config=config)
    elif 'harmony' in problem_type:
        reward_fn = setup_reward_manager_harmony(tokenizer=tokenizer, num_examine=0, config=config)
        # Note that we always use function-based RM for validation
        val_reward_fn = setup_reward_manager_harmony(tokenizer=tokenizer, num_examine=1, config=config)
    else:
        reward_fn = setup_reward_manager_mas_r1(tokenizer=tokenizer, num_examine=0, config=config)
        # Note that we always use function-based RM for validation
        val_reward_fn = setup_reward_manager_mas_r1(tokenizer=tokenizer, num_examine=1, config=config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    from verl.utils.dataset.rl_dataset import collate_fn

    trainer = MASR1Trainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        _mas_r1_task_run(config)


if __name__ == "__main__":
    main()
