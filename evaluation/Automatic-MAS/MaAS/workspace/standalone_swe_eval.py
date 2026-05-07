#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone SWE-Bench CoT Evaluation Script
独立的 SWE-Bench CoT 评估脚本
"""

import asyncio
import json
import re
import os
import shutil
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse
from pathlib import Path
from collections import Counter

from maas.llm import LLM
from maas.configs.llm_config import LLMConfig
from maas.context import Context
from maas.logs import logger

# Import SWE evaluation utilities
from maas.ext.maas.benchmark.swe_utils import run_swebench_evaluation


@dataclass
class SWECoTConfig:
    """SWE CoT 配置"""
    num_samples: int = 3  # CoT-SC 生成的样本数量
    temperature: float = 0.7  # 温度参数


AGENTLESS_REPAIR = """
You must make sure 
(1) the patch is correct and can be applied to the code. 
(2) Please note that the patch REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
(3) Wrap each patch in a code block as shown in the example above. If you have multiple patchs, use a separate code block for each one. For example,
(5) Your patch must be significant enough to change the PASS or FAIL status of potential test cases. DO NOT include trivial patch like change the doc string, add empty lines, add comments or change the vairable names as these trivial patches cannot change a failed test cases to passed.
(6) The patch must be COMPLETE CODE and without any syntax error. Please implement complete, reliable, reusable code snippets.
(7) A user will run unix's patch program directly to apply the patch, so please make sure the patch is correct and directly runnable by the unix's patch program.
"""

SWE_SC_ENSEMBLE_PROMPT = """Given the following bug fix problem and multiple proposed patches, select the best one.

## Problem Context
{problem_context}

## Proposed Patches
{patches}

## Task
Analyze each patch and select the one most likely to correctly fix the bug.
Consider:
1. Does the patch address the root cause?
2. Is the patch minimal and focused?
3. Does it follow good coding practices?
4. Are there any potential side effects?

Output the letter (A, B, C, etc.) of the best patch, then provide the complete patch content.

## Your Selection
"""


class SWECoT:
    """SWE-Bench CoT 操作子"""
    
    def __init__(self, llm: Any):
        self.llm = llm
    
    async def generate_patch(self, problem: Dict[str, Any]) -> str:
        """
        生成单个补丁
        
        Args:
            problem: SWE-Bench 问题字典
            
        Returns:
            生成的补丁（diff 格式）
        """
        # 使用 problem["text"] 而不是 problem["problem_statement"]
        # problem["text"] 包含完整的代码库上下文和问题描述
        input_text = problem.get("text", "")
        
        # 添加 AGENTLESS_REPAIR 指令，与 swe.py 保持一致
        input_text = input_text + AGENTLESS_REPAIR
        
        try:
            response = await self.llm.aask(input_text)
            return self._extract_patch(response)
        except Exception as e:
            logger.error(f"Patch generation failed: {e}")
            return ""
    
    async def generate_multi_patches(
        self, 
        problem: Dict[str, Any], 
        num_samples: int = 3
    ) -> List[str]:
        """
        生成多个补丁（用于 CoT-SC）
        """
        tasks = [self.generate_patch(problem) for _ in range(num_samples)]
        responses = await asyncio.gather(*tasks)
        return [r for r in responses if r]  # 过滤空响应
    
    def _extract_patch(self, response: str) -> str:
        """从响应中提取 diff 补丁"""
        # 尝试提取 ```diff ... ``` 代码块
        diff_match = re.search(r'```diff\s*(.*?)```', response, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()
        
        # 尝试提取 ``` ... ``` 代码块（可能没有 diff 标记）
        code_match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
        if code_match:
            content = code_match.group(1).strip()
            # 检查是否看起来像 diff
            if content.startswith('---') or content.startswith('diff ') or '@@ ' in content:
                return content
        
        # 尝试直接找 diff 内容
        lines = response.split('\n')
        diff_lines = []
        in_diff = False
        
        for line in lines:
            if line.startswith('---') or line.startswith('diff '):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
                # 检测 diff 结束（遇到非 diff 行）
                if line and not any(line.startswith(p) for p in ['---', '+++', '@@', '+', '-', ' ', 'diff ']):
                    if not line.strip():
                        continue
                    break
        
        if diff_lines:
            return '\n'.join(diff_lines).strip()
        
        return response.strip()


class SWECoTSC:
    """SWE-Bench CoT-SC 集成器"""
    
    def __init__(self, llm: Any):
        self.llm = llm
    
    async def ensemble(self, patches: List[str], problem: Dict[str, Any]) -> str:
        """
        使用自洽性集成选择最佳补丁
        """
        if not patches:
            return ""
        
        if len(patches) == 1:
            return patches[0]
        
        # 如果所有补丁相同，直接返回
        if len(set(patches)) == 1:
            return patches[0]
        
        # 构建补丁文本
        patches_text = ""
        for idx, patch in enumerate(patches):
            letter = chr(65 + idx)  # A, B, C...
            patches_text += f"### Patch {letter}:\n```diff\n{patch}\n```\n\n"
        
        # 使用 problem["text"] 获取完整上下文，如果没有则使用 problem_statement
        problem_context = problem.get("text", problem.get("problem_statement", ""))
        
        prompt = SWE_SC_ENSEMBLE_PROMPT.format(
            problem_context=problem_context,
            patches=patches_text,
        )
        
        try:
            response = await self.llm.aask(prompt)
            
            # 尝试从响应中提取选择的补丁
            # 首先看是否提到了某个字母
            letter_match = re.search(r'\b([A-Z])\b', response)
            if letter_match:
                idx = ord(letter_match.group(1)) - 65
                if 0 <= idx < len(patches):
                    return patches[idx]
            
            # 否则尝试提取响应中的 diff
            diff_match = re.search(r'```diff\s*(.*?)```', response, re.DOTALL)
            if diff_match:
                return diff_match.group(1).strip()
            
            # 降级：使用投票选择最常见的补丁
            return Counter(patches).most_common(1)[0][0]
            
        except Exception as e:
            logger.error(f"SC ensemble failed: {e}")
            # 降级：返回第一个补丁
            return patches[0]


class SWEEvaluator:
    """SWE-Bench 评估器"""
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        cot_config: SWECoTConfig = None,
        config_path: str = None,
        output_dir: str = None,
    ):
        self.cot_config = cot_config or SWECoTConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("swe_eval_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Context 和 LLM
        if config_path:
            os.environ['MAAS_CONFIG_PATH'] = config_path
        
        context = Context()
        
        if llm_config:
            logger.info(f"Using LLM config: model={llm_config.model}")
            self.llm = context.llm_with_cost_manager_from_llm_config(llm_config)
        else:
            logger.info("Using default LLM config from config2.yaml")
            self.llm = context.llm()
        
        # 初始化操作子
        self.cot = SWECoT(self.llm)
        self.cot_sc = SWECoTSC(self.llm)
    
    async def evaluate_cot(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """使用单个 CoT 评估"""
        
        instance_id = problem.get("instance_id", "unknown")
        
        # 生成补丁
        logger.info(f"[{instance_id}] Generating patch...")
        patch = await self.cot.generate_patch(problem)
        
        if not patch:
            logger.warning(f"[{instance_id}] Empty patch generated")
            return {
                "instance_id": instance_id,
                "patch": "",
                "score": 0.0,
                "method": "CoT",
                "error": "Empty patch"
            }
        
        # 运行 SWE-Bench 评估
        logger.info(f"[{instance_id}] Running SWE-Bench evaluation...")
        
        # 在评测之前，删除对应的 log 目录，避免旧日志干扰
        try:
            instance_prefix = instance_id.replace("-", "_")
            run_id = f"{instance_prefix}_cot_standalone"
            log_dir = self.output_dir / "logs" / "run_evaluation" / run_id
            if log_dir.exists():
                logger.info(f"[{instance_id}] Removing existing log dir before evaluation: {log_dir}")
                shutil.rmtree(log_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"[{instance_id}] Failed to clean log dir before evaluation: {e}")

        score = await self._run_evaluation(problem, patch, "cot")

        
        return {
            "instance_id": instance_id,
            "patch": patch,
            "score": score,
            "method": "CoT"
        }
    
    async def evaluate_cot_sc(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """使用 CoT-SC 评估"""
        instance_id = problem.get("instance_id", "unknown")
        
        # 生成多个补丁
        logger.info(f"[{instance_id}] Generating {self.cot_config.num_samples} patches...")
        patches = await self.cot.generate_multi_patches(
            problem, 
            self.cot_config.num_samples
        )
        
        if not patches:
            logger.warning(f"[{instance_id}] No patches generated")
            return {
                "instance_id": instance_id,
                "patches": [],
                "final_patch": "",
                "score": 0.0,
                "method": "CoT-SC",
                "error": "No patches generated"
            }
        
        # 使用自洽性集成选择最佳补丁
        logger.info(f"[{instance_id}] Selecting best patch from {len(patches)} candidates...")
        final_patch = await self.cot_sc.ensemble(patches, problem)
        
        # 运行 SWE-Bench 评估
        logger.info(f"[{instance_id}] Running SWE-Bench evaluation...")
        
        # 在评测之前，删除对应的 log 目录，避免旧日志干扰
        try:
            instance_prefix = instance_id.replace("-", "_")
            run_id = f"{instance_prefix}_cot_sc_standalone"
            log_dir = self.output_dir / "logs" / "run_evaluation" / run_id
            if log_dir.exists():
                logger.info(f"[{instance_id}] Removing existing log dir before evaluation: {log_dir}")
                shutil.rmtree(log_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"[{instance_id}] Failed to clean log dir before evaluation: {e}")

        score = await self._run_evaluation(problem, final_patch, "cot_sc")
        
        return {
            "instance_id": instance_id,
            "patches": patches,
            "final_patch": final_patch,
            "score": score,
            "method": "CoT-SC"
        }
    
    async def _run_evaluation(
        self, 
        problem: Dict[str, Any], 
        patch: str, 
        technique: str
    ) -> float:
        """运行 SWE-Bench 评估"""
        instance_id = problem.get("instance_id", "unknown")
        
        # 创建临时数据文件
        temp_data_path = self.output_dir / f"temp_{instance_id}.jsonl"
        with open(temp_data_path, 'w') as f:
            f.write(json.dumps(problem) + '\n')
        
        score = await run_swebench_evaluation(
            judge_path=str(self.output_dir),
            instance_id=instance_id,
            extracted_answer=patch,
            technique=technique,
            solution_name="standalone",
            file_path=str(temp_data_path),
        )
        if temp_data_path.exists():
            temp_data_path.unlink()
        return score

            
    
    async def evaluate_dataset(
        self, 
        data_path: str, 
        output_path: str = None,
        method: str = "cot-sc",
        limit: int = None,
        skip_no_image: bool = True,
    ) -> Dict[str, Any]:
        """
        评估整个数据集
        
        Args:
            data_path: 数据集路径（.jsonl 格式）
            output_path: 输出结果路径
            method: 方法 ("cot" 或 "cot-sc")
            limit: 限制评估样本数量
            skip_no_image: 是否跳过没有 Docker 镜像的实例
        """
        # 加载数据
        problems = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        if limit:
            problems = problems[:limit]
        
        logger.info(f"Loaded {len(problems)} problems from {data_path}")
        logger.info(f"Using method: {method}")
        
        # 检查 Docker 镜像可用性
        if skip_no_image:
            available_problems = await self._filter_available_instances(problems)
            logger.info(f"Found {len(available_problems)}/{len(problems)} instances with Docker images")
            problems = available_problems
        
        if not problems:
            logger.error("No problems to evaluate!")
            return {"error": "No problems to evaluate"}
        
        # 评估
        eval_func = self.evaluate_cot if method == "cot" else self.evaluate_cot_sc
        
        async def evaluate_with_retry(problem, idx):
            """异步评估单个问题，带重试机制"""
            instance_id = problem.get("instance_id", f"problem_{idx}")
            logger.info(f"[{idx + 1}/{len(problems)}] Evaluating {instance_id}")
            
            max_retries = 10
            result = None
            
            for attempt in range(1, max_retries + 1):
                logger.info(f"[{instance_id}] Attempt {attempt}/{max_retries}")
                try:
                    result = await eval_func(problem)
                    break
                except Exception as e:
                    # 如果还有重试机会，继续重试
                    if attempt < max_retries:
                        logger.warning(f"[{instance_id}] Evaluation exception on attempt {attempt}/{max_retries}: {e}, retrying...")
                        await asyncio.sleep(2)  # 等待 2 秒后重试
                    else:
                        logger.error(f"[{instance_id}] Evaluation failed after {max_retries} attempts: {e}")
                        result = {
                            "instance_id": instance_id,
                            "patch": "",
                            "score": 0.0,
                            "method": method,
                            "error": str(e)
                        }
                        break
            
            # 如果所有重试都失败，使用默认的失败结果
            if result is None:
                result = {
                    "instance_id": instance_id,
                    "patch": "",
                    "score": 0.0,
                    "method": method,
                    "error": "All retries failed"
                }
            
            logger.info(f"[{instance_id}] Score: {result['score']}")
            
            # 保存中间结果
            self._save_intermediate_result(result)
            
            return result
        
        # 异步并发执行所有评估任务
        tasks = [evaluate_with_retry(problem, idx) for idx, problem in enumerate(problems)]
        results = await asyncio.gather(*tasks)
        
        # 计算总分
        total_score = sum(result["score"] for result in results)
        
        # 计算统计
        accuracy = total_score / len(problems) if problems else 0.0
        
        stats = {
            "method": method,
            "total_problems": len(problems),
            "resolved": int(total_score),
            "accuracy": accuracy,
            "results": results
        }
        
        # 保存结果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Complete!")
        logger.info(f"Method: {method}")
        logger.info(f"Resolved: {int(total_score)}/{len(problems)} ({accuracy:.2%})")
        logger.info(f"{'='*50}\n")
        
        return stats
    
    async def _filter_available_instances(self, problems: List[Dict]) -> List[Dict]:
        """过滤出有 Docker 镜像的实例"""
        import docker
        
        try:
            client = docker.from_env()
            
            # 获取所有 swebench 镜像
            eval_tags = set()
            for img in client.images.list():
                for tag in img.tags:
                    if 'sweb.eval' in tag:
                        # 提取 instance ID 部分
                        eval_tags.add(tag.lower())
            
            available = []
            for problem in problems:
                instance_id = problem.get("instance_id", "")
                # 检查是否有匹配的镜像
                issue = instance_id.split('__')[-1].lower()
                for tag in eval_tags:
                    if issue in tag:
                        available.append(problem)
                        break
            
            return available
            
        except Exception as e:
            logger.warning(f"Failed to check Docker images: {e}")
            return problems
    
    def _save_intermediate_result(self, result: Dict):
        """保存中间结果"""
        instance_id = result.get("instance_id", "unknown")
        result_path = self.output_dir / f"result_{instance_id}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Standalone SWE-Bench CoT/CoT-SC Evaluation")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="xx",
        help="Path to SWE-Bench dataset (.jsonl file)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save results (JSON)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="swe_eval_results",
        help="Directory for intermediate results and reports"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["cot", "cot-sc"],
        default="cot",
        help="Evaluation method"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=3,
        help="Number of samples for CoT-SC"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of problems to evaluate"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="LLM model name (e.g., gpt-4, gpt-5)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config2.yaml",
        help="Path to config2.yaml file"
    )
    parser.add_argument(
        "--no_skip_missing",
        action="store_true",
        help="Don't skip instances without Docker images"
    )
    
    args = parser.parse_args()
    
    # 设置输出路径
    if args.output_path is None:
        data_name = Path(args.data_path).stem
        args.output_path = f"swe_{args.method}_{data_name}_results.json"
    
    # 加载 LLM 配置
    os.environ['MAAS_CONFIG_PATH'] = args.config
    
    llm_config = None
    if args.model:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if 'models' in config_dict and args.model in config_dict['models']:
            model_config_dict = config_dict['models'][args.model]
            llm_config = LLMConfig(**model_config_dict)
            logger.info(f"Loaded model config for {args.model}")
        else:
            if 'llm' in config_dict:
                llm_config = LLMConfig(**config_dict['llm'])
                llm_config.model = args.model
                logger.info(f"Using default config with model: {args.model}")
    
    # 创建配置
    cot_config = SWECoTConfig(num_samples=args.num_samples)
    
    # 创建评估器
    evaluator = SWEEvaluator(
        llm_config=llm_config,
        cot_config=cot_config,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    
    # 运行评估
    asyncio.run(
        evaluator.evaluate_dataset(
            data_path=args.data_path,
            output_path=args.output_path,
            method=args.method,
            limit=args.limit,
            skip_no_image=not args.no_skip_missing,
        )
    )


if __name__ == "__main__":
    main()

