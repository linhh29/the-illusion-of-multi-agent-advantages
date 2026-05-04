#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone CoT and CoT-SC Evaluation Script
独立的CoT和CoT-SC评估脚本，最小化对MaAS框架的依赖
"""

import asyncio
import json
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse
from pathlib import Path
from math import isclose

from maas.llm import LLM
from maas.configs.llm_config import LLMConfig
from maas.context import Context
from maas.logs import logger


@dataclass
class CoTConfig:
    """CoT配置"""
    num_samples: int = 3  # CoT-SC生成的样本数量
    temperature: float = 0.7  # 温度参数


class StandaloneCoT:
    """独立的CoT操作子"""
    
    def __init__(self, llm: Any, task_type: str = "gpqa"):
        """
        初始化CoT操作子
        
        Args:
            llm: LLM实例
            task_type: 任务类型 (gpqa, gsm8k, hlemath, math)
        """
        self.llm = llm
        self.task_type = task_type
        self.cot_prompt = self._get_cot_prompt(task_type)
    
    def _get_cot_prompt(self, task_type: str) -> str:
        """根据任务类型获取CoT提示词"""
        
        if task_type == "gpqa":
            return """Graduate-Level Scientific Reasoning Instruction
{instruction}

Current Problem:
{input}

Demonstration Examples (GPQA style):

1. Problem: A quantum state has a lifetime of 10^-9 sec. To clearly distinguish its energy level from another state...
   Analysis:
   Using Heisenberg uncertainty principle: ΔE·Δt ≥ ℏ/2
   Energy uncertainty: ΔE ≥ ℏ/(2Δt) = (1.055×10^-34)/(2×10^-9) ≈ 5.3×10^-26 J
   Convert to eV: ΔE ≈ 3.3×10^-7 eV
   For clear distinction, energy difference should be >> ΔE
   Comparing options: 10^-4 eV >> 3.3×10^-7 eV ✓
   \\boxed{{D}}

2. Problem: Identify the product of a chemical reaction involving trans-cinnamaldehyde...
   Analysis:
   Step 1: Grignard addition forms secondary alcohol
   Step 2: PCC oxidizes secondary alcohol to ketone
   Step 3: Wittig reaction forms alkene, extending carbon chain
   Count carbons: Starting material (9C) + reagent (2C) = 11C total
   \\boxed{{C}}

Solution Protocol:
1. Parse the question carefully, identifying key concepts
2. Apply relevant scientific principles (physics, chemistry, biology)
3. Perform step-by-step logical reasoning
4. Verify intermediate conclusions
5. Present final answer in \\boxed{{option letter}} format

Remember: GPQA questions are at the graduate level and require deep domain knowledge. Think carefully!"""
        
        elif task_type == "hlemath":
            return """High-Level Mathematical Reasoning Instruction
{instruction}

Current Problem:
{input}

Demonstration Examples (HLEMath style):

1. Problem: How many finite groups contain maximal by inclusion product-free sets of size 2?
   Analysis:
   By Theorem 1.1 and Tables 1-3 from Giudici and Hart's paper "Small maximal sum-free sets":
   A product-free set S has no elements a,b such that ab ∈ S
   Maximal by inclusion means no proper superset is product-free
   For |S|=2, must analyze all finite group structures
   Direct enumeration shows exactly 12 such groups
   \\boxed{{12}}

2. Problem: For a nonsingular real polynomial P in ℝ³ of degree D...
   Analysis:
   Using Guth's theorem on polynomial partitioning (2016, JAMS)
   For zero set Z(P,T) in cylinder T with angle constraint > 1/10
   Unit ball covering requires O(D^k) balls
   By theorem 4.1, optimal k = 3 (sharpness shown by grid construction)
   \\boxed{{3}}

Solution Protocol:
1. Parse the mathematical problem carefully
2. Identify relevant theorems and mathematical structures
3. Perform rigorous step-by-step derivation
4. Verify logical consistency at each step
5. Present final answer in \\boxed{{}} notation
6. Ensure all mathematical notation is precise

Remember: HLEMath problems are research-level mathematics. Deep understanding and rigorous reasoning are essential!"""
        
        elif task_type == "bcp":
            return """Answer the question concisely and accurately.
{instruction}

Question:
{input}

Examples:
Q: The series set in a nursing home features pianist and actress XYZ; what is its title?
A: Gentle Notes

Q: A 2021 documentary about surfing in Portugal was directed by João Silva. Title?
A: Maré Alta

Think briefly if needed, then give a single final answer enclosed in <answer></answer> tags.
Output format (strict):
<answer>FINAL_ANSWER_HERE</answer>"""
        
        elif task_type in ["gsm8k", "math"]:
            return """Mathematical Problem Solving with Chain-of-Thought
{instruction}

Current Problem:
{input}

Solution Protocol:
1. Carefully read and understand the problem
2. Identify what is being asked and what information is given
3. Break down the problem into smaller steps
4. Solve each step with clear reasoning
5. Verify your calculations
6. Present the final answer in \\boxed{{answer}} format

Please provide step-by-step reasoning before arriving at the final answer."""
        
        else:
            return """{instruction}

Problem:
{input}

Please think step-by-step and provide your reasoning before giving the final answer."""
    
    async def generate_cot(self, input_text: str, instruction: str = "") -> str:
        """
        生成单个CoT推理
        
        Args:
            input_text: 输入问题
            instruction: 任务指令
            
        Returns:
            CoT推理结果
        """
        prompt = self.cot_prompt.format(input=input_text, instruction=instruction)
        
        try:
            response = await self.llm.aask(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"CoT generation failed: {e}")
            return ""
    
    async def generate_multi_cot(
        self, 
        input_text: str, 
        instruction: str = "", 
        num_samples: int = 3
    ) -> List[str]:
        """
        生成多个CoT推理（用于CoT-SC）
        
        Args:
            input_text: 输入问题
            instruction: 任务指令
            num_samples: 生成样本数量
            
        Returns:
            多个CoT推理结果列表
        """
        tasks = [
            self.generate_cot(input_text, instruction) 
            for _ in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks)
        return [r for r in responses if r]  # 过滤空响应


class StandaloneCoTSC:
    """独立的CoT-SC (Self-Consistency) 操作子"""
    
    def __init__(self, llm: Any):
        self.llm = llm
        self.sc_prompt = """Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

Provide your analysis and then output the letter (A, B, C, etc.) or the final answer that appears most consistently. For multiple choice questions, output only the option letter. For math problems, output the numerical answer.

Your response:"""
    
    async def ensemble(self, solutions: List[str], problem: str) -> str:
        """
        使用自洽性集成选择最佳答案
        
        Args:
            solutions: 多个CoT推理结果
            problem: 原始问题
            
        Returns:
            最终答案
        """
        if not solutions:
            return ""
        
        if len(solutions) == 1:
            return solutions[0]
        
        # 构建解决方案文本
        solution_text = ""
        for idx, solution in enumerate(solutions):
            letter = chr(65 + idx)  # A, B, C...
            solution_text += f"{letter}: \n{solution}\n\n\n"
        
        prompt = self.sc_prompt.format(problem=problem, solutions=solution_text)
        
        try:
            response = await self.llm.aask(prompt)
            
            # 尝试从多个解决方案中提取最一致的答案
            # 如果LLM返回了字母，映射回对应的solution
            answer_match = re.search(r'\b([A-Z])\b', response)
            if answer_match:
                idx = ord(answer_match.group(1)) - 65
                if 0 <= idx < len(solutions):
                    return solutions[idx]
            
            # 否则返回LLM的直接响应
            return response.strip()
            
        except Exception as e:
            logger.error(f"SC ensemble failed: {e}")
            # 降级：返回第一个解决方案
            return solutions[0]
    
    def extract_answer(self, text: str, task_type: str = "gpqa") -> str:
        """
        从文本中提取最终答案
        
        Args:
            text: 包含答案的文本
            task_type: 任务类型
            
        Returns:
            提取的答案
        """
        # 全局优先：提取 <answer>...</answer> 标签
        tag_matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        if tag_matches:
            candidate = tag_matches[-1]
            candidate = candidate.strip().rstrip('.,;!?').strip('"').strip("'")
            return candidate

        # 尝试提取 \boxed{} 中的内容（支持嵌套括号）
        if '\\boxed{' in text:
            try:
                # 匹配可能包含嵌套括号的boxed内容
                pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
                matches = re.findall(pattern, text)
                if matches:
                    return matches[-1].strip()
            except:
                pass
        
        # 对于多选题，提取选项字母
        if task_type in ["gpqa"]:
            matches = re.findall(r'\b([A-Da-d])\b', text)
            if matches:
                return matches[-1].upper()

        # BCP：优先取引号/末尾句
        if task_type == "bcp":
            # 优先取引号中的内容
            quote_matches = re.findall(r'"([^"]+)"', text)
            if quote_matches:
                candidate = quote_matches[-1]
                candidate = candidate.strip().rstrip('.,;!?')
                return candidate
            # 其次取最后一行/句
            parts = [p.strip() for p in re.split(r'[\\n\\r]|(?<!\\d)[.!?]\\s+', text) if p.strip()]
            if parts:
                candidate = parts[-1].strip().rstrip('.,;!?')
                return candidate
            return text.strip().rstrip('.,;!?')
        
        # 对于hlemath，提取最后一句话（如果没有boxed）
        if task_type == "hlemath":
            # 按句子分割
            sentence_end_pattern = r'(?<!\d)[.!?]\s+'
            sentences = re.split(sentence_end_pattern, text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if sentences:
                return sentences[-1]
        
        # 对于数学题，提取数字
        if task_type in ["gsm8k", "math"]:
            # 尝试找最后一个数字
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if numbers:
                return numbers[-1]
        
        return text.strip()


BCP_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()


class StandaloneEvaluator:
    """独立评估器"""
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        task_type: str = "gpqa",
        cot_config: CoTConfig = None,
        config_path: str = None,
        use_llm_grader: bool = False,
    ):
        """
        初始化评估器
        
        Args:
            llm_config: LLM配置
            task_type: 任务类型
            cot_config: CoT配置
            config_path: config2.yaml路径
        """
        self.task_type = task_type
        self.cot_config = cot_config or CoTConfig()
        self.use_llm_grader = use_llm_grader
        
        # 初始化Context和LLM
        import os
        if config_path:
            # 设置环境变量，让Context加载指定的配置文件
            os.environ['MAAS_CONFIG_PATH'] = config_path
        
        context = Context()
        
        # 初始化LLM
        if llm_config:
            logger.info(f"Using LLM config: model={llm_config.model}, api_type={llm_config.api_type}")
            logger.info(f"API key starts with: {llm_config.api_key[:10] if llm_config.api_key else 'None'}...")
            self.llm = context.llm_with_cost_manager_from_llm_config(llm_config)
        else:
            # 使用context中的默认LLM配置（从config2.yaml加载）
            logger.info("Using default LLM config from config2.yaml")
            self.llm = context.llm()
        
        # 初始化操作子
        self.cot = StandaloneCoT(self.llm, task_type)
        self.cot_sc = StandaloneCoTSC(self.llm)
    
    def is_digit(self, num) -> bool:
        """检查是否可以解析为数字"""
        return self.parse_digits(num) is not None
    
    def parse_digits(self, num) -> Optional[float]:
        """
        解析字符串为浮点数
        支持普通数字和百分比
        """
        import regex
        num_str = regex.sub(",", "", str(num))
        try:
            return float(num_str)
        except Exception:
            # 尝试解析百分比
            if num_str.endswith("%"):
                num_str = num_str[:-1]
                if num_str.endswith("\\"):
                    num_str = num_str[:-1]
                try:
                    return float(num_str) / 100
                except Exception:
                    pass
        return None
    
    def symbolic_equal(self, a, b) -> bool:
        """
        使用sympy检查符号等价性
        """
        try:
            from sympy import N, simplify
            from sympy.parsing.latex import parse_latex
            from sympy.parsing.sympy_parser import parse_expr
            
            def _parse(s):
                """尝试将字符串解析为数学表达式"""
                for f in [parse_latex, parse_expr]:
                    try:
                        return f(s)
                    except Exception:
                        pass
                return s
            
            a = _parse(a)
            b = _parse(b)
            
            # 尝试符号简化
            try:
                if simplify(a - b) == 0:
                    return True
            except Exception:
                pass
            
            # 尝试数值求值
            try:
                if isclose(N(a), N(b), abs_tol=1e-3):
                    return True
            except Exception:
                pass
            
            return False
        except Exception:
            return False
    
    def math_equal(self, prediction: Any, reference: Any) -> bool:
        """
        检查两个数学表达式是否等价（用于hlemath）
        尝试三种方法：
        1. 字符串等价
        2. 数值等价（对于数字）
        3. 符号等价（使用sympy）
        """
        # 方法1：直接字符串比较
        if str(prediction) == str(reference):
            return True
        
        # 方法2：数值比较
        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction_num = self.parse_digits(prediction)
                reference_num = self.parse_digits(reference)
                if prediction_num is not None and reference_num is not None:
                    return isclose(prediction_num, reference_num, abs_tol=1e-3)
        except Exception:
            pass
        
        # 方法3：符号比较
        try:
            return self.symbolic_equal(prediction, reference)
        except Exception:
            pass
        
        return False
    
    def normalize_answer(self, answer: str, task_type: str = None) -> str:
        """标准化答案用于比较"""
        task_type = task_type or self.task_type
        
        # 提取答案
        answer = self.cot_sc.extract_answer(answer, task_type)
        
        # 转小写并去除空格，同时去除首尾引号/标点
        answer = answer.strip().strip('"').strip("'").rstrip('.,;!?')
        return answer.lower().strip()
    
    async def grade_with_llm(self, question: str, ground_truth: str, prediction: str) -> Optional[float]:
        """可选的 LLM Grader（仅 bcp 使用），返回 1/0/None."""
        if self.task_type != "bcp" or not self.use_llm_grader:
            return None
        try:
            grader_prompt = BCP_GRADER_TEMPLATE.format(
                question=question,
                response=prediction,
                correct_answer=ground_truth,
            )
            grading_response = await self.llm.aask(grader_prompt)
            # 兼容 **correct:** yes / correct: yes / correct：yes
            correct_match = re.search(r"correct\s*[:：]\s*(yes|no)", grading_response, re.IGNORECASE)
            if correct_match:
                return 1.0 if correct_match.group(1).lower() == "yes" else 0.0
            logger.warning("LLM grader did not return parsable 'correct' field; falling back to local match.")
            return None
        except Exception as e:
            logger.warning(f"LLM grader failed, fallback to string match. Error: {e}")
            return None

    def calculate_score_local(self, prediction: str, ground_truth: str) -> float:
        """本地宽松判分（无 LLM）。"""
        # 对于hlemath，使用数学等价性判断
        if self.task_type == "hlemath":
            pred_answer = self.cot_sc.extract_answer(prediction, self.task_type)
            gt_answer = self.cot_sc.extract_answer(ground_truth, self.task_type)
            return 1.0 if self.math_equal(pred_answer, gt_answer) else 0.0
        
        # 对于其他任务类型，使用标准化字符串比较
        pred_norm = self.normalize_answer(prediction)
        gt_norm = self.normalize_answer(ground_truth)
        
        if pred_norm == gt_norm:
            return 1.0
        
        # BCP：更宽松，允许包含关系（处理“the name is XX”）
        if self.task_type == "bcp":
            if gt_norm and pred_norm:
                if (gt_norm in pred_norm) or (pred_norm in gt_norm):
                    # 避免过短噪声，要求至少3字符
                    if min(len(gt_norm), len(pred_norm)) >= 3:
                        return 1.0
        
        return 0.0
    
    async def evaluate_cot(
        self, 
        problem: Dict[str, Any], 
        instruction: str = ""
    ) -> Dict[str, Any]:
        """
        使用单个CoT评估
        
        Args:
            problem: 问题字典，包含 'question' 和 'answer'
            instruction: 任务指令
            
        Returns:
            评估结果
        """
        input_text = problem.get("question", problem.get("input", ""))
        ground_truth = problem.get("answer", "")
        
        # 生成CoT推理
        prediction = await self.cot.generate_cot(input_text, instruction)
        
        # 计算得分（先 LLM grader，后本地宽松）
        score_llm = await self.grade_with_llm(input_text, ground_truth, prediction)
        if score_llm is None:
            score = self.calculate_score_local(prediction, ground_truth)
        else:
            score = score_llm
        
        return {
            "input": input_text,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "score": score,
            "method": "CoT"
        }
    
    async def evaluate_cot_sc(
        self, 
        problem: Dict[str, Any], 
        instruction: str = ""
    ) -> Dict[str, Any]:
        """
        使用CoT-SC评估
        
        Args:
            problem: 问题字典
            instruction: 任务指令
            
        Returns:
            评估结果
        """
        input_text = problem.get("question", problem.get("input", ""))
        ground_truth = problem.get("answer", "")
        
        # 生成多个CoT推理
        solutions = await self.cot.generate_multi_cot(
            input_text, 
            instruction, 
            self.cot_config.num_samples
        )
        
        # 使用自洽性集成
        prediction = await self.cot_sc.ensemble(solutions, input_text)
        
        # 计算得分（先 LLM grader，后本地宽松）
        score_llm = await self.grade_with_llm(input_text, ground_truth, prediction)
        if score_llm is None:
            score = self.calculate_score_local(prediction, ground_truth)
        else:
            score = score_llm
        
        return {
            "input": input_text,
            "solutions": solutions,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "score": score,
            "method": "CoT-SC"
        }
    
    async def evaluate_dataset(
        self, 
        data_path: str, 
        output_path: str = None,
        method: str = "cot-sc",
        instruction: str = "",
        limit: int = None
    ) -> Dict[str, Any]:
        """
        评估整个数据集
        
        Args:
            data_path: 数据集路径（.jsonl格式）
            output_path: 输出结果路径
            method: 方法 ("cot" 或 "cot-sc")
            instruction: 任务指令
            limit: 限制评估样本数量
            
        Returns:
            评估结果统计
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
        
        # 评估
        results = []
        total_score = 0.0
        
        eval_func = self.evaluate_cot if method == "cot" else self.evaluate_cot_sc
        
        for idx, problem in enumerate(problems):
            logger.info(f"Evaluating problem {idx + 1}/{len(problems)}")
            
            try:
                result = await eval_func(problem, instruction)
                results.append(result)
                total_score += result["score"]
                
                logger.info(f"Score: {result['score']}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate problem {idx + 1}: {e}")
                results.append({
                    "input": problem.get("question", ""),
                    "prediction": f"Error: {e}",
                    "ground_truth": problem.get("answer", ""),
                    "score": 0.0,
                    "method": method
                })
        
        # 计算统计
        accuracy = total_score / len(problems) if problems else 0.0
        
        stats = {
            "method": method,
            "task_type": self.task_type,
            "total_problems": len(problems),
            "correct": int(total_score),
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
        logger.info(f"Accuracy: {accuracy:.4f} ({int(total_score)}/{len(problems)})")
        logger.info(f"{'='*50}\n")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Standalone CoT/CoT-SC Evaluation")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None,
        help="Path to dataset (.jsonl file); not required when --stock_levels is set",
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save results (JSON)"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["cot", "cot-sc"],
        default="cot-sc",
        help="Evaluation method"
    )
    parser.add_argument(
        "--task_type", 
        type=str, 
        choices=["gpqa", "gsm8k", "math", "hlemath", "humaneval", "bcp"],
        default="gpqa",
        help="Task type"
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
        help=(
            "Cap how many problems to run. With --stock_levels: first N lines per level "
            "from each stock_level_{n}.jsonl (same as limit_per_level). Example: --limit 1 for a smoke test."
        ),
    )
    parser.add_argument(
        "--instruction", 
        type=str, 
        default="",
        help="Custom instruction for the task"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="LLM model name (e.g., gpt-4, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config2.yaml",
        help="Path to config2.yaml file"
    )
    parser.add_argument(
        "--use_llm_grader",
        action="store_true",
        help="For bcp: use LLM grader before local string match."
    )
    parser.add_argument(
        "--stock_levels",
        nargs="*",
        type=int,
        default=None,
        help="If set, run STOCKS multi-level eval (AFlow-aligned). Example: --stock_levels 2 3 4 5 6",
    )
    parser.add_argument(
        "--stock_data_dir",
        type=str,
        default="maas/ext/maas/data",
        help="Directory with stock_level_{n}.jsonl; relative to MaAS repo root (…/MaAS/maas/ext/maas/data)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=30,
        help="Max concurrent STOCKS tasks (used with --stock_levels)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Override LLMConfig.max_token (max output tokens per LLM call)",
    )
    
    args = parser.parse_args()

    if args.stock_levels is not None and len(args.stock_levels) > 0:
        from workspace.stock_standalone_eval import run_stock_levels

        if args.output_path is None:
            args.output_path = (
                f"results_{args.method}_stock_{Path(args.stock_data_dir).name}.json"
            )

        os.environ["MAAS_CONFIG_PATH"] = args.config
        run_stock_levels(
            stock_levels=list(args.stock_levels),
            stock_data_dir=args.stock_data_dir,
            output_path=args.output_path,
            method=args.method,
            model=args.model,
            config_path=args.config,
            max_concurrent=args.max_concurrent,
            num_samples=args.num_samples,
            limit_per_level=args.limit,
            max_tokens=args.max_tokens,
        )
        return

    if not args.data_path:
        raise SystemExit("error: --data_path is required unless --stock_levels is provided")
    
    # 设置输出路径
    if args.output_path is None:
        data_name = Path(args.data_path).stem
        args.output_path = f"results_{args.method}_{data_name}.json"
    
    # 加载配置文件并获取LLM配置
    os.environ['MAAS_CONFIG_PATH'] = args.config
    
    llm_config = None
    if args.model:
        # 从config2.yaml中加载指定模型的配置
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 检查是否在models中定义了该模型
        if 'models' in config_dict and args.model in config_dict['models']:
            model_config_dict = config_dict['models'][args.model]
            llm_config = LLMConfig(**model_config_dict)
            logger.info(f"Loaded model config for {args.model} from config2.yaml")
            logger.info(f"Model: {llm_config.model}, API Type: {llm_config.api_type}")
        else:
            # 如果没有在models中找到，使用默认llm配置
            logger.warning(f"Model {args.model} not found in config2.yaml models section")
            if 'llm' in config_dict:
                llm_config = LLMConfig(**config_dict['llm'])
                llm_config.model = args.model
                logger.info(f"Using default LLM config with model override: {args.model}")
            else:
                raise ValueError(f"Model {args.model} not found and no default llm config available")
    
    # 创建CoT配置
    cot_config = CoTConfig(num_samples=args.num_samples)
    
    # 创建评估器
    evaluator = StandaloneEvaluator(
        llm_config=llm_config,
        task_type=args.task_type,
        cot_config=cot_config,
        config_path=args.config,
        use_llm_grader=args.use_llm_grader,
    )
    
    # 运行评估
    asyncio.run(
        evaluator.evaluate_dataset(
            data_path=args.data_path,
            output_path=args.output_path,
            method=args.method,
            instruction=args.instruction,
            limit=args.limit
        )
    )


if __name__ == "__main__":
    main()

