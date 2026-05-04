import asyncio
import csv
import glob
import json
import time
import torch
import os
import numpy as np
from typing import List, Literal

from pydantic import BaseModel, Field
from maas.ext.maas.scripts.evaluator import DatasetType
from maas.ext.maas.scripts.optimizer_utils.data_utils import DataUtils               
from maas.ext.maas.scripts.optimizer_utils.experience_utils import ExperienceUtils
from maas.ext.maas.scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from maas.ext.maas.scripts.optimizer_utils.graph_utils import GraphUtils           
from maas.logs import logger
from maas.ext.maas.models.utils import get_sentence_embedding
from maas.ext.maas.models.controller import MultiLayerController

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


def _stock_aggregate_from_csv(level_dir: str) -> dict:
    """Read the latest CSV in *level_dir* and compute direct/code stats from eval_details."""
    csvs = sorted(glob.glob(os.path.join(level_dir, "*.csv")))
    if not csvs:
        return {}
    csv_path = csvs[-1]
    total = 0
    direct_full = 0
    code_full = 0
    code_failed = 0
    both = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            ed_str = row.get("eval_details", "")
            if not ed_str:
                continue
            try:
                ed = json.loads(ed_str)
            except Exception:
                continue
            d = bool(ed.get("direct_full"))
            c = bool(ed.get("code_full"))
            cf = bool(ed.get("code_exec_info") and not c
                      and "success\": false" in str(ed.get("code_exec_info", "")))
            if d:
                direct_full += 1
            if c:
                code_full += 1
            if cf:
                code_failed += 1
            if d and c:
                both += 1
    return {
        "total_samples": total,
        "count_direct_full": direct_full,
        "count_code_full": code_full,
        "count_code_exec_failed": code_failed,
        "count_direct_and_code_full": both,
        "rate_direct_full": direct_full / total if total else 0.0,
        "rate_code_full": code_full / total if total else 0.0,
    }


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        optimized_path: str = None,
        round: int = 1,
        batch_size: int = 4,
        lr: float = 0.01,
        is_textgrad: bool = False,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.execute_llm_config = exec_llm_config
        self.dataset = dataset
        self.type = question_type
        self.graph = None
        self.operators = operators
        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = round
        self.batch_size = batch_size
        self.lr = lr
        self.is_textgrad = is_textgrad
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.controller = MultiLayerController(device=self.device).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.lr)          

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            score = loop.run_until_complete(self.test())
            return None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        retry_count = 0
        max_retries = 1
        round = 1

        while retry_count < max_retries:
            try:
                score = loop.run_until_complete(self._optimize_graph_maas()) 
                break
            except Exception as e:
                retry_count += 1
                logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                if retry_count == max_retries:
                    logger.info("Max retries reached. Moving to next round.")
                    score = None

                wait_time = 5 * retry_count
                time.sleep(wait_time)

            if retry_count < max_retries: 
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        logger.info(f"Score for round {round}: {score}")
        round += 1
        
        time.sleep(5)

    async def _optimize_graph_maas(self):
        graph_path = f"{self.root_path}/train"
        data = self.data_utils.load_results(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators) 
        precomputed_operator_embeddings = torch.stack([get_sentence_embedding(op_desc) for op_desc in operator_descriptions]).to(self.device)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)
        logger.info(directory)

        self.graph = self.graph_utils.load_graph_maas(graph_path)

        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config, 
            "dataset": self.dataset, 
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": self.is_textgrad,
        }

        avg_score = await self.evaluation_utils.evaluate_graph_maas(self, directory, data, initial=False, params=params)

        return avg_score

    async def test(self):
        if self.dataset == "Stock":
            return await self._test_stock_multi_level()
        return await self._test_single()

    async def _test_single(self):
        """Standard single-dataset test (used by all non-Stock benchmarks)."""
        data = []
        graph_path = f"{self.root_path}/test"
        
        json_file_path = self.data_utils.get_results_file_path(graph_path)
        data = self.data_utils.load_results(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators) 
        precomputed_operator_embeddings = torch.stack([get_sentence_embedding(op_desc) for op_desc in operator_descriptions]).to(self.device)

        self.graph = self.graph_utils.load_graph_maas(graph_path)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)

        pth_path = f"{self.root_path}/train"
        pth_directory = self.graph_utils.create_round_directory(pth_path, self.round)
        controller_path = os.path.join(pth_directory,  f"{self.dataset}_controller_sample{self.sample}.pth")
        logger.info(controller_path)

        if os.path.exists(controller_path):
            checkpoint = torch.load(controller_path, map_location=self.device)
            self.controller.load_state_dict(checkpoint)
            self.controller.eval()
        else:
            raise FileNotFoundError(f"Controller model file not found at {controller_path}")         

        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config,  
            "dataset": self.dataset,                        
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": False,
        }

        result = await self.evaluation_utils.evaluate_graph_test_maas(self, directory, is_test=True, params=params)
        
        # Handle optional operator statistics
        if len(result) == 5:
            score, avg_cost, total_cost, token, operator_stats = result
        else:
            score, avg_cost, total_cost, token = result
            operator_stats = None

        new_data = self.data_utils.create_result_data(self.round, score, avg_cost, total_cost, token)
        
        # Add operator statistics if available
        if operator_stats:
            new_data['operator_statistics'] = operator_stats
            logger.info(f"Operator usage statistics: {operator_stats['summary']}")
        
        data.append(new_data)

        self.data_utils.save_results(json_file_path, data)

        return score

    async def _test_stock_multi_level(self):
        """Test Stock benchmark across complexity levels 2-6, recording each result."""
        graph_path = f"{self.root_path}/test"

        # --- load controller trained weights ---
        pth_path = f"{self.root_path}/train"
        pth_directory = self.graph_utils.create_round_directory(pth_path, self.round)
        controller_path = os.path.join(pth_directory, f"{self.dataset}_controller_sample{self.sample}.pth")
        logger.info(f"Loading controller from {controller_path}")

        if os.path.exists(controller_path):
            checkpoint = torch.load(controller_path, map_location=self.device)
            self.controller.load_state_dict(checkpoint)
            self.controller.eval()
        else:
            raise FileNotFoundError(f"Controller model file not found at {controller_path}")

        # --- pre-compute operator embeddings ---
        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators)
        precomputed_operator_embeddings = torch.stack(
            [get_sentence_embedding(desc) for desc in operator_descriptions]
        ).to(self.device)

        self.graph = self.graph_utils.load_graph_maas(graph_path)

        # --- iterate over levels ---
        stock_levels = self.stock_levels if hasattr(self, "stock_levels") else [2, 3, 4, 5, 6]
        all_level_data = []
        summary = {}

        for level in stock_levels:
            data_file = f"maas/ext/maas/data/stock_level_{level}.jsonl"
            if not os.path.exists(data_file):
                logger.warning(f"Data file not found, skipping level {level}: {data_file}")
                continue

            round_dir = self.graph_utils.create_round_directory(graph_path, self.round)
            level_dir = os.path.join(round_dir, f"level_{level}")
            os.makedirs(level_dir, exist_ok=True)

            logger.info(f"{'='*60}")
            logger.info(f"Evaluating Stock level {level}  ({data_file})")
            logger.info(f"{'='*60}")

            params = {
                "operator_embeddings": precomputed_operator_embeddings,
                "controller": self.controller,
                "execute_llm_config": self.execute_llm_config,
                "dataset": self.dataset,
                "optimizer": self.optimizer,
                "sample": self.sample,
                "is_textgrad": False,
                    "data_path_override": data_file,
            }

            result = await self.evaluation_utils.evaluate_graph_test_maas(
                self, level_dir, is_test=True, params=params
            )

            if len(result) == 5:
                score, avg_cost, total_cost, token, operator_stats = result
            else:
                score, avg_cost, total_cost, token = result
                operator_stats = None

            level_result = self.data_utils.create_result_data(
                self.round, score, avg_cost, total_cost, token
            )
            level_result["level"] = level
            if operator_stats:
                level_result["operator_statistics"] = operator_stats

            # --- attach stock direct/code aggregate from eval_details ---
            try:
                level_result["stock_aggregate"] = _stock_aggregate_from_csv(level_dir)
            except Exception as e:
                logger.warning(f"Could not compute stock_aggregate for level {level}: {e}")

            all_level_data.append(level_result)
            summary[level] = {"score": score, "avg_cost": avg_cost, "total_cost": total_cost}

            logger.info(
                f"Level {level} — score: {score:.4f}, "
                f"avg_cost: ${avg_cost:.5f}, total_cost: ${total_cost:.5f}"
            )

        # --- save consolidated results ---
        json_file_path = self.data_utils.get_results_file_path(graph_path)
        existing = self.data_utils.load_results(graph_path)
        existing.extend(all_level_data)
        self.data_utils.save_results(json_file_path, existing)

        # --- print summary table ---
        logger.info("")
        logger.info("=" * 60)
        logger.info("  STOCK MULTI-LEVEL TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  {'Level':<10} {'Score':<12} {'Avg Cost':<14} {'Total Cost'}")
        logger.info(f"  {'-'*10} {'-'*12} {'-'*14} {'-'*12}")
        for lvl in sorted(summary.keys()):
            s = summary[lvl]
            logger.info(
                f"  {lvl:<10} {s['score']:<12.4f} ${s['avg_cost']:<13.5f} ${s['total_cost']:.5f}"
            )
        if summary:
            avg_score = np.mean([s["score"] for s in summary.values()])
            logger.info(f"  {'-'*10} {'-'*12} {'-'*14} {'-'*12}")
            logger.info(f"  {'Average':<10} {avg_score:<12.4f}")
        logger.info("=" * 60)

        return avg_score if summary else 0.0
