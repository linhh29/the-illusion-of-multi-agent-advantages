from typing import Dict, Literal, Tuple

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.ext.maas.benchmark.gsm8k import GSM8KBenchmark
from maas.ext.maas.benchmark.humaneval import HumanEvalBenchmark
from maas.ext.maas.benchmark.math import MATHBenchmark
from maas.ext.maas.benchmark.gpqa import GPQABenchmark
from maas.ext.maas.benchmark.hlemath import HLEMATHBenchmark
from maas.ext.maas.benchmark.bcp import BCPBenchmark
from maas.ext.maas.benchmark.swe import SWEBenchmark
from maas.ext.maas.benchmark.smfr import SmfrBenchmark

DatasetType = Literal["HumanEval", "GSM8K", "MATH", "GPQA", "HLEMATH", "BCP", "SWE", "Smfr"]


class Evaluator:
    def __init__(self, eval_path: str, batch_size: int):
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "GPQA": GPQABenchmark,
            "HLEMATH": HLEMATHBenchmark,
            "BCP": BCPBenchmark,
            "SWE": SWEBenchmark,
            "Smfr": SmfrBenchmark,
        }

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False
    ):
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        if "data_path_override" in params:
            data_path = params["data_path_override"]
        else:
            data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]

        benchmark_kwargs = dict(
            name=dataset,
            file_path=data_path,
            log_path=path,
            batch_size=self.batch_size,
            controller=params["controller"],
            operator_embeddings=params["operator_embeddings"],
            optimizer=params["optimizer"],
        )
        if dataset == "Smfr" and "eval_mode" in params:
            pass  # eval_mode removed; kept for backward compat with old configs
        benchmark = benchmark_class(**benchmark_kwargs)
        configured_graph = await self._configure_graph(dataset, graph, params)
        if is_test:
            va_list = None
        else:
            va_list = None

        result = await benchmark.run_evaluation(configured_graph, va_list, is_test, params["sample"], params["is_textgrad"])

        # Extract operator statistics if available
        operator_stats = None
        if hasattr(configured_graph, 'get_operator_statistics'):
            operator_stats = configured_graph.get_operator_statistics()

        # Return results with operator statistics
        if operator_stats:
            return (*result, operator_stats)
        return result

    async def _configure_graph(self, dataset, graph, params: dict):
        controller = params.get("controller")
        operator_embeddings = params.get("operator_embeddings")
        llm_config = params.get("execute_llm_config")
        dataset_config = params.get("dataset")
        graph_kwargs = dict(
            name=dataset,
            llm_config=llm_config,
            dataset=dataset_config,
            controller=controller,
            operator_embeddings=operator_embeddings,
        )
        if dataset == "Smfr" and "eval_mode" in params:
            pass  # eval_mode removed; kept for backward compat with old configs
        configured_graph = graph(**graph_kwargs)
        return configured_graph

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"maas/ext/maas/data/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_train.jsonl"
