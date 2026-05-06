"""
Inference script for the manual MAS pipeline.

Usage:
  python run_inference.py --input <data_dir>/smfr_test.jsonl --model gpt-4.1 --test
  python run_inference.py --input <data_dir>/smfr_test.jsonl --model gpt-4.1 --slice 20
  python run_inference.py --input <data_dir>/smfr_test.jsonl --model gpt-4.1

Writes results to: {model}_{dataset_name}.jsonl
Then automatically runs evaluation and prints accuracy + cost summary.

Output format (one JSON per line):
  {
    "input":  {<original sample fields>},
    "model":  "<model name>",
    "output": {
      "investor_dates": {"Alice": [...], ...},
      "comparison":     {"Alice": "...", ...},
      "answer":         ["Alice"]
    },
    "stats": {
      "model": "gpt-4.1-2025-04-14",
      "llm_calls": 12,
      "input_tokens": 45000,
      "output_tokens": 3200,
      "total_tokens": 48200,
      "cost_usd": {"input": 0.09, "output": 0.026, "total": 0.116, "pricing_found": true},
      "phases": {"meta": {...}, "transactions": {...}, ...}
    }
  }
"""

import sys
import os
import json
import asyncio
import argparse
import logging
import traceback
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluate"))

from pipeline import run_pipeline
from agents import create_model_client, ModelClient, PipelineStats

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ALIASES = {
    # OpenAI
    "gpt-4o":         "gpt-4o-2024-08-06",
    "gpt-4.1":        "gpt-4.1-2025-04-14",
    "o3":             "o3-2025-04-16",
    "o4-mini":        "o4-mini-2025-04-16",
    "gpt-5":          "gpt-5-2025-08-07",
    # Custom gateway models (no alias expansion — name is the endpoint identifier)
    # "gpt-oss-120b": used as-is
    # Google Gemini
    "gemini-flash":   "gemini-2.5-flash",
    "gemini-pro":     "gemini-2.5-pro",
    "gemini-flash-lite": "gemini-2.5-flash-lite-preview-06-17",
    # Anthropic Claude (add when backend is wired up)
}


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _aggregate_stats(all_stats: list[PipelineStats]) -> dict:
    """Aggregate PipelineStats across all samples into a summary dict."""
    total_calls = sum(s.llm_calls for s in all_stats)
    total_input = sum(s.input_tokens for s in all_stats)
    total_output = sum(s.output_tokens for s in all_stats)
    total_tokens = total_input + total_output

    # Cost (use first non-empty stats for pricing since model is the same)
    sample_stats = next((s for s in all_stats if s.model_name), None)
    if sample_stats:
        # Build a temporary combined stats to reuse cost()
        combined = PipelineStats(
            model_name=sample_stats.model_name,
            llm_calls=total_calls,
            input_tokens=total_input,
            output_tokens=total_output,
        )
        cost = combined.cost()
    else:
        cost = {"input": 0.0, "output": 0.0, "total": 0.0, "pricing_found": False}

    n = len(all_stats)
    return {
        "samples": n,
        "total_llm_calls": total_calls,
        "avg_llm_calls_per_sample": round(total_calls / n, 1) if n else 0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": round(total_tokens / n, 0) if n else 0,
        "total_cost_usd": cost,
        "avg_cost_per_sample_usd": round(cost["total"] / n, 6) if n else 0,
    }


def _print_cost_summary(agg: dict, model: str):
    cost = agg["total_cost_usd"]
    print(f"\n{'=' * 60}")
    print("COST / TOKEN SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model:                    {model}")
    print(f"  Samples:                  {agg['samples']}")
    print(f"  Total LLM calls:          {agg['total_llm_calls']}  "
          f"(avg {agg['avg_llm_calls_per_sample']} / sample)")
    print(f"  Total input tokens:       {agg['total_input_tokens']:,}")
    print(f"  Total output tokens:      {agg['total_output_tokens']:,}")
    print(f"  Total tokens:             {agg['total_tokens']:,}  "
          f"(avg {int(agg['avg_tokens_per_sample']):,} / sample)")
    if cost.get("pricing_found"):
        print(f"  Total cost (input):       ${cost['input']:.4f}")
        print(f"  Total cost (output):      ${cost['output']:.4f}")
        print(f"  Total cost:               ${cost['total']:.4f}  "
              f"(avg ${agg['avg_cost_per_sample_usd']:.4f} / sample)")
    else:
        print(f"  Cost:                     N/A (model not in pricing table)")
    print(f"{'=' * 60}")


async def process_sample(
    sample: dict, model_client: ModelClient, semaphore: asyncio.Semaphore
) -> tuple[dict, PipelineStats, dict]:
    """Run the full MAS pipeline for one sample under semaphore control."""
    async with semaphore:
        try:
            output, stats, trace = await run_pipeline(sample["problem"], model_client)
        except Exception:
            traceback.print_exc()
            output = {"investor_dates": {}, "comparison": {}, "answer": None}
            stats = PipelineStats(model_name=model_client.model_name)
            trace = {}
        return output, stats, trace


async def run_one_rep(
    dataset: list,
    model_client,
    concurrency: int,
    model_alias: str,
    output_path: str,
) -> None:
    """Run the full MAS pipeline for one repetition and write results + eval."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [process_sample(sample, model_client, semaphore) for sample in dataset]
    results_raw = await tqdm_asyncio.gather(*tasks, desc="Running MAS pipeline")

    records = []
    all_stats: list[PipelineStats] = []
    for sample, (output, stats, trace) in zip(dataset, results_raw):
        records.append({
            "input": sample,
            "model": model_alias,
            "output": {"answer": output},
            "stats": stats.to_dict(),
            "trace": trace,
        })
        all_stats.append(stats)

    write_jsonl(output_path, records)
    print(f"\nWrote {len(records)} results to {output_path}")

    agg = _aggregate_stats(all_stats)
    _print_cost_summary(agg, model_alias)

    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate stats saved to: {stats_path}")

    print(f"\n{'=' * 60}")
    print("RUNNING EVALUATION")
    print(f"{'=' * 60}")
    eval_output = output_path.replace(".jsonl", "_eval.json")
    try:
        from evaluate_smfr_answer_code import evaluate_responses
        evaluate_responses(output_path, eval_output)
        print(f"Eval results saved to: {eval_output}")
    except Exception:
        traceback.print_exc()
        print("Evaluation failed — check output file manually.")


async def main(args):
    model_name = MODEL_ALIASES.get(args.model, args.model)
    model_client = create_model_client(model_name)

    dataset = load_jsonl(args.input)
    if args.test:
        dataset = dataset[:2]
    elif args.slice:
        dataset = dataset[:args.slice]

    dataset_name = Path(args.input).stem

    print(f"Running MAS pipeline on {len(dataset)} samples  model={args.model}  reps={args.reps}")
    print(f"Input:  {args.input}")

    for rep in range(args.reps):
        if args.reps > 1:
            print(f"\n{'=' * 60}")
            print(f"REP {rep} / {args.reps - 1}")
            print(f"{'=' * 60}")

        if args.output and args.reps == 1:
            output_path = args.output
        else:
            output_path = f"{args.model}_run-{rep}__{dataset_name}.jsonl"

        print(f"Output: {output_path}\n")
        await run_one_rep(dataset, model_client, args.concurrency, args.model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run manual MAS pipeline on smfr benchmark")
    parser.add_argument("--input", required=True, help="Path to input JSONL dataset")
    parser.add_argument(
        "--model", default="gpt-4.1",
        help="Model alias (gpt-4.1, gpt-4o, o3, o4-mini, gpt-5) or full model ID",
    )
    parser.add_argument("--output", default=None, help="Output JSONL path (default: auto)")
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max parallel pipelines (each makes ~10-20 LLM calls internally)",
    )
    parser.add_argument(
        "--reps", type=int, default=1,
        help="Number of repetitions (writes run-0, run-1, ... output files)",
    )
    parser.add_argument("--test", action="store_true", help="Run only 2 samples")
    parser.add_argument("--slice", type=int, default=None, help="Run first N samples")
    args = parser.parse_args()

    asyncio.run(main(args))
