#!/usr/bin/env python3
"""Plot selection-index histograms from self_selection_details.csv files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASETS = ["browsecomp-plus", "gpqa-diamond", "hle-math", "stock"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read self_selection_details.csv files under outputs and plot grouped "
            "bar charts for GPT-4o vs GPT-5 selection frequencies."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Root directory to search. Default: outputs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/selection_histograms_4datasets.png"),
        help="Output image path. Default: outputs/selection_histograms_4datasets.png",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=50,
        help="Ignore details CSVs with fewer than this many rows. Default: 50",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to plot. Default: {' '.join(DEFAULT_DATASETS)}",
    )
    return parser.parse_args()


def normalize_dataset(name: str) -> str:
    return name.replace("gpqa_diamond", "gpqa-diamond").replace("hle_math", "hle-math")


def infer_model(run_name: str) -> str | None:
    if "gpt-4o" in run_name:
        return "GPT-4o"
    if "gpt-5" in run_name or "gpt5" in run_name:
        return "GPT-5"
    return None


def load_run_rows(root: Path, datasets: set[str], min_rows: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/question/meta_agent/workflow_search/*/self_selection_details.csv")):
        rel_parts = path.relative_to(root).parts
        if len(rel_parts) < 6:
            continue
        run = rel_parts[0]
        dataset = normalize_dataset(rel_parts[4])
        if dataset not in datasets:
            continue

        model = infer_model(run)
        if model is None:
            continue

        selections: list[int] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    selections.append(int(row["selection"]))
                except Exception:
                    continue

        if len(selections) < min_rows:
            continue

        counts = Counter(selections)
        total = len(selections)
        freq_by_index = {idx: counts[idx] / total for idx in range(9)}
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "run": run,
                "n": total,
                **{f"idx{idx}": freq_by_index[idx] for idx in range(9)},
            }
        )
    return rows


def average_run_frequencies(
    run_rows: list[dict[str, object]], datasets: list[str]
) -> dict[tuple[str, str], list[float]]:
    freq_map: dict[tuple[str, str], list[float]] = {}
    for model in ("GPT-4o", "GPT-5"):
        for dataset in datasets:
            matched = [row for row in run_rows if row["model"] == model and row["dataset"] == dataset]
            if not matched:
                continue
            freq_map[(model, dataset)] = [
                sum(float(row[f"idx{idx}"]) for row in matched) / len(matched) for idx in range(9)
            ]
    return freq_map


def plot_histograms(freq_map: dict[tuple[str, str], list[float]], datasets: list[str], out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5.0), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    colors = {"GPT-4o": "#4285F4", "GPT-5": "#EA4335"}
    indices = np.arange(9)
    width = 0.34

    mapping = {
        "gpqa-diamond": "GPQA Diamond",
        "hle-math": "HLE Math",
        "browsecomp-plus": "Browsecomp Plus",
        "stock": "SMFR"
    }

    for ax, dataset in zip(axes, datasets):
        for offset, model in [(-width / 2, "GPT-4o"), (width / 2, "GPT-5")]:
            values = freq_map.get((model, dataset), [0.0] * 9)
            ax.bar(indices + offset, np.array(values) * 100.0, width=width, color=colors[model], label=model)

        ax.set_xticks(indices)
        ax.set_xlabel(mapping[dataset], fontsize=16)
        ax.set_ylim(0, 60)
        ax.tick_params(axis="both", labelsize=10)

    axes[0].set_ylabel("Frequency (%)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    print(labels, handles)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        fontsize=16,
        handlelength=1.0,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    datasets = [normalize_dataset(dataset) for dataset in args.datasets]

    run_rows = load_run_rows(root, set(datasets), args.min_rows)
    if not run_rows:
        raise SystemExit(f"No valid self_selection_details.csv files found under {root}")

    freq_map = average_run_frequencies(run_rows, datasets)
    plot_histograms(freq_map, datasets, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
