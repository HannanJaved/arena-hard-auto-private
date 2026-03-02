#!/usr/bin/env python3
"""Plot weighted hard_prompt + creative_writing scores for Qwen3 SFT/DPO checkpoints."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

HARD_WEIGHT = 2 / 3
CREATIVE_WEIGHT = 1 / 3

LR_PATTERN = re.compile(r"Lr([0-9.eE+]+)")
BETA_PATTERN = re.compile(r"Beta(\d+)")


def read_scores(csv_path: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["Model"].strip()
            score = float(row["Scores (%)"].strip())
            scores[model] = score
    return scores


def weighted_scores(
    hard_scores: Dict[str, float], creative_scores: Dict[str, float]
) -> Dict[str, float]:
    weighted: Dict[str, float] = {}
    common_models = hard_scores.keys() & creative_scores.keys()
    for model in common_models:
        weighted[model] = HARD_WEIGHT * hard_scores[model] + CREATIVE_WEIGHT * creative_scores[model]
    return weighted


def parse_lr(model_name: str) -> float:
    match = LR_PATTERN.search(model_name)
    if not match:
        raise ValueError(f"Could not parse Lr from {model_name}")
    return float(match.group(1))


def parse_beta(model_name: str) -> int | None:
    match = BETA_PATTERN.search(model_name)
    if match:
        return int(match.group(1))
    return None


def filter_models(models: Iterable[str], keyword: str) -> List[str]:
    return sorted([model for model in models if keyword in model])


def plot_sft(
    models: List[str],
    scores: Dict[str, float],
    baseline_score: float,
    blue_line_score: float,
    output_path: Path,
) -> None:
    points = sorted(((parse_lr(model), scores[model]) for model in models), key=lambda x: x[0])
    lrs, values = zip(*points)

    plt.figure(figsize=(8, 5))
    plt.plot(lrs, values, marker="o", linestyle="-", label="SFT checkpoints")
    plt.axhline(baseline_score, color="red", linestyle=":", label="Baseline = 50")
    plt.axhline(blue_line_score, color="blue", linestyle=":", label="qwen3-1.7b-base")
    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Weighted score")
    plt.title("SFT checkpoints (weighted hard_prompt + creative_writing)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_dpo(
    models: List[str],
    scores: Dict[str, float],
    baseline_score: float,
    blue_line_score: float,
    output_path: Path,
) -> None:
    beta_values = sorted({parse_beta(model) for model in models if parse_beta(model) is not None})
    color_map = plt.get_cmap("viridis", len(beta_values))

    plt.figure(figsize=(9, 5.5))
    for index, beta in enumerate(beta_values):
        beta_models = [model for model in models if parse_beta(model) == beta]
        points = sorted(((parse_lr(model), scores[model]) for model in beta_models), key=lambda x: x[0])
        if not points:
            continue
        lrs, values = zip(*points)
        plt.scatter(
            lrs,
            values,
            label=f"Beta {beta}",
            color=color_map(index),
            s=70,
            edgecolor="black",
        )

    plt.axhline(baseline_score, color="red", linestyle=":", label="Baseline = 50")
    plt.axhline(
        blue_line_score,
        color="blue",
        linestyle=":",
        label="qwen3-1.7b-olmo3-sft-Lr5e5",
    )
    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Weighted score")
    plt.title("DPO checkpoints (weighted hard_prompt + creative_writing)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Qwen3 SFT/DPO weighted scores.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "qwen3-sft-dpo",
        help="Directory containing hard_prompt_leaderboard_all.csv and creative_writing_leaderboard_all.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (defaults to <results-dir>/plots)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots"

    hard_path = results_dir / "hard_prompt_leaderboard_all.csv"
    creative_path = results_dir / "creative_writing_leaderboard_all.csv"

    hard_scores = read_scores(hard_path)
    creative_scores = read_scores(creative_path)
    combined_scores = weighted_scores(hard_scores, creative_scores)

    sft_models = filter_models(combined_scores.keys(), "-sft-")
    dpo_models = filter_models(combined_scores.keys(), "-dpo-")

    if "qwen3-1.7b-base" not in combined_scores:
        raise ValueError("Missing qwen3-1.7b-base score for baseline line.")
    if "qwen3-1.7b-olmo3-sft-Lr5e5" not in combined_scores:
        raise ValueError("Missing qwen3-1.7b-olmo3-sft-Lr5e5 score for DPO blue line.")

    plot_sft(
        models=sft_models,
        scores=combined_scores,
        baseline_score=50,
        blue_line_score=combined_scores["qwen3-1.7b-base"],
        output_path=output_dir / "sft_lr_vs_weighted_score.png",
    )

    plot_dpo(
        models=dpo_models,
        scores=combined_scores,
        baseline_score=50,
        blue_line_score=combined_scores["qwen3-1.7b-olmo3-sft-Lr5e5"],
        output_path=output_dir / "dpo_lr_vs_weighted_score.png",
    )

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
