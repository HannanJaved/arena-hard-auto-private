#!/usr/bin/env python3
"""Compute Spearman correlation between intermediate checkpoint AH scores and final AH scores.

Weighted score definition:
    weighted = (2/3) * hard_prompt + (1/3) * creative_writing

By default, this script:
- reads intermediate checkpoint leaderboards from a directory
- reads final checkpoint leaderboards from another directory
- maps intermediate model names like `...-lr1e5-step24000` to final model names like
  `qwen3-1.7b-olmo3-sft-Lr1e5`
- assigns every intermediate checkpoint the weighted final score/rank of its LR group
- computes Spearman correlation between:
  1. intermediate weighted score vs final weighted score
  2. intermediate weighted rank vs final weighted rank

It also prints per-LR checkpoint trajectories ordered by step, which is usually the more useful
view for the question "do earlier checkpoints predict the final outcome?".
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INTERMEDIATE_DIR = Path(
    "/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/results/qwen3-sft-intermediate"
)
DEFAULT_FINAL_DIR = Path(
    "/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/results/qwen3-sft-final"
)

LR_STEP_PATTERN = re.compile(r"-lr(?P<lr>[^-]+)-step(?P<step>\d+)$", re.IGNORECASE)
FINAL_LR_PATTERN = re.compile(r"-Lr(?P<lr>.+)$")


@dataclass(frozen=True)
class WeightedConfig:
    hard_prompt_weight: float = 2.0 / 3.0
    creative_writing_weight: float = 1.0 / 3.0


def rankdata_average(values: Iterable[float]) -> list[float]:
    """Return average ranks (1-indexed) with ties averaged, matching SciPy semantics."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1
    return ranks


def pearsonr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return math.nan
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [v - mean_x for v in x]
    dy = [v - mean_y for v in y]
    denom_x = math.sqrt(sum(v * v for v in dx))
    denom_y = math.sqrt(sum(v * v for v in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return math.nan
    return sum(a * b for a, b in zip(dx, dy)) / (denom_x * denom_y)


def spearmanr(x: list[float], y: list[float]) -> float:
    return pearsonr(rankdata_average(x), rankdata_average(y))


def load_leaderboard_scores(results_dir: Path, category: str) -> pd.DataFrame:
    csv_path = results_dir / f"{category}_leaderboard_all.csv"
    df = pd.read_csv(csv_path)
    required = {"Model", "Scores (%)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return df[["Model", "Scores (%)"]].rename(columns={"Scores (%)": category})


def load_weighted_scores(results_dir: Path, weights: WeightedConfig) -> pd.DataFrame:
    hard_df = load_leaderboard_scores(results_dir, "hard_prompt")
    creative_df = load_leaderboard_scores(results_dir, "creative_writing")
    merged = hard_df.merge(creative_df, on="Model", how="inner", validate="one_to_one")
    merged["weighted_score"] = (
        weights.hard_prompt_weight * merged["hard_prompt"]
        + weights.creative_writing_weight * merged["creative_writing"]
    )
    merged["weighted_rank"] = merged["weighted_score"].rank(method="average", ascending=False)
    return merged.sort_values("weighted_score", ascending=False).reset_index(drop=True)


def parse_intermediate_model(model_name: str) -> tuple[str, int]:
    match = LR_STEP_PATTERN.search(model_name)
    if not match:
        raise ValueError(
            f"Could not parse learning rate / step from intermediate model name: {model_name}"
        )
    return match.group("lr"), int(match.group("step"))


def parse_final_lr(model_name: str) -> str | None:
    match = FINAL_LR_PATTERN.search(model_name)
    if not match:
        return None
    return match.group("lr")


def build_joined_dataframe(intermediate_dir: Path, final_dir: Path) -> pd.DataFrame:
    weights = WeightedConfig()
    intermediate = load_weighted_scores(intermediate_dir, weights)
    final = load_weighted_scores(final_dir, weights)

    intermediate = intermediate.copy()
    intermediate = intermediate[intermediate["Model"].str.contains(r"-step\d+$", regex=True)].copy()
    parsed = intermediate["Model"].apply(parse_intermediate_model)
    intermediate["lr"] = parsed.apply(lambda item: item[0])
    intermediate["step"] = parsed.apply(lambda item: item[1])

    final = final.copy()
    final["lr"] = final["Model"].apply(parse_final_lr)
    final = final.dropna(subset=["lr"])

    final = final[["Model", "lr", "hard_prompt", "creative_writing", "weighted_score", "weighted_rank"]].rename(
        columns={
            "Model": "final_model",
            "hard_prompt": "final_hard_prompt",
            "creative_writing": "final_creative_writing",
            "weighted_score": "final_weighted_score",
            "weighted_rank": "final_weighted_rank",
        }
    )

    joined = intermediate.merge(final, on="lr", how="inner", validate="many_to_one")
    joined = joined.rename(
        columns={
            "Model": "intermediate_model",
            "hard_prompt": "intermediate_hard_prompt",
            "creative_writing": "intermediate_creative_writing",
            "weighted_score": "intermediate_weighted_score",
            "weighted_rank": "intermediate_weighted_rank",
        }
    )
    return joined.sort_values(["final_weighted_rank", "step"]).reset_index(drop=True)


def summarize_correlations(joined: pd.DataFrame) -> dict[str, float]:
    x_score = joined["intermediate_weighted_score"].tolist()
    y_score = joined["final_weighted_score"].tolist()
    x_rank = joined["intermediate_weighted_rank"].tolist()
    y_rank = joined["final_weighted_rank"].tolist()
    return {
        "n_intermediate_checkpoints": float(len(joined)),
        "n_final_models": float(joined["lr"].nunique()),
        "spearman_weighted_score_vs_final_score": spearmanr(x_score, y_score),
        "spearman_intermediate_rank_vs_final_rank": spearmanr(x_rank, y_rank),
    }


def summarize_by_step_peak(joined: pd.DataFrame) -> pd.DataFrame:
    idx = joined.groupby("lr")["intermediate_weighted_score"].idxmax()
    best = joined.loc[idx].copy()
    best = best.sort_values("final_weighted_rank")
    best["checkpoint_label"] = best.apply(
        lambda row: f"{row['lr']}@{int(row['step'])}", axis=1
    )
    return best


def build_step_correlation_dataframe(joined: pd.DataFrame) -> pd.DataFrame:
    step_rows = []
    for step, group in joined.groupby("step"):
        if group["lr"].nunique() < 2:
            continue
        step_rows.append(
            {
                "step": int(step),
                "n_lrs": int(group["lr"].nunique()),
                "spearman_score_vs_final": spearmanr(
                    group["intermediate_weighted_score"].tolist(),
                    group["final_weighted_score"].tolist(),
                ),
                "spearman_rank_vs_final": spearmanr(
                    group["intermediate_weighted_rank"].tolist(),
                    group["final_weighted_rank"].tolist(),
                ),
            }
        )
    return pd.DataFrame(step_rows).sort_values("step").reset_index(drop=True)


def save_lr_trajectory_plot(joined: pd.DataFrame, output_path: Path, overall_spearman: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for _, lr_group in joined.groupby("lr"):
        lr_group = lr_group.sort_values("step")
        lr = lr_group["lr"].iloc[0]
        final_score = lr_group["final_weighted_score"].iloc[0]
        final_rank = int(lr_group["final_weighted_rank"].iloc[0])
        ax.plot(
            lr_group["step"],
            lr_group["intermediate_weighted_score"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=f"lr={lr} (final rank {final_rank}, final={final_score:.2f})",
        )

    ax.set_title(
        "Intermediate weighted AH score trajectory by learning rate\n"
        f"Overall Spearman(intermediate score, final score) = {overall_spearman:.4f}"
    )
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Weighted AH score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scatter_plot(joined: pd.DataFrame, output_path: Path, overall_spearman: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    lrs = sorted(joined["lr"].unique())
    cmap = plt.get_cmap("tab10")

    for idx, lr in enumerate(lrs):
        group = joined[joined["lr"] == lr].sort_values("step")
        ax.scatter(
            group["intermediate_weighted_score"],
            group["final_weighted_score"],
            s=50,
            alpha=0.8,
            color=cmap(idx % 10),
            label=f"lr={lr}",
        )
        best_row = group.loc[group["intermediate_weighted_score"].idxmax()]
        ax.annotate(
            f"{lr}@{int(best_row['step'])}",
            (best_row["intermediate_weighted_score"], best_row["final_weighted_score"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    x_min = joined["intermediate_weighted_score"].min()
    x_max = joined["intermediate_weighted_score"].max()
    y_min = joined["final_weighted_score"].min()
    y_max = joined["final_weighted_score"].max()
    lo = min(x_min, y_min)
    hi = max(x_max, y_max)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray", alpha=0.7)

    ax.set_title(f"Intermediate vs final weighted AH score\nSpearman = {overall_spearman:.4f}")
    ax.set_xlabel("Intermediate weighted score")
    ax.set_ylabel("Final weighted score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_spearman_by_step_plot(step_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(
        step_df["step"],
        step_df["spearman_rank_vs_final"],
        marker="s",
        linewidth=2,
        label="Intermediate rank vs final rank",
    )
    ax.set_title("Spearman correlation with final outcome across checkpoints")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=DEFAULT_INTERMEDIATE_DIR,
        help="Directory containing intermediate leaderboard CSVs.",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=DEFAULT_FINAL_DIR,
        help="Directory containing final leaderboard CSVs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the joined per-checkpoint dataset as CSV.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/results/qwen3-sft-intermediate/plots"),
        help="Directory to save trajectory/scatter/Spearman plots.",
    )
    args = parser.parse_args()

    joined = build_joined_dataframe(args.intermediate_dir, args.final_dir)
    summary = summarize_correlations(joined)
    best = summarize_by_step_peak(joined)
    step_df = build_step_correlation_dataframe(joined)

    print("Weighted score: 2/3 hard_prompt + 1/3 creative_writing")
    print(
        f"Matched {int(summary['n_intermediate_checkpoints'])} intermediate checkpoints "
        f"across {int(summary['n_final_models'])} final LR groups."
    )
    print(
        "Spearman(intermediate weighted score, final weighted score) = "
        f"{format_float(summary['spearman_weighted_score_vs_final_score'])}"
    )
    print(
        "Spearman(intermediate weighted rank, final weighted rank) = "
        f"{format_float(summary['spearman_intermediate_rank_vs_final_rank'])}"
    )
    print()
    print("Best intermediate checkpoint per LR (by weighted score):")
    print(
        best[
            [
                "lr",
                "step",
                "intermediate_weighted_score",
                "final_weighted_score",
                "intermediate_weighted_rank",
                "final_weighted_rank",
            ]
        ].to_string(index=False)
    )

    if not step_df.empty:
        print()
        print("Per-step correlation across LR groups:")
        print(step_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot_dir = args.plot_dir
    trajectory_plot = plot_dir / "intermediate_lr_trajectories.png"
    scatter_plot = plot_dir / "intermediate_vs_final_scatter.png"
    spearman_plot = plot_dir / "spearman_vs_checkpoint.png"
    save_lr_trajectory_plot(
        joined,
        trajectory_plot,
        summary["spearman_weighted_score_vs_final_score"],
    )
    save_scatter_plot(joined, scatter_plot, summary["spearman_weighted_score_vs_final_score"])
    if not step_df.empty:
        save_spearman_by_step_plot(step_df, spearman_plot)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(args.output_csv, index=False)
        print()
        print(f"Saved joined checkpoint/final comparison to {args.output_csv}")

    print(f"Saved LR trajectory plot to {trajectory_plot}")
    print(f"Saved intermediate-vs-final scatter plot to {scatter_plot}")
    if not step_df.empty:
        print(f"Saved Spearman-by-step plot to {spearman_plot}")


if __name__ == "__main__":
    main()
