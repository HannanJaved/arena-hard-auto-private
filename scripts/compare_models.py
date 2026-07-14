#!/usr/bin/env python3
"""
compare_models.py — pull together every eval result we have for two models
and print a side-by-side comparison.

Sources (all relative to WORKSPACE):
  Arena-Hard  : arena-hard-auto/results/<arena-hard-dir>/{hard_prompt,creative_writing}_leaderboard_all.csv
                composite = hard_weight * hard_prompt + creative_weight * creative_writing
  ELO         : evaluation_results/openjury-elo/**/summary.json  (matched by checkpoint path)
  MT-Bench    : evaluation_results/judgearena-mtbench/**/results-*.json (matched by checkpoint path)
  Static evals: evaluation_results/{task}/{sanitized_checkpoint_path}/results_*.json
                tasks: arc_challenge, gpqa, gsm8k, hellaswag, ifeval, piqa, truthfulqa

For ELO and static evals, --pretrain-baseline (default qwen3-4b-base) is also
pulled in as a reference row/column with deltas, since that's the pre-SFT
checkpoint both model-a and model-b were fine-tuned from.

Usage:
    python compare_models.py
    python compare_models.py --model-a qwen3-4b-sft-lr3e-5 --model-b qwen3-4b-sft-lr3e-5-AO
    python compare_models.py --arena-hard-dir qwen3-4b-compared-to-AO --json-out report.json
    python compare_models.py --plot arena-hard-auto/plots/qwen3-4b-sft-lr3e-5_vs_AO.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

WORKSPACE = Path("/data/horse/ws/hama901h-BFTranslation")
ARENA_HARD_RESULTS = WORKSPACE / "arena-hard-auto" / "results"
API_CONFIG_PATH = WORKSPACE / "arena-hard-auto" / "config" / "api_config.yaml"
MTBENCH_RESULT_DIR = WORKSPACE / "evaluation_results" / "judgearena-mtbench"
ELO_RESULT_DIR = WORKSPACE / "evaluation_results" / "openjury-elo"
EVAL_RESULTS_DIR = WORKSPACE / "evaluation_results"

STATIC_TASKS = ["arc_challenge", "gpqa", "gsm8k", "hellaswag", "ifeval", "piqa", "truthfulqa"]

# Metric to headline per static task. The sentinel AVERAGE_METRICS means:
# average every numeric metric reported for that task instead of picking one.
AVERAGE_METRICS = "__average__"
HEADLINE_METRIC = {
    "arc_challenge": "acc_norm,none",
    "gpqa": "acc_norm,none",
    "gsm8k": AVERAGE_METRICS,   # avg of exact_match,strict-match + exact_match,flexible-extract
    "hellaswag": "acc_norm,none",
    "ifeval": AVERAGE_METRICS,  # avg of {prompt,inst}_level_{strict,loose}_acc
    "piqa": "acc_norm,none",
    "truthfulqa": "acc,none",
}


def load_api_config() -> dict:
    with open(API_CONFIG_PATH) as fh:
        return yaml.safe_load(fh) or {}


def model_path(api_config: dict, model_name: str) -> str:
    entry = api_config.get(model_name)
    if not isinstance(entry, dict) or "model" not in entry:
        print(f"ERROR: model '{model_name}' not found in {API_CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    return entry["model"]


# ---------------------------------------------------------------------------
# Arena-Hard
# ---------------------------------------------------------------------------

def _read_leaderboard_csv(path: Path) -> dict:
    """Return {model_name: {'score': float, 'ci': str}}."""
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["Model"].strip()
            try:
                score = float(row["Scores (%)"])
            except (KeyError, ValueError):
                continue
            out[name] = {"score": score, "ci": row.get("CI (%)", "").strip()}
    return out


def get_arena_hard(arena_hard_dir: str, model_names: list[str], hard_weight: float, creative_weight: float) -> dict:
    base_dir = ARENA_HARD_RESULTS / arena_hard_dir
    hard = _read_leaderboard_csv(base_dir / "hard_prompt_leaderboard_all.csv")
    creative = _read_leaderboard_csv(base_dir / "creative_writing_leaderboard_all.csv")

    result = {}
    for name in model_names:
        h = hard.get(name)
        c = creative.get(name)
        entry = {"hard_prompt": h, "creative_writing": c, "composite": None}
        if h and c:
            entry["composite"] = hard_weight * h["score"] + creative_weight * c["score"]
        result[name] = entry
    return result


# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------

def get_elo(path: str) -> dict | None:
    target = f"VLLM/{path}"
    matches = []
    for summary_file in ELO_RESULT_DIR.rglob("summary.json"):
        try:
            data = json.loads(summary_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("model") != target:
            continue
        elo_entry = next((e for e in data.get("elo_summary", []) if e.get("model") == target), None)
        matches.append({
            "file": str(summary_file.relative_to(WORKSPACE)),
            "mtime": summary_file.stat().st_mtime,
            "judge": data.get("judge"),
            "n_battles": data.get("n_battles"),
            "wins": data.get("wins"),
            "losses": data.get("losses"),
            "ties": data.get("ties"),
            "winrate": data.get("winrate"),
            "elo_mean": elo_entry.get("elo_mean") if elo_entry else None,
            "elo_std": elo_entry.get("elo_std") if elo_entry else None,
        })
    if not matches:
        return None
    matches.sort(key=lambda m: m["mtime"], reverse=True)
    return matches[0]


# ---------------------------------------------------------------------------
# MT-Bench (pairwise battle winrate, model_A = subject, model_B = baseline)
# ---------------------------------------------------------------------------

def get_mtbench(path: str, opponent_path: str | None = None) -> dict | None:
    target = f"VLLM/{path}"
    opponent = f"VLLM/{opponent_path}" if opponent_path else None
    matches = []
    for results_file in MTBENCH_RESULT_DIR.rglob("results-*.json"):
        try:
            data = json.loads(results_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("model_A") != target:
            continue
        matches.append({
            "file": str(results_file.relative_to(WORKSPACE)),
            "mtime": results_file.stat().st_mtime,
            "baseline": data.get("model_B"),
            "num_battles": data.get("num_battles"),
            "winrate": data.get("winrate"),
            "num_wins": data.get("num_wins"),
            "num_losses": data.get("num_losses"),
            "num_ties": data.get("num_ties"),
            "num_missing": data.get("num_missing"),
        })
    if not matches:
        return None
    if opponent:
        vs_opponent = [m for m in matches if m["baseline"] == opponent]
        if vs_opponent:
            matches = vs_opponent
    matches.sort(key=lambda m: m["mtime"], reverse=True)
    return matches[0]


# ---------------------------------------------------------------------------
# Static LM-eval tasks
# ---------------------------------------------------------------------------

def _sanitize_path(path: str) -> str:
    return path.replace("/", "__")


def get_static_evals(path: str) -> dict:
    sanitized = _sanitize_path(path)
    out = {}
    for task in STATIC_TASKS:
        task_dir = EVAL_RESULTS_DIR / task / sanitized
        files = sorted(task_dir.glob("results_*.json")) if task_dir.exists() else []
        if not files:
            out[task] = None
            continue
        try:
            data = json.loads(files[-1].read_text())
        except (json.JSONDecodeError, OSError):
            out[task] = None
            continue
        results = data.get("results", {})
        metrics = {}
        for _subtask, vals in results.items():
            for k, v in vals.items():
                if isinstance(v, (int, float)) and "stderr" not in k and k != "sample_len":
                    metrics[k] = v
        headline_key = HEADLINE_METRIC.get(task)
        if headline_key == AVERAGE_METRICS:
            # Full metric names (e.g. "prompt_level_strict_acc,none") get long once
            # averaged together; keep the label short here and leave the detail in
            # all_metrics so axis/table labels don't overflow or wrap.
            headline_label = f"avg({len(metrics)} metrics)" if metrics else None
            headline = sum(metrics.values()) / len(metrics) if metrics else None
        else:
            headline_label = headline_key
            headline = metrics.get(headline_key) if headline_key else None
        out[task] = {
            "file": str(files[-1].relative_to(WORKSPACE)),
            "headline_metric": headline_label,
            "headline_value": headline,
            "all_metrics": metrics,
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x, digits=2):
    return f"{x * 100:.{digits}f}%" if isinstance(x, (int, float)) else "N/A"


def fmt(x, digits=2):
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "N/A"


def build_report(model_a: str, model_b: str, arena_hard_dir: str, hard_weight: float, creative_weight: float,
                  pretrain_baseline: str | None = None) -> dict:
    api_config = load_api_config()
    path_a = model_path(api_config, model_a)
    path_b = model_path(api_config, model_b)

    arena_hard = get_arena_hard(arena_hard_dir, [model_a, model_b], hard_weight, creative_weight)
    elo_a, elo_b = get_elo(path_a), get_elo(path_b)
    mtbench_a = get_mtbench(path_a, opponent_path=path_b)
    mtbench_b = get_mtbench(path_b, opponent_path=path_a)
    static_a, static_b = get_static_evals(path_a), get_static_evals(path_b)

    pretrain_baseline_entry = None
    if pretrain_baseline:
        base_path = model_path(api_config, pretrain_baseline)
        pretrain_baseline_entry = {
            "name": pretrain_baseline,
            "path": base_path,
            "elo": get_elo(base_path),
            "static_evals": get_static_evals(base_path),
        }

    return {
        "model_a": {"name": model_a, "path": path_a},
        "model_b": {"name": model_b, "path": path_b},
        "pretrain_baseline": pretrain_baseline_entry,
        "arena_hard": {"dir": arena_hard_dir, "hard_weight": hard_weight,
                        "creative_weight": creative_weight, "results": arena_hard},
        "elo": {model_a: elo_a, model_b: elo_b},
        "mtbench": {model_a: mtbench_a, model_b: mtbench_b},
        "static_evals": {model_a: static_a, model_b: static_b},
    }


def print_report(report: dict) -> None:
    a_name, b_name = report["model_a"]["name"], report["model_b"]["name"]
    w = max(len(a_name), len(b_name), 20)

    print("=" * 78)
    print(f"  Model comparison: {a_name}  vs.  {b_name}")
    print("=" * 78)
    print(f"  {a_name}: {report['model_a']['path']}")
    print(f"  {b_name}: {report['model_b']['path']}")

    # --- Arena-Hard ---
    ah = report["arena_hard"]
    hw, cw = ah["hard_weight"], ah["creative_weight"]
    print(f"\n--- Arena-Hard (dir: {ah['dir']}, composite = {hw:.3g}*hard_prompt + {cw:.3g}*creative_writing) ---")
    print(f"  {'Model':<{w}}  {'hard_prompt':>24}  {'creative_writing':>24}  {'composite':>10}")
    for name, e in ah["results"].items():
        hp = e["hard_prompt"]
        cwv = e["creative_writing"]
        hp_s = f"{hp['score']:.1f}% {hp['ci']}" if hp else "N/A"
        cw_s = f"{cwv['score']:.1f}% {cwv['ci']}" if cwv else "N/A"
        comp_s = f"{e['composite']:.2f}%" if e["composite"] is not None else "N/A"
        print(f"  {name:<{w}}  {hp_s:>24}  {cw_s:>24}  {comp_s:>10}")

    # --- ELO ---
    pb = report["pretrain_baseline"]
    pb_elo = pb["elo"] if pb else None
    pb_name = pb["name"] if pb else None
    print(f"\n--- ELO (OpenJury / LMArena) ---")
    delta_header = f"{'Delta vs ' + pb_name:>14}" if pb else ""
    print(f"  {'Model':<{w}}  {'elo_mean':>10}  {'elo_std':>8}  {'winrate':>8}  {'W-L-T':>12}  {'n_battles':>9}  {delta_header}")
    if pb:
        row_names = [pb_name, a_name, b_name]
    else:
        row_names = [a_name, b_name]
    for name in row_names:
        e = pb_elo if (pb and name == pb_name) else report["elo"][name]
        if not e:
            line = f"  {name:<{w}}  {'N/A':>10}  {'N/A':>8}  {'N/A':>8}  {'N/A':>12}  {'N/A':>9}"
        else:
            wlt = f"{e['wins']}-{e['losses']}-{e['ties']}"
            line = (f"  {name:<{w}}  {fmt(e['elo_mean']):>10}  {fmt(e['elo_std']):>8}  "
                    f"{pct(e['winrate']):>8}  {wlt:>12}  {e['n_battles']!s:>9}")
        if pb:
            if name == pb_name:
                delta_s = "-"
            elif e and pb_elo and e.get("elo_mean") is not None and pb_elo.get("elo_mean") is not None:
                delta_s = f"{e['elo_mean'] - pb_elo['elo_mean']:+.2f}"
            else:
                delta_s = "N/A"
            line += f"  {delta_s:>14}"
        print(line)

    # --- MT-Bench ---
    print(f"\n--- MT-Bench (JudgeArena pairwise winrate, model as subject) ---")
    print(f"  {'Model':<{w}}  {'winrate':>8}  {'W-L-T':>12}  {'n_battles':>9}  baseline")
    for name in (a_name, b_name):
        m = report["mtbench"][name]
        if not m:
            print(f"  {name:<{w}}  {'N/A':>8}  {'N/A':>12}  {'N/A':>9}  (no MT-Bench run found for this model)")
            continue
        wlt = f"{m['num_wins']}-{m['num_losses']}-{m['num_ties']}"
        print(f"  {name:<{w}}  {pct(m['winrate']):>8}  {wlt:>12}  {m['num_battles']!s:>9}  {m['baseline']}")

    # --- Static evals ---
    pb_static = pb["static_evals"] if pb else None
    print(f"\n--- Static evals (lm-evaluation-harness) ---")
    if pb:
        print(f"  {'Task':<15}  {'Metric':<16}  {pb_name:>13}  {a_name:>22}  {b_name:>22}")
    else:
        print(f"  {'Task':<15}  {'Metric':<16}  {a_name:>14}  {b_name:>14}")
    for task in STATIC_TASKS:
        sa = report["static_evals"][a_name][task]
        sb = report["static_evals"][b_name][task]
        sp = pb_static[task] if pb_static else None
        metric = (sa or sb or sp or {}).get("headline_metric") or "N/A"
        va = sa["headline_value"] if sa else None
        vb = sb["headline_value"] if sb else None
        if pb:
            vp = sp["headline_value"] if sp else None
            va_s = pct(va) if va is None else f"{pct(va)} ({(va - vp) * 100:+.2f}pp)" if vp is not None else pct(va)
            vb_s = pct(vb) if vb is None else f"{pct(vb)} ({(vb - vp) * 100:+.2f}pp)" if vp is not None else pct(vb)
            print(f"  {task:<15}  {metric:<16}  {pct(vp):>13}  {va_s:>22}  {vb_s:>22}")
        else:
            print(f"  {task:<15}  {metric:<16}  {pct(va):>14}  {pct(vb):>14}")

    print("=" * 78)


# ---------------------------------------------------------------------------
# Ratio chart: model_b's score as a multiplier of model_a's (base = 1.0x)
# ---------------------------------------------------------------------------

# Status palette (fixed, reserved meaning) — see dataviz skill references/palette.md
STATUS_GOOD = "#0ca30c"      # ratio >= 1 : model_b scores higher
STATUS_CRITICAL = "#d03b3b"  # ratio < 1  : model_b scores lower
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def build_ratio_rows(report: dict) -> list[dict]:
    """Flatten every comparable metric into {label, base, other, ratio}, ratio = other/base."""
    a_name, b_name = report["model_a"]["name"], report["model_b"]["name"]
    rows = []

    ah = report["arena_hard"]["results"]
    a_ah, b_ah = ah[a_name], ah[b_name]
    if a_ah["composite"] is not None and b_ah["composite"] is not None:
        rows.append({"label": "Arena-Hard composite", "base": a_ah["composite"], "other": b_ah["composite"]})
    if a_ah["hard_prompt"] and b_ah["hard_prompt"]:
        rows.append({"label": "Arena-Hard hard_prompt", "base": a_ah["hard_prompt"]["score"], "other": b_ah["hard_prompt"]["score"]})
    if a_ah["creative_writing"] and b_ah["creative_writing"]:
        rows.append({"label": "Arena-Hard creative_writing", "base": a_ah["creative_writing"]["score"], "other": b_ah["creative_writing"]["score"]})

    elo_a, elo_b = report["elo"][a_name], report["elo"][b_name]
    if elo_a and elo_b and elo_a.get("elo_mean") is not None and elo_b.get("elo_mean") is not None:
        rows.append({"label": "ELO (elo_mean)", "base": elo_a["elo_mean"], "other": elo_b["elo_mean"]})

    mtb_a, mtb_b = report["mtbench"][a_name], report["mtbench"][b_name]
    if mtb_a and mtb_b and mtb_a.get("winrate") is not None and mtb_b.get("winrate") is not None:
        rows.append({"label": "MT-Bench winrate", "base": mtb_a["winrate"], "other": mtb_b["winrate"]})

    static_a, static_b = report["static_evals"][a_name], report["static_evals"][b_name]
    for task in STATIC_TASKS:
        sa, sb = static_a.get(task), static_b.get(task)
        if sa and sb and sa["headline_value"] is not None and sb["headline_value"] is not None:
            rows.append({
                "label": f"{task} ({sa['headline_metric']})",
                "base": sa["headline_value"],
                "other": sb["headline_value"],
            })

    for r in rows:
        r["ratio"] = (r["other"] / r["base"]) if r["base"] else None
    return [r for r in rows if r["ratio"] is not None]


def plot_ratio_chart(report: dict, rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_name, b_name = report["model_a"]["name"], report["model_b"]["name"]
    plot_rows = list(reversed(rows))  # first metric ends up on top

    labels = [r["label"] for r in plot_rows]
    ratios = [r["ratio"] for r in plot_rows]
    colors = [STATUS_GOOD if r >= 1 else STATUS_CRITICAL for r in ratios]

    fig_h = max(3.5, 0.5 * len(plot_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    y = list(range(len(plot_rows)))
    # Diverging bars anchored at the baseline (1.0x): barh's default left=0 is
    # invalid on a log axis and silently collapses near-1.0 bars against the
    # view edge, so each bar is drawn explicitly from 1.0 to its ratio.
    lefts = [min(1.0, r) for r in ratios]
    widths = [abs(r - 1.0) for r in ratios]
    ax.barh(y, widths, left=lefts, height=0.5, color=colors, zorder=3)

    ax.set_xscale("log", base=2)
    ax.axvline(1.0, color=BASELINE_AXIS, linewidth=1, zorder=2)

    max_ratio, min_ratio = max(ratios + [1.0]), min(ratios + [1.0])
    # Log-scale barh autoscales xlim to the tight data min/max (no default margin),
    # so the smallest bar otherwise lands flush on the axes edge, on top of the
    # y-tick labels. Pad explicitly, in log2 space, with a floor so a tightly
    # clustered set of ratios still gets breathing room for labels.
    lo2, hi2 = math.log2(min_ratio), math.log2(max_ratio)
    pad2 = max((hi2 - lo2) * 0.12, 0.5)
    ax.set_xlim(2 ** (lo2 - pad2), 2 ** (hi2 + pad2))

    candidate_ticks = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    ticks = sorted({t for t in candidate_ticks if min_ratio * 0.85 <= t <= max_ratio * 1.35} | {1})
    ax.set_xticks(ticks)
    ax.set_xticklabels([("1.0x (base)" if t == 1 else f"{t:g}x") for t in ticks], color=INK_MUTED, fontsize=9)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK_PRIMARY, fontsize=10)
    ax.set_ylim(-0.6, len(plot_rows) - 0.4)

    for yi, r in zip(y, plot_rows):
        ratio = r["ratio"]
        label = f"{ratio:.2f}x"
        ha = "left" if ratio >= 1 else "right"
        offset = (6, 0) if ratio >= 1 else (-6, 0)
        ax.annotate(label, xy=(ratio, yi), xytext=offset, textcoords="offset points",
                    va="center", ha=ha, fontsize=9, color=INK_SECONDARY, zorder=4)

    ax.grid(axis="x", color=GRID_HAIRLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(axis="both", length=0)

    ax.set_title(f"{b_name}\nas a multiplier of {a_name} (base = 1.0x)",
                 fontsize=12, color=INK_PRIMARY, loc="left", pad=14)
    ax.set_xlabel(f"{b_name} / {a_name}  (log scale)", fontsize=9, color=INK_MUTED)

    good_handle = plt.Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=STATUS_GOOD,
                              markeredgecolor="none", markersize=10, label=f"{b_name} higher")
    bad_handle = plt.Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=STATUS_CRITICAL,
                             markeredgecolor="none", markersize=10, label=f"{b_name} lower")
    legend = ax.legend(handles=[good_handle, bad_handle], loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    if not any(r["label"] == "MT-Bench winrate" for r in rows):
        fig.text(0.01, 0.005, "MT-Bench: no completed run found for either model, omitted", fontsize=8, color=INK_MUTED)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-a", default="qwen3-4b-sft-lr3e-5", help="First model key in api_config.yaml.")
    parser.add_argument("--model-b", default="qwen3-4b-sft-lr3e-5-AO", help="Second model key in api_config.yaml.")
    parser.add_argument("--arena-hard-dir", default="qwen3-4b-compared-to-AO",
                         help="Subdirectory of arena-hard-auto/results/ holding the leaderboard CSVs.")
    parser.add_argument("--hard-weight", type=float, default=2 / 3, help="Weight for hard_prompt in the composite score.")
    parser.add_argument("--creative-weight", type=float, default=1 / 3, help="Weight for creative_writing in the composite score.")
    parser.add_argument("--pretrain-baseline", default="qwen3-4b-base",
                         help="Pre-SFT checkpoint to show as a reference row/column with deltas for ELO and static "
                              "evals (both model-a and model-b were SFT'd from this). Pass '' to omit it.")
    parser.add_argument("--json-out", help="Optional path to also write the full report as JSON.")
    parser.add_argument("--plot", nargs="?", const="__default__",
                         help="Write a PNG indexing model-b's score to model-a's score (base = 1.0x) per metric. "
                              "Pass a path, or omit the value to use arena-hard-auto/plots/<model_a>_vs_<model_b>_multiplier.png.")
    args = parser.parse_args()

    report = build_report(args.model_a, args.model_b, args.arena_hard_dir, args.hard_weight, args.creative_weight,
                           pretrain_baseline=args.pretrain_baseline or None)
    print_report(report)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {args.json_out}")

    if args.plot:
        rows = build_ratio_rows(report)
        if not rows:
            print("\nWARNING: no metrics available for both models, skipping plot.", file=sys.stderr)
        else:
            plot_path = args.plot
            if plot_path == "__default__":
                plot_path = WORKSPACE / "arena-hard-auto" / "plots" / f"{args.model_a}_vs_{args.model_b}_multiplier.png"
            else:
                plot_path = Path(plot_path)
            plot_ratio_chart(report, rows, plot_path)
            print(f"\nRatio chart written to {plot_path}")


if __name__ == "__main__":
    main()
