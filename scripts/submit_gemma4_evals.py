#!/usr/bin/env python3
from __future__ import annotations
"""
submit_gemma4_evals.py — submit Gemma 4 Instruct judge jobs only.

Runs (and only runs):
  1. Arena-Hard judgment   (no generation)
  2. AlpacaEval judgment   (no generation)
  3. MT-Bench
  4. ELO ratings

Does not submit static LM-eval tasks or any generation jobs. Expects Arena-Hard
answers and AlpacaEval model_outputs.json to already exist.

Usage:
    python submit_gemma4_evals.py --models-file models.txt --baseline <model> [--submit]
    python submit_gemma4_evals.py --models-file models.txt --baseline <model> --skip-completed [--submit]
    python submit_gemma4_evals.py --models-file models.txt --baseline <model> --tasks mtbench elo [--submit]

Judge default: Gemma4-31B-it

Result isolation vs Qwen3-Next:
    Arena-Hard judgments → model_judgment/Gemma4-31B-it/
    AlpacaEval judgments → alpaca_eval_outputs_gemma4/
    MT-Bench             → evaluation_results/judgearena-mtbench-gemma4/
    ELO                  → evaluation_results/openjury-elo-gemma4/
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

WORKSPACE = Path("/data/horse/ws/hama901h-BFTranslation")
SCRIPTS = WORKSPACE / "arena-hard-auto" / "scripts"

VENV_ARENA = WORKSPACE / "arena-hard-auto" / "venv" / "bin" / "python"
VENV_ALPACA = WORKSPACE / "venv-alpacaeval" / "bin" / "python"
VENV_OPENJURY = WORKSPACE / "venv-openjury" / "bin" / "python"

DEFAULT_JUDGE_MODEL = "Gemma4-31B-it"

ARENA_HARD_JUDGE_SCRIPT = SCRIPTS / "automate_arena_hard_judgment_gemma4.py"
ALPACA_JUDGE_SCRIPT = SCRIPTS / "automate_alpaca_eval_judgment_gemma4.py"
MTBENCH_SCRIPT = WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench_gemma4.py"
ELO_SCRIPT = WORKSPACE / "OpenJury" / "scripts" / "automate_elo_estimation_gemma4.py"

TASKS = [
    "arena_hard_judgment",
    "alpaca_eval_judgment",
    "mtbench",
    "elo",
]

TASK_GROUPS: dict[str, list[str]] = {
    "arena-hard": ["arena_hard_judgment"],
    "arena_hard": ["arena_hard_judgment"],
    "alpaca-eval": ["alpaca_eval_judgment"],
    "alpaca_eval": ["alpaca_eval_judgment"],
    "all": list(TASKS),
}

ARENA_HARD_JUDGMENT_DIR = (
    WORKSPACE / "arena-hard-auto" / "data" / "arena-hard-v2.0" / "model_judgment"
)
ALPACA_EVAL_JUDGE_OUT_DIR = WORKSPACE / "alpaca_eval_outputs_gemma4"
MTBENCH_RESULT_DIR = WORKSPACE / "evaluation_results" / "judgearena-mtbench-gemma4"
ELO_RESULT_DIR = WORKSPACE / "evaluation_results" / "openjury-elo-gemma4"
API_CONFIG_PATH = WORKSPACE / "arena-hard-auto" / "config" / "api_config.yaml"


def run(label: str, python: Path, script: Path, extra_args: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    cmd = [str(python), str(script)] + extra_args
    print("  CMD:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: {label} exited with code {result.returncode}", file=sys.stderr)


def _read_models(models_file: str) -> list[str]:
    models = []
    for line in Path(models_file).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            models.append(line)
    return models


def _filter_missing_paths(models: list[str], api_config: dict) -> list[str]:
    """Skip models whose checkpoint is missing or still training (no config.json)."""
    valid = []
    for m in models:
        path = _model_path(api_config, m)
        if path and path.startswith("/"):
            ckpt = Path(path)
            if not ckpt.exists():
                print(f"  SKIP (missing checkpoint): {m}\n    path: {path}", file=sys.stderr)
                continue
            if not (ckpt / "config.json").exists():
                print(
                    f"  SKIP (incomplete checkpoint, no config.json yet): {m}\n    path: {path}",
                    file=sys.stderr,
                )
                continue
        valid.append(m)
    return valid


def _write_temp_models(models: list[str]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write("\n".join(models) + "\n")
    f.close()
    return f.name


def _load_api_config() -> dict:
    try:
        with open(API_CONFIG_PATH) as fh:
            return yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as e:
        print(f"  WARNING: could not load api_config: {e}", file=sys.stderr)
        return {}


def _model_path(api_config: dict, model_name: str) -> str | None:
    entry = api_config.get(model_name)
    if isinstance(entry, dict):
        return entry.get("model")
    return None


def _print_completion_summary(label: str, completed: list[str], pending: list[str]) -> None:
    print(f"\n  [{label}]  {len(completed)} completed, {len(pending)} pending")
    for m in completed:
        print(f"    SKIP  {m}")
    for m in pending:
        print(f"    RUN   {m}")


def _is_arena_hard_judgment_completed(model_name: str, judge_model: str, baseline: str) -> bool:
    judge_dir = ARENA_HARD_JUDGMENT_DIR / judge_model
    for path in judge_dir.rglob(f"{model_name}.jsonl"):
        try:
            with open(path) as fh:
                first_line = fh.readline()
            data = json.loads(first_line)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("baseline") == baseline:
            return True
    return False


def _is_alpaca_eval_judgment_completed(model_name: str) -> bool:
    if (ALPACA_EVAL_JUDGE_OUT_DIR / model_name / "leaderboard_length_controlled.csv").exists():
        return True
    return any(ALPACA_EVAL_JUDGE_OUT_DIR.glob(f"*/{model_name}/leaderboard_length_controlled.csv"))


def _is_mtbench_completed(model_name: str, api_config: dict, judge_model: str) -> bool:
    path = _model_path(api_config, model_name)
    if not path or not MTBENCH_RESULT_DIR.exists():
        return False
    vllm_model = f"VLLM/{path}"
    judge_served = re.sub(r"[^A-Za-z0-9_.-]", "_", judge_model)
    expected_judges = {
        f"ChatOpenAI/{judge_served}",
        f"ChatOpenAI/{judge_model}",
        judge_model,
        judge_served,
    }
    for results_file in MTBENCH_RESULT_DIR.rglob("results-*.json"):
        try:
            data = json.loads(results_file.read_text())
            if data.get("model_A") != vllm_model:
                continue
            jm = data.get("judge_model")
            if jm is None or jm in expected_judges or judge_model in str(jm) or judge_served in str(jm):
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


def _is_elo_completed(model_name: str, api_config: dict, judge_model: str) -> bool:
    path = _model_path(api_config, model_name)
    judge_path = _model_path(api_config, judge_model)
    if not path or not ELO_RESULT_DIR.exists():
        return False
    vllm_model = f"VLLM/{path}"
    vllm_judge = f"VLLM/{judge_path}" if judge_path else None
    for summary_file in ELO_RESULT_DIR.rglob("summary.json"):
        try:
            data = json.loads(summary_file.read_text())
            if data.get("model") != vllm_model:
                continue
            if vllm_judge is None or data.get("judge") == vllm_judge:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


def _partition(models: list[str], check_fn) -> tuple[list[str], list[str]]:
    pending, completed = [], []
    for m in models:
        (completed if check_fn(m) else pending).append(m)
    return pending, completed


def _resolve_tasks(task_filter: list[str] | None) -> set[str]:
    if not task_filter:
        return set(TASKS)

    selected: set[str] = set()
    unknown: list[str] = []
    for raw_item in task_filter:
        for raw_task in raw_item.split(","):
            task = raw_task.strip()
            if not task:
                continue
            if task in TASK_GROUPS:
                selected.update(TASK_GROUPS[task])
            elif task in TASKS:
                selected.add(task)
            else:
                unknown.append(task)

    if unknown:
        print("Unknown task IDs:", ", ".join(unknown), file=sys.stderr)
        print(
            "Valid task IDs:",
            ", ".join(sorted(TASKS + list(TASK_GROUPS.keys()))),
            file=sys.stderr,
        )
        sys.exit(2)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Submit Gemma 4 Instruct judge jobs only: Arena-Hard judgment, "
            "AlpacaEval judgment, MT-Bench, and ELO (no generation, no static LM-eval)."
        )
    )
    parser.add_argument("--models-file", required=True, help="Text file with one model name per line.")
    parser.add_argument("--submit", action="store_true", help="Submit SLURM jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit.")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline model key from api_config.yaml (used by Arena-Hard and MT-Bench).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model key from api_config.yaml (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help=(
            "Subset of tasks (default: all four). "
            "Valid: arena_hard_judgment, alpaca_eval_judgment, mtbench, elo. "
            "Groups: arena-hard, alpaca-eval, all."
        ),
    )
    parser.add_argument("--rerun", action="store_true", help="Re-run MT-Bench for all (no skipping).")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip models that already have Gemma4 results. "
            f"Arena-Hard: model_judgment/{DEFAULT_JUDGE_MODEL}/... "
            "AlpacaEval: alpaca_eval_outputs_gemma4/{model}/leaderboard_length_controlled.csv. "
            "MT-Bench: evaluation_results/judgearena-mtbench-gemma4/**/results-*.json. "
            "ELO: evaluation_results/openjury-elo-gemma4/**/summary.json."
        ),
    )
    args = parser.parse_args()
    selected_tasks = _resolve_tasks(args.tasks)
    models_file = str(args.models_file)
    auto_flag = ["--submit"] if args.submit else (["--dry-run"] if args.dry_run else [])

    def submit_one(label: str, python: Path, script: Path, models_path: str, extra: list[str]) -> None:
        run(label, python, script, ["--models-file", models_path] + extra + auto_flag)

    if args.skip_completed:
        print(f"\n{'=' * 60}")
        print("  Checking completed Gemma4 judge jobs ...")
        print(f"{'=' * 60}")
        all_models = _filter_missing_paths(_read_models(models_file), _load_api_config())
        api_config = _load_api_config()
        temp_files: list[str] = []

        def submit_filtered(label: str, check_fn, run_fn) -> None:
            pending, completed = _partition(all_models, check_fn)
            _print_completion_summary(label, completed, pending)
            if not pending:
                print(f"  -> All models already done for {label}, skipping.")
                return
            tmp = _write_temp_models(pending)
            temp_files.append(tmp)
            run_fn(tmp)

        if "arena_hard_judgment" in selected_tasks:
            submit_filtered(
                "Arena-Hard judgment (Gemma4)",
                lambda m: _is_arena_hard_judgment_completed(m, args.judge_model, args.baseline),
                lambda tmp: submit_one(
                    "Arena-Hard judgment (automate_arena_hard_judgment_gemma4)",
                    VENV_ARENA,
                    ARENA_HARD_JUDGE_SCRIPT,
                    tmp,
                    ["--baseline", args.baseline, "--judge-model", args.judge_model],
                ),
            )

        if "alpaca_eval_judgment" in selected_tasks:
            submit_filtered(
                "AlpacaEval judgment (Gemma4)",
                _is_alpaca_eval_judgment_completed,
                lambda tmp: submit_one(
                    "AlpacaEval judgment (automate_alpaca_eval_judgment_gemma4)",
                    VENV_ALPACA,
                    ALPACA_JUDGE_SCRIPT,
                    tmp,
                    ["--judge-model", args.judge_model],
                ),
            )

        if "mtbench" in selected_tasks:
            mtbench_flag = ["--rerun-all"] if args.rerun else ["--skip-existing"]
            submit_filtered(
                "MT-Bench (Gemma4)",
                lambda m: _is_mtbench_completed(m, api_config, args.judge_model),
                lambda tmp: submit_one(
                    "MT-Bench (automate_mtbench_gemma4)",
                    VENV_OPENJURY,
                    MTBENCH_SCRIPT,
                    tmp,
                    ["--baseline-model", args.baseline, "--judge", args.judge_model] + mtbench_flag,
                ),
            )

        if "elo" in selected_tasks:
            submit_filtered(
                "ELO estimation (Gemma4)",
                lambda m: _is_elo_completed(m, api_config, args.judge_model),
                lambda tmp: submit_one(
                    "ELO estimation (automate_elo_estimation_gemma4)",
                    VENV_OPENJURY,
                    ELO_SCRIPT,
                    tmp,
                    ["--judge", args.judge_model],
                ),
            )

        for f in temp_files:
            Path(f).unlink(missing_ok=True)

    else:
        if "arena_hard_judgment" in selected_tasks:
            submit_one(
                "Arena-Hard judgment (automate_arena_hard_judgment_gemma4)",
                VENV_ARENA,
                ARENA_HARD_JUDGE_SCRIPT,
                models_file,
                ["--baseline", args.baseline, "--judge-model", args.judge_model],
            )

        if "alpaca_eval_judgment" in selected_tasks:
            submit_one(
                "AlpacaEval judgment (automate_alpaca_eval_judgment_gemma4)",
                VENV_ALPACA,
                ALPACA_JUDGE_SCRIPT,
                models_file,
                ["--judge-model", args.judge_model],
            )

        if "mtbench" in selected_tasks:
            mtbench_flag = ["--rerun-all"] if args.rerun else ["--skip-existing"]
            submit_one(
                "MT-Bench (automate_mtbench_gemma4)",
                VENV_OPENJURY,
                MTBENCH_SCRIPT,
                models_file,
                ["--baseline-model", args.baseline, "--judge", args.judge_model] + mtbench_flag,
            )

        if "elo" in selected_tasks:
            submit_one(
                "ELO estimation (automate_elo_estimation_gemma4)",
                VENV_OPENJURY,
                ELO_SCRIPT,
                models_file,
                ["--judge", args.judge_model],
            )

    print(f"\n{'=' * 60}")
    print("Gemma4 judge jobs submitted (judgment / MT-Bench / ELO only).")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
