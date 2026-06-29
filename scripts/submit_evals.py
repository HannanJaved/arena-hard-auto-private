#!/usr/bin/env python3
"""
submit_evals.py — submit all evaluation jobs for a list of models.

Usage:
    python submit_evals.py --models-file models.txt [--submit] [--dry-run]
    python submit_evals.py --models-file models.txt --skip-completed [--submit]

Venv assignments:
    arena-hard-auto/venv  → arena-hard generation
    venv-alpacaeval       → alpaca-eval generation + judgment
    venv-openjury         → MT-Bench (JudgeArena) + ELO estimation (OpenJury)
    venv-lm-eval          → arc_challenge, gpqa, gsm8k, hellaswag, ifeval, piqa, truthfulqa
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

VENV_ARENA    = WORKSPACE / "arena-hard-auto" / "venv" / "bin" / "python"
VENV_ALPACA   = WORKSPACE / "venv-alpacaeval" / "bin" / "python"
VENV_OPENJURY = WORKSPACE / "venv-openjury" / "bin" / "python"
VENV_LMEVAL   = WORKSPACE / "venv-lm-eval" / "bin" / "python"

STATIC_TASKS = ["arc_challenge", "gpqa", "gsm8k", "hellaswag", "ifeval", "piqa", "truthfulqa"]

LM_EVAL_LOG_DIR     = WORKSPACE / "logs" / "LM-eval"
ARENA_HARD_ANS_DIR  = WORKSPACE / "arena-hard-auto" / "data" / "arena-hard-v2.0" / "model_answer"
ALPACA_EVAL_OUT_DIR = WORKSPACE / "alpaca_eval_outputs"
MTBENCH_RESULT_DIR  = WORKSPACE / "evaluation_results" / "judgearena-mtbench"
ELO_RESULT_DIR      = WORKSPACE / "evaluation_results" / "openjury-elo"
API_CONFIG_PATH     = WORKSPACE / "arena-hard-auto" / "config" / "api_config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(label: str, python: Path, script: Path, extra_args: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
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
    """Skip models whose checkpoint path does not exist on disk."""
    valid = []
    for m in models:
        path = _model_path(api_config, m)
        if path and path.startswith("/") and not Path(path).exists():
            print(f"  SKIP (missing checkpoint): {m}\n    path: {path}", file=sys.stderr)
        else:
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


# ---------------------------------------------------------------------------
# Per-task completion checks
# ---------------------------------------------------------------------------

def _lmeval_sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:128]


def _is_lmeval_completed(task: str, model_name: str) -> bool:
    prefix = f"{task}_{_lmeval_sanitize(model_name)}_"
    for log_file in LM_EVAL_LOG_DIR.glob(f"{prefix}*.out"):
        try:
            if "END TIME:" in log_file.read_text(errors="replace"):
                return True
        except OSError:
            pass
    return False


def _is_arena_hard_completed(model_name: str) -> bool:
    return (ARENA_HARD_ANS_DIR / f"{model_name}.jsonl").exists()


def _is_alpaca_eval_completed(model_name: str) -> bool:
    # Direct path first, then one level of subdirectory (e.g. BO/model_name/)
    if (ALPACA_EVAL_OUT_DIR / model_name / "model_outputs.json").exists():
        return True
    return any(ALPACA_EVAL_OUT_DIR.glob(f"*/{model_name}/model_outputs.json"))


def _is_mtbench_completed(model_name: str, api_config: dict) -> bool:
    path = _model_path(api_config, model_name)
    if not path or not MTBENCH_RESULT_DIR.exists():
        return False
    # Directory names are truncated + hashed when too long, so we match on
    # the model_A field inside results-*.json instead of the directory name.
    vllm_model = f"VLLM/{path}"
    for results_file in MTBENCH_RESULT_DIR.rglob("results-*.json"):
        try:
            data = json.loads(results_file.read_text())
            if data.get("model_A") == vllm_model:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


def _is_elo_completed(model_name: str, api_config: dict) -> bool:
    path = _model_path(api_config, model_name)
    if not path or not ELO_RESULT_DIR.exists():
        return False
    vllm_model = f"VLLM/{path}"
    for summary_file in ELO_RESULT_DIR.rglob("summary.json"):
        try:
            data = json.loads(summary_file.read_text())
            if data.get("model") == vllm_model:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


# ---------------------------------------------------------------------------
# Partition helpers
# ---------------------------------------------------------------------------

def _partition(models: list[str], check_fn) -> tuple[list[str], list[str]]:
    pending, completed = [], []
    for m in models:
        (completed if check_fn(m) else pending).append(m)
    return pending, completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Submit all evaluation jobs for a model list.")
    parser.add_argument("--models-file", required=True, help="Text file with one model name per line.")
    parser.add_argument("--submit",  action="store_true", help="Submit SLURM jobs (automation scripts).")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit.")
    parser.add_argument("--baseline", required=True, help="Baseline model for MT-Bench.")
    parser.add_argument("--rerun", action="store_true", help="Re-run MT-Bench for all (no skipping).")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Check output files / logs before each task and skip models that are already done. "
            "Arena-hard: checks model_answer/{model}.jsonl. "
            "Alpaca-eval: checks alpaca_eval_outputs/{model}/model_outputs.json. "
            "MT-Bench: scans evaluation_results/judgearena-mtbench/**/results-*.json for model_A match. "
            "ELO: scans evaluation_results/openjury-elo/**/summary.json for model match. "
            "LM-eval tasks: checks logs/LM-eval/{task}_{model}_*.out for 'END TIME:'."
        ),
    )
    args = parser.parse_args()

    models_file = str(args.models_file)

    # For the automation scripts (arena-hard, alpaca-eval, mtbench, elo):
    #   --submit  → pass --submit
    #   --dry-run → pass --dry-run
    #   neither   → generate only (no flag)
    auto_flag = ["--submit"] if args.submit else (["--dry-run"] if args.dry_run else [])

    # For the static submit_*_from_list scripts:
    #   they submit by default; --dry-run suppresses submission
    static_flag = ["--dry-run"] if (args.dry_run or not args.submit) else []

    if args.skip_completed:
        print(f"\n{'='*60}")
        print("  Checking completed jobs ...")
        print(f"{'='*60}")
        all_models = _read_models(models_file)
        api_config = _load_api_config()
        all_models = _filter_missing_paths(all_models, api_config)
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

        # 1. Arena-Hard generation
        submit_filtered(
            "Arena-Hard generation",
            _is_arena_hard_completed,
            lambda tmp: run(
                "Arena-Hard generation (automate_arena_hard_generation_olmo3)",
                VENV_ARENA,
                SCRIPTS / "automate_arena_hard_generation_olmo3.py",
                ["--models-file", tmp] + auto_flag,
            ),
        )

        # 2. AlpacaEval generation
        submit_filtered(
            "AlpacaEval generation",
            _is_alpaca_eval_completed,
            lambda tmp: run(
                "AlpacaEval generation (automate_alpaca_eval)",
                VENV_ALPACA,
                SCRIPTS / "automate_alpaca_eval.py",
                ["--models-file", tmp] + auto_flag,
            ),
        )

        # 3. MT-Bench / JudgeArena
        mtbench_flag = (["--rerun-all"] if args.rerun else ["--skip-existing"]) + auto_flag
        submit_filtered(
            "MT-Bench / JudgeArena",
            lambda m: _is_mtbench_completed(m, api_config),
            lambda tmp: run(
                "MT-Bench / JudgeArena (automate_mtbench)",
                VENV_OPENJURY,
                WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench.py",
                ["--models-file", tmp, "--baseline-model", args.baseline] + mtbench_flag,
            ),
        )

        # 4. ELO estimation / OpenJury
        submit_filtered(
            "ELO estimation",
            lambda m: _is_elo_completed(m, api_config),
            lambda tmp: run(
                "ELO estimation (automate_elo_estimation)",
                VENV_OPENJURY,
                WORKSPACE / "OpenJury" / "scripts" / "automate_elo_estimation.py",
                ["--models-file", tmp] + auto_flag,
            ),
        )

        # 5. Static LM-eval tasks
        for task in STATIC_TASKS:
            submit_filtered(
                task,
                lambda m, t=task: _is_lmeval_completed(t, m),
                lambda tmp, t=task: run(
                    f"Static eval: {t}",
                    VENV_LMEVAL,
                    SCRIPTS / f"submit_{t}_from_list.py",
                    ["--models-file", tmp] + static_flag,
                ),
            )

        for f in temp_files:
            Path(f).unlink(missing_ok=True)

    else:
        # 1. Arena-Hard generation
        run(
            "Arena-Hard generation (automate_arena_hard_generation_olmo3)",
            VENV_ARENA,
            SCRIPTS / "automate_arena_hard_generation_olmo3.py",
            ["--models-file", models_file] + auto_flag,
        )

        # 2. AlpacaEval generation
        run(
            "AlpacaEval generation (automate_alpaca_eval)",
            VENV_ALPACA,
            SCRIPTS / "automate_alpaca_eval.py",
            ["--models-file", models_file] + auto_flag,
        )

        # 3. MT-Bench / JudgeArena
        if args.rerun:
            run(
                "MT-Bench / JudgeArena (automate_mtbench)",
                VENV_OPENJURY,
                WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench.py",
                ["--models-file", models_file, "--baseline-model", args.baseline, "--rerun-all"] + auto_flag,
            )
        else:
            run(
                "MT-Bench / JudgeArena (automate_mtbench)",
                VENV_OPENJURY,
                WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench.py",
                ["--models-file", models_file, "--baseline-model", args.baseline, "--skip-existing"] + auto_flag,
            )

        # 4. ELO estimation / OpenJury
        run(
            "ELO estimation (automate_elo_estimation)",
            VENV_OPENJURY,
            WORKSPACE / "OpenJury" / "scripts" / "automate_elo_estimation.py",
            ["--models-file", models_file] + auto_flag,
        )

        # 5. Static evals
        for task in STATIC_TASKS:
            run(
                f"Static eval: {task}",
                VENV_LMEVAL,
                SCRIPTS / f"submit_{task}_from_list.py",
                ["--models-file", models_file] + static_flag,
            )

    print(f"\n{'='*60}")
    print("Evaluation Jobs Submitted.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
