#!/usr/bin/env python3
from __future__ import annotations
"""
submit_evals_alpha.py — submit all evaluation jobs for a list of models,
targeting the `alpha` partition (A100 40GB) instead of `capella` (H100).

This is a variant of submit_evals.py — see that file for the capella/H100
version (kept untouched per request). The judge model (~80B params, FP8,
~84GB of weight shards) does not fit on a single A100 40GB, so every judge
server here is sharded across multiple GPUs with vLLM tensor parallelism
(--judge-tp-size, default 4 -> ~21GB weights/GPU, leaving headroom for KV
cache; TP=2 does not fit). Candidate-model generation steps only need to
fit one (usually much smaller) model, so those keep TP=1 and just move to
the alpha partition to relieve capella queue pressure.

Usage:
    python submit_evals_alpha.py --models-file models.txt --baseline <model> [--submit] [--dry-run]
    python submit_evals_alpha.py --models-file models.txt --baseline <model> --skip-completed [--submit]
    python submit_evals_alpha.py --models-file models.txt --baseline <model> --evals dynamic [--submit]
    python submit_evals_alpha.py --models-file models.txt --baseline <model> --evals static [--submit]
    python submit_evals_alpha.py --models-file models.txt --baseline <model> --tasks mtbench arena-hard [--submit]
    python submit_evals_alpha.py --models-file models.txt --baseline <model> --judge-tp-size 8 [--submit]

Venv assignments (same as submit_evals.py):
    arena-hard-auto/venv  → arena-hard generation
    venv-alpacaeval       → alpaca-eval generation + judgment
    venv-openjury         → MT-Bench (JudgeArena) + ELO estimation (OpenJury)
    venv-lm-eval          → arc_challenge, gpqa, gsm8k, hellaswag, ifeval, piqa, truthfulqa

Automation scripts used here (alpha variants):
    automate_arena_hard_generation_alpha.py
    automate_arena_hard_judgment_alpha.py
    automate_alpaca_eval_alpha.py
    automate_alpaca_eval_judgment_alpha.py
    JudgeArena/scripts/automate_mtbench_alpha.py
    OpenJury/scripts/automate_elo_estimation_alpha.py
Static LM-eval tasks reuse the original submit_{task}_from_list.py scripts,
which already accept --partition, so they're just pointed at alpha here.
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

AUTOMATION_TASKS = [
    "arena_hard_generation",
    "arena_hard_judgment",
    "alpaca_eval_generation",
    "alpaca_eval_judgment",
    "mtbench",
    "elo",
]

# Dynamic = ArenaHard / AlpacaEval / MT-Bench / ELO (generation + judgment / rating).
DYNAMIC_TASKS = list(AUTOMATION_TASKS)

TASK_GROUPS: dict[str, list[str]] = {
    "arena-hard": ["arena_hard_generation", "arena_hard_judgment"],
    "arena_hard": ["arena_hard_generation", "arena_hard_judgment"],
    "alpaca-eval": ["alpaca_eval_generation", "alpaca_eval_judgment"],
    "alpaca_eval": ["alpaca_eval_generation", "alpaca_eval_judgment"],
    "lm-eval": STATIC_TASKS,
    "static": STATIC_TASKS,
    "dynamic": DYNAMIC_TASKS,
}

EVAL_SUITES: dict[str, list[str]] = {
    "static": STATIC_TASKS,
    "dynamic": DYNAMIC_TASKS,
    "both": AUTOMATION_TASKS + STATIC_TASKS,
}

ALL_TASK_IDS = set(AUTOMATION_TASKS + STATIC_TASKS + list(TASK_GROUPS.keys()))

LM_EVAL_LOG_DIR     = WORKSPACE / "logs" / "LM-eval"
ARENA_HARD_ANS_DIR      = WORKSPACE / "arena-hard-auto" / "data" / "arena-hard-v2.0" / "model_answer"
ARENA_HARD_JUDGMENT_DIR = WORKSPACE / "arena-hard-auto" / "data" / "arena-hard-v2.0" / "model_judgment"
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


def _run_capturing(label: str, python: Path, script: Path, extra_args: list[str]) -> list[str]:
    """Run script, print its output, and return SLURM job IDs parsed from stdout."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    cmd = [str(python), str(script)] + extra_args
    print("  CMD:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print(f"  WARNING: {label} exited with code {result.returncode}", file=sys.stderr)
    return re.findall(r"Job ID:\s*(\d+)", result.stdout)


def _dependency_str(job_ids: list[str]) -> str:
    """Build a SLURM afterok dependency string from a list of job IDs."""
    return "afterok:" + ":".join(job_ids) if job_ids else ""


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
            # Intermediate DPO/SFT dirs exist before the final HF export; vLLM needs config.json.
            if not (ckpt / "config.json").exists():
                print(
                    f"  SKIP (incomplete checkpoint, no config.json yet): {m}\n    path: {path}",
                    file=sys.stderr,
                )
                continue
        valid.append(m)
    return valid


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")[:120] or "models"


def _temp_models_prefix(models: list[str], eval_name: str) -> str:
    eval_token = _safe_name(eval_name.lower().replace(" ", "_"))
    if not models:
        model_token = "empty"
    elif len(models) == 1:
        model_token = _safe_name(models[0])
    else:
        model_token = f"{_safe_name(models[0])}_plus{len(models) - 1}"
    return f"{eval_token}__{model_token}__"


def _write_temp_models(models: list[str], eval_name: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=_temp_models_prefix(models, eval_name),
        suffix=".txt",
        delete=False,
    )
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
    if (ARENA_HARD_ANS_DIR / f"{model_name}.jsonl").exists():
        return True
    return any(ARENA_HARD_ANS_DIR.rglob(f"{model_name}.jsonl"))


def _is_arena_hard_judgment_completed(model_name: str, judge_model: str, baseline: str) -> bool:
    # Judgment files land in model_judgment/{judge_model}/compared_with_{baseline}/{model}.jsonl,
    # but directory placement isn't trustworthy on its own (files can be moved/renamed), so
    # confirm the "baseline" field recorded inside the judgment file matches the requested baseline.
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
    if (ALPACA_EVAL_OUT_DIR / model_name / "leaderboard_length_controlled.csv").exists():
        return True
    return any(ALPACA_EVAL_OUT_DIR.glob(f"*/{model_name}/leaderboard_length_controlled.csv"))


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


def _resolve_tasks(task_filter: list[str] | None, evals: str = "both") -> set[str]:
    suite = set(EVAL_SUITES[evals])
    if not task_filter:
        return suite

    selected: set[str] = set()
    unknown: list[str] = []
    for raw_item in task_filter:
        for raw_task in raw_item.split(","):
            task = raw_task.strip()
            if not task:
                continue
            if task in TASK_GROUPS:
                selected.update(TASK_GROUPS[task])
            elif task in ALL_TASK_IDS:
                selected.add(task)
            else:
                unknown.append(task)

    if unknown:
        print("Unknown task IDs:", ", ".join(unknown), file=sys.stderr)
        print(
            "Valid task IDs:",
            ", ".join(sorted(AUTOMATION_TASKS + STATIC_TASKS + sorted(TASK_GROUPS))),
            file=sys.stderr,
        )
        sys.exit(2)

    filtered = selected & suite
    dropped = sorted(selected - suite)
    if dropped:
        print(
            f"WARNING: --evals {evals} dropped tasks outside that suite: "
            + ", ".join(dropped),
            file=sys.stderr,
        )
    if not filtered:
        print(
            f"ERROR: no tasks left after applying --evals {evals} to --tasks.",
            file=sys.stderr,
        )
        sys.exit(2)
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Submit all evaluation jobs for a model list on the alpha (A100) partition.")
    parser.add_argument("--models-file", required=True, help="Text file with one model name per line.")
    parser.add_argument("--submit",  action="store_true", help="Submit SLURM jobs (automation scripts).")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit.")
    parser.add_argument("--baseline", required=True,
                        help="Baseline model: a legacy alias (instruct, base, ...) or any api_config.yaml model key.")
    parser.add_argument(
        "--judge-model", default="Qwen3-Next-80B-A3B-Instruct-FP8",
        help="Judge model name used for arena-hard and alpaca-eval judgment (default: Qwen3-Next-80B-A3B-Instruct-FP8).",
    )
    parser.add_argument(
        "--judge-tp-size", type=int, default=4,
        help=(
            "Number of A100 40GB GPUs to shard every judge server across via vLLM "
            "tensor parallelism (default: 4). Applies to arena-hard judgment, "
            "alpaca-eval judgment, MT-Bench, and ELO. TP=2 does not fit an ~80B "
            "FP8 judge in 40GB; TP=4 leaves ~19GB/GPU headroom for KV cache."
        ),
    )
    parser.add_argument(
        "--evals",
        choices=["static", "dynamic", "both"],
        default="both",
        help=(
            "Which eval suite to submit (default: both). "
            "dynamic = ArenaHard, AlpacaEval, MT-Bench, ELO; "
            "static = LM-eval tasks "
            f"({', '.join(STATIC_TASKS)}); "
            "both = dynamic + static. Can be combined with --tasks to further narrow."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help=(
            "Subset of evals to run within --evals (default: all tasks in that suite). "
            "Pass multiple tasks separated by spaces and/or commas, e.g. "
            "--tasks mtbench arena-hard. "
            "Automation tasks: arena_hard_generation, arena_hard_judgment, "
            "alpaca_eval_generation, alpaca_eval_judgment, mtbench, elo. "
            "Static LM-eval tasks: "
            + ", ".join(STATIC_TASKS)
            + ". Groups: arena-hard, alpaca-eval, dynamic, lm-eval (or static)."
        ),
    )
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
    selected_tasks = _resolve_tasks(args.tasks, args.evals)

    models_file = str(args.models_file)
    judge_tp = str(args.judge_tp_size)

    # For the automation scripts (arena-hard, alpaca-eval, mtbench, elo):
    #   --submit  → pass --submit
    #   --dry-run → pass --dry-run
    #   neither   → generate only (no flag)
    auto_flag = ["--submit"] if args.submit else (["--dry-run"] if args.dry_run else [])

    # For the static submit_*_from_list scripts:
    #   they submit by default; --dry-run suppresses submission.
    # Always route to the alpha partition.
    static_flag = (["--dry-run"] if (args.dry_run or not args.submit) else []) + ["--partition", "alpha"]

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
            tmp = _write_temp_models(pending, label)
            temp_files.append(tmp)
            run_fn(tmp)

        # 1. Arena-Hard generation → capture job IDs for dependency chaining
        arena_gen_job_ids: list[str] = []
        if "arena_hard_generation" in selected_tasks:
            arena_gen_pending, arena_gen_completed = _partition(all_models, _is_arena_hard_completed)
            _print_completion_summary("Arena-Hard generation", arena_gen_completed, arena_gen_pending)
            if arena_gen_pending:
                tmp = _write_temp_models(arena_gen_pending, "Arena-Hard generation")
                temp_files.append(tmp)
                if args.submit:
                    arena_gen_job_ids = _run_capturing(
                        "Arena-Hard generation (automate_arena_hard_generation_alpha)",
                        VENV_ARENA,
                        SCRIPTS / "automate_arena_hard_generation_alpha.py",
                        ["--models-file", tmp, "--submit"],
                    )
                else:
                    run(
                        "Arena-Hard generation (automate_arena_hard_generation_alpha)",
                        VENV_ARENA,
                        SCRIPTS / "automate_arena_hard_generation_alpha.py",
                        ["--models-file", tmp] + auto_flag,
                    )
            else:
                print("  -> All models already done for Arena-Hard generation, skipping.")

        # 1b. Arena-Hard judgment — only for models missing judgment, depends on new gen jobs
        if "arena_hard_judgment" in selected_tasks:
            arena_judg_check = lambda m: _is_arena_hard_judgment_completed(m, args.judge_model, args.baseline)
            submit_filtered(
                "Arena-Hard judgment",
                arena_judg_check,
                lambda tmp: run(
                    "Arena-Hard judgment (automate_arena_hard_judgment_alpha)",
                    VENV_ARENA,
                    SCRIPTS / "automate_arena_hard_judgment_alpha.py",
                    ["--models-file", tmp,
                     "--baseline", args.baseline,
                     "--judge-model", args.judge_model,
                     "--tensor-parallel-size", judge_tp]
                    + auto_flag
                    + (["--dependency", _dependency_str(arena_gen_job_ids)] if arena_gen_job_ids else []),
                ),
            )

        # 2. AlpacaEval generation → capture job IDs for dependency chaining
        alpaca_gen_job_ids: list[str] = []
        if "alpaca_eval_generation" in selected_tasks:
            alpaca_gen_pending, alpaca_gen_completed = _partition(all_models, _is_alpaca_eval_completed)
            _print_completion_summary("AlpacaEval generation", alpaca_gen_completed, alpaca_gen_pending)
            if alpaca_gen_pending:
                tmp = _write_temp_models(alpaca_gen_pending, "AlpacaEval generation")
                temp_files.append(tmp)
                if args.submit:
                    alpaca_gen_job_ids = _run_capturing(
                        "AlpacaEval generation (automate_alpaca_eval_alpha)",
                        VENV_ALPACA,
                        SCRIPTS / "automate_alpaca_eval_alpha.py",
                        ["--models-file", tmp, "--submit"],
                    )
                else:
                    run(
                        "AlpacaEval generation (automate_alpaca_eval_alpha)",
                        VENV_ALPACA,
                        SCRIPTS / "automate_alpaca_eval_alpha.py",
                        ["--models-file", tmp] + auto_flag,
                    )
            else:
                print("  -> All models already done for AlpacaEval generation, skipping.")

        # 2b. AlpacaEval judgment — only for models missing judgment, depends on new gen jobs
        if "alpaca_eval_judgment" in selected_tasks:
            submit_filtered(
                "AlpacaEval judgment",
                _is_alpaca_eval_judgment_completed,
                lambda tmp: run(
                    "AlpacaEval judgment (automate_alpaca_eval_judgment_alpha)",
                    VENV_ALPACA,
                    SCRIPTS / "automate_alpaca_eval_judgment_alpha.py",
                    ["--models-file", tmp,
                     "--judge-model", args.judge_model,
                     "--tensor-parallel-size", judge_tp]
                    + auto_flag
                    + (["--dependency", _dependency_str(alpaca_gen_job_ids)] if alpaca_gen_job_ids else []),
                ),
            )

        # 3. MT-Bench / JudgeArena
        if "mtbench" in selected_tasks:
            mtbench_flag = (["--rerun-all"] if args.rerun else ["--skip-existing"]) + auto_flag
            submit_filtered(
                "MT-Bench / JudgeArena",
                lambda m: _is_mtbench_completed(m, api_config),
                lambda tmp: run(
                    "MT-Bench / JudgeArena (automate_mtbench_alpha)",
                    VENV_OPENJURY,
                    WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench_alpha.py",
                    ["--models-file", tmp, "--baseline-model", args.baseline,
                     "--judge-server-tp", judge_tp] + mtbench_flag,
                ),
            )

        # 4. ELO estimation / OpenJury
        if "elo" in selected_tasks:
            submit_filtered(
                "ELO estimation",
                lambda m: _is_elo_completed(m, api_config),
                lambda tmp: run(
                    "ELO estimation (automate_elo_estimation_alpha)",
                    VENV_OPENJURY,
                    WORKSPACE / "OpenJury" / "scripts" / "automate_elo_estimation_alpha.py",
                    ["--models-file", tmp, "--judge-tp-size", judge_tp] + auto_flag,
                ),
            )

        # 5. Static LM-eval tasks
        for task in STATIC_TASKS:
            if task not in selected_tasks:
                continue
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
        # 1. Arena-Hard generation → capture job IDs when submitting
        arena_gen_job_ids: list[str] = []
        if "arena_hard_generation" in selected_tasks:
            if args.submit:
                arena_gen_job_ids = _run_capturing(
                    "Arena-Hard generation (automate_arena_hard_generation_alpha)",
                    VENV_ARENA,
                    SCRIPTS / "automate_arena_hard_generation_alpha.py",
                    ["--models-file", models_file, "--submit"],
                )
            else:
                run(
                    "Arena-Hard generation (automate_arena_hard_generation_alpha)",
                    VENV_ARENA,
                    SCRIPTS / "automate_arena_hard_generation_alpha.py",
                    ["--models-file", models_file] + auto_flag,
                )

        # 1b. Arena-Hard judgment — depends on generation jobs
        if "arena_hard_judgment" in selected_tasks:
            arena_judg_extra = ["--dependency", _dependency_str(arena_gen_job_ids)] if arena_gen_job_ids else []
            run(
                "Arena-Hard judgment (automate_arena_hard_judgment_alpha)",
                VENV_ARENA,
                SCRIPTS / "automate_arena_hard_judgment_alpha.py",
                ["--models-file", models_file,
                 "--baseline", args.baseline,
                 "--judge-model", args.judge_model,
                 "--tensor-parallel-size", judge_tp] + auto_flag + arena_judg_extra,
            )

        # 2. AlpacaEval generation → capture job IDs when submitting
        alpaca_gen_job_ids: list[str] = []
        if "alpaca_eval_generation" in selected_tasks:
            if args.submit:
                alpaca_gen_job_ids = _run_capturing(
                    "AlpacaEval generation (automate_alpaca_eval_alpha)",
                    VENV_ALPACA,
                    SCRIPTS / "automate_alpaca_eval_alpha.py",
                    ["--models-file", models_file, "--submit"],
                )
            else:
                run(
                    "AlpacaEval generation (automate_alpaca_eval_alpha)",
                    VENV_ALPACA,
                    SCRIPTS / "automate_alpaca_eval_alpha.py",
                    ["--models-file", models_file] + auto_flag,
                )

        # 2b. AlpacaEval judgment — depends on generation jobs
        if "alpaca_eval_judgment" in selected_tasks:
            alpaca_judg_extra = ["--dependency", _dependency_str(alpaca_gen_job_ids)] if alpaca_gen_job_ids else []
            run(
                "AlpacaEval judgment (automate_alpaca_eval_judgment_alpha)",
                VENV_ALPACA,
                SCRIPTS / "automate_alpaca_eval_judgment_alpha.py",
                ["--models-file", models_file,
                 "--judge-model", args.judge_model,
                 "--tensor-parallel-size", judge_tp] + auto_flag + alpaca_judg_extra,
            )

        # 3. MT-Bench / JudgeArena
        if "mtbench" in selected_tasks:
            if args.rerun:
                run(
                    "MT-Bench / JudgeArena (automate_mtbench_alpha)",
                    VENV_OPENJURY,
                    WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench_alpha.py",
                    ["--models-file", models_file, "--baseline-model", args.baseline,
                     "--judge-server-tp", judge_tp, "--rerun-all"] + auto_flag,
                )
            else:
                run(
                    "MT-Bench / JudgeArena (automate_mtbench_alpha)",
                    VENV_OPENJURY,
                    WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench_alpha.py",
                    ["--models-file", models_file, "--baseline-model", args.baseline,
                     "--judge-server-tp", judge_tp, "--skip-existing"] + auto_flag,
                )

        # 4. ELO estimation / OpenJury
        if "elo" in selected_tasks:
            run(
                "ELO estimation (automate_elo_estimation_alpha)",
                VENV_OPENJURY,
                WORKSPACE / "OpenJury" / "scripts" / "automate_elo_estimation_alpha.py",
                ["--models-file", models_file, "--judge-tp-size", judge_tp] + auto_flag,
            )

        # 5. Static evals
        for task in STATIC_TASKS:
            if task not in selected_tasks:
                continue
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
