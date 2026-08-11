#!/usr/bin/env python3
from __future__ import annotations
"""
submit_evals.py — submit all evaluation jobs for a list of models.

Usage:
    python submit_evals.py --models-file models.txt --baseline <model> [--submit] [--dry-run]
    python submit_evals.py --models-file models.txt --baseline <model> --skip-completed [--submit]
    python submit_evals.py --models-file models.txt --baseline <model> --evals dynamic [--submit]
    python submit_evals.py --models-file models.txt --baseline <model> --evals static \\
        --partition romeo --cpu-only --skip-completed [--submit]
    python submit_evals.py --models-file models.txt --baseline <model> --tasks mtbench arena-hard [--submit]

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
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKSPACE = Path("/data/horse/ws/hama901h-BFTranslation")
SCRIPTS = WORKSPACE / "arena-hard-auto" / "scripts"

VENV_ARENA    = WORKSPACE / "arena-hard-auto" / "venv" / "bin" / "python"
VENV_ALPACA   = WORKSPACE / "venv-alpacaeval" / "bin" / "python"
VENV_OPENJURY = WORKSPACE / "venv-openjury" / "bin" / "python"
VENV_LMEVAL   = WORKSPACE / "venv-lm-eval" / "bin" / "python"

STATIC_TASKS = ["arc_challenge", "gpqa", "gsm8k", "hellaswag", "ifeval", "piqa", "truthfulqa"]

CPU_PARTITIONS = {"romeo", "barnard"}
GPU_PARTITIONS = {"alpha", "capella"}
DEFAULT_CPU_PARTITION = "romeo"
DEFAULT_GPU_PARTITION = "capella"
DEFAULT_CPU_CPUS_PER_TASK = 32
DEFAULT_CPU_MEM = "128G"
CPU_TIME_MULTIPLIER = 10


@dataclass(frozen=True)
class StaticTaskSpec:
    """Config for routing a static task through submit_lmeval_task_from_list.py (CPU path)."""

    key: str                 # STATIC_TASKS id / skip-completed log prefix
    lm_eval_task: str        # lm_eval --tasks name
    num_fewshot: int
    batch_size: int
    time: str
    output_subdir: str


# Defaults mirror submit_{task}_from_list.py single-GPU settings.
STATIC_TASK_SPECS: dict[str, StaticTaskSpec] = {
    "arc_challenge": StaticTaskSpec("arc_challenge", "arc_challenge", 25, 8, "01:00:00", "arc_challenge"),
    "gpqa": StaticTaskSpec("gpqa", "gpqa_diamond_zeroshot", 0, 16, "01:00:00", "gpqa"),
    "gsm8k": StaticTaskSpec("gsm8k", "gsm8k", 5, 64, "03:00:00", "gsm8k"),
    "hellaswag": StaticTaskSpec("hellaswag", "hellaswag", 10, 16, "03:00:00", "hellaswag"),
    "ifeval": StaticTaskSpec("ifeval", "ifeval", 0, 16, "04:00:00", "ifeval"),
    "piqa": StaticTaskSpec("piqa", "piqa", 0, 32, "01:00:00", "piqa"),
    "truthfulqa": StaticTaskSpec("truthfulqa", "truthfulqa_mc2", 0, 32, "01:00:00", "truthfulqa"),
}

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
            alias_paths = [Path(p) for p in _model_path_aliases(path)]
            existing_ckpt = next((ckpt for ckpt in alias_paths if ckpt.exists()), None)
            if existing_ckpt is None:
                print(f"  SKIP (missing checkpoint): {m}\n    path: {path}", file=sys.stderr)
                continue
            # Intermediate DPO/SFT dirs exist before the final HF export; vLLM needs config.json.
            if not (existing_ckpt / "config.json").exists():
                print(
                    f"  SKIP (incomplete checkpoint, no config.json yet): {m}\n    path: {existing_ckpt}",
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


def _model_path_aliases(path: str | None) -> list[str]:
    if not path:
        return []
    aliases = [path]
    for old, new in [("Qwen3-114B", "Qwen3-14B"), ("Qwen3-14B", "Qwen3-114B")]:
        if old in path:
            aliases.append(path.replace(old, new))
    deduped = []
    seen = set()
    for item in aliases:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


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
    vllm_models = {f"VLLM/{candidate}" for candidate in _model_path_aliases(path)}
    for results_file in MTBENCH_RESULT_DIR.rglob("results-*.json"):
        try:
            data = json.loads(results_file.read_text())
            if data.get("model_A") in vllm_models:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


def _is_elo_completed(model_name: str, api_config: dict) -> bool:
    path = _model_path(api_config, model_name)
    if not path or not ELO_RESULT_DIR.exists():
        return False
    vllm_models = {f"VLLM/{candidate}" for candidate in _model_path_aliases(path)}
    for summary_file in ELO_RESULT_DIR.rglob("summary.json"):
        try:
            data = json.loads(summary_file.read_text())
            if data.get("model") in vllm_models:
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


def _scale_time(time_str: str, multiplier: int) -> str:
    hours, minutes, seconds = (int(part) for part in time_str.split(":"))
    total_seconds = (hours * 3600 + minutes * 60 + seconds) * multiplier
    hh, remainder = divmod(total_seconds, 3600)
    mm, ss = divmod(remainder, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _resolve_partition(cpu_only: bool, partition: str | None) -> tuple[str | None, bool]:
    """Returns (partition_to_pass, is_cpu). None partition means leave from_list defaults."""
    if partition:
        if partition not in CPU_PARTITIONS and partition not in GPU_PARTITIONS:
            print(
                f"  WARNING: unrecognized partition '{partition}' "
                f"(known GPU: {', '.join(sorted(GPU_PARTITIONS))}; "
                f"known CPU: {', '.join(sorted(CPU_PARTITIONS))})",
                file=sys.stderr,
            )
        is_cpu = partition in CPU_PARTITIONS
        if cpu_only and partition in GPU_PARTITIONS:
            print(
                f"ERROR: --cpu-only conflicts with GPU partition '{partition}'.",
                file=sys.stderr,
            )
            sys.exit(2)
        return partition, is_cpu

    if cpu_only:
        return DEFAULT_CPU_PARTITION, True

    return None, False


def _static_submit_args(
    task: str,
    models_file: str,
    *,
    is_cpu: bool,
    partition: str | None,
    cpus_per_task: int | None,
    mem: str | None,
    time_override: str | None,
    dry_run: bool,
) -> tuple[Path, list[str]]:
    """Build (script, args) for one static LM-eval task."""
    dry = ["--dry-run"] if dry_run else []

    if is_cpu:
        # Dedicated from_list scripts are GPU-oriented (CUDA module / nvidia-smi).
        # Route CPU jobs through submit_lmeval_task_from_list.py which supports --device cpu.
        spec = STATIC_TASK_SPECS[task]
        time_value = time_override or _scale_time(spec.time, CPU_TIME_MULTIPLIER)
        cpus = cpus_per_task if cpus_per_task is not None else DEFAULT_CPU_CPUS_PER_TASK
        memory = mem if mem is not None else DEFAULT_CPU_MEM
        args = [
            "--models-file", models_file,
            "--task", spec.lm_eval_task,
            "--num-fewshot", str(spec.num_fewshot),
            "--batch-size", str(spec.batch_size),
            "--job-name-prefix", f"{spec.key}_",
            "--output-dir", str(WORKSPACE / "evaluation_results" / spec.output_subdir),
            "--partition", partition or DEFAULT_CPU_PARTITION,
            "--gres", "none",
            "--cpus-per-task", str(cpus),
            "--mem", memory,
            "--time", time_value,
        ] + dry
        return SCRIPTS / "submit_lmeval_task_from_list.py", args

    args = ["--models-file", models_file] + dry
    if partition:
        args += ["--partition", partition]
    if cpus_per_task is not None:
        args += ["--cpus-per-task", str(cpus_per_task)]
    if mem is not None:
        args += ["--mem", mem]
    if time_override is not None:
        args += ["--time", time_override]
    return SCRIPTS / f"submit_{task}_from_list.py", args


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
    parser = argparse.ArgumentParser(description="Submit all evaluation jobs for a model list.")
    parser.add_argument("--models-file", required=True, help="Text file with one model name per line.")
    parser.add_argument("--submit",  action="store_true", help="Submit SLURM jobs (automation scripts).")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit.")
    parser.add_argument("--baseline", required=True,
                        help="Baseline model: a legacy alias (instruct, base, ...) or any api_config.yaml model key.")
    parser.add_argument(
        "--judge-model", default="Qwen3-Next-80B-A3B-Instruct-FP8",
        help="Judge model name used for arena-hard, alpaca-eval, and other judge-based evals (default: Qwen3-Next-80B-A3B-Instruct-FP8).",
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
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help=(
            f"Run static LM-eval tasks on a CPU partition (default '{DEFAULT_CPU_PARTITION}'). "
            "Uses submit_lmeval_task_from_list.py with --device cpu. "
            "Dynamic/judge tasks still require GPUs and are rejected with --cpu-only."
        ),
    )
    parser.add_argument(
        "--partition",
        help=(
            "Slurm partition for static LM-eval tasks. "
            f"GPU: {', '.join(sorted(GPU_PARTITIONS))}. CPU: {', '.join(sorted(CPU_PARTITIONS))}. "
            f"Defaults to each from_list script's default, or '{DEFAULT_CPU_PARTITION}' with --cpu-only."
        ),
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        help=(
            f"Slurm CPUs per task for static LM-eval "
            f"(default: {DEFAULT_CPU_CPUS_PER_TASK} for CPU jobs, else from_list script default)."
        ),
    )
    parser.add_argument(
        "--mem",
        help=f"Slurm memory for static LM-eval (default: {DEFAULT_CPU_MEM} for CPU jobs, else from_list default).",
    )
    parser.add_argument(
        "--time",
        help=(
            "Override Slurm wall time for all static LM-eval tasks (HH:MM:SS). "
            f"For CPU jobs without this set, each task's default time is scaled by {CPU_TIME_MULTIPLIER}x."
        ),
    )
    args = parser.parse_args()
    selected_tasks = _resolve_tasks(args.tasks, args.evals)
    partition, is_cpu = _resolve_partition(args.cpu_only, args.partition)

    if is_cpu:
        dynamic_selected = sorted(t for t in selected_tasks if t in AUTOMATION_TASKS)
        if dynamic_selected:
            print(
                "ERROR: --cpu-only / CPU partition cannot run dynamic/judge tasks: "
                + ", ".join(dynamic_selected)
                + ". Use --evals static (or --tasks with only LM-eval tasks).",
                file=sys.stderr,
            )
            sys.exit(2)

    models_file = str(args.models_file)

    # For the automation scripts (arena-hard, alpaca-eval, mtbench, elo):
    #   --submit  → pass --submit
    #   --dry-run → pass --dry-run
    #   neither   → generate only (no flag)
    auto_flag = ["--submit"] if args.submit else (["--dry-run"] if args.dry_run else [])

    # Static scripts submit by default; --dry-run (or omitting --submit) suppresses submission.
    static_dry_run = args.dry_run or not args.submit

    def run_static(task: str, models_path: str) -> None:
        script, extra = _static_submit_args(
            task,
            models_path,
            is_cpu=is_cpu,
            partition=partition,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time_override=args.time,
            dry_run=static_dry_run,
        )
        run(f"Static eval: {task}", VENV_LMEVAL, script, extra)

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
                        "Arena-Hard generation (automate_arena_hard_generation_olmo3)",
                        VENV_ARENA,
                        SCRIPTS / "automate_arena_hard_generation_olmo3.py",
                        ["--models-file", tmp, "--submit"],
                    )
                else:
                    run(
                        "Arena-Hard generation (automate_arena_hard_generation_olmo3)",
                        VENV_ARENA,
                        SCRIPTS / "automate_arena_hard_generation_olmo3.py",
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
                    "Arena-Hard judgment (automate_arena_hard_judgment_olmo3)",
                    VENV_ARENA,
                    SCRIPTS / "automate_arena_hard_judgment_olmo3.py",
                    ["--models-file", tmp,
                     "--baseline", args.baseline,
                     "--judge-model", args.judge_model]
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
                        "AlpacaEval generation (automate_alpaca_eval)",
                        VENV_ALPACA,
                        SCRIPTS / "automate_alpaca_eval.py",
                        ["--models-file", tmp, "--submit"],
                    )
                else:
                    run(
                        "AlpacaEval generation (automate_alpaca_eval)",
                        VENV_ALPACA,
                        SCRIPTS / "automate_alpaca_eval.py",
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
                    "AlpacaEval judgment (automate_alpaca_eval_judgment)",
                    VENV_ALPACA,
                    SCRIPTS / "automate_alpaca_eval_judgment.py",
                    ["--models-file", tmp,
                     "--judge-model", args.judge_model]
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
                    "MT-Bench / JudgeArena (automate_mtbench)",
                    VENV_OPENJURY,
                    WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench.py",
                    ["--models-file", tmp, "--baseline-model", args.baseline] + mtbench_flag,
                ),
            )

        # 4. ELO estimation / OpenJury
        if "elo" in selected_tasks:
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
            if task not in selected_tasks:
                continue
            submit_filtered(
                task,
                lambda m, t=task: _is_lmeval_completed(t, m),
                lambda tmp, t=task: run_static(t, tmp),
            )

        for f in temp_files:
            Path(f).unlink(missing_ok=True)

    else:
        # 1. Arena-Hard generation → capture job IDs when submitting
        arena_gen_job_ids: list[str] = []
        if "arena_hard_generation" in selected_tasks:
            if args.submit:
                arena_gen_job_ids = _run_capturing(
                    "Arena-Hard generation (automate_arena_hard_generation_olmo3)",
                    VENV_ARENA,
                    SCRIPTS / "automate_arena_hard_generation_olmo3.py",
                    ["--models-file", models_file, "--submit"],
                )
            else:
                run(
                    "Arena-Hard generation (automate_arena_hard_generation_olmo3)",
                    VENV_ARENA,
                    SCRIPTS / "automate_arena_hard_generation_olmo3.py",
                    ["--models-file", models_file] + auto_flag,
                )

        # 1b. Arena-Hard judgment — depends on generation jobs
        if "arena_hard_judgment" in selected_tasks:
            arena_judg_extra = ["--dependency", _dependency_str(arena_gen_job_ids)] if arena_gen_job_ids else []
            run(
                "Arena-Hard judgment (automate_arena_hard_judgment_olmo3)",
                VENV_ARENA,
                SCRIPTS / "automate_arena_hard_judgment_olmo3.py",
                ["--models-file", models_file,
                 "--baseline", args.baseline,
                 "--judge-model", args.judge_model] + auto_flag + arena_judg_extra,
            )

        # 2. AlpacaEval generation → capture job IDs when submitting
        alpaca_gen_job_ids: list[str] = []
        if "alpaca_eval_generation" in selected_tasks:
            if args.submit:
                alpaca_gen_job_ids = _run_capturing(
                    "AlpacaEval generation (automate_alpaca_eval)",
                    VENV_ALPACA,
                    SCRIPTS / "automate_alpaca_eval.py",
                    ["--models-file", models_file, "--submit"],
                )
            else:
                run(
                    "AlpacaEval generation (automate_alpaca_eval)",
                    VENV_ALPACA,
                    SCRIPTS / "automate_alpaca_eval.py",
                    ["--models-file", models_file] + auto_flag,
                )

        # 2b. AlpacaEval judgment — depends on generation jobs
        if "alpaca_eval_judgment" in selected_tasks:
            alpaca_judg_extra = ["--dependency", _dependency_str(alpaca_gen_job_ids)] if alpaca_gen_job_ids else []
            run(
                "AlpacaEval judgment (automate_alpaca_eval_judgment)",
                VENV_ALPACA,
                SCRIPTS / "automate_alpaca_eval_judgment.py",
                ["--models-file", models_file,
                 "--judge-model", args.judge_model] + auto_flag + alpaca_judg_extra,
            )

        # 3. MT-Bench / JudgeArena
        if "mtbench" in selected_tasks:
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
        if "elo" in selected_tasks:
            run(
                "ELO estimation (automate_elo_estimation)",
                VENV_OPENJURY,
                WORKSPACE / "OpenJury" / "scripts" / "automate_elo_estimation.py",
                ["--models-file", models_file] + auto_flag,
            )

        # 5. Static evals
        for task in STATIC_TASKS:
            if task not in selected_tasks:
                continue
            run_static(task, models_file)

    print(f"\n{'='*60}")
    print("Evaluation Jobs Submitted.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
