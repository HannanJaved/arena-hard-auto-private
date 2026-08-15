#!/usr/bin/env python3
"""
submit_core_evals.py — submit the core 13-task LM-eval suite for a list of models.

Task set and few-shot counts follow scripts/tasks.txt:
    COPA, OpenBookQA, Lambada-OpenAI, Winogrande, Social IQA (0-shot)
    MMLU, MMLU-cont. (5-shot)
    CommonsenseQA, PIQA, ARC-Challenge, ARC-Easy, HellaSwag, BoolQ (10-shot)

Usage:
    python submit_core_evals.py --models-file models.txt [--submit] [--dry-run]
    python submit_core_evals.py --models-file models.txt --skip-completed [--submit]
    python submit_core_evals.py --models-file models.txt --tasks mmlu,boolq [--submit]

Venv assignment:
    venv-lm-eval → all core tasks via submit_lmeval_task_from_list.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKSPACE = Path("/data/horse/ws/hama901h-BFTranslation")
SCRIPTS = WORKSPACE / "arena-hard-auto" / "scripts"

VENV_LMEVAL = WORKSPACE / "venv-lm-eval" / "bin" / "python"
SUBMIT_SCRIPT = SCRIPTS / "submit_lmeval_task_from_list.py"

LM_EVAL_LOG_DIR = WORKSPACE / "logs" / "LM-eval"
OUTPUT_ROOT = WORKSPACE / "evaluation_results" / "core_evals"
API_CONFIG_PATH = WORKSPACE / "arena-hard-auto" / "config" / "api_config.yaml"


@dataclass(frozen=True)
class CoreTask:
    """One core-suite evaluation configuration."""

    task_id: str
    num_fewshot: int
    label: str
    time: str = "01:00:00"
    batch_size: int = 16


# Core 13-task suite with lm_eval task names and few-shot settings from scripts/tasks.txt.
CORE_TASKS: list[CoreTask] = [
    CoreTask("copa", 0, "COPA"),
    CoreTask("openbookqa", 0, "OpenBookQA"),
    CoreTask("lambada_openai", 0, "LAMBADA (OpenAI)"),
    CoreTask("winogrande", 0, "Winogrande"),
    CoreTask("social_iqa", 0, "Social IQA"),
    CoreTask("mmlu_continuation", 5, "MMLU (continuation)", time="02:00:00", batch_size=8),
    CoreTask("mmlu", 5, "MMLU", time="02:00:00", batch_size=8),
    CoreTask("commonsense_qa", 10, "CommonsenseQA", batch_size=4),
    CoreTask("piqa", 10, "PIQA", batch_size=4),
    CoreTask("arc_challenge", 10, "ARC Challenge", batch_size=4),
    CoreTask("arc_easy", 10, "ARC Easy", batch_size=4),
    CoreTask("hellaswag", 10, "HellaSwag", batch_size=4),
    CoreTask("boolq", 10, "BoolQ", batch_size=4),
]

CORE_TASK_IDS = {task.task_id for task in CORE_TASKS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(label: str, extra_args: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    cmd = [str(VENV_LMEVAL), str(SUBMIT_SCRIPT)] + extra_args
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
    valid = []
    for model_name in models:
        entry = api_config.get(model_name)
        if not isinstance(entry, dict):
            valid.append(model_name)
            continue
        path = entry.get("model")
        if path and path.startswith("/") and not Path(path).exists():
            print(f"  SKIP (missing checkpoint): {model_name}\n    path: {path}", file=sys.stderr)
        else:
            valid.append(model_name)
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


def _print_completion_summary(label: str, completed: list[str], pending: list[str]) -> None:
    print(f"\n  [{label}]  {len(completed)} completed, {len(pending)} pending")
    for model_name in completed:
        print(f"    SKIP  {model_name}")
    for model_name in pending:
        print(f"    RUN   {model_name}")


def _lmeval_sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:128]


def _job_name_prefix(task: CoreTask) -> str:
    return f"core_{task.task_id}_"


def _output_dir(task: CoreTask) -> str:
    return str(OUTPUT_ROOT / task.task_id)


def _lm_eval_model_subdir(model_path: str) -> str:
    return model_path.replace("/", "__")


def _has_results_json(task: CoreTask, model_path: str) -> bool:
    subdir = OUTPUT_ROOT / task.task_id / _lm_eval_model_subdir(model_path)
    if not subdir.is_dir():
        return False
    return any(subdir.glob("results_*.json"))


def _log_completion_status(task: CoreTask, model_name: str) -> bool | None:
    """Check this model+task's own Slurm logs for success/failure.

    Returns True/False if at least one matching log exists, or None if there's no
    log at all (e.g. results predate current log retention).
    """
    prefix = f"{_job_name_prefix(task)}{_lmeval_sanitize(model_name)}_"
    log_files = list(LM_EVAL_LOG_DIR.glob(f"{prefix}*.out"))
    if not log_files:
        return None

    for log_file in log_files:
        try:
            text = log_file.read_text(errors="replace")
            if "END TIME:" not in text:
                continue
            err_file = log_file.with_suffix(".err")
            if err_file.exists():
                err_text = err_file.read_text(errors="replace")
                if any(
                    marker in err_text
                    for marker in (
                        "Traceback (most recent call last)",
                        "OutOfMemoryError",
                        "RuntimeError",
                        "AssertionError",
                        "TypeError",
                    )
                ):
                    continue
            return True
        except OSError:
            pass
    return False


def _is_lmeval_completed(
    task: CoreTask,
    model_name: str,
    api_config: dict | None = None,
) -> bool:
    # A model's own Slurm log is authoritative when present: a results_*.json living
    # under the model's checkpoint path can be stale or (if two model names briefly
    # pointed at the same path in api_config.yaml) belong to a *different* model's run,
    # so it must not override a log that shows this model's own job actually failed.
    log_status = _log_completion_status(task, model_name)
    if log_status is not None:
        return log_status

    if api_config is not None:
        entry = api_config.get(model_name)
        if isinstance(entry, dict):
            model_path = entry.get("model")
            if model_path and _has_results_json(task, model_path):
                return True

    return False


def _partition(models: list[str], check_fn) -> tuple[list[str], list[str]]:
    pending, completed = [], []
    for model_name in models:
        (completed if check_fn(model_name) else pending).append(model_name)
    return pending, completed


def _resolve_tasks(task_filter: str | None) -> list[CoreTask]:
    if not task_filter:
        return list(CORE_TASKS)

    selected: list[CoreTask] = []
    unknown: list[str] = []
    for task_id in task_filter.split(","):
        task_id = task_id.strip()
        if not task_id:
            continue
        match = next((task for task in CORE_TASKS if task.task_id == task_id), None)
        if match is None:
            unknown.append(task_id)
        else:
            selected.append(match)

    if unknown:
        print("Unknown task IDs:", ", ".join(unknown), file=sys.stderr)
        print("Valid task IDs:", ", ".join(sorted(CORE_TASK_IDS)), file=sys.stderr)
        sys.exit(2)

    return selected


CPU_PARTITIONS = {"romeo", "barnard"}
GPU_PARTITIONS = {"alpha", "capella"}
DEFAULT_CPU_PARTITION = "barnard"
DEFAULT_GPU_PARTITION = "capella"

# romeo: 128 cores/node, ~505GB/node (~3.9GB/core). barnard: 104 cores/node, ~500GB/node (~4.8GB/core).
# 32 cores is a moderate slice of either shared node; mem scales proportionally with headroom.
DEFAULT_CPU_CPUS_PER_TASK = 32
DEFAULT_CPU_MEM = "128G"

# CPU inference has no GPU acceleration; the per-task times above are tuned for GPU runs.
CPU_TIME_MULTIPLIER = 8


def _scale_time(time_str: str, multiplier: int) -> str:
    hours, minutes, seconds = (int(part) for part in time_str.split(":"))
    total_seconds = (hours * 3600 + minutes * 60 + seconds) * multiplier
    hh, remainder = divmod(total_seconds, 3600)
    mm, ss = divmod(remainder, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _resolve_partition(cpu_only: bool, partition: str | None) -> tuple[str | None, bool]:
    """Returns (partition_to_pass, is_cpu)."""
    if partition:
        if partition not in CPU_PARTITIONS and partition not in GPU_PARTITIONS:
            print(
                f"  WARNING: unrecognized partition '{partition}' "
                f"(known GPU: {', '.join(sorted(GPU_PARTITIONS))}; known CPU: {', '.join(sorted(CPU_PARTITIONS))})",
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

    return DEFAULT_GPU_PARTITION, False


def _submit_task(
    task: CoreTask,
    models_file: str,
    *,
    dry_run: bool,
    submit: bool,
    partition: str | None = None,
    is_cpu: bool = False,
    cpus_per_task: int | None = None,
    mem: str | None = None,
    time_override: str | None = None,
    use_module_torch: bool = False,
    trust_remote_code: bool = False,
) -> None:
    if time_override:
        time_value = time_override
    elif is_cpu:
        time_value = _scale_time(task.time, CPU_TIME_MULTIPLIER)
    else:
        time_value = task.time

    if is_cpu and cpus_per_task is None:
        cpus_per_task = DEFAULT_CPU_CPUS_PER_TASK
    if is_cpu and mem is None:
        mem = DEFAULT_CPU_MEM

    label = f"Core eval: {task.label}"
    extra_args = [
        "--models-file", models_file,
        "--task", task.task_id,
        "--num-fewshot", str(task.num_fewshot),
        "--batch-size", str(task.batch_size),
        "--job-name-prefix", _job_name_prefix(task),
        "--output-dir", _output_dir(task),
        "--time", time_value,
    ]
    if partition:
        extra_args += ["--partition", partition]
    if is_cpu:
        extra_args += ["--gres", "none"]
    if cpus_per_task is not None:
        extra_args += ["--cpus-per-task", str(cpus_per_task)]
    if mem is not None:
        extra_args += ["--mem", mem]
    if use_module_torch:
        extra_args.append("--use-module-torch")
    if trust_remote_code:
        extra_args.append("--trust-remote-code")
    if dry_run or not submit:
        extra_args.append("--dry-run")
    run(label, extra_args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the core 13-task LM-eval suite for a model list.",
    )
    parser.add_argument("--models-file", required=True, help="Text file with one model name per line.")
    parser.add_argument("--submit", action="store_true", help="Submit SLURM jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit.")
    parser.add_argument(
        "--tasks",
        help="Comma-separated subset of core task IDs (default: all). Valid: " + ", ".join(sorted(CORE_TASK_IDS)),
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip models that already have results JSON in evaluation_results/core_evals "
            "or a successful SLURM log (END TIME without traceback/OOM in .err)."
        ),
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help=f"Submit on a CPU-only partition (default '{DEFAULT_CPU_PARTITION}') instead of the default GPU partition.",
    )
    parser.add_argument(
        "--partition",
        help=(
            "Slurm partition to submit on. "
            f"GPU: {', '.join(sorted(GPU_PARTITIONS))}. CPU: {', '.join(sorted(CPU_PARTITIONS))}. "
            f"Defaults to '{DEFAULT_GPU_PARTITION}', or '{DEFAULT_CPU_PARTITION}' if --cpu-only is set."
        ),
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        help=(
            f"Slurm CPUs per task (default: {DEFAULT_CPU_CPUS_PER_TASK} for --cpu-only/CPU-partition jobs, "
            "else the submit script's default of 4)."
        ),
    )
    parser.add_argument(
        "--mem",
        help=f"Slurm memory request, e.g. '64G' (default: {DEFAULT_CPU_MEM} for CPU jobs, else 16G).",
    )
    parser.add_argument(
        "--time",
        help=(
            "Override Slurm wall time for all tasks (HH:MM:SS), instead of each task's built-in default. "
            f"For CPU jobs without this set, each task's default time is scaled by {CPU_TIME_MULTIPLIER}x."
        ),
    )
    parser.add_argument(
        "--use-module-torch",
        action="store_true",
        help=(
            "Use the cluster's module-provided PyTorch/2.3.0 (via venv-lm-eval-module) instead of the "
            "pip-installed torch in venv-lm-eval, whose large .so files suffer catastrophic Lustre read latency."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True in lm_eval model_args, for models with custom modeling code.",
    )
    args = parser.parse_args()

    tasks = _resolve_tasks(args.tasks)
    models_file = str(args.models_file)
    partition, is_cpu = _resolve_partition(args.cpu_only, args.partition)

    if args.skip_completed:
        print(f"\n{'='*60}")
        print("  Checking completed jobs ...")
        print(f"{'='*60}")

        all_models = _read_models(models_file)
        api_config = _load_api_config()
        all_models = _filter_missing_paths(all_models, api_config)
        temp_files: list[str] = []

        for task in tasks:
            pending, completed = _partition(
                all_models,
                lambda model_name, current_task=task: _is_lmeval_completed(
                    current_task, model_name, api_config
                ),
            )
            _print_completion_summary(task.label, completed, pending)
            if not pending:
                print(f"  -> All models already done for {task.label}, skipping.")
                continue

            tmp = _write_temp_models(pending)
            temp_files.append(tmp)
            _submit_task(
                task, tmp, dry_run=args.dry_run, submit=args.submit, partition=partition, is_cpu=is_cpu,
                cpus_per_task=args.cpus_per_task, mem=args.mem, time_override=args.time,
                use_module_torch=args.use_module_torch, trust_remote_code=args.trust_remote_code,
            )

        for temp_path in temp_files:
            Path(temp_path).unlink(missing_ok=True)
    else:
        for task in tasks:
            _submit_task(
                task, models_file, dry_run=args.dry_run, submit=args.submit, partition=partition, is_cpu=is_cpu,
                cpus_per_task=args.cpus_per_task, mem=args.mem, time_override=args.time,
                use_module_torch=args.use_module_torch, trust_remote_code=args.trust_remote_code,
            )

    print(f"\n{'='*60}")
    print("Core Evaluation Jobs Submitted.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
