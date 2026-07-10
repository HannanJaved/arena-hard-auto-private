#!/usr/bin/env python3
"""
submit_dclm_core_evals.py — submit DCLM-core (22-task) LM-eval jobs for a list of models.

DCLM-core is the 22-task benchmark suite from DataComp-LM used to compute Core
centered accuracy. Task names and few-shot counts follow DCLM's mmlu_and_lowvar.yaml
configuration (excluding MMLU).

Usage:
    python submit_dclm_core_evals.py --models-file models.txt [--submit] [--dry-run]
    python submit_dclm_core_evals.py --models-file models.txt --skip-completed [--submit]
    python submit_dclm_core_evals.py --models-file models.txt --tasks arc_easy,boolq [--submit]

Venv assignment:
    venv-lm-eval → all DCLM-core tasks via submit_lmeval_task_from_list.py
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
OUTPUT_ROOT = WORKSPACE / "evaluation_results" / "dclm_core"
API_CONFIG_PATH = WORKSPACE / "arena-hard-auto" / "config" / "api_config.yaml"


@dataclass(frozen=True)
class DclmCoreTask:
    """One DCLM-core evaluation configuration."""

    task_id: str
    lm_eval_task: str
    num_fewshot: int
    label: str
    time: str = "01:00:00"
    batch_size: int = 16
    note: str | None = None


# DCLM-core tasks with lm_eval mappings and DCLM few-shot settings.
# See: https://github.com/mlfoundations/dclm/blob/main/eval/mmlu_and_lowvar.yaml
DCLM_CORE_TASKS: list[DclmCoreTask] = [
    DclmCoreTask("agi_eval_lsat_ar", "agieval_lsat_ar", 3, "AGI Eval LSAT-AR", batch_size=8),
    DclmCoreTask("arc_easy", "arc_easy", 10, "ARC Easy", batch_size=4),
    DclmCoreTask("arc_challenge", "arc_challenge", 10, "ARC Challenge", batch_size=4),
    DclmCoreTask("bigbench_qa_wikidata", "bigbench_qa_wikidata_generate_until", 10, "Big-Bench: QA Wikidata"),
    DclmCoreTask("bigbench_dyck_languages", "bigbench_dyck_languages_generate_until", 10, "Big-Bench: Dyck Languages"),
    DclmCoreTask("bigbench_operators", "bigbench_operators_generate_until", 10, "Big-Bench: Operators"),
    DclmCoreTask("bigbench_repeat_copy_logic", "bigbench_repeat_copy_logic_generate_until", 10, "Big-Bench: Repeat Copy Logic"),
    DclmCoreTask("bigbench_cs_algorithms", "bigbench_cs_algorithms_generate_until", 10, "Big-Bench: CS Algorithms"),
    DclmCoreTask(
        "bigbench_language_identification",
        "bigbench_language_identification_multiple_choice",
        10,
        "Big-Bench: Language Identification",
        batch_size=4,
    ),
    DclmCoreTask("boolq", "boolq", 10, "BoolQ", batch_size=4),
    DclmCoreTask("commonsense_qa", "commonsense_qa", 10, "CommonsenseQA", batch_size=4),
    DclmCoreTask("copa", "copa", 0, "COPA"),
    DclmCoreTask("coqa", "coqa", 0, "CoQA", time="02:00:00", batch_size=4),
    DclmCoreTask("hellaswag_zeroshot", "hellaswag", 0, "HellaSwag (0-shot)"),
    DclmCoreTask("hellaswag", "hellaswag", 10, "HellaSwag (10-shot)", batch_size=4),
    DclmCoreTask(
        "jeopardy",
        "",
        10,
        "Jeopardy",
        note="No standard lm_eval task; use the DCLM eval harness (eval/mmlu_and_lowvar.yaml) instead.",
    ),
    DclmCoreTask("lambada_openai", "lambada_openai", 0, "LAMBADA"),
    DclmCoreTask("openbook_qa", "openbookqa", 0, "OpenBookQA"),
    DclmCoreTask("piqa", "piqa", 10, "PIQA", batch_size=4),
    DclmCoreTask(
        "squad",
        "squad_completion",
        10,
        "SQuAD",
        batch_size=4,
        note="Uses lm_eval squad_completion as closest available task to DCLM's squad.",
    ),
    DclmCoreTask("winograd", "wsc273", 0, "The Winograd Schema Challenge"),
    DclmCoreTask("winogrande", "winogrande", 0, "The Winogrande"),
]

DCLM_CORE_TASK_IDS = {task.task_id for task in DCLM_CORE_TASKS}


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


def _job_name_prefix(task: DclmCoreTask) -> str:
    return f"dclm_{task.task_id}_"


def _output_dir(task: DclmCoreTask) -> str:
    return str(OUTPUT_ROOT / task.task_id)


def _lm_eval_model_subdir(model_path: str) -> str:
    return model_path.replace("/", "__")


def _has_results_json(task: DclmCoreTask, model_path: str) -> bool:
    subdir = OUTPUT_ROOT / task.task_id / _lm_eval_model_subdir(model_path)
    if not subdir.is_dir():
        return False
    return any(subdir.glob("results_*.json"))


def _is_lmeval_completed(
    task: DclmCoreTask,
    model_name: str,
    api_config: dict | None = None,
) -> bool:
    if api_config is not None:
        entry = api_config.get(model_name)
        if isinstance(entry, dict):
            model_path = entry.get("model")
            if model_path and _has_results_json(task, model_path):
                return True

    prefix = f"{_job_name_prefix(task)}{_lmeval_sanitize(model_name)}_"
    for log_file in LM_EVAL_LOG_DIR.glob(f"{prefix}*.out"):
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


def _partition(models: list[str], check_fn) -> tuple[list[str], list[str]]:
    pending, completed = [], []
    for model_name in models:
        (completed if check_fn(model_name) else pending).append(model_name)
    return pending, completed


def _resolve_tasks(task_filter: str | None) -> list[DclmCoreTask]:
    if not task_filter:
        return list(DCLM_CORE_TASKS)

    selected: list[DclmCoreTask] = []
    unknown: list[str] = []
    for task_id in task_filter.split(","):
        task_id = task_id.strip()
        if not task_id:
            continue
        match = next((task for task in DCLM_CORE_TASKS if task.task_id == task_id), None)
        if match is None:
            unknown.append(task_id)
        else:
            selected.append(match)

    if unknown:
        print("Unknown task IDs:", ", ".join(unknown), file=sys.stderr)
        print("Valid task IDs:", ", ".join(sorted(DCLM_CORE_TASK_IDS)), file=sys.stderr)
        sys.exit(2)

    return selected


def _submit_task(
    task: DclmCoreTask,
    models_file: str,
    *,
    dry_run: bool,
    submit: bool,
) -> None:
    if not task.lm_eval_task:
        print(f"\n  SKIP {task.label}: {task.note}", file=sys.stderr)
        return

    label = f"DCLM-core: {task.label}"
    extra_args = [
        "--models-file", models_file,
        "--task", task.lm_eval_task,
        "--num-fewshot", str(task.num_fewshot),
        "--batch-size", str(task.batch_size),
        "--job-name-prefix", _job_name_prefix(task),
        "--output-dir", _output_dir(task),
        "--time", task.time,
    ]
    if task.note:
        print(f"  NOTE ({task.task_id}): {task.note}")
    if dry_run or not submit:
        extra_args.append("--dry-run")
    run(label, extra_args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit DCLM-core LM-eval jobs for a model list.",
    )
    parser.add_argument("--models-file", required=True, help="Text file with one model name per line.")
    parser.add_argument("--submit", action="store_true", help="Submit SLURM jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit.")
    parser.add_argument(
        "--tasks",
        help="Comma-separated subset of DCLM-core task IDs (default: all).",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip models that already have results JSON in evaluation_results/dclm_core "
            "or a successful SLURM log (END TIME without traceback/OOM in .err)."
        ),
    )
    args = parser.parse_args()

    tasks = _resolve_tasks(args.tasks)
    models_file = str(args.models_file)

    if args.skip_completed:
        print(f"\n{'='*60}")
        print("  Checking completed jobs ...")
        print(f"{'='*60}")

        all_models = _read_models(models_file)
        api_config = _load_api_config()
        all_models = _filter_missing_paths(all_models, api_config)
        temp_files: list[str] = []

        for task in tasks:
            if not task.lm_eval_task:
                print(f"\n  SKIP {task.label}: {task.note}", file=sys.stderr)
                continue

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
            _submit_task(task, tmp, dry_run=args.dry_run, submit=args.submit)

        for temp_path in temp_files:
            Path(temp_path).unlink(missing_ok=True)
    else:
        for task in tasks:
            _submit_task(task, models_file, dry_run=args.dry_run, submit=args.submit)

    print(f"\n{'='*60}")
    print("DCLM-core Evaluation Jobs Submitted.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
