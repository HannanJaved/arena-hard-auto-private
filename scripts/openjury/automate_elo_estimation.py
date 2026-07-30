#!/usr/bin/env python3
"""
Automation script to submit ELO estimation jobs for multiple models using OpenJury.
Generates SLURM scripts that call openjury/estimate_elo_ratings.py with a VLLM provider prefix.

Models and judge are referenced by key in arena-hard-auto/config/api_config.yaml.
The VLLM provider loads models in-process (no separate server needed).
"""

import os
import re
import json
import yaml
import argparse
import subprocess
from pathlib import Path

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
OPENJURY_DIR = f"{WORKSPACE_ROOT}/OpenJury"
API_CONFIG_PATH = f"{WORKSPACE_ROOT}/arena-hard-auto/config/api_config.yaml"
LOGS_DIR = f"{WORKSPACE_ROOT}/logs/openjury-elo"
SCRIPTS_DIR = f"{WORKSPACE_ROOT}/generated_sbatch_jobs"

# Defaults
DEFAULT_JUDGE = "Qwen3-Next-80B-A3B-Instruct-FP8"
DEFAULT_ARENA = "LMArena"
DEFAULT_N_INSTRUCTIONS = 500
DEFAULT_SWAP_MODE = "both"
DEFAULT_RESULT_FOLDER = f"{WORKSPACE_ROOT}/evaluation_results/openjury-elo/"
DEFAULT_PARTITION = "capella"
DEFAULT_TIME = "04:00:00"
DEFAULT_GPUS = 1
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MAX_OUT_TOKENS_MODELS = 4096
DEFAULT_MAX_OUT_TOKENS_JUDGE = 2048
OPENJURY_DATA=f"{WORKSPACE_ROOT}/openjury-data"


def load_api_config():
    with open(API_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_model_path(api_config, model_key):
    """Return the model path from api_config for a given key."""
    entry = api_config.get(model_key)
    if entry is None:
        raise ValueError(f"Model key '{model_key}' not found in api_config.yaml.")
    if isinstance(entry, dict) and "model" in entry:
        return entry["model"]
    raise ValueError(f"No 'model' field found for key '{model_key}' in api_config.yaml.")


def safe_name(name: str) -> str:
    """Sanitize a string for use in filenames and job names."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def create_slurm_script(
    model_key: str,
    model_path: str,
    judge_key: str,
    judge_path: str,
    script_path: str,
    arena: str,
    n_instructions: int,
    n_instructions_per_language: int | None,
    languages: list[str] | None,
    swap_mode: str,
    provide_explanation: bool,
    result_folder: str,
    max_out_tokens_models: int,
    max_out_tokens_judge: int,
    max_model_len: int | None,
    num_gpus: int,
    model_tp_size: int,
    partition: str,
    time_limit: str,
    ignore_cache: bool,
):
    safe_model = safe_name(model_key)
    job_name = f"elo-{safe_model[:40]}"
    log_dir = f"{LOGS_DIR}/{safe_model}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # VLLM provider prefix: "VLLM/<model_path>"
    vllm_model = f"VLLM/{model_path}"
    vllm_judge = f"VLLM/{judge_path}"

    judge_tp_size = 1
    cuda_devices = ",".join(str(i) for i in range(num_gpus))

    # Build optional CLI flags
    optional_flags = []
    if n_instructions is not None:
        optional_flags.append(f"    --n_instructions {n_instructions} \\")
    if n_instructions_per_language is not None:
        optional_flags.append(f"    --n_instructions_per_language {n_instructions_per_language} \\")
    if languages:
        optional_flags.append(f"    --languages {' '.join(languages)} \\")
    if provide_explanation:
        optional_flags.append("    --provide_explanation \\")
    if ignore_cache:
        optional_flags.append("    --ignore_cache \\")
    if max_model_len is not None:
        optional_flags.append(f"    --max_model_len {max_model_len} \\")

    optional_flags_str = "\n".join(optional_flags)
    optional_flags_block = (optional_flags_str + "\n") if optional_flags_str else ""

    # Cap judge concurrency: Qwen3-Next MoE on 1xH100 intermittently hangs at
    # high in-flight batch counts (progress stuck, Triton MoE JIT mid-run).
    model_engine_kwargs = (
        f'{{"tensor_parallel_size": {model_tp_size}, "enforce_eager": true}}'
    )
    judge_engine_kwargs = (
        f'{{"tensor_parallel_size": {judge_tp_size}, "enforce_eager": true, '
        f'"max_num_seqs": 16, "max_num_batched_tokens": 8192}}'
    )

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --error={log_dir}/elo_%j.err
#SBATCH --output={log_dir}/elo_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{num_gpus}

set -e

echo "=== ELO estimation ==="
echo "  Model : {model_key}"
echo "  Judge : {judge_key}"
echo "  Arena : {arena}"
echo "  Started: $(date)"

# --- ENVIRONMENT ---
source {WORKSPACE_ROOT}/venv-openjury/bin/activate
PYTHON_EXEC={WORKSPACE_ROOT}/venv-openjury/bin/python
module load CUDA
source {WORKSPACE_ROOT}/cache.sh

echo "Python: $PYTHON_EXEC"

# Use specified GPUs for in-process vLLM
export CUDA_VISIBLE_DEVICES={cuda_devices}

# Avoid FlashInfer FP8 MoE JIT builds (can fail on some nodes)
export VLLM_USE_FLASHINFER_MOE_FP8=0
# Force Triton MoE backend to avoid nvcc JIT compilation (FLASHINFER_CUTLASS gets OOM-killed)
export VLLM_MOE_FP8_BACKEND=TRITON

# ===================================================================
# Run ELO estimation (vLLM loads models in-process)
# ===================================================================
cd {OPENJURY_DIR}

$PYTHON_EXEC {OPENJURY_DIR}/openjury/estimate_elo_ratings.py \\
    --model "{vllm_model}" \\
    --judge "{vllm_judge}" \\
    --arena {arena} \\
    --swap_mode {swap_mode} \\
    --max_out_tokens_models {max_out_tokens_models} \\
    --max_out_tokens_judge {max_out_tokens_judge} \\
    --result_folder "{result_folder}" \\
{optional_flags_block}    --engine_kwargs '{model_engine_kwargs}' \\
    --judge_engine_kwargs '{judge_engine_kwargs}'

echo "Done: $(date)"
"""

    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)


def find_existing_results(result_folder: str, model_path: str, arena: str) -> list[Path]:
    """Return result dirs whose summary.json matches the given VLLM model path and arena."""
    vllm_model = f"VLLM/{model_path}"
    result_dir = Path(result_folder)
    matches = []
    if not result_dir.exists():
        return matches
    for summary_file in result_dir.rglob("summary.json"):
        try:
            with open(summary_file) as f:
                data = json.load(f)
            if data.get("model") == vllm_model and data.get("arena") == arena:
                matches.append(summary_file.parent)
        except (json.JSONDecodeError, OSError):
            pass
    return matches


def load_models_from_file(file_path):
    models = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    models.append(line)
    except FileNotFoundError:
        print(f"Model list file not found: {file_path}")
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit SLURM jobs for OpenJury ELO estimation."
    )

    # Model / judge selection
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model keys from api_config.yaml to evaluate, or a single .txt file with one key per line.",
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default=f"{WORKSPACE_ROOT}/arena_hard_models_to_test.txt",
        help="File containing model keys to evaluate (default: arena_hard_models_to_test.txt).",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=DEFAULT_JUDGE,
        help=f"Judge model key from api_config.yaml (default: {DEFAULT_JUDGE}).",
    )

    # estimate_elo_ratings.py options
    parser.add_argument(
        "--arena",
        choices=["LMArena", "ComparIA"],
        default=DEFAULT_ARENA,
        help=f"Arena to sample battles from (default: {DEFAULT_ARENA}).",
    )
    parser.add_argument(
        "--n_instructions",
        type=int,
        default=DEFAULT_N_INSTRUCTIONS,
        help=f"Number of battles for LLM-judge evaluation (default: {DEFAULT_N_INSTRUCTIONS}).",
    )
    parser.add_argument(
        "--n_instructions_per_language",
        type=int,
        default=None,
        help="Max battles per language (default: no limit).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help='Language codes to evaluate, e.g. "en fr de" (default: all).',
    )
    parser.add_argument(
        "--swap_mode",
        choices=["fixed", "both"],
        default=DEFAULT_SWAP_MODE,
        help=f"Swap mode for position-bias correction (default: {DEFAULT_SWAP_MODE}).",
    )
    parser.add_argument(
        "--provide_explanation",
        action="store_true",
        help="Ask judge to provide an explanation before scoring.",
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="Ignore cached completions and rerun everything.",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default=DEFAULT_RESULT_FOLDER,
        help=f"Folder to save results (default: {DEFAULT_RESULT_FOLDER}).",
    )
    parser.add_argument(
        "--max_out_tokens_models",
        type=int,
        default=DEFAULT_MAX_OUT_TOKENS_MODELS,
        help=(
            "Max generation tokens for model responses. "
            f"Default: {DEFAULT_MAX_OUT_TOKENS_MODELS}."
        ),
    )
    parser.add_argument(
        "--max_out_tokens_judge",
        type=int,
        default=DEFAULT_MAX_OUT_TOKENS_JUDGE,
        help=(
            "Max generation tokens for judge responses. "
            f"Default: {DEFAULT_MAX_OUT_TOKENS_JUDGE}."
        ),
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=(
            "VLLM max_model_len (context window cap, useful to avoid OOM). "
            f"Default: {DEFAULT_MAX_MODEL_LEN}."
        ),
    )

    # SLURM options
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_GPUS,
        help=f"Number of GPUs to allocate per job (default: {DEFAULT_GPUS}). Model and judge each use tp=1 by default and run sequentially, so 1 GPU is usually sufficient.",
    )
    parser.add_argument(
        "--model-tp-size",
        type=int,
        default=1,
        help="tensor_parallel_size for the candidate model (default: 1). Use a smaller value than --num-gpus when the model is small and the judge is large, to avoid GPU OOM when both load sequentially.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=DEFAULT_PARTITION,
        help=f"SLURM partition (default: {DEFAULT_PARTITION}).",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=DEFAULT_TIME,
        dest="time_limit",
        help=f"SLURM time limit (default: {DEFAULT_TIME}).",
    )

    # Submission options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts but do not submit.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs after generating scripts.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Automatically skip models that already have results (no interactive prompt).",
    )
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="Automatically rerun all models even if results already exist (no interactive prompt).",
    )

    args = parser.parse_args()

    # Create output directories
    Path(SCRIPTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # Load API config
    api_config = load_api_config()

    # Resolve judge
    try:
        judge_path = get_model_path(api_config, args.judge)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # Resolve models to evaluate
    if args.models:
        if len(args.models) == 1 and args.models[0].endswith(".txt"):
            model_keys = load_models_from_file(args.models[0])
        else:
            model_keys = args.models
    else:
        model_keys = load_models_from_file(args.models_file)
        if not model_keys:
            print(
                f"No models found in {args.models_file}. Use --models to specify models directly."
            )
            return

    # Validate all model keys exist in api_config
    valid_models = {}
    for key in model_keys:
        try:
            path = get_model_path(api_config, key)
            valid_models[key] = path
        except ValueError as e:
            print(f"Warning: {e} — skipping.")

    if not valid_models:
        print("No valid models to process. Exiting.")
        return

    print(f"Judge  : {args.judge}")
    print(f"         {judge_path}")
    print(f"Arena  : {args.arena}")
    print(f"Models : {len(valid_models)}")
    # for k, p in valid_models.items():
    #     print(f"  - {k}  ({p})")
    print()

    # Check for existing results
    models_with_results = {}
    models_without_results = {}
    for key, path in valid_models.items():
        existing = find_existing_results(args.result_folder, path, args.arena)
        if existing:
            models_with_results[key] = (path, existing)
        else:
            models_without_results[key] = path

    if models_with_results:
        print(f"Found existing results for {len(models_with_results)} model(s):")
        for key, (path, dirs) in models_with_results.items():
            print(f"  - {key}")
            for d in dirs:
                print(f"      {d}")
        print()

        if models_without_results:
            print(f"Models without existing results ({len(models_without_results)}):")
            for key in models_without_results:
                print(f"  - {key}")
            print()

        if args.rerun_all:
            models_to_run = valid_models
            print(f"--rerun-all set: submitting all {len(valid_models)} model(s) (including {len(models_with_results)} with existing results).\n")
        elif args.skip_existing:
            models_to_run = models_without_results
            print(f"--skip-existing set: skipping {len(models_with_results)} model(s) with existing results, submitting {len(models_without_results)} missing.\n")
        else:
            prompt = (
                f"{len(models_with_results)} model(s) already have results.\n"
                "  [a] Rerun all models\n"
                "  [m] Only run missing models\n"
                "  [q] Quit\n"
                "Choice [a/m/q]: "
            )
            while True:
                choice = input(prompt).strip().lower()
                if choice == "a":
                    models_to_run = valid_models
                    break
                elif choice == "m":
                    models_to_run = models_without_results
                    if not models_to_run:
                        print("All models already have results. Nothing to do.")
                        return
                    break
                elif choice == "q":
                    print("Aborted.")
                    return
                else:
                    print("Please enter 'a', 'm', or 'q'.")
    else:
        models_to_run = valid_models

    if not models_to_run:
        print("No models to run. Exiting.")
        return

    job_scripts = []
    for model_key, model_path in models_to_run.items():
        safe_model = safe_name(model_key)
        script_path = f"{SCRIPTS_DIR}/elo_{safe_model}.sh"

        create_slurm_script(
            model_key=model_key,
            model_path=model_path,
            judge_key=args.judge,
            judge_path=judge_path,
            script_path=script_path,
            arena=args.arena,
            n_instructions=args.n_instructions,
            n_instructions_per_language=args.n_instructions_per_language,
            languages=args.languages,
            swap_mode=args.swap_mode,
            provide_explanation=args.provide_explanation,
            result_folder=args.result_folder,
            max_out_tokens_models=args.max_out_tokens_models,
            max_out_tokens_judge=args.max_out_tokens_judge,
            max_model_len=args.max_model_len,
            num_gpus=args.num_gpus,
            model_tp_size=args.model_tp_size,
            partition=args.partition,
            time_limit=args.time_limit,
            ignore_cache=args.ignore_cache,
        )
        print(f"  Created: {script_path}")
        job_scripts.append(script_path)

    print(f"\nGenerated {len(job_scripts)} scripts in {SCRIPTS_DIR}")

    if args.dry_run:
        print("\nDry run — scripts generated but not submitted. To submit manually:")
        for s in job_scripts:
            print(f"  sbatch {s}")
    elif args.submit:
        print("\nSubmitting jobs...")
        submitted = []
        for s in job_scripts:
            try:
                result = subprocess.run(
                    ["sbatch", s], capture_output=True, text=True, check=True
                )
                job_id = result.stdout.strip().split()[-1]
                submitted.append((s, job_id))
                print(f"  {os.path.basename(s)} -> Job ID: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  Failed to submit {s}: {e}")
        print(f"\nSubmitted {len(submitted)} jobs.")
        print("Monitor with: squeue -u $USER")
    else:
        print("\nTo submit, rerun with --submit. Or manually:")
        for s in job_scripts[:3]:
            print(f"  sbatch {s}")
        if len(job_scripts) > 3:
            print(f"  ... and {len(job_scripts) - 3} more")


if __name__ == "__main__":
    main()
