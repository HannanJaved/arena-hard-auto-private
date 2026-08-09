#!/usr/bin/env python3
"""
Automation script to generate Arena Hard answers for multiple models WITHOUT vLLM.
This script creates individual SLURM jobs for each model to run in parallel on GPU.

Approach:
- Generates per-model answer config (bench + model_list)
- Generates per-model endpoint config with local_engine=True
- Submits SLURM jobs that call gen_answer.py directly (no API server)
"""

import argparse
import os
import subprocess
from pathlib import Path

import yaml

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
ARENA_HARD_AUTO_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto"
LOGS_DIR = f"{WORKSPACE_ROOT}/logs/arena-hard"
SCRIPTS_DIR = f"{WORKSPACE_ROOT}/generated_scripts"
CONFIGS_DIR = f"{WORKSPACE_ROOT}/generated_configs"


def load_api_config():
    """Load the API configuration file."""
    config_path = f"{ARENA_HARD_AUTO_DIR}/config/api_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_directories():
    """Create necessary directories for scripts, configs, and logs."""
    for directory in [SCRIPTS_DIR, CONFIGS_DIR, LOGS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def create_gen_answer_config(model_name, output_path, bench_name):
    """Create a gen_answer_config.yaml file for a specific model."""
    config = {
        "bench_name": bench_name,
        "model_list": [model_name],
    }
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def create_local_endpoint_config(model_name, model_config, output_path, trust_remote_code=False):
    """
    Create per-model endpoint config for local engine usage (no vLLM).

    This preserves most fields from api_config[model_name], but enforces:
    - local_engine: true
    - model: <path from api_config>
    - api_base/endpoints removed (not needed for local generation)
    """
    endpoint_entry = dict(model_config) if isinstance(model_config, dict) else {}

    endpoint_entry["local_engine"] = True
    if trust_remote_code:
        endpoint_entry["trust_remote_code"] = True

    # Keep api_type if present; if absent, default to huggingface-ish local handler.
    # If your registry expects another key, update here.
    endpoint_entry.setdefault("api_type", "huggingface")

    # Ensure model path exists in entry
    if "model" not in endpoint_entry:
        raise ValueError(f"Model config for '{model_name}' missing required 'model' path.")

    # Remove remote API-only keys
    endpoint_entry.pop("api_base", None)
    endpoint_entry.pop("endpoints", None)

    config = {model_name: endpoint_entry}
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def extract_model_details(model_name):
    """Extract model details from model name for directory structure."""
    parts = model_name.split("-")

    rank = None
    alpha = None
    step = None

    for i, part in enumerate(parts):
        if part.startswith("rank"):
            rank = part
        elif part.startswith("alpha"):
            alpha = f"{part}-{parts[i + 1]}" if i + 1 < len(parts) else part
        elif part.startswith("step") or part == "final":
            step = part

    return rank, alpha, step


def create_slurm_script(model_name, script_path):
    """Create a SLURM script for a specific model (no vLLM)."""
    rank, alpha, step = extract_model_details(model_name)

    # Create log directory structure
    log_subdir = f"{rank}/{alpha}" if rank and alpha else "misc"
    log_dir = f"{LOGS_DIR}/{log_subdir}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Generated config paths
    answer_config_file = f"{CONFIGS_DIR}/gen_answer_config_{model_name}.yaml"
    endpoint_config_file = f"{CONFIGS_DIR}/api_config_{model_name}_local.yaml"

    script_content = f"""#!/bin/bash
#SBATCH --job-name={model_name}-novllm
#SBATCH --error={log_dir}/{model_name}_novllm.err
#SBATCH --output={log_dir}/{model_name}_novllm.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --partition=capella
#SBATCH --gres=gpu:1

set -e

echo "Setting up environment for {model_name} (no vLLM)..."
source {WORKSPACE_ROOT}/ah-eval/bin/activate

PYTHON_EXEC={WORKSPACE_ROOT}/ah-eval/bin/python
echo "Using Python executable at: $PYTHON_EXEC"

module load release/24.10
module load CUDA/12.4.0

echo "--- Sanity Checks ---"
$PYTHON_EXEC -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "---------------------"

cd {ARENA_HARD_AUTO_DIR}

echo "Running local generation (no vLLM) for {model_name}"
echo "Answer config: {answer_config_file}"
echo "Endpoint config: {endpoint_config_file}"

# Pin to one GPU slot for this job
export CUDA_VISIBLE_DEVICES=0

$PYTHON_EXEC {ARENA_HARD_AUTO_DIR}/gen_answer.py \\
    --config-file {answer_config_file} \\
    --endpoint-file {endpoint_config_file}

echo "Job completed successfully for {model_name} (no vLLM)"
"""
    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)


def load_models_from_file(file_path):
    """Load model names from a text file, ignoring comments and empty lines."""
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


def resolve_models_to_process(args, api_config):
    """Resolve model subset from CLI options."""
    if args.models:
        if len(args.models) == 1 and args.models[0].endswith(".txt"):
            model_names_from_file = load_models_from_file(args.models[0])
            if not model_names_from_file:
                print(f"No models found in {args.models[0]}.")
                return {}
            models_list = model_names_from_file
        else:
            models_list = args.models

        models_to_process = {}
        for model in models_list:
            if model in api_config:
                models_to_process[model] = api_config[model]
            else:
                print(f"Warning: Model '{model}' not found in API config")
        return models_to_process

    if args.missing_models_file:
        model_names_from_file = load_models_from_file(args.missing_models_file)
        if not model_names_from_file:
            print(f"No models found in {args.missing_models_file}.")
            return {}

        models_to_process = {}
        for model in model_names_from_file:
            if model in api_config:
                models_to_process[model] = api_config[model]
            else:
                print(f"Warning: Model '{model}' from missing models file not found in API config")
        print(f"Processing {len(models_to_process)} missing models from {args.missing_models_file}")
        return models_to_process

    if args.all:
        return {k: v for k, v in api_config.items() if k.startswith("tulu3-") and "alpha" in k}

    model_names_from_file = load_models_from_file(args.models_file)
    if not model_names_from_file:
        print(
            f"No models found in {args.models_file}. "
            "Use --all to process all models or --models to specify models directly."
        )
        return {}

    models_to_process = {}
    for model in model_names_from_file:
        if model in api_config:
            models_to_process[model] = api_config[model]
        else:
            print(f"Warning: Model '{model}' from file not found in API config")
    return models_to_process


def main():
    parser = argparse.ArgumentParser(
        description="Automate Arena Hard answer generation for multiple models (NO vLLM)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to process, or a single .txt file containing model names (one per line)",
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default=f"{WORKSPACE_ROOT}/arena_hard_models_to_test.txt",
        help="File containing list of models to process (default: arena_hard_models_to_test.txt)",
    )
    parser.add_argument(
        "--missing-models-file",
        type=str,
        help="File containing missing models (generated by check_missing_model_answers.py)",
    )
    parser.add_argument("--all", action="store_true", help="Process all tulu3 models from API config")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit jobs")
    parser.add_argument("--submit", action="store_true", help="Submit jobs after generating scripts")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code for local Hugging Face loading",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="arena-hard-v2.0",
        help="Benchmark name to use in config files (default: arena-hard-v2.0)",
        choices=[
            "arena-hard-v0.1",
            "arena-hard-v2.0",
            "hard_prompt",
            "coding",
            "math",
            "creative_writing",
        ],
    )

    args = parser.parse_args()

    create_directories()
    api_config = load_api_config()
    models_to_process = resolve_models_to_process(args, api_config)

    print(f"Found {len(models_to_process)} models to process:")
    for model_name in models_to_process.keys():
        print(f"  - {model_name}")

    if not models_to_process:
        print("No models to process. Exiting.")
        return

    job_scripts = []
    for model_name, model_config in models_to_process.items():
        print(f"\nProcessing {model_name}...")

        answer_config_path = f"{CONFIGS_DIR}/gen_answer_config_{model_name}.yaml"
        create_gen_answer_config(model_name, answer_config_path, args.bench_name)
        print(f"  Created answer config: {answer_config_path}")

        endpoint_config_path = f"{CONFIGS_DIR}/api_config_{model_name}_local.yaml"
        create_local_endpoint_config(
            model_name,
            model_config,
            endpoint_config_path,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"  Created endpoint config: {endpoint_config_path}")

        script_path = f"{SCRIPTS_DIR}/run_arena_hard_no_vllm_{model_name}.sh"
        create_slurm_script(model_name, script_path)
        print(f"  Created script: {script_path}")

        job_scripts.append(script_path)

    print(f"\nGenerated {len(job_scripts)} job scripts in {SCRIPTS_DIR}")
    print(f"Generated {len(models_to_process)} answer configs in {CONFIGS_DIR}")
    print(f"Generated {len(models_to_process)} local endpoint configs in {CONFIGS_DIR}")

    if args.dry_run:
        print("\nDry run complete. Scripts generated but not submitted.")
        print("To submit jobs manually, run:")
        for script in job_scripts:
            print(f"  sbatch {script}")
    elif args.submit:
        print("\nSubmitting jobs...")
        submitted_jobs = []
        for script in job_scripts:
            try:
                result = subprocess.run(["sbatch", script], capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs.append((script, job_id))
                print(f"  Submitted {os.path.basename(script)} -> Job ID: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  Failed to submit {script}: {e}")

        print(f"\nSuccessfully submitted {len(submitted_jobs)} jobs")
        print("\nTo monitor jobs:")
        print("  squeue -u $USER")
        print("\nTo cancel all jobs:")
        print("  scancel -u $USER")
    else:
        print("\nTo submit jobs, run:")
        print(f"  python {__file__} --submit")
        print("\nOr submit individual jobs with:")
        for script in job_scripts[:3]:
            print(f"  sbatch {script}")
        if len(job_scripts) > 3:
            print(f"  ... and {len(job_scripts) - 3} more")


if __name__ == "__main__":
    main()