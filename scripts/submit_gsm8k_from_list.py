#!/usr/bin/env python3
"""Submit GSM8K jobs for models listed in a text file.

The text file should contain one model name per line matching keys in
`arena-hard-auto/config/api_config.yaml`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List

import yaml

DEFAULT_API_CONFIG = "/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml"
# gsm8k is a generate_until task; vLLM's batched/paged-attention decoding is much
# faster than HF's .generate() loop here, so this runs out of venv-vllm (which has
# both vllm and lm_eval installed) instead of venv-lm-eval (no vllm package).
DEFAULT_VENV_ACTIVATE = "/data/horse/ws/hama901h-BFTranslation/venv-vllm/bin/activate"
DEFAULT_LM_EVAL_DIR = "/data/horse/ws/hama901h-BFTranslation/lm-evaluation-harness"
DEFAULT_LOG_DIR = "/data/horse/ws/hama901h-BFTranslation/logs/LM-eval/"
DEFAULT_OUTPUT_DIR = "/data/horse/ws/hama901h-BFTranslation/evaluation_results/gsm8k"
DEFAULT_HF_HOME = "/data/horse/ws/hama901h-BFTranslation/.cache"
DEFAULT_HF_DATASETS_CACHE = "/data/horse/ws/hama901h-BFTranslation/.cache"
DEFAULT_PYTHONPATH = "/data/horse/ws/hama901h-BFTranslation/venv-vllm/lib/python3.11/site-packages"

SBATCH_HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --account={account}
{exclude_nodes}
{exclusive}

"""

SBATCH_BODY = """\
echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA
source {venv_activate}

export HF_HOME="{hf_home}"
export HF_DATASETS_CACHE="{hf_datasets_cache}"
export PYTHONPATH="{pythonpath}"

cd {lm_eval_dir}

echo "JOBNAME" $SLURM_JOB_NAME
pwd -P

mkdir -p {output_dir}

export CMD="lm_eval --model vllm \
    --model_args pretrained={model_path},dtype=\"{dtype}\",gpu_memory_utilization=0.9 \
    --tasks gsm8k \
    --num_fewshot 5 \
    --gen_kwargs max_new_tokens=1024 \
    --batch_size {batch_size} \
    --output_path {output_dir}"

python -m $CMD

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
"""

SBATCH_BODY_MULTI_GPU = """\
echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA
source {venv_activate}

export HF_HOME="{hf_home}"
export HF_DATASETS_CACHE="{hf_datasets_cache}"
export PYTHONPATH="{pythonpath}"

cd {lm_eval_dir}

echo "JOBNAME" $SLURM_JOB_NAME
pwd -P

mkdir -p {output_dir}

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

# vLLM shards across GPUs itself via tensor_parallel_size (spawns its own worker
# processes internally) -- no accelerate/srun multi-process launch needed here,
# unlike the HF-backend static-eval scripts.
export CMD="lm_eval --model vllm \
    --model_args pretrained={model_path},dtype=\"{dtype}\",gpu_memory_utilization=0.9,tensor_parallel_size=$NPROC_PER_NODE \
    --tasks gsm8k \
    --num_fewshot 5 \
    --gen_kwargs max_new_tokens=1024 \
    --batch_size {batch_size} \
    --output_path {output_dir}"

python -m $CMD

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit GSM8K jobs for a list of model names.")
    parser.add_argument("--models-file", required=True, help="Path to text file with model names.")
    parser.add_argument("--api-config", default=DEFAULT_API_CONFIG, help="Path to api_config.yaml.")
    parser.add_argument("--job-name-prefix", default="gsm8k_", help="Prefix for Slurm job name.")
    parser.add_argument("--batch-size", default="auto", help="Batch size for lm_eval (int, or 'auto'/'auto:N'). vLLM manages its own request batching internally.")
    parser.add_argument("--dtype", default="bfloat16", help="dtype passed to lm_eval model_args.")
    parser.add_argument("--partition", default="capella", help="Slurm partition.")
    parser.add_argument("--account", choices=["p_neurasearch", "p_scads_nas"], default="p_neurasearch",
                        help="SLURM account to charge jobs to (default: p_neurasearch).")
    parser.add_argument("--time", default="03:00:00", help="Slurm wall time.")
    parser.add_argument("--gres", default="gpu:1", help="Slurm gres.")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="Slurm CPUs per task.")
    parser.add_argument("--mem", default="32G", help="Slurm memory.")
    parser.add_argument("--exclusive", action="store_true", help="Request exclusive node.")
    parser.add_argument("--no-exclusive", dest="exclusive", action="store_false")
    parser.set_defaults(exclusive=False)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Directory for Slurm logs.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where lm_eval saves results.")
    parser.add_argument("--venv-activate", default=DEFAULT_VENV_ACTIVATE, help="Path to venv activate script.")
    parser.add_argument("--lm-eval-dir", default=DEFAULT_LM_EVAL_DIR, help="Path to lm-evaluation-harness.")
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME, help="HF cache home.")
    parser.add_argument("--hf-datasets-cache", default=DEFAULT_HF_DATASETS_CACHE, help="HF datasets cache.")
    parser.add_argument("--pythonpath", default=DEFAULT_PYTHONPATH, help="PYTHONPATH for lm-eval venv.")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch scripts without submitting.")
    return parser.parse_args()


def read_models_list(path: str) -> List[str]:
    models: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line in seen:
                continue
            seen.add(line)
            models.append(line)
    return models


def load_api_config(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"api_config.yaml should be a mapping, got {type(data)}")
    return data


def sanitize_job_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return sanitized[:128]


def build_sbatch_script(
    *,
    model_name: str,
    model_path: str,
    args: argparse.Namespace,
) -> str:
    exclusive_line = "#SBATCH --exclusive" if args.exclusive else ""
    exclude_nodes_line = "#SBATCH --exclude=c52" if args.partition == "capella" else ""
    header = SBATCH_HEADER.format(
        job_name=f"{args.job_name_prefix}{sanitize_job_name(model_name)}",
        log_dir=args.log_dir,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time=args.time,
        partition=args.partition,
        account=args.account,
        exclude_nodes=exclude_nodes_line,
        exclusive=exclusive_line,
    )

    if args.gres == "gpu:1":
        body = SBATCH_BODY.format(
            venv_activate=args.venv_activate,
            hf_home=args.hf_home,
            hf_datasets_cache=args.hf_datasets_cache,
            pythonpath=args.pythonpath,
            lm_eval_dir=args.lm_eval_dir,
            model_path=model_path,
            dtype=args.dtype,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
    else:
        body = SBATCH_BODY_MULTI_GPU.format(
            venv_activate=args.venv_activate,
            hf_home=args.hf_home,
            hf_datasets_cache=args.hf_datasets_cache,
            pythonpath=args.pythonpath,
            lm_eval_dir=args.lm_eval_dir,
            model_path=model_path,
            dtype=args.dtype,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

    return f"{header}{body}"


def submit_job(script_contents: str, dry_run: bool) -> None:
    if dry_run:
        print("\n" + "=" * 80)
        print(script_contents)
        return

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(script_contents)
        temp_path = handle.name

    try:
        result = subprocess.run(["sbatch", temp_path], capture_output=True, text=True)
        if result.returncode != 0:
            print("sbatch failed.")
            if result.stdout:
                print("\nSTDOUT:\n" + result.stdout.strip())
            if result.stderr:
                print("\nSTDERR:\n" + result.stderr.strip())
            print(f"Temporary script preserved at: {temp_path}")
            return
        print(result.stdout.strip())
    finally:
        if result.returncode == 0:
            os.unlink(temp_path)


def main() -> int:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    models = read_models_list(args.models_file)
    if not models:
        print("No models found in list.")
        return 1

    api_config = load_api_config(args.api_config)

    missing: List[str] = []
    skipped: List[str] = []

    for model_name in models:
        entry = api_config.get(model_name)
        if entry is None:
            missing.append(model_name)
            continue

        if not isinstance(entry, dict):
            skipped.append(model_name)
            print(f"Skipping {model_name}: config entry is not a mapping.")
            continue

        model_path = entry.get("model")
        if not model_path:
            skipped.append(model_name)
            print(f"Skipping {model_name}: no 'model' path found in api_config.")
            continue

        output_subdir = model_path.replace("/", "__")
        result_dir = os.path.join(args.output_dir, output_subdir)
        if os.path.isdir(result_dir) and any(
            f.startswith("results_") and f.endswith(".json")
            for f in os.listdir(result_dir)
        ):
            print(f"Skipping {model_name}: results already exist in {result_dir}")
            skipped.append(model_name)
            continue

        script_contents = build_sbatch_script(
            model_name=model_name,
            model_path=model_path,
            args=args,
        )
        submit_job(script_contents, args.dry_run)

    if missing:
        print("\nMissing models in api_config:")
        for name in missing:
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
