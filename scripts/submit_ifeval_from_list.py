#!/usr/bin/env python3
"""Submit IFEval jobs for models listed in a text file.

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
import textwrap
from typing import Dict, Iterable, List

import yaml

DEFAULT_API_CONFIG = "/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml"
DEFAULT_VENV_ACTIVATE = "/data/horse/ws/hama901h-BFTranslation/venv-lm-eval/bin/activate"
DEFAULT_LM_EVAL_DIR = "/data/horse/ws/hama901h-BFTranslation/lm-evaluation-harness"
DEFAULT_LOG_DIR = "/data/horse/ws/hama901h-BFTranslation/logs/LM-eval"
DEFAULT_OUTPUT_DIR = "/data/horse/ws/hama901h-BFTranslation/evaluation_results/ifeval"
DEFAULT_HF_HOME = "/data/cat/ws/hama901h-Posttraining/.cache"
DEFAULT_HF_DATASETS_CACHE = "/data/cat/ws/hama901h-Posttraining/.cache"
DEFAULT_PYTHONPATH = "/data/horse/ws/hama901h-BFTranslation/venv-lm-eval/lib/python3.12/site-packages"

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

export CMD="lm_eval --model hf \
    --model_args pretrained={model_path},dtype=\"{dtype}\" \
    --tasks ifeval \
    --batch_size {batch_size} \
    --output_path {output_dir}"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

export ACC_LAUNCHER="accelerate launch -m "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER $CMD"

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
TOTAL_BATCH_SIZE=$((NPROC_PER_NODE*{batch_size}))

export CMD="lm_eval --model hf \
    --model_args pretrained={model_path},dtype=\"{dtype}\" \
    --tasks ifeval \
    --batch_size $TOTAL_BATCH_SIZE \
    --output_path {output_dir}"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

export ACC_LAUNCHER="accelerate launch -m "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit IFEval jobs for a list of model names.")
    parser.add_argument("--models-txt", required=True, help="Path to text file with model names.")
    parser.add_argument("--api-config", default=DEFAULT_API_CONFIG, help="Path to api_config.yaml.")
    parser.add_argument("--job-name-prefix", default="ifeval_", help="Prefix for Slurm job name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU for lm_eval.")
    parser.add_argument("--dtype", default="bfloat16", help="dtype passed to lm_eval model_args.")
    parser.add_argument("--partition", default="capella", help="Slurm partition.")
    parser.add_argument("--time", default="01:00:00", help="Slurm wall time.")
    parser.add_argument("--gres", default="gpu:1", help="Slurm gres.")
    parser.add_argument("--cpus-per-task", type=int, default=14, help="Slurm CPUs per task.")
    parser.add_argument("--mem", default="32G", help="Slurm memory.")
    parser.add_argument("--exclusive", action="store_true", help="Request exclusive node.")
    parser.add_argument("--no-exclusive", dest="exclusive", action="store_false")
    parser.set_defaults(exclusive=True)
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
    header = SBATCH_HEADER.format(
        job_name=f"{args.job_name_prefix}{sanitize_job_name(model_name)}",
        log_dir=args.log_dir,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time=args.time,
        partition=args.partition,
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
        result = subprocess.run(["sbatch", temp_path], check=True, capture_output=True, text=True)
        print(result.stdout.strip())
    finally:
        os.unlink(temp_path)


def main() -> int:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    models = read_models_list(args.models_txt)
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
