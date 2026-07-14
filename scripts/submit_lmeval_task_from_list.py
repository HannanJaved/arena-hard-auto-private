#!/usr/bin/env python3
"""Submit lm_eval jobs for a task and models listed in a text file.

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
from pathlib import Path
from typing import Dict, List

import yaml

DEFAULT_API_CONFIG = "/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml"
DEFAULT_VENV_ACTIVATE = "/data/horse/ws/hama901h-BFTranslation/venv-lm-eval/bin/activate"
DEFAULT_LM_EVAL_DIR = "/data/horse/ws/hama901h-BFTranslation/lm-evaluation-harness"
DEFAULT_LOG_DIR = "/data/horse/ws/hama901h-BFTranslation/logs/LM-eval"
DEFAULT_OUTPUT_ROOT = "/data/horse/ws/hama901h-BFTranslation/evaluation_results"
DEFAULT_HF_HOME = "/data/horse/ws/hama901h-BFTranslation/.cache"
DEFAULT_HF_DATASETS_CACHE = "/data/horse/ws/hama901h-BFTranslation/.cache"
DEFAULT_PYTHONPATH = "/data/horse/ws/hama901h-BFTranslation/venv-lm-eval/lib/python3.11/site-packages"

# --use-module-torch: use the cluster's module-provided PyTorch (fast, non-Lustre import) instead of the
# pip-installed torch in venv-lm-eval (whose large .so files suffer catastrophic Lustre read latency).
# Requires venv-lm-eval-module, a `python -m venv --system-site-packages` venv built on top of the module's
# Python, with transformers pinned to a version that still supports torch 2.3.0 (the cluster's newest module).
DEFAULT_VENV_ACTIVATE_MODULE = "/data/horse/ws/hama901h-BFTranslation/venv-lm-eval-module/bin/activate"
DEFAULT_PYTHONPATH_MODULE = "/data/horse/ws/hama901h-BFTranslation/venv-lm-eval-module/lib/python3.11/site-packages"
# lm_eval>=0.4.12 uses TypedDict(..., extra_items=...) which needs typing_extensions>=4.13.0.
# venv-lm-eval-module has no bundled torch and often no typing_extensions, so the cluster module's
# older copy would be picked up unless we install typing_extensions into that venv explicitly:
#   venv-lm-eval/bin/pip install 'typing_extensions>=4.13.0' \
#     --target=venv-lm-eval-module/lib/python3.11/site-packages --upgrade
MIN_MODULE_TYPING_EXTENSIONS = (4, 13, 0)
MODULE_TORCH_PRELUDE_CPU = "module purge\nmodule load release/24.10 GCC/13.2.0 OpenMPI/4.1.6\nmodule load PyTorch/2.3.0"
MODULE_TORCH_PRELUDE_GPU = "module purge\nmodule load release/24.10 GCC/13.2.0 OpenMPI/4.1.6\nmodule load PyTorch/2.3.0-CUDA-12.4.0"

HEARTBEAT_START = """\
( while true; do sleep 60; echo "HEARTBEAT: still running at $(date), elapsed ${SECONDS}s"; done ) &
HEARTBEAT_PID=$!
"""

HEARTBEAT_STOP = "kill $HEARTBEAT_PID 2>/dev/null"

SBATCH_HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
{gres_line}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --partition={partition}
{exclusive}

"""

SBATCH_BODY = """\
echo "JOB NAME" $SLURM_JOB_NAME

{module_prelude}
source {venv_activate}

export HF_HOME="{hf_home}"
export HF_DATASETS_CACHE="{hf_datasets_cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="{pythonpath}"

cd {lm_eval_dir}

echo "JOBNAME" $SLURM_JOB_NAME
pwd -P

mkdir -p {output_dir}

export CMD="lm_eval --model hf \
    --model_args pretrained={model_path},dtype=\"{dtype}\"{model_args_extra} \
    --tasks {task} \
    --num_fewshot {num_fewshot} \
    --batch_size {batch_size} \
    --output_path {output_dir}"

echo "CMD: $CMD"
echo "START TIME: $(date)"
{heartbeat_start}
python -u -m $CMD
EXIT_CODE=$?
{heartbeat_stop}

if [ $EXIT_CODE -eq 0 ]; then
  echo "END TIME: $(date)"
fi

echo "END $SLURM_JOBID: $(date)"
exit $EXIT_CODE
"""

SBATCH_BODY_CPU = """\
echo "JOB NAME" $SLURM_JOB_NAME

{module_prelude}
source {venv_activate}

export HF_HOME="{hf_home}"
export HF_DATASETS_CACHE="{hf_datasets_cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="{pythonpath}"

cd {lm_eval_dir}

echo "JOBNAME" $SLURM_JOB_NAME
pwd -P

mkdir -p {output_dir}

echo "CPUS: $SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export CMD="lm_eval --model hf \
    --model_args pretrained={model_path},dtype=\"{dtype}\"{model_args_extra} \
    --tasks {task} \
    --num_fewshot {num_fewshot} \
    --batch_size {batch_size} \
    --device cpu \
    --output_path {output_dir}"

echo "CMD: $CMD"
echo "START TIME: $(date)"
{heartbeat_start}
python -u -m $CMD
EXIT_CODE=$?
{heartbeat_stop}

if [ $EXIT_CODE -eq 0 ]; then
  echo "END TIME: $(date)"
fi

echo "END $SLURM_JOBID: $(date)"
exit $EXIT_CODE
"""

SBATCH_BODY_MULTI_GPU = """\
echo "JOB NAME" $SLURM_JOB_NAME

{module_prelude}
source {venv_activate}

export HF_HOME="{hf_home}"
export HF_DATASETS_CACHE="{hf_datasets_cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="{pythonpath}"

cd {lm_eval_dir}

echo "JOBNAME" $SLURM_JOB_NAME
pwd -P

mkdir -p {output_dir}

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
TOTAL_BATCH_SIZE=$((NPROC_PER_NODE*{batch_size}))
echo "NPROC_PER_NODE: $NPROC_PER_NODE, TOTAL_BATCH_SIZE: $TOTAL_BATCH_SIZE"

export PYTHONUNBUFFERED=1
export CMD="lm_eval --model hf \
    --model_args pretrained={model_path},dtype=\"{dtype}\"{model_args_extra} \
    --tasks {task} \
    --num_fewshot {num_fewshot} \
    --batch_size $TOTAL_BATCH_SIZE \
    --output_path {output_dir}"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

export ACC_LAUNCHER="accelerate launch -m "

echo "CMD: $ACC_LAUNCHER $CMD"
echo "START TIME: $(date)"
{heartbeat_start}
srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER $CMD"
EXIT_CODE=$?
{heartbeat_stop}

if [ $EXIT_CODE -eq 0 ]; then
  echo "END TIME: $(date)"
fi

echo "END $SLURM_JOBID: $(date)"
exit $EXIT_CODE
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit lm_eval jobs for a task and model list.")
    parser.add_argument("--models-file", required=True, help="Path to text file with model names.")
    parser.add_argument("--task", required=True, help="lm_eval task name.")
    parser.add_argument("--num-fewshot", type=int, required=True, help="Number of few-shot examples.")
    parser.add_argument("--api-config", default=DEFAULT_API_CONFIG, help="Path to api_config.yaml.")
    parser.add_argument("--job-name-prefix", help="Prefix for Slurm job name (default: <task>_<num_fewshot>shot_).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU for lm_eval.")
    parser.add_argument("--dtype", default="bfloat16", help="dtype passed to lm_eval model_args.")
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Pass trust_remote_code=True in lm_eval model_args, for models with custom modeling code.",
    )
    parser.add_argument("--partition", default="alpha", help="Slurm partition.")
    parser.add_argument("--time", default="01:00:00", help="Slurm wall time.")
    parser.add_argument("--gres", default="gpu:1", help="Slurm gres.")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="Slurm CPUs per task.")
    parser.add_argument("--mem", default="16G", help="Slurm memory.")
    parser.add_argument("--exclusive", action="store_true", help="Request exclusive node.")
    parser.add_argument("--no-exclusive", dest="exclusive", action="store_false")
    parser.set_defaults(exclusive=False)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Directory for Slurm logs.")
    parser.add_argument("--output-dir", help="Directory where lm_eval saves results.")
    parser.add_argument(
        "--venv-activate", default=None,
        help=f"Path to venv activate script (default: {DEFAULT_VENV_ACTIVATE}, or "
             f"{DEFAULT_VENV_ACTIVATE_MODULE} with --use-module-torch).",
    )
    parser.add_argument("--lm-eval-dir", default=DEFAULT_LM_EVAL_DIR, help="Path to lm-evaluation-harness.")
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME, help="HF cache home.")
    parser.add_argument("--hf-datasets-cache", default=DEFAULT_HF_DATASETS_CACHE, help="HF datasets cache.")
    parser.add_argument(
        "--pythonpath", default=None,
        help=f"PYTHONPATH for lm-eval venv (default: {DEFAULT_PYTHONPATH}, or "
             f"{DEFAULT_PYTHONPATH_MODULE} with --use-module-torch).",
    )
    parser.add_argument(
        "--use-module-torch", action="store_true",
        help=(
            "Use the cluster's module-provided PyTorch/2.3.0 (fast, non-Lustre import) via venv-lm-eval-module "
            "instead of the pip-installed torch in venv-lm-eval, which suffers catastrophic Lustre read latency "
            "on its large .so files. Also switches the 'module load CUDA' prelude to the matching PyTorch module."
        ),
    )
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


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def _check_module_typing_extensions(pythonpath: str) -> None:
    typing_file = Path(pythonpath) / "typing_extensions.py"
    if not typing_file.exists():
        print(
            f"ERROR: --use-module-torch requires typing_extensions>={'.'.join(map(str, MIN_MODULE_TYPING_EXTENSIONS))} "
            f"in {pythonpath}.\n"
            "Install with:\n"
            f"  {DEFAULT_VENV_ACTIVATE.replace('/bin/activate', '/bin/pip')} "
            f"install 'typing_extensions>={'.'.join(map(str, MIN_MODULE_TYPING_EXTENSIONS))}' "
            f"--target={pythonpath} --upgrade",
            file=sys.stderr,
        )
        sys.exit(2)

    version = None
    for meta in Path(pythonpath).glob("typing_extensions-*.dist-info/METADATA"):
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
                break
        if version:
            break

    if version and _parse_version(version) < MIN_MODULE_TYPING_EXTENSIONS:
        print(
            f"ERROR: typing_extensions {version} in {pythonpath} is too old; "
            f"need >={'.'.join(map(str, MIN_MODULE_TYPING_EXTENSIONS))}.",
            file=sys.stderr,
        )
        sys.exit(2)


def build_sbatch_script(
    *,
    model_name: str,
    model_path: str,
    args: argparse.Namespace,
) -> str:
    exclusive_line = "#SBATCH --exclusive" if args.exclusive else ""
    cpu_only = not args.gres or args.gres.lower() == "none"
    gres_line = "" if cpu_only else f"#SBATCH --gres={args.gres}"
    job_name_prefix = args.job_name_prefix or f"{args.task}_{args.num_fewshot}shot_"
    header = SBATCH_HEADER.format(
        job_name=f"{job_name_prefix}{sanitize_job_name(model_name)}",
        log_dir=args.log_dir,
        gres_line=gres_line,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time=args.time,
        partition=args.partition,
        exclusive=exclusive_line,
    )

    if args.use_module_torch:
        venv_activate = args.venv_activate or DEFAULT_VENV_ACTIVATE_MODULE
        pythonpath = args.pythonpath or DEFAULT_PYTHONPATH_MODULE
        module_prelude = MODULE_TORCH_PRELUDE_CPU if cpu_only else MODULE_TORCH_PRELUDE_GPU
    else:
        venv_activate = args.venv_activate or DEFAULT_VENV_ACTIVATE
        pythonpath = args.pythonpath or DEFAULT_PYTHONPATH
        module_prelude = "" if cpu_only else "module load CUDA"

    model_args_extra = ",trust_remote_code=True" if args.trust_remote_code else ""

    body_kwargs = dict(
        venv_activate=venv_activate,
        hf_home=args.hf_home,
        hf_datasets_cache=args.hf_datasets_cache,
        pythonpath=pythonpath,
        module_prelude=module_prelude,
        lm_eval_dir=args.lm_eval_dir,
        model_path=model_path,
        dtype=args.dtype,
        model_args_extra=model_args_extra,
        task=args.task,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        heartbeat_start=HEARTBEAT_START,
        heartbeat_stop=HEARTBEAT_STOP,
    )

    if cpu_only:
        body = SBATCH_BODY_CPU.format(**body_kwargs)
    elif args.gres == "gpu:1":
        body = SBATCH_BODY.format(**body_kwargs)
    else:
        body = SBATCH_BODY_MULTI_GPU.format(**body_kwargs)

    return f"{header}{body}"


def submit_job(script_contents: str, dry_run: bool) -> None:
    if dry_run:
        print("\n" + "=" * 80)
        print(script_contents)
        return

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(script_contents)
        temp_path = handle.name

    result = subprocess.run(["sbatch", temp_path], capture_output=True, text=True)
    try:
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
    if args.use_module_torch:
        pythonpath = args.pythonpath or DEFAULT_PYTHONPATH_MODULE
        _check_module_typing_extensions(pythonpath)

    if not args.output_dir:
        args.output_dir = f"{DEFAULT_OUTPUT_ROOT}/{args.task}"
    if not args.job_name_prefix:
        args.job_name_prefix = f"{args.task}_{args.num_fewshot}shot_"

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
