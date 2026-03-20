#!/usr/bin/env python3
"""Submit OpenJury ELO estimation jobs for models listed in a text file.

The models and judge are referenced by key in `arena-hard-auto/config/api_config.yaml`.
This helper generates Slurm scripts that call `openjury/estimate_elo_ratings.py`
with a VLLM provider prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List

import yaml

DEFAULT_API_CONFIG = "/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml"
DEFAULT_VENV_ACTIVATE = "/data/horse/ws/hama901h-BFTranslation/OpenJury/.venv/bin/activate"
DEFAULT_OPENJURY_DIR = "/data/horse/ws/hama901h-BFTranslation/OpenJury"
DEFAULT_LOG_DIR = "/data/horse/ws/hama901h-BFTranslation/logs/openjury-elo"
DEFAULT_RESULT_DIR = "/data/horse/ws/hama901h-BFTranslation/evaluation_results/openjury-elo"
DEFAULT_HF_HOME = "/data/cat/ws/hama901h-Posttraining/.cache"
DEFAULT_HF_DATASETS_CACHE = "/data/cat/ws/hama901h-Posttraining/.cache"

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
export PYTHONPATH="{openjury_dir}:$PYTHONPATH"

cd {openjury_dir}

mkdir -p {result_dir}

python openjury/estimate_elo_ratings.py \
  --arena {arena} \
  --model {model_name} \
  --judge {judge_name} \
  --n_instructions {n_instructions} \
  --swap_mode {swap_mode} \
  --truncate_all_input_chars {truncate_all_input_chars} \
  --max_out_tokens_models {max_out_tokens_models} \
  --max_out_tokens_judge {max_out_tokens_judge} \
  --result_folder {result_dir} \
    --engine_kwargs '{engine_kwargs}'{n_instructions_per_language_flag}{max_model_len_flag}{extra_args}

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit OpenJury estimate_elo_ratings jobs for a list of models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names or a single .txt file containing one model per line.",
    )
    parser.add_argument("--judge-model", required=True, help="Judge model key in api_config.yaml.")
    parser.add_argument("--api-config", default=DEFAULT_API_CONFIG, help="Path to api_config.yaml.")
    parser.add_argument("--job-name-prefix", default="elo_", help="Prefix for Slurm job name.")
    parser.add_argument("--arena", default="LMArena", choices=["LMArena", "ComparIA"], help="Arena to use.")
    parser.add_argument("--n-instructions", type=int, default=200, help="Number of battles to sample.")
    parser.add_argument(
        "--n-instructions-per-language",
        type=int,
        default=None,
        help="Max instructions per language (None means no cap).",
    )
    parser.add_argument("--swap-mode", default="fixed", choices=["fixed", "both"], help="Swap mode.")
    parser.add_argument("--truncate-all-input-chars", type=int, default=8192)
    parser.add_argument("--max-out-tokens-models", type=int, default=32768)
    parser.add_argument("--max-out-tokens-judge", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--engine-kwargs",
        default="{}",
        help="JSON dict of engine kwargs passed to OpenJury (vLLM settings).",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional raw args appended to estimate_elo_ratings.py.",
    )
    parser.add_argument("--partition", default="capella", help="Slurm partition.")
    parser.add_argument("--time", default="04:00:00", help="Slurm wall time.")
    parser.add_argument("--gres", default="gpu:1", help="Slurm gres.")
    parser.add_argument("--cpus-per-task", type=int, default=8, help="Slurm CPUs per task.")
    parser.add_argument("--mem", default="64G", help="Slurm memory.")
    parser.add_argument("--exclusive", action="store_true", help="Request exclusive node.")
    parser.add_argument("--no-exclusive", dest="exclusive", action="store_false")
    parser.set_defaults(exclusive=True)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Directory for Slurm logs.")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR, help="Directory for result outputs.")
    parser.add_argument("--venv-activate", default=DEFAULT_VENV_ACTIVATE, help="Path to venv activate script.")
    parser.add_argument("--openjury-dir", default=DEFAULT_OPENJURY_DIR, help="Path to OpenJury repo.")
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME, help="HF cache home.")
    parser.add_argument("--hf-datasets-cache", default=DEFAULT_HF_DATASETS_CACHE, help="HF datasets cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch scripts without submitting.")
    return parser.parse_args()


def read_models_list_from_file(path: str) -> List[str]:
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


def resolve_models_arg(values: List[str]) -> List[str]:
    if len(values) == 1 and values[0].endswith(".txt"):
        return read_models_list_from_file(values[0])

    models: List[str] = []
    seen = set()
    for name in values:
        if name in seen:
            continue
        seen.add(name)
        models.append(name)
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


def resolve_model_path(api_config: Dict[str, dict], name: str) -> str | None:
    entry = api_config.get(name)
    if entry is None or not isinstance(entry, dict):
        return None
    model_path = entry.get("model")
    return model_path


def vllm_name(model_path: str) -> str:
    if model_path.startswith("/"):
        return f"VLLM{model_path}"
    return f"VLLM/{model_path}"


def build_sbatch_script(
    *,
    model_name: str,
    model_path: str,
    judge_name: str,
    judge_path: str,
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

    try:
        json.loads(args.engine_kwargs)
    except Exception as exc:
        raise ValueError(f"--engine-kwargs must be valid JSON: {exc}")

    body = SBATCH_BODY.format(
        venv_activate=args.venv_activate,
        hf_home=args.hf_home,
        hf_datasets_cache=args.hf_datasets_cache,
        openjury_dir=args.openjury_dir,
        arena=args.arena,
    model_name=vllm_name(model_path),
    judge_name=vllm_name(judge_path),
        n_instructions=args.n_instructions,
        swap_mode=args.swap_mode,
        truncate_all_input_chars=args.truncate_all_input_chars,
        max_out_tokens_models=args.max_out_tokens_models,
        max_out_tokens_judge=args.max_out_tokens_judge,
        result_dir=args.result_dir,
        engine_kwargs=args.engine_kwargs.replace("'", "\\'"),
        n_instructions_per_language_flag=(
            f" \\\n+  --n_instructions_per_language {args.n_instructions_per_language}"
            if args.n_instructions_per_language is not None
            else ""
        ),
        max_model_len_flag=(
            f" \\\n+  --max_model_len {args.max_model_len}"
            if args.max_model_len is not None
            else ""
        ),
        extra_args=(f" {args.extra_args}" if args.extra_args else ""),
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
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "(no stderr captured)"
        stdout = exc.stdout.strip() if exc.stdout else "(no stdout captured)"
        print("sbatch failed with non-zero exit status.")
        print(f"stdout: {stdout}")
        print(f"stderr: {stderr}")
        print(f"Script path: {temp_path}")
        raise
    finally:
        os.unlink(temp_path)


def main() -> int:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    models = resolve_models_arg(args.models)
    if not models:
        print("No models found in list.")
        return 1

    api_config = load_api_config(args.api_config)
    judge_path = resolve_model_path(api_config, args.judge_model)
    if not judge_path:
        print(f"Judge model '{args.judge_model}' not found or missing 'model' path.")
        return 1

    missing: List[str] = []
    skipped: List[str] = []

    for model_name in models:
        model_path = resolve_model_path(api_config, model_name)
        if not model_path:
            missing.append(model_name)
            continue

        try:
            script_contents = build_sbatch_script(
                model_name=model_name,
                model_path=model_path,
                judge_name=args.judge_model,
                judge_path=judge_path,
                args=args,
            )
        except ValueError as exc:
            print(str(exc))
            return 1

        submit_job(script_contents, args.dry_run)

    if missing:
        print("\nMissing models in api_config:")
        for name in missing:
            print(f"  - {name}")

    if skipped:
        print("\nSkipped models:")
        for name in skipped:
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
