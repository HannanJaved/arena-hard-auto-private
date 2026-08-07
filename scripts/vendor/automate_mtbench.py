#!/usr/bin/env python3
"""
Automation script to submit MT-Bench evaluation jobs for JudgeArena.

This version mirrors the vLLM server workflow used by arena-hard and alpaca-eval:
- Starts a vLLM OpenAI-compatible API server for the *judge* model.
- Runs JudgeArena MT-Bench with model_A/model_B using in-process providers
  (e.g., VLLM/...) and the judge via ChatOpenAI + local base URL.

This avoids in-process vLLM CUDA-graph issues for large Mamba models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
JUDGEARENA_DIR = f"{WORKSPACE_ROOT}/JudgeArena"
DEFAULT_API_CONFIG = f"{WORKSPACE_ROOT}/arena-hard-auto/config/api_config.yaml"
DEFAULT_LOGS_DIR = f"{WORKSPACE_ROOT}/logs/judgearena-mtbench"
DEFAULT_SCRIPTS_DIR = f"{WORKSPACE_ROOT}/generated_sbatch_jobs"
DEFAULT_RESULT_FOLDER = f"{WORKSPACE_ROOT}/evaluation_results/judgearena-mtbench"
DEFAULT_MODELS_FILE = f"{WORKSPACE_ROOT}/mtbench_models_to_test.txt"

# Defaults
DEFAULT_JUDGE = "Qwen3-Next-80B-A3B-Instruct-FP8"
DEFAULT_SWAP_MODE = "fixed"
DEFAULT_PARTITION = "capella"
DEFAULT_TIME = "04:00:00"
DEFAULT_GPUS = 2
DEFAULT_CPUS = 4
DEFAULT_MEM = "128G"
DEFAULT_MAX_OUT_TOKENS_MODELS = 4096
DEFAULT_MAX_OUT_TOKENS_JUDGE = 2048
DEFAULT_TRUNCATE_INPUT_CHARS = 8192
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_PROVIDER_PREFIX = "VLLM"
DEFAULT_ENGINE_KWARGS = {
    "tensor_parallel_size": 1,
    "max_batch_size": 64,
    "max_num_seqs": 64,
    "max_num_batched_tokens": 65536,
}
DEFAULT_PYTHON_EXEC = f"{WORKSPACE_ROOT}/venv-openjury/bin/python"
DEFAULT_ACTIVATE = f"{WORKSPACE_ROOT}/venv-openjury/bin/activate"
DEFAULT_VLLM_EXEC = f"{WORKSPACE_ROOT}/arena-hard-auto/venv/bin/python"

# Judge server defaults (vLLM OpenAI server)
DEFAULT_JUDGE_SERVER_PORT = 8010
DEFAULT_JUDGE_SERVER_TP = 1
DEFAULT_JUDGE_SERVER_MAX_MODEL_LEN = 26304
DEFAULT_JUDGE_SERVER_MAX_NUM_SEQS = 32
DEFAULT_JUDGE_SERVER_MAX_NUM_BATCHED_TOKENS = 16384
DEFAULT_JUDGE_SERVER_CUDAGRAPH_MODE = "FULL"


@dataclass(frozen=True)
class ModelPair:
    model_a: str
    model_b: str


def load_api_config(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"api_config.yaml should be a mapping, got {type(data)}")
    return data


def get_model_path(api_config: dict, model_key: str) -> str:
    entry = api_config.get(model_key)
    if entry is None:
        raise ValueError(f"Model key '{model_key}' not found in api_config.yaml.")
    if isinstance(entry, dict) and "model" in entry:
        return entry["model"]
    raise ValueError(f"No 'model' field found for key '{model_key}' in api_config.yaml.")


def resolve_model_path(api_config: dict, model_key: str) -> str:
    if "/" in model_key:
        return model_key
    return get_model_path(api_config, model_key)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def safe_script_name(prefix: str, model_a: str, model_b: str, max_len: int = 180) -> str:
    base = safe_name(f"{prefix}_{model_a}_vs_{model_b}")
    if len(base) <= max_len:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    head = base[: max(0, max_len - 11)]
    return f"{head}-{digest}"


def read_lines(path: str) -> list[str]:
    lines = []
    try:
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if line and not line.startswith("#"):
                    lines.append(line)
    except FileNotFoundError:
        print(f"Model list file not found: {path}")
    return lines


def parse_pairs_file(path: str) -> list[ModelPair]:
    pairs = []
    for line in read_lines(path):
        if "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in line.split() if p.strip()]
        if len(parts) < 2:
            print(f"Skipping invalid pair line: {line}")
            continue
        pairs.append(ModelPair(parts[0], parts[1]))
    return pairs


def build_engine_kwargs(model_tp_size: int, gpu_memory_utilization: float | None, extra: str) -> dict:
    engine_kwargs = dict(DEFAULT_ENGINE_KWARGS)
    engine_kwargs["tensor_parallel_size"] = model_tp_size
    if gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if extra:
        extra_kwargs = json.loads(extra)
        if not isinstance(extra_kwargs, dict):
            raise ValueError("--engine-kwargs must be a JSON object")
        engine_kwargs.update(extra_kwargs)
    if "max_batch_size" in engine_kwargs and "max_num_seqs" not in engine_kwargs:
        engine_kwargs["max_num_seqs"] = engine_kwargs["max_batch_size"]
    return engine_kwargs


def resolve_model_spec(api_config: dict, model_key: str, provider_prefix: str) -> str:
    if "/" in model_key:
        return model_key
    model_path = get_model_path(api_config, model_key)
    if provider_prefix:
        if model_path.startswith(f"{provider_prefix}/"):
            return model_path
        # model_path may be an absolute filesystem path (starts with "/");
        # use "VLLM:" style separation won't work — just concatenate carefully.
        return f"{provider_prefix}/{model_path}"
    return model_path


def build_mtbench_result_name(
    dataset: str,
    model_a: str,
    model_b: str,
    judge_model: str,
    swap_mode: str,
    suffix: str | None = None,
) -> str:
    name = f"{dataset}-{model_a}-{model_b}-{judge_model}-{swap_mode}"
    if suffix:
        name += f"-{suffix}"
    return name.replace("/", "_")


def safe_result_name(name: str, max_len: int = 160) -> str:
    if len(name) <= max_len:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    head = name[: max(0, max_len - 13)]
    return f"{head}-{digest}"


def find_existing_result_folder(
    result_folder: str,
    dataset: str,
    model_a: str,
    model_b: str,
    judge_model: str,
    swap_mode: str,
) -> Path | None:
    name = build_mtbench_result_name(dataset, model_a, model_b, judge_model, swap_mode)
    folder = Path(result_folder) / safe_result_name(name)
    if folder.exists():
        return folder
    return None


def create_slurm_script(
    *,
    script_path: str,
    model_a: str,
    model_b: str,
    judge_model: str,
    judge_model_path: str,
    judge_server_port: int,
    judge_server_tp: int,
    judge_server_max_model_len: int,
    judge_server_max_num_seqs: int,
    judge_server_max_num_batched_tokens: int,
    judge_server_cudagraph_mode: str,
    dataset: str,
    n_instructions: int | None,
    swap_mode: str,
    provide_explanation: bool,
    ignore_cache: bool,
    use_tqdm: bool,
    result_folder: str,
    truncate_all_input_chars: int,
    max_out_tokens_models: int,
    max_out_tokens_judge: int,
    max_model_len: int | None,
    chat_template: str | None,
    engine_kwargs: dict,
    num_gpus: int,
    cpus: int,
    mem: str,
    partition: str,
    time_limit: str,
    python_exec: str,
    vllm_exec: str,
    activate_path: str | None,
    logs_dir: str,
):
    job_name = f"mtbench-{safe_name(model_a)[:30]}"
    log_dir = Path(logs_dir) / safe_name(model_a)
    log_dir.mkdir(parents=True, exist_ok=True)

    optional_flags = []
    if n_instructions is not None:
        optional_flags.append(f"    --n_instructions {n_instructions} \\")
    if provide_explanation:
        optional_flags.append("    --provide_explanation \\")
    if ignore_cache:
        optional_flags.append("    --ignore_cache \\")
    if max_model_len is not None:
        optional_flags.append(f"    --max_model_len {max_model_len} \\")
    if chat_template:
        optional_flags.append(f"    --chat_template {shlex.quote(chat_template)} \\")
    # use_tqdm=True serializes vLLM inference (one request at a time via ainvoke);
    # always omit it so the batch path is used.

    optional_flags_block = "\n".join(optional_flags)
    if optional_flags_block:
        optional_flags_block += "\n"

    engine_kwargs_json = json.dumps(engine_kwargs)
    compilation_config_json = json.dumps({"cudagraph_mode": judge_server_cudagraph_mode})

    activate_line = ""
    if activate_path:
        activate_line = f"source {activate_path}\n"

    judge_served_name = safe_name(judge_model)

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --error={log_dir}/mtbench_%j.err
#SBATCH --output={log_dir}/mtbench_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{num_gpus}

set -e

echo "=== MT-Bench evaluation ==="
echo "  Model A : {model_a}"
echo "  Model B : {model_b}"
echo "  Judge   : {judge_model}"
echo "  Dataset : {dataset}"
echo "  Started : $(date)"

# --- ENVIRONMENT ---
{activate_line}PYTHON_EXEC={python_exec}
VLLM_EXEC={vllm_exec}
export PATH=$(dirname {vllm_exec}):$PATH
module load CUDA

echo "Python: $PYTHON_EXEC"

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(({num_gpus}-1)))
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_MOE_FP8_BACKEND=TRITON
# Throttle FlashInfer/nvcc JIT compilation so concurrent cicc processes
# don't exhaust host RAM and get OOM-killed (signal 9), which previously
# left the judge server unable to start.
export MAX_JOBS=2
export NVCC_THREADS=1

source {WORKSPACE_ROOT}/cache.sh

export PYTHONPATH={JUDGEARENA_DIR}:$PYTHONPATH
export PYTHONUNBUFFERED=1
export JUDGEARENA_DATA={WORKSPACE_ROOT}/judgearena-data

# Start vLLM OpenAI server for the judge
JUDGE_PORT={judge_server_port}
JUDGE_MODEL_PATH="{judge_model_path}"
JUDGE_SERVED_NAME="{judge_served_name}"
JUDGE_LOG="{log_dir}/judge_vllm_${{SLURM_JOB_ID}}.log"

# Prefer an explicit chat_template file from the judge checkpoint; otherwise
# let vLLM use the tokenizer_config chat template.
CHAT_TEMPLATE_FLAG=""
if [ -f "$JUDGE_MODEL_PATH/chat_template.jinja" ]; then
    CHAT_TEMPLATE_FLAG="--chat-template $JUDGE_MODEL_PATH/chat_template.jinja"
elif [ -f "$JUDGE_MODEL_PATH/chat_template.j2" ]; then
    CHAT_TEMPLATE_FLAG="--chat-template $JUDGE_MODEL_PATH/chat_template.j2"
fi
echo "Judge chat template: ${{CHAT_TEMPLATE_FLAG:-tokenizer_config (auto)}}"

CUDA_VISIBLE_DEVICES=0 $VLLM_EXEC -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL_PATH" \
    --port $JUDGE_PORT \
    --tensor-parallel-size {judge_server_tp} \
    --served-model-name "$JUDGE_SERVED_NAME" \
    --max-model-len {judge_server_max_model_len} \
    --max-num-seqs {judge_server_max_num_seqs} \
    --max-num-batched-tokens {judge_server_max_num_batched_tokens} \
    --compilation-config '{compilation_config_json}' \
    $CHAT_TEMPLATE_FLAG \
    > "$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!

sleep 5
if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
    echo "ERROR: Judge server failed to start. Check log: $JUDGE_LOG"
    tail -n 50 "$JUDGE_LOG"
    exit 1
fi

echo "Waiting for judge server to become ready..."
MAX_WAIT=3600
ELAPSED=0
SLEEP_INTERVAL=30
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$JUDGE_PORT/health > /dev/null 2>&1; then
        echo "Judge server is ready after $ELAPSED seconds."
        break
    fi
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
    if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
        echo "ERROR: Judge server process died."
        tail -n 50 "$JUDGE_LOG"
        exit 1
    fi
    echo "Still waiting... ($ELAPSED / $MAX_WAIT)"
 done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Judge server not ready within $MAX_WAIT seconds"
    tail -n 100 "$JUDGE_LOG"
    kill $JUDGE_PID
    exit 1
fi

sleep 10

export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE="http://localhost:$JUDGE_PORT/v1"
export OPENAI_BASE_URL="http://localhost:$JUDGE_PORT/v1"

cd {JUDGEARENA_DIR}

CUDA_VISIBLE_DEVICES=1 $PYTHON_EXEC -m judgearena.generate_and_evaluate \
    --dataset {dataset} \
    --model_A {shlex.quote(model_a)} \
    --model_B {shlex.quote(model_b)} \
    --judge {shlex.quote(f"ChatOpenAI/{judge_served_name}")} \
    --swap_mode {swap_mode} \
    --truncate_all_input_chars {truncate_all_input_chars} \
    --max_out_tokens_models {max_out_tokens_models} \
    --max_out_tokens_judge {max_out_tokens_judge} \
    --result_folder {shlex.quote(result_folder)} \
{optional_flags_block}    --engine_kwargs {shlex.quote(engine_kwargs_json)}

echo "Stopping judge server (PID: $JUDGE_PID)..."
kill $JUDGE_PID
sleep 10

echo "Done: $(date)"
"""

    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit SLURM jobs for JudgeArena MT-Bench evaluations."
    )

    parser.add_argument(
        "--api-config",
        type=str,
        default=DEFAULT_API_CONFIG,
        help=f"Path to api_config.yaml (default: {DEFAULT_API_CONFIG}).",
    )
    parser.add_argument(
        "--provider-prefix",
        type=str,
        default=DEFAULT_PROVIDER_PREFIX,
        help="Provider prefix to prepend to model paths from api_config (default: VLLM).",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Model keys (or provider-prefixed strings) to evaluate as Model A.",
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default=DEFAULT_MODELS_FILE,
        help=f"File containing model keys (default: {DEFAULT_MODELS_FILE}).",
    )
    parser.add_argument(
        "--pairs-file",
        type=str,
        default=None,
        help="Optional file of explicit model pairs (model_a, model_b per line).",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        help="Baseline model key (or provider-prefixed string) to use as Model B.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=DEFAULT_JUDGE,
        help=f"Judge model key (default: {DEFAULT_JUDGE}).",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mt-bench",
        help="Dataset name (default: mt-bench).",
    )
    parser.add_argument(
        "--swap_mode",
        choices=["fixed", "both"],
        default=DEFAULT_SWAP_MODE,
        help=f"Swap mode (default: {DEFAULT_SWAP_MODE}).",
    )
    parser.add_argument(
        "--n_instructions",
        type=int,
        default=None,
        help="Number of MT-Bench questions to evaluate (default: all).",
    )
    parser.add_argument(
        "--provide_explanation",
        action="store_true",
        help="Ask judge to provide explanation before scoring.",
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="Ignore cached completions and rerun everything.",
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",
        help="Use tqdm progress bars (may not work with some providers).",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default=DEFAULT_RESULT_FOLDER,
        help=f"Folder to save results (default: {DEFAULT_RESULT_FOLDER}).",
    )
    parser.add_argument(
        "--truncate_all_input_chars",
        type=int,
        default=DEFAULT_TRUNCATE_INPUT_CHARS,
        help=f"Character-level truncation (default: {DEFAULT_TRUNCATE_INPUT_CHARS}).",
    )
    parser.add_argument(
        "--max_out_tokens_models",
        type=int,
        default=DEFAULT_MAX_OUT_TOKENS_MODELS,
        help=f"Max output tokens for models (default: {DEFAULT_MAX_OUT_TOKENS_MODELS}).",
    )
    parser.add_argument(
        "--max_out_tokens_judge",
        type=int,
        default=DEFAULT_MAX_OUT_TOKENS_JUDGE,
        help=f"Max output tokens for judge (default: {DEFAULT_MAX_OUT_TOKENS_JUDGE}).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"VLLM max_model_len (default: {DEFAULT_MAX_MODEL_LEN}).",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Override chat template (Jinja2 string).",
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_GPUS,
        help=f"Number of GPUs per job (default: {DEFAULT_GPUS}).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=DEFAULT_CPUS,
        help=f"CPUs per task (default: {DEFAULT_CPUS}).",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default=DEFAULT_MEM,
        help=f"Memory per task (default: {DEFAULT_MEM}).",
    )
    parser.add_argument(
        "--model-tp-size",
        type=int,
        default=1,
        help="tensor_parallel_size for model generation (default: 1).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional vLLM gpu_memory_utilization for model generation.",
    )
    parser.add_argument(
        "--engine-kwargs",
        type=str,
        default="{}",
        help="Extra engine kwargs as JSON string (merged into defaults).",
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
        dest="time_limit",
        default=DEFAULT_TIME,
        help=f"SLURM time limit (default: {DEFAULT_TIME}).",
    )

    parser.add_argument(
        "--python-exec",
        type=str,
        default=DEFAULT_PYTHON_EXEC,
        help=f"Python executable for judgearena (default: {DEFAULT_PYTHON_EXEC}).",
    )
    parser.add_argument(
        "--vllm-exec",
        type=str,
        default=DEFAULT_VLLM_EXEC,
        help=f"Python executable for the vLLM server (default: {DEFAULT_VLLM_EXEC}).",
    )
    parser.add_argument(
        "--activate",
        type=str,
        default=DEFAULT_ACTIVATE,
        help=f"Optional venv activate script (default: {DEFAULT_ACTIVATE}).",
    )

    parser.add_argument(
        "--scripts-dir",
        type=str,
        default=DEFAULT_SCRIPTS_DIR,
        help=f"Where to write generated sbatch scripts (default: {DEFAULT_SCRIPTS_DIR}).",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=DEFAULT_LOGS_DIR,
        help=f"Where to write job logs (default: {DEFAULT_LOGS_DIR}).",
    )

    parser.add_argument(
        "--judge-server-port",
        type=int,
        default=DEFAULT_JUDGE_SERVER_PORT,
        help=f"Port for judge vLLM server (default: {DEFAULT_JUDGE_SERVER_PORT}).",
    )
    parser.add_argument(
        "--judge-server-tp",
        type=int,
        default=DEFAULT_JUDGE_SERVER_TP,
        help=f"Tensor parallel size for judge server (default: {DEFAULT_JUDGE_SERVER_TP}).",
    )
    parser.add_argument(
        "--judge-server-max-model-len",
        type=int,
        default=DEFAULT_JUDGE_SERVER_MAX_MODEL_LEN,
        help=f"Judge server max_model_len (default: {DEFAULT_JUDGE_SERVER_MAX_MODEL_LEN}).",
    )
    parser.add_argument(
        "--judge-server-max-num-seqs",
        type=int,
        default=DEFAULT_JUDGE_SERVER_MAX_NUM_SEQS,
        help=f"Judge server max_num_seqs (default: {DEFAULT_JUDGE_SERVER_MAX_NUM_SEQS}).",
    )
    parser.add_argument(
        "--judge-server-max-num-batched-tokens",
        type=int,
        default=DEFAULT_JUDGE_SERVER_MAX_NUM_BATCHED_TOKENS,
        help=(
            "Judge server max_num_batched_tokens "
            f"(default: {DEFAULT_JUDGE_SERVER_MAX_NUM_BATCHED_TOKENS})."
        ),
    )
    parser.add_argument(
        "--judge-server-cudagraph-mode",
        type=str,
        default=DEFAULT_JUDGE_SERVER_CUDAGRAPH_MODE,
        help="Judge server cudagraph_mode (default: NONE).",
    )

    parser.add_argument("--dry-run", action="store_true", help="Only generate scripts.")
    parser.add_argument("--submit", action="store_true", help="Submit jobs after generating scripts.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pairs that already have result folders.",
    )
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="Rerun all pairs even if results already exist.",
    )

    args = parser.parse_args()

    Path(args.scripts_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    api_config = load_api_config(args.api_config)

    judge_model_path = resolve_model_path(api_config, args.judge)

    if args.pairs_file:
        raw_pairs = parse_pairs_file(args.pairs_file)
        if not raw_pairs:
            print(f"No valid pairs found in {args.pairs_file}.")
            return
        pairs = raw_pairs
    else:
        if args.models:
            model_keys = args.models
        else:
            model_keys = read_lines(args.models_file)
            if not model_keys:
                print(
                    f"No models found in {args.models_file}. Use --models to specify models directly."
                )
                return
        if not args.baseline_model:
            print("ERROR: --baseline-model is required unless --pairs-file is provided.")
            return
        pairs = [ModelPair(model_key, args.baseline_model) for model_key in model_keys]

    resolved_pairs: list[ModelPair] = []
    for pair in pairs:
        try:
            model_a = resolve_model_spec(api_config, pair.model_a, args.provider_prefix)
            model_b = resolve_model_spec(api_config, pair.model_b, args.provider_prefix)
            resolved_pairs.append(ModelPair(model_a, model_b))
        except ValueError as e:
            print(f"Warning: {e} — skipping pair {pair}")

    if not resolved_pairs:
        print("No valid model pairs to process. Exiting.")
        return

    engine_kwargs = build_engine_kwargs(
        model_tp_size=args.model_tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        extra=args.engine_kwargs,
    )

    existing_pairs: list[tuple[ModelPair, Path]] = []
    missing_pairs: list[ModelPair] = []

    for pair in resolved_pairs:
        existing = find_existing_result_folder(
            result_folder=args.result_folder,
            dataset=args.dataset,
            model_a=pair.model_a,
            model_b=pair.model_b,
            judge_model=args.judge,
            swap_mode=args.swap_mode,
        )
        if existing:
            existing_pairs.append((pair, existing))
        else:
            missing_pairs.append(pair)

    if existing_pairs:
        print(f"Found existing results for {len(existing_pairs)} pair(s):")
        for pair, folder in existing_pairs:
            print(f"  - {pair.model_a} vs {pair.model_b} -> {folder}")
        print()

        if missing_pairs:
            print(f"Pairs without existing results ({len(missing_pairs)}):")
            for pair in missing_pairs:
                print(f"  - {pair.model_a} vs {pair.model_b}")
            print()

        if args.rerun_all:
            pairs_to_run = resolved_pairs
            print(
                f"--rerun-all set: submitting all {len(resolved_pairs)} pair(s).\n"
            )
        elif args.skip_existing:
            pairs_to_run = missing_pairs
            print(
                f"--skip-existing set: submitting {len(missing_pairs)} pair(s).\n"
            )
        else:
            prompt = (
                f"{len(existing_pairs)} pair(s) already have results.\n"
                "  [a] Rerun all pairs\n"
                "  [m] Only run missing pairs\n"
                "  [q] Quit\n"
                "Choice [a/m/q]: "
            )
            while True:
                choice = input(prompt).strip().lower()
                if choice == "a":
                    pairs_to_run = resolved_pairs
                    break
                if choice == "m":
                    if not missing_pairs:
                        print("All pairs already have results. Nothing to do.")
                        return
                    pairs_to_run = missing_pairs
                    break
                if choice == "q":
                    print("Aborted.")
                    return
                print("Please enter 'a', 'm', or 'q'.")
    else:
        pairs_to_run = resolved_pairs

    if not pairs_to_run:
        print("No pairs to run. Exiting.")
        return

    activate_path = args.activate if Path(args.activate).exists() else None

    job_scripts: list[str] = []
    for pair in pairs_to_run:
        safe_pair = safe_script_name("mtbench", pair.model_a, pair.model_b)
        script_path = f"{args.scripts_dir}/{safe_pair}.sh"

        create_slurm_script(
            script_path=script_path,
            model_a=pair.model_a,
            model_b=pair.model_b,
            judge_model=args.judge,
            judge_model_path=judge_model_path,
            judge_server_port=args.judge_server_port,
            judge_server_tp=args.judge_server_tp,
            judge_server_max_model_len=args.judge_server_max_model_len,
            judge_server_max_num_seqs=args.judge_server_max_num_seqs,
            judge_server_max_num_batched_tokens=args.judge_server_max_num_batched_tokens,
            judge_server_cudagraph_mode=args.judge_server_cudagraph_mode,
            dataset=args.dataset,
            n_instructions=args.n_instructions,
            swap_mode=args.swap_mode,
            provide_explanation=args.provide_explanation,
            ignore_cache=args.ignore_cache,
            use_tqdm=args.use_tqdm,
            result_folder=args.result_folder,
            truncate_all_input_chars=args.truncate_all_input_chars,
            max_out_tokens_models=args.max_out_tokens_models,
            max_out_tokens_judge=args.max_out_tokens_judge,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            engine_kwargs=engine_kwargs,
            num_gpus=args.num_gpus,
            cpus=args.cpus,
            mem=args.mem,
            partition=args.partition,
            time_limit=args.time_limit,
            python_exec=args.python_exec,
            vllm_exec=args.vllm_exec,
            activate_path=activate_path,
            logs_dir=args.logs_dir,
        )
        print(f"  Created: {script_path}")
        job_scripts.append(script_path)

    print(f"\nGenerated {len(job_scripts)} scripts in {args.scripts_dir}")

    if args.dry_run:
        print("\nDry run — scripts generated but not submitted. To submit manually:")
        for script in job_scripts:
            print(f"  sbatch {script}")
    elif args.submit:
        print("\nSubmitting jobs...")
        submitted = []
        for script in job_scripts:
            try:
                result = subprocess.run(
                    ["sbatch", script], capture_output=True, text=True, check=True
                )
                job_id = result.stdout.strip().split()[-1]
                submitted.append((script, job_id))
                print(f"  {os.path.basename(script)} -> Job ID: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  Failed to submit {script}: {e}")
        print(f"\nSubmitted {len(submitted)} jobs.")
        print("Monitor with: squeue -u $USER")
    else:
        print("\nTo submit, rerun with --submit. Or manually:")
        for script in job_scripts[:3]:
            print(f"  sbatch {script}")
        if len(job_scripts) > 3:
            print(f"  ... and {len(job_scripts) - 3} more")


if __name__ == "__main__":
    main()
