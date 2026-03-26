#!/usr/bin/env python3
"""Submit IFBench jobs for models listed in a text file.

This script generates IFBench responses using a vLLM server (GPU) and then
runs IFBench evaluation, similar in spirit to automate_arena_hard_generation.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional
from urllib.parse import urlparse

import yaml

DEFAULT_API_CONFIG = "/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml"
DEFAULT_VENV_ACTIVATE = "/data/horse/ws/hama901h-BFTranslation/venv-lm-eval/bin/activate"
DEFAULT_IFBENCH_DIR = "/data/horse/ws/hama901h-BFTranslation/IFBench"
DEFAULT_INPUT_DATA = "/data/horse/ws/hama901h-BFTranslation/IFBench/data/IFBench_test.jsonl"
DEFAULT_RESPONSE_DIR = "/data/horse/ws/hama901h-BFTranslation/IFBench/outputs"
DEFAULT_RESPONSE_SUFFIX = ".jsonl"
DEFAULT_LOG_DIR = "/data/horse/ws/hama901h-BFTranslation/logs/IFBench"
DEFAULT_OUTPUT_DIR = "/data/horse/ws/hama901h-BFTranslation/evaluation_results/ifbench"
DEFAULT_HF_HOME = "/data/cat/ws/hama901h-Posttraining/.cache"
DEFAULT_HF_DATASETS_CACHE = "/data/cat/ws/hama901h-Posttraining/.cache"

DEFAULT_GEN_TEMPERATURE = 0.0
DEFAULT_GEN_MAX_TOKENS = 512
DEFAULT_GEN_PROMPT_BUFFER = 512
DEFAULT_GEN_SEED = 42
DEFAULT_GEN_WORKERS = 8

DEFAULT_VLLM_TENSOR_PARALLEL = 1
DEFAULT_VLLM_GPU_UTIL = 0.9
DEFAULT_VLLM_MAX_MODEL_LEN = 4096

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
set -e

echo "JOB NAME" $SLURM_JOB_NAME

module load CUDA
source {venv_activate}

export HF_HOME="{hf_home}"
export HF_DATASETS_CACHE="{hf_datasets_cache}"

cd {ifbench_dir}

mkdir -p {output_dir}

{generate_block}

python3 -m run_eval \
    --input_data={input_data} \
    --input_response_data={input_response_data} \
    --output_dir={output_dir}

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit IFBench jobs for a list of model names.")
    parser.add_argument("--models-file", required=True, help="Path to text file with model names.")
    parser.add_argument("--api-config", default=DEFAULT_API_CONFIG, help="Path to api_config.yaml.")
    parser.add_argument("--api-endpoint-index", type=int, default=0)
    parser.add_argument("--api-model-field", default="name")
    parser.add_argument("--api-model-override", default=None)
    parser.add_argument("--job-name-prefix", default="ifbench_")
    parser.add_argument("--partition", default="capella")
    parser.add_argument("--time", default="00:45:00")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--exclusive", action="store_true")
    parser.add_argument("--no-exclusive", dest="exclusive", action="store_false")
    parser.set_defaults(exclusive=False)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--venv-activate", default=DEFAULT_VENV_ACTIVATE)
    parser.add_argument("--ifbench-dir", default=DEFAULT_IFBENCH_DIR)
    parser.add_argument("--input-data", default=DEFAULT_INPUT_DATA)
    parser.add_argument("--response-dir", default=DEFAULT_RESPONSE_DIR)
    parser.add_argument("--response-suffix", default=DEFAULT_RESPONSE_SUFFIX)
    parser.add_argument(
        "--response-template",
        default="{response_dir}/{model_name}{response_suffix}",
    )
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument("--hf-datasets-cache", default=DEFAULT_HF_DATASETS_CACHE)
    parser.add_argument("--gen-temperature", type=float, default=DEFAULT_GEN_TEMPERATURE)
    parser.add_argument("--gen-max-tokens", type=int, default=DEFAULT_GEN_MAX_TOKENS)
    parser.add_argument(
        "--gen-prompt-buffer",
        type=int,
        default=DEFAULT_GEN_PROMPT_BUFFER,
        help="Reserve this many tokens for the prompt when computing max_tokens.",
    )
    parser.add_argument("--gen-seed", type=int, default=DEFAULT_GEN_SEED)
    parser.add_argument("--gen-workers", type=int, default=DEFAULT_GEN_WORKERS)
    parser.add_argument("--gen-resume", action="store_true")
    parser.add_argument("--no-gen-resume", dest="gen_resume", action="store_false")
    parser.set_defaults(gen_resume=True)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=DEFAULT_VLLM_TENSOR_PARALLEL)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=DEFAULT_VLLM_GPU_UTIL)
    parser.add_argument("--vllm-max-model-len", type=int, default=DEFAULT_VLLM_MAX_MODEL_LEN)
    parser.add_argument("--vllm-extra-args", default="")
    parser.add_argument(
        "--vllm-chat-template",
        default=None,
        help="Path to a chat template Jinja file for vLLM.",
    )
    parser.add_argument("--dry-run", action="store_true")
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


def parse_port_from_api_base(api_base_url: Optional[str]) -> Optional[int]:
    if not api_base_url:
        return None
    try:
        parsed = urlparse(api_base_url)
        if parsed.port:
            return parsed.port
        if parsed.netloc and ":" in parsed.netloc:
            return int(parsed.netloc.split(":")[-1])
    except Exception:
        return None
    return None


def get_endpoint(entry: dict, index: int) -> Optional[dict]:
    endpoints = entry.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        return None
    if index < 0 or index >= len(endpoints):
        return None
    endpoint = endpoints[index]
    if not isinstance(endpoint, dict):
        return None
    return endpoint


def build_response_path(model_name: str, args: argparse.Namespace) -> str:
    return args.response_template.format(
        response_dir=args.response_dir,
        model_name=model_name,
        response_suffix=args.response_suffix,
    )


def build_vllm_block(*, model_path: str, served_model_name: str, port: int, log_path: str, args: argparse.Namespace) -> str:
    extra_args = args.vllm_extra_args.strip()
    extra_args = f" {extra_args}" if extra_args else ""
    chat_template_arg = f" --chat-template {args.vllm_chat_template}" if args.vllm_chat_template else ""
    return f"""echo \"Starting vLLM server on GPU 0 (port {port})...\"
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \\
        --model \"{model_path}\" \\
        --port {port} \\
        --tensor-parallel-size {args.vllm_tensor_parallel_size} \\
        --gpu-memory-utilization {args.vllm_gpu_memory_utilization} \\
        --max-model-len {args.vllm_max_model_len} \\
        --served-model-name \"{served_model_name}\"{chat_template_arg}{extra_args} \\
        > {log_path} 2>&1 &
VLLM_PID=$!
sleep 5
if ! kill -0 $VLLM_PID > /dev/null 2>&1; then
    echo \"ERROR: vLLM server failed to start. Check log: {log_path}\"
    tail -n 100 {log_path}
    exit 1
fi
echo \"Waiting for vLLM health endpoint...\"
MAX_WAIT=1200
ELAPSED=0
SLEEP_INTERVAL=10
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:{port}/health > /dev/null 2>&1; then
        echo \"vLLM is ready after $ELAPSED seconds\"
        break
    fi
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
    if ! kill -0 $VLLM_PID > /dev/null 2>&1; then
        echo \"ERROR: vLLM process died.\"
        tail -n 100 {log_path}
        exit 1
    fi
done
if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo \"ERROR: vLLM failed to become ready.\"
    tail -n 100 {log_path}
    kill $VLLM_PID
    exit 1
fi
"""


def response_file_has_non_empty_outputs(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                response = payload.get("response")
                if isinstance(response, str) and response.strip():
                    return True
    except FileNotFoundError:
        return False
    except json.JSONDecodeError:
        return True
    return False


def build_sbatch_script(*, model_name: str, response_path: str, output_dir: str, generate_block: str, args: argparse.Namespace) -> str:
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
    body = SBATCH_BODY.format(
        venv_activate=args.venv_activate,
        hf_home=args.hf_home,
        hf_datasets_cache=args.hf_datasets_cache,
        ifbench_dir=args.ifbench_dir,
        input_data=args.input_data,
        input_response_data=response_path,
        output_dir=output_dir,
        generate_block=generate_block,
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
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.response_dir, exist_ok=True)

    models = read_models_list(args.models_file)
    if not models:
        print("No models found in list.")
        return 1

    api_config = load_api_config(args.api_config)

    skipped: List[str] = []

    for model_name in models:
        entry = api_config.get(model_name)
        if entry is None:
            skipped.append(model_name)
            print(f"Skipping {model_name}: missing in api_config.yaml")
            continue
        if not isinstance(entry, dict):
            skipped.append(model_name)
            print(f"Skipping {model_name}: config entry is not a mapping.")
            continue

        endpoint = get_endpoint(entry, args.api_endpoint_index)
        if endpoint is None:
            skipped.append(model_name)
            print(f"Skipping {model_name}: endpoint {args.api_endpoint_index} not found.")
            continue

        api_base = endpoint.get("api_base")
        api_key = endpoint.get("api_key")
        if not api_base:
            skipped.append(model_name)
            print(f"Skipping {model_name}: endpoint missing api_base.")
            continue

        model_path = entry.get("model")
        if not model_path:
            skipped.append(model_name)
            print(f"Skipping {model_name}: no 'model' path in api_config.")
            continue

        api_model = args.api_model_override
        if not api_model:
            if args.api_model_field == "name":
                api_model = model_name
            else:
                api_model = entry.get(args.api_model_field) or model_name

        port = parse_port_from_api_base(api_base) or 8000
        api_base = f"http://localhost:{port}/v1"

        response_path = build_response_path(model_name, args)
        response_dir = os.path.dirname(response_path)
        response_dir_line = f"mkdir -p {response_dir}" if response_dir else ""
        resume_flag = "--resume" if args.gen_resume else ""
        api_key_flag = f"--api-key {api_key}" if api_key and api_key != "-" else ""
        effective_max_tokens = min(
            args.gen_max_tokens,
            max(1, args.vllm_max_model_len - args.gen_prompt_buffer),
        )

        if args.gen_resume and os.path.exists(response_path):
            if not response_file_has_non_empty_outputs(response_path):
                os.remove(response_path)
                resume_flag = ""

        vllm_log = os.path.join(args.log_dir, f"{sanitize_job_name(model_name)}_vllm.log")
        vllm_block = build_vllm_block(
            model_path=model_path,
            served_model_name=api_model,
            port=port,
            log_path=vllm_log,
            args=args,
        )

        generate_block = (
            f"{response_dir_line}\n"
            f"{vllm_block}"
            "python3 -m generate_responses \\\n    --api-base {api_base} \\\n    --model {api_model} \\\n    --input-file {input_data} \\\n    --output-file {output_file} \\\n    --temperature {temperature} \\\n    --max-tokens {max_tokens} \\\n    --workers {workers} \\\n    --seed {seed} {resume_flag} {api_key_flag}\n"
            "GEN_EXIT_CODE=$?\n"
            "if [ $GEN_EXIT_CODE -ne 0 ]; then\n"
            "  echo \"Generation failed with exit code $GEN_EXIT_CODE\"\n"
            "  exit $GEN_EXIT_CODE\n"
            "fi\n"
            "if [ -n \"${{VLLM_PID:-}}\" ]; then\n"
            "  echo \"Stopping vLLM server (PID: $VLLM_PID)\"\n"
            "  kill $VLLM_PID\n"
            "  sleep 5\n"
            "fi\n"
        ).format(
            api_base=api_base,
            api_model=api_model,
            input_data=args.input_data,
            output_file=response_path,
            temperature=args.gen_temperature,
            max_tokens=effective_max_tokens,
            workers=args.gen_workers,
            seed=args.gen_seed,
            resume_flag=resume_flag,
            api_key_flag=api_key_flag,
        )

        model_output_dir = os.path.join(args.output_dir, sanitize_job_name(model_name))
        script_contents = build_sbatch_script(
            model_name=model_name,
            response_path=response_path,
            output_dir=model_output_dir,
            generate_block=generate_block,
            args=args,
        )
        submit_job(script_contents, args.dry_run)

    if skipped:
        print("\nSkipped models:")
        for name in skipped:
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
