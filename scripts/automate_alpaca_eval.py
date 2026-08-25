#!/usr/bin/env python3
"""
Automation script to generate AlpacaEval answers using VLLM servers.
Creates individual SLURM jobs for each model.
"""

import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yaml

WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
LOGS_DIR = f"{WORKSPACE_ROOT}/logs/alpaca_eval"
SCRIPTS_DIR = f"{WORKSPACE_ROOT}/generated_alpaca_eval_scripts"
CONFIGS_DIR = f"{WORKSPACE_ROOT}/generated_alpaca_eval_configs"
OUTPUTS_DIR = f"{WORKSPACE_ROOT}/alpaca_eval_outputs"
RUN_SCRIPT = f"{WORKSPACE_ROOT}/arena-hard-auto/scripts/run_alpaca_eval.py"
DEFAULT_PROMPT_TEMPLATE = "{instruction}"
DEFAULT_CHAT_TEMPLATE = f"{WORKSPACE_ROOT}/checkpoints/olmo3_chat_template.jinja"


def load_api_config():
    config_path = f"{WORKSPACE_ROOT}/arena-hard-auto/config/api_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _find_api_base(obj):
    if isinstance(obj, dict):
        if "api_base" in obj and isinstance(obj["api_base"], str):
            return obj["api_base"]
        for v in obj.values():
            found = _find_api_base(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_api_base(item)
            if found:
                return found
    return None


def parse_port_from_api_base(api_base_url):
    if not api_base_url:
        return None
    try:
        parsed = urlparse(api_base_url)
        if parsed.port:
            return parsed.port
        netloc = parsed.netloc
        if ":" in netloc:
            return int(netloc.split(":")[-1])
    except Exception:
        return None
    return None


def get_port_for_model(api_config, model_name=None, default=8000):
    try:
        if model_name and model_name in api_config:
            entry = api_config[model_name]
            api_base = None
            if isinstance(entry, dict) and "api_base" in entry:
                api_base = entry["api_base"]
            else:
                api_base = _find_api_base(entry)
            port = parse_port_from_api_base(api_base)
            if port:
                return port
    except Exception:
        pass

    api_base = _find_api_base(api_config)
    port = parse_port_from_api_base(api_base)
    return port or default


def create_directories():
    for directory in [SCRIPTS_DIR, CONFIGS_DIR, LOGS_DIR, OUTPUTS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


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


def extract_model_details(model_name):
    parts = model_name.split("-")
    rank = None
    alpha = None
    step = None

    for i, part in enumerate(parts):
        if part.startswith("rank"):
            rank = part
        elif part.startswith("alpha"):
            alpha = f"{part}-{parts[i+1]}" if i + 1 < len(parts) else part
        elif part.startswith("step") or part == "final":
            step = part

    return rank, alpha, step


def resolve_model_path(model_config):
    if isinstance(model_config, dict):
        if "model" in model_config:
            return model_config["model"]
        if "model_path" in model_config:
            return model_config["model_path"]
    return None


def create_slurm_script(model_name, model_path, script_path, args, model_port):
    rank, alpha, step = extract_model_details(model_name)
    log_subdir = f"{rank}/{alpha}" if rank and alpha else "misc"
    log_dir = f"{LOGS_DIR}/{log_subdir}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    config_file = f"{CONFIGS_DIR}/openai_config_{model_name}.yaml"
    chat_template_flag = f"--chat-template {args.chat_template}" if args.chat_template else ""
    trust_remote_code_flag = "--trust-remote-code" if args.trust_remote_code else ""
    batch_flag = f"--batch-size {args.batch_size}" if args.batch_size else ""
    num_procs_flag = f"--num-procs {args.num_procs}" if args.num_procs else ""
    max_instances_flag = f"--max-instances {args.max_instances}" if args.max_instances else ""
    requires_chatml_flag = "--requires-chatml" if args.requires_chatml else ""

    script_content = f"""#!/bin/bash
#SBATCH --job-name=alpaca-gen-{model_name}
#SBATCH --error={log_dir}/{model_name}.err
#SBATCH --output={log_dir}/{model_name}.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=capella
#SBATCH --exclude=c80,c81
#SBATCH --gres=gpu:1
#SBATCH --account=p_neurasearch

set -e

echo "Setting up the environment for AlpacaEval generation: {model_name}..."
source {WORKSPACE_ROOT}/venv-alpacaeval/bin/activate

PYTHON_EXEC={WORKSPACE_ROOT}/venv-alpacaeval/bin/python
VLLM_EXEC={WORKSPACE_ROOT}/arena-hard-auto/venv/bin/python
module load CUDA
source {WORKSPACE_ROOT}/cache.sh

MODEL_PORT={model_port}
OPENAI_CONFIG_FILE={config_file}

cat > $OPENAI_CONFIG_FILE <<EOF
default:
  - api_key: "EMPTY"
    base_url: "http://localhost:$MODEL_PORT/v1"
EOF

echo "Starting VLLM server for {model_name} on port $MODEL_PORT..."
CUDA_VISIBLE_DEVICES=0 $VLLM_EXEC -m vllm.entrypoints.openai.api_server \
    --model "{model_path}" \
    --port $MODEL_PORT \
    --tensor-parallel-size 1 \
    --served-model-name "{model_name}" \
    {chat_template_flag} \
    {trust_remote_code_flag} \
    > {log_dir}/{model_name}_vllm_server.log 2>&1 &
MODEL_PID=$!

sleep 5
if ! kill -0 $MODEL_PID > /dev/null 2>&1; then
    echo "ERROR: Model server failed to start."
    tail -n 50 {log_dir}/{model_name}_vllm_server.log
    exit 1
fi

echo "Waiting for model server to become ready..."
MAX_WAIT=2400
ELAPSED=0
SLEEP_INTERVAL=30
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$MODEL_PORT/health > /dev/null 2>&1; then
        echo "Model server is ready after $ELAPSED seconds."
        break
    fi
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
    if ! kill -0 $MODEL_PID > /dev/null 2>&1; then
        echo "ERROR: Model server process died."
        tail -n 50 {log_dir}/{model_name}_vllm_server.log
        exit 1
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Model server failed to become ready within $MAX_WAIT seconds"
    tail -n 100 {log_dir}/{model_name}_vllm_server.log
    kill $MODEL_PID
    exit 1
fi

sleep 10

$PYTHON_EXEC {RUN_SCRIPT} \
    --model-name "{model_name}" \
    --served-model-name "{model_name}" \
    --output-dir "{OUTPUTS_DIR}" \
    --dataset-file "{args.dataset_file}" \
    --dataset-repo "{args.dataset_repo}" \
    --prompt-template "{args.prompt_template}" \
    --openai-config-path "$OPENAI_CONFIG_FILE" \
    --openai-base-url "http://localhost:$MODEL_PORT/v1" \
    --max-new-tokens {args.max_new_tokens} \
    --temperature {args.temperature} \
    --top-p {args.top_p} \
    {batch_flag} \
    {num_procs_flag} \
    {requires_chatml_flag} \
    {max_instances_flag}

echo "Generation complete. Stopping server..."
kill $MODEL_PID
sleep 10
"""

    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)


def main():
    parser = argparse.ArgumentParser(description="Automate AlpacaEval answer generation with VLLM")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to process, or a single .txt file containing model names",
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default=f"{WORKSPACE_ROOT}/alpaca_eval_models_to_test.txt",
        help="File containing list of models to process",
    )
    parser.add_argument("--all", action="store_true", help="Process all tulu3 models from API config")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit jobs")
    parser.add_argument("--submit", action="store_true", help="Submit jobs after generating scripts")

    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--dataset-file", default="alpaca_eval_gpt4_baseline.json")
    parser.add_argument("--dataset-repo", default="tatsu-lab/alpaca_eval")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--num-procs", type=int, default=32,
        help="Concurrent completions requests against the vLLM server (alpaca_eval's "
             "OPENAI_MAX_CONCURRENCY otherwise defaults to 5, which leaves the server "
             "mostly idle). Default: 32.",
    )
    parser.add_argument("--requires-chatml", action="store_true")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--chat-template", default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()

    create_directories()

    api_config = load_api_config()

    if args.models:
        if len(args.models) == 1 and args.models[0].endswith(".txt"):
            models_list = load_models_from_file(args.models[0])
        else:
            models_list = args.models
        models_to_process = {m: api_config[m] for m in models_list if m in api_config}
        for model in models_list:
            if model not in api_config:
                print(f"Warning: Model '{model}' not found in API config")
    elif args.all:
        models_to_process = {k: v for k, v in api_config.items() if k.startswith("tulu3-") and "alpha" in k}
    else:
        models_list = load_models_from_file(args.models_file)
        if not models_list:
            print(f"No models found in {args.models_file}. Use --all or --models to specify models.")
            return
        models_to_process = {m: api_config[m] for m in models_list if m in api_config}
        for model in models_list:
            if model not in api_config:
                print(f"Warning: Model '{model}' not found in API config")

    if not models_to_process:
        print("No models to process. Exiting.")
        return

    print(f"Found {len(models_to_process)} models to process:")
    for model_name in models_to_process.keys():
        print(f"  - {model_name}")

    job_scripts = []
    for model_name, model_config in models_to_process.items():
        print(f"\nProcessing {model_name}...")
        model_path = resolve_model_path(model_config)
        if not model_path:
            print(f"  Skipping {model_name}: missing model path in api_config")
            continue

        script_path = f"{SCRIPTS_DIR}/run_alpaca_eval_generation_{model_name}.sh"
        model_port = get_port_for_model(api_config, model_name=model_name, default=8000)
        create_slurm_script(model_name, model_path, script_path, args, model_port)
        print(f"  Created script: {script_path}")

        job_scripts.append(script_path)

    print(f"\nGenerated {len(job_scripts)} job scripts in {SCRIPTS_DIR}")

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
