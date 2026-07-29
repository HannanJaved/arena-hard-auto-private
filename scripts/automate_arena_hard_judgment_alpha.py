#!/usr/bin/env python3
"""
Automation script to generate Arena Hard judgments for multiple models,
running the judge server on the `alpha` partition (A100 40GB) instead of
`capella` (H100). The judge model (~80B params, FP8) does not fit on a
single A100 40GB, so the judge server is sharded across multiple GPUs with
vLLM tensor parallelism (--tensor-parallel-size, default 4 -> ~21GB
weights/GPU, leaving headroom for KV cache).

This is a variant of automate_arena_hard_judgment_olmo3.py — see that file
for the capella/H100/TP=1 version. Kept as a separate script per request so
the original is untouched.

Use for Tulu: --chat-template {WORKSPACE_ROOT}/checkpoints/meta-llama/tulu_template.j2 \\
"""

import os
import yaml
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse

# Configuration
WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
ARENA_HARD_AUTO_DIR = f"{WORKSPACE_ROOT}/arena-hard-auto"
LOGS_DIR = f"{WORKSPACE_ROOT}/logs/arena-hard-alpha"
SCRIPTS_DIR = f"{WORKSPACE_ROOT}/generated_judgment_scripts_alpha"
CONFIGS_DIR = f"{WORKSPACE_ROOT}/generated_judgment_configs_alpha"
JUDGE_UTILS_PATH = f"{ARENA_HARD_AUTO_DIR}/utils/judge_utils.py"

# vLLM + client environment. Note: `ah-eval` (used by the original capella
# script) currently has no vllm/openai/tqdm installed and would fail to run
# gen_judgment.py regardless of partition. arena-hard-auto/venv has vllm,
# openai, and tqdm installed and is the same venv submit_evals.py already
# uses for the arena-hard automation scripts, so it's used here instead.
JUDGE_VENV = f"{ARENA_HARD_AUTO_DIR}/venv"

# Judge configuration
JUDGE_MODEL = "neuralmagic-llama3.1-70b-instruct-fp8"
JUDGE_PATH = "/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Meta-Llama-3.1-70B-Instruct-FP8"

# Baseline configurations
BASELINE_CONFIGS = {
    "instruct": "llama3.1-8b-instruct",
    "base": "llama3.1-8b",
    "tulu_finetuned": "llama3.1-8b-TULU",
    "tulu_sft": "llama3.1-8b-TULU-SFT",
    "tulu_dpo": "llama3.1-8b-TULU-DPO"
}

# Default baseline
DEFAULT_BASELINE = "instruct"

# Default tensor-parallel degree for the judge server on A100 40GB.
# 80B-param FP8 judge (~84GB of weight shards on disk) needs TP>=4 to fit:
#   TP=2 -> ~42GB/GPU (does not fit in 40GB)
#   TP=4 -> ~21GB/GPU (fits, leaves ~19GB/GPU for KV cache/activations)
DEFAULT_TP_SIZE = 4


def load_api_config():
    """Load the API configuration file."""
    config_path = f"{ARENA_HARD_AUTO_DIR}/config/api_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _find_api_base(obj):
    """Recursively search the api_config for the first 'api_base' value.

    Returns the URL string or None.
    """
    if isinstance(obj, dict):
        if 'api_base' in obj and isinstance(obj['api_base'], str):
            return obj['api_base']
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
    """Parse and return port (int) from a URL like 'http://localhost:8001/v1'.

    Returns None if not found or cannot parse.
    """
    if not api_base_url:
        return None
    try:
        parsed = urlparse(api_base_url)
        if parsed.port:
            return parsed.port
        netloc = parsed.netloc
        if ':' in netloc:
            return int(netloc.split(':')[-1])
    except Exception:
        return None
    return None


def get_port_for_model(api_config, model_name=None, default=8001):
    """Get port to use for model server.

    If model_name is provided, try to read api_base under that model's config.
    Otherwise return the first api_base port found in the config. If none
    found, return the provided default.
    """
    try:
        if model_name and model_name in api_config:
            entry = api_config[model_name]
            api_base = None
            if isinstance(entry, dict) and 'api_base' in entry:
                api_base = entry['api_base']
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

def load_models_from_file(file_path):
    """Load model names from a text file, ignoring comments and empty lines."""
    models = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    models.append(line)
    except FileNotFoundError:
        print(f"Model list file not found: {file_path}")
    return models

def extract_model_details(model_name):
    """Extract model details from model name for directory structure."""
    # Parse model name like: tulu3-8b-rank64-alpha1e5-001-step48000
    parts = model_name.split('-')

    # Extract rank, alpha, and step information
    rank = None
    alpha = None
    step = None

    for i, part in enumerate(parts):
        if part.startswith('rank'):
            rank = part
        elif part.startswith('alpha'):
            alpha = f"{part}-{parts[i+1]}" if i+1 < len(parts) else part
        elif part.startswith('step') or part == 'final':
            step = part
        elif part == 'default':
            alpha = 'default'
            step = parts[i+1] if i+1 < len(parts) else 'unknown'

    return rank, alpha, step

def resolve_baseline_model(baseline, api_config):
    """Resolve the baseline model name.

    If `baseline` is a key in BASELINE_CONFIGS, return the mapped model name.
    Otherwise, treat `baseline` as a literal api_config model key.
    """
    if baseline in BASELINE_CONFIGS:
        return BASELINE_CONFIGS[baseline]
    if not isinstance(api_config, dict):
        api_config = {}
    if baseline in api_config:
        return baseline
    raise ValueError(
        f"Baseline '{baseline}' is not a known alias in BASELINE_CONFIGS and was not "
        f"found in api_config.yaml. Available aliases: {list(BASELINE_CONFIGS.keys())}. "
        f"Available api_config models (first 10): {list(api_config.keys())[:10]}"
    )


def create_directories():
    """Create necessary directories for scripts, configs, and logs."""
    for directory in [SCRIPTS_DIR, CONFIGS_DIR, LOGS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def update_baseline_in_judge_utils(baseline_model):
    """Update the BASELINE_MODEL in judge_utils.py to match the selected baseline."""
    # Read the current judge_utils.py
    with open(JUDGE_UTILS_PATH, 'r') as f:
        content = f.read()

    # Find and replace the BASELINE_MODEL line
    import re
    pattern = r'BASELINE_MODEL = "[^"]*"'
    replacement = f'BASELINE_MODEL = "{baseline_model}"'

    new_content = re.sub(pattern, replacement, content)

    # Write back to file
    with open(JUDGE_UTILS_PATH, 'w') as f:
        f.write(new_content)

    print(f"Updated BASELINE_MODEL in judge_utils.py to: {baseline_model}")

def extract_judge_path_from_api_config(api_config, model_name):
    """Extract the model path for the judge model from the API config."""
    entry = api_config.get(model_name, {})
    if isinstance(entry, dict):
        if 'model_path' in entry:
            return entry['model_path']
        elif 'model' in entry:
            return entry['model']
    return JUDGE_PATH  # Fallback to default judge path

def create_judgment_config(models_to_judge, output_path, baseline_model, judge_model=JUDGE_MODEL):
    """Create an arena-hard config file for judging specific models."""
    config = {
        'judge_model': judge_model,
        'baseline': baseline_model,
        'temperature': 0.0,
        'max_tokens': 4096,
        'bench_name': 'arena-hard-v2.0',
        'reference': None,
        'regex_patterns': [
            '\\[\\[([AB<>=]+)\\]\\]',
            '\\[([AB<>=]+)\\]'
        ],
        'prompt_template': "<|User Prompt|>\\n{QUESTION}\\n\\n<|The Start of Assistant A's Answer|>\\n{ANSWER_A}\\n<|The End of Assistant A's Answer|>\\n\\n<|The Start of Assistant B's Answer|>\\n{ANSWER_B}\\n<|The End of Assistant B's Answer|>",
        'model_list': models_to_judge
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_judgment_slurm_script(models_to_judge, script_path, config_file_path, baseline_label, baseline_model, judge_model=JUDGE_MODEL, judge_path=JUDGE_PATH, judge_port=8001, tp_size=DEFAULT_TP_SIZE):
    """Create a SLURM script for judging a batch of models on the alpha (A100) partition."""
    # Create a meaningful job name from the first and last model
    if len(models_to_judge) == 1:
        job_name = f"judge-alpha-{models_to_judge[0]}"
        rank, alpha, step = extract_model_details(models_to_judge[0])
    else:
        first_model = models_to_judge[0]
        last_model = models_to_judge[-1]
        rank1, alpha1, step1 = extract_model_details(first_model)
        rank2, alpha2, step2 = extract_model_details(last_model)

        if rank1 == rank2 and alpha1 == alpha2:
            job_name = f"judge-alpha-{rank1}-{alpha1}-{step1}-to-{step2}"
        elif rank1 == rank2:
            job_name = f"judge-alpha-{rank1}-batch"
        else:
            job_name = f"judge-alpha-batch-{len(models_to_judge)}models"

    # Create log directory structure
    log_subdir = "judgment_batches"
    log_dir = f"{LOGS_DIR}/{log_subdir}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Use SLURM_JOB_ID for unique filenames (no shell commands in SBATCH directives)
    log_file_base = f"{job_name}_$SLURM_JOB_ID"

    judge_port_val = judge_port or 8001

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --error={log_dir}/{log_file_base}.err
#SBATCH --output={log_dir}/{log_file_base}.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=alpha
#SBATCH --gres=gpu:{tp_size}

# Exit on any error
set -e

# Create timestamp for additional log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
UNIQUE_ID="${{SLURM_JOB_ID}}_${{TIMESTAMP}}"
SERVER_LOG_FILE="{log_dir}/{job_name}_${{UNIQUE_ID}}_vllm_judge_server.log"

# --- SETUP ENVIRONMENT ---
echo "Setting up the environment for Arena Hard judgment (alpha/A100, TP={tp_size})..."
source {JUDGE_VENV}/bin/activate

PYTHON_EXEC={JUDGE_VENV}/bin/python
echo "Using Python executable at: $PYTHON_EXEC"

module load CUDA

# [DEBUG] Verify the environment and installation
echo "--- Sanity Checks ---"
echo "Python Executable: $PYTHON_EXEC"
echo "vLLM Installation:"
$PYTHON_EXEC -m pip list | grep vllm || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "---------------------"

# --- DEFINE PATHS AND PORTS ---
JUDGE_PATH="{judge_path}"
API_CONFIG_FILE="{ARENA_HARD_AUTO_DIR}/config/api_config.yaml"
JUDGMENT_CONFIG_FILE="{config_file_path}"
JUDGE_PORT={judge_port_val}

echo "### JUDGING MODELS: {', '.join(models_to_judge)} ###"
echo "Judge Model: {judge_model}"
echo "Baseline Model: {baseline_model}"
echo "Tensor-parallel size: {tp_size} (GPUs allocated by SLURM via --gres=gpu:{tp_size})"
echo "Config File: $JUDGMENT_CONFIG_FILE"
echo "Judge Server Log: $SERVER_LOG_FILE"

# ===================================================================
# Start Judge Server and Generate Judgments
# ===================================================================
echo "Starting judge server sharded across {tp_size} GPUs (Port {judge_port_val})..."
$PYTHON_EXEC -m vllm.entrypoints.openai.api_server \\
    --model "$JUDGE_PATH" --port $JUDGE_PORT --tensor-parallel-size {tp_size} \\
    --max-model-len 26304 \\
    --gpu-memory-utilization 0.90 \\
    --chat-template {WORKSPACE_ROOT}/checkpoints/olmo3_chat_template.jinja \\
    > "$SERVER_LOG_FILE" 2>&1 &
JUDGE_PID=$!

sleep 5
if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
    echo "ERROR: Judge server failed to start. Check log: $SERVER_LOG_FILE"
    cat "$SERVER_LOG_FILE"
    exit 1
fi
echo "Judge server started with PID: $JUDGE_PID. Tailing log for 10s..."
tail -n 100 "$SERVER_LOG_FILE"

echo "Waiting for judge server to become ready (checking health endpoint)..."
MAX_WAIT=5400  # 90 min: observed ~55-60 min just to load the 76GB judge checkpoint's 8 shards on this Lustre filesystem
ELAPSED=0
SLEEP_INTERVAL=30

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$JUDGE_PORT/health > /dev/null 2>&1; then
        echo "Judge server is ready after $ELAPSED seconds!"
        break
    fi
    echo "Server not ready yet... waiting (elapsed: ${{ELAPSED}}s / max: ${{MAX_WAIT}}s)"
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))

    # Check if server process is still running
    if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
        echo "ERROR: Judge server process died. Check logs:"
        tail -n 50 "$SERVER_LOG_FILE"
        exit 1
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Judge server failed to become ready within $MAX_WAIT seconds"
    echo "Last 100 lines of server log:"
    tail -n 100 "$SERVER_LOG_FILE"
    kill $JUDGE_PID
    exit 1
fi

sleep 10  # Extra buffer after health check passes

echo "Verifying judge can actually serve requests (vLLM /health can return ready before torch.compile/CUDA-graph warmup finishes, which takes longer for a TP={tp_size}-sharded 80B judge)..."
PROBE_READY=0
PROBE_MAX_WAIT=1800
PROBE_ELAPSED=0
PROBE_INTERVAL=20
while [ $PROBE_ELAPSED -lt $PROBE_MAX_WAIT ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{{http_code}}" -X POST "http://localhost:$JUDGE_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{{"model": "{judge_path}", "messages": [{{"role": "user", "content": "hi"}}], "max_tokens": 1}}')
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Judge is ready to serve requests (probe succeeded after ${{PROBE_ELAPSED}}s)."
        PROBE_READY=1
        break
    fi
    echo "  Probe returned HTTP $HTTP_CODE (not ready yet)... (elapsed: ${{PROBE_ELAPSED}}s / max: ${{PROBE_MAX_WAIT}}s)"
    if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
        echo "ERROR: Judge server process died while waiting for readiness probe."
        exit 1
    fi
    sleep $PROBE_INTERVAL
    PROBE_ELAPSED=$((PROBE_ELAPSED + PROBE_INTERVAL))
done
if [ "$PROBE_READY" != "1" ]; then
    echo "ERROR: Judge never became ready to serve requests within ${{PROBE_MAX_WAIT}}s of passing /health"
    kill $JUDGE_PID
    exit 1
fi

cd {ARENA_HARD_AUTO_DIR}

echo "Running gen_judgment.py with config: $JUDGMENT_CONFIG_FILE"
$PYTHON_EXEC {ARENA_HARD_AUTO_DIR}/gen_judgment.py \\
    --setting-file "$JUDGMENT_CONFIG_FILE" \\
    --endpoint-file "{ARENA_HARD_AUTO_DIR}/config/api_config.yaml"

echo "Judgment generation complete. Killing judge server (PID: $JUDGE_PID)..."
kill $JUDGE_PID
sleep 10

echo "Judgment job completed successfully for models: {', '.join(models_to_judge)}"

# Display summary of generated judgments
echo "--- Judgment Summary ---"
JUDGMENT_DIR="{ARENA_HARD_AUTO_DIR}/data/arena-hard-v2.0/model_judgment/{judge_model}/compared_with_{baseline_label}"
mkdir -p "$JUDGMENT_DIR"
# Move generated judgment files to JUDGMENT_DIR
for model in {' '.join(models_to_judge)}; do
    src_file="{ARENA_HARD_AUTO_DIR}/data/arena-hard-v2.0/model_judgment/$model.jsonl"
    if [ -f "$src_file" ]; then
        mv "$src_file" "$JUDGMENT_DIR/"
    fi
done
if [ -d "$JUDGMENT_DIR" ]; then
    echo "Generated judgment files:"
    for model in {' '.join(models_to_judge)}; do
        if [ -f "$JUDGMENT_DIR/$model.jsonl" ]; then
            lines=$(wc -l < "$JUDGMENT_DIR/$model.jsonl")
            echo "  $model.jsonl: $lines judgments"
        else
            echo "  $model.jsonl: NOT FOUND"
        fi
    done
else
    echo "Judgment directory not found: $JUDGMENT_DIR"
fi
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make script executable
    os.chmod(script_path, 0o755)

def validate_models_exist(models_to_judge, baseline_model, api_config):
    """Validate that all models exist in the API config and have generated answers."""
    available_models = set(api_config.keys())
    missing_models = []
    missing_answers = []

    answer_dir = f"{ARENA_HARD_AUTO_DIR}/data/arena-hard-v2.0/model_answer"

    for model in models_to_judge:
        if model not in available_models:
            missing_models.append(model)
        else:
            # Check if answer file exists
            answer_file = f"{answer_dir}/{model}.jsonl"
            if not os.path.exists(answer_file):
                missing_answers.append(model)

    # Check baseline model
    if baseline_model not in available_models:
        missing_models.append(f"{baseline_model} (baseline)")
    else:
        baseline_answer_file = f"{answer_dir}/{baseline_model}.jsonl"
        if not os.path.exists(baseline_answer_file):
            missing_answers.append(f"{baseline_model} (baseline)")

    return missing_models, missing_answers

def main():
    parser = argparse.ArgumentParser(description='Automate Arena Hard judgment generation for multiple models on the alpha (A100) partition')
    parser.add_argument('--models', nargs='+', help='Specific model names to judge, or a single .txt file containing model names (one per line)')
    parser.add_argument('--models-file', type=str, default=f'{WORKSPACE_ROOT}/arena_hard_models_to_test.txt',
                       help='File containing list of models to judge (default: arena_hard_models_to_test.txt)')
    parser.add_argument('--missing-models-file', type=str,
                       help='File containing list of missing/incomplete models to judge')
    parser.add_argument('--baseline', type=str, default=DEFAULT_BASELINE,
                       help=(
                           f'Baseline model: a known alias ({", ".join(BASELINE_CONFIGS.keys())}) '
                           'or any model key from api_config.yaml (default: %(default)s)'
                       ))
    parser.add_argument('--judge-model', type=str, default=JUDGE_MODEL,
                       help=f'Judge model to use (default: {JUDGE_MODEL})')
    parser.add_argument('--tensor-parallel-size', type=int, default=DEFAULT_TP_SIZE,
                       help=(
                           f'Number of A100 40GB GPUs to shard the judge server across via vLLM '
                           f'tensor parallelism (default: {DEFAULT_TP_SIZE}). Also sets '
                           '--gres=gpu:N in the generated SLURM script. TP=2 does not fit an '
                           '~80B FP8 judge in 40GB; TP=4 leaves ~19GB/GPU headroom for KV cache.'
                       ))
    parser.add_argument('--all', action='store_true', help='Judge all tulu3 models from API config')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of models to judge per job (default: 1)')
    parser.add_argument('--dry-run', action='store_true', help='Generate scripts but do not submit jobs')
    parser.add_argument('--submit', action='store_true', help='Submit jobs after generating scripts')
    parser.add_argument('--validate-only', action='store_true', help='Only validate models without generating scripts')
    parser.add_argument('--dependency', type=str, default='',
                        help='SLURM dependency string passed to sbatch (e.g. afterok:12345:67890). '
                             'Skips missing-answer validation since generation will finish before this job runs.')

    args = parser.parse_args()

    # Create necessary directories
    create_directories()

    # Load API configuration
    api_config = load_api_config()

    # Get models to process
    if args.models:
        # Check if first argument is a .txt file
        if len(args.models) == 1 and args.models[0].endswith('.txt'):
            # Load models from the specified file
            models_to_judge = load_models_from_file(args.models[0])
            if not models_to_judge:
                print(f"No models found in {args.models[0]}.")
                return
        else:
            # Use models as provided
            models_to_judge = args.models
    elif args.missing_models_file:
        # Load models from missing models file
        models_to_judge = load_models_from_file(args.missing_models_file)
        if not models_to_judge:
            print(f"No models found in {args.missing_models_file}.")
            return
        print(f"Processing models from missing models file: {args.missing_models_file}")
    elif args.all:
        # Extract all tulu3 models
        models_to_judge = [model for model in api_config.keys() if model.startswith('tulu3-8b-rank')]
    else:
        # Load models from file
        models_to_judge = load_models_from_file(args.models_file)
        if not models_to_judge:
            print(f"No models found in {args.models_file}. Use --all to judge all models or --models to specify models directly.")
            return

    print(f"Found {len(models_to_judge)} models to judge:")
    for model in models_to_judge[:5]:  # Show first 5
        print(f"  - {model}")
    if len(models_to_judge) > 5:
        print(f"  ... and {len(models_to_judge) - 5} more")

    if not models_to_judge:
        print("No models to judge. Exiting.")
        return

    try:
        baseline_model = resolve_baseline_model(args.baseline, api_config)
    except ValueError as exc:
        print(exc)
        return

    # Validate models
    missing_models, missing_answers = validate_models_exist(models_to_judge, baseline_model, api_config)

    if missing_models:
        print(f"\\nERROR: The following models are not found in API config:")
        for model in missing_models:
            print(f"  - {model}")
        return

    if missing_answers:
        print(f"\\nWARNING: The following models don't have generated answers yet:")
        for model in missing_answers:
            print(f"  - {model}")
        if args.dependency:
            print("\\nDependency set — judgment jobs will wait for generation to finish before running.")
        else:
            print("\\nYou need to generate answers first before judging.")
            if not args.validate_only:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    return

    if args.validate_only:
        print(f"\\nValidation complete. {len(models_to_judge)} models ready for judgment.")
        return

    # Update baseline in judge_utils.py
    print(f"\\nUpdating baseline configuration...")
    update_baseline_in_judge_utils(baseline_model)

    # Create batches of models
    model_batches = []
    for i in range(0, len(models_to_judge), args.batch_size):
        batch = models_to_judge[i:i + args.batch_size]
        model_batches.append(batch)

    print(f"\\nCreating {len(model_batches)} judgment batches (batch size: {args.batch_size})")

    # Generate scripts and configs for each batch
    job_scripts = []
    for batch_idx, model_batch in enumerate(model_batches):
        print(f"\\nProcessing batch {batch_idx + 1}/{len(model_batches)} ({len(model_batch)} models)...")

        # Create judgment config file
        config_filename = f"arena_hard_judgment_batch_{batch_idx + 1}.yaml"
        config_path = f"{CONFIGS_DIR}/{config_filename}"
        create_judgment_config(model_batch, config_path, baseline_model, judge_model=args.judge_model)
        print(f"  Created config: {config_path}")

        # Create SLURM script
        script_filename = f"run_arena_hard_judgment_alpha_batch_{batch_idx + 1}.sh"
        script_path = f"{SCRIPTS_DIR}/{script_filename}"
        # Determine judge port from api_config (prefer judge model entry, fallback to first api_base or 8001)

        if args.judge_model:
            judge_port = get_port_for_model(api_config, model_name=args.judge_model, default=8001)
            model_name = args.judge_model
            model_path = extract_judge_path_from_api_config(api_config, args.judge_model)
        else:
            judge_port = get_port_for_model(api_config, model_name=JUDGE_MODEL, default=8001)

        create_judgment_slurm_script(
                model_batch,
                script_path,
                config_path,
                args.baseline,
                baseline_model,
                judge_model=model_name,
                judge_path=model_path,
                judge_port=judge_port,
                tp_size=args.tensor_parallel_size,
            )
        print(f"  Created script: {script_path}")

        job_scripts.append(script_path)

    print(f"\\nGenerated {len(job_scripts)} judgment job scripts in {SCRIPTS_DIR}")
    print(f"Generated {len(model_batches)} config files in {CONFIGS_DIR}")

    if args.dry_run:
        print("\\nDry run complete. Scripts generated but not submitted.")
        print("To submit jobs manually, run:")
        for script in job_scripts:
            print(f"  sbatch {script}")
    elif args.submit:
        print("\\nSubmitting judgment jobs...")
        submitted_jobs = []
        for script in job_scripts:
            try:
                sbatch_cmd = ['sbatch']
                if args.dependency:
                    sbatch_cmd += ['--dependency', args.dependency]
                sbatch_cmd.append(script)
                result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs.append((script, job_id))
                print(f"  Submitted {os.path.basename(script)} -> Job ID: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  Failed to submit {script}: {e}")

        print(f"\\nSuccessfully submitted {len(submitted_jobs)} judgment jobs")
        print("\\nTo monitor jobs:")
        print("  squeue -u $USER")
        print("\\nTo cancel all jobs:")
        print("  scancel -u $USER")
    else:
        print("\\nTo submit jobs, run:")
        print(f"  python {__file__} --submit")
        print("\\nOr submit individual jobs with:")
        for script in job_scripts[:3]:  # Show first 3 as examples
            print(f"  sbatch {script}")
        if len(job_scripts) > 3:
            print(f"  ... and {len(job_scripts) - 3} more")

if __name__ == "__main__":
    main()
