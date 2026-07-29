#!/usr/bin/env python3
"""
Automation script to generate Arena Hard judgments for multiple models.
This script creates individual SLURM jobs for judging each model against the baseline.

Use this for Tulu: --chat-template {WORKSPACE_ROOT}/checkpoints/meta-llama/tulu_template.j2 \\
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
LOGS_DIR = f"{WORKSPACE_ROOT}/logs/arena-hard"
SCRIPTS_DIR = f"{WORKSPACE_ROOT}/generated_judgment_scripts"
CONFIGS_DIR = f"{WORKSPACE_ROOT}/generated_judgment_configs"
JUDGE_UTILS_PATH = f"{ARENA_HARD_AUTO_DIR}/utils/judge_utils.py"

# Judge configuration
# JUDGE_MODEL = "neuralmagic-llama3.1-70b-instruct-fp8"
JUDGE_MODEL = "Qwen3-Next-80B-A3B-Instruct-FP8"
JUDGE_PATH = f"{WORKSPACE_ROOT}/checkpoints/Qwen3-Next-80B-A3B-Instruct-FP8"
# JUDGE_PATH = "/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Meta-Llama-3.1-70B-Instruct-FP8"

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


def safe_batch_id(value, max_len=180):
    """Create a filesystem-safe identifier with hashing for long names."""
    import hashlib
    import re

    safe_value = re.sub(r'[^A-Za-z0-9_.-]', '_', value)
    if len(safe_value) <= max_len:
        return safe_value
    digest = hashlib.sha1(safe_value.encode("utf-8")).hexdigest()[:12]
    head = safe_value[: max(0, max_len - 13)]
    return f"{head}-{digest}"

def create_directories():
    """Create necessary directories for scripts, configs, and logs."""
    for directory in [SCRIPTS_DIR, CONFIGS_DIR, LOGS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def update_baseline_in_judge_utils(baseline_name, api_config):
    """Update the BASELINE_MODEL in judge_utils.py to match the selected baseline."""
    baseline_model = resolve_baseline_model(baseline_name, api_config)
    
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

def create_judgment_config(models_to_judge, output_path, baseline_name, judge_model=JUDGE_MODEL, api_config=None):
    """Create an arena-hard config file for judging specific models."""
    baseline_model = resolve_baseline_model(baseline_name, api_config or {})
    
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

def create_judgment_slurm_script(models_to_judge, script_path, config_file_path, baseline_name, judge_model=JUDGE_MODEL, judge_path=JUDGE_PATH, judge_port=8001, api_config=None):
    """Create a SLURM script for judging a batch of models."""
    
    baseline_model = resolve_baseline_model(baseline_name, api_config or {})
    
    # Create a meaningful job name from the first and last model
    if len(models_to_judge) == 1:
        job_name = f"judge-{models_to_judge[0]}"
        rank, alpha, step = extract_model_details(models_to_judge[0])
    else:
        first_model = models_to_judge[0]
        last_model = models_to_judge[-1]
        rank1, alpha1, step1 = extract_model_details(first_model)
        rank2, alpha2, step2 = extract_model_details(last_model)
        
        if rank1 == rank2 and alpha1 == alpha2:
            job_name = f"judge-{rank1}-{alpha1}-{step1}-to-{step2}"
        elif rank1 == rank2:
            job_name = f"judge-{rank1}-batch"
        else:
            job_name = f"judge-batch-{len(models_to_judge)}models"
    
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
#SBATCH --cpus-per-task=6        
#SBATCH --mem=32G                
#SBATCH --time=02:00:00          
#SBATCH --partition=capella
#SBATCH --gres=gpu:1

# Exit on any error
set -e

# Create timestamp for additional log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
UNIQUE_ID="${{SLURM_JOB_ID}}_${{TIMESTAMP}}"
SERVER_LOG_FILE="{log_dir}/{job_name}_${{UNIQUE_ID}}_vllm_judge_server.log"

# --- SETUP ENVIRONMENT ---
echo "Setting up the environment for Arena Hard judgment..."
source {WORKSPACE_ROOT}/arena-hard-auto/venv/bin/activate

PYTHON_EXEC={WORKSPACE_ROOT}/arena-hard-auto/venv/bin/python
echo "Using Python executable at: $PYTHON_EXEC"

module load CUDA
export PATH={WORKSPACE_ROOT}/arena-hard-auto/venv/bin:$PATH
source {WORKSPACE_ROOT}/cache.sh

# [DEBUG] Verify the environment and installation
echo "--- Sanity Checks ---"
echo "Python Executable: $PYTHON_EXEC"
echo "vLLM Installation:"
$PYTHON_EXEC -m pip list | grep vllm || true
echo "---------------------"

# --- DEFINE PATHS AND PORTS ---
JUDGE_PATH="{judge_path}"
API_CONFIG_FILE="{ARENA_HARD_AUTO_DIR}/config/api_config.yaml"
JUDGMENT_CONFIG_FILE="{config_file_path}"
JUDGE_PORT={judge_port_val}

echo "### JUDGING MODELS: {', '.join(models_to_judge)} ###"
echo "Judge Model: {judge_model}"
echo "Baseline Model: {baseline_model}"
echo "Config File: $JUDGMENT_CONFIG_FILE"
echo "Judge Server Log: $SERVER_LOG_FILE"

# ===================================================================
# Start Judge Server and Generate Judgments
# ===================================================================
echo "Starting judge server on GPU 0 (Port {judge_port_val})..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \\
    --model "$JUDGE_PATH" --port $JUDGE_PORT --tensor-parallel-size 1 \\
    --max-model-len 26304 \\
    --max-num-seqs 512 \\
    --gpu-memory-utilization 0.95 \\
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
MAX_WAIT=3600  # 60 minutes max wait time
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
JUDGMENT_DIR="{ARENA_HARD_AUTO_DIR}/data/arena-hard-v2.0/model_judgment/{judge_model}/compared_with_{baseline_name}"
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

def resolve_baseline_model(baseline, api_config):
    """Resolve the baseline model name.

    If `baseline` is a key in BASELINE_CONFIGS, return the mapped model name.
    Otherwise, treat `baseline` as a literal model name and verify it exists in
    api_config. Raises ValueError if it cannot be resolved.
    """
    if baseline in BASELINE_CONFIGS:
        return BASELINE_CONFIGS[baseline]
    if not isinstance(api_config, dict):
        api_config = {}
    # Treat as a direct model name — look it up in api_config
    if baseline in api_config:
        return baseline
    raise ValueError(
        f"Baseline '{baseline}' is not a known alias in BASELINE_CONFIGS and was not "
        f"found in api_config.yaml. Available aliases: {list(BASELINE_CONFIGS.keys())}. "
        f"Available api_config models (first 10): {list(api_config.keys())[:10]}"
    )


def validate_models_exist(models_to_judge, baseline, api_config):
    """Validate that all models exist in the API config and have generated answers."""
    available_models = set(api_config.keys())
    missing_models = []
    missing_answers = []
    
    # Get baseline model name from configuration (alias or direct name)
    baseline_model = resolve_baseline_model(baseline, api_config)
    
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
    parser = argparse.ArgumentParser(description='Automate Arena Hard judgment generation for multiple models')
    parser.add_argument('--models', nargs='+', help='Specific model names to judge, or a single .txt file containing model names (one per line)')
    parser.add_argument('--models-file', type=str, default=f'{WORKSPACE_ROOT}/arena_hard_models_to_test.txt',
                       help='File containing list of models to judge (default: arena_hard_models_to_test.txt)')
    parser.add_argument('--baseline', type=str, default=DEFAULT_BASELINE,
                       help=f'Baseline model: a known alias ({", ".join(BASELINE_CONFIGS.keys())}) '
                            f'or a model name from api_config.yaml (default: {DEFAULT_BASELINE})')
    parser.add_argument('--judge-model', type=str, default=JUDGE_MODEL,
                       help=f'Judge model to use (default: {JUDGE_MODEL})')
    parser.add_argument('--all', action='store_true', help='Judge all tulu3 models from API config')
    parser.add_argument('--batch-size', type=int, default=1, 
                       help='Number of models to judge per job (default: 1)')
    parser.add_argument('--dry-run', action='store_true', help='Generate scripts but do not submit jobs')
    parser.add_argument('--submit', action='store_true', help='Submit jobs after generating scripts')
    parser.add_argument('--validate-only', action='store_true', help='Only validate models without generating scripts')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Load API configuration
    api_config = load_api_config()
    
    # Get models to process
    if args.models:
        models_to_judge = args.models
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
    
    missing_models, missing_answers = validate_models_exist(models_to_judge, args.baseline, api_config)
    
    if missing_models:
        print(f"\nERROR: The following models are not found in API config:")
        for model in missing_models:
            print(f"  - {model}")
        return
        
    if missing_answers:
        print(f"\\nWARNING: The following models don't have generated answers yet:")
        for model in missing_answers:
            print(f"  - {model}")
        print("\\nYou need to generate answers first before judging.")
        if not args.validate_only:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return
    
    if args.validate_only:
        print(f"\\nValidation complete. {len(models_to_judge)} models ready for judgment.")
        return
    
    # NOTE: Do NOT update judge_utils.py here. Baseline is now taken from the
    # per-job config file to avoid cross-job races in SLURM.
    
    # Create batches of models
    model_batches = []
    for i in range(0, len(models_to_judge), args.batch_size):
        batch = models_to_judge[i:i + args.batch_size]
        model_batches.append(batch)

    baseline_model_name = resolve_baseline_model(args.baseline, api_config)
    
    print(f"\\nCreating {len(model_batches)} judgment batches (batch size: {args.batch_size})")
    
    # Generate scripts and configs for each batch
    job_scripts = []

    for batch_idx, model_batch in enumerate(model_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{len(model_batches)} ({len(model_batch)} models)...")

        # Determine config and script filenames based on model names
        if len(model_batch) == 1:
            model_id = model_batch[0]
        else:
            model_id = '__'.join(model_batch)

        # Include baseline in filenames to avoid overwriting across runs
        combined_id = f"baseline_{baseline_model_name}__{model_id}"
        safe_model_id = safe_batch_id(combined_id)

        config_filename = f"arena_hard_judgment_{safe_model_id}.yaml"
        config_path = f"{CONFIGS_DIR}/{config_filename}"
        create_judgment_config(
            model_batch,
            config_path,
            args.baseline,
            judge_model=args.judge_model,
            api_config=api_config,
        )
        print(f"  Created config: {config_path}")

        script_filename = f"run_arena_hard_judgment_{safe_model_id}.sh"
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
            judge_model=model_name,
            judge_path=model_path,
            judge_port=judge_port,
            api_config=api_config,
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
                result = subprocess.run(['sbatch', script], capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs.append((script, job_id))
                print(f"  Submitted {os.path.basename(script)} -> Job ID: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  Failed to submit {script}: {e}")
        
        print(f"\nSuccessfully submitted {len(submitted_jobs)} judgment jobs")
        print("\nTo monitor jobs:")
        print("  squeue -u $USER")
        print("\nTo cancel all jobs:")
        print("  scancel -u $USER")
    else:
        print("\nTo submit jobs, run:")
        print(f"  python {__file__} --submit")
        print("\nOr submit individual jobs with:")
        for script in job_scripts[:3]:  # Show first 3 as examples
            print(f"  sbatch {script}")
        if len(job_scripts) > 3:
            print(f"  ... and {len(job_scripts) - 3} more")

if __name__ == "__main__":
    main()