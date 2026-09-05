#!/usr/bin/env python3
"""
Automation script to generate Arena Hard answers for multiple models.
This script creates individual SLURM jobs for each model to run in parallel.
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
SCRIPTS_DIR = f"{WORKSPACE_ROOT}/generated_scripts"
CONFIGS_DIR = f"{WORKSPACE_ROOT}/generated_configs"

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
    """Parse and return port (int) from a URL like 'http://localhost:8000/v1'.

    Returns None if not found or cannot parse.
    """
    if not api_base_url:
        return None
    try:
        parsed = urlparse(api_base_url)
        if parsed.port:
            return parsed.port
        # Fallback: try to parse port from netloc
        netloc = parsed.netloc
        if ':' in netloc:
            return int(netloc.split(':')[-1])
    except Exception:
        return None
    return None


def get_port_for_model(api_config, model_name=None, default=8000):
    """Get port to use for model server.

    If model_name is provided, try to read api_base under that model's config.
    Otherwise return the first api_base port found in the config. If none
    found, return the provided default.
    """
    # Try model-specific entry first
    try:
        if model_name and model_name in api_config:
            entry = api_config[model_name]
            api_base = None
            if isinstance(entry, dict) and 'api_base' in entry:
                api_base = entry['api_base']
            else:
                # might be nested
                api_base = _find_api_base(entry)
            port = parse_port_from_api_base(api_base)
            if port:
                return port
    except Exception:
        pass

    # Fallback: search entire config for any api_base
    api_base = _find_api_base(api_config)
    port = parse_port_from_api_base(api_base)
    return port or default

def extract_tulu_models(api_config):
    """Extract all tulu3 models from API config that contain an alpha setting."""
    tulu_models = {}
    for model_name, config in api_config.items():
        if model_name.startswith('tulu3-') and 'alpha' in model_name:
            tulu_models[model_name] = config
    return tulu_models

def create_directories():
    """Create necessary directories for scripts, configs, and logs."""
    for directory in [SCRIPTS_DIR, CONFIGS_DIR, LOGS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def create_gen_answer_config(model_name, output_path, bench_name):
    """Create a gen_answer_config.yaml file for a specific model."""
    config = {
        'bench_name': bench_name,
        'model_list': [model_name]
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def extract_model_details(model_name):
    """Extract model details from model name for directory structure."""
    # Parse model name like: tulu3-8b-rank64-alpha1e5-001-step48000
    parts = model_name.split('-')

    rank = None
    alpha = None
    step = None

    for i, part in enumerate(parts):
        if part.startswith('rank'):
            rank = part
        elif part.startswith('alpha'):
            alpha = f"{part}-{parts[i+1]}" if i + 1 < len(parts) else part
        elif part.startswith('step') or part == 'final':
            step = part

    return rank, alpha, step
    # return rank, lr, step

def create_slurm_script(model_name, model_path, script_path, model_port=8000, account="p_neurasearch"):
    """Create a SLURM script for a specific model."""
    rank, alpha, step = extract_model_details(model_name)
    
    # Create log directory structure
    log_subdir = f"{rank}/{alpha}" if rank and alpha else "misc"
    log_dir = f"{LOGS_DIR}/{log_subdir}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate config file path
    config_file = f"{CONFIGS_DIR}/gen_answer_config_{model_name}.yaml"
    
# Use the configured model port
    model_port_val = model_port or 8000

    script_content = f"""#!/bin/bash
#SBATCH --job-name={model_name}
#SBATCH --error={log_dir}/{model_name}.err
#SBATCH --output={log_dir}/{model_name}.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=01:00:00
#SBATCH --partition=capella
#SBATCH --exclude=c52,c78,c93
#SBATCH --gres=gpu:1
#SBATCH --account={account}

# Exit on any error
set -e

# --- SETUP ENVIRONMENT ---
echo "Setting up the environment for {model_name}..."
source {WORKSPACE_ROOT}/arena-hard-auto/venv/bin/activate

PYTHON_EXEC={WORKSPACE_ROOT}/arena-hard-auto/venv/bin/python
echo "Using Python executable at: $PYTHON_EXEC"

module load CUDA

# [DEBUG] Verify the environment and installation
echo "--- Sanity Checks ---"
echo "Python Executable: $PYTHON_EXEC"
echo "vLLM Installation:"
$PYTHON_EXEC -m pip list | grep vllm || true
echo "---------------------"

# --- DEFINE PATHS AND PORTS ---
MODEL_PATH="{model_path}"
API_CONFIG_FILE="{ARENA_HARD_AUTO_DIR}/config/api_config.yaml"
MODEL_PORT={model_port_val}

# ===================================================================
# Generate Answers for {model_name}
# ===================================================================
echo "### GENERATING ANSWERS FOR {model_name} ###"
echo "Starting model server on GPU 0 (Port {model_port_val})..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \\
    --model "$MODEL_PATH" --port $MODEL_PORT --tensor-parallel-size 1 \\
    --trust-remote-code true \\
    --chat-template {WORKSPACE_ROOT}/checkpoints/olmo3_chat_template.jinja \\
    > {log_dir}/{model_name}_vllm_model_server.log 2>&1 &
MODEL_PID=$!

sleep 5
if ! kill -0 $MODEL_PID > /dev/null 2>&1; then
    echo "ERROR: Model server failed to start. Check vllm_model_server.log for details."
    cat {log_dir}/{model_name}_vllm_model_server.log
    exit 1
fi
echo "Model server started with PID: $MODEL_PID. Tailing log for 10s..."
tail -n 100 {log_dir}/{model_name}_vllm_model_server.log

echo "Waiting for model server to become ready (checking health endpoint)..."
MAX_WAIT=2400  # 40 minutes max wait time
ELAPSED=0
SLEEP_INTERVAL=30

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$MODEL_PORT/health > /dev/null 2>&1; then
        echo "Model server is ready after $ELAPSED seconds!"
        break
    fi
    echo "Server not ready yet... waiting (elapsed: ${{ELAPSED}}s / max: ${{MAX_WAIT}}s)"
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
    
    # Check if server process is still running
    if ! kill -0 $MODEL_PID > /dev/null 2>&1; then
        echo "ERROR: Model server process died. Check logs:"
        tail -n 50 {log_dir}/{model_name}_vllm_model_server.log
        exit 1
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Model server failed to become ready within $MAX_WAIT seconds"
    echo "Last 100 lines of server log:"
    tail -n 100 {log_dir}/{model_name}_vllm_model_server.log
    kill $MODEL_PID
    exit 1
fi

sleep 10  # Extra buffer after health check passes

cd {ARENA_HARD_AUTO_DIR}

echo "Running gen_answer.py with config: {config_file}"
$PYTHON_EXEC {ARENA_HARD_AUTO_DIR}/gen_answer.py --config-file {config_file}

echo "Answer generation complete for {model_name}. Killing model server (PID: $MODEL_PID)..."
kill $MODEL_PID
sleep 10

echo "Job completed successfully for {model_name}"
"""

    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)

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

def main():
    parser = argparse.ArgumentParser(description='Automate Arena Hard answer generation for multiple models')
    parser.add_argument('--models', nargs='+', help='Specific model names to process, or a single .txt file containing model names (one per line)')
    parser.add_argument('--models-file', type=str, default=f'{WORKSPACE_ROOT}/arena_hard_models_to_test.txt',
                       help='File containing list of models to process (default: arena_hard_models_to_test.txt)')
    parser.add_argument('--missing-models-file', type=str, 
                       help='File containing missing models (generated by check_missing_model_answers.py)')
    parser.add_argument('--all', action='store_true', help='Process all tulu3 models from API config')
    parser.add_argument('--dry-run', action='store_true', help='Generate scripts but do not submit jobs')
    parser.add_argument('--submit', action='store_true', help='Submit jobs after generating scripts')
    parser.add_argument('--bench-name', type=str, default='arena-hard-v2.0',
                       help='Benchmark name to use in config files (default: arena-hard-v2.0)',
                       choices=['arena-hard-v0.1', 'arena-hard-v2.0', 'hard_prompt', 'coding', 'math', 'creative_writing'])
    parser.add_argument('--account', choices=['p_neurasearch', 'p_scads_nas'], default='p_neurasearch',
                       help='SLURM account to charge jobs to (default: p_neurasearch).')

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
            model_names_from_file = load_models_from_file(args.models[0])
            if not model_names_from_file:
                print(f"No models found in {args.models[0]}.")
                return
            models_list = model_names_from_file
        else:
            # Use models as provided
            models_list = args.models
            
        models_to_process = {}
        # Use the raw api_config to check availability of provided models
        available_models = api_config
        for model in models_list:
            if model in available_models:
                models_to_process[model] = available_models[model]
            else:
                print(f"Warning: Model '{model}' not found in API config")
    elif args.missing_models_file:
        # Load models from missing models file
        model_names_from_file = load_models_from_file(args.missing_models_file)
        if not model_names_from_file:
            print(f"No models found in {args.missing_models_file}.")
            return
        
        # Use the raw api_config to check availability of provided models
        available_models = api_config
        models_to_process = {}
        for model in model_names_from_file:
            if model in available_models:
                models_to_process[model] = available_models[model]
            else:
                print(f"Warning: Model '{model}' from missing models file not found in API config")
        
        print(f"Processing {len(models_to_process)} missing models from {args.missing_models_file}")
    elif args.all:
        # Process all tulu3 models (build inline instead of calling helper)
        models_to_process = {k: v for k, v in api_config.items() if k.startswith('tulu3-') and 'alpha' in k}
    else:
        # Load models from file
        model_names_from_file = load_models_from_file(args.models_file)
        if not model_names_from_file:
            print(f"No models found in {args.models_file}. Use --all to process all models or --models to specify models directly.")
            return

        available_models = api_config
        models_to_process = {}       
        
        for model in model_names_from_file:
            if model in available_models:
                models_to_process[model] = available_models[model]
            else:
                print(f"Warning: Model '{model}' from file not found in API config")
         
        
    print(f"Found {len(models_to_process)} models to process:")
    for model_name in models_to_process.keys():
        print(f"  - {model_name}")
    
    if not models_to_process:
        print("No models to process. Exiting.")
        return
    
    bench_name = args.bench_name

    # Generate scripts and configs for each model
    job_scripts = []
    for model_name, model_config in models_to_process.items():
        print(f"\nProcessing {model_name}...")

        # Create gen_answer config file
        config_path = f"{CONFIGS_DIR}/gen_answer_config_{model_name}.yaml"
        create_gen_answer_config(model_name, config_path, bench_name)
        print(f"  Created config: {config_path}")

        # Create SLURM script
        script_path = f"{SCRIPTS_DIR}/run_arena_hard_{model_name}.sh"
        model_path = model_config['model']
        # Determine model port from api_config (fall back to 8000)
        model_port = get_port_for_model(api_config, model_name=model_name, default=8000)
        create_slurm_script(model_name, model_path, script_path, model_port=model_port, account=args.account)
        print(f"  Created script: {script_path}")

        job_scripts.append(script_path)
    
    print(f"\\nGenerated {len(job_scripts)} job scripts in {SCRIPTS_DIR}")
    print(f"Generated {len(models_to_process)} config files in {CONFIGS_DIR}")
    
    if args.dry_run:
        print("\\nDry run complete. Scripts generated but not submitted.")
        print("To submit jobs manually, run:")
        for script in job_scripts:
            print(f"  sbatch {script}")
    elif args.submit:
        print("\\nSubmitting jobs...")
        submitted_jobs = []
        for script in job_scripts:
            try:
                result = subprocess.run(['sbatch', script], capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs.append((script, job_id))
                print(f"  Submitted {os.path.basename(script)} -> Job ID: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  Failed to submit {script}: {e}")
        
        print(f"\\nSuccessfully submitted {len(submitted_jobs)} jobs")
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
