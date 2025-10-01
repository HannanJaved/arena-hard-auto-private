#!/usr/bin/env python3
"""
Temporary script to generate Arena Hard answers for tulu3 models with 'lalpha' in their name.
"""

import os
import yaml
import argparse
import subprocess
from pathlib import Path

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

def extract_lalpha_models(api_config):
    """Extract all tulu3 models from API config that have 'lalpha' in their name."""
    lalpha_models = {}
    for model_name, config in api_config.items():
        if model_name.startswith('tulu3-') and 'lalpha' in model_name:
            lalpha_models[model_name] = config
    return lalpha_models

def create_directories():
    """Create necessary directories for scripts, configs, and logs."""
    for directory in [SCRIPTS_DIR, CONFIGS_DIR, LOGS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def create_gen_answer_config(model_name, output_path):
    """Create a gen_answer_config.yaml file for a specific model."""
    config = {
        'bench_name': 'arena-hard-v2.0',
        'model_list': [model_name]
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def extract_model_details(model_name):
    """Extract model details from model name for directory structure."""
    parts = model_name.split('-')
    
    rank = None
    lr = None
    step = None
    alpha = None
    
    for i, part in enumerate(parts):
        if part.startswith('rank'):
            rank = part
        elif part.startswith('alpha'):
            lr = f"{part}-{parts[i+1]}" if i+1 < len(parts) else part
        elif part.startswith('lalpha'):
            alpha = part
        elif part.startswith('step') or part == 'final':
            step = part
    
    return rank, lr, step, alpha

def create_slurm_script(model_name, model_path, script_path):
    """Create a SLURM script for a specific model."""
    rank, lr, step, alpha  = extract_model_details(model_name)
    
    log_subdir = f"{rank}/{alpha}" if rank and alpha else "misc"
    log_dir = f"{LOGS_DIR}/{log_subdir}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    config_file = f"{CONFIGS_DIR}/gen_answer_config_{model_name}.yaml"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={model_name}
#SBATCH --error={log_dir}/{step or model_name}.err
#SBATCH --output={log_dir}/{step or model_name}.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=01:00:00          
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --exclude=c3

set -e

echo "Setting up the environment for {model_name}..."
source {WORKSPACE_ROOT}/ah-eval/bin/activate

PYTHON_EXEC={WORKSPACE_ROOT}/ah-eval/bin/python
echo "Using Python executable at: $PYTHON_EXEC"

module load CUDA/12.4.0

echo "--- Sanity Checks ---"
echo "Python Executable: $PYTHON_EXEC"
echo "vLLM Installation:"
$PYTHON_EXEC -m pip list | grep vllm
echo "---------------------"

MODEL_PATH="{model_path}"
MODEL_PORT=8000

echo "### GENERATING ANSWERS FOR {model_name} ###"
echo "Starting model server on GPU 0 (Port 8000)..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \\
    --model "$MODEL_PATH" --port 8000 --tensor-parallel-size 1 \\
    --chat-template {WORKSPACE_ROOT}/checkpoints/meta-llama/llama3_template.j2 \\
    > {log_dir}/{step or model_name}_vllm_model_server.log 2>&1 &
MODEL_PID=$!

sleep 5
if ! kill -0 $MODEL_PID > /dev/null 2>&1; then
    echo "ERROR: Model server failed to start. Check vllm_model_server.log for details."
    cat {log_dir}/{step or model_name}_vllm_model_server.log
    exit 1
fi
echo "Model server started with PID: $MODEL_PID. Tailing log for 10s..."
tail -n 100 {log_dir}/{step or model_name}_vllm_model_server.log

echo "Waiting 15 mins for server to load..."
sleep 900

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
    
    os.chmod(script_path, 0o755)

def main():
    parser = argparse.ArgumentParser(description='Generate Arena Hard answers for tulu3 models with lalpha')
    parser.add_argument('--dry-run', action='store_true', help='Generate scripts but do not submit jobs')
    parser.add_argument('--submit', action='store_true', help='Submit jobs after generating scripts')
    
    args = parser.parse_args()
    
    create_directories()
    
    api_config = load_api_config()
    
    models_to_process = extract_lalpha_models(api_config)
    
    print(f"Found {len(models_to_process)} lalpha models to process:")
    for model_name in models_to_process.keys():
        print(f"  - {model_name}")
    
    if not models_to_process:
        print("No models to process. Exiting.")
        return
    
    job_scripts = []
    for model_name, model_config in models_to_process.items():
        print(f"\nProcessing {model_name}...")
        
        config_path = f"{CONFIGS_DIR}/gen_answer_config_{model_name}.yaml"
        create_gen_answer_config(model_name, config_path)
        print(f"  Created config: {config_path}")
        
        script_path = f"{SCRIPTS_DIR}/run_lalpha_{model_name}.sh"
        model_path = model_config['model']
        create_slurm_script(model_name, model_path, script_path)
        print(f"  Created script: {script_path}")
        
        job_scripts.append(script_path)
    
    print(f"\nGenerated {len(job_scripts)} job scripts in {SCRIPTS_DIR}")
    print(f"Generated {len(models_to_process)} config files in {CONFIGS_DIR}")
    
    if args.dry_run:
        print("\nDry run complete. Scripts generated but not submitted.")
        print("To submit jobs manually, run:")
        for script in job_scripts[:5]:
            print(f"  sbatch {script}")
        if len(job_scripts) > 5:
            print(f"  ... and {len(job_scripts) - 5} more")
    elif args.submit:
        print("\nSubmitting jobs...")
        submitted_jobs = []
        for script in job_scripts:
            try:
                result = subprocess.run(['sbatch', script], capture_output=True, text=True, check=True)
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
        print("\nTo submit jobs, run this script with --submit")
        print("\nOr submit individual jobs with:")
        for script in job_scripts[:5]:
            print(f"  sbatch {script}")
        if len(job_scripts) > 5:
            print(f"  ... and {len(job_scripts) - 5} more")

if __name__ == "__main__":
    main()
