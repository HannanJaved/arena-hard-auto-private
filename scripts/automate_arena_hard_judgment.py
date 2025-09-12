#!/usr/bin/env python3
"""
Automation script to generate Arena Hard judgments for multiple models.
This script creates individual SLURM jobs for judging each model against the baseline.
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
SCRIPTS_DIR = f"{WORKSPACE_ROOT}/generated_judgment_scripts"
CONFIGS_DIR = f"{WORKSPACE_ROOT}/generated_judgment_configs"

# Judge configuration
JUDGE_MODEL = "neuralmagic-llama3.1-70b-instruct-fp8"
BASELINE_MODEL = "llama3.1-8b-TULU"
JUDGE_PATH = "/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Meta-Llama-3.1-70B-Instruct-FP8"

def load_api_config():
    """Load the API configuration file."""
    config_path = f"{ARENA_HARD_AUTO_DIR}/config/api_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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

def create_directories():
    """Create necessary directories for scripts, configs, and logs."""
    for directory in [SCRIPTS_DIR, CONFIGS_DIR, LOGS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def create_judgment_config(models_to_judge, output_path):
    """Create an arena-hard config file for judging specific models."""
    config = {
        'judge_model': JUDGE_MODEL,
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

def create_judgment_slurm_script(models_to_judge, script_path, config_file_path):
    """Create a SLURM script for judging a batch of models."""
    
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
    log_file_base = f"{job_name}_${{SLURM_JOB_ID}}"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --error={log_dir}/{log_file_base}.err
#SBATCH --output={log_dir}/{log_file_base}.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        
#SBATCH --mem=64G                
#SBATCH --time=00:45:00          
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --exclude=c3

# Exit on any error
set -e

# Create timestamp for additional log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
UNIQUE_ID="${{SLURM_JOB_ID}}_${{TIMESTAMP}}"

# --- SETUP ENVIRONMENT ---
echo "Setting up the environment for Arena Hard judgment..."
source {WORKSPACE_ROOT}/ah-eval/bin/activate

PYTHON_EXEC={WORKSPACE_ROOT}/ah-eval/bin/python
echo "Using Python executable at: $PYTHON_EXEC"

module load CUDA/12.4.0

# [DEBUG] Verify the environment and installation
echo "--- Sanity Checks ---"
echo "Python Executable: $PYTHON_EXEC"
echo "vLLM Installation:"
$PYTHON_EXEC -m pip list | grep vllm
echo "---------------------"

# --- DEFINE PATHS AND PORTS ---
JUDGE_PATH="{JUDGE_PATH}"
API_CONFIG_FILE="{ARENA_HARD_AUTO_DIR}/config/api_config.yaml"
JUDGMENT_CONFIG_FILE="{config_file_path}"
JUDGE_PORT=8001

echo "### JUDGING MODELS: {', '.join(models_to_judge)} ###"
echo "Judge Model: {JUDGE_MODEL}"
echo "Baseline Model: {BASELINE_MODEL}"
echo "Config File: $JUDGMENT_CONFIG_FILE"

# ===================================================================
# Start Judge Server and Generate Judgments
# ===================================================================
echo "Starting judge server on GPU 0 (Port 8001)..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \\
    --model "$JUDGE_PATH" --port 8001 --tensor-parallel-size 1 \\
    --max-model-len 26304 \\
    --chat-template {WORKSPACE_ROOT}/checkpoints/meta-llama/llama3_template.j2 \\
    > {log_dir}/{job_name}_${{UNIQUE_ID}}_vllm_judge_server.log 2>&1 &
JUDGE_PID=$!

sleep 5
if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
    echo "ERROR: Judge server failed to start. Check vllm_judge_server.log for details."
    cat {log_dir}/{job_name}_${{UNIQUE_ID}}_vllm_judge_server.log
    exit 1
fi
echo "Judge server started with PID: $JUDGE_PID. Tailing log for 10s..."
tail -n 100 {log_dir}/{job_name}_${{UNIQUE_ID}}_vllm_judge_server.log

echo "Waiting 15 mins for judge server to load..."
sleep 900

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
JUDGMENT_DIR="{ARENA_HARD_AUTO_DIR}/data/arena-hard-v2.0/model_judgment/{JUDGE_MODEL}"
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

def validate_models_exist(models_to_judge, api_config):
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
    if BASELINE_MODEL not in available_models:
        missing_models.append(f"{BASELINE_MODEL} (baseline)")
    else:
        baseline_answer_file = f"{answer_dir}/{BASELINE_MODEL}.jsonl"
        if not os.path.exists(baseline_answer_file):
            missing_answers.append(f"{BASELINE_MODEL} (baseline)")
    
    return missing_models, missing_answers

def main():
    parser = argparse.ArgumentParser(description='Automate Arena Hard judgment generation for multiple models')
    parser.add_argument('--models', nargs='+', help='Specific model names to judge')
    parser.add_argument('--models-file', type=str, default=f'{WORKSPACE_ROOT}/arena_hard_models_to_test.txt',
                       help='File containing list of models to judge (default: arena_hard_models_to_test.txt)')
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
    
    # Validate models
    missing_models, missing_answers = validate_models_exist(models_to_judge, api_config)
    
    if missing_models:
        print(f"\\nERROR: The following models are not found in API config:")
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
        create_judgment_config(model_batch, config_path)
        print(f"  Created config: {config_path}")
        
        # Create SLURM script
        script_filename = f"run_arena_hard_judgment_batch_{batch_idx + 1}.sh"
        script_path = f"{SCRIPTS_DIR}/{script_filename}"
        create_judgment_slurm_script(model_batch, script_path, config_path)
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
