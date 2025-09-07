#!/bin/bash

# This script submits sbatch jobs for all tulu models found in api_config.yaml.

set -e

# The base script to modify for each job
BASE_SCRIPT_PATH="/data/horse/ws/hama901h-BFTranslation/run_arena_hard_gen_answers.sh"
# The YAML file containing model configurations
API_CONFIG_FILE="/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml"
# Directory for SLURM logs
LOG_DIR_BASE="/data/horse/ws/hama901h-BFTranslation/logs/arena-hard"
PYTHON_EXEC="/data/horse/ws/hama901h-BFTranslation/ah-eval/bin/python"
YAML_PARSER_SCRIPT="/data/horse/ws/hama901h-BFTranslation/parse_yaml.py"

# Ensure the base log directory exists
mkdir -p "$LOG_DIR_BASE"

# Use a Python script to parse the yaml file and get all model names and paths starting with tulu
# Then iterate over them
"$PYTHON_EXEC" "$YAML_PARSER_SCRIPT" "$API_CONFIG_FILE" | while read -r model_name model_path; do
    
    # Construct a job name from the model name
    job_name=$(echo "$model_name" | sed 's/tulu3-8b-//g')

    # Create a specific log directory for this job
    log_dir_job="$LOG_DIR_BASE/$(dirname "$job_name")"
    mkdir -p "$log_dir_job"
    
    # Define log file paths
    log_file_base_name=$(basename "$job_name")
    error_log="$log_dir_job/${log_file_base_name}.err"
    output_log="$log_dir_job/${log_file_base_name}.out"

    echo "---------------------------------"
    echo "Submitting job for model: $model_name"
    echo "Model Path: $model_path"
    echo "Job Name: $job_name"
    echo "Error Log: $error_log"
    echo "Output Log: $output_log"
    echo "---------------------------------"

    # Create a temporary script for this job
    temp_script=$(mktemp)

    # Read the base script and modify it for the current job
    sed -e "s|#SBATCH --job-name=.*|#SBATCH --job-name=$job_name|" \
        -e "s|#SBATCH --error=.*|#SBATCH --error=$error_log|" \
        -e "s|#SBATCH --output=.*|#SBATCH --output=$output_log|" \
        -e "s|MODEL_PATH=.*|MODEL_PATH=\"$model_path\"|" \
        "$BASE_SCRIPT_PATH" > "$temp_script"

    # Submit the job
    sbatch "$temp_script"

    # Clean up the temporary file
    rm "$temp_script"

    # Sleep for a moment to avoid overwhelming the scheduler
    sleep 3
done

echo "All jobs submitted."
