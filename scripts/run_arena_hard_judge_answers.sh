#!/bin/bash
#SBATCH --job-name=Rank64/alpha_1e5_001/step_6000
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/logs/arena-hard/Rank64/alpha_1e5_001/step_6000_judgement.err
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/logs/arena-hard/Rank64/alpha_1e5_001/step_6000_judgement.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=00:30:00          
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --exclude=c3


# Exit on any error
set -e

# --- 1. SETUP YOUR ENVIRONMENT ---
echo "Setting up the environment..."
# IMPORTANT: Replace with the command to activate your environment
source /data/horse/ws/hama901h-BFTranslation/ah-eval/bin/activate

PYTHON_EXEC=/data/horse/ws/hama901h-BFTranslation/ah-eval/bin/python
echo "Using Python executable at: $PYTHON_EXEC"

module load CUDA/12.4.0

# [DEBUG] Verify the environment and installation
echo "--- Sanity Checks ---"
echo "Python Executable: $PYTHON_EXEC"
echo "vLLM Installation:"
$PYTHON_EXEC -m pip list | grep vllm
echo "---------------------"


# --- 2. DEFINE PATHS AND PORTS ---
# MODEL_PATH="/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct"
JUDGE_PATH="/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Meta-Llama-3.1-70B-Instruct-FP8"
API_CONFIG_FILE="/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml"
JUDGE_PORT=8001

# ===================================================================
# Part 2: Generate Judgments with the Judge Model
# ===================================================================
echo "### PHASE 2: GENERATING JUDGMENTS ###"
echo "Starting judge server on GPU 1 (Port 8001)..."
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_PATH" --port 8001 --tensor-parallel-size 1 --max-model-len 26304 --chat-template /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/llama3_template.j2  > /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/Rank64/alpha_1e5_001/step_6000_vllm_judge_server.log 2>&1 &
JUDGE_PID=$!

sleep 5
if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
    echo "ERROR: Judge server failed to start. Check vllm_judge_server.log for details."
    cat /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/vllm_judge_server.log
    exit 1
fi
echo "Judge server started with PID: $JUDGE_PID. Tailing log for 10s..."
tail -n 100 /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/vllm_judge_server.log

echo "Waiting 15 mins for server to load..."
sleep 900

cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto

echo "Running gen_judgment.py..."
$PYTHON_EXEC /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/gen_judgment.py

echo "Judgment generation complete. Killing judge server (PID: $JUDGE_PID)..."
kill $JUDGE_PID
sleep 10
