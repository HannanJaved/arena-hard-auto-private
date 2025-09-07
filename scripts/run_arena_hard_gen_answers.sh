#!/bin/bash
#SBATCH --job-name=Rank64/alpha_1e5_001/step_48000
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/logs/arena-hard/Rank64/alpha_1e5_001/step_48000.err
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/logs/arena-hard/Rank64/alpha_1e5_001/step_48000.out
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
MODEL_PATH="/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank64/alpha_1e5_001/step_48000"
API_CONFIG_FILE="/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml"
MODEL_PORT=8000

# ===================================================================
# Part 1: Generate Answers for Your Model
# ===================================================================
echo "### PHASE 1: GENERATING ANSWERS ###"
echo "Starting model server on GPU 0 (Port 8000)..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --port 8000 --tensor-parallel-size 1 --chat-template /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/llama3_template.j2  > /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/Rank64/alpha_1e5_001/step_48000_vllm_model_server.log 2>&1 &
MODEL_PID=$!

sleep 5
if ! kill -0 $MODEL_PID > /dev/null 2>&1; then
    echo "ERROR: Model server failed to start. Check vllm_model_server.log for details."
    cat /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/vllm_model_server.log
    exit 1
fi
echo "Model server started with PID: $MODEL_PID. Tailing log for 10s..."
tail -n 100 /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/vllm_model_server.log

echo "Waiting 15 mins for server to load..."
sleep 900

cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto

echo "Running gen_answer.py..."
$PYTHON_EXEC /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/gen_answer.py

echo "Answer generation complete. Killing model server (PID: $MODEL_PID)..."
kill $MODEL_PID
sleep 10