#!/bin/bash
#SBATCH --job-name=arenahard
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/logs/arena-hard/arenahard.err
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/logs/arena-hard/arenahard.out
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=2-00:00:00          
#SBATCH --partition=capella
#SBATCH --gres=gpu:2
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
MODEL_PATH="/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct"
# JUDGE_PATH="/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Meta-Llama-3.1-70B-Instruct-FP8"
API_CONFIG_FILE="/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml"
MODEL_PORT=8000
# JUDGE_PORT=8001

# ===================================================================
# Part 1: Generate Answers for Your Model
# ===================================================================
echo "### PHASE 1: GENERATING ANSWERS ###"
echo "Starting model server on GPU 0 (Port 8000)..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --port 8000 --tensor-parallel-size 1 --chat-template /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/llama3_template.j2  > /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/vllm_model_server.log 2>&1 &
MODEL_PID=$!
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL_PATH" --port 8000 --tensor-parallel-size 1 &
# MODEL_PID=$!

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

# echo "Waiting 60 seconds for server to load..."
# sleep 60

# cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto

# echo "Running gen_answer.py..."
# python gen_answer.py

# echo "Answer generation complete. Killing model server (PID: $MODEL_PID)..."
# kill $MODEL_PID
# sleep 10 # Give time for the port to be freed


# ===================================================================
# Part 2: Generate Judgments with the Judge Model
# ===================================================================
# echo "### PHASE 2: GENERATING JUDGMENTS ###"
# echo "Starting judge server on GPU 1 (Port 8001)..."
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#     --model "$JUDGE_PATH" --port 8001 --tensor-parallel-size 1 --max-model-len 26304 --chat-template /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/llama3_template.j2  > /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/vllm_judge_server.log 2>&1 &
# JUDGE_PID=$!

# sleep 5
# if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
#     echo "ERROR: Judge server failed to start. Check vllm_judge_server.log for details."
#     cat /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/vllm_judge_server.log
#     exit 1
# fi
# echo "Judge server started with PID: $JUDGE_PID. Tailing log for 10s..."
# tail -n 100 /data/horse/ws/hama901h-BFTranslation/logs/arena-hard/vllm_judge_server.log

# echo "Waiting 15 mins for server to load..."
# sleep 900
# # echo "Waiting 180 seconds for server to load..."
# # sleep 180

# cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto

# echo "Running gen_judgment.py..."
# $PYTHON_EXEC /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/gen_judgment.py

# echo "Judgment generation complete. Killing judge server (PID: $JUDGE_PID)..."
# kill $JUDGE_PID
# sleep 10

# # ===================================================================
# # Part 3: Show the Final Results
# # ===================================================================
# echo "### PHASE 3: SHOWING RESULTS ###"
# $PYTHON_EXEC /data/horse/ws/hama901h-BFTranslation/arena-hard-auto/show_result.py --judge-names neuralmagic-llama3.1-70b-instruct-fp8

# # --- FINAL CLEANUP ---
# # Revert the config file change so it's clean for the next run
# git checkout -- "$API_CONFIG_FILE"
# echo "Job completed successfully."