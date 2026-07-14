#!/usr/bin/env python3
"""
Automation script to generate AlpacaEval judgments using a VLLM judge model,
running the judge server on the `alpha` partition (A100 40GB) instead of
`capella` (H100). The judge model (~80B params, FP8) does not fit on a
single A100 40GB, so the judge server is sharded across multiple GPUs with
vLLM tensor parallelism (--tensor-parallel-size, default 4 -> ~21GB
weights/GPU, leaving headroom for KV cache).

This is a variant of automate_alpaca_eval_judgment.py — see that file for
the capella/H100/TP=1 version. Kept as a separate script per request so the
original is untouched.
"""

import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yaml

WORKSPACE_ROOT = "/data/horse/ws/hama901h-BFTranslation"
LOGS_DIR = f"{WORKSPACE_ROOT}/logs/alpaca_eval-alpha"
SCRIPTS_DIR = f"{WORKSPACE_ROOT}/generated_alpaca_eval_judgment_scripts_alpha"
CONFIGS_DIR = f"{WORKSPACE_ROOT}/generated_alpaca_eval_judgment_configs_alpha"
OUTPUTS_DIR = f"{WORKSPACE_ROOT}/alpaca_eval_outputs"
RUN_SCRIPT = f"{WORKSPACE_ROOT}/arena-hard-auto/scripts/run_alpaca_eval_judgment.py"
DEFAULT_JUDGE_MODEL = "Qwen3-Next-80B-A3B-Instruct-FP8"
DEFAULT_PROMPT_TEMPLATE = (
    f"{WORKSPACE_ROOT}/alpaca_eval/src/alpaca_eval/evaluators_configs/"
    "alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt"
)

# Default tensor-parallel degree for the judge server on A100 40GB.
# 80B-param FP8 judge (~84GB of weight shards on disk) needs TP>=4 to fit:
#   TP=2 -> ~42GB/GPU (does not fit in 40GB)
#   TP=4 -> ~21GB/GPU (fits, leaves ~19GB/GPU for KV cache/activations)
DEFAULT_TP_SIZE = 4


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


def get_port_for_model(api_config, model_name=None, default=8001):
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
    for directory in [SCRIPTS_DIR, CONFIGS_DIR, LOGS_DIR]:
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


def resolve_model_path(model_config):
    if isinstance(model_config, dict):
        if "model" in model_config:
            return model_config["model"]
        if "model_path" in model_config:
            return model_config["model_path"]
    return None


def create_judge_config(output_path, judge_model_name, prompt_template):
    config = {
        "alpaca_eval_vllm_judge": {
            "prompt_template": prompt_template,
            "fn_completions": "openai_completions",
            "completions_kwargs": {
                "model_name": judge_model_name,
                "max_tokens": 1,
                "temperature": 1,
                "logprobs": True,
                "top_logprobs": 5,
                "requires_chatml": True,
            },
            "fn_completion_parser": "logprob_parser",
            "completion_parser_kwargs": {
                "numerator_token": "m",
                "denominator_tokens": ["m", "M"],
                "is_binarize": False,
            },
            "completion_key": "completions_all",
            "batch_size": 1,
        }
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def create_slurm_script(models_to_judge, script_path, config_path, args, judge_model_path, judge_port):
    job_name = "alpaca-judge-alpha-" + (models_to_judge[0] if len(models_to_judge) == 1 else f"batch-{len(models_to_judge)}")

    log_dir = f"{LOGS_DIR}/judgments"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    openai_config_file = f"{CONFIGS_DIR}/openai_judge_config_{job_name}.yaml"

    save_raw_flag = "--save-raw" if args.save_raw else ""
    save_lc_flag = "--save-length-controlled" if args.save_length_controlled else ""
    gdn_prefill_flag = f"--gdn-prefill-backend {args.gdn_prefill_backend}" if args.gdn_prefill_backend else ""
    tp_size = args.tensor_parallel_size

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --error={log_dir}/{job_name}.err
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --partition=alpha
#SBATCH --gres=gpu:{tp_size}

set -e

echo "Setting up the environment for AlpacaEval judgment (alpha/A100, TP={tp_size})..."
source {WORKSPACE_ROOT}/venv-alpacaeval/bin/activate
PYTHON_EXEC={WORKSPACE_ROOT}/venv-alpacaeval/bin/python
module load CUDA
export PATH={WORKSPACE_ROOT}/venv-alpacaeval/bin:$PATH
source {WORKSPACE_ROOT}/cache.sh

nvidia-smi --query-gpu=index,name,memory.total --format=csv

JUDGE_PORT={judge_port}
OPENAI_CONFIG_FILE={openai_config_file}

cat > $OPENAI_CONFIG_FILE <<EOF
default:
  - api_key: "EMPTY"
    base_url: "http://localhost:$JUDGE_PORT/v1"
EOF

echo "Starting judge server sharded across {tp_size} GPUs on port $JUDGE_PORT..."
$PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model "{judge_model_path}" \
    --max-model-len 26304 \
    --port $JUDGE_PORT \
    --tensor-parallel-size {tp_size} \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.90 \
    --served-model-name "{args.judge_model}" \
    {gdn_prefill_flag} \
    > {log_dir}/{job_name}_vllm_judge.log 2>&1 &
JUDGE_PID=$!

sleep 5
if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
    echo "ERROR: Judge server failed to start."
    tail -n 50 {log_dir}/{job_name}_vllm_judge.log
    exit 1
fi

MAX_WAIT=5400  # 90 min: observed ~55-60 min just to load the 76GB judge checkpoint's 8 shards on this Lustre filesystem
ELAPSED=0
SLEEP_INTERVAL=30
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$JUDGE_PORT/health > /dev/null 2>&1; then
        echo "Judge server is ready after $ELAPSED seconds."
        break
    fi
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
    if ! kill -0 $JUDGE_PID > /dev/null 2>&1; then
        echo "ERROR: Judge server process died."
        tail -n 50 {log_dir}/{job_name}_vllm_judge.log
        exit 1
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Judge server failed to become ready within $MAX_WAIT seconds"
    tail -n 100 {log_dir}/{job_name}_vllm_judge.log
    kill $JUDGE_PID
    exit 1
fi

sleep 10

echo "Verifying judge can actually serve requests (vLLM /health can return ready before torch.compile/CUDA-graph warmup finishes, which takes longer for a TP={tp_size}-sharded 80B judge)..."
PROBE_READY=0
PROBE_MAX_WAIT=1800
PROBE_ELAPSED=0
PROBE_INTERVAL=20
while [ $PROBE_ELAPSED -lt $PROBE_MAX_WAIT ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{{http_code}}" -X POST "http://localhost:$JUDGE_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{{"model": "{args.judge_model}", "messages": [{{"role": "user", "content": "hi"}}], "max_tokens": 1}}')
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

for model in {' '.join(models_to_judge)}; do
    $PYTHON_EXEC {RUN_SCRIPT} \
        --model-name "$model" \
        --annotators-config "{config_path}" \
        --dataset-file "{args.dataset_file}" \
        --dataset-repo "{args.dataset_repo}" \
        --openai-config-path "$OPENAI_CONFIG_FILE" \
        {save_raw_flag} \
        {save_lc_flag}
done

echo "Judgment complete. Stopping judge server..."
kill $JUDGE_PID
sleep 10
"""

    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)


def main():
    parser = argparse.ArgumentParser(description="Automate AlpacaEval judgments with a VLLM judge on the alpha (A100) partition")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to judge, or a single .txt file containing model names",
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default=f"{WORKSPACE_ROOT}/alpaca_eval_models_to_test.txt",
        help="File containing list of models to judge",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit jobs")
    parser.add_argument("--submit", action="store_true", help="Submit jobs after generating scripts")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--dataset-file", default="alpaca_eval_gpt4_baseline.json")
    parser.add_argument("--dataset-repo", default="tatsu-lab/alpaca_eval")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-length-controlled", action="store_true")
    parser.add_argument("--gdn-prefill-backend", default="triton")
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TP_SIZE,
                        help=(
                            f"Number of A100 40GB GPUs to shard the judge server across via vLLM "
                            f"tensor parallelism (default: {DEFAULT_TP_SIZE}). Also sets "
                            "--gres=gpu:N in the generated SLURM script. TP=2 does not fit an "
                            "~80B FP8 judge in 40GB; TP=4 leaves ~19GB/GPU headroom for KV cache."
                        ))
    parser.add_argument("--dependency", type=str, default="",
                        help="SLURM dependency string passed to sbatch (e.g. afterok:12345:67890). "
                             "Skips missing-output validation since generation will finish before this job runs.")

    args = parser.parse_args()

    create_directories()

    api_config = load_api_config()

    if args.models:
        if len(args.models) == 1 and args.models[0].endswith(".txt"):
            models_to_judge = load_models_from_file(args.models[0])
        else:
            models_to_judge = args.models
    else:
        models_to_judge = load_models_from_file(args.models_file)
        if not models_to_judge:
            print(f"No models found in {args.models_file}. Use --models to specify models.")
            return

    missing_outputs = []
    for model in models_to_judge:
        outputs_path = Path(OUTPUTS_DIR) / model / "model_outputs.json"
        if not outputs_path.exists():
            missing_outputs.append(model)

    if missing_outputs:
        print("\nWARNING: Missing model outputs for:")
        for model in missing_outputs:
            print(f"  - {model}")
        if args.dependency:
            print("\nDependency set — judgment jobs will wait for generation to finish before running.")
        elif not args.dry_run:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                return

    judge_config_entry = api_config.get(args.judge_model)
    judge_model_path = resolve_model_path(judge_config_entry) if judge_config_entry else None
    if not judge_model_path:
        print(f"ERROR: Judge model '{args.judge_model}' not found in api_config.yaml")
        return

    judge_port = get_port_for_model(api_config, model_name=args.judge_model, default=8001)

    config_path = f"{CONFIGS_DIR}/alpaca_eval_judge_{args.judge_model}.yaml"
    create_judge_config(config_path, args.judge_model, args.prompt_template)

    model_batches = [
        models_to_judge[i : i + args.batch_size]
        for i in range(0, len(models_to_judge), args.batch_size)
    ]

    job_scripts = []
    for batch_idx, model_batch in enumerate(model_batches):
        script_path = f"{SCRIPTS_DIR}/run_alpaca_eval_judgment_alpha_batch_{batch_idx + 1}.sh"
        create_slurm_script(model_batch, script_path, config_path, args, judge_model_path, judge_port)
        job_scripts.append(script_path)
        print(f"Created script: {script_path}")

    print(f"\nGenerated {len(job_scripts)} judgment job scripts in {SCRIPTS_DIR}")

    if args.dry_run:
        print("\nDry run complete. Scripts generated but not submitted.")
        print("To submit jobs manually, run:")
        for script in job_scripts:
            print(f"  sbatch {script}")
    elif args.submit:
        print("\nSubmitting judgment jobs...")
        submitted_jobs = []
        for script in job_scripts:
            try:
                sbatch_cmd = ["sbatch"]
                if args.dependency:
                    sbatch_cmd += ["--dependency", args.dependency]
                sbatch_cmd.append(script)
                result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
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
