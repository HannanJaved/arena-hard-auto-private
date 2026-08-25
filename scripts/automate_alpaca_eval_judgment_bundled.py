#!/usr/bin/env python3
"""
Bundle AlpacaEval judgment jobs by baseline.

The per-model automate_alpaca_eval_judgment.py already supports --batch-size
(multiple models judged one-by-one in a bash loop against a single judge-server
start) but only for one --dataset-file (AlpacaEval's baseline reference
answers) per invocation. This wrapper groups an input list of models by
baseline first, then reuses automate_alpaca_eval_judgment.py's own
config/script generation per group so the resulting SLURM jobs (judge server
startup, one-by-one judging loop) are unchanged - only the batching is new.

Models file format (one entry per line, '#' comments allowed):
    model_name
    model_name,dataset_file
    model_name<TAB>dataset_file
A line without an explicit dataset_file uses --dataset-file.
"""

import argparse
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import automate_alpaca_eval_judgment as base

DEFAULT_BATCH_SIZE = 3


def load_models_with_baseline(file_path, default_baseline):
    """Parse a models file into an ordered list of (model_name, baseline) tuples."""
    entries = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                model, baseline = (p.strip() for p in line.split(",", 1))
            elif "\t" in line:
                model, baseline = (p.strip() for p in line.split("\t", 1))
            else:
                parts = line.split()
                model, baseline = (parts[0], parts[1]) if len(parts) == 2 else (line, default_baseline)
            entries.append((model, baseline or default_baseline))
    return entries


def group_by_baseline(entries):
    groups = OrderedDict()
    for model, baseline in entries:
        groups.setdefault(baseline, []).append(model)
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Bundle AlpacaEval judgment jobs: models sharing the same baseline "
                    "(dataset-file) are judged one-by-one in a single SLURM job, up to "
                    "--batch-size per job."
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Model names to judge (space separated), all using --dataset-file as baseline. "
             "For mixed baselines, use --models-file instead.",
    )
    parser.add_argument(
        "--models-file", type=str,
        default=f"{base.WORKSPACE_ROOT}/alpaca_eval_models_to_test.txt",
        help="File with one model per line; optionally 'model,dataset_file' to mix baselines.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Max number of models judged per bundled job (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit jobs")
    parser.add_argument("--submit", action="store_true", help="Submit jobs after generating scripts")
    parser.add_argument("--judge-model", default=base.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--prompt-template", default=base.DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--dataset-file", default="alpaca_eval_gpt4_baseline.json",
        help="Default baseline (reference-answer dataset file) for models without an explicit one.",
    )
    parser.add_argument("--dataset-repo", default="tatsu-lab/alpaca_eval")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-length-controlled", action="store_true")
    parser.add_argument("--gdn-prefill-backend", default="triton")
    parser.add_argument(
        "--dependency", type=str, default="",
        help="SLURM dependency string passed to sbatch (e.g. afterok:12345:67890), applied to every bundled job.",
    )
    args = parser.parse_args()

    base.create_directories()
    api_config = base.load_api_config()

    if args.models:
        entries = [(m, args.dataset_file) for m in args.models]
    else:
        entries = load_models_with_baseline(args.models_file, args.dataset_file)
        if not entries:
            print(f"No models found in {args.models_file}. Use --models to specify models directly.")
            return

    groups = group_by_baseline(entries)
    print(f"Found {len(entries)} models across {len(groups)} baseline (dataset-file) group(s):")
    for baseline, models in groups.items():
        print(f"  baseline '{baseline}': {len(models)} model(s)")

    all_missing_outputs = []
    for baseline, models in groups.items():
        for model in models:
            outputs_path = Path(base.OUTPUTS_DIR) / model / "model_outputs.json"
            if not outputs_path.exists():
                all_missing_outputs.append(model)

    if all_missing_outputs:
        print("\nWARNING: Missing model outputs for:")
        for model in all_missing_outputs:
            print(f"  - {model}")
        if args.dependency:
            print("\nDependency set — judgment jobs will wait for generation to finish before running.")
        elif not args.dry_run:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                return

    judge_config_entry = api_config.get(args.judge_model)
    judge_model_path = base.resolve_model_path(judge_config_entry) if judge_config_entry else None
    if not judge_model_path:
        print(f"ERROR: Judge model '{args.judge_model}' not found in api_config.yaml")
        return

    judge_config_path = f"{base.CONFIGS_DIR}/alpaca_eval_judge_{args.judge_model}.yaml"
    base.create_judge_config(judge_config_path, args.judge_model, args.prompt_template)

    job_scripts = []
    port_offset = 0
    for baseline, models in groups.items():
        model_batches = [models[i:i + args.batch_size] for i in range(0, len(models), args.batch_size)]
        print(f"\nBaseline '{baseline}': {len(model_batches)} bundled job(s), batch size {args.batch_size}")

        for model_batch in model_batches:
            # Unique port across all groups/batches (8001, 8002, ...) to avoid stale :8000 conflicts.
            judge_port = base.DEFAULT_JUDGE_SERVER_PORT_BASE + port_offset
            port_offset += 1

            model_id = model_batch[0] if len(model_batch) == 1 else "__".join(model_batch)
            script_path = f"{base.SCRIPTS_DIR}/run_alpaca_eval_judgment_{base.safe_batch_id(model_id)}.sh"

            # Per-batch args namespace so each group's --dataset-file (baseline) flows
            # into create_slurm_script's `for model in ...` loop correctly.
            batch_args = argparse.Namespace(**vars(args))
            batch_args.dataset_file = baseline

            base.create_slurm_script(model_batch, script_path, judge_config_path, batch_args, judge_model_path, judge_port)
            print(f"  Created: {script_path} ({len(model_batch)} models: {', '.join(model_batch)}, judge port {judge_port})")
            job_scripts.append(script_path)

    print(f"\nGenerated {len(job_scripts)} bundled judgment job scripts in {base.SCRIPTS_DIR}")

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
    else:
        print("\nTo submit jobs, run:")
        print(f"  python {__file__} --submit ...")
        print("\nOr submit individual jobs with:")
        for script in job_scripts[:3]:
            print(f"  sbatch {script}")
        if len(job_scripts) > 3:
            print(f"  ... and {len(job_scripts) - 3} more")


if __name__ == "__main__":
    main()
