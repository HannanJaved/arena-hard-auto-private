#!/usr/bin/env python3
"""
Bundle Arena Hard judgment jobs by baseline.

The per-model automate_arena_hard_judgment.py already supports --batch-size
(multiple models judged one-by-one against a single judge-server start, via
gen_judgment.py's model_list loop) but only for one --baseline per
invocation. This wrapper groups an input list of models by baseline first,
then reuses automate_arena_hard_judgment.py's own config/script generation
per group so the resulting SLURM jobs (judge server startup, one-by-one
judging loop, result handling) are unchanged - only the batching is new.

Models file format (one entry per line, '#' comments allowed):
    model_name
    model_name,baseline
    model_name<TAB>baseline
A line without an explicit baseline uses --baseline.
"""

import argparse
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import automate_arena_hard_judgment as base

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
        description="Bundle Arena Hard judgment jobs: models sharing the same baseline are "
                    "judged one-by-one in a single SLURM job, up to --batch-size per job."
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Model names to judge (space separated), all using --baseline. "
             "For mixed baselines, use --models-file instead.",
    )
    parser.add_argument(
        "--models-file", type=str,
        default=f"{base.WORKSPACE_ROOT}/arena_hard_models_to_test.txt",
        help="File with one model per line; optionally 'model,baseline' to mix baselines.",
    )
    parser.add_argument(
        "--baseline", type=str, default=base.DEFAULT_BASELINE,
        help="Default baseline for models without an explicit baseline "
             f"(alias: {', '.join(base.BASELINE_CONFIGS.keys())}, or a literal model name from api_config.yaml).",
    )
    parser.add_argument("--judge-model", type=str, default=base.JUDGE_MODEL)
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Max number of models judged per bundled job (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit jobs")
    parser.add_argument("--submit", action="store_true", help="Submit jobs after generating scripts")
    parser.add_argument("--validate-only", action="store_true", help="Only validate models without generating scripts")
    parser.add_argument(
        "--dependency", type=str, default="",
        help="SLURM dependency string passed to sbatch (e.g. afterok:12345:67890), applied to every bundled job.",
    )
    args = parser.parse_args()

    base.create_directories()
    api_config = base.load_api_config()

    if args.models:
        entries = [(m, args.baseline) for m in args.models]
    else:
        entries = load_models_with_baseline(args.models_file, args.baseline)
        if not entries:
            print(f"No models found in {args.models_file}. Use --models to specify models directly.")
            return

    groups = group_by_baseline(entries)
    print(f"Found {len(entries)} models across {len(groups)} baseline group(s):")
    for baseline, models in groups.items():
        print(f"  baseline '{baseline}': {len(models)} model(s)")

    all_missing_models, all_missing_answers = [], []
    for baseline, models in groups.items():
        missing_models, missing_answers = base.validate_models_exist(models, baseline, api_config)
        all_missing_models += missing_models
        all_missing_answers += missing_answers

    if all_missing_models:
        print("\nERROR: The following models/baselines are not found in API config:")
        for m in all_missing_models:
            print(f"  - {m}")
        return

    if all_missing_answers:
        print("\nWARNING: The following models don't have generated answers yet:")
        for m in all_missing_answers:
            print(f"  - {m}")
        print("\nYou need to generate answers first before judging.")
        if args.dependency:
            print("Dependency set — judgment jobs will wait for generation to finish before running.")
        elif not args.validate_only and not args.dry_run:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                return

    if args.validate_only:
        print(f"\nValidation complete. {len(entries)} models ready for judgment across {len(groups)} baseline group(s).")
        return

    job_scripts = []
    for baseline, models in groups.items():
        baseline_model_name = base.resolve_baseline_model(baseline, api_config)
        model_batches = [models[i:i + args.batch_size] for i in range(0, len(models), args.batch_size)]
        print(f"\nBaseline '{baseline}' ({baseline_model_name}): {len(model_batches)} bundled job(s), batch size {args.batch_size}")

        judge_key = args.judge_model
        judge_logical = base.resolve_judge_logical_name(api_config, judge_key)
        model_path = base.extract_judge_path_from_api_config(api_config, judge_key)

        for model_batch in model_batches:
            model_id = model_batch[0] if len(model_batch) == 1 else "__".join(model_batch)
            combined_id = f"baseline_{baseline_model_name}__{model_id}"
            safe_model_id = base.safe_batch_id(combined_id)
            config_path = f"{base.CONFIGS_DIR}/arena_hard_judgment_{safe_model_id}.yaml"
            script_path = f"{base.SCRIPTS_DIR}/run_arena_hard_judgment_{safe_model_id}.sh"
            judge_port = base.judge_port_for_job(safe_model_id)

            base.create_judgment_config(
                model_batch, config_path, baseline, judge_model=judge_logical, api_config=api_config,
            )
            base.create_judgment_slurm_script(
                model_batch, script_path, config_path, baseline,
                judge_model=judge_logical, judge_path=model_path,
                judge_port=judge_port, api_config=api_config,
            )
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

        print(f"\nSuccessfully submitted {len(submitted_jobs)} judgment jobs")
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
