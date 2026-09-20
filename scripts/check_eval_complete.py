#!/usr/bin/env python3
"""Check whether every eval task (7 static LM-eval + 6 dynamic: Arena-Hard
gen/judge, AlpacaEval gen/judge, MT-Bench, ELO) is complete for every model
in a models-file, reusing submit_evals.py's own completion-check functions
(the same ones --skip-completed uses) so "complete" here means exactly what
it would mean to a fresh submit_evals.py run.

Exit code 0 = fully complete (nothing left for submit_evals.py to do).
Exit code 1 = still missing something (prints a per-task breakdown).

Usage:
    python check_eval_complete.py --models-file ../../qwen3-1.7b-ultrafeedback-dpo-ao.txt --baseline qwen3-1.7b-sft-lr3e-5-AO
    python check_eval_complete.py --models-file ... --baseline ... --quiet   # exit code only, no per-task printout
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit_evals as se  # noqa: E402


def check(models_file: str, baseline: str, judge_model: str = "Qwen3-Next-80B-A3B-Instruct-FP8") -> tuple[bool, dict[str, tuple[int, int]]]:
    models = [line.strip() for line in open(models_file) if line.strip() and not line.strip().startswith("#")]
    api_config = se._load_api_config()
    judge_logical = se._resolve_judge_logical_name(api_config, judge_model)

    results: dict[str, tuple[int, int]] = {}
    counts = {task: 0 for task in se.AUTOMATION_TASKS + se.STATIC_TASKS}
    for m in models:
        counts["arena_hard_generation"] += se._is_arena_hard_completed(m)
        counts["arena_hard_judgment"] += se._is_arena_hard_judgment_completed(m, judge_logical, baseline)
        counts["alpaca_eval_generation"] += se._is_alpaca_eval_completed(m)
        counts["alpaca_eval_judgment"] += se._is_alpaca_eval_judgment_completed(m)
        counts["mtbench"] += se._is_mtbench_completed(m, api_config)
        counts["elo"] += se._is_elo_completed(m, api_config)
        for task in se.STATIC_TASKS:
            counts[task] += se._is_lmeval_completed(task, m, api_config, None)

    for task, done in counts.items():
        results[task] = (done, len(models))

    fully_done = all(done == total for done, total in results.values())
    return fully_done, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-file", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--judge-model", default="Qwen3-Next-80B-A3B-Instruct-FP8")
    parser.add_argument("--quiet", action="store_true", help="Exit code only, no printout.")
    args = parser.parse_args()

    fully_done, results = check(args.models_file, args.baseline, args.judge_model)

    if not args.quiet:
        label = Path(args.models_file).stem
        print(f"=== {label} ===")
        for task, (done, total) in results.items():
            marker = "OK" if done == total else "--"
            print(f"  [{marker}] {task:<25} {done:>3}/{total}")
        print("COMPLETE" if fully_done else "NOT COMPLETE")

    sys.exit(0 if fully_done else 1)


if __name__ == "__main__":
    main()
