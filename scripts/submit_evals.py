#!/usr/bin/env python3
"""
submit_evals.py — submit all evaluation jobs for a list of models.

Usage:
    python submit_evals.py --models-file models.txt [--submit] [--dry-run]

Venv assignments:
    arena-hard-auto/venv  → arena-hard generation
    venv-alpacaeval       → alpaca-eval generation + judgment
    venv-openjury         → MT-Bench (JudgeArena) + ELO estimation (OpenJury)
    venv-lm-eval          → arc_challenge, gpqa, gsm8k, hellaswag, ifeval, piqa, truthfulqa
"""

import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/data/horse/ws/hama901h-BFTranslation")
SCRIPTS = WORKSPACE / "arena-hard-auto" / "scripts"

VENV_ARENA    = WORKSPACE / "arena-hard-auto" / "venv" / "bin" / "python"
VENV_ALPACA   = WORKSPACE / "venv-alpacaeval" / "bin" / "python"
VENV_OPENJURY = WORKSPACE / "venv-openjury" / "bin" / "python"
VENV_LMEVAL   = WORKSPACE / "venv-lm-eval" / "bin" / "python"

STATIC_TASKS = ["arc_challenge", "gpqa", "gsm8k", "hellaswag", "ifeval", "piqa", "truthfulqa"]


def run(label: str, python: Path, script: Path, extra_args: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    cmd = [str(python), str(script)] + extra_args
    print("  CMD:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: {label} exited with code {result.returncode}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit all evaluation jobs for a model list.")
    parser.add_argument("--models-file", required=True, help="Text file with one model name per line.")
    parser.add_argument("--submit",  action="store_true", help="Submit SLURM jobs (automation scripts).")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit.")
    parser.add_argument("--baseline", required=True, help="Baseline model for MT-Bench.")
    parser.add_argument("--rerun", action="store_true", help="Re-run MT-Bench for all (no skipping).")
    args = parser.parse_args()

    models_file = str(args.models_file)

    # For the automation scripts (arena-hard, alpaca-eval, mtbench, elo):
    #   --submit  → pass --submit
    #   --dry-run → pass --dry-run
    #   neither   → generate only (no flag)
    auto_flag = ["--submit"] if args.submit else (["--dry-run"] if args.dry_run else [])

    # For the static submit_*_from_list scripts:
    #   they submit by default; --dry-run suppresses submission
    static_flag = ["--dry-run"] if (args.dry_run or not args.submit) else []

    # 1. Arena-Hard generation
    run(
        "Arena-Hard generation (automate_arena_hard_generation_olmo3)",
        VENV_ARENA,
        SCRIPTS / "automate_arena_hard_generation_olmo3.py",
        ["--models-file", models_file] + auto_flag,
    )

    # 2. AlpacaEval generation
    run(
        "AlpacaEval generation (automate_alpaca_eval)",
        VENV_ALPACA,
        SCRIPTS / "automate_alpaca_eval.py",
        ["--models-file", models_file] + auto_flag,
    )

    # 3. MT-Bench / JudgeArena
    # if args.rerun:
    #     run(
    #         "MT-Bench / JudgeArena (automate_mtbench)",
    #         VENV_OPENJURY,
    #         WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench.py",
    #         ["--models-file", models_file, "--baseline-model", args.baseline, "--rerun-all"] + auto_flag,
    #     )
    # else:
    #     run(
    #         "MT-Bench / JudgeArena (automate_mtbench)",
    #         VENV_OPENJURY,
    #         WORKSPACE / "JudgeArena" / "scripts" / "automate_mtbench.py",
    #         ["--models-file", models_file, "--baseline-model", args.baseline, "--skip-existing"] + auto_flag,
        # )

    # 4. ELO estimation / OpenJury
    run(
        "ELO estimation (automate_elo_estimation)",
        VENV_OPENJURY,
        WORKSPACE / "OpenJury" / "scripts" / "automate_elo_estimation.py",
        ["--models-file", models_file] + auto_flag,
    )

    # 5. Static evals
    for task in STATIC_TASKS:
        run(
            f"Static eval: {task}",
            VENV_LMEVAL,
            SCRIPTS / f"submit_{task}_from_list.py",
            ["--models-file", models_file] + static_flag,
        )

    print(f"\n{'='*60}")
    print("Evaluation Jobs Submitted.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
