#!/usr/bin/env python3
"""Cancel pending static LM-eval jobs whose results already exist (e.g. because the romeo copy finished).

Job names are <task>_<model> (submit_evals.py). Uses submit_evals' own completion check, so a task only
counts as done when its results_*.json is on disk. Dry run unless --go. --watch repeats every 10 minutes.
Run with arena-hard-auto/venv/bin/python (needs pyyaml).
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit_evals as se  # noqa: E402

NAME = re.compile(r"^(%s)_(.+)$" % "|".join(se.STATIC_TASKS))


def sweep(go):
    cfg = se._load_api_config()
    out = subprocess.run(["squeue", "--me", "-h", "-t", "PENDING", "-o", "%i|%j"], capture_output=True, text=True).stdout
    done = []
    for line in out.splitlines():
        jid, name = line.split("|", 1)
        m = NAME.match(name)
        if m and se._is_lmeval_completed(m.group(1), m.group(2), cfg, None):
            done.append((jid, name))
    for jid, name in done:
        print(("cancel " if go else "would cancel ") + f"{jid} {name}")
    if go and done:
        subprocess.run(["scancel"] + [j for j, _ in done], check=True)
    print(f"{len(done)} pending static jobs already have results")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    while True:
        sweep(a.go)
        if not a.watch:
            break
        time.sleep(600)
