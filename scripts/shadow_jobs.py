#!/usr/bin/env python3
"""Shadow copies of pending capella jobs under the other Slurm account; whichever copy starts first wins.

  shadow_jobs.py clone  [--go]   copy every pending job of --from-account (name filter --name-regex) to --to-account
  shadow_jobs.py watch           poll the queue; when one copy leaves PENDING, scancel its pending sibling
  shadow_jobs.py undo   [--go]   scancel the still-pending copies

A copy is `sbatch` of the original's batch script (scontrol write batch_script), same name/logs/resources.
Dependencies: a job waiting on afterok:G is rewritten as afterok:G?afterok:G' (G' = G's copy), for the original
and its copy, so whichever generation job survives still releases both judgment jobs.
Pairs are kept in --state. Default is a dry run; clone and undo change nothing without --go.
Duplicates never write at the same time in practice: the watcher polls every 20 s and a job takes minutes to
load its model before it writes any output, but `watch` must be running.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WS = Path("/data/horse/ws/hama901h-BFTranslation")
STATE = WS / "shadow_pairs.json"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)}: {r.stderr.strip()}")
    return r.stdout


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(pairs):
    STATE.write_text(json.dumps(pairs, indent=1))


def pending_jobs():
    """[(id, name, account, dependency ids)] of my pending jobs, ascending id."""
    rows = []
    for line in sh("squeue", "--me", "-h", "-t", "PENDING", "-o", "%i|%j|%a|%E").splitlines():
        jid, name, acct, dep = line.split("|", 3)
        rows.append((jid, name, acct, re.findall(r"after\w+:(\d+)", dep)))
    return sorted(rows, key=lambda r: int(r[0]))


def shadow_dependency(dep_ids, pairs):
    """OR-dependency string for a job waiting on dep_ids, or None if unchanged / unsupported."""
    if len(dep_ids) != 1:
        return None
    d = dep_ids[0]
    twin = pairs.get(d) or next((o for o, c in pairs.items() if c == d), None)
    return f"afterok:{d}?afterok:{twin}" if twin else f"afterok:{d}"


def cmd_clone(a):
    pairs = load()
    done = set(pairs) | set(pairs.values())
    rx = re.compile(a.name_regex)
    todo = [j for j in pending_jobs() if j[2] == a.from_account and rx.search(j[1]) and j[0] not in done]
    print(f"{len(todo)} pending {a.from_account} jobs match; copies go to {a.to_account}")
    for jid, name, _, deps in todo:
        if len(deps) > 1:
            print(f"  skip {jid} {name}: {len(deps)} dependencies")
            continue
        dep = shadow_dependency(deps, pairs)
        print(f"  {jid} {name}" + (f"  dep {dep}" if deps else ""))
        if not a.go:
            continue
        script = Path(tempfile.mkdtemp()) / f"{jid}.sh"
        sh("scontrol", "write", "batch_script", jid, str(script))
        cmd = ["sbatch", "--parsable", "--account", a.to_account, "--job-name", name] + ([f"--dependency={dep}"] if deps else [])
        new = sh(*cmd, str(script)).strip().split(";")[0]
        if deps and "?" in dep:
            sh("scontrol", "update", f"JobId={jid}", f"Dependency={dep}")   # original follows whichever twin survives
        pairs[jid] = new
        save(pairs)
    if not a.go:
        print("dry run; pass --go to submit")


def decide(states, pairs, finished_ok=lambda jid: False):
    """Job ids to cancel: the PENDING twin of a pair whose other member has started (or finished OK)."""
    cancel = []
    for a, b in pairs.items():
        sa, sb = states.get(a), states.get(b)
        started = lambda s, j: (s is not None and s != "PENDING") or (s is None and finished_ok(j))
        if sb == "PENDING" and started(sa, a):
            cancel.append(b)
        elif sa == "PENDING" and started(sb, b):
            cancel.append(a)
    return cancel


def cmd_watch(a):
    pairs = load()
    print(f"watching {len(pairs)} pairs every {a.interval}s", flush=True)
    ok = lambda j: sh("sacct", "-j", j, "-X", "-n", "-o", "State").split()[:1] == ["COMPLETED"]
    while True:
        states = dict(l.split() for l in sh("squeue", "--me", "-h", "-o", "%i %T").splitlines())
        for j in decide(states, pairs, ok):
            sh("scancel", j)
            print(time.strftime("%F %T"), f"cancelled {j} (its twin started)", flush=True)
            states[j] = "CANCELLED"
        if not any(states.get(j) == "PENDING" for p in pairs.items() for j in p):
            print("no pending pair members left; done")
            return
        time.sleep(a.interval)


def cmd_undo(a):
    pairs = load()
    states = dict(l.split() for l in sh("squeue", "--me", "-h", "-o", "%i %T").splitlines())
    todo = [c for c in pairs.values() if states.get(c) == "PENDING"]
    print(f"{len(todo)} pending copies")
    if a.go and todo:
        sh("scancel", *todo)
    elif not a.go:
        print("dry run; pass --go to cancel")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clone")
    c.add_argument("--from-account", default="p_scads_nas")
    c.add_argument("--to-account", default="p_neurasearch")
    c.add_argument("--name-regex", default=".", help="only jobs whose name matches (python re.search)")
    c.add_argument("--go", action="store_true")
    w = sub.add_parser("watch")
    w.add_argument("--interval", type=int, default=20)
    u = sub.add_parser("undo")
    u.add_argument("--go", action="store_true")
    a = ap.parse_args()
    {"clone": cmd_clone, "watch": cmd_watch, "undo": cmd_undo}[a.cmd](a)
