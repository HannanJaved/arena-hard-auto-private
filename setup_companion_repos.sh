#!/bin/bash
# Clones the JudgeArena (MT-Bench) and OpenJury (ELO) companion repos as siblings
# of this arena-hard-auto checkout. automate_mtbench.py, automate_elo_estimation.py,
# and submit_evals.py all expect them at that fixed sibling location.
#
# Run this once after cloning arena-hard-auto into a new workspace:
#   ./setup_companion_repos.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

clone_or_skip() {
    local name="$1" url="$2" branch="$3"
    local dest="$WORKSPACE_ROOT/$name"
    if [ -d "$dest/.git" ]; then
        echo "$name already present at $dest - skipping."
        return
    fi
    echo "Cloning $name ($branch) into $dest..."
    git clone --branch "$branch" "$url" "$dest"
}

clone_or_skip "JudgeArena" "https://github.com/OpenEuroLLM/JudgeArena.git" "main"
clone_or_skip "OpenJury" "https://github.com/OpenEuroLLM/OpenJury.git" "elo"

echo ""
echo "Done. Companion repos are ready alongside arena-hard-auto at: $WORKSPACE_ROOT"
echo ""
echo "NOTE: WORKSPACE_ROOT is still hardcoded (not auto-detected) in:"
echo "  - arena-hard-auto/scripts/submit_evals.py"
echo "  - JudgeArena/scripts/automate_mtbench.py"
echo "  - OpenJury/scripts/automate_elo_estimation.py"
echo "If this workspace's path differs from what those files currently hardcode"
echo "($WORKSPACE_ROOT expected), update WORKSPACE_ROOT/WORKSPACE in each before"
echo "running evals - this script only places the repos, it doesn't repoint them."
