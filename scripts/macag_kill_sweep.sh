#!/usr/bin/env bash
# Kill orphaned MACAG sweep workers (shard shells, pipeline, GPU python).
#
# Use after Ctrl+C if nvidia-smi still shows macag/python processes, or before
# relaunching a sweep on a busy GPU.
#
# Usage:
#   scripts/macag_kill_sweep.sh                  # all MACAG processes for $USER
#   scripts/macag_kill_sweep.sh results/macag_mib  # only that OUTROOT
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTROOT="${1:-}"
# shellcheck source=scripts/macag_parallel_common.sh
source scripts/macag_parallel_common.sh

echo ">>> Killing MACAG sweep processes for user=${USER:-$(whoami)}${OUTROOT:+ (OUTROOT=$OUTROOT)} ..."
macag_kill_macag_procs "$OUTROOT"

echo ">>> Remaining GPU compute apps (if any):"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null \
    || echo "(none or nvidia-smi unavailable)"
else
  echo "(nvidia-smi not found)"
fi
