#!/usr/bin/env bash
# TMLR-250 v4 Pass A — launch all three CLT sweeps.
#
#   scripts/slurm/submit_all_tmlr250_v4.sh              # cancel existing v4 jobs, then launch
#   scripts/slurm/submit_all_tmlr250_v4.sh --no-cancel  # launch without cancelling
#
# After the arrays finish, aggregate (see macag/docs/run_todo_v4.md):
#   RESULTS_ROOT=/gscratch/$USER/macag_mib_tmlr250v4_h200 \
#     RUN_ANALYSIS=1 SEEDS=0 scripts/run_mib_benchmark.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CANCEL=1
if [[ "${1:-}" == "--no-cancel" ]]; then
  CANCEL=0
fi

if [[ "$CANCEL" == "1" ]]; then
  echo ">>> Cancelling existing v4 jobs for user $USER ..."
  # Match only this campaign's job names (macag_v4_*); never scancel blindly.
  while IFS= read -r jobid; do
    [[ -z "$jobid" ]] && continue
    echo ">>> scancel $jobid"
    scancel "$jobid" || true
  done < <(squeue -h -u "$USER" -o '%i %j' 2>/dev/null | awk '$2 ~ /^macag_v4_/ {print $1}')
else
  echo ">>> --no-cancel: leaving existing v4 jobs running"
fi

sbatch scripts/slurm/submit_macag_mib_h200_v4.sh
sbatch scripts/slurm/submit_macag_mib_gemma25m_v4.sh
sbatch scripts/slurm/submit_macag_mib_llama_v4.sh

echo ">>> Launched v4 Pass A (3 CLTs x 4-node arrays). Watch with: squeue -u \$USER -n macag_v4_h200,macag_v4_gemma25m,macag_v4_llama"
