#!/usr/bin/env bash
# Launch Gemma BaseJump KAN + matched linear STE control.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm spline_sae/baselines

JOBS=()
for cfg in \
  spline_sae/configs/probe_gemma_kan_basejump.yaml \
  spline_sae/configs/probe_gemma_linear_basejump_ctrl.yaml
do
  jid=$(sbatch --parsable scripts/slurm/spline_sae_run.sbatch "$cfg")
  echo "submitted ${jid}  ${cfg}"
  JOBS+=("$jid")
done
printf '%s\n' "${JOBS[@]}" > spline_sae/baselines/basejump_gemma_jobids.txt
echo "Submitted ${#JOBS[@]} BaseJump measure jobs"
