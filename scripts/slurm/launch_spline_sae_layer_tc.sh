#!/usr/bin/env bash
# Launch Gemma L24 full-block transcoder (resid_pre → Δ resid_post).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm spline_sae/baselines

JOBS=()
for cfg in \
  spline_sae/configs/probe_gemma_layer_tc_l24_kan_basejump.yaml \
  spline_sae/configs/probe_gemma_layer_tc_l24_linear_basejump_ctrl.yaml
do
  jid=$(sbatch --parsable scripts/slurm/spline_sae_run.sbatch "$cfg")
  echo "submitted ${jid}  ${cfg}"
  JOBS+=("$jid")
done
printf '%s\n' "${JOBS[@]}" > spline_sae/baselines/basejump_gemma_layer_tc_l24_jobids.txt
echo "Submitted ${#JOBS[@]} layer transcoder L24 (resid_pre→Δ resid_post) jobs"
