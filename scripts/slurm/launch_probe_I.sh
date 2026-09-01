#!/usr/bin/env bash
# Launch Probe I: spline-only encoder (no SiLU base) detailed bakeoff.
#
# Arms:
#   linear | silu_base | kan | spline_only (update_grid on) | spline_only_nogrid
# Recipe: unlocked decoder, θ calibrated then locked, λ=1e-6, target_l0 ∈ {32,64,112}
# d_t ∈ {512, 2048}
#
# Usage: bash scripts/slurm/launch_probe_I.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/slurm /gscratch/ssuresh/results/paper_probes/I_spline_only

STEPS="${STEPS:-3000}"
LAMBDA="${LAMBDA:-1e-6}"
SBATCH=scripts/slurm/paper_probes.sbatch

ARMS=(linear silu_base kan spline_only spline_only_nogrid)
DTS=(512 2048)
TARGETS=(32 64 112)

submit_cell() {
  local arm="$1" dt="$2" target="$3"
  local job_name="I-${arm}-dt${dt}-t${target}"
  echo "submit arm=$arm d_t=$dt target_l0=$target λ=$LAMBDA steps=$STEPS name=$job_name"
  sbatch --job-name="$job_name" "$SBATCH" I "$arm" "$dt" "$target" "$STEPS" "$LAMBDA"
}

echo "[launch_I] STEPS=$STEPS λ=$LAMBDA arms=${ARMS[*]} targets=${TARGETS[*]}"
echo "[launch_I] → /gscratch/ssuresh/results/paper_probes/I_spline_only"

n=0
for arm in "${ARMS[@]}"; do
  for dt in "${DTS[@]}"; do
    for target in "${TARGETS[@]}"; do
      submit_cell "$arm" "$dt" "$target"
      n=$((n + 1))
    done
  done
done

echo "[launch_I] submitted $n cells. After finish:"
echo "  sbatch scripts/slurm/paper_probes.sbatch I_agg"
