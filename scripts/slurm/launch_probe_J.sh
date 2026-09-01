#!/usr/bin/env bash
# Launch Probe J: Tier-1 routing knobs on full KAN (dt=2048, target_l0=64).
#
# B2: scale_base=0.2, scale_spline=1.0, lr_spline_mult=1 (λ_kan=0)
# B3: same scales + lr_spline_mult=5
#
# Recipe matches Probe I: unlocked decoder, locked θ, λ=1e-6, 3000 steps.
# Compare to I_spline_only/{kan,silu_base}_dt2048_t64.
#
# Usage: bash scripts/slurm/launch_probe_J.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/slurm /gscratch/ssuresh/results/paper_probes/J_tier1_routing

STEPS="${STEPS:-20000}"
LAMBDA="${LAMBDA:-1e-5}"
D_T="${D_T:-2048}"
TARGET="${TARGET:-64}"
SBATCH=scripts/slurm/paper_probes.sbatch
# 20k steps ≈ 6–7× Probe I (3k); give headroom on GH200.
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

echo "[launch_J] STEPS=$STEPS λ_peak=$LAMBDA (r5b: warmup 2000, decay 4500→3e-6, bw=0.12)"
echo "[launch_J] d_t=$D_T target_l0=$TARGET variants=B2,B3 time=$TIME_LIMIT"
echo "[launch_J] → /gscratch/ssuresh/results/paper_probes/J_tier1_routing"

for variant in B2 B3; do
  job_name="J-${variant}-dt${D_T}-t${TARGET}"
  echo "submit variant=$variant d_t=$D_T target_l0=$TARGET name=$job_name"
  sbatch --job-name="$job_name" --time="$TIME_LIMIT" "$SBATCH" J "$variant" "$D_T" "$TARGET" "$STEPS" "$LAMBDA"
done

echo "[launch_J] submitted 2 cells."
echo "  Compare late spline_contribution_frac + rel_fro to:"
echo "    I_spline_only/kan_dt${D_T}_t${TARGET}"
echo "    I_spline_only/silu_base_dt${D_T}_t${TARGET}"
