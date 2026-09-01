#!/usr/bin/env bash
# Launch Probe H: matched-L0 bakeoff (open-gate θ, then lock).
#
# Claim tests at L0≈100 and L0≈200:
#   1) silu/kan@512 vs linear@2048 (capacity)
#   2) full kan vs silu_base (does spline help once L0 is healthy?)
#
# θ calibrated to target_l0 then locked; decoder unlocked; λ=1e-6.
# target_l0 grid brackets both bands for linear and SiLU (from G curves).
#
# Usage:
#   bash scripts/slurm/launch_probe_H.sh
#   WAVE=core bash scripts/slurm/launch_probe_H.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/slurm /gscratch/ssuresh/results/paper_probes/H_matched_l0

STEPS="${STEPS:-3000}"
WAVE="${WAVE:-full}"
LAMBDA="${LAMBDA:-1e-6}"
SBATCH=scripts/slurm/paper_probes.sbatch

# Bracket final L0 ~100 and ~200 for both linear and SiLU (G: silu t32→55, t128→238;
# linear t32→108, t128→441).
TARGETS=(32 48 64 80 96 112 128)
DTS=(512 2048)
ARMS=(linear silu_base)
if [[ "$WAVE" == "full" ]]; then
  ARMS+=(kan)
fi

submit_cell() {
  local arm="$1" dt="$2" target="$3"
  local job_name="H-${arm}-dt${dt}-t${target}"
  echo "submit arm=$arm d_t=$dt target_l0=$target λ=$LAMBDA steps=$STEPS name=$job_name"
  sbatch --job-name="$job_name" "$SBATCH" H "$arm" "$dt" "$target" "$STEPS" "$LAMBDA"
}

echo "[launch_H] STEPS=$STEPS WAVE=$WAVE λ=$LAMBDA targets=${TARGETS[*]}"
echo "[launch_H] → /gscratch/ssuresh/results/paper_probes/H_matched_l0"

for arm in "${ARMS[@]}"; do
  for dt in "${DTS[@]}"; do
    for target in "${TARGETS[@]}"; do
      submit_cell "$arm" "$dt" "$target"
    done
  done
done

echo "[launch_H] submitted ($(( ${#ARMS[@]} * ${#DTS[@]} * ${#TARGETS[@]} )) cells). After finish:"
echo "  sbatch scripts/slurm/paper_probes.sbatch H_agg"
