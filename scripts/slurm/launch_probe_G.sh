#!/usr/bin/env bash
# Launch Probe G: JumpReLU / θ schedule levers.
#
# Fixed from F: unlocked θ/decoder, λ=1e-6, proper θ Adam group.
# Variants: baseline, target128, target256, bw1x, delay30, relu
#
# Usage:
#   bash scripts/slurm/launch_probe_G.sh
#   WAVE=core bash scripts/slurm/launch_probe_G.sh   # linear + silu only
#   WAVE=full bash scripts/slurm/launch_probe_G.sh   # + kan (default)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/slurm /gscratch/ssuresh/results/paper_probes/G_jumprelu_theta

STEPS="${STEPS:-3000}"
WAVE="${WAVE:-full}"
LAMBDA="${LAMBDA:-1e-6}"
SBATCH=scripts/slurm/paper_probes.sbatch

VARIANTS=(baseline target128 target256 bw1x delay30 relu)
DTS=(512 2048)
ARMS=(linear silu_base)
if [[ "$WAVE" == "full" ]]; then
  ARMS+=(kan)
fi

submit_cell() {
  local arm="$1" dt="$2" variant="$3"
  local job_name="G-${arm}-dt${dt}-${variant}"
  echo "submit arm=$arm d_t=$dt variant=$variant λ=$LAMBDA steps=$STEPS name=$job_name"
  sbatch --job-name="$job_name" "$SBATCH" G "$arm" "$dt" "$variant" "$STEPS" "$LAMBDA"
}

echo "[launch_G] STEPS=$STEPS WAVE=$WAVE λ=$LAMBDA variants=${VARIANTS[*]}"
echo "[launch_G] → /gscratch/ssuresh/results/paper_probes/G_jumprelu_theta"

for arm in "${ARMS[@]}"; do
  for dt in "${DTS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      submit_cell "$arm" "$dt" "$variant"
    done
  done
done

echo "[launch_G] submitted. After all finish:"
echo "  sbatch scripts/slurm/paper_probes.sbatch G_agg"
