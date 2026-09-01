#!/usr/bin/env bash
# Launch Probe F capacity / L0-retarget grid (one GPU job per cell).
#
# Claim: SiLU/KAN at smaller d_t can match linear at larger d_t once L0 is healthy.
# Unlocked θ + decoder. λ swept on nonlinear arms; linear uses reference λ.
#
# NOTE: paper_probes.sbatch uses --export=NONE, so cells are passed positionally:
#   sbatch ... F <arm> <d_t> <lambda> <steps>
#
# Usage (from repo root):
#   bash scripts/slurm/launch_probe_F.sh
#   STEPS=3000 WAVE=core bash scripts/slurm/launch_probe_F.sh
# Extend a running grid with lower λ only:
#   ONLY_LAMBDAS="1e-5 1e-6" INCLUDE_LINEAR=1 bash scripts/slurm/launch_probe_F.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/slurm /gscratch/ssuresh/results/paper_probes/F_capacity_l0

STEPS="${STEPS:-3000}"
WAVE="${WAVE:-full}"   # core = linear+silu; full = +kan
SBATCH=scripts/slurm/paper_probes.sbatch

LAMBDAS=(0.005 0.002 0.001 0.0005 0.0002 1e-5 1e-6)
DTS=(512 2048)
REF_LAMBDA=0.005

# If ONLY_LAMBDAS is set (space-separated), submit just those λ (extend grid).
if [[ -n "${ONLY_LAMBDAS:-}" ]]; then
  # shellcheck disable=SC2206
  LAMBDAS=($ONLY_LAMBDAS)
fi

submit_cell() {
  local arm="$1" dt="$2" lam="$3"
  local job_name="F-${arm}-dt${dt}-l${lam}"
  job_name="${job_name//./p}"
  echo "submit arm=$arm d_t=$dt λ=$lam steps=$STEPS name=$job_name"
  sbatch --job-name="$job_name" "$SBATCH" F "$arm" "$dt" "$lam" "$STEPS"
}

echo "[launch_F] STEPS=$STEPS WAVE=$WAVE LAMBDAS=${LAMBDAS[*]} → /gscratch/ssuresh/results/paper_probes/F_capacity_l0"

# Full launch: one linear cell per d_t at REF_LAMBDA.
# Extend mode (ONLY_LAMBDAS): skip linear unless INCLUDE_LINEAR=1.
if [[ -z "${ONLY_LAMBDAS:-}" ]]; then
  for dt in "${DTS[@]}"; do
    submit_cell linear "$dt" "$REF_LAMBDA"
  done
elif [[ "${INCLUDE_LINEAR:-0}" == "1" ]]; then
  for dt in "${DTS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
      submit_cell linear "$dt" "$lam"
    done
  done
fi

for dt in "${DTS[@]}"; do
  for lam in "${LAMBDAS[@]}"; do
    submit_cell silu_base "$dt" "$lam"
  done
done

if [[ "$WAVE" == "full" ]]; then
  for dt in "${DTS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
      submit_cell kan "$dt" "$lam"
    done
  done
fi

echo "[launch_F] submitted. After all finish:"
echo "  sbatch scripts/slurm/paper_probes.sbatch F_agg"
echo "  # or: python -m experiments.probes.run_probes F_agg"
