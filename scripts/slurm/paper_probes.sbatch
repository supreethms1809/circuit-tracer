#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J paper-probes
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -p ai4wy-2
#SBATCH -t 12:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/probes_%j.out
#SBATCH -e logs/slurm/probes_%j.err
#SBATCH --export=NONE

# Probe F: sbatch ... F <arm> <d_t> <lambda> [steps]
# Probe G: sbatch ... G <arm> <d_t> <variant> [steps] [lambda]
# Probe H/I: sbatch ... H|I <arm> <d_t> <target_l0> [steps] [lambda]
# Probe J: sbatch ... J <B2|B3> <d_t> <target_l0> [steps] [lambda]

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/ssuresh/circuit-tracer}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm" /gscratch/ssuresh/results/paper_probes

module load arcc miniconda3
module load slurm/ai4wy/25.05 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

PROBE="${1:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-512}"
BATCH="${BATCH:-8}"
F_ARM=""
G_VARIANT=""
J_VARIANT=""
TARGET_L0=""

export PAPER_PROBES_ROOT=/gscratch/ssuresh/results/paper_probes
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA=()
if [[ "$PROBE" == "F" ]]; then
  F_ARM="${2:-}"; D_T="${3:-2048}"; LAMBDA="${4:-0.005}"; STEPS="${5:-3000}"
  [[ -n "$F_ARM" ]] || { echo "ERROR: F <arm> <d_t> <lambda> [steps]" >&2; exit 2; }
  EXTRA+=(--f-arm "$F_ARM" --lambda-sparsity "$LAMBDA")
elif [[ "$PROBE" == "G" ]]; then
  F_ARM="${2:-}"; D_T="${3:-2048}"; G_VARIANT="${4:-}"; STEPS="${5:-3000}"; LAMBDA="${6:-1e-6}"
  [[ -n "$F_ARM" && -n "$G_VARIANT" ]] || { echo "ERROR: G <arm> <d_t> <variant> ..." >&2; exit 2; }
  EXTRA+=(--f-arm "$F_ARM" --g-variant "$G_VARIANT" --lambda-sparsity "$LAMBDA")
elif [[ "$PROBE" == "H" || "$PROBE" == "I" ]]; then
  F_ARM="${2:-}"; D_T="${3:-2048}"; TARGET_L0="${4:-}"; STEPS="${5:-3000}"; LAMBDA="${6:-1e-6}"
  [[ -n "$F_ARM" && -n "$TARGET_L0" ]] || { echo "ERROR: $PROBE <arm> <d_t> <target_l0> ..." >&2; exit 2; }
  EXTRA+=(--f-arm "$F_ARM" --target-l0 "$TARGET_L0" --lambda-sparsity "$LAMBDA")
elif [[ "$PROBE" == "J" ]]; then
  J_VARIANT="${2:-}"; D_T="${3:-2048}"; TARGET_L0="${4:-}"; STEPS="${5:-3000}"; LAMBDA="${6:-1e-6}"
  [[ -n "$J_VARIANT" && -n "$TARGET_L0" ]] || { echo "ERROR: J <B2|B3> <d_t> <target_l0> [steps] [lambda]" >&2; exit 2; }
  EXTRA+=(
    --j-variant "$J_VARIANT"
    --target-l0 "$TARGET_L0"
    --lambda-sparsity "$LAMBDA"
    --scale-base "${SCALE_BASE:-0.2}"
    --scale-spline "${SCALE_SPLINE:-1.0}"
  )
else
  D_T="${D_T:-2048}"; LAMBDA="${LAMBDA:-0.005}"; STEPS="${STEPS:-3000}"
fi

echo "[probes] host=$(hostname) probe=$PROBE steps=$STEPS d_t=$D_T λ=$LAMBDA f_arm=${F_ARM:-none} g_variant=${G_VARIANT:-none} j_variant=${J_VARIANT:-none} target_l0=${TARGET_L0:-none}"
nvidia-smi -L || true

python -m experiments.probes.run_probes "$PROBE" \
  --device cuda \
  --steps "$STEPS" \
  --d-transcoder "$D_T" \
  --batch-size "$BATCH" \
  --max-samples "$MAX_SAMPLES" \
  --seed 101 \
  ${EXTRA[@]+"${EXTRA[@]}"}
