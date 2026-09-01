#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J sae-featcmp
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-1
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/sae_featcmp_%j.out
#SBATCH -e logs/slurm/sae_featcmp_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/slurm

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
PYTHON="${CONDA_PREFIX:-}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON=/home/ssuresh/.conda/envs/ct/bin/python

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false

LIN="${1:-/gscratch/ssuresh/results/spline_sae/probe_gemma_layer_tc_l24_linear_basejump_ctrl}"
KAN="${2:-/gscratch/ssuresh/results/spline_sae/probe_gemma_layer_tc_l24_kan_basejump}"
OUT="${3:-/gscratch/ssuresh/results/spline_sae/feature_compare_layer_tc_l24}"

echo "[$(date -u +%FT%TZ)] compare linear=${LIN} kan=${KAN}"
"${PYTHON}" -m spline_sae.compare_features \
  --linear-dir "${LIN}" \
  --kan-dir "${KAN}" \
  --output-dir "${OUT}" \
  --match-threshold 0.7 \
  --n-tokens 16384
echo "[$(date -u +%FT%TZ)] done → ${OUT}"
