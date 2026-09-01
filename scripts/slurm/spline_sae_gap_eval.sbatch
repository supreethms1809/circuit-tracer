#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J sae-gap-eval
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-1
#SBATCH -t 04:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/sae_gap_eval_%j.out
#SBATCH -e logs/slurm/sae_gap_eval_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/slurm /gscratch/ssuresh/results/spline_sae

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
PYTHON="${CONDA_PREFIX:-}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON=/home/ssuresh/.conda/envs/ct/bin/python

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" -m spline_sae.eval_gap \
  --checkpoints \
    /gscratch/ssuresh/results/spline_sae/gemma2_2b_l12_kan_jumprelu_w16k \
    /gscratch/ssuresh/results/spline_sae/llama31_8b_l16_kan_topk_w16k \
    /gscratch/ssuresh/results/spline_sae/qwen25_7b_it_l20_kan_jumprelu_w16k \
    /gscratch/ssuresh/results/spline_sae/gemma2_2b_l12_linear_jumprelu_w16k \
  --n-batches 16 \
  --batch-tokens 2048 \
  --output /gscratch/ssuresh/results/spline_sae/gap_eval_phase1.json
