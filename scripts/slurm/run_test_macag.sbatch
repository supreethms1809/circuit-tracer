#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J test-macag
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 04:00:00
#SBATCH --mem=32G
#SBATCH -o logs/slurm/test_macag_%j.out
#SBATCH -e logs/slurm/test_macag_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

export HF_HOME=/gscratch/ssuresh
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

python -u /home/ssuresh/.gemini/antigravity-ide/brain/148b87a9-5dae-477d-876a-5f29852c4ba5/scratch/test_macag_game1.py
