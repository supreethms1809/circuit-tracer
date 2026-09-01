#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J ravel-rescore
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p gp-1
#SBATCH -t 02:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/ravel_rescore_%j.out
#SBATCH -e logs/slurm/ravel_rescore_%j.err

# Re-score hub + spline RAVEL graphs with reconstruction_fve=None.
# Submit:  sbatch scripts/slurm/ravel_rescore_graphs.sbatch

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source activate ct
export PYTHONUNBUFFERED=1

python -m rebuttal_eval.rescore_ravel_graphs \
  --out-dir /gscratch/ssuresh/results/paper_r5b_b3/ravel_rescored_raw_replacement
