#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J diag-fsdp-threshold
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -p ai4wy-2
#SBATCH -t 00:10:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/diag_fsdp_threshold_%j.out
#SBATCH -e logs/slurm/diag_fsdp_threshold_%j.err
#SBATCH --export=NONE

set -euo pipefail

cd "/home/ssuresh/circuit-tracer"
module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

torchrun --standalone --nproc_per_node=2 scripts/diag_fsdp_threshold_optimizer.py
