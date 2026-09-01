#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J diag-threshold-init
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -p ai4wy-2
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/diag_threshold_calibration_%j.out
#SBATCH -e logs/slurm/diag_threshold_calibration_%j.err
#SBATCH --export=NONE

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 ENCODER_TYPE" >&2
    exit 2
fi

cd "/home/ssuresh/circuit-tracer"
module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

python scripts/diag_threshold_calibration.py \
    --encoder-type "$1" \
    --activation-dir "/gscratch/ssuresh/shared/activations/probe_lambda_sweep/gpt2_small/val/gpt2" \
    --out-dir "results/rebuttal/linear_training_diagnostics" \
    --target-l0 32 \
    --calibration-samples 16 \
    --eval-samples 32 \
    --values-per-sample 32768 \
    --seed 101 \
    --device cuda
