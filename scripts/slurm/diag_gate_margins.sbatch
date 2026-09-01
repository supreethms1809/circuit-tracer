#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J diag-gate-margins
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -p ai4wy-2
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/diag_gate_margins_%j.out
#SBATCH -e logs/slurm/diag_gate_margins_%j.err
#SBATCH --export=NONE

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: sbatch $0 CHECKPOINT LABEL" >&2
    exit 2
fi

cd "/home/ssuresh/circuit-tracer"
module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

python scripts/diag_gate_margins.py \
    --checkpoint "$1" \
    --activation-dir "/gscratch/ssuresh/shared/activations/probe_lambda_sweep/gpt2_small/val/gpt2" \
    --label "$2" \
    --out-dir "results/rebuttal/linear_training_diagnostics" \
    --n-samples 32 \
    --seed 101 \
    --device cuda
