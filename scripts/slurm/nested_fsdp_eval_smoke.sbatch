#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J nested-fsdp-eval
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -p ai4wy-2
#SBATCH -t 00:20:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/nested_fsdp_eval_%j.out
#SBATCH -e logs/slurm/nested_fsdp_eval_%j.err
#SBATCH --export=NONE

# 2-GPU smoke for nested-FSDP train + sharded evaluate (no summon).
set -euo pipefail
cd /home/ssuresh/circuit-tracer
mkdir -p logs/slurm

module load arcc miniconda3
module load slurm/ai4wy/25.05
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -m pytest tests/test_nested_fsdp_kan.py -v --tb=short \
  -k 'eval_without_summon or eval_call_site or two_gpu_smoke or summon_nested'
