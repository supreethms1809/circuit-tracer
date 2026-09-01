#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J diag-val-eval
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -p ai4wy-2
#SBATCH -t 00:20:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/diag_val_eval_%j.out
#SBATCH -e logs/slurm/diag_val_eval_%j.err
#SBATCH --signal=B:TERM@60

# Reproduce the val_loss discrepancy (logged ~0.2375 vs true recon ~0.73) on the
# real 2-GPU FSDP path. Read-only: loads the _best checkpoint, runs eval three ways,
# writes nothing to the run dir. Safe to run alongside the live training job (7745)
# — it lands on a separate idle node.
#
# Submit:  sbatch scripts/slurm/diag_val_eval.sbatch
# Optional env: DIAG_MAX (val seq cap, default 2048; 0 = full streaming val)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"

echo "[$(date -u +%FT%TZ)] diag-val-eval on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv || true

torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port="${MASTER_PORT:-29533}" \
    scripts/diag_val_eval.py

echo "[$(date -u +%FT%TZ)] diag-val-eval complete"
