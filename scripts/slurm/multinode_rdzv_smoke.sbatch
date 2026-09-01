#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rdzv-smoke
#SBATCH -N 2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -p gp-2
#SBATCH -t 00:10:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rdzv_smoke_%j.out
#SBATCH -e logs/slurm/rdzv_smoke_%j.err
#SBATCH --export=NONE

# Multi-node torch.distributed rendezvous smoke (c10d + NCCL all-reduce).
# Same launch shape as run_paper_eval_multinode.sbatch — use this after cluster
# hostname / DNS cutovers before resubmitting long Gemma training.
#
#   sbatch scripts/slurm/multinode_rdzv_smoke.sbatch

set -euo pipefail

cd /home/ssuresh/circuit-tracer
mkdir -p logs/slurm

module load arcc miniconda3
module load slurm/ai4wy/25.05 || true
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

# --export=NONE leaves srun with a stripped PATH; inherit the batch env.
export SLURM_EXPORT_ENV=ALL

export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Same pin as run_paper_eval_multinode.sbatch: nodes expose mlx5_0 (IB) and
# mlx5_1 (RoCE); NCCL must not mix them across nodes.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"

echo "[$(date -u +%FT%TZ)] rdzv-smoke start"
echo "  Node list:     ${SLURM_JOB_NODELIST}"
echo "  Head node:     ${HEAD_NODE}"
echo "  nnodes:        ${SLURM_NNODES}"
echo "  nproc/node:    ${NPROC_PER_NODE}"
echo "  rdzv endpoint: ${HEAD_NODE}:${MASTER_PORT}"
echo "  hostnames:"
srun --ntasks-per-node=1 bash -lc 'echo "    $(hostname) fqdn=$(hostname -f) ip=$(hostname -I | awk "{print \$1}") aliases=$(getent hosts $(hostname -s) | head -1)"'

echo "[$(date -u +%FT%TZ)] launching torchrun..."
srun --ntasks-per-node=1 --kill-on-bad-exit=1 torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${HEAD_NODE}:${MASTER_PORT}" \
    scripts/diag_multinode_rdzv.py

echo "[$(date -u +%FT%TZ)] rdzv-smoke finished OK"
