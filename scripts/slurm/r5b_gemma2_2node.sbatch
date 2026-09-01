#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J r5b-gemma2
#SBATCH -N 2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -p gp-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/r5b_gemma2_%j.out
#SBATCH -e logs/slurm/r5b_gemma2_%j.err
#SBATCH --signal=B:TERM@120
#SBATCH --export=NONE

# Gemma-2-2B Spline-CLT with nested FSDP KAN sharding: 2 nodes / 4 GPUs,
# batch_size=8 keeps global batch = 32 (same as prior 4×2 × batch 4).
# Usage: sbatch scripts/slurm/r5b_gemma2_2node.sbatch r5b_b3_gemma2_2b_spline_dt6144_basejump
#    or: sbatch scripts/slurm/r5b_gemma2_2node.sbatch r5b_b3_gemma2_2b_spline_dt10112_basejump

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 LAUNCH_SUITE_NAME" >&2
    exit 2
fi

export SUITE_GLOB="experiments/paper_configs_v3/launch_suites/*.json"
export NPROC_PER_NODE="2"
export PAPER_OUTPUT_ROOT="/gscratch/ssuresh/results/paper_r5b_b3"
# gp-2 mlx5_0 (IB) is DOWN/INIT cluster-wide as of 2026-08-19; mlx5_1 (RoCE) is
# 4:ACTIVE on every probed node. Pin NCCL to the live HCA so 2-node jobs do not
# abort in preflight. Do not mix mlx5_0 and mlx5_1 across ranks.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_1:1}"

cd "/home/ssuresh/circuit-tracer"
exec bash scripts/slurm/run_paper_eval_multinode.sbatch "$1"
