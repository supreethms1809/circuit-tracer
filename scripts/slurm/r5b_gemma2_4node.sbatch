#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J r5b-gemma2
#SBATCH -N 4
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -p gp-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/r5b_gemma2_%j.out
#SBATCH -e logs/slurm/r5b_gemma2_%j.err
#SBATCH --signal=B:TERM@120
#SBATCH --export=NONE

# Gemma-2-2B Spline-CLT: 4 nodes / 8 GPUs for large d_t (param-match dt10112).
# Pair with launch-suite batch_size=4 (global batch 32) and total_steps=48828
# for ~200M tokens — lower per-GPU batch than 8 for HBM headroom.
# Usage: sbatch scripts/slurm/r5b_gemma2_4node.sbatch r5b_b3_gemma2_2b_spline_dt10112_basejump

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 LAUNCH_SUITE_NAME" >&2
    exit 2
fi

export SUITE_GLOB="experiments/paper_configs_v3/launch_suites/*.json"
export NPROC_PER_NODE="2"
export PAPER_OUTPUT_ROOT="/gscratch/ssuresh/results/paper_r5b_b3"

cd "/home/ssuresh/circuit-tracer"
exec bash scripts/slurm/run_paper_eval_multinode.sbatch "$1"
