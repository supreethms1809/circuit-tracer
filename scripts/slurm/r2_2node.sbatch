#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J r2-2node
#SBATCH -N 2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -p ai4wy-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/r2_%j.out
#SBATCH -e logs/slurm/r2_%j.err
#SBATCH --signal=B:TERM@120
#SBATCH --export=NONE

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 LAUNCH_SUITE_NAME" >&2
    exit 2
fi

export SUITE_GLOB="experiments/paper_configs_v3/launch_suites/*.json"
export NPROC_PER_NODE="2"
export PAPER_OUTPUT_ROOT="/gscratch/ssuresh/results/paper_rebuttal"

cd "/home/ssuresh/circuit-tracer"
exec bash scripts/slurm/run_paper_eval_multinode.sbatch "$1"
