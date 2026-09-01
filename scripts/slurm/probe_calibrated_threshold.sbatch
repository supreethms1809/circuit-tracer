#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J probe-calibrated-threshold
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -p ai4wy-2
#SBATCH -t 06:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/probe_calibrated_threshold_%j.out
#SBATCH -e logs/slurm/probe_calibrated_threshold_%j.err
#SBATCH --signal=B:TERM@120
#SBATCH --export=NONE

set -euo pipefail

export SUITE_GLOB="experiments/paper_configs_v3/suites/*.json"
export NPROC_PER_NODE="2"
export PAPER_OUTPUT_ROOT="/cluster/ai4wy/gscratch/ssuresh/results/paper"
export WORKER_ID="${1:-0}"
export NUM_WORKERS="${2:-1}"

cd "/home/ssuresh/circuit-tracer"
exec bash scripts/slurm/run_paper_eval_multinode.sbatch \
    probe_calibrated_threshold_gpt2_small
