#!/usr/bin/env bash
# TMLR-250 v4 Pass A — gemma2-2.5M MIB sweep (4 nodes x 8 GPUs).
#
# Same v4 protocol as submit_macag_mib_h200_v4.sh; fewer shards per node
# because the 2.5M CLT + Gemma-2-2B needs more HBM per worker.
#
#   sbatch scripts/slurm/submit_macag_mib_gemma25m_v4.sh
#SBATCH --job-name=macag_v4_gemma25m
#SBATCH -A uwyo-0002
#SBATCH --partition=gp-2
#SBATCH --array=0-3
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

# --- v4 Pass A knobs (do not retune mid-pass; see run_todo_v4.md) ---
export JSON="${JSON:-macag/data/mib_benchmark_prompts.json}"
export OUTROOT="${OUTROOT:-/gscratch/${USER}/macag_mib_tmlr250v4_gemma25m/macag_mib_seed0}"
export CLTS="${CLTS:-gemma2-2.5M}"
export TASKS="${TASKS:-ioi mcqa arc_easy}"
export FREEZE_MODE="${FREEZE_MODE:-both}"
export SCORE_KINDS="${SCORE_KINDS:-logit_gap}"
export SOLVERS="${SOLVERS:-abr fp}"
export BASELINE_METHODS="${BASELINE_METHODS:-influence}"
export REPORT_BUDGET="${REPORT_BUDGET:-128}"
export NODE_THRESHOLD="${NODE_THRESHOLD:-1.0}"
export EDGE_THRESHOLD="${EDGE_THRESHOLD:-1.0}"
export CONNECTED="${CONNECTED:-0}"
export PREFILTER_TOP_K="${PREFILTER_TOP_K:-}"
export KL_RESCORE="${KL_RESCORE:-1}"
export SHAPLEY_SEED="${SHAPLEY_SEED:-0}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-0}"
export STAGGER="${STAGGER:-45}"

export WORKERS_GEMMA2_2_5M="${WORKERS_GEMMA2_2_5M:-3}"
export NODE_RANK="${SLURM_ARRAY_TASK_ID:-0}"
export TOTAL_WORKERS="$(( 4 * WORKERS_GEMMA2_2_5M ))"

echo ">>> v4 Pass A gemma2-2.5M | array=${SLURM_ARRAY_TASK_ID:-0}/4 total_workers=$TOTAL_WORKERS"
echo ">>> out=$OUTROOT"
nvidia-smi -L || true

scripts/run_macag_mib_parallel.sh
