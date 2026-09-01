#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J compute-bench-v3reb
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 02:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_compute_bench_v3reb_%j.out
#SBATCH -e logs/slurm/rebuttal_compute_bench_v3reb_%j.err
#SBATCH --signal=B:TERM@60

# AC-Q5 compute/memory numbers re-measured on the four paper_v3_rebuttal_*
# gpt2-small checkpoints — the ones actually reported in the rebuttal tables.
#
# The existing results/rebuttal/{attr_scaling,inference_bench}/ artifacts were
# measured on paper_v3_gpt2_small, which differs from the rebuttal generation on
# THREE axes, all of which inflate the linear arms' active-feature count:
#   threshold_init_strategy : none (frozen theta=0.001)  vs  data_quantile@L0=32
#   jumprelu_bandwidth      : unset -> class default 2   vs  0.001
#   lambda_sparsity         : 0.002 (under-regularized)  vs  0.005
# That is the theta^2 gradient trap (dL/dlog theta ~ theta^2/bw). Those linear
# arms never sparsified (l0_active_per_pos 4069 / 9226 vs ~10 for spline), which
# is what produces the reported "4-7x lower peak memory for spline".
#
# Read-only over checkpoints and activations; writes only to results/rebuttal/.
# Writes to *_v3reb/ so the older-generation artifacts stay intact for continuity.
#
# Submit:  sbatch scripts/slurm/rebuttal_compute_bench_v3rebuttal.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Reap any of our own stray GPU processes left on this node before starting.
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

R=/gscratch/ssuresh/results/paper
ACT_S=/gscratch/ssuresh/shared/activations/paper_v3/gpt2_small_1b/val/gpt2

SPLINE_FM="$R/paper_v3_rebuttal_gpt2_small_spline_fm/runs/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best"
SPLINE_PM="$R/paper_v3_rebuttal_gpt2_small_spline_pm/runs/spline_param_match_gpt2_small_pv3/seed_101/checkpoints/spline_pm_gpt2_small_pv3_seed101_best"
LINEAR_FM="$R/paper_v3_rebuttal_gpt2_small_linear_fm/runs/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best"
LINEAR_PM="$R/paper_v3_rebuttal_gpt2_small_linear_pm/runs/linear_param_match_gpt2_small_pv3/seed_101/checkpoints/linear_pm_gpt2_small_pv3_seed101_best"

for c in "${SPLINE_FM}" "${SPLINE_PM}" "${LINEAR_FM}" "${LINEAR_PM}"; do
    [[ -d "${c}" ]] || { echo "FATAL: missing checkpoint ${c}" >&2; exit 1; }
done

# --- Forward latency + peak forward memory + true trained L0 -----------------
# n-iters/seed match the original run so rows are directly comparable.
echo "[$(date -u +%FT%TZ)] inference_bench (v3-rebuttal checkpoints)"
python -m rebuttal_eval.inference_bench \
    --checkpoints "${SPLINE_FM}" "${SPLINE_PM}" "${LINEAR_FM}" "${LINEAR_PM}" \
    --activation-dir "${ACT_S}" \
    --out-dir "${REPO_ROOT}/results/rebuttal/inference_bench_v3reb" \
    --seed 101 \
    --n-iters 30 \
    --label gpt2_small_v3reb

# --- Attribution wall-time + peak attribution memory + active features -------
# n-prompts/max-features/batch-size match the original run.
echo "[$(date -u +%FT%TZ)] attr_scaling (v3-rebuttal checkpoints)"
python -m rebuttal_eval.attr_scaling \
    --checkpoints "${SPLINE_FM}" "${SPLINE_PM}" "${LINEAR_FM}" "${LINEAR_PM}" \
    --model gpt2 \
    --n-prompts 5 \
    --max-features 7500 \
    --batch-size 512 \
    --out-dir "${REPO_ROOT}/results/rebuttal/attr_scaling_v3reb" \
    --label gpt2_small_v3reb

echo "[$(date -u +%FT%TZ)] v3-rebuttal compute/memory bench complete"
