#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J recon-checks-v3reb
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 02:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_recon_v3reb_%j.out
#SBATCH -e logs/slurm/rebuttal_recon_v3reb_%j.err
#SBATCH --signal=B:TERM@60

# §4.1-4.3 blocking checks over the four paper_v3_rebuttal_* gpt2-small
# checkpoints — the ones actually reported in the rebuttal tables. The existing
# rebuttal_check_reconstruction.sbatch covers the older paper_v3_gpt2_small /
# v2 / v1 generations, whose linear arms have the frozen-JumpReLU defect and
# whose spline arms sit at a different operating point (L0 ~11 vs ~4).
#
# Read-only over checkpoints and activations; writes only to results/rebuttal/.
#
# Submit:  sbatch scripts/slurm/rebuttal_check_reconstruction_v3rebuttal.sbatch
# Optional env: N_SAMPLES (default 256)

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
OUT="${REPO_ROOT}/results/rebuttal/check_reconstruction"
N="${N_SAMPLES:-256}"

run_check () {
    local label="$1" ckpt="$2" n="$3"
    if [[ ! -d "${ckpt}" ]]; then
        echo "SKIP ${label}: checkpoint not present (${ckpt})"
        return 0
    fi
    echo "[$(date -u +%FT%TZ)] check_reconstruction ${label} (n=${n})"
    python -m rebuttal_eval.check_reconstruction \
        --checkpoint "${ckpt}" \
        --activation-dir "${ACT_S}" \
        --out-dir "${OUT}" \
        --n-samples "${n}" \
        --seed 101 \
        --label "${label}"
}

# Smoke first: fails fast on a genuine metric-bug signature.
run_check smoke_v3reb_spline_fm \
    "$R/paper_v3_rebuttal_gpt2_small_spline_fm/runs/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best" 8

# The four reported arms. linear_pm is still training; its _best checkpoint is
# evaluated as-is and the result must be labelled mid-training.
run_check gpt2s_v3reb_spline_fm_s101 \
    "$R/paper_v3_rebuttal_gpt2_small_spline_fm/runs/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best" "${N}"
run_check gpt2s_v3reb_spline_pm_s101 \
    "$R/paper_v3_rebuttal_gpt2_small_spline_pm/runs/spline_param_match_gpt2_small_pv3/seed_101/checkpoints/spline_pm_gpt2_small_pv3_seed101_best" "${N}"
run_check gpt2s_v3reb_linear_fm_s101 \
    "$R/paper_v3_rebuttal_gpt2_small_linear_fm/runs/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best" "${N}"
run_check gpt2s_v3reb_linear_pm_s101_midtrain \
    "$R/paper_v3_rebuttal_gpt2_small_linear_pm/runs/linear_param_match_gpt2_small_pv3/seed_101/checkpoints/linear_pm_gpt2_small_pv3_seed101_best" "${N}"

echo "[$(date -u +%FT%TZ)] v3-rebuttal reconstruction checks complete"
