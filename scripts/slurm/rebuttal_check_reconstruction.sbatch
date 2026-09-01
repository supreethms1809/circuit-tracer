#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rebuttal-recon-checks
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 06:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_recon_%j.out
#SBATCH -e logs/slurm/rebuttal_recon_%j.err
#SBATCH --signal=B:TERM@60

# §4.1-4.3 blocking reconstruction checks (rebuttal_eval.check_reconstruction)
# over every available converged checkpoint plus the submitted-era (93M-token)
# gpt2-small checkpoints, all on identical val subsets per model. Read-only
# over checkpoints and activations; writes only to results/rebuttal/.
#
# Submit:  sbatch scripts/slurm/rebuttal_check_reconstruction.sbatch
# Optional env: N_SAMPLES (default 256), SMOKE_ONLY=1 (stop after smoke)

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
ACT_L=/gscratch/ssuresh/shared/activations/paper_v2/gpt2_large/val/gpt2-large
ACT_Q=/gscratch/ssuresh/shared/activations/paper_v2/qwen3_06b/val/Qwen_Qwen3-0.6B
OUT="${REPO_ROOT}/results/rebuttal/check_reconstruction"
N="${N_SAMPLES:-256}"

run_check () {
    local label="$1" ckpt="$2" acts="$3" n="$4"
    echo "[$(date -u +%FT%TZ)] check_reconstruction ${label} (n=${n})"
    python -m rebuttal_eval.check_reconstruction \
        --checkpoint "${ckpt}" \
        --activation-dir "${acts}" \
        --out-dir "${OUT}" \
        --n-samples "${n}" \
        --seed 101 \
        --label "${label}"
}

# Smoke: tiny sample count on one spline + one linear checkpoint; fails fast.
run_check smoke_spline_fm_v3 \
    "$R/paper_v3_gpt2_small/runs/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best" \
    "${ACT_S}" 8
run_check smoke_linear_fm_v3 \
    "$R/paper_v3_gpt2_small/runs/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best" \
    "${ACT_S}" 8
if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    echo "SMOKE_ONLY=1 — stopping after smoke."
    exit 0
fi

# gpt2-small: converged v3 (all four variants, seed 101; spline FM also 202)
run_check gpt2s_v3_spline_fm_s101 \
    "$R/paper_v3_gpt2_small/runs/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best" \
    "${ACT_S}" "${N}"
run_check gpt2s_v3_spline_fm_s202 \
    "$R/paper_v3_gpt2_small/runs/spline_feature_match_gpt2_small_pv3/seed_202/checkpoints/spline_fm_gpt2_small_pv3_seed202_best" \
    "${ACT_S}" "${N}"
run_check gpt2s_v3_spline_pm_s101 \
    "$R/paper_v3_gpt2_small/runs/spline_param_match_gpt2_small_pv3/seed_101/checkpoints/spline_pm_gpt2_small_pv3_seed101_best" \
    "${ACT_S}" "${N}"
run_check gpt2s_v3_linear_fm_s101 \
    "$R/paper_v3_gpt2_small/runs/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best" \
    "${ACT_S}" "${N}"
run_check gpt2s_v3_linear_pm_s101 \
    "$R/paper_v3_gpt2_small/runs/linear_param_match_gpt2_small_pv3/seed_101/checkpoints/linear_pm_gpt2_small_pv3_seed101_best" \
    "${ACT_S}" "${N}"

# gpt2-small: submitted-era (93M tokens) checkpoints, same val subset.
# These are EXPECTED to show negative cosines (§4.1 diagnosis); the script
# exits nonzero only on a genuine metric-bug signature, so use `|| true`
# guarded by explicit logging rather than masking real failures silently.
set +e
run_check gpt2s_v1_spline_fm_submitted \
    "$R/paper_gpt2_small/runs/spline_feature_match_gpt2_small/seed_101/checkpoints/spline_fm_gpt2_small_seed101_best" \
    "${ACT_S}" "${N}"
V1_SPLINE_RC=$?
run_check gpt2s_v1_linear_fm_submitted \
    "$R/paper_gpt2_small/runs/linear_feature_match_gpt2_small/seed_101/checkpoints/linear_fm_gpt2_small_seed101_best" \
    "${ACT_S}" "${N}"
V1_LINEAR_RC=$?
set -e
echo "submitted-era exit codes: spline=${V1_SPLINE_RC} linear=${V1_LINEAR_RC}"

# gpt2-large v2
run_check gpt2l_v2_spline_fm_s101 \
    "$R/paper_v2_gpt2_large/runs/spline_feature_match_gpt2_large_pv2/seed_101/checkpoints/spline_fm_gpt2_large_pv2_seed101_best" \
    "${ACT_L}" "${N}"
run_check gpt2l_v2_linear_fm_s101 \
    "$R/paper_v2_gpt2_large/runs/linear_feature_match_gpt2_large_pv2/seed_101/checkpoints/linear_fm_gpt2_large_pv2_seed101_best" \
    "${ACT_L}" "${N}"

# qwen3-0.6b v2
run_check qwen_v2_spline_fm_s101 \
    "$R/paper_v2_qwen3_06b/runs/spline_feature_match_qwen3_06b_pv2/seed_101/checkpoints/spline_fm_qwen3_06b_pv2_seed101_best" \
    "${ACT_Q}" "${N}"
run_check qwen_v2_linear_fm_s101 \
    "$R/paper_v2_qwen3_06b/runs/linear_feature_match_qwen3_06b_pv2/seed_101/checkpoints/linear_fm_qwen3_06b_pv2_seed101_best" \
    "${ACT_Q}" "${N}"

# Submitted-vs-converged delta tables (§4.1 compare mode).
python -m rebuttal_eval.check_reconstruction --out-dir "${OUT}/compare_spline_fm" \
    --compare "${OUT}/check_reconstruction_gpt2s_v1_spline_fm_submitted.json" \
              "${OUT}/check_reconstruction_gpt2s_v3_spline_fm_s101.json"
python -m rebuttal_eval.check_reconstruction --out-dir "${OUT}/compare_linear_fm" \
    --compare "${OUT}/check_reconstruction_gpt2s_v1_linear_fm_submitted.json" \
              "${OUT}/check_reconstruction_gpt2s_v3_linear_fm_s101.json"

echo "[$(date -u +%FT%TZ)] all reconstruction checks complete"
