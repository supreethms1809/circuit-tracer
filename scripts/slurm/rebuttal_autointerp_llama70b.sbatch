#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J rebuttal-autointerp-l70
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 12:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_autointerp_l70_%j.out
#SBATCH -e logs/slurm/rebuttal_autointerp_l70_%j.err
#SBATCH --signal=B:TERM@120

# REQ-4 / REQ-9 scorer-matched re-score.
#
# The published detection/fuzzing numbers we compare against (Paulo et al.
# 2024 arXiv:2410.13928; Paulo, Shabalin & Belrose 2025 arXiv:2501.18823) all
# use Llama-3.1-70B as explainer+scorer. Our first pass used
# Qwen2.5-72B-Instruct, so absolute values were not comparable to the
# literature. This re-runs explanation + scoring for all eight arms under
# Llama-3.1-70B-Instruct and nothing else changes.
#
# Crucially this REUSES each arm's existing collection.json, so the sampled
# feature set, contexts, windows and seed are byte-identical to the Qwen pass.
# Qwen-vs-Llama is therefore a paired comparison on the same features, which
# also gives us a direct estimate of the scorer's calibration offset.
#
# Requires the vLLM server serving Llama-3.1-70B-Instruct:
#   VLLM_MODEL=meta-llama/Llama-3.1-70B-Instruct sbatch scripts/slurm/launch_vllm.sbatch
#
# Submit (results land in the live tree even when the script lives in a worktree):
#   sbatch --export=ALL,OUT=/cluster/ai4wy/home/ssuresh/circuit-tracer/results/rebuttal \
#       scripts/slurm/rebuttal_autointerp_llama70b.sbatch
#
# Env: N_FEATURES (default 200), ENDPOINT_FILE, LLM_MODEL, OUT

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

R=/gscratch/ssuresh/results/paper
V3=$R/paper_v3_gpt2_small/runs
# OUT defaults to this tree but is overridable, so the script can live in a
# worktree while reading/writing the live tree's results/rebuttal.
OUT="${OUT:-${REPO_ROOT}/results/rebuttal}"
ENDPOINT_FILE="${ENDPOINT_FILE:-${OUT}/vllm_endpoint.json}"
N_FEATURES="${N_FEATURES:-200}"
LLM_MODEL="${LLM_MODEL:-}"
SUFFIX=llama70b

echo "OUT=${OUT}"
echo "ENDPOINT_FILE=${ENDPOINT_FILE}"

# Re-score one arm against the Llama scorer, reusing the Qwen pass's
# collection.json so the feature sample is identical.
rescore () {
    local label="$1" base_model="$2" ckpt="$3"
    local src="${OUT}/autointerp_${label}"
    local dst="${OUT}/autointerp_${SUFFIX}_${label}"

    if [[ ! -f "${src}/collection.json" ]]; then
        echo "MISSING ${src}/collection.json — cannot re-score ${label}" >&2
        return 1
    fi
    mkdir -p "${dst}"
    # Copy once; on requeue the existing copy (and any partial
    # explanations/feature_scores JSONL) is kept so the run resumes.
    if [[ ! -f "${dst}/collection.json" ]]; then
        cp "${src}/collection.json" "${dst}/collection.json"
    fi

    echo "[$(date -u +%FT%TZ)] re-score ${label} (${base_model}) -> ${dst}"
    python -m rebuttal_eval.autointerp.run \
        --checkpoint "${ckpt}" \
        --model "${base_model}" \
        --out-dir "${dst}" \
        --backend openai_compat \
        --endpoint-file "${ENDPOINT_FILE}" \
        --llm-model "${LLM_MODEL}" \
        --n-features "${N_FEATURES}" \
        --skip-collect \
        --label "${label}"
}

# gpt2-small v3 feature-match pair
rescore spline_fm_v3 gpt2 \
    "$V3/spline_feature_match_gpt2_small_pv3/seed_101/checkpoints/spline_fm_gpt2_small_pv3_seed101_best"
rescore linear_fm_v3 gpt2 \
    "$V3/linear_feature_match_gpt2_small_pv3/seed_101/checkpoints/linear_fm_gpt2_small_pv3_seed101_best"

# gpt2-small v3 param-match pair
rescore spline_pm_v3 gpt2 \
    "$V3/spline_param_match_gpt2_small_pv3/seed_101/checkpoints/spline_pm_gpt2_small_pv3_seed101_best"
rescore linear_pm_v3 gpt2 \
    "$V3/linear_param_match_gpt2_small_pv3/seed_101/checkpoints/linear_pm_gpt2_small_pv3_seed101_best"

# gpt2-large v2 pair
rescore spline_fm_gpt2l gpt2-large \
    "$R/paper_v2_gpt2_large/runs/spline_feature_match_gpt2_large_pv2/seed_101/checkpoints/spline_fm_gpt2_large_pv2_seed101_best"
rescore linear_fm_gpt2l gpt2-large \
    "$R/paper_v2_gpt2_large/runs/linear_feature_match_gpt2_large_pv2/seed_101/checkpoints/linear_fm_gpt2_large_pv2_seed101_best"

# qwen3-0.6b v2 pair
rescore spline_fm_qwen "Qwen/Qwen3-0.6B" \
    "$R/paper_v2_qwen3_06b/runs/spline_feature_match_qwen3_06b_pv2/seed_101/checkpoints/spline_fm_qwen3_06b_pv2_seed101_best"
rescore linear_fm_qwen "Qwen/Qwen3-0.6B" \
    "$R/paper_v2_qwen3_06b/runs/linear_feature_match_qwen3_06b_pv2/seed_101/checkpoints/linear_fm_qwen3_06b_pv2_seed101_best"

echo "[$(date -u +%FT%TZ)] llama-70b re-score complete"
