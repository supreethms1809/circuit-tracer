#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J r2-ravel-interp-spl
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 20:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_autointerp_ravel_r2_spline_%j.out
#SBATCH -e logs/slurm/rebuttal_autointerp_ravel_r2_spline_%j.err
#SBATCH --signal=B:TERM@120

# Graph-targeted autointerp for the r2 gpt2_small SPLINE ravel eval suites.
# Companion to rebuttal_autointerp_ravel_r2_linear.sbatch (same vLLM
# endpoint, split so a single failure doesn't take out all 14 r2 arms).
#
# Prerequisites:
#   - r2 gpt2_small ravel eval suites finished (jobs 8606-8613) under
#     /gscratch/ssuresh/results/paper_rebuttal/ravel_eval_suite_r2_<label>
#   - vLLM endpoint at ENDPOINT_FILE (scripts/slurm/launch_vllm.sbatch)
#
# Submit:
#   sbatch --dependency=afterany:8606,8607,8608,8609,8610,8611,8612,8613 \
#       scripts/slurm/rebuttal_autointerp_ravel_r2_spline.sbatch
#
# Env: ENDPOINT_FILE, LLM_MODEL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
PYTHON="${CONDA_PREFIX:-}/bin/python"
if [[ ! -x "${PYTHON}" ]] || ! "${PYTHON}" -c "import torch" 2>/dev/null; then
    PYTHON=/home/ssuresh/.conda/envs/ct/bin/python
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[$(date -u +%FT%TZ)] python=${PYTHON}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

R=/gscratch/ssuresh/results/paper_rebuttal
ENDPOINT_FILE="${ENDPOINT_FILE:-${REPO_ROOT}/results/rebuttal/vllm_endpoint.json}"
LLM_MODEL="${LLM_MODEL:-}"
OUT="${R}"

run_name_graphs () {
    local label="$1" graphs_dir="$2" ckpt="$3"
    local out_dir="${OUT}/autointerp_ravel_${label}"
    local n_graphs
    n_graphs=$(find "${graphs_dir}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    if [[ "${n_graphs}" -lt 1 ]]; then
        echo "[$(date -u +%FT%TZ)] SKIP ${label}: no graphs in ${graphs_dir}" >&2
        return 0
    fi
    if [[ ! -d "${ckpt}" ]]; then
        echo "[$(date -u +%FT%TZ)] SKIP ${label}: checkpoint missing ${ckpt}" >&2
        return 0
    fi
    echo "[$(date -u +%FT%TZ)] name_graphs ${label} (n_graphs=${n_graphs}) -> ${out_dir}"
    if ! "${PYTHON}" -m rebuttal_eval.autointerp.name_graphs \
        --graphs-dir "${graphs_dir}" \
        --checkpoint "${ckpt}" \
        --model gpt2 \
        --out-dir "${out_dir}" \
        --backend openai_compat \
        --endpoint-file "${ENDPOINT_FILE}" \
        --llm-model "${LLM_MODEL}" \
        --label "${label}" \
        --include-scores-in-clerp; then
        echo "[$(date -u +%FT%TZ)] FAILED ${label} (continuing)" >&2
        return 0
    fi

    local mirror
    mirror="$(dirname "${graphs_dir}")/graphs_autointerp"
    mkdir -p "${mirror}"
    cp -f "${out_dir}/annotated_graphs/"*.json "${mirror}/" 2>/dev/null || true
    if [[ -f "${out_dir}/annotated_graphs/clerp_writeback_summary.json" ]]; then
        cp -f "${out_dir}/annotated_graphs/clerp_writeback_summary.json" "${mirror}/"
    fi
    echo "[$(date -u +%FT%TZ)] mirrored annotated graphs -> ${mirror}"
}

run_name_graphs r2_spline_dt768 \
    "$R/ravel_eval_suite_r2_spline_dt768/runs/r2_gpt2_small_spline_dt768/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_spline_dt768/runs/gpt2_small_spline_dt768_r2/seed_101/checkpoints/gpt2_small_spline_dt768_r2_seed101_best"

run_name_graphs r2_spline_dt1536 \
    "$R/ravel_eval_suite_r2_spline_dt1536/runs/r2_gpt2_small_spline_dt1536/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_spline_dt1536/runs/gpt2_small_spline_dt1536_r2/seed_101/checkpoints/gpt2_small_spline_dt1536_r2_seed101_best"

run_name_graphs r2_spline_dt3072 \
    "$R/ravel_eval_suite_r2_spline_dt3072/runs/r2_gpt2_small_spline_dt3072/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_spline_dt3072/runs/gpt2_small_spline_dt3072_r2/seed_101/checkpoints/gpt2_small_spline_dt3072_r2_seed101_best"

run_name_graphs r2_spline_dt5568 \
    "$R/ravel_eval_suite_r2_spline_dt5568/runs/r2_gpt2_small_spline_dt5568/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_spline_dt5568/runs/gpt2_small_spline_dt5568_r2/seed_101/checkpoints/gpt2_small_spline_dt5568_r2_seed101_best"

run_name_graphs r2_spline_dt6144 \
    "$R/ravel_eval_suite_r2_spline_dt6144/runs/r2_gpt2_small_spline_dt6144/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_spline_dt6144/runs/gpt2_small_spline_dt6144_r2/seed_101/checkpoints/gpt2_small_spline_dt6144_r2_seed101_best"

run_name_graphs r2_spline_dt12288 \
    "$R/ravel_eval_suite_r2_spline_dt12288/runs/r2_gpt2_small_spline_dt12288/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_spline_dt12288/runs/gpt2_small_spline_dt12288_r2/seed_101/checkpoints/gpt2_small_spline_dt12288_r2_seed101_best"

run_name_graphs r2_spline_feature_match \
    "$R/ravel_eval_suite_r2_spline_feature_match/runs/r2_gpt2_small_spline_feature_match/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_spline_feature_match/runs/gpt2_small_spline_feature_match_r2/seed_101/checkpoints/gpt2_small_spline_feature_match_r2_seed101_best"

run_name_graphs r2_spline_param_match \
    "$R/ravel_eval_suite_r2_spline_param_match/runs/r2_gpt2_small_spline_param_match/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_spline_param_match/runs/gpt2_small_spline_param_match_r2/seed_101/checkpoints/gpt2_small_spline_param_match_r2_seed101_best"

echo "[$(date -u +%FT%TZ)] r2 spline ravel graph autointerp complete"
