#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J r2-ravel-interp-lin
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p ai4wy-2
#SBATCH -t 20:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_autointerp_ravel_r2_linear_%j.out
#SBATCH -e logs/slurm/rebuttal_autointerp_ravel_r2_linear_%j.err
#SBATCH --signal=B:TERM@120

# Graph-targeted autointerp for the r2 gpt2_small LINEAR ravel eval suites:
# extract unique feature IDs from evaluation/graphs/*.json, collect->explain->
# score on exactly those IDs (wikitext-2 val+test corpus), write explanations
# into annotated graph clerps. Companion to
# rebuttal_autointerp_ravel_r2_spline.sbatch (same vLLM endpoint, split so a
# single failure doesn't take out all 14 r2 arms).
#
# Prerequisites:
#   - r2 gpt2_small ravel eval suites finished (jobs 8601-8605, 8614) under
#     /gscratch/ssuresh/results/paper_rebuttal/ravel_eval_suite_r2_<label>
#   - vLLM endpoint at ENDPOINT_FILE (scripts/slurm/launch_vllm.sbatch)
#
# Submit:
#   sbatch --dependency=afterany:8601,8602,8603,8604,8605,8614 \
#       scripts/slurm/rebuttal_autointerp_ravel_r2_linear.sbatch
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

run_name_graphs r2_linear_dt13504 \
    "$R/ravel_eval_suite_r2_linear_dt13504/runs/r2_gpt2_small_linear_dt13504/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_linear_dt13504/runs/gpt2_small_linear_dt13504_r2/seed_101/checkpoints/gpt2_small_linear_dt13504_r2_seed101_best"

run_name_graphs r2_linear_dt1664 \
    "$R/ravel_eval_suite_r2_linear_dt1664/runs/r2_gpt2_small_linear_dt1664/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_linear_dt1664/runs/gpt2_small_linear_dt1664_r2/seed_101/checkpoints/gpt2_small_linear_dt1664_r2_seed101_best"

run_name_graphs r2_linear_dt3392 \
    "$R/ravel_eval_suite_r2_linear_dt3392/runs/r2_gpt2_small_linear_dt3392/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_linear_dt3392/runs/gpt2_small_linear_dt3392_r2/seed_101/checkpoints/gpt2_small_linear_dt3392_r2_seed101_best"

run_name_graphs r2_linear_dt6784 \
    "$R/ravel_eval_suite_r2_linear_dt6784/runs/r2_gpt2_small_linear_dt6784/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_linear_dt6784/runs/gpt2_small_linear_dt6784_r2/seed_101/checkpoints/gpt2_small_linear_dt6784_r2_seed101_best"

run_name_graphs r2_linear_feature_match \
    "$R/ravel_eval_suite_r2_linear_feature_match/runs/r2_gpt2_small_linear_feature_match/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_linear_feature_match/runs/gpt2_small_linear_feature_match_r2/seed_101/checkpoints/gpt2_small_linear_feature_match_r2_seed101_best"

run_name_graphs r2_linear_param_match \
    "$R/ravel_eval_suite_r2_linear_param_match/runs/r2_gpt2_small_linear_param_match/seed_101/evaluation/graphs" \
    "$R/paper_r2_gpt2_small_linear_param_match/runs/gpt2_small_linear_param_match_r2/seed_101/checkpoints/gpt2_small_linear_param_match_r2_seed101_best"

echo "[$(date -u +%FT%TZ)] r2 linear ravel graph autointerp complete"
