#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J gemma2-ravel-interp
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p gp-1
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/rebuttal_autointerp_ravel_gemma2_%j.out
#SBATCH -e logs/slurm/rebuttal_autointerp_ravel_gemma2_%j.err
#SBATCH --signal=B:TERM@120

# Graph-targeted autointerp for Gemma-2-2B RAVEL circuits:
#   1) spline dt6144 (r5b_b3 BaseJump local checkpoint)
#   2) published linear hub CLT (mntss/clt-gemma-2-2b-426k)
#
# Client needs 1×H200 on gp-1 (gp101–gp112). vLLM judge stays on a
# separate 2-GPU gp-2 job (scripts/slurm/launch_vllm.sbatch).
#
# Do NOT use --dependency=afterok:<vllm_job>: vLLM is a long-lived server, so
# afterok only fires when it exits (and the endpoint is then gone). Submit
# independently once results/rebuttal/vllm_endpoint.json exists, or use
# --dependency=after:<vllm_job> (starts after vLLM begins) and let this
# script wait on the endpoint file / health check.
#
# Submit:
#   sbatch scripts/slurm/launch_vllm.sbatch
#   # after endpoint file appears:
#   sbatch scripts/slurm/rebuttal_autointerp_ravel_gemma2.sbatch
#
# Env: ENDPOINT_FILE, LLM_MODEL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
# gp* nodes: /home/ssuresh is not a writable home mount. Point caches there via
# the real cluster path (same fix as launch_vllm.sbatch / job 13556).
export HOME=/cluster/ai4wy/home/ssuresh
export XDG_CACHE_HOME="${HOME}/.cache"
mkdir -p "${XDG_CACHE_HOME}"

CT_ENV="${CT_ENV:-/cluster/ai4wy/home/ssuresh/.conda/envs/ct}"
PYTHON="${CT_ENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "ct python missing: ${PYTHON}" >&2
    exit 1
fi
export PATH="${CT_ENV}/bin:${PATH}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[$(date -u +%FT%TZ)] python=${PYTHON}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

ENDPOINT_FILE="${ENDPOINT_FILE:-${REPO_ROOT}/results/rebuttal/vllm_endpoint.json}"
LLM_MODEL="${LLM_MODEL:-}"
OUT=/gscratch/ssuresh/results/paper_r5b_b3
BASE_MODEL="google/gemma-2-2b"

SPLINE_GRAPHS=/gscratch/ssuresh/results/paper_r5b_b3/ravel_eval_suite_r5b_b3_gemma2_2b_spline_dt6144/runs/r5b_b3_gemma2_2b_spline_dt6144/seed_101/evaluation/graphs
SPLINE_CKPT=/gscratch/ssuresh/results/paper_r5b_b3/paper_r5b_b3_gemma2_2b_spline_dt6144_basejump/runs/gemma2_2b_spline_dt6144_r5b_b3_bj/seed_101/checkpoints/gemma2_2b_spline_dt6144_r5b_b3_basejump_seed101_best

HUB_GRAPHS=/gscratch/ssuresh/results/paper/ravel_eval_suite_v3_hub_gemma2/runs/hub_gemma2_426k/seed_101/evaluation/graphs
HUB_CKPT=$("${PYTHON}" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("mntss/clt-gemma-2-2b-426k", local_files_only=True))
PY
)

run_name_graphs () {
    local label="$1" graphs_dir="$2" ckpt="$3"
    local out_dir="${OUT}/autointerp_ravel_${label}"
    local n_graphs
    n_graphs=$(find "${graphs_dir}" -maxdepth 1 -name '*.json' ! -name 'graph-metadata.json' 2>/dev/null | wc -l)
    if [[ "${n_graphs}" -lt 1 ]]; then
        echo "[$(date -u +%FT%TZ)] SKIP ${label}: no graphs in ${graphs_dir}" >&2
        return 0
    fi
    if [[ ! -d "${ckpt}" ]]; then
        echo "[$(date -u +%FT%TZ)] SKIP ${label}: checkpoint missing ${ckpt}" >&2
        return 0
    fi
    if [[ ! -f "${ENDPOINT_FILE}" ]]; then
        echo "[$(date -u +%FT%TZ)] FAIL ${label}: missing vLLM endpoint file ${ENDPOINT_FILE}" >&2
        return 1
    fi
    echo "[$(date -u +%FT%TZ)] name_graphs ${label} (n_graphs=${n_graphs}) ckpt=${ckpt} -> ${out_dir}"
    if ! "${PYTHON}" -m rebuttal_eval.autointerp.name_graphs \
        --graphs-dir "${graphs_dir}" \
        --checkpoint "${ckpt}" \
        --model "${BASE_MODEL}" \
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

if [[ "${SKIP_SPLINE:-0}" != "1" ]]; then
    run_name_graphs r5b_b3_gemma2_2b_spline_dt6144 "${SPLINE_GRAPHS}" "${SPLINE_CKPT}"
else
    echo "[$(date -u +%FT%TZ)] SKIP_SPLINE=1 — reusing existing spline autointerp artifacts"
fi
run_name_graphs r5b_b3_gemma2_2b_hub_linear_426k "${HUB_GRAPHS}" "${HUB_CKPT}"

# Structural circuit comparison (feature-ID spaces differ — report sizes +
# per-prompt retained counts / completeness, not cross-model Jaccard).
echo "[$(date -u +%FT%TZ)] writing circuit comparison summary"
"${PYTHON}" -m rebuttal_eval.autointerp.compare_ravel_circuits \
    --spline-graphs "${SPLINE_GRAPHS}" \
    --linear-graphs "${HUB_GRAPHS}" \
    --spline-autointerp "${OUT}/autointerp_ravel_r5b_b3_gemma2_2b_spline_dt6144" \
    --linear-autointerp "${OUT}/autointerp_ravel_r5b_b3_gemma2_2b_hub_linear_426k" \
    --out-dir "${OUT}/ravel_circuit_compare_gemma2_spline_vs_hub" \
    || echo "[$(date -u +%FT%TZ)] compare step failed (non-fatal)" >&2

echo "[$(date -u +%FT%TZ)] gemma2 ravel graph autointerp complete"
