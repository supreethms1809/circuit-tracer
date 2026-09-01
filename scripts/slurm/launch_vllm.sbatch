#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J vllm-autointerp
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -p gp-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/vllm_%j.out
#SBATCH -e logs/slurm/vllm_%j.err
#SBATCH --signal=B:TERM@60

# Serve an OpenAI-compatible LLM endpoint for auto-interp scoring on a GPU
# node, and advertise it via an endpoint file that
# rebuttal_eval.autointerp (--endpoint-file) polls from any other node.
#
# Submit:  sbatch scripts/slurm/launch_vllm.sbatch
# Env: VLLM_MODEL (default Qwen/Qwen2.5-72B-Instruct), VLLM_TP (default 2),
#      VLLM_MAX_LEN (default 8192), VLLM_PORT (default 8192),
#      ENDPOINT_FILE (default <repo>/results/rebuttal/vllm_endpoint.json)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm" "${REPO_ROOT}/results/rebuttal"

module load arcc miniconda3
# Compute nodes (gp*) do not provide a writable /home/ssuresh mount.
# vLLM/flashinfer still expand caches under $HOME → PermissionError (job 13550).
export HOME=/cluster/ai4wy/home/ssuresh
export XDG_CACHE_HOME="${HOME}/.cache"
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME}/vllm"
export FLASHINFER_WORKSPACE_DIR="${XDG_CACHE_HOME}/flashinfer"
mkdir -p "${VLLM_CACHE_ROOT}" "${FLASHINFER_WORKSPACE_DIR}" \
    "${XDG_CACHE_HOME}/huggingface" "${HOME}/.cache/vllm/torch_compile_cache"

# Invoke via cluster-path python: the vllm console-script shebang hardcodes
# #!/home/ssuresh/... which does not exist on gp* nodes (job 13547).
VLLM_ENV="${VLLM_ENV:-/cluster/ai4wy/home/ssuresh/.conda/envs/vllm_env}"
VLLM_PY="${VLLM_ENV}/bin/python3.12"
VLLM_BIN="${VLLM_ENV}/bin/vllm"
if [[ ! -x "${VLLM_PY}" || ! -f "${VLLM_BIN}" ]]; then
    echo "vllm env incomplete under ${VLLM_ENV}" >&2
    exit 1
fi
export PATH="${VLLM_ENV}/bin:${PATH}"
export VIRTUAL_ENV="${VLLM_ENV}"
echo "[$(date -u +%FT%TZ)] HOME=${HOME} using ${VLLM_PY} ${VLLM_BIN}"

# Model cache lives on gscratch (bashrc export doesn't reach batch shells).
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
# This source build selects the DeepGEMM FP8 backend on Hopper and aborts if
# deep_gemm is missing (job 8354). Force the non-FP8 kernel path instead.
export VLLM_USE_DEEP_GEMM=0

# Reap any of our own stray GPU processes left on this node before starting.
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
TP="${VLLM_TP:-2}"
MAX_LEN="${VLLM_MAX_LEN:-8192}"
PORT="${VLLM_PORT:-8192}"
ENDPOINT_FILE="${ENDPOINT_FILE:-${REPO_ROOT}/results/rebuttal/vllm_endpoint.json}"
HOST_NAME="$(hostname)"

rm -f "${ENDPOINT_FILE}"

echo "[$(date -u +%FT%TZ)] starting vllm serve ${MODEL} (tp=${TP}, max_len=${MAX_LEN}) on ${HOST_NAME}:${PORT}"
"${VLLM_PY}" "${VLLM_BIN}" serve "${MODEL}" --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size "${TP}" --max-model-len "${MAX_LEN}" &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true; rm -f "${ENDPOINT_FILE}"' EXIT TERM

# Wait for the health endpoint, then advertise. Model load can take minutes.
for _ in $(seq 1 240); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vllm serve exited during startup" >&2
        exit 1
    fi
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        break
    fi
    sleep 10
done
if ! curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "vllm health endpoint never came up" >&2
    exit 1
fi

cat > "${ENDPOINT_FILE}" <<EOF
{"host": "${HOST_NAME}", "port": ${PORT}, "model": "${MODEL}", "job_id": "${SLURM_JOB_ID}"}
EOF
echo "[$(date -u +%FT%TZ)] endpoint ready: ${HOST_NAME}:${PORT} (${ENDPOINT_FILE})"

wait "${VLLM_PID}"
