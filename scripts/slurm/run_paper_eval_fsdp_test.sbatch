#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J paper-fsdp-qwen3
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -p ai4wy-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/paper_fsdp_test_%j.out
#SBATCH -e logs/slurm/paper_fsdp_test_%j.err
# Deliver SIGTERM to the *batch shell* (B:) 120s before the wall-time limit so
# the cleanup trap below can tear torchrun down gracefully — letting workers
# release their CUDA context — before SLURM's hard SIGKILL. Without this, the
# elastic launcher's workers can get stuck in the NVIDIA driver (D-state) on an
# abrupt kill and leak the whole GPU until the node is rebooted.
#SBATCH --signal=B:TERM@120

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

# Use an expandable CUDA allocator: the chunked-offline loop builds/tears down a
# fresh FSDP model every chunk, which fragments the caching allocator. Expandable
# segments let freed blocks be returned/reused across those rebuilds instead of
# accumulating as "reserved but unallocated", which otherwise drives GPU memory
# up to the cap and thrashes the allocator (8x slower steps).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NCCL async error handling + heartbeat monitor. When a rank dies or is SIGTERM'd
# (walltime via --signal=B:TERM@120, or scancel) while a peer is blocked inside a
# collective, this makes NCCL abort the communicator and tear the process down
# instead of the peer wedging forever in an uninterruptible (D-state) GPU driver
# call — which previously left unkillable orphans that forced node reboots.
# NOTE: do NOT shorten the init_process_group timeout (4h, set in train.py); the
# long timeout is deliberate for >10-min rank-0 chunk collection skew. The
# watchdog/heartbeat below is the abort mechanism, not a shorter collective timeout.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"

export PAPER_OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-/cluster/ai4wy/gscratch/${USER}/results/paper}"

SUITE_GLOB="${SUITE_GLOB:-experiments/paper_configs/suites/*.json}"
DEFAULT_SUITE="paper_gpt2_fsdp_test"
SUITE_NAME="${SUITE_NAME:-${1:-${DEFAULT_SUITE}}}"

shopt -s nullglob
SUITES=( ${SUITE_GLOB} )
shopt -u nullglob

if [[ ${#SUITES[@]} -eq 0 ]]; then
    echo "No suite configs matched pattern: ${SUITE_GLOB}" >&2
    exit 1
fi

IFS=$'\n' SUITES=( $(printf '%s\n' "${SUITES[@]}" | sort) )
unset IFS

SELECTED_SUITE=""
for suite in "${SUITES[@]}"; do
    base="$(basename "${suite}")"
    stem="${base%.json}"
    if [[ "${SUITE_NAME}" == "${base}" || "${SUITE_NAME}" == "${stem}" ]]; then
        SELECTED_SUITE="${suite}"
        break
    fi
done

if [[ -z "${SELECTED_SUITE}" ]]; then
    echo "No suite matched SUITE_NAME='${SUITE_NAME}' under pattern ${SUITE_GLOB}" >&2
    echo "Available suite names:" >&2
    for suite in "${SUITES[@]}"; do
        base="$(basename "${suite}")"
        echo "  - ${base%.json}" >&2
    done
    exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[$(date -u +%FT%TZ)] Starting FSDP paper-eval test"
echo "  Host:               $(hostname)"
echo "  Suite name:         ${SUITE_NAME}"
echo "  Selected suite:     ${SELECTED_SUITE}"
echo "  Suite pattern:      ${SUITE_GLOB}"
echo "  Conda env:          ct"
echo "  Paper output_root:  ${PAPER_OUTPUT_ROOT}"
echo "  nproc_per_node:     ${NPROC_PER_NODE}"
echo "  master_port:        ${MASTER_PORT}"

rm -rf /lscratch/*

# --- Reap stray GPU processes before launching ---------------------------------
# Killed/timed-out jobs can leave orphaned worker processes that keep holding the
# GPU (the elastic launcher does not always tear its workers down cleanly). Each
# run starts by killing any of THIS user's GPU compute processes on the allocated
# node so we get a clean slate. We only touch processes owned by ${USER} so we
# never disturb another user's job. Processes wedged in the NVIDIA driver
# (uninterruptible D-state) cannot be killed from userspace — those are reported
# and the pre-flight check below bails out so the job can reschedule elsewhere.
kill_stray_gpu_procs() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local pids pid owner st
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
    if [[ -z "${pids}" ]]; then
        echo "[gpu-clean] no GPU compute processes present"
        return 0
    fi
    for pid in ${pids}; do
        owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ')"
        if [[ "${owner}" == "${USER}" ]]; then
            echo "[gpu-clean] SIGTERM stray GPU process ${pid} (user=${owner})"
            kill -TERM "${pid}" 2>/dev/null || true
        elif [[ -n "${owner}" ]]; then
            echo "[gpu-clean] leaving GPU process ${pid} owned by '${owner}' (not ${USER})"
        fi
    done
    sleep 5
    for pid in ${pids}; do
        owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ')"
        if [[ "${owner}" == "${USER}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "[gpu-clean] SIGKILL stubborn GPU process ${pid}"
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
    sleep 3
    for pid in ${pids}; do
        if kill -0 "${pid}" 2>/dev/null; then
            st="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ')"
            echo "[gpu-clean] WARNING: process ${pid} survived (state=${st:-?}) —" >&2
            echo "            likely wedged in the NVIDIA driver; node needs a reboot/gpu-reset." >&2
        fi
    done
}

echo "[$(date -u +%FT%TZ)] Reaping stray GPU processes on $(hostname) before launch..."
kill_stray_gpu_procs

# --- Pre-flight: after the reap, fail fast if a GPU is still mostly occupied
# (i.e. an unkillable driver-wedged leftover). Better to bail and reschedule than
# to waste minutes initializing only to OOM. ---
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] Pre-flight GPU check on $(hostname):"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv || true
    min_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)"
    if [[ -n "${min_free_mib}" && "${min_free_mib}" -lt 100000 ]]; then
        echo "[ABORT] A GPU on $(hostname) still has only ${min_free_mib} MiB free after reaping —" >&2
        echo "        an unkillable (D-state) leftover is holding it. Resubmit excluding this node:" >&2
        echo "          sbatch --exclude=$(hostname) ${BASH_SOURCE[0]} ${SUITE_NAME}" >&2
        echo "        and ask ARCC to reboot/gpu-reset $(hostname)." >&2
        exit 1
    fi
fi

# On exit, reap this run's own GPU processes too, so a crash/timeout leaves the
# node clean for the next job (best-effort; the startup reap is the guarantee).
trap kill_stray_gpu_procs EXIT

# echo "[$(date -u +%FT%TZ)] Stage 1/2: collect dataset (single process)"
# python -m spline_clt.paper_eval \
#     --suite "${SELECTED_SUITE}" \
#     --stages collect

# echo "[$(date -u +%FT%TZ)] Stage 2/2: train (torchrun, ${NPROC_PER_NODE} processes)"
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m spline_clt.paper_eval \
    --suite "${SELECTED_SUITE}"

echo "[$(date -u +%FT%TZ)] FSDP paper-eval test complete"
