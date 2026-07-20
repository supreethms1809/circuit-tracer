#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J paper-multinode
#SBATCH -N 4
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -p ai4wy-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/paper_multinode_%j.out
#SBATCH -e logs/slurm/paper_multinode_%j.err
# Deliver SIGTERM to the *batch shell* (B:) 120s before the wall-time limit so
# the trap below can forward it to the srun step (srun propagates the signal to
# torchrun on every node), letting workers checkpoint and release their CUDA
# context before SLURM's hard SIGKILL.
#SBATCH --signal=B:TERM@120

# Multi-node paper-eval launcher.
#
# Differences from run_paper_eval_fsdp_test.sbatch (single node):
#   - /lscratch cleanup + stray-GPU reaping + pre-flight run on EVERY node via srun
#     (the batch shell itself only executes on the head node).
#   - torchrun uses c10d rendezvous on the head node instead of localhost, and is
#     launched once per node via srun.
#   - Activation chunks are collected per node (LOCAL_RANK==0 on each node writes
#     its own /lscratch replica) — handled inside the paper runner.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
# Under `#SBATCH --export=NONE` the batch env is built fresh on the compute node.
# On nodes where slurm/ai4wy is not auto-loaded, `module load arcc` does NOT add
# it, leaving scontrol/srun off PATH (deterministic `scontrol: command not found`
# at rendezvous setup). Load it explicitly; it's a no-op when already present.
module load slurm/ai4wy/25.05
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

# The wrapper header keeps `#SBATCH --export=NONE` (CLI --export=ALL triggers
# user_env_retrieval_failed_requeued_held on this cluster). That sets
# SLURM_EXPORT_ENV=NONE, so srun steps below inherit a minimal env without
# /usr/bin (bash/nvidia-smi/torchrun `execve(): No such file or directory`).
# Now that the batch shell env is fully populated (modules + conda), override
# propagation so all srun steps inherit it.
export SLURM_EXPORT_ENV=ALL

# ~/.bashrc sets HF_HOME=/gscratch/ssuresh, but `--export=NONE` + a non-interactive
# batch shell means .bashrc is never sourced, so HF defaults the datasets/hub cache
# to /home/$USER/.cache/huggingface. /home has a 50 GB quota; collecting activations
# streams the raw corpus and blows it (`OSError [Errno 122] Disk quota exceeded`).
# Pin it to gscratch (5 TB), where the codelion corpus is already cached — no re-download.
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"

# See run_paper_eval_fsdp_test.sbatch for the rationale behind each of these.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"

# The ai4wy nodes expose two ACTIVE Mellanox ports with DIFFERENT link types:
# mlx5_0 (InfiniBand) and mlx5_1 (Ethernet/RoCE). NCCL auto-selects both and
# then fails cross-node with "Remote RoCE device is incompatible with the
# local IB" at the first collective. Pin NCCL to the real IB HCA only.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0:1}"

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
MASTER_PORT="${MASTER_PORT:-29500}"
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"

echo "[$(date -u +%FT%TZ)] Starting multi-node paper-eval"
echo "  Head node:          ${HEAD_NODE}"
echo "  Node list:          ${SLURM_JOB_NODELIST}"
echo "  Nodes:              ${SLURM_NNODES}"
echo "  Suite name:         ${SUITE_NAME}"
echo "  Selected suite:     ${SELECTED_SUITE}"
echo "  Conda env:          ct"
echo "  Paper output_root:  ${PAPER_OUTPUT_ROOT}"
echo "  nproc_per_node:     ${NPROC_PER_NODE}"
echo "  rdzv endpoint:      ${HEAD_NODE}:${MASTER_PORT}"

# --- Per-node preparation ------------------------------------------------------
# The batch shell runs on the head node only, so scratch cleanup, stray-GPU
# reaping, and the pre-flight free-memory check must be fanned out with srun.
# Semantics per node are identical to the single-node script:
#   1. wipe the node-local activation cache (regenerable; a stale cache would
#      desync the per-node chunk replicas),
#   2. SIGTERM→SIGKILL this user's leftover GPU compute processes,
#   3. abort if a GPU is still mostly occupied (unkillable D-state leftover).
NODE_PREP=$(cat <<'PREP_EOF'
set -uo pipefail
host="$(hostname)"

rm -rf /lscratch/* 2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
    if [[ -n "${pids}" ]]; then
        for pid in ${pids}; do
            owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ')"
            if [[ "${owner}" == "${USER}" ]]; then
                echo "[gpu-clean ${host}] SIGTERM stray GPU process ${pid}"
                kill -TERM "${pid}" 2>/dev/null || true
            elif [[ -n "${owner}" ]]; then
                echo "[gpu-clean ${host}] leaving GPU process ${pid} owned by '${owner}' (not ${USER})"
            fi
        done
        sleep 5
        for pid in ${pids}; do
            owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ')"
            if [[ "${owner}" == "${USER}" ]] && kill -0 "${pid}" 2>/dev/null; then
                echo "[gpu-clean ${host}] SIGKILL stubborn GPU process ${pid}"
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        done
        sleep 3
        for pid in ${pids}; do
            if kill -0 "${pid}" 2>/dev/null; then
                st="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ')"
                echo "[gpu-clean ${host}] WARNING: process ${pid} survived (state=${st:-?}) —" >&2
                echo "            likely wedged in the NVIDIA driver; node needs a reboot/gpu-reset." >&2
            fi
        done
    else
        echo "[gpu-clean ${host}] no GPU compute processes present"
    fi

    echo "[preflight ${host}]"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv || true
    min_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)"
    if [[ -n "${min_free_mib}" && "${min_free_mib}" -lt 100000 ]]; then
        echo "[ABORT ${host}] A GPU still has only ${min_free_mib} MiB free after reaping —" >&2
        echo "        an unkillable (D-state) leftover is holding it. Resubmit excluding this node" >&2
        echo "        (sbatch --exclude=${host} ...) and ask ARCC to reboot/gpu-reset it." >&2
        exit 1
    fi
fi
PREP_EOF
)

echo "[$(date -u +%FT%TZ)] Per-node prep (scratch wipe + GPU reap + preflight) on ${SLURM_NNODES} node(s)..."
srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "${NODE_PREP}"

# Best-effort end-of-job reap on every node so a crash/timeout leaves the nodes
# clean for the next job (the startup prep above is the guarantee).
reap_all_nodes() {
    srun --ntasks-per-node=1 bash -c "${NODE_PREP}" 2>/dev/null || true
}
trap reap_all_nodes EXIT

# --- Launch --------------------------------------------------------------------
# One torchrun per node (srun task), all rendezvousing at the head node. srun
# forwards SIGTERM to every task, torchrun forwards it to its workers, and the
# training loop's cooperative-shutdown handler checkpoints and exits cleanly.
echo "[$(date -u +%FT%TZ)] Launching torchrun on ${SLURM_NNODES} node(s) x ${NPROC_PER_NODE} GPU(s)"
srun --ntasks-per-node=1 --kill-on-bad-exit=1 torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${HEAD_NODE}:${MASTER_PORT}" \
    -m spline_clt.paper_eval \
    --suite "${SELECTED_SUITE}" &
LAUNCH_PID=$!

on_term() {
    echo "[$(date -u +%FT%TZ)] SIGTERM received — forwarding to srun step for graceful shutdown"
    kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
}
trap on_term TERM

# First wait can be interrupted by the TERM trap (returns >128); wait again to
# reap the real exit status of the srun step.
set +e
wait "${LAUNCH_PID}"
rc=$?
wait "${LAUNCH_PID}" 2>/dev/null
rc2=$?
if [[ ${rc2} -ne 127 ]]; then
    rc=${rc2}
fi
set -e

echo "[$(date -u +%FT%TZ)] Multi-node paper-eval finished with exit code ${rc}"
exit "${rc}"
