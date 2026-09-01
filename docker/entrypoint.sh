#!/usr/bin/env bash
# Collect activations if missing, then train Spline-CLT.
# MODE=train (default) uses experiments/train_spline_clt.py.
# MODE=paper-eval runs python -m spline_clt.paper_eval (optional torchrun).
# Set NSYS=1 to wrap the train/paper-eval command in Nsight Systems.
set -euo pipefail

cd /workspace/circuit-tracer

MODE="${MODE:-train}"
TRAIN_CONFIG="${TRAIN_CONFIG:-experiments/configs/gpt2_small.yaml}"
PAPER_SUITE="${PAPER_SUITE:-}"
DEVICE="${DEVICE:-cuda}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
N_TOKENS="${N_TOKENS:-}"
DATASET_NAME="${DATASET_NAME:-}"
RESUME="${RESUME:-1}"
NSYS="${NSYS:-0}"
NSYS_STEPS="${NSYS_STEPS:-}"
TOTAL_STEPS="${TOTAL_STEPS:-${NSYS_STEPS}}"
NSYS_OUTPUT="${NSYS_OUTPUT:-profiles/nsys_spline_clt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PAPER_OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-/workspace/circuit-tracer/results/paper}"
export PAPER_OUTPUT_ROOT
export PYTHONPATH="${PYTHONPATH:-/workspace/circuit-tracer}"

echo "=== Spline-CLT container ==="
echo "  mode:      ${MODE}"
echo "  config:    ${TRAIN_CONFIG}"
echo "  suite:     ${PAPER_SUITE:-<none>}"
echo "  device:    ${DEVICE}"
echo "  nproc:     ${NPROC_PER_NODE}"
echo "  nsys:      ${NSYS}"
echo "  steps:     ${TOTAL_STEPS:-<from config>}"
echo "  cwd:       $(pwd)"
python - <<'PY'
import torch
print(f"  torch:     {torch.__version__}")
print(f"  cuda:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu:       {torch.cuda.get_device_name(0)}")
    print(f"  gpu_count: {torch.cuda.device_count()}")
    print(f"  gpu_mem:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

if [[ "${DEVICE}" == "cuda" ]]; then
    python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not visible inside the container.", file=sys.stderr)
    print("Pass --gpus all (or use docker compose) and confirm nvidia-smi works on the host.", file=sys.stderr)
    sys.exit(1)
PY
fi

run_cmd() {
    if [[ "${NSYS}" == "1" ]]; then
        if ! command -v nsys >/dev/null 2>&1; then
            echo "ERROR: NSYS=1 but nsys is not on PATH. Rebuild the image from the current Dockerfile." >&2
            exit 1
        fi
        mkdir -p "$(dirname "${NSYS_OUTPUT}")"
        # shellcheck disable=SC2206
        nsys_extra=( ${NSYS_EXTRA_ARGS:-} )
        echo "=== Nsight Systems → ${NSYS_OUTPUT}.nsys-rep ==="
        exec nsys profile \
            --force-overwrite=true \
            --output="${NSYS_OUTPUT}" \
            --trace=cuda,nvtx,osrt,cudnn,cublas \
            --cuda-memory-usage=true \
            --stats=true \
            --sample=process-tree \
            "${nsys_extra[@]}" \
            "$@"
    fi
    exec "$@"
}

if [[ "${MODE}" == "paper-eval" ]]; then
    if [[ -z "${PAPER_SUITE}" ]]; then
        echo "ERROR: MODE=paper-eval requires PAPER_SUITE (path to a suite JSON)." >&2
        exit 1
    fi
    paper_cmd=()
    if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
        paper_cmd+=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}")
    fi
    paper_cmd+=(python -m spline_clt.paper_eval --suite "${PAPER_SUITE}")
    if [[ -n "${STAGES:-}" ]]; then
        # shellcheck disable=SC2206
        paper_cmd+=(--stages ${STAGES})
    fi
    paper_cmd+=("$@")
    echo "=== paper-eval ==="
    echo "  cmd: ${paper_cmd[*]}"
    echo "  PAPER_OUTPUT_ROOT=${PAPER_OUTPUT_ROOT}"
    run_cmd "${paper_cmd[@]}"
fi

if [[ "${MODE}" != "train" ]]; then
    echo "ERROR: unknown MODE='${MODE}' (expected train or paper-eval)." >&2
    exit 1
fi

data_dir="$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${TRAIN_CONFIG}")) or {}
print(cfg.get("data_dir", "data/activations"))
PY
)"
activations_present=0
if [[ -f "${data_dir}/mlp_inputs_train.npy" || -f "${data_dir}/mlp_inputs.npy" || -f "${data_dir}/mlp_inputs.pt" ]]; then
    activations_present=1
fi

if [[ "${SKIP_COLLECT}" != "1" && "${activations_present}" -eq 0 ]]; then
    echo "=== Collecting activations (none found under ${data_dir}) ==="
    collect_args=(
        --collect-data
        --collect-only
        --config "${TRAIN_CONFIG}"
        --device "${DEVICE}"
    )
    if [[ -n "${N_TOKENS}" ]]; then
        collect_args+=(--n-tokens "${N_TOKENS}")
    fi
    if [[ -n "${DATASET_NAME}" ]]; then
        collect_args+=(--dataset-name "${DATASET_NAME}")
    fi
    python experiments/train_spline_clt.py "${collect_args[@]}"
    echo "=== Collection finished; starting training ==="
elif [[ "${activations_present}" -eq 1 ]]; then
    echo "=== Found existing activations in ${data_dir}; skipping collection ==="
else
    echo "=== SKIP_COLLECT=1; using existing data in ${data_dir} ==="
fi

train_cmd=(
    python experiments/train_spline_clt.py
    --config "${TRAIN_CONFIG}"
    --device "${DEVICE}"
)
if [[ "${RESUME}" == "1" ]]; then
    train_cmd+=(--resume)
fi
if [[ -n "${TOTAL_STEPS}" ]]; then
    train_cmd+=(--total-steps "${TOTAL_STEPS}")
fi
train_cmd+=("$@")

echo "=== Training Spline-CLT ==="
echo "  cmd: ${train_cmd[*]}"
run_cmd "${train_cmd[@]}"
