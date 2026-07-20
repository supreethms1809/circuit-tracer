#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J collect-acts
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 06:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/collect_acts_%j.out
#SBATCH -e logs/slurm/collect_acts_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

SUITE_GLOB="${SUITE_GLOB:-experiments/paper_configs/suites/*.json}"
SUITE_NAME="${SUITE_NAME:-${1:-}}"

shopt -s nullglob
SUITES=( ${SUITE_GLOB} )
shopt -u nullglob

if [[ ${#SUITES[@]} -eq 0 ]]; then
    echo "No suite configs matched pattern: ${SUITE_GLOB}" >&2
    exit 1
fi

IFS=$'\n' SUITES=( $(printf '%s\n' "${SUITES[@]}" | sort) )
unset IFS

if [[ -z "${SUITE_NAME}" ]]; then
    echo "SUITE_NAME is required." >&2
    echo "Available suite names:" >&2
    for suite in "${SUITES[@]}"; do
        base="$(basename "${suite}")"
        echo "  - ${base%.json}" >&2
    done
    exit 1
fi

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
    exit 1
fi

export REPO_ROOT SELECTED_SUITE

SUITE_STEM="$(basename "${SELECTED_SUITE}" .json)"
LOCAL_BASE="${LOCAL_SCRATCH_BASE:-/lscratch/${SLURM_JOB_ID:-manual}}"
LOCAL_SUITE_ROOT="${LOCAL_BASE}/paper_eval_collect/${SUITE_STEM}"
mkdir -p "${LOCAL_SUITE_ROOT}"

echo "[$(date -u +%FT%TZ)] Starting collection staging run"
echo "  Suite:            ${SELECTED_SUITE}"
echo "  Local scratch:    ${LOCAL_SUITE_ROOT}"

SUITE_INFO_JSON="$(python - <<'PY'
import json
import os
from pathlib import Path

from spline_clt.paper.config import load_suite_config

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
suite_path = Path(os.environ["SELECTED_SUITE"]).resolve()
config, _ = load_suite_config(suite_path)
suite_root = (repo_root / config.output_root / config.suite_name).resolve()
datasets_root = (suite_root / config.dataset.cache_dir).resolve()
model_names = sorted({variant.model_name or config.dataset.model_name for variant in config.model_variants.values()})

print(
    json.dumps(
        {
            "dataset_name": config.dataset.dataset_name,
            "dataset_config": config.dataset.dataset_config,
            "n_tokens": config.dataset.n_tokens,
            "seq_len": config.dataset.seq_len,
            "batch_size": config.dataset.batch_size,
            "seed": config.dataset.seed,
            "device": config.dataset.device,
            "datasets_root": str(datasets_root),
            "models": model_names,
        }
    )
)
PY
)"

export SUITE_INFO_JSON

mapfile -t MODEL_NAMES < <(python - <<'PY'
import json
import os

payload = json.loads(os.environ["SUITE_INFO_JSON"])
for model_name in payload["models"]:
    print(model_name)
PY
)

DATASET_NAME="$(python - <<'PY'
import json
import os
print(json.loads(os.environ["SUITE_INFO_JSON"])["dataset_name"])
PY
)"
DATASET_CONFIG="$(python - <<'PY'
import json
import os
print(json.loads(os.environ["SUITE_INFO_JSON"])["dataset_config"])
PY
)"
N_TOKENS="$(python - <<'PY'
import json
import os
print(json.loads(os.environ["SUITE_INFO_JSON"])["n_tokens"])
PY
)"
SEQ_LEN="$(python - <<'PY'
import json
import os
print(json.loads(os.environ["SUITE_INFO_JSON"])["seq_len"])
PY
)"
BATCH_SIZE="$(python - <<'PY'
import json
import os
print(json.loads(os.environ["SUITE_INFO_JSON"])["batch_size"])
PY
)"
DATASET_SEED="$(python - <<'PY'
import json
import os
print(json.loads(os.environ["SUITE_INFO_JSON"])["seed"])
PY
)"
DATASET_DEVICE="$(python - <<'PY'
import json
import os
value = json.loads(os.environ["SUITE_INFO_JSON"])["device"]
print("" if value is None else value)
PY
)"
GSCRATCH_DATASETS_ROOT="$(python - <<'PY'
import json
import os
print(json.loads(os.environ["SUITE_INFO_JSON"])["datasets_root"])
PY
)"

echo "  Dataset source:   ${DATASET_NAME}"
echo "  Dataset config:   ${DATASET_CONFIG:-<default>}"
echo "  Tokens/seq/batch: ${N_TOKENS}/${SEQ_LEN}/${BATCH_SIZE}"
echo "  Dataset root out: ${GSCRATCH_DATASETS_ROOT}"
echo "  Models:           ${MODEL_NAMES[*]}"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    LOCAL_MODEL_DIR="${LOCAL_SUITE_ROOT}/${MODEL_NAME}"
    DEST_MODEL_DIR="${GSCRATCH_DATASETS_ROOT}/${MODEL_NAME}"

    if [[ -f "${DEST_MODEL_DIR}/mlp_inputs.npy" && -f "${DEST_MODEL_DIR}/mlp_outputs.npy" ]]; then
        echo "[$(date -u +%FT%TZ)] ${MODEL_NAME}: already exists on gscratch, skipping"
        continue
    fi

    rm -rf "${LOCAL_MODEL_DIR}"
    mkdir -p "${LOCAL_MODEL_DIR}"

    echo "[$(date -u +%FT%TZ)] ${MODEL_NAME}: collecting to local scratch ${LOCAL_MODEL_DIR}"

    export MODEL_NAME LOCAL_MODEL_DIR DATASET_NAME DATASET_CONFIG N_TOKENS SEQ_LEN BATCH_SIZE DATASET_SEED DATASET_DEVICE
    python - <<'PY'
import os

from spline_clt.training.data import DataConfig, collect_activations

device = os.environ["DATASET_DEVICE"] or None
cfg = DataConfig(
    model_name=os.environ["MODEL_NAME"],
    dataset_name=os.environ["DATASET_NAME"],
    dataset_config=os.environ["DATASET_CONFIG"],
    n_tokens=int(os.environ["N_TOKENS"]),
    seq_len=int(os.environ["SEQ_LEN"]),
    batch_size=int(os.environ["BATCH_SIZE"]),
    save_dir=os.environ["LOCAL_MODEL_DIR"],
    device=device,
    seed=int(os.environ["DATASET_SEED"]),
    load_after_collect=False,
)
collect_activations(cfg)
PY

    if [[ ! -f "${LOCAL_MODEL_DIR}/mlp_inputs.npy" || ! -f "${LOCAL_MODEL_DIR}/mlp_outputs.npy" ]]; then
        echo "Collection completed but expected files are missing for ${MODEL_NAME}" >&2
        exit 1
    fi

    mkdir -p "$(dirname "${DEST_MODEL_DIR}")"
    DEST_TMP="${DEST_MODEL_DIR}.tmp_${SLURM_JOB_ID:-manual}_$$"
    rm -rf "${DEST_TMP}"
    mkdir -p "${DEST_TMP}"

    echo "[$(date -u +%FT%TZ)] ${MODEL_NAME}: syncing local -> ${DEST_TMP}"
    rsync -a "${LOCAL_MODEL_DIR}/" "${DEST_TMP}/"

    if [[ ! -f "${DEST_TMP}/mlp_inputs.npy" || ! -f "${DEST_TMP}/mlp_outputs.npy" ]]; then
        echo "Synced copy missing required files for ${MODEL_NAME}" >&2
        exit 1
    fi

    rm -rf "${DEST_MODEL_DIR}"
    mv "${DEST_TMP}" "${DEST_MODEL_DIR}"
    echo "[$(date -u +%FT%TZ)] ${MODEL_NAME}: published to ${DEST_MODEL_DIR}"
done

echo "[$(date -u +%FT%TZ)] Collection staging run complete"
