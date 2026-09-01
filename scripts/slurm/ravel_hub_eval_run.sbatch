#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J ravel-hub-eval
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -p gp-1
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/ravel_hub_eval_%j.out
#SBATCH -e logs/slurm/ravel_hub_eval_%j.err
#SBATCH --signal=B:TERM@120

# RAVEL replacement-fidelity eval for a published hub CLT suite
# (mntss/clt-llama-3.2-1b-524k or mntss/clt-gemma-2-2b-426k).
# Runs on gp-1 single-GPU nodes (gp101–gp112).
#
# Submit:  sbatch scripts/slurm/ravel_hub_eval_run.sbatch <suite.json>
# Env:     SKIP_GRAPHS=1  (replacement fidelity only)

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 SUITE_JSON_PATH" >&2
    exit 2
fi
SUITE="$1"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
# Explicit interpreter — `conda activate` can leave a non-ct python on some nodes.
PYTHON="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${PYTHON}" ]] || ! "${PYTHON}" -c "import torch" 2>/dev/null; then
    PYTHON=/home/ssuresh/.conda/envs/ct/bin/python
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/gscratch/ssuresh}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
# Models are already in the local hub cache; force offline so tokenizer
# loads don't hit huggingface.co (429 rate limits kill the job).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [[ "$(ps -o user= -p "${pid}" 2>/dev/null)" == "${USER}" ]]; then
        echo "reaping stray GPU pid ${pid}"
        kill -9 "${pid}" || true
    fi
done

SUITE_NAME=$("${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))['suite_name'])" "${SUITE}")
OUT_DIR="${OUT_DIR:-/gscratch/ssuresh/results/paper/${SUITE_NAME}}"
EXTRA=()
if [[ "${SKIP_GRAPHS:-0}" == "1" ]]; then
    EXTRA+=(--skip-graphs)
fi

echo "[$(date -u +%FT%TZ)] ravel hub eval: ${SUITE} -> ${OUT_DIR} (python=${PYTHON})"
"${PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"${PYTHON}" -m rebuttal_eval.ravel_hub_eval \
    --suite "${SUITE}" \
    --out-dir "${OUT_DIR}" \
    "${EXTRA[@]}"
echo "[$(date -u +%FT%TZ)] ravel hub eval complete: ${SUITE}"
