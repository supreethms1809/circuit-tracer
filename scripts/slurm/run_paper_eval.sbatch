#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J paper-eval
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/paper_eval_%j.out
#SBATCH -e logs/slurm/paper_eval_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs/slurm"

module load arcc miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct

export PAPER_OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-/cluster/ai4wy/gscratch/${USER}/results/paper}"

SUITE_GLOB="${SUITE_GLOB:-experiments/paper_configs/suites/*.json}"
shopt -s nullglob
SUITES=( ${SUITE_GLOB} )
shopt -u nullglob

if [[ ${#SUITES[@]} -eq 0 ]]; then
    echo "No suite configs matched pattern: ${SUITE_GLOB}" >&2
    exit 1
fi

IFS=$'\n' SUITES=( $(printf '%s\n' "${SUITES[@]}" | sort) )
unset IFS

SUITE_NAME="${SUITE_NAME:-${1:-}}"

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
    echo "Available suite names:" >&2
    for suite in "${SUITES[@]}"; do
        base="$(basename "${suite}")"
        echo "  - ${base%.json}" >&2
    done
    exit 1
fi

echo "[$(date -u +%FT%TZ)] Starting paper-eval"
echo "  Host:              $(hostname)"
echo "  Suite name:        ${SUITE_NAME}"
echo "  Selected suite:    ${SELECTED_SUITE}"
echo "  Suite pattern:     ${SUITE_GLOB}"
echo "  Conda env:         ct"
echo "  Paper output_root:${PAPER_OUTPUT_ROOT}"

rm -rf /lscratch/* 2>/dev/null || true

python -m spline_clt.paper_eval --suite "${SELECTED_SUITE}"

echo "[$(date -u +%FT%TZ)] paper-eval complete"
