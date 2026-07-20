#!/usr/bin/env bash
# Launch every paper_configs_v2 suite as its own 2-node (2x2 GPU) FSDP job,
# all submitted at once. Each suite has a distinct target model, so the jobs
# are fully independent: separate node pairs, separate node-local /lscratch
# train caches (collected per node), and separate shared val dirs.
#
# Usage:
#   bash scripts/slurm/launch_paper_v2_all.sh            # submit all suites
#   bash scripts/slurm/launch_paper_v2_all.sh gpt2_small # submit a subset (substring match)
#
# Environment overrides:
#   NODES_PER_JOB   (default 2)
#   NPROC_PER_NODE  (default 2 GPUs per node)
#   SBATCH_EXTRA    extra sbatch args, e.g. SBATCH_EXTRA="--exclude=badnode01"
#   PAPER_OUTPUT_ROOT

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PAPER_OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-/cluster/ai4wy/gscratch/${USER}/results/paper}"
FILTER="${1:-}"

# Nodes per suite. The step is compute-bound per GPU (the encoder all-reduce is
# 0.1% of step time), so tokens/s scales ~linearly with GPUs and the only reason
# to take a second node is to fit the model. gpt2-small does fit on one, so
# holding two for it just idles a node.
default_nodes_for() {
    case "$1" in
        paper_v2_gpt2_small*) echo 1 ;;
        paper_v2_gpt2_large) echo 3 ;;  # 36 layers, d_t=5120: 3x2 GPU, total_steps=38650
        *)                   echo 2 ;;
    esac
}

# Override to launch one-off probes without them being swept by the no-filter run,
# e.g. SUITE_DIR=experiments/paper_configs_v2/probes bash launch_paper_v2_all.sh <name>
SUITE_DIR="${SUITE_DIR:-experiments/paper_configs_v2/suites}"
LAUNCHER="scripts/slurm/run_paper_eval_multinode.sbatch"
WRAPPER_DIR="${REPO_ROOT}/logs/slurm/wrappers"
mkdir -p logs/slurm "${WRAPPER_DIR}"

shopt -s nullglob
SUITES=( "${SUITE_DIR}"/*.json )
shopt -u nullglob
if [[ ${#SUITES[@]} -eq 0 ]]; then
    echo "No suite configs found under ${SUITE_DIR}" >&2
    exit 1
fi

submitted=0
for suite in "${SUITES[@]}"; do
    stem="$(basename "${suite}" .json)"
    if [[ -n "${FILTER}" && "${stem}" != *"${FILTER}"* ]]; then
        continue
    fi
    nodes="${NODES_PER_JOB:-$(default_nodes_for "${stem}")}"
    echo "Submitting ${stem}: ${nodes} node(s) x ${NPROC_PER_NODE} GPU(s)"

    # Write a thin sbatch wrapper with #SBATCH --export=NONE and bake env vars
    # into the script body. CLI --export=ALL / --export=VAR=... on this cluster
    # triggers user_env_retrieval_failed_requeued_held (BatchFlag=2); embedding
    # --export=NONE in the batch script headers avoids that (BatchFlag=1).
    wrapper="${WRAPPER_DIR}/${stem}.sbatch"
    cat > "${wrapper}" <<EOF
#!/usr/bin/env bash
#SBATCH -A uwyo-0002
#SBATCH -J pv2-${stem}
#SBATCH -N ${nodes}
#SBATCH --gpus-per-node=${NPROC_PER_NODE}
#SBATCH --ntasks-per-node=1
#SBATCH -p ai4wy-2
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH -o logs/slurm/${stem}_%j.out
#SBATCH -e logs/slurm/${stem}_%j.err
#SBATCH --signal=B:TERM@120
#SBATCH --export=NONE
set -euo pipefail
export SUITE_GLOB="${SUITE_DIR}/*.json"
export NPROC_PER_NODE="${NPROC_PER_NODE}"
export PAPER_OUTPUT_ROOT="${PAPER_OUTPUT_ROOT}"
cd "${REPO_ROOT}"
exec bash ${LAUNCHER} ${stem}
EOF
    chmod +x "${wrapper}"

    sbatch ${SBATCH_EXTRA:-} "${wrapper}"
    submitted=$((submitted + 1))
done

if [[ ${submitted} -eq 0 ]]; then
    echo "No suites matched filter '${FILTER}'. Available:" >&2
    for suite in "${SUITES[@]}"; do
        echo "  - $(basename "${suite}" .json)" >&2
    done
    exit 1
fi

echo "Submitted ${submitted} job(s). Monitor with: squeue -u \${USER}"
