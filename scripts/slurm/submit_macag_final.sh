#!/usr/bin/env bash
# Final paper campaign, one CLT per array job, one stage at a time.
#
# 5 nodes x 2 GPUs x 4 workers = 40 slots. Without a pilot cost table the
# prompts are sharded evenly. Do not launch until scripts/macag_check_eval_ready.py passes
# (clean tree, tag macag-final-v3, prompt digest).
#
# Stages follow the eval plan: graph, then Game 1, then baselines
# (ACDC reads Game 1 |E*|), then Game 2. Launch the next stage only after
# the previous one has its artifacts. PAPER_STAGE defaults to graph.
#
#   sbatch --job-name=paper_final_g426k_graph \
#     --export=ALL,CLTS=gemma2-426k,PAPER_STAGE=graph \
#     scripts/slurm/submit_macag_final.sh
#   # later: PAPER_STAGE=game1 | baselines | game2
#
#SBATCH --job-name=macag_final
#SBATCH -A uwyo-0002
#SBATCH --partition=gp-2
#SBATCH --array=0-4
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export JSON="${JSON:-macag/data/mib_benchmark_prompts.json}"
export PROMPTS_SHA256="${PROMPTS_SHA256:-}"
# One tree for the paper: paper_final/macag/<clt-tag>/<prompt-id>/
export OUTROOT="${OUTROOT:-/gscratch/${USER}/paper_final/macag}"
export CLTS="${CLTS:-llama32-524k}"
export TASKS="${TASKS:-ioi mcqa arc_easy}"
export FREEZE_MODE="${FREEZE_MODE:-both}"
export SCORE_KINDS="${SCORE_KINDS:-logit_gap}"
export SOLVERS="${SOLVERS:-abr}"
export GAME2_BETAS="${GAME2_BETAS:-0 0.2}"
export GAME2_SCORE_KIND="${GAME2_SCORE_KIND:-negative_loss}"
export GAME2_FREEZE_ATTENTION="${GAME2_FREEZE_ATTENTION:-false}"
export GAME2_SUBSET="${GAME2_SUBSET:-p3}"
export ABR_ITERS="${ABR_ITERS:-8}"
export BASELINE_METHODS="${BASELINE_METHODS:-random,influence,eap_syed,singleton,acdc}"
export BASELINE_LEGS="${BASELINE_LEGS:-both}"
export RANDOM_DRAWS="${RANDOM_DRAWS:-10}"
export ACDC_TARGET_FROM_GAME1="${ACDC_TARGET_FROM_GAME1:-1}"
export REPORT_BUDGET="${REPORT_BUDGET:-128}"
export NODE_THRESHOLD="${NODE_THRESHOLD:-0.9}"
export EDGE_THRESHOLD="${EDGE_THRESHOLD:-1.0}"
export CONNECTED="${CONNECTED:-0}"
export PREFILTER_TOP_K="${PREFILTER_TOP_K:-}"
export KL_RESCORE="${KL_RESCORE:-1}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-0}"
export MACAG_WORKERS_PER_GPU="${MACAG_WORKERS_PER_GPU:-4}"
export MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-4}"
# One stage per launch. "all" is intentionally not a choice.
export PAPER_STAGE="${PAPER_STAGE:-graph}"
case "$PAPER_STAGE" in
  graph)
    export SKIP_GAME1=1 SKIP_GAME2=1 SKIP_BASELINES=1
    ;;
  game1)
    export SKIP_GAME1=0 SKIP_GAME2=1 SKIP_BASELINES=1
    ;;
  baselines)
    export SKIP_GAME1=1 SKIP_GAME2=1 SKIP_BASELINES=0
    ;;
  game2)
    export SKIP_GAME1=1 SKIP_GAME2=0 SKIP_BASELINES=1
    ;;
  *)
    echo "ERROR: PAPER_STAGE must be graph|game1|baselines|game2 (got '$PAPER_STAGE')" >&2
    exit 2
    ;;
esac
echo ">>> PAPER_STAGE=$PAPER_STAGE skip_game1=$SKIP_GAME1 skip_game2=$SKIP_GAME2 skip_baselines=$SKIP_BASELINES"
case "$CLTS" in
  llama32-524k) export MIB_MODELS="${MIB_MODELS:-llama3}" ;;
  *) export MIB_MODELS="${MIB_MODELS:-gemma2}" ;;
esac

# 4 workers share each of 2 GPUs. The existing slot lock may serialize them
# if the node would otherwise OOM; the assignment file is still the shard.
LOCAL_WORKERS="${LOCAL_WORKERS:-8}"
export TOTAL_WORKERS="${TOTAL_WORKERS:-40}"
export NODE_RANK="${SLURM_ARRAY_TASK_ID:-0}"
case "$CLTS" in
  gemma2-426k) export WORKERS_GEMMA2_426K="$LOCAL_WORKERS" ;;
  gemma2-2.5M) export WORKERS_GEMMA2_2_5M="$LOCAL_WORKERS" ;;
  llama32-524k) export WORKERS_LLAMA32_524K="$LOCAL_WORKERS" ;;
esac

python scripts/macag_check_eval_ready.py \
  --tag "${MACAG_TAG:-macag-final-v3}" \
  --prompts "$JSON" \
  ${PROMPTS_SHA256:+--prompts-sha256 "$PROMPTS_SHA256"}

if [[ -n "${COST_TABLE:-}" ]]; then
  export ASSIGNMENT_FILE="${ASSIGNMENT_FILE:-$OUTROOT/assignment.json}"
  python scripts/macag_assign_workers.py \
    --costs "$COST_TABLE" --workers "$TOTAL_WORKERS" --output "$ASSIGNMENT_FILE"
fi

echo ">>> final campaign CLTS=$CLTS node=$NODE_RANK/$TOTAL_WORKERS out=$OUTROOT"
nvidia-smi -L || true
scripts/run_macag_mib_parallel.sh
