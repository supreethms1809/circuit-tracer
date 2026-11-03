#!/usr/bin/env bash
# Chain 24h game2 waves (afterany) until NUM_WAVES follow-ups are queued.
# Does not submit the first wave — pass the running or last wave job id.
#
#   PREV_JOB=49302 CLTS=gemma2-426k WAVE_START=9 NUM_WAVES=4 \
#     bash scripts/slurm/submit_macag_final_game2_chain.sh
#
set -euo pipefail

PREV_JOB="${PREV_JOB:?set PREV_JOB to last array job id (e.g. 49302)}"
CLTS="${CLTS:?set CLTS}"
WAVE_START="${WAVE_START:-9}"
NUM_WAVES="${NUM_WAVES:-4}"
PAPER_STAGE=game2
SCRIPT="${SCRIPT:-scripts/slurm/submit_macag_final_allprompts.sh}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
case "$CLTS" in
  gemma2-426k)  PREFIX=g426k ;;
  gemma2-2.5M)  PREFIX=g25m ;;
  llama32-524k) PREFIX=llama ;;
  *) echo "unknown CLTS=$CLTS" >&2; exit 2 ;;
esac

prev="$PREV_JOB"
for ((i = 0; i < NUM_WAVES; i++)); do
  r=$((WAVE_START + i))
  name="paper_final_${PREFIX}_game2_r${r}"
  jid=$(sbatch --parsable \
    --dependency="afterany:${prev}" \
    --job-name="$name" \
    --export=ALL,CLTS="$CLTS",PAPER_STAGE="$PAPER_STAGE" \
    "$SCRIPT")
  echo "submitted $name job=$jid afterany:$prev"
  prev="$jid"
done
echo ">>> chain tail job id: $prev"
