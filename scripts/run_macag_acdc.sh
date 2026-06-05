#!/usr/bin/env bash
# Run the full MACAG pipeline (attribute -> game1 -> game2 -> annotate) over the
# ACDC benchmark prompts (macag/data/acdc_benchmark_prompts.json) for all three
# runnable CLTs. These are NEW prompts, so each needs a fresh attribution graph.
#
# Usage:
#   scripts/run_macag_acdc.sh                # all tasks, all 3 CLTs
#   TASKS="indirect_object_identification" scripts/run_macag_acdc.sh   # one task
#   LIMIT=1 scripts/run_macag_acdc.sh        # first N prompts per task (smoke)
#
# Output: results/macag_acdc/<clt_tag>/<slug>/  (game1, game2, graph, annotated)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JSON="${JSON:-macag/data/acdc_benchmark_prompts.json}"
OUTROOT="${OUTROOT:-results/macag_acdc}"
DEVICE="${DEVICE:-cuda}"
TASKS="${TASKS:-}"     # space-separated task filter; empty = all tasks
LIMIT="${LIMIT:-0}"    # >0 = first N prompts per task (for smoke tests)
CLTS="${CLTS:-}"       # space-separated clt-tag filter; empty = all 3 CLTs
# greater_than is excluded by default: its target/foil (e.g. "42"/"40") share a
# first token, so first-token logit_gap scoring gives a degenerate gap of 0.
SKIP_TASKS="${SKIP_TASKS:-greater_than}"

# clt_tag  model  transcoder_set
CLT_TAGS=(gemma2-426k gemma2-2.5M llama32-524k)
CLT_MODEL=(google/gemma-2-2b google/gemma-2-2b meta-llama/Llama-3.2-1B)
CLT_TSET=(mntss/clt-gemma-2-2b-426k mntss/clt-gemma-2-2b-2.5M mntss/clt-llama-3.2-1b-524k)

mkdir -p "$OUTROOT"
STATUS="$OUTROOT/status.txt"; : > "$STATUS"
echo ">>> MACAG ACDC sweep | json=$JSON tasks=${TASKS:-ALL} limit=$LIMIT device=$DEVICE"

# Emit "task<TAB>idx" per prompt. Only task+idx are line-safe; the prompt itself
# may contain newlines (docstring task), so it is fetched separately by index.
emit_prompts() {
  JSON="$JSON" TASKS="$TASKS" LIMIT="$LIMIT" SKIP_TASKS="$SKIP_TASKS" python - <<'PY'
import json, os
d = json.load(open(os.environ["JSON"]))
want = set(os.environ.get("TASKS", "").split()) or None
skip = set(os.environ.get("SKIP_TASKS", "").split())
limit = int(os.environ.get("LIMIT", "0"))
for task, items in d["tasks"].items():
    if (want and task not in want) or task in skip:
        continue
    for i in range(len(items)):
        if limit and i >= limit:
            break
        print(f"{task}\t{i}")
PY
}

# get_field <task> <idx> <field> -> raw value (newline-safe; no trailing newline).
get_field() {
  JSON="$JSON" python - "$1" "$2" "$3" <<'PY'
import json, os, sys
d = json.load(open(os.environ["JSON"]))
task, idx, field = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.stdout.write(d["tasks"][task][idx][field])
PY
}

for ci in "${!CLT_TAGS[@]}"; do
  tag="${CLT_TAGS[$ci]}"; model="${CLT_MODEL[$ci]}"; tset="${CLT_TSET[$ci]}"
  if [[ -n "$CLTS" && " $CLTS " != *" $tag "* ]]; then continue; fi
  echo ""; echo ">>> ===== CLT $tag ($model | $tset) ====="
  while IFS=$'\t' read -r task idx; do
    [[ -z "$task" ]] && continue
    slug=$(get_field "$task" "$idx" id)
    prompt=$(get_field "$task" "$idx" clean_prompt)
    target=$(get_field "$task" "$idx" correct_token)
    foil=$(get_field "$task" "$idx" incorrect_token)
    out="$OUTROOT/$tag/$slug"
    if [[ -f "$out/macag_game2.json" ]]; then
      echo "SKIP-done $tag/$slug" | tee -a "$STATUS"; continue
    fi
    echo ">>> [$tag/$slug] ($task) target=$target foil=$foil"
    if scripts/run_macag_pipeline.sh \
          --prompt "$prompt" --target "$target" --foil "$foil" \
          --model "$model" --transcoder-set "$tset" \
          --slug "$slug" --outdir "$out" --device "$DEVICE" \
          > "$OUTROOT/$tag-$slug.log" 2>&1; then
      echo "OK   $tag/$slug ($task)" | tee -a "$STATUS"
    else
      echo "FAIL $tag/$slug ($task)  (see $OUTROOT/$tag-$slug.log)" | tee -a "$STATUS"
    fi
  done < <(emit_prompts)
done

echo ""; echo ">>> DONE. Status summary:"
sort "$STATUS" | uniq -c | sort -rn | head
