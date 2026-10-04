#!/usr/bin/env bash
# Dallas–Austin Game 1 ablation (same graph; does not overwrite job 13913).
#
# Env:
#   SCORE_KIND          kl_divergence | logit_gap (default kl_divergence)
#   PREFILTER_TOP_K     int, or empty/0/off = no prefilter
#   ALPHA               0.5 default; 0 = pure N; 1 = pure S
#   EPS                 faithfulness_eps (default 0.1)
#   FORCE=1             overwrite existing output
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}:."

SLUG="${SLUG:-dallas-austin}"
CLT_TAG="${CLT_TAG:-llama32-524k}"
OUTDIR="${OUTDIR:-/gscratch/${USER}/macag_dallas_austin_llama/${CLT_TAG}/${SLUG}}"
SCORE_KIND="${SCORE_KIND:-kl_divergence}"
KIND_DIR="$OUTDIR/$SCORE_KIND"
GRAPH="$OUTDIR/graphs/$SLUG.json"
KWARGS="$KIND_DIR/oracle_kwargs.json"
ORACLE_FACTORY="macag.factories.replacement_model:create_replacement_model_oracle"

PREFILTER_TOP_K="${PREFILTER_TOP_K:-}"
ALPHA="${ALPHA:-0.5}"
LAM="${LAM:-0.02}"
EPS="${EPS:-0.1}"
FREEZE_MODE="${FREEZE_MODE:-both}"
CONNECTED="${CONNECTED:-0}"
CAP_SUFFICIENCY="${CAP_SUFFICIENCY:-0}"

mkdir -p "$KIND_DIR" "$OUTDIR/logs"

if [[ ! -f "$GRAPH" ]]; then
  echo "ERROR: missing graph $GRAPH" >&2
  exit 2
fi

# Seed logit_gap kwargs from the KL setup if needed (same prompt/tokens/graph).
if [[ ! -f "$KWARGS" ]]; then
  SRC="$OUTDIR/kl_divergence/oracle_kwargs.json"
  [[ -f "$SRC" ]] || { echo "ERROR: no kwargs at $KWARGS or $SRC" >&2; exit 2; }
  SCORE_KIND="$SCORE_KIND" SRC="$SRC" DST="$KWARGS" python - <<'PY'
import json, os
from pathlib import Path
src, dst, kind = os.environ["SRC"], os.environ["DST"], os.environ["SCORE_KIND"]
kw = json.loads(Path(src).read_text())
kw["score_kind"] = kind
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(json.dumps(kw, indent=2) + "\n")
print(f">>> wrote {dst} score_kind={kind}")
PY
fi

use_prefilter=0
if [[ -n "${PREFILTER_TOP_K}" && "${PREFILTER_TOP_K}" != "0" && "${PREFILTER_TOP_K}" != "off" && "${PREFILTER_TOP_K}" != "none" ]]; then
  use_prefilter=1
fi

# Filename tags so ablations never clobber each other.
STEM="macag_game1"
if (( use_prefilter )); then
  if [[ "$EPS" == "0.1" || "$EPS" == "0.10" ]]; then
    STEM="macag_game1_prefilter${PREFILTER_TOP_K}"
  else
    STEM="macag_game1_prefilter${PREFILTER_TOP_K}_eps${EPS}"
  fi
fi
if [[ "$ALPHA" != "0.5" && "$ALPHA" != "0.50" ]]; then
  STEM="${STEM}_alpha${ALPHA}"
fi
cap_args=()
if [[ "$CAP_SUFFICIENCY" == "1" || "$CAP_SUFFICIENCY" == "true" ]]; then
  STEM="${STEM}_capsuff"
  cap_args=(--cap-sufficiency)
fi
OUT="$KIND_DIR/${STEM}.json"
ID="${SLUG}-${STEM}"

if [[ -f "$OUT" && "${FORCE:-0}" != "1" ]]; then
  echo ">>> skip (exists $OUT); set FORCE=1 to overwrite"
  exit 0
fi

connected_args=(--no-connected)
if [[ "$CONNECTED" == "1" || "$CONNECTED" == "true" ]]; then
  connected_args=(--connected)
fi
prefilter_args=()
if (( use_prefilter )); then
  prefilter_args=(--prefilter-top-k "$PREFILTER_TOP_K")
fi

echo ">>> Dallas Game1 ablation"
echo ">>> graph=$GRAPH kwargs=$KWARGS"
echo ">>> score_kind=$SCORE_KIND alpha=$ALPHA eps=$EPS prefilter=${PREFILTER_TOP_K:-off} freeze=$FREEZE_MODE cap_sufficiency=$CAP_SUFFICIENCY"
echo ">>> out=$OUT"

START=$(date +%s)
python -m macag.cli.run_macag game1 \
  --graph-json "$GRAPH" --target y --input-id "$ID" \
  --oracle-factory "$ORACLE_FACTORY" \
  --oracle-kwargs-file "$KWARGS" \
  "${connected_args[@]}" \
  "${prefilter_args[@]}" \
  --alpha "$ALPHA" --lam "$LAM" \
  --faithfulness-eps "$EPS" \
  --freeze-mode "$FREEZE_MODE" \
  "${cap_args[@]}" \
  --checkpoint-json "${OUT%.json}.ckpt.json" \
  --output-json "$OUT"
END=$(date +%s)
ELAPSED=$((END - START))

python - "$OUT" "$ELAPSED" "$ALPHA" "${PREFILTER_TOP_K:-}" "$EPS" "$SCORE_KIND" <<'PY'
import json, sys
from pathlib import Path
out_path = Path(sys.argv[1])
elapsed, alpha, k, eps, kind = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
pre = json.loads(out_path.read_text())
summary = {
    "ablation": "game1",
    "score_kind": kind,
    "alpha": float(alpha),
    "prefilter_top_k": None if k in ("", "off", "none") else int(k),
    "faithfulness_eps": float(eps),
    "wall_seconds": int(elapsed),
    "output": str(out_path),
    "legs": {},
}
for leg in ("frozen", "unfrozen"):
    if leg not in pre:
        continue
    s, st, e = pre[leg]["scores"], pre[leg]["stats"], pre[leg]["evidence"]
    row = {
        "size": len(e.get("E_star") or []),
        "F": s.get("faithfulness"),
        "S": s.get("sufficiency"),
        "N": s.get("necessity"),
        "all": s.get("all"),
        "empty": s.get("empty"),
        "keep_only": s.get("keep_only"),
        "remove": s.get("remove"),
        "sufficiency_uncapped": s.get("sufficiency_uncapped"),
        "sufficiency_capped": s.get("sufficiency_capped"),
        "keep_distance": s.get("keep_distance"),
        "oracle_calls": st.get("oracle_calls"),
        "total_candidates": st.get("total_candidates"),
        "prefiltered_candidate_count": st.get("prefiltered_candidate_count"),
        "E_star": e.get("E_star"),
    }
    summary["legs"][leg] = row
cmp = out_path.with_name(out_path.stem + "_compare.json")
cmp.write_text(json.dumps(summary, indent=2) + "\n")
print(">>> summary ->", cmp)
print(json.dumps(summary, indent=2))
PY

echo ">>> Game1 complete in ${ELAPSED}s -> $OUT"
