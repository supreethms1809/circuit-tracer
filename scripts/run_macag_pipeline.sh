#!/usr/bin/env bash
#
# End-to-end MACAG pipeline over a linear-CLT circuit-tracer graph:
#   attribute (build graph) -> game1 + game2 -> annotate -> (optional) serve
#
# Assumes the conda env is already activated (uses `python` directly).
# Run from the repo root, or it will cd there based on this script's location.
#
# Usage:
#   scripts/run_macag_pipeline.sh \
#     --prompt "Fact: The capital of the state containing Dallas is" \
#     --target " Austin" --foil " Texas"
#
# Override any default via --flag value (see DEFAULTS below). Common ones:
#   --model --transcoder-set --slug --outdir --device --dtype
#   --prefilter-top-k --budget --beta --abr-iters --alpha --lam --eps
#   --max-feature-nodes --batch-size --node-threshold --edge-threshold
#   --skip-attribute   reuse an existing graph at <outdir>/graphs/<slug>.json
#   --serve            launch the visualization server at the end
#   --port             server port (default 8041)
set -euo pipefail

# --- locate repo root (parent of this script's dir) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-.}:."

# --- DEFAULTS ---------------------------------------------------------------
PROMPT="Fact: The capital of the state containing Dallas is"
TARGET=" Austin"
FOIL=" Texas"
MODEL="google/gemma-2-2b"
TRANSCODER_SET="mntss/clt-gemma-2-2b-426k"
SLUG="dallas-austin"
OUTDIR="results/macag_demo"
DEVICE="cuda"          # cuda | cpu  (mps is unsupported: safetensors lazy-decoder)
DTYPE="bfloat16"
# graph (attribution) params
MAX_FEATURE_NODES=7500
BATCH_SIZE=256
NODE_THRESHOLD=0.8
EDGE_THRESHOLD=0.98
# game params
PREFILTER_TOP_K=20
BUDGET=8
BETA=0.2
ABR_ITERS=4
ALPHA=0.5
LAM=0.02
EPS=0.1
# candidate universe: empty = full feature-node universe (recommended on CUDA).
# Set to a path (.json list or text) to restrict, e.g. for CPU tractability.
CANDIDATES_FILE=""
SKIP_ATTRIBUTE=0
SERVE=0
PORT=8041

# --- parse --flag value -----------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2;;
    --target) TARGET="$2"; shift 2;;
    --foil) FOIL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --transcoder-set) TRANSCODER_SET="$2"; shift 2;;
    --slug) SLUG="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --max-feature-nodes) MAX_FEATURE_NODES="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --node-threshold) NODE_THRESHOLD="$2"; shift 2;;
    --edge-threshold) EDGE_THRESHOLD="$2"; shift 2;;
    --prefilter-top-k) PREFILTER_TOP_K="$2"; shift 2;;
    --budget) BUDGET="$2"; shift 2;;
    --beta) BETA="$2"; shift 2;;
    --abr-iters) ABR_ITERS="$2"; shift 2;;
    --alpha) ALPHA="$2"; shift 2;;
    --lam) LAM="$2"; shift 2;;
    --eps) EPS="$2"; shift 2;;
    --candidates-file) CANDIDATES_FILE="$2"; shift 2;;
    --skip-attribute) SKIP_ATTRIBUTE=1; shift;;
    --serve) SERVE=1; shift;;
    --port) PORT="$2"; shift 2;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

GRAPH="$OUTDIR/graphs/$SLUG.json"
KWARGS="$OUTDIR/oracle_kwargs.json"
G1_OUT="$OUTDIR/macag_game1.json"
G2_OUT="$OUTDIR/macag_game2.json"
ANNOTATED="$OUTDIR/graphs/${SLUG}_macag.json"
mkdir -p "$OUTDIR/graphs"

CAND_ARG=()
[[ -n "$CANDIDATES_FILE" ]] && CAND_ARG=(--candidates-file "$CANDIDATES_FILE")

echo ">>> MACAG pipeline | model=$MODEL clt=$TRANSCODER_SET device=$DEVICE"
echo ">>> prompt: $PROMPT"
echo ">>> target=$TARGET  foil=$FOIL  -> $OUTDIR"

# --- 1) attribution graph ---------------------------------------------------
if [[ "$SKIP_ATTRIBUTE" -eq 0 ]]; then
  echo ">>> [1/5] Building attribution graph ..."
  python -m circuit_tracer attribute \
    -m "$MODEL" \
    -t "$TRANSCODER_SET" \
    -p "$PROMPT" \
    --dtype "$DTYPE" \
    --batch_size "$BATCH_SIZE" \
    --max_feature_nodes "$MAX_FEATURE_NODES" \
    --slug "$SLUG" \
    --graph_file_dir "$OUTDIR/graphs" \
    --node_threshold "$NODE_THRESHOLD" --edge_threshold "$EDGE_THRESHOLD" \
    --verbose
else
  echo ">>> [1/5] Skipping attribution; reusing $GRAPH"
  [[ -f "$GRAPH" ]] || { echo "ERROR: $GRAPH not found"; exit 1; }
fi

# --- 2) oracle kwargs (JSON written safely via python) ----------------------
echo ">>> [2/5] Writing oracle kwargs -> $KWARGS"
PROMPT="$PROMPT" TARGET="$TARGET" FOIL="$FOIL" MODEL="$MODEL" \
TRANSCODER_SET="$TRANSCODER_SET" GRAPH="$GRAPH" DEVICE="$DEVICE" DTYPE="$DTYPE" \
KWARGS="$KWARGS" python - <<'PY'
import json, os
kw = {
    "model_name": os.environ["MODEL"],
    "transcoder_set": os.environ["TRANSCODER_SET"],
    "prompt": os.environ["PROMPT"],
    "graph_json": os.environ["GRAPH"],
    "backend": "transformerlens",
    "score_kind": "logit_gap",
    "strict_single_token": False,
    "freeze_attention": True,
    "target_token_by_label": {"y": os.environ["TARGET"], "y_foil": os.environ["FOIL"]},
    "foil_by_target": {"y": "y_foil", "y_foil": "y"},
    "model_kwargs": {"dtype": os.environ["DTYPE"], "device": os.environ["DEVICE"]},
}
with open(os.environ["KWARGS"], "w") as f:
    json.dump(kw, f, indent=2)
print("wrote", os.environ["KWARGS"])
PY

ORACLE_FACTORY="macag.factories.replacement_model:create_replacement_model_oracle"

# --- 3) Game 1: minimal faithful set ---------------------------------------
echo ">>> [3/5] Game 1 (minimal faithful set for target) ..."
python -m macag.cli.run_macag game1 \
  --graph-json "$GRAPH" --target y \
  --oracle-factory "$ORACLE_FACTORY" \
  --oracle-kwargs-file "$KWARGS" \
  "${CAND_ARG[@]}" \
  --prefilter-top-k "$PREFILTER_TOP_K" --budget "$BUDGET" \
  --alpha "$ALPHA" --lam "$LAM" --faithfulness-eps "$EPS" \
  --output-json "$G1_OUT"

# --- 4) Game 2: contrastive allocation -------------------------------------
echo ">>> [4/5] Game 2 (contrastive target vs foil) ..."
python -m macag.cli.run_macag game2 \
  --graph-json "$GRAPH" --target y --foil y_foil \
  --oracle-factory "$ORACLE_FACTORY" \
  --oracle-kwargs-file "$KWARGS" \
  "${CAND_ARG[@]}" \
  --prefilter-top-k "$PREFILTER_TOP_K" --budget "$BUDGET" \
  --beta "$BETA" --abr-iters "$ABR_ITERS" \
  --alpha "$ALPHA" --lam "$LAM" \
  --output-json "$G2_OUT"

# --- 5) annotate graph with both games' evidence ---------------------------
echo ">>> [5/5] Annotating graph -> $ANNOTATED"
python -m macag.cli.annotate_graph \
  --graph-json "$GRAPH" \
  --macag-result-json "$G2_OUT" --label-prefix "MACAG:g2" \
  --output-json "$ANNOTATED"
python -m macag.cli.annotate_graph \
  --graph-json "$ANNOTATED" \
  --macag-result-json "$G1_OUT" --label-prefix "MACAG:g1" \
  --output-json "$ANNOTATED"

echo ">>> Done."
echo "    graph     : $GRAPH"
echo "    game1     : $G1_OUT"
echo "    game2     : $G2_OUT"
echo "    annotated : $ANNOTATED"

# --- optional: serve --------------------------------------------------------
if [[ "$SERVE" -eq 1 ]]; then
  echo ">>> Serving $OUTDIR/graphs on port $PORT (Ctrl-C to stop) ..."
  python -m circuit_tracer start-server --graph_file_dir "$OUTDIR/graphs" --port "$PORT"
else
  echo ">>> To visualize:"
  echo "    python -m circuit_tracer start-server --graph_file_dir $OUTDIR/graphs --port $PORT"
fi
