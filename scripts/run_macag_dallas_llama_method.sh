#!/usr/bin/env bash
# Run one mutually-exclusive dallas-austin / Llama method shard.
# Waits for setup READY_FLAG, then runs exactly METHOD (no cross-talk).
#
#   METHOD=game1 scripts/run_macag_dallas_llama_method.sh
#   METHOD=eap_edge scripts/run_macag_dallas_llama_method.sh
#
# Known METHOD values:
#   game1 | game2 | influence | eap_syed | acdc | shapley
#   eap_edge | acdc_edge | finalize
#
# Env knobs (defaults match the KL / no-budget Game 1 dallas plan):
#   OUTDIR SCORE_KIND SLUG REPORT_BUDGET ALPHA LAM EPS
#   SHAPLEY_PERMUTATIONS SHAPLEY_SEED ACDC_TAUS ACDC_TARGET_K
#   ACDC_EDGE_TAUS ACDC_EDGE_TARGET_K EDGE_BUDGET
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-.}:."

METHOD="${METHOD:?set METHOD=game1|game2|influence|eap_syed|acdc|shapley|eap_edge|acdc_edge|finalize}"
SLUG="${SLUG:-dallas-austin}"
CLT_TAG="${CLT_TAG:-llama32-524k}"
OUTDIR="${OUTDIR:-/gscratch/${USER}/macag_dallas_austin_llama/${CLT_TAG}/${SLUG}}"
SCORE_KIND="${SCORE_KIND:-kl_divergence}"
KIND_DIR="$OUTDIR/$SCORE_KIND"
GRAPH="$OUTDIR/graphs/$SLUG.json"
KWARGS="$KIND_DIR/oracle_kwargs.json"
READY_FLAG="$OUTDIR/.setup_ready"
ORACLE_FACTORY="macag.factories.replacement_model:create_replacement_model_oracle"

# Game 1: omit budget (None). Do NOT pass --budget 0 — that stops with |E|=0.
# Ranking baselines need a finite budget for prefix curves / top-k.
REPORT_BUDGET="${REPORT_BUDGET:-128}"
ALPHA="${ALPHA:-0.5}"
LAM="${LAM:-0.02}"
EPS="${EPS:-0.1}"
BETA="${BETA:-0.2}"
ABR_ITERS="${ABR_ITERS:-4}"
FP_TOL="${FP_TOL:-1e-3}"
FREEZE_MODE="${FREEZE_MODE:-both}"
CONNECTED="${CONNECTED:-0}"
SHAPLEY_PERMUTATIONS="${SHAPLEY_PERMUTATIONS:-64}"
SHAPLEY_SEED="${SHAPLEY_SEED:-0}"
ACDC_TAUS="${ACDC_TAUS:-0.001,0.01,0.05,0.1,0.2,0.5}"
# Empty / unset / 0 => native feature ACDC τ-sweep (no hard size match).
ACDC_TARGET_K="${ACDC_TARGET_K:-}"
# Original edge track (native AutoCircuit).
EDGE_BUDGET="${EDGE_BUDGET:-0}"
ACDC_EDGE_TAUS="${ACDC_EDGE_TAUS:-0.1}"
ACDC_EDGE_METRIC="${ACDC_EDGE_METRIC:-kl}"
ACDC_EDGE_TARGET_K="${ACDC_EDGE_TARGET_K:-0}"
EDGE_SNF="${EDGE_SNF:-1}"
SETUP_WAIT_SECS="${SETUP_WAIT_SECS:-7200}"
PREFILTER_TOP_K="${PREFILTER_TOP_K:-}"
GAME2_SOLVERS="${GAME2_SOLVERS:-abr fp}"

mkdir -p "$KIND_DIR" "$OUTDIR/logs"

wait_for_setup() {
  local waited=0
  while [[ ! -f "$READY_FLAG" || ! -f "$GRAPH" || ! -f "$KWARGS" ]]; do
    if (( waited >= SETUP_WAIT_SECS )); then
      echo "ERROR: timed out waiting for setup ($READY_FLAG / graph / kwargs)" >&2
      exit 2
    fi
    echo ">>> METHOD=$METHOD waiting for setup ... (${waited}s)"
    sleep 30
    waited=$((waited + 30))
  done
  echo ">>> METHOD=$METHOD setup ready -> $OUTDIR"
}

connected_args=()
if [[ "$CONNECTED" == "1" || "$CONNECTED" == "true" ]]; then
  connected_args=(--connected)
else
  connected_args=(--no-connected)
fi

run_feature_baseline() {
  local method="$1"
  local tag=""
  if [[ -n "${OUTPUT_TAG:-}" ]]; then
    tag="_${OUTPUT_TAG}"
  elif [[ -n "${PREFILTER_TOP_K}" && "${PREFILTER_TOP_K}" != "0" && "${PREFILTER_TOP_K}" != "off" ]]; then
    tag="_prefilter${PREFILTER_TOP_K}"
  fi
  local out="$KIND_DIR/macag_baselines_${method}${tag}.json"
  if [[ -f "$out" ]]; then
    echo ">>> skip $method (exists $out)"
    return 0
  fi
  local cmd=(
    python -m macag.cli.run_baselines
    --graph-json "$GRAPH" --target y --input-id "$SLUG"
    --oracle-factory "$ORACLE_FACTORY"
    --oracle-kwargs-file "$KWARGS"
    --budget "$REPORT_BUDGET" --alpha "$ALPHA" --lam "$LAM"
    --methods "$method"
    --shapley-permutations "$SHAPLEY_PERMUTATIONS"
    --shapley-seed "$SHAPLEY_SEED"
    --output-json "$out"
  )
  if [[ "${CAP_SUFFICIENCY:-0}" == "1" || "${CAP_SUFFICIENCY:-0}" == "true" ]]; then
    cmd+=(--cap-sufficiency)
  fi
  if [[ -n "${CANDIDATES_FILE:-}" ]]; then
    cmd+=(--candidates-file "$CANDIDATES_FILE")
  fi
  ((${#connected_args[@]})) && cmd+=("${connected_args[@]}")
  if [[ "$method" == "acdc" ]]; then
    cmd+=(--acdc-taus "$ACDC_TAUS")
    # Omit --acdc-target-k for native τ-sweep (None). Do not pass 0 (empty-circuit
    # match). Use -1 to hard-match REPORT_BUDGET, or a positive k for an explicit cap.
    if [[ -n "${ACDC_TARGET_K}" && "${ACDC_TARGET_K}" != "0" ]]; then
      cmd+=(--acdc-target-k "$ACDC_TARGET_K")
    fi
  fi
  if [[ "$method" == "eap_syed" ]]; then
    # Prefer kwargs corrupted_prompt; CLI flag as belt-and-suspenders.
    local corrupt
    corrupt=$(python - "$KWARGS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("corrupted_prompt") or "", end="")
PY
)
    [[ -n "$corrupt" ]] && cmd+=(--eap-corrupted-prompt "$corrupt")
  fi
  echo ">>> feature baseline $method -> $out (report_budget=$REPORT_BUDGET)"
  "${cmd[@]}"
}

run_edge_snf() {
  local method="$1"
  local orig="$KIND_DIR/macag_original_baselines_${method}.json"
  local out="$KIND_DIR/macag_original_edge_logit_gap_faithfulness_${method}.json"
  if [[ "$EDGE_SNF" != "1" && "$EDGE_SNF" != "true" ]]; then
    return 0
  fi
  if [[ -f "$out" ]]; then
    echo ">>> skip edge S/N/F for $method (exists $out)"
    return 0
  fi
  [[ -f "$orig" ]] || { echo "ERROR: missing $orig for S/N/F" >&2; return 1; }
  echo ">>> edge logit-gap S/N/F for $method -> $out"
  RUN_DIR="$KIND_DIR" ORIG_JSON="$orig" OUT="$out" METHOD="$method" ALPHA="$ALPHA" \
  python - <<'PY'
from __future__ import annotations
import json, os
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from macag.baselines.original.autocircuit_bridge import (
    build_prompt_loader,
    logit_gap_faithfulness_for_edges,
    make_patchable_model,
)
from macag.baselines.original.metrics import resolve_token_index
from macag.baselines.original.tl_compat import enable_autocircuit_tl3_compat

enable_autocircuit_tl3_compat()
run_dir = Path(os.environ["RUN_DIR"])
orig = json.loads(Path(os.environ["ORIG_JSON"]).read_text())
kwargs = json.loads((run_dir / "oracle_kwargs.json").read_text())
method = os.environ["METHOD"]
alpha = float(os.environ.get("ALPHA", "0.5"))
entry = (orig.get("methods") or {}).get(method) or {}

# Collect named circuits to score.
circuits: dict[str, list[str]] = {}
kept = list(entry.get("edges_kept") or [])
ranking = list(entry.get("ranking") or [])
sweep = list(entry.get("sweep") or [])
if kept:
    circuits[f"{method}_final"] = kept
elif method == "eap_edge" and ranking:
    for k in (8, 64, 256, 1024, 4096):
        if k <= len(ranking):
            circuits[f"eap_edge_k{k}"] = ranking[:k]
    if sweep:
        last = sweep[-1]
        circuits[f"eap_edge_k{int(last['size'])}"] = list(last.get("edges_kept") or [])
elif sweep:
    last = sweep[-1]
    circuits[f"{method}_k{int(last['size'])}"] = list(last.get("edges_kept") or [])
if not circuits:
    raise SystemExit(f"no edges to score for {method}")

clean = str(kwargs["prompt"])
corrupt = str(kwargs["corrupted_prompt"])
model_name = str(kwargs["model_name"])
token_by_label = kwargs.get("target_token_by_label") or {}
target_map = kwargs.get("target_to_logit_idx") or {}
foil_by = kwargs.get("foil_by_target") or {}
target_token = str(token_by_label["y"])
foil_label = foil_by.get("y")
foil_token = str(token_by_label[foil_label]) if foil_label in token_by_label else None
target_idx = int(target_map["y"]) if "y" in target_map else None
foil_idx = int(target_map[foil_label]) if foil_label in target_map else None

mk = dict(kwargs.get("model_kwargs") or {})
mk.pop("torch_dtype", None)
mk["dtype"] = torch.float32  # auto-circuit requires float32
print(f">>> loading {model_name} for edge S/N/F", flush=True)
tl_model = HookedTransformer.from_pretrained(model_name, **mk)
tl_model.cfg.use_attn_result = True
if target_idx is None:
    target_idx = resolve_token_index(tl_model, target_token)
if foil_token is not None and foil_idx is None:
    foil_idx = resolve_token_index(tl_model, foil_token)
loader, _ = build_prompt_loader(
    tl_model,
    clean_prompt=clean,
    corrupted_prompt=corrupt,
    target_idx=target_idx,
    foil_idx=foil_idx,
)
patch_model = make_patchable_model(tl_model)
clean_tokens = tl_model.to_tokens(clean)

payload = {
    "method": method,
    "alpha": alpha,
    "metric": "logit_gap",
    "circuits": {},
}
for name, edges in circuits.items():
    print(f">>> scoring {name} size={len(edges)}", flush=True)
    block = logit_gap_faithfulness_for_edges(
        patch_model,
        loader,
        edges,
        target_idx=target_idx,
        foil_idx=foil_idx,
        alpha=alpha,
        clean_tokens=clean_tokens,
    )
    block["edges_kept"] = edges
    payload["circuits"][name] = block
    sc = block["scores"]
    print(
        f"  S={sc['sufficiency']:.4f} N={sc['necessity']:.4f} F={sc['faithfulness']:.4f}",
        flush=True,
    )

Path(os.environ["OUT"]).write_text(json.dumps(payload, indent=2) + "\n")
print("wrote", os.environ["OUT"], flush=True)
PY
}

wait_for_setup

case "$METHOD" in
  game1)
    out="$KIND_DIR/macag_game1.json"
    if [[ -f "$out" ]]; then
      echo ">>> skip game1 (exists $out)"
      exit 0
    fi
    echo ">>> Game 1 dual-freeze | score_kind=$SCORE_KIND | NO budget | eps=$EPS"
    # Intentionally omit --budget so solver uses budget=None (unbounded).
    python -m macag.cli.run_macag game1 \
      --graph-json "$GRAPH" --target y --input-id "$SLUG" \
      --oracle-factory "$ORACLE_FACTORY" \
      --oracle-kwargs-file "$KWARGS" \
      "${connected_args[@]}" \
      --alpha "$ALPHA" --lam "$LAM" \
      --faithfulness-eps "$EPS" \
      --freeze-mode "$FREEZE_MODE" \
      --output-json "$out"
    ln -sfn "$SCORE_KIND/macag_game1.json" "$OUTDIR/macag_game1.json"
    ;;

  game2)
    use_prefilter=0
    pf_args=()
    pf_tag=""
    if [[ -n "${PREFILTER_TOP_K}" && "${PREFILTER_TOP_K}" != "0" && "${PREFILTER_TOP_K}" != "off" && "${PREFILTER_TOP_K}" != "none" ]]; then
      use_prefilter=1
      pf_args=(--prefilter-top-k "$PREFILTER_TOP_K")
      pf_tag="_prefilter${PREFILTER_TOP_K}"
    fi
    alpha_tag=""
    if [[ "$ALPHA" != "0.5" && "$ALPHA" != "0.50" ]]; then
      alpha_tag="_alpha${ALPHA}"
    fi
    beta_tag=""
    if [[ "$BETA" != "0.2" && "$BETA" != "0.20" ]]; then
      beta_tag="_beta${BETA}"
    fi
    read -r -a SOLVER_LIST <<< "$GAME2_SOLVERS"
    for solver in "${SOLVER_LIST[@]}"; do
      out="$KIND_DIR/macag_game2_${solver}${pf_tag}${alpha_tag}${beta_tag}.json"
      if [[ -f "$out" ]]; then
        echo ">>> skip game2/$solver (exists $out)"
        continue
      fi
      ckpt="$KIND_DIR/macag_game2_${solver}${pf_tag}${alpha_tag}${beta_tag}.ckpt.json"
      echo ">>> Game 2 solver=$solver | score_kind=$SCORE_KIND | alpha=$ALPHA | NO budget | prefilter=${PREFILTER_TOP_K:-off} | ckpt=$ckpt"
      # Full-universe logs/ckpts must not seed a prefiltered run (different pool).
      if (( ! use_prefilter )); then
        python - "$ckpt" "$solver" "$OUTDIR/logs" <<'PY'
import sys
from pathlib import Path
from macag.games.game2_contrastive import checkpoint_from_abr_logs, _atomic_write_json
ckpt, solver, logdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
if solver != "abr":
    raise SystemExit(0)
logs = sorted(logdir.glob("method.game2*.out"))
harvested = checkpoint_from_abr_logs(logs, solver="abr")
if harvested is None:
    print(">>> no ABR log harvest")
    raise SystemExit(0)
dest = Path(ckpt)
if dest.is_file():
    import json
    old = json.loads(dest.read_text())
    old_n = len(old.get("next_y") or []) + len(old.get("next_foil") or [])
    new_n = len(harvested.get("next_y") or []) + len(harvested.get("next_foil") or [])
    if new_n <= old_n:
        print(f">>> keep existing ckpt (|y|+|foil|={old_n} >= harvested {new_n})")
        raise SystemExit(0)
_atomic_write_json(dest, harvested)
print(
    f">>> harvested ABR ckpt iter={harvested['iteration']} phase={harvested['phase']} "
    f"|y|={len(harvested['next_y'])} |foil|={len(harvested['next_foil'])} -> {dest}"
)
PY
      fi
      python -m macag.cli.run_macag game2 \
        --graph-json "$GRAPH" --target y --foil y_foil --input-id "$SLUG" \
        --oracle-factory "$ORACLE_FACTORY" \
        --oracle-kwargs-file "$KWARGS" \
        "${connected_args[@]}" \
        "${pf_args[@]}" \
        --beta "$BETA" --abr-iters "$ABR_ITERS" \
        --solver "$solver" --fp-tol "$FP_TOL" \
        --alpha "$ALPHA" --lam "$LAM" \
        --checkpoint-json "$ckpt" \
        --output-json "$out"
    done
    if [[ -z "$pf_tag" && -z "$alpha_tag" && -z "$beta_tag" && -f "$KIND_DIR/macag_game2_abr.json" ]]; then
      cp -f "$KIND_DIR/macag_game2_abr.json" "$KIND_DIR/macag_game2.json"
      ln -sfn "$SCORE_KIND/macag_game2.json" "$OUTDIR/macag_game2.json"
    fi
    ;;

  influence|eap_syed|acdc|shapley)
    run_feature_baseline "$METHOD"
    ;;

  eap_edge|acdc_edge)
    out="$KIND_DIR/macag_original_baselines_${METHOD}.json"
    if [[ -f "$out" ]]; then
      echo ">>> skip $METHOD (exists $out)"
    else
      echo ">>> original $METHOD -> $out"
      python -m macag.cli.run_original_baselines \
        --oracle-kwargs-file "$KWARGS" \
        --input-id "$SLUG" \
        --budget "$EDGE_BUDGET" \
        --methods "$METHOD" \
        --acdc-taus "$ACDC_EDGE_TAUS" \
        --acdc-metric "$ACDC_EDGE_METRIC" \
        --acdc-target-k "$ACDC_EDGE_TARGET_K" \
        --output-json "$out"
    fi
    run_edge_snf "$METHOD"
    ;;

  finalize)
    # Merge feature sidecars -> macag_baselines.json; merge edge sidecars;
    # optional KL rescore is a no-op check when selection already used KL.
    echo ">>> finalize merges under $KIND_DIR"
    feature_methods=(influence eap_syed acdc shapley)
    main=""
    for m in "${feature_methods[@]}"; do
      side="$KIND_DIR/macag_baselines_${m}.json"
      [[ -f "$side" ]] || { echo "WARN: missing feature sidecar $side"; continue; }
      if [[ -z "$main" ]]; then
        cp -f "$side" "$KIND_DIR/macag_baselines.json"
        main="$KIND_DIR/macag_baselines.json"
        echo ">>> seeded macag_baselines.json from $m"
      else
        python -m macag.cli.merge_baselines --main "$main" --extra "$side" --progress
      fi
    done
    if [[ -n "$main" ]]; then
      ln -sfn "$SCORE_KIND/macag_baselines.json" "$OUTDIR/macag_baselines.json"
    fi

    # Merge original edge sidecars (simple dict union; no merge_baselines schema).
    python - "$KIND_DIR" <<'PY'
import json, sys
from pathlib import Path
kind = Path(sys.argv[1])
blocks = {}
meta = None
for method in ("eap_edge", "acdc_edge"):
    p = kind / f"macag_original_baselines_{method}.json"
    if not p.is_file():
        print(f"WARN: missing {p}")
        continue
    d = json.loads(p.read_text())
    if meta is None:
        meta = {k: d.get(k) for k in (
            "schema_version", "track", "game", "input_id", "experiment_identity", "comparison"
        )}
    blocks.update(d.get("methods") or {})
out = kind / "macag_original_baselines.json"
if blocks:
    payload = dict(meta or {})
    payload["methods"] = blocks
    # Prefer combined S/N/F if both exist.
    snf = {}
    for method in ("eap_edge", "acdc_edge"):
        sp = kind / f"macag_original_edge_logit_gap_faithfulness_{method}.json"
        if sp.is_file():
            snf[method] = json.loads(sp.read_text())
    if snf:
        payload["logit_gap_faithfulness"] = snf
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out)
else:
    print("no original edge sidecars to merge")
PY
    ln -sfn "$SCORE_KIND/macag_original_baselines.json" "$OUTDIR/macag_original_baselines.json" 2>/dev/null || true

    if [[ "$SCORE_KIND" != "kl_divergence" ]]; then
      echo ">>> KL rescore (selection was not KL)"
      python -m macag.cli.rescore_kl --run-dir "$KIND_DIR" --progress || \
        echo ">>> KL rescore failed; selection artifacts unchanged"
    else
      echo ">>> selection already kl_divergence; skipping post-hoc KL rescore"
    fi
    echo ">>> finalize done"
    ;;

  *)
    echo "ERROR: unknown METHOD=$METHOD" >&2
    exit 2
    ;;
esac

echo ">>> METHOD=$METHOD complete"
