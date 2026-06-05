#!/usr/bin/env python
"""Aggregate a MACAG generalization sweep into a comparison table + signature.

For each prompt run produced by ``scripts/run_macag_sweep.sh`` this reads the
Game 1 / Game 2 result JSONs and the attribution graph, then reports:

  * whether the model actually preferred the target at baseline (sign of `all`),
  * the headline faithfulness metrics at the achieved sparsity,
  * the *structural signature* of the minimal faithful set: each evidence
    feature as (layer, token-role), where token-role is the prompt token it
    reads from and its reverse position (0 = final token where the model
    predicts). Feature indices differ per fact; what should generalize is this
    (layer, token-role) pattern.

Usage:
  python experiments/analyze_macag_sweep.py --sweep-dir results/macag_sweep
  python experiments/analyze_macag_sweep.py --sweep-dir results/macag_sweep \
      --csv results/macag_sweep/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Any, Optional


def _load(path: str) -> Optional[dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n["node_id"]: n for n in graph.get("nodes", []) if "node_id" in n}


def _token_role(node_id: str, graph: dict[str, Any],
                node_idx: dict[str, dict[str, Any]]) -> tuple[int, str, int]:
    """Return (layer, token_text, reverse_pos) for a feature node id.

    reverse_pos is computed from the token count (0 = final token where the
    model predicts). The graph's own ``reverse_ctx_idx`` field is unreliable
    here (it is 0 for every node in the current export), so we do not use it.
    """
    layer_s, _feat, pos_s = node_id.split("_")
    layer, pos = int(layer_s), int(pos_s)
    toks = graph.get("metadata", {}).get("prompt_tokens", [])
    rev = (len(toks) - 1 - pos) if toks else -1
    tok = toks[pos] if 0 <= pos < len(toks) else "?"
    return layer, tok, int(rev)


def analyze_run(run_dir: str) -> Optional[dict[str, Any]]:
    slug = os.path.basename(run_dir.rstrip("/"))
    g1 = _load(os.path.join(run_dir, "macag_game1.json"))
    g2 = _load(os.path.join(run_dir, "macag_game2.json"))
    graph = _load(os.path.join(run_dir, "graphs", f"{slug}.json"))
    if g1 is None or graph is None:
        return None

    node_idx = _node_index(graph)
    s1 = g1.get("scores", {})
    e_star = g1.get("evidence", {}).get("E_star", [])
    signature = [_token_role(nid, graph, node_idx) for nid in e_star]

    # infer model depth from transcoder/error nodes (logit nodes use a sentinel)
    layer_ints = [
        int(n["layer"]) for n in graph.get("nodes", [])
        if n.get("feature_type") in ("cross layer transcoder", "mlp reconstruction error")
        and str(n.get("layer", "")).lstrip("-").isdigit()
    ]
    n_layers = (max(layer_ints) + 1) if layer_ints else 0

    rec = {
        "slug": slug,
        "prompt": graph.get("metadata", {}).get("prompt", ""),
        "target": g1.get("target"),
        "foil": g1.get("foil"),
        # baseline: did the model actually prefer the target token?
        "all": s1.get("all"),
        "empty": s1.get("empty"),
        "recoverable_range": s1.get("recoverable_range"),
        "target_preferred": (s1.get("all") or 0.0) > 0,
        # game 1 headline
        "n_E_star": len(e_star),
        "sufficiency_norm": s1.get("sufficiency_normalized"),
        "necessity_norm": s1.get("necessity_normalized"),
        "faithfulness_norm": s1.get("faithfulness_normalized"),
        "sparsity": g1.get("stats", {}).get("sparsity"),
        "n_layers": n_layers,
        "signature": signature,  # list of (layer, token, reverse_pos)
    }
    if g2 is not None:
        ev = g2.get("evidence", {})
        sc = g2.get("scores", {})
        rec.update({
            "g2_overlap_rate": sc.get("overlap_rate"),
            "g2_n_y": len(ev.get("unique_y", [])),
            "g2_n_foil": len(ev.get("unique_foil", [])),
        })
    return rec


def _fmt(x: Any, nd: int = 2) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return "-" if x is None else str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", required=True, help="dir containing one subdir per prompt")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args()

    subdirs = sorted(
        d for d in (os.path.join(args.sweep_dir, x) for x in os.listdir(args.sweep_dir))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "macag_game1.json"))
    )
    recs = [r for r in (analyze_run(d) for d in subdirs) if r is not None]
    if not recs:
        print(f"No completed runs found under {args.sweep_dir}")
        return

    # --- per-prompt table ---------------------------------------------------
    print("\n=== Per-prompt metrics ===")
    hdr = ["slug", "pref?", "faith_norm", "suff_norm", "nec_norm",
           "|E*|", "sparsity", "g2_overlap", "|E_y|", "|E_foil|"]
    print("  ".join(f"{h:>11}" for h in hdr))
    for r in recs:
        row = [
            r["slug"][:11],
            "Y" if r["target_preferred"] else "n",
            _fmt(r["faithfulness_norm"]), _fmt(r["sufficiency_norm"]),
            _fmt(r["necessity_norm"]), str(r["n_E_star"]),
            _fmt(r["sparsity"], 3),
            _fmt(r.get("g2_overlap_rate")), _fmt(r.get("g2_n_y")),
            _fmt(r.get("g2_n_foil")),
        ]
        print("  ".join(f"{c:>11}" for c in row))

    n = len(recs)
    pref = sum(1 for r in recs if r["target_preferred"])
    fn = [r["faithfulness_norm"] for r in recs if r["faithfulness_norm"] is not None]
    ov = [r.get("g2_overlap_rate") for r in recs if r.get("g2_overlap_rate") is not None]
    print(f"\n  prompts={n}  target_preferred={pref}/{n}", end="")
    if fn:
        print(f"  mean_faith_norm={sum(fn)/len(fn):.2f}", end="")
    if ov:
        print(f"  mean_g2_overlap={sum(ov)/len(ov):.2f}", end="")
    print()

    # --- structural signature ----------------------------------------------
    print("\n=== Minimal-faithful-set signature per prompt ===")
    print("    (layer, token, rev_pos)   rev_pos 0 = final token where model predicts")
    for r in recs:
        sig = ", ".join(f"(L{l},{t.strip()!r},r{rp})" for l, t, rp in r["signature"])
        print(f"  {r['slug']:>22}: {sig}")

    # --- aggregate signature: which (layer-band, token-role) recur? ---------
    # bands are thirds of model depth so models of different n_layers compare.
    def band(layer: int, n_layers: int) -> str:
        if n_layers <= 0:
            return f"L{layer}"
        third = n_layers / 3.0
        if layer < third:
            return "early(0-1/3)"
        if layer < 2 * third:
            return "mid(1/3-2/3)"
        return "late(2/3-1)"

    band_ctr: Counter[str] = Counter()
    role_ctr: Counter[str] = Counter()
    bandrole_ctr: Counter[tuple[str, str]] = Counter()
    for r in recs:
        seen_b, seen_r, seen_br = set(), set(), set()
        for layer, tok, rp in r["signature"]:
            b = band(layer, r.get("n_layers", 0))
            role = "FINAL(' is')" if rp == 0 else tok.strip() or "?"
            seen_b.add(b); seen_r.add(role); seen_br.add((b, role))
        band_ctr.update(seen_b)
        role_ctr.update(seen_r)
        bandrole_ctr.update(seen_br)

    print(f"\n=== Signature consistency across {n} prompts (count = # prompts that use it) ===")
    print("  Layer band:")
    for b, c in sorted(band_ctr.items(), key=lambda x: -x[1]):
        print(f"    {b:>12}: {c}/{n}")
    print("  Token role read:")
    for role, c in sorted(role_ctr.items(), key=lambda x: -x[1]):
        print(f"    {role:>14}: {c}/{n}")
    print("  (layer-band x token-role):")
    for (b, role), c in sorted(bandrole_ctr.items(), key=lambda x: -x[1]):
        print(f"    {b:>12} @ {role:<14}: {c}/{n}")

    print("\nReading guide:")
    print("  * target_preferred should be Y for every prompt; if 'n', the model")
    print("    didn't make the two-hop and that row's circuit explains a non-behavior.")
    print("  * A reusable algorithm shows the SAME (layer-band x token-role) cells")
    print("    lit across most prompts (e.g. early@city + late@FINAL), even though")
    print("    the concrete feature indices differ per fact.")
    print("  * Low, stable g2_overlap means the target/foil hops stay cleanly split.")

    # --- CSV ----------------------------------------------------------------
    if args.csv:
        cols = ["slug", "prompt", "target", "foil", "target_preferred", "all",
                "empty", "recoverable_range", "n_E_star", "faithfulness_norm",
                "sufficiency_norm", "necessity_norm", "sparsity",
                "g2_overlap_rate", "g2_n_y", "g2_n_foil"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow(r)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
