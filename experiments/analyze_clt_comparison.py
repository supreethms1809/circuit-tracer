#!/usr/bin/env python
"""Cross-CLT comparison for the multi-CLT MACAG experiment.

Loads the per-prompt MACAG results for every cross-layer transcoder listed in
``experiments/macag_clt_compare.json`` and tabulates them side by side. The
headline question: does increasing capacity (gemma 426k -> 2.5M) or switching
model eliminate the reconstruction failures the 426k CLT shows, where the
behavior lives in error nodes rather than features (negative recoverable_range
=> negative normalized faithfulness, e.g. philadelphia)?

Usage:
  python experiments/analyze_clt_comparison.py --config experiments/macag_clt_compare.json \
      --csv results/macag_clt_compare/comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Optional

from analyze_macag_sweep import analyze_run


def _load_runs(sweep_dir: str) -> dict[str, dict[str, Any]]:
    """slug -> record, for every completed prompt run under sweep_dir."""
    if not os.path.isdir(sweep_dir):
        return {}
    out = {}
    for name in sorted(os.listdir(sweep_dir)):
        d = os.path.join(sweep_dir, name)
        if os.path.isfile(os.path.join(d, "macag_game1.json")):
            rec = analyze_run(d)
            if rec is not None:
                out[rec["slug"]] = rec
    return out


def _cell(rec: Optional[dict[str, Any]], key: str, nd: int = 2) -> str:
    if rec is None:
        return "  --"
    v = rec.get(key)
    if v is None:
        return "  --"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="experiments/macag_clt_compare.json")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    clts = cfg["clts"]
    # tag -> {slug -> record}
    data = {c["tag"]: _load_runs(c["sweep_dir"]) for c in clts}
    tags = [c["tag"] for c in clts if data[c["tag"]]]
    if not tags:
        print("No completed CLT runs found yet.")
        return

    all_slugs = sorted({s for tag in tags for s in data[tag]})

    def matrix(title: str, key: str, nd: int = 2, note: str = "") -> None:
        print(f"\n=== {title} ===")
        if note:
            print(f"    {note}")
        hdr = ["prompt"] + tags
        print("  ".join(f"{h[:13]:>13}" for h in hdr))
        for slug in all_slugs:
            row = [slug[:13]] + [_cell(data[t].get(slug), key, nd) for t in tags]
            print("  ".join(f"{c:>13}" for c in row))

    matrix("recoverable_range  (NEGATIVE = CLT reconstruction failure: behavior in error nodes)",
           "recoverable_range", 2,
           note="all - empty. <0 means ablating all features INCREASES the target-foil gap.")
    matrix("faithfulness_normalized  (~1.0 good; <0 = anomalous, see recoverable_range)",
           "faithfulness_norm", 2)
    matrix("target_preferred  (Y = model made the two-hop at baseline; n = invalid row)",
           "target_preferred", 0)
    matrix("sparsity  (|E*| fraction sparsity; higher = fewer features explain it)",
           "sparsity", 3)
    matrix("g2_overlap_rate  (0 = clean target/foil hop separation)",
           "g2_overlap_rate", 2)

    # --- per-CLT aggregates -------------------------------------------------
    print("\n=== Per-CLT aggregate ===")
    print("    (recon_fail and mean_faith* counted only over pref=Y prompts)")
    cols = ["CLT", "n", "pref", "recon_fail", "mean_faith*", "mean_sparsity", "mean_overlap"]
    print("  ".join(f"{c:>13}" for c in cols))
    agg_rows = []
    for t in tags:
        recs = list(data[t].values())
        n = len(recs)
        pref = sum(1 for r in recs if r.get("target_preferred"))
        # reconstruction failures only count among prompts the model ACTUALLY
        # performs (target_preferred): a negative range on a behavior the model
        # doesn't do is meaningless, not a transcoder failure.
        recon_fail = sum(1 for r in recs
                         if r.get("target_preferred")
                         and isinstance(r.get("recoverable_range"), (int, float))
                         and r["recoverable_range"] < 0)
        # mean faithfulness over VALID rows only: model does the behavior AND
        # the recoverable range is positive (otherwise normalization is degenerate)
        valid_fn = [r["faithfulness_norm"] for r in recs
                    if r.get("target_preferred")
                    and isinstance(r.get("recoverable_range"), (int, float))
                    and r["recoverable_range"] > 0
                    and r.get("faithfulness_norm") is not None]
        spars = [r["sparsity"] for r in recs if r.get("sparsity") is not None]
        ov = [r["g2_overlap_rate"] for r in recs if r.get("g2_overlap_rate") is not None]
        mf = sum(valid_fn) / len(valid_fn) if valid_fn else float("nan")
        ms = sum(spars) / len(spars) if spars else float("nan")
        mo = sum(ov) / len(ov) if ov else float("nan")
        row = [t[:13], str(n), str(pref), str(recon_fail),
               f"{mf:.2f}", f"{ms:.3f}", f"{mo:.2f}"]
        print("  ".join(f"{c:>13}" for c in row))
        agg_rows.append({
            "CLT": t, "n_prompts": n, "target_preferred": pref,
            "recon_failures_neg_range": recon_fail,
            "mean_faith_norm_valid": mf, "mean_sparsity": ms, "mean_overlap": mo,
        })
    print("    * mean_faith* averages only rows with positive recoverable_range "
          "(reconstruction-failure rows are not faithfulness-meaningful).")

    print("\nReading guide:")
    print("  * Capacity control: compare gemma2-426k vs gemma2-2.5M columns. If 2.5M")
    print("    flips a negative recoverable_range positive (e.g. philadelphia), the")
    print("    failure was capacity-limited -> spline must beat linear at MATCHED")
    print("    capacity. If it stays negative -> failure is representational (motivates")
    print("    the nonlinear encoder).")
    print("  * 'recon_fail' count per CLT = how many prompts that transcoder cannot")
    print("    faithfully reconstruct. Lower is better.")
    print("  * Cross-model (llama / gpt-oss) tests whether the patterns are general,")
    print("    but differ in model+size so read them as generality, not a clean control.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)
        # also dump the full per-prompt-per-CLT long table
        long_path = args.csv.replace(".csv", "_per_prompt.csv")
        with open(long_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["CLT", "slug", "target_preferred", "recoverable_range",
                        "faithfulness_norm", "sparsity", "g2_overlap_rate", "n_E_star"])
            for t in tags:
                for slug in all_slugs:
                    r = data[t].get(slug)
                    if r is None:
                        continue
                    w.writerow([t, slug, r.get("target_preferred"),
                                r.get("recoverable_range"), r.get("faithfulness_norm"),
                                r.get("sparsity"), r.get("g2_overlap_rate"),
                                r.get("n_E_star")])
        print(f"\nWrote {args.csv} and {long_path}")


if __name__ == "__main__":
    main()
