#!/usr/bin/env python
"""Robust frozen-vs-unfrozen comparison using UNNORMALIZED scores.

The normalized faithfulness blew up under unfrozen attention only because its
denominator (recoverable_range = all - empty) goes degenerate when ablate-all +
free attention collapses the `empty` baseline. The RAW scores -- sufficiency
(keep_only - empty), necessity (all - remove), faithfulness_delta -- have no such
denominator and are already stored in every Game 1 output. The minimal set was
also selected on raw utility, so feature composition is unaffected.

This re-reads the stored raw scores (NO new model runs) and asks: under a stable
metric, does unfreezing still (a) recruit upstream features and (b) change
faithfulness in an interpretable way?
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Optional

# frozen source dirs per CLT; the unfrozen dir is <unfrozen_root>/<tag>
FROZEN = {
    "gemma2-426k": "results/macag_sweep",
    "gemma2-2.5M": "results/macag_clt_compare/gemma2-2.5M",
    "llama32-524k": "results/macag_clt_compare/llama32-524k",
}
SLUGS = ["dallas-austin", "houston-austin", "chicago-springfield", "miami-tallahassee",
         "detroit-lansing", "portland-salem", "cleveland-columbus", "philadelphia-harrisburg"]


def read(run_dir: str, slug: str) -> Optional[dict[str, Any]]:
    g1p = os.path.join(run_dir, "macag_game1.json")
    grp = os.path.join(run_dir, "graphs", f"{slug}.json")
    if not os.path.isfile(g1p) or not os.path.isfile(grp):
        return None
    g1 = json.load(open(g1p))
    s = g1.get("scores", {})
    n_tok = len(json.load(open(grp)).get("metadata", {}).get("prompt_tokens", []))
    e_star = g1.get("evidence", {}).get("E_star", [])
    final_pos = n_tok - 1
    upstream = sum(1 for nid in e_star if int(nid.split("_")[2]) != final_pos)
    return {
        "suff": s.get("sufficiency"),            # keep_only - empty  (stable)
        "nec": s.get("necessity"),               # all - remove       (stable)
        "faith": s.get("faithfulness"),          # raw delta          (stable)
        "range": s.get("recoverable_range"),     # all - empty  (the unstable denom)
        "n": len(e_star),
        "upstream": upstream,
        "pref": (s.get("all") or 0.0) > 0,
    }


def fnum(x: Any, nd: int = 2) -> str:
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unfrozen-root", default="results/macag_unfrozen_raw",
                    help="dir holding <tag>/<slug>/ unfrozen runs (default: raw_relative-stop rerun)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    csv_path = args.csv or os.path.join(args.unfrozen_root, "robust_frozen_vs_unfrozen.csv")
    CLTS = {tag: (fdir, f"{args.unfrozen_root}/{tag}") for tag, fdir in FROZEN.items()}

    rows = []
    for tag, (fdir, udir) in CLTS.items():
        print(f"\n===== {tag}  (RAW sufficiency / faithfulness -- stable) =====")
        print(f"  {'prompt':22} {'rawSuff f→u':>16} {'rawFaith f→u':>16} "
              f"{'|E*| f→u':>9} {'upstr f→u':>10} {'range f→u':>16} {'pref':>5}")
        for slug in SLUGS:
            fr = read(f"{fdir}/{slug}", slug)
            uf = read(f"{udir}/{slug}", slug)
            if fr is None and uf is None:
                continue
            fr = fr or {}; uf = uf or {}
            print(f"  {slug:22} "
                  f"{fnum(fr.get('suff'))+'→'+fnum(uf.get('suff')):>16} "
                  f"{fnum(fr.get('faith'))+'→'+fnum(uf.get('faith')):>16} "
                  f"{str(fr.get('n','-'))+'→'+str(uf.get('n','-')):>9} "
                  f"{str(fr.get('upstream','-'))+'→'+str(uf.get('upstream','-')):>10} "
                  f"{fnum(fr.get('range'))+'→'+fnum(uf.get('range')):>16} "
                  f"{('Y' if uf.get('pref') else 'n'):>5}")
            rows.append({"clt": tag, "slug": slug, "pref": uf.get("pref"),
                         "suff_f": fr.get("suff"), "suff_u": uf.get("suff"),
                         "faith_f": fr.get("faith"), "faith_u": uf.get("faith"),
                         "E_f": fr.get("n"), "E_u": uf.get("n"),
                         "upstream_f": fr.get("upstream"), "upstream_u": uf.get("upstream"),
                         "range_f": fr.get("range"), "range_u": uf.get("range")})

    print("\n===== Aggregate (pref=Y only) =====")
    print(f"  {'CLT':14} {'n':>3} {'mean rawSuff f→u':>20} {'mean rawFaith f→u':>20} "
          f"{'mean upstr f→u':>16} {'degen range_u':>14}")
    for tag in CLTS:
        rs = [r for r in rows if r["clt"] == tag and r["pref"]]
        if not rs:
            print(f"  {tag:14}  (no valid prompts)"); continue
        def m(k):
            v = [r[k] for r in rs if isinstance(r[k], (int, float))]
            return sum(v) / len(v) if v else float("nan")
        degen = sum(1 for r in rs if isinstance(r["range_u"], (int, float)) and r["range_u"] <= 0)
        print(f"  {tag:14} {len(rs):>3} "
              f"{m('suff_f'):>9.2f}→{m('suff_u'):<9.2f} "
              f"{m('faith_f'):>9.2f}→{m('faith_u'):<9.2f} "
              f"{m('upstream_f'):>7.1f}→{m('upstream_u'):<7.1f} {degen:>5}/{len(rs)}")

    print("\nReading:")
    print("  * rawSuff / rawFaith are stable (no denominator) -> compare these, not faith_norm.")
    print("  * 'degen range_u' = # prompts where the NORMALIZED metric's denominator went <=0")
    print("    unfrozen (i.e. where faith_norm was garbage). Raw scores stay finite there.")
    print("  * If mean upstream still rises f->u under raw scoring, upstream recruitment is real;")
    print("    if rawFaith stays comparable while faith_norm exploded, the instability was")
    print("    purely the normalization (a metric artifact, not a circuit effect).")

    if rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
