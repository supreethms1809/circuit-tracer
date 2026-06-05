#!/usr/bin/env python
"""Compare frozen vs unfrozen-attention Game 1 across CLTs.

Frozen runs come from the original sweep dirs; unfrozen runs from
results/macag_unfrozen/<tag>/<slug>/ (produced by scripts/run_macag_unfrozen.sh).

Per (CLT, prompt) reports the minimal-faithful-set size, the count of UPSTREAM
features (rev_pos > 0 -- features NOT at the final prediction token, i.e. the
city/structure features that frozen attention otherwise carries for free), the
count of EARLY-layer features, normalized faithfulness, recoverable_range, and
whether it hit the budget cap. The headline test: unfreezing attention should
recruit upstream/early features the frozen metric dropped as redundant.
"""
from __future__ import annotations

import argparse
import csv
from typing import Any, Optional

from analyze_macag_sweep import analyze_run

# tag -> (frozen_dir, unfrozen_dir)
CLTS = {
    "gemma2-426k": ("results/macag_sweep", "results/macag_unfrozen/gemma2-426k"),
    "gemma2-2.5M": ("results/macag_clt_compare/gemma2-2.5M", "results/macag_unfrozen/gemma2-2.5M"),
    "llama32-524k": ("results/macag_clt_compare/llama32-524k", "results/macag_unfrozen/llama32-524k"),
}
SLUGS = ["dallas-austin", "houston-austin", "chicago-springfield", "miami-tallahassee",
         "detroit-lansing", "portland-salem", "cleveland-columbus", "philadelphia-harrisburg"]
FROZEN_BUDGET, UNFROZEN_BUDGET = 8, 20


def feats(rec: Optional[dict[str, Any]]) -> dict[str, Any]:
    if rec is None:
        return {}
    sig = rec.get("signature", [])
    nl = rec.get("n_layers", 0) or 0
    third = nl / 3.0 if nl else 0
    return {
        "n": rec.get("n_E_star"),
        "upstream": sum(1 for (_l, _t, rp) in sig if rp > 0),
        "early": sum(1 for (l, _t, _rp) in sig if third and l < third),
        "faith": rec.get("faithfulness_norm"),
        "range": rec.get("recoverable_range"),
        "pref": rec.get("target_preferred"),
    }


def f(x: Any, nd: int = 2) -> str:
    return f"{x:+.{nd}f}" if isinstance(x, float) else ("--" if x is None else str(x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/macag_unfrozen/frozen_vs_unfrozen.csv")
    args = ap.parse_args()

    rows = []
    for tag, (fdir, udir) in CLTS.items():
        print(f"\n===== {tag} =====")
        print(f"  {'prompt':22} {'|E*| f→u':>10} {'upstream f→u':>14} {'early f→u':>11} "
              f"{'faith f→u':>16} {'range f→u':>16} {'cap?':>5} {'pref':>5}")
        for slug in SLUGS:
            fr = feats(analyze_run(f"{fdir}/{slug}"))
            uf = feats(analyze_run(f"{udir}/{slug}"))
            if not fr and not uf:
                continue
            hit_cap = "Y" if uf.get("n") == UNFROZEN_BUDGET else "n"
            pref = "Y" if uf.get("pref") else ("n" if uf else fr.get("pref") and "Y" or "?")
            print(f"  {slug:22} {str(fr.get('n','-'))+'→'+str(uf.get('n','-')):>10} "
                  f"{str(fr.get('upstream','-'))+'→'+str(uf.get('upstream','-')):>14} "
                  f"{str(fr.get('early','-'))+'→'+str(uf.get('early','-')):>11} "
                  f"{f(fr.get('faith'))+'→'+f(uf.get('faith')):>16} "
                  f"{f(fr.get('range'))+'→'+f(uf.get('range')):>16} {hit_cap:>5} {pref:>5}")
            rows.append({"clt": tag, "slug": slug,
                         "E_frozen": fr.get("n"), "E_unfrozen": uf.get("n"),
                         "upstream_frozen": fr.get("upstream"), "upstream_unfrozen": uf.get("upstream"),
                         "early_frozen": fr.get("early"), "early_unfrozen": uf.get("early"),
                         "faith_frozen": fr.get("faith"), "faith_unfrozen": uf.get("faith"),
                         "range_frozen": fr.get("range"), "range_unfrozen": uf.get("range"),
                         "unfrozen_hit_cap": uf.get("n") == UNFROZEN_BUDGET,
                         "target_preferred": uf.get("pref")})

    # aggregate (pref=Y only)
    print("\n===== Aggregate (pref=Y prompts only) =====")
    print(f"  {'CLT':14} {'n':>3} {'mean|E*| f→u':>14} {'mean upstream f→u':>18} {'hit_cap':>8}")
    for tag in CLTS:
        rs = [r for r in rows if r["clt"] == tag and r["target_preferred"]]
        if not rs:
            print(f"  {tag:14}  (no valid prompts)"); continue
        def m(key):
            vals = [r[key] for r in rs if isinstance(r[key], (int, float))]
            return sum(vals) / len(vals) if vals else float("nan")
        cap = sum(1 for r in rs if r["unfrozen_hit_cap"])
        print(f"  {tag:14} {len(rs):>3} "
              f"{m('E_frozen'):>6.1f}→{m('E_unfrozen'):<6.1f} "
              f"{m('upstream_frozen'):>8.1f}→{m('upstream_unfrozen'):<8.1f} {cap:>5}/{len(rs)}")

    print("\nReading: upstream = features NOT at the final token (the city/structure")
    print("features frozen attention carries for free). Frozen→unfrozen upstream going")
    print("from ~0 to >0, and |E*| growing / hitting the cap, confirms frozen attention")
    print("was masking the upstream circuit. hit_cap=Y means it needs MORE than budget 20.")

    if rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
