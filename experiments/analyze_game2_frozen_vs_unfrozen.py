#!/usr/bin/env python
"""Frozen-vs-unfrozen Game 2 (contrastive) comparison across CLTs.

Game 2's headline is overlap_rate (target/foil hop separation) plus the
unique_y / unique_foil set sizes. The frozen runs gave overlap_rate = 0.00
everywhere (perfectly disjoint hops). This asks whether that clean separation
survives unfreezing attention -- i.e. is the contrastive structure a property of
the circuit or an artifact of the frozen-attention convention?

Game 2 selects on raw game2_utility (no normalized stop), so unlike Game 1 it
was never affected by the denominator bug; overlap_rate is denominator-free too.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Optional

FROZEN = {
    "gemma2-426k": "results/macag_sweep",
    "gemma2-2.5M": "results/macag_clt_compare/gemma2-2.5M",
    "llama32-524k": "results/macag_clt_compare/llama32-524k",
}
SLUGS = ["dallas-austin", "houston-austin", "chicago-springfield", "miami-tallahassee",
         "detroit-lansing", "portland-salem", "cleveland-columbus", "philadelphia-harrisburg"]


def read(run_dir: str) -> Optional[dict[str, Any]]:
    p = os.path.join(run_dir, "macag_game2.json")
    if not os.path.isfile(p):
        return None
    g = json.load(open(p))
    ev, sc = g.get("evidence", {}), g.get("scores", {})
    tgt = sc.get("target", {}) if isinstance(sc.get("target"), dict) else {}
    return {
        "overlap": sc.get("overlap_rate"),
        "ny": len(ev.get("unique_y", [])),
        "nfoil": len(ev.get("unique_foil", [])),
        "nshared": len(ev.get("shared", [])),
        "pref": (tgt.get("all") or 0.0) > 0,  # did the model prefer the target?
    }


def fnum(x: Any, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unfrozen-root", default="results/macag_unfrozen_raw")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    csv_path = args.csv or os.path.join(args.unfrozen_root, "game2_frozen_vs_unfrozen.csv")

    rows = []
    for tag, fdir in FROZEN.items():
        udir_base = f"{args.unfrozen_root}/{tag}"
        print(f"\n===== {tag} =====")
        print(f"  {'prompt':22} {'overlap f→u':>14} {'|uniq_y| f→u':>14} "
              f"{'|uniq_foil| f→u':>16} {'shared f→u':>12} {'pref':>5}")
        for slug in SLUGS:
            fr = read(f"{fdir}/{slug}")
            uf = read(f"{udir_base}/{slug}")
            if fr is None and uf is None:
                continue
            fr = fr or {}; uf = uf or {}
            print(f"  {slug:22} "
                  f"{fnum(fr.get('overlap'))+'→'+fnum(uf.get('overlap')):>14} "
                  f"{str(fr.get('ny','-'))+'→'+str(uf.get('ny','-')):>14} "
                  f"{str(fr.get('nfoil','-'))+'→'+str(uf.get('nfoil','-')):>16} "
                  f"{str(fr.get('nshared','-'))+'→'+str(uf.get('nshared','-')):>12} "
                  f"{('Y' if uf.get('pref') else 'n'):>5}")
            rows.append({"clt": tag, "slug": slug, "pref": uf.get("pref"),
                         "overlap_f": fr.get("overlap"), "overlap_u": uf.get("overlap"),
                         "uniq_y_f": fr.get("ny"), "uniq_y_u": uf.get("ny"),
                         "uniq_foil_f": fr.get("nfoil"), "uniq_foil_u": uf.get("nfoil"),
                         "shared_f": fr.get("nshared"), "shared_u": uf.get("nshared")})

    print("\n===== Aggregate (pref=Y only) =====")
    print(f"  {'CLT':14} {'n':>3} {'mean overlap f→u':>20} {'mean |uniq_y| f→u':>20} "
          f"{'mean |uniq_foil| f→u':>22} {'#overlap>0 unfrozen':>20}")
    for tag in FROZEN:
        rs = [r for r in rows if r["clt"] == tag and r["pref"]]
        if not rs:
            print(f"  {tag:14}  (no valid prompts)"); continue
        def m(k):
            v = [r[k] for r in rs if isinstance(r[k], (int, float))]
            return sum(v) / len(v) if v else float("nan")
        nz = sum(1 for r in rs if isinstance(r["overlap_u"], (int, float)) and r["overlap_u"] > 0)
        print(f"  {tag:14} {len(rs):>3} {m('overlap_f'):>9.2f}→{m('overlap_u'):<9.2f} "
              f"{m('uniq_y_f'):>9.1f}→{m('uniq_y_u'):<9.1f} "
              f"{m('uniq_foil_f'):>10.1f}→{m('uniq_foil_u'):<10.1f} {nz:>8}/{len(rs)}")

    print("\nReading: frozen overlap_rate was ~0 everywhere (disjoint hops). If it stays")
    print("~0 unfrozen, the contrastive target/foil separation is a circuit property,")
    print("not a frozen-attention artifact. '#overlap>0 unfrozen' counts prompts where")
    print("unfreezing introduced target/foil feature sharing.")

    if rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
