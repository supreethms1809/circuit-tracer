#!/usr/bin/env python
"""Compare Game 2 ABR vs fictitious-play (fp) solvers across the ACDC runs.

Reads results/macag_acdc/<clt_tag>/<slug>/{macag_game2_abr.json,
macag_game2_fp.json}. For each run it reports the selected-evidence sizes,
overlap_rate, and the two agents' utilities under each solver, plus how much the
two solvers actually disagree (Jaccard of E_y and E_foil between ABR and FP).

The question this answers: does fictitious play (best-responding to the
opponent's empirical history) pick a materially different contrastive evidence
set, or land in the same place as last-iterate best response (ABR)?
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Optional

BENCH = "macag/data/acdc_benchmark_prompts.json"
CLTS = ["gemma2-426k", "gemma2-2.5M", "llama32-524k"]


def slug_to_task(bench_path: str) -> dict[str, str]:
    d = json.load(open(bench_path))
    return {it["id"]: task for task, items in d["tasks"].items() for it in items}


def _ev_sets(g2: dict[str, Any]) -> tuple[set, set]:
    ev = g2.get("evidence", {})
    return set(ev.get("E_y", [])), set(ev.get("E_foil", []))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def read(run_dir: str) -> Optional[dict[str, Any]]:
    pa = os.path.join(run_dir, "macag_game2_abr.json")
    pf = os.path.join(run_dir, "macag_game2_fp.json")
    if not (os.path.isfile(pa) and os.path.isfile(pf)):
        return None
    a, f = json.load(open(pa)), json.load(open(pf))
    sa, sf = a.get("scores", {}), f.get("scores", {})
    Ey_a, Ef_a = _ev_sets(a)
    Ey_f, Ef_f = _ev_sets(f)
    return {
        # sizes
        "n_y_abr": len(Ey_a), "n_y_fp": len(Ey_f),
        "n_foil_abr": len(Ef_a), "n_foil_fp": len(Ef_f),
        # overlap_rate (shared/union of the two agents within a solver)
        "ovl_abr": sa.get("overlap_rate"), "ovl_fp": sf.get("overlap_rate"),
        # combined utilities
        "uy_abr": sa.get("utility_target"), "uy_fp": sf.get("utility_target"),
        "uf_abr": sa.get("utility_foil"), "uf_fp": sf.get("utility_foil"),
        # solver agreement: how much ABR and FP picked the same nodes
        "jac_y": _jaccard(Ey_a, Ey_f), "jac_foil": _jaccard(Ef_a, Ef_f),
        # convergence (best_iteration is the retained best-iterate)
        "abr_converged": a.get("stats", {}).get("converged"),
        "abr_iters": a.get("stats", {}).get("iterations"),
        "fp_converged": f.get("stats", {}).get("converged"),
        "fp_iters": f.get("stats", {}).get("iterations"),
    }


def fnum(x: Any, nd: int = 2) -> str:
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/macag_acdc")
    ap.add_argument("--csv", default="results/macag_acdc/abr_vs_fp.csv")
    ap.add_argument("--bench", default=BENCH,
                    help="prompt JSON for slug->task labels (use the nonlinear set when "
                         "analyzing results/macag_nonlinear*)")
    args = ap.parse_args()
    task_of = slug_to_task(args.bench)

    rows = []
    for tag in CLTS:
        base = os.path.join(args.root, tag)
        if not os.path.isdir(base):
            continue
        slugs = sorted(s for s in os.listdir(base)
                       if os.path.isfile(os.path.join(base, s, "macag_game2_fp.json")))
        if not slugs:
            continue
        print(f"\n===== {tag} =====")
        print(f"  {'slug':14} {'|Ey|a/f':>8} {'|Ef|a/f':>8} {'ovl a/f':>9} "
              f"{'Uy a/f':>14} {'Uf a/f':>14} {'jacEy':>6} {'jacEf':>6}")
        for slug in slugs:
            r = read(os.path.join(base, slug))
            if r is None:
                continue
            print(f"  {slug:14} "
                  f"{r['n_y_abr']:>3}/{r['n_y_fp']:<4} "
                  f"{r['n_foil_abr']:>3}/{r['n_foil_fp']:<4} "
                  f"{fnum(r['ovl_abr'],2)}/{fnum(r['ovl_fp'],2)} "
                  f"{fnum(r['uy_abr']):>6}/{fnum(r['uy_fp']):<6} "
                  f"{fnum(r['uf_abr']):>6}/{fnum(r['uf_fp']):<6} "
                  f"{r['jac_y']:>6.2f} {r['jac_foil']:>6.2f}")
            rows.append({"clt": tag, "task": task_of.get(slug, "?"), "slug": slug, **r})

    print("\n===== Aggregate per CLT x task (mean over runs) =====")
    print(f"  {'CLT':14} {'task':24} {'n':>3} {'jacEy':>6} {'jacEf':>6} "
          f"{'dUy(fp-abr)':>11} {'dUf(fp-abr)':>11} {'fp_conv':>8}")
    agg = defaultdict(list)
    for r in rows:
        agg[(r["clt"], r["task"])].append(r)

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return sum(xs) / len(xs) if xs else float("nan")

    for (tag, task), rs in sorted(agg.items()):
        jy = mean([r["jac_y"] for r in rs])
        jf = mean([r["jac_foil"] for r in rs])
        duy = mean([r["uy_fp"] - r["uy_abr"] for r in rs
                    if isinstance(r["uy_fp"], (int, float)) and isinstance(r["uy_abr"], (int, float))])
        duf = mean([r["uf_fp"] - r["uf_abr"] for r in rs
                    if isinstance(r["uf_fp"], (int, float)) and isinstance(r["uf_abr"], (int, float))])
        nconv = sum(1 for r in rs if r["fp_converged"])
        print(f"  {tag:14} {task[:24]:24} {len(rs):>3} {jy:>6.2f} {jf:>6.2f} "
              f"{duy:>+11.3f} {duf:>+11.3f} {nconv:>3}/{len(rs):<3}")

    # headline: how often do ABR and FP pick identical sets?
    ident_y = sum(1 for r in rows if r["jac_y"] == 1.0)
    ident_f = sum(1 for r in rows if r["jac_foil"] == 1.0)
    print(f"\nIdentical E_y (jac=1.0):    {ident_y}/{len(rows)} runs")
    print(f"Identical E_foil (jac=1.0): {ident_f}/{len(rows)} runs")

    if rows:
        with open(args.csv, "w", newline="") as f:
            cols = ["clt", "task", "slug", "n_y_abr", "n_y_fp", "n_foil_abr", "n_foil_fp",
                    "ovl_abr", "ovl_fp", "uy_abr", "uy_fp", "uf_abr", "uf_fp",
                    "jac_y", "jac_foil", "abr_converged", "abr_iters",
                    "fp_converged", "fp_iters"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
