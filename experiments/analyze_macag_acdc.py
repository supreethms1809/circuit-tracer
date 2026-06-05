#!/usr/bin/env python
"""Analyze the ACDC-benchmark MACAG runs across CLTs, grouped by task.

Reads results/macag_acdc/<clt_tag>/<slug>/{macag_game1.json, macag_game2.json}.
Reports RAW sufficiency / faithfulness (stable; IOI shows negative
recoverable_range because it is an attention-mediated task, so the normalized
faithfulness is unreliable there) plus Game 2 overlap_rate, grouped by task.
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


def read(run_dir: str) -> Optional[dict[str, Any]]:
    g1p = os.path.join(run_dir, "macag_game1.json")
    if not os.path.isfile(g1p):
        return None
    s = json.load(open(g1p)).get("scores", {})
    e_star = json.load(open(g1p)).get("evidence", {}).get("E_star", [])
    rec = {
        "suff": s.get("sufficiency"), "faith": s.get("faithfulness"),
        "range": s.get("recoverable_range"), "n": len(e_star),
        "pref": (s.get("all") or 0.0) > 0,
    }
    g2p = os.path.join(run_dir, "macag_game2.json")
    if os.path.isfile(g2p):
        rec["overlap"] = json.load(open(g2p)).get("scores", {}).get("overlap_rate")
    return rec


def fnum(x: Any, nd: int = 2) -> str:
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/macag_acdc")
    ap.add_argument("--csv", default="results/macag_acdc/summary.csv")
    args = ap.parse_args()
    task_of = slug_to_task(BENCH)

    rows = []
    for tag in CLTS:
        base = os.path.join(args.root, tag)
        if not os.path.isdir(base):
            continue
        slugs = sorted(s for s in os.listdir(base)
                       if os.path.isfile(os.path.join(base, s, "macag_game1.json")))
        print(f"\n===== {tag} =====")
        print(f"  {'task':14} {'slug':14} {'pref':>5} {'rawSuff':>8} {'rawFaith':>9} "
              f"{'range':>8} {'|E*|':>5} {'g2ovl':>6}")
        for slug in slugs:
            r = read(os.path.join(base, slug))
            if r is None:
                continue
            task = task_of.get(slug, "?")
            print(f"  {task[:14]:14} {slug:14} {('Y' if r['pref'] else 'n'):>5} "
                  f"{fnum(r.get('suff')):>8} {fnum(r.get('faith')):>9} "
                  f"{fnum(r.get('range')):>8} {str(r.get('n','-')):>5} {fnum(r.get('overlap')):>6}")
            rows.append({"clt": tag, "task": task, "slug": slug, **r})

    # aggregate per (CLT, task), pref=Y only
    print("\n===== Aggregate per CLT x task (pref=Y only) =====")
    print(f"  {'CLT':14} {'task':30} {'n_pref':>7} {'recon_fail':>11} "
          f"{'mean rawFaith':>14} {'mean g2ovl':>11}")
    agg = defaultdict(list)
    for r in rows:
        agg[(r["clt"], r["task"])].append(r)
    arows = []
    for (tag, task), rs in sorted(agg.items()):
        pref = [r for r in rs if r["pref"]]
        if not pref:
            print(f"  {tag:14} {task[:30]:30} {0:>7}  (model performs none)")
            continue
        recon = sum(1 for r in pref if isinstance(r["range"], (int, float)) and r["range"] < 0)
        mf = [r["faith"] for r in pref if isinstance(r["faith"], (int, float))]
        ov = [r["overlap"] for r in pref if isinstance(r.get("overlap"), (int, float))]
        mfv = sum(mf) / len(mf) if mf else float("nan")
        ovv = sum(ov) / len(ov) if ov else float("nan")
        print(f"  {tag:14} {task[:30]:30} {len(pref):>7} {recon:>5}/{len(pref):<5} "
              f"{mfv:>13.2f} {ovv:>10.2f}")
        arows.append({"clt": tag, "task": task, "n_pref": len(pref),
                      "recon_fail": recon, "mean_raw_faith": mfv, "mean_overlap": ovv})

    print("\nReading: IOI is attention-mediated -> expect negative range / recon_fail under")
    print("frozen attention (answer carried by frozen attention, not MLP features). docstring")
    print("is feature-carried -> positive range. Compare raw faith + overlap across CLTs.")

    if rows:
        with open(args.csv, "w", newline="") as f:
            cols = ["clt", "task", "slug", "pref", "suff", "faith", "range", "n", "overlap"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
