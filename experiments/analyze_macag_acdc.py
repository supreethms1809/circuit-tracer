#!/usr/bin/env python
"""Analyze the ACDC-benchmark MACAG runs across CLTs, grouped by task.

Reads results/macag_acdc/<clt_tag>/<slug>/{macag_game1.json, macag_game2.json}.
Supports single-mode Game 1 outputs and dual-freeze outputs (``freeze_mode='both'``);
the summary table reports the frozen leg by default (adds ``verdict`` when dual).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Optional

from macag.kl_rescore import read_game1_kl_faith
from spline_clt.paper.reporting import bootstrap_mean_ci

BENCH = "macag/data/acdc_benchmark_prompts.json"
CLTS = ["gemma2-426k", "gemma2-2.5M", "llama32-524k"]


def slug_to_task(bench_path: str) -> dict[str, str]:
    d = json.load(open(bench_path))
    return {it["id"]: task for task, items in d["tasks"].items() for it in items}


def _game1_leg(g1: dict[str, Any], leg: str = "frozen") -> dict[str, Any]:
    if g1.get("freeze_mode") == "both":
        return g1.get(leg, {})
    return g1


def read(run_dir: str) -> Optional[dict[str, Any]]:
    g1p = os.path.join(run_dir, "macag_game1.json")
    if not os.path.isfile(g1p):
        return None
    g1 = json.load(open(g1p))
    leg = _game1_leg(g1)
    s = leg.get("scores", {})
    e_star = leg.get("evidence", {}).get("E_star", [])
    rec: dict[str, Any] = {
        "suff": s.get("sufficiency"), "faith": s.get("faithfulness"),
        "range": s.get("recoverable_range"), "n": len(e_star),
        "pref": (s.get("all") or 0.0) > 0,
    }
    if g1.get("freeze_mode") == "both":
        diag = g1.get("attention_mediation", {})
        rec["verdict"] = diag.get("verdict")
        rec["range_flip"] = diag.get("range_flip")
    kl_faith = read_game1_kl_faith(run_dir)
    if kl_faith is not None:
        rec["kl_faith"] = kl_faith
        if isinstance(rec.get("faith"), (int, float)):
            rec["faith_kl_gap"] = rec["faith"] - kl_faith
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
    ap.add_argument("--bench", default=BENCH,
                    help="prompt JSON for slug->task labels (use the nonlinear set when "
                         "analyzing results/macag_nonlinear*)")
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    task_of = slug_to_task(args.bench)

    rows = []
    for tag in CLTS:
        base = os.path.join(args.root, tag)
        if not os.path.isdir(base):
            continue
        slugs = sorted(s for s in os.listdir(base)
                       if os.path.isfile(os.path.join(base, s, "macag_game1.json")))
        print(f"\n===== {tag} =====")
        print(f"  {'task':14} {'slug':14} {'pref':>5} {'rawSuff':>8} {'rawFaith':>9} "
              f"{'klFaith':>8} {'range':>8} {'|E*|':>5} {'g2ovl':>6} {'verdict':>12}")
        for slug in slugs:
            r = read(os.path.join(base, slug))
            if r is None:
                continue
            task = task_of.get(slug, "?")
            verdict = str(r.get("verdict") or "")[:12]
            print(f"  {task[:14]:14} {slug:14} {('Y' if r['pref'] else 'n'):>5} "
                  f"{fnum(r.get('suff')):>8} {fnum(r.get('faith')):>9} "
                  f"{fnum(r.get('kl_faith')):>8} "
                  f"{fnum(r.get('range')):>8} {str(r.get('n','-')):>5} {fnum(r.get('overlap')):>6} "
                  f"{verdict:>12}")
            rows.append({"clt": tag, "task": task, "slug": slug, **r})

    # aggregate per (CLT, task), pref=Y only
    print("\n===== Aggregate per CLT x task (pref=Y only) =====")
    print(f"  {'CLT':14} {'task':30} {'n_pref':>7} {'recon_fail':>11} "
          f"{'mean rawFaith':>14} {'mean klFaith':>12} {'mean g2ovl':>11} {'flip_rate [95% CI]':>22}")
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
        mk = [r["kl_faith"] for r in pref if isinstance(r.get("kl_faith"), (int, float))]
        ov = [r["overlap"] for r in pref if isinstance(r.get("overlap"), (int, float))]
        flips = [r["range_flip"] for r in pref if r.get("range_flip") is True]
        mfv = sum(mf) / len(mf) if mf else float("nan")
        mkv = sum(mk) / len(mk) if mk else float("nan")
        ovv = sum(ov) / len(ov) if ov else float("nan")
        flip_rate = len(flips) / len(pref) if pref else float("nan")
        flip_ind = [1.0 if r.get("range_flip") is True else 0.0 for r in pref]
        flip_lo, flip_hi = bootstrap_mean_ci(
            flip_ind, args.bootstrap_samples, 0.95, args.seed)
        print(f"  {tag:14} {task[:30]:30} {len(pref):>7} {recon:>5}/{len(pref):<5} "
              f"{mfv:>13.2f} {mkv:>10.2f} {ovv:>10.2f} {flip_rate:>9.1%} "
              f"[{flip_lo:.1%}, {flip_hi:.1%}]")
        arows.append({"clt": tag, "task": task, "n_pref": len(pref),
                      "recon_fail": recon, "mean_raw_faith": mfv, "mean_kl_faith": mkv,
                      "mean_overlap": ovv, "flip_rate": flip_rate,
                      "flip_lo": flip_lo, "flip_hi": flip_hi})

    print("\nReading: IOI is attention-mediated -> expect negative range / recon_fail under")
    print("frozen attention (answer carried by frozen attention, not MLP features). docstring")
    print("is feature-carried -> positive range. Compare raw faith + overlap across CLTs.")

    if rows:
        with open(args.csv, "w", newline="") as f:
            cols = ["clt", "task", "slug", "pref", "suff", "faith", "kl_faith", "faith_kl_gap",
                    "range", "n", "overlap", "verdict", "range_flip"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.csv}")
    if arows:
        agg_csv = os.path.splitext(args.csv)[0] + "_agg.csv"
        with open(agg_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(arows[0].keys()))
            w.writeheader(); w.writerows(arows)
        print(f"Wrote {agg_csv}")


if __name__ == "__main__":
    main()
