#!/usr/bin/env python
"""ACDC frozen-vs-unfrozen Game 1, grouped by task.

Frozen runs: results/macag_acdc/<tag>/<slug>/ ; unfrozen (raw_relative stop):
results/macag_acdc_unfrozen/<tag>/<slug>/. Tests the attention-mediation
hypothesis: gemma IOI was 10/10 reconstruction failures frozen (negative range,
answer in attention). If unfreezing recruits features and flips range positive,
that confirms the answer lived in frozen attention.

Uses RAW sufficiency/faithfulness (stable) + recoverable_range sign + upstream
feature count (features NOT at the final prediction token).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Optional

BENCH = "macag/data/acdc_benchmark_prompts.json"
TAGS = ["gemma2-426k", "gemma2-2.5M", "llama32-524k"]
FROZEN_ROOT = "results/macag_acdc"
UNFROZEN_ROOT = "results/macag_acdc_unfrozen"


def slug_to_task() -> dict[str, str]:
    d = json.load(open(BENCH))
    return {it["id"]: task for task, items in d["tasks"].items() for it in items}


def read(run_dir: str, slug: str) -> Optional[dict[str, Any]]:
    g1p = os.path.join(run_dir, "macag_game1.json")
    grp = os.path.join(run_dir, "graphs", f"{slug}.json")
    if not os.path.isfile(g1p):
        return None
    g1 = json.load(open(g1p))
    s = g1.get("scores", {})
    e_star = g1.get("evidence", {}).get("E_star", [])
    upstream = None
    if os.path.isfile(grp):
        ntok = len(json.load(open(grp)).get("metadata", {}).get("prompt_tokens", []))
        final = ntok - 1
        upstream = sum(1 for nid in e_star if int(nid.split("_")[2]) != final)
    return {"faith": s.get("faithfulness"), "range": s.get("recoverable_range"),
            "n": len(e_star), "upstream": upstream, "pref": (s.get("all") or 0.0) > 0}


def f(x: Any, nd: int = 1) -> str:
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/macag_acdc_unfrozen/acdc_frozen_vs_unfrozen.csv")
    args = ap.parse_args()
    task_of = slug_to_task()

    rows = []
    for tag in TAGS:
        fbase, ubase = f"{FROZEN_ROOT}/{tag}", f"{UNFROZEN_ROOT}/{tag}"
        if not os.path.isdir(fbase):
            continue
        slugs = sorted(s for s in os.listdir(fbase)
                       if os.path.isfile(os.path.join(fbase, s, "macag_game1.json")))
        print(f"\n===== {tag} =====")
        print(f"  {'slug':14} {'task':6} {'range f→u':>16} {'rawFaith f→u':>16} "
              f"{'|E*| f→u':>9} {'upstr f→u':>10} {'flip?':>6}")
        for slug in slugs:
            fr = read(f"{fbase}/{slug}", slug)
            uf = read(f"{ubase}/{slug}", slug)
            if fr is None:
                continue
            uf = uf or {}
            task = task_of.get(slug, "?")
            tshort = "IOI" if task.startswith("indirect") else ("doc" if task.startswith("docstring") else task[:6])
            flip = ""
            if isinstance(fr.get("range"), (int, float)) and isinstance(uf.get("range"), (int, float)):
                if fr["range"] < 0 <= uf["range"]:
                    flip = "NEG→POS"
                elif fr["range"] >= 0 > uf["range"]:
                    flip = "pos→neg"
            print(f"  {slug:14} {tshort:6} "
                  f"{f(fr.get('range'))+'→'+f(uf.get('range')):>16} "
                  f"{f(fr.get('faith'))+'→'+f(uf.get('faith')):>16} "
                  f"{str(fr.get('n','-'))+'→'+str(uf.get('n','-')):>9} "
                  f"{str(fr.get('upstream','-'))+'→'+str(uf.get('upstream','-')):>10} {flip:>7}")
            rows.append({"clt": tag, "task": task, "slug": slug,
                         "range_f": fr.get("range"), "range_u": uf.get("range"),
                         "faith_f": fr.get("faith"), "faith_u": uf.get("faith"),
                         "E_f": fr.get("n"), "E_u": uf.get("n"),
                         "upstream_f": fr.get("upstream"), "upstream_u": uf.get("upstream"),
                         "pref": fr.get("pref")})

    print("\n===== Aggregate per CLT x task (pref=Y) =====")
    print(f"  {'CLT':14} {'task':6} {'n':>3} {'recon_fail f→u':>16} "
          f"{'mean range f→u':>18} {'mean upstr f→u':>16}")
    agg = defaultdict(list)
    for r in rows:
        if r["pref"]:
            agg[(r["clt"], "IOI" if r["task"].startswith("indirect") else "doc")].append(r)
    for (tag, t), rs in sorted(agg.items()):
        def cnt(key):
            return sum(1 for r in rs if isinstance(r[key], (int, float)) and r[key] < 0)
        def mean(key):
            v = [r[key] for r in rs if isinstance(r[key], (int, float))]
            return sum(v) / len(v) if v else float("nan")
        print(f"  {tag:14} {t:6} {len(rs):>3} {cnt('range_f'):>6}/{len(rs)} → {cnt('range_u'):>2}/{len(rs)}   "
              f"{mean('range_f'):>7.1f}→{mean('range_u'):<7.1f} "
              f"{mean('upstream_f'):>6.1f}→{mean('upstream_u'):<6.1f}")

    print("\nReading: recon_fail = # prompts with negative recoverable_range. If gemma IOI")
    print("goes from N/N (frozen) toward 0/N (unfrozen) with rising upstream features, the")
    print("answer was carried by frozen attention -> confirmed attention-mediated.")

    if rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
