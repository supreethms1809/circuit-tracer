#!/usr/bin/env python
"""ACDC frozen-vs-unfrozen Game 1, grouped by task.

Primary path: dual-freeze ``macag_game1.json`` (``freeze_mode='both'``) under
``--root`` — reads ``frozen`` / ``unfrozen`` legs and ``attention_mediation`` from
one file per prompt.

Legacy fallback: pair ``--frozen-root`` and ``--unfrozen-root`` directories from
the old two-pass driver (separate macag_game1.json per mode).

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

from macag.kl_rescore import read_game1_kl_faith

BENCH = "macag/data/acdc_benchmark_prompts.json"
TAGS = ["gemma2-426k", "gemma2-2.5M", "llama32-524k"]
DEFAULT_ROOT = "results/macag_acdc"
FROZEN_ROOT = "results/macag_acdc"
UNFROZEN_ROOT = "results/macag_acdc_unfrozen"


def slug_to_task(bench_path: str = BENCH) -> dict[str, str]:
    d = json.load(open(bench_path))
    return {it["id"]: task for task, items in d["tasks"].items() for it in items}


def _leg_kl_faith(g1_leg: dict[str, Any], run_dir: str, leg: str | None) -> float | None:
    embedded = (g1_leg.get("kl_faithfulness") or {}).get("faithfulness")
    if isinstance(embedded, (int, float)):
        return float(embedded)
    if leg is not None:
        return read_game1_kl_faith(run_dir, leg=leg)
    return None


def _leg_metrics(
    g1_leg: dict[str, Any],
    slug: str,
    run_dir: str,
    leg: str | None = None,
) -> dict[str, Any]:
    s = g1_leg.get("scores", {})
    e_star = g1_leg.get("evidence", {}).get("E_star", [])
    upstream = None
    grp = os.path.join(run_dir, "graphs", f"{slug}.json")
    if os.path.isfile(grp):
        ntok = len(json.load(open(grp)).get("metadata", {}).get("prompt_tokens", []))
        final = ntok - 1
        upstream = sum(1 for nid in e_star if int(nid.split("_")[2]) != final)
    return {
        "faith": s.get("faithfulness"),
        "kl_faith": _leg_kl_faith(g1_leg, run_dir, leg),
        "range": s.get("recoverable_range"),
        "n": len(e_star),
        "upstream": upstream,
        "pref": (s.get("all") or 0.0) > 0,
    }


def read_dual(run_dir: str, slug: str) -> Optional[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    g1p = os.path.join(run_dir, "macag_game1.json")
    if not os.path.isfile(g1p):
        return None
    g1 = json.load(open(g1p))
    if g1.get("freeze_mode") != "both":
        return None
    fr = _leg_metrics(g1["frozen"], slug, run_dir, leg="frozen")
    uf = _leg_metrics(g1["unfrozen"], slug, run_dir, leg="unfrozen")
    diag = g1.get("attention_mediation", {})
    return fr, uf, diag


def read_legacy(run_dir: str, slug: str) -> Optional[dict[str, Any]]:
    g1p = os.path.join(run_dir, "macag_game1.json")
    if not os.path.isfile(g1p):
        return None
    g1 = json.load(open(g1p))
    if g1.get("freeze_mode") == "both":
        return None
    return _leg_metrics(g1, slug, run_dir)


def aggregate_flip_stats(
    rows: list[dict[str, Any]],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per-(clt, task) flip/negative-range rates with bootstrap CIs over prompts.

    Only ``pref`` (target-preferred) rows enter the aggregates. Rates are means
    of 0/1 indicators, so ``bootstrap_mean_ci`` on the indicators gives the CI
    on the proportion (the roadmap-B1.2 "fraction of prompts with
    recoverable_range < 0" quantity).
    """
    from spline_clt.paper.reporting import bootstrap_mean_ci

    def rate_ci(indicators: list[float]) -> tuple[float, float, float]:
        if not indicators:
            return float("nan"), float("nan"), float("nan")
        lo, hi = bootstrap_mean_ci(indicators, samples, confidence, seed)
        return sum(indicators) / len(indicators), lo, hi

    def mean(rs: list[dict[str, Any]], key: str) -> float:
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    agg: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("pref"):
            task = r.get("task", "?")
            tshort = "IOI" if task.startswith("indirect") else (
                "doc" if task.startswith("docstring") else task[:6])
            agg[(r["clt"], tshort)].append(r)

    records: list[dict[str, Any]] = []
    for (tag, t), rs in sorted(agg.items()):
        flip_rate, flip_lo, flip_hi = rate_ci(
            [1.0 if r.get("range_flip") is True else 0.0 for r in rs])
        neg_f = [1.0 if isinstance(r.get("range_f"), (int, float)) and r["range_f"] < 0
                 else 0.0 for r in rs]
        neg_u = [1.0 if isinstance(r.get("range_u"), (int, float)) and r["range_u"] < 0
                 else 0.0 for r in rs]
        negf_rate, negf_lo, negf_hi = rate_ci(neg_f)
        negu_rate, negu_lo, negu_hi = rate_ci(neg_u)
        records.append({
            "clt": tag, "task": t, "n": len(rs),
            "flip_rate": flip_rate, "flip_lo": flip_lo, "flip_hi": flip_hi,
            "negrange_f_rate": negf_rate, "negrange_f_lo": negf_lo, "negrange_f_hi": negf_hi,
            "negrange_u_rate": negu_rate, "negrange_u_lo": negu_lo, "negrange_u_hi": negu_hi,
            "recon_fail_f": int(sum(neg_f)), "recon_fail_u": int(sum(neg_u)),
            "mean_range_f": mean(rs, "range_f"), "mean_range_u": mean(rs, "range_u"),
            "mean_upstream_f": mean(rs, "upstream_f"), "mean_upstream_u": mean(rs, "upstream_u"),
        })
    return records


def f(x: Any, nd: int = 1) -> str:
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="Root with dual-freeze macag_game1.json (primary path)")
    ap.add_argument("--frozen-root", default=FROZEN_ROOT,
                    help="Legacy: separate frozen pass output root")
    ap.add_argument("--unfrozen-root", default=UNFROZEN_ROOT,
                    help="Legacy: separate unfrozen pass output root")
    ap.add_argument("--bench", default=BENCH,
                    help="prompt JSON for slug->task labels")
    ap.add_argument("--csv", default=None,
                    help="output CSV (default: <root>/frozen_vs_unfrozen.csv)")
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.csv is None:
        args.csv = os.path.join(args.root, "frozen_vs_unfrozen.csv")
    task_of = slug_to_task(args.bench)

    rows = []
    for tag in TAGS:
        base = os.path.join(args.root, tag)
        if not os.path.isdir(base):
            continue
        slugs = sorted(s for s in os.listdir(base)
                       if os.path.isfile(os.path.join(base, s, "macag_game1.json")))
        print(f"\n===== {tag} =====")
        print(f"  {'slug':14} {'task':6} {'range f→u':>16} {'rawFaith f→u':>16} "
              f"{'|E*| f→u':>9} {'upstr f→u':>10} {'flip?':>6} {'verdict':>12}")
        for slug in slugs:
            dual = read_dual(f"{base}/{slug}", slug)
            if dual is not None:
                fr, uf, diag = dual
                flip = "NEG→POS" if diag.get("range_flip") else ""
                verdict = str(diag.get("verdict") or "")[:12]
            else:
                fr = read_legacy(f"{args.frozen_root}/{tag}/{slug}", slug)
                uf = read_legacy(f"{args.unfrozen_root}/{tag}/{slug}", slug)
                flip = verdict = ""
                if fr is None:
                    continue
                if uf and isinstance(fr.get("range"), (int, float)) and isinstance(uf.get("range"), (int, float)):
                    if fr["range"] < 0 <= uf["range"]:
                        flip = "NEG→POS"
                    elif fr["range"] >= 0 > uf["range"]:
                        flip = "pos→neg"
            uf = uf or {}
            task = task_of.get(slug, "?")
            tshort = "IOI" if task.startswith("indirect") else ("doc" if task.startswith("docstring") else task[:6])
            print(f"  {slug:14} {tshort:6} "
                  f"{f(fr.get('range'))+'→'+f(uf.get('range')):>16} "
                  f"{f(fr.get('faith'))+'→'+f(uf.get('faith')):>16} "
                  f"{str(fr.get('n','-'))+'→'+str(uf.get('n','-')):>9} "
                  f"{str(fr.get('upstream','-'))+'→'+str(uf.get('upstream','-')):>10} "
                  f"{flip:>7} {verdict:>12}")
            rows.append({"clt": tag, "task": task, "slug": slug,
                         "range_f": fr.get("range"), "range_u": uf.get("range"),
                         "faith_f": fr.get("faith"), "faith_u": uf.get("faith"),
                         "kl_faith_f": fr.get("kl_faith"), "kl_faith_u": uf.get("kl_faith"),
                         "E_f": fr.get("n"), "E_u": uf.get("n"),
                         "upstream_f": fr.get("upstream"), "upstream_u": uf.get("upstream"),
                         "pref": fr.get("pref"), "range_flip": diag.get("range_flip") if dual else None,
                         "verdict": diag.get("verdict") if dual else None})

    print("\n===== Aggregate per CLT x task (pref=Y) =====")
    print(f"  {'CLT':14} {'task':6} {'n':>3} {'recon_fail f→u':>16} "
          f"{'mean range f→u':>18} {'mean upstr f→u':>16} {'flip_rate [95% CI]':>22}")
    agg_records = aggregate_flip_stats(
        rows, samples=args.bootstrap_samples, seed=args.seed)
    for rec in agg_records:
        print(f"  {rec['clt']:14} {rec['task']:6} {rec['n']:>3} "
              f"{rec['recon_fail_f']:>6}/{rec['n']} → {rec['recon_fail_u']:>2}/{rec['n']}   "
              f"{rec['mean_range_f']:>7.1f}→{rec['mean_range_u']:<7.1f} "
              f"{rec['mean_upstream_f']:>6.1f}→{rec['mean_upstream_u']:<6.1f} "
              f"{rec['flip_rate']:>6.1%} [{rec['flip_lo']:.1%}, {rec['flip_hi']:.1%}]")

    print("\nReading: recon_fail = # prompts with negative recoverable_range. If gemma IOI")
    print("goes from N/N (frozen) toward 0/N (unfrozen) with rising upstream features, the")
    print("answer was carried by frozen attention -> confirmed attention-mediated.")

    if rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.csv}")
    if agg_records:
        agg_csv = os.path.join(os.path.dirname(args.csv) or ".",
                               "frozen_vs_unfrozen_agg.csv")
        with open(agg_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(agg_records[0].keys()))
            w.writeheader(); w.writerows(agg_records)
        print(f"Wrote {agg_csv}")


if __name__ == "__main__":
    main()
