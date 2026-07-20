#!/usr/bin/env python
"""Aggregate head-to-head baseline results from the ACDC MACAG runs.

Reads results/macag_acdc/<clt_tag>/<slug>/macag_baselines.json (produced by
``scripts/run_macag_pipeline.sh`` via ``run_macag_acdc.sh``). Reports
faithfulness@budget, faithfulness AUC, precision@k vs Shapley-gold, and oracle
call counts per method, grouped by task.
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
METHODS = ("influence", "eap", "shapley", "game1", "acdc")


def slug_to_task(bench_path: str) -> dict[str, str]:
    d = json.load(open(bench_path))
    return {it["id"]: task for task, items in d["tasks"].items() for it in items}


def _budget_k(payload: dict[str, Any]) -> int:
    return int(payload.get("params", {}).get("budget", 8))


def _faith_at_own_k(curve: dict[str, Any], budget: int) -> tuple[Optional[float], Optional[int]]:
    """Faithfulness of a method's OWN final selected set.

    ``curve`` is the prefix-cumulative faithfulness_at_k map {"1": v, "2": v, ...}.
    Methods that stop early (e.g. Game 1's raw_relative stop) only populate keys up
    to their selected size |E|, so scoring at the global budget k under-counts them.
    The method's own selected size is the largest key present (capped at budget).
    """
    ks = sorted(int(x) for x in curve)
    if not ks:
        return None, None
    capped = [k for k in ks if k <= budget]
    own = max(capped) if capped else min(ks)
    return curve.get(str(own)), own


def read(run_dir: str) -> Optional[dict[str, Any]]:
    path = os.path.join(run_dir, "macag_baselines.json")
    if not os.path.isfile(path):
        return None
    payload = json.load(open(path))
    budget = _budget_k(payload)
    k = str(budget)
    comp = payload.get("comparison", {})
    faith_at_k = comp.get("faithfulness_at_k", {})
    auc = comp.get("auc_raw_faithfulness", {})
    agreement = comp.get("agreement_vs_shapley", {})

    rec: dict[str, Any] = {"budget": budget}
    for method in METHODS:
        curve = faith_at_k.get(method, {})
        # Fair primary metric: faithfulness at the method's OWN selected size, so
        # an early-stopping method (Game 1) is not penalised against fixed-budget
        # ones. faith_budget keeps the at-budget value for reference (may be None).
        faith_own, k_own = _faith_at_own_k(curve, budget)
        rec[f"faith_{method}"] = faith_own
        rec[f"k_{method}"] = k_own
        rec[f"faith_budget_{method}"] = curve.get(k)
        # Efficiency: faithfulness gained per selected feature (the parsimony win).
        rec[f"fpf_{method}"] = (faith_own / k_own) if (faith_own is not None and k_own) else None
        rec[f"auc_{method}"] = auc.get(method)
        rec[f"oracle_{method}"] = (
            payload.get("methods", {}).get(method, {}).get("selection_stats", {}).get("oracle_calls")
        )
        if method != "shapley" and method in agreement:
            rec[f"prec_{method}"] = agreement[method].get(k, {}).get("precision_at_k")
            rec[f"jac_{method}"] = agreement[method].get(k, {}).get("jaccard")
        elif method == "shapley":
            rec["prec_shapley"] = 1.0
            rec["jac_shapley"] = 1.0

    # ACDC reports best-by-size, not nested prefixes. Prefer the budget bucket;
    # fall back to the largest available size so it is never spuriously blank.
    acdc = payload.get("methods", {}).get("acdc", {})
    best_by_size = acdc.get("best_by_size", {}) or {}
    best = best_by_size.get(k)
    if best is None and best_by_size:
        largest = max(best_by_size, key=lambda s: int(s))
        best = best_by_size.get(largest)
        rec["k_acdc"] = int(largest)
    elif best is not None:
        rec["k_acdc"] = budget
    if best is not None:
        rec["faith_acdc"] = best.get("scores", {}).get("faithfulness")
        if rec.get("k_acdc"):
            rec["fpf_acdc"] = rec["faith_acdc"] / rec["k_acdc"]
    rec["oracle_acdc"] = acdc.get("selection_stats", {}).get("oracle_calls")

    kl_path = os.path.join(run_dir, "macag_kl_faithfulness.json")
    if os.path.isfile(kl_path):
        kl_payload = json.load(open(kl_path))
        kl_methods = (kl_payload.get("baselines") or {}).get("methods") or {}
        for method in METHODS:
            if method in kl_methods:
                kl_faith = (kl_methods[method].get("kl_divergence") or {}).get("faithfulness")
            else:
                embedded = (
                    (payload.get("methods") or {}).get(method, {}).get("kl_faithfulness") or {}
                ).get("faithfulness")
                kl_faith = embedded
            if isinstance(kl_faith, (int, float)):
                rec[f"kl_faith_{method}"] = kl_faith
    else:
        for method in METHODS:
            embedded = (
                (payload.get("methods") or {}).get(method, {}).get("kl_faithfulness") or {}
            ).get("faithfulness")
            if isinstance(embedded, (int, float)):
                rec[f"kl_faith_{method}"] = embedded
    return rec


def fnum(x: Any, nd: int = 2) -> str:
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else "--"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/macag_acdc")
    ap.add_argument("--csv", default="results/macag_acdc/baselines.csv")
    ap.add_argument("--bench", default=BENCH)
    args = ap.parse_args()
    task_of = slug_to_task(args.bench)

    rows: list[dict[str, Any]] = []
    for tag in CLTS:
        base = os.path.join(args.root, tag)
        if not os.path.isdir(base):
            continue
        slugs = sorted(
            s for s in os.listdir(base)
            if os.path.isfile(os.path.join(base, s, "macag_baselines.json"))
        )
        if not slugs:
            continue
        print(f"\n===== {tag} =====  (faith@own-|E|; g1|E| = Game 1 selected size)")
        print(
            f"  {'slug':14} {'task':14} "
            f"{'faith_g1':>9} {'g1|E|':>5} {'fpf_g1':>7} {'faith_sh':>9} {'sh|E|':>5} "
            f"{'faith_eap':>9} {'prec_g1':>8} {'oracle_sh':>10}"
        )
        for slug in slugs:
            r = read(os.path.join(base, slug))
            if r is None:
                continue
            task = task_of.get(slug, "?")
            print(
                f"  {slug:14} {task[:14]:14} "
                f"{fnum(r.get('faith_game1')):>9} {str(r.get('k_game1', '-')):>5} "
                f"{fnum(r.get('fpf_game1')):>7} "
                f"{fnum(r.get('faith_shapley')):>9} {str(r.get('k_shapley', '-')):>5} "
                f"{fnum(r.get('faith_eap')):>9} "
                f"{fnum(r.get('prec_game1'), 2):>8} {str(r.get('oracle_shapley', '-')):>10}"
            )
            rows.append({"clt": tag, "task": task, "slug": slug, **r})

    print("\n===== Aggregate per CLT x task (mean faith@own-|E|  /  mean faith-per-feature) =====")
    print(
        f"  {'CLT':14} {'task':24} {'n':>3} "
        f"{'game1':>14} {'shapley':>14} {'eap':>14} {'infl':>14} {'acdc':>14}"
    )
    agg: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        agg[(r["clt"], r["task"])].append(r)

    def mean(xs: list[Any]) -> float:
        vals = [x for x in xs if isinstance(x, (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    def cell(rs: list[dict[str, Any]], method: str) -> str:
        f = mean([r.get(f"faith_{method}") for r in rs])
        e = mean([r.get(f"fpf_{method}") for r in rs])
        fs = f"{f:+.2f}" if f == f else "--"        # NaN check
        es = f"{e:+.2f}" if e == e else "--"
        return f"{fs}/{es}"

    for (tag, task), rs in sorted(agg.items()):
        print(
            f"  {tag:14} {task[:24]:24} {len(rs):>3} "
            f"{cell(rs, 'game1'):>14} {cell(rs, 'shapley'):>14} {cell(rs, 'eap'):>14} "
            f"{cell(rs, 'influence'):>14} {cell(rs, 'acdc'):>14}"
        )

    print("\nReading: faith_* is faithfulness at each method's OWN final |E| (fair to")
    print("early-stopping Game 1); faith_budget_* is the at-budget value. fpf = faith")
    print("per feature (efficiency). Game 1 reaching comparable faith at smaller |E| /")
    print("higher fpf with ~100x fewer oracle calls than Shapley is the parsimony win.")

    if rows:
        cols = ["clt", "task", "slug", "budget"]
        for method in METHODS:
            cols.extend([f"faith_{method}", f"k_{method}", f"faith_budget_{method}",
                         f"fpf_{method}", f"auc_{method}", f"oracle_{method}",
                         f"kl_faith_{method}"])
            if method != "shapley":
                cols.extend([f"prec_{method}", f"jac_{method}"])
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
