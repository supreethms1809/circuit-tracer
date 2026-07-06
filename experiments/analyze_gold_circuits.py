#!/usr/bin/env python
"""Score MACAG evidence against the (layer-band, token-role) IOI gold spec (B4.1).

Walks ``<root>/<clt>/<slug>/`` for the selected task, assigns token roles from
the benchmark manifest (metadata S1/IO; heuristic fallback otherwise), and
scores each Game 1 evidence leg — plus, with ``--include-baselines``, every
baseline selector's chosen set — for precision / component-recall / F1 against
``macag.eval.gold_circuits.IOI_GOLD``.

Outputs ``<root>/gold_circuits.csv`` (per prompt × leg × method) and prints a
per-(clt, leg, method) aggregate with bootstrap CIs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from macag.eval.gold_circuits import IOI_GOLD, assign_token_roles, score_evidence_against_gold
from macag.graph import CircuitGraph
from spline_clt.paper.reporting import bootstrap_mean_ci

BENCH = "macag/data/acdc_benchmark_prompts.json"
CLTS = ["gemma2-426k", "gemma2-2.5M", "llama32-524k"]
BASELINE_METHODS = ("influence", "eap", "shapley", "acdc")


def _bench_items(bench_path: str, task: str) -> dict[str, dict[str, Any]]:
    d = json.load(open(bench_path))
    return {it["id"]: it for it in d.get("tasks", {}).get(task, [])}


def _game1_legs(g1: dict[str, Any]) -> dict[str, list[str]]:
    if g1.get("freeze_mode") == "both":
        return {
            leg: (g1.get(leg, {}).get("evidence", {}) or {}).get("E_star") or []
            for leg in ("frozen", "unfrozen")
        }
    return {"single": (g1.get("evidence", {}) or {}).get("E_star") or []}


def _baseline_evidence(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Each baseline's own selected set at (up to) the budget."""
    budget = int(payload.get("params", {}).get("budget", 8))
    out: dict[str, list[str]] = {}
    for method in BASELINE_METHODS:
        entry = payload.get("methods", {}).get(method)
        if not entry:
            continue
        if method == "acdc":
            matched = entry.get("matched_k")
            if matched and matched.get("evidence"):
                out[method] = list(matched["evidence"])
                continue
            best = entry.get("best_by_size") or {}
            if best:
                sizes = sorted(int(s) for s in best)
                capped = [s for s in sizes if s <= budget]
                pick = str(max(capped) if capped else min(sizes))
                out[method] = list(best[pick].get("evidence") or [])
            continue
        results = entry.get("results") or {}
        if not results:
            continue
        ks = sorted(int(k) for k in results)
        capped = [k for k in ks if k <= budget]
        own = str(max(capped) if capped else min(ks))
        out[method] = list(results[own].get("evidence") or [])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/macag_acdc")
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--task", default="indirect_object_identification")
    ap.add_argument("--csv", default=None, help="default: <root>/gold_circuits.csv")
    ap.add_argument("--include-baselines", action="store_true",
                    help="also score each baseline selector's evidence set")
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.csv is None:
        args.csv = os.path.join(args.root, "gold_circuits.csv")

    items = _bench_items(args.bench, args.task)
    if not items:
        raise SystemExit(f"no '{args.task}' items in {args.bench}")

    rows: list[dict[str, Any]] = []
    for tag in CLTS:
        base = Path(args.root) / tag
        if not base.is_dir():
            continue
        for slug, item in sorted(items.items()):
            run_dir = base / slug
            g1_path = run_dir / "macag_game1.json"
            graph_path = run_dir / "graphs" / f"{slug}.json"
            if not g1_path.is_file() or not graph_path.is_file():
                continue
            graph_payload = json.loads(graph_path.read_text())
            graph = CircuitGraph.from_dict(graph_payload)
            prompt_tokens = graph_payload.get("metadata", {}).get("prompt_tokens") or []
            roles = assign_token_roles(prompt_tokens, item.get("metadata"))

            evidence_sets: dict[tuple[str, str], list[str]] = {}
            for leg, evidence in _game1_legs(json.loads(g1_path.read_text())).items():
                evidence_sets[(leg, "game1")] = evidence
            if args.include_baselines:
                bl_path = run_dir / "macag_baselines.json"
                if bl_path.is_file():
                    for method, evidence in _baseline_evidence(
                        json.loads(bl_path.read_text())
                    ).items():
                        evidence_sets[("frozen", method)] = evidence

            for (leg, method), evidence in evidence_sets.items():
                if not evidence:
                    continue
                try:
                    score = score_evidence_against_gold(graph, evidence, roles, IOI_GOLD)
                except ValueError as err:
                    print(f"skip {tag}/{slug} ({leg}/{method}): {err}")
                    continue
                row: dict[str, Any] = {
                    "clt": tag, "task": args.task, "slug": slug, "leg": leg,
                    "method": method, "roles_resolved": len(roles),
                    **{k: v for k, v in score.to_dict().items() if k != "component_hits"},
                }
                for component in IOI_GOLD:
                    row[f"hits_{component.name}"] = score.component_hits[component.name]
                rows.append(row)

    if not rows:
        raise SystemExit(f"no scoreable runs for task '{args.task}' under {args.root}")

    print(f"\n===== Gold-circuit aggregate ({args.task}) =====")
    print(f"  {'CLT':14} {'leg':9} {'method':10} {'n':>3} "
          f"{'precision [95% CI]':>22} {'recall [95% CI]':>22} {'f1':>6}")
    agg: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        agg[(row["clt"], row["leg"], row["method"])].append(row)
    kwargs = (args.bootstrap_samples, 0.95, args.seed)
    for (tag, leg, method), rs in sorted(agg.items()):
        def ci(key: str) -> tuple[float, float, float]:
            vals = [r[key] for r in rs]
            lo, hi = bootstrap_mean_ci(vals, *kwargs)
            return sum(vals) / len(vals), lo, hi
        p, p_lo, p_hi = ci("precision")
        r, r_lo, r_hi = ci("recall")
        f1_mean, _, _ = ci("f1")
        print(f"  {tag:14} {leg:9} {method:10} {len(rs):>3} "
              f"{p:>7.2f} [{p_lo:.2f}, {p_hi:.2f}] "
              f"{r:>7.2f} [{r_lo:.2f}, {r_hi:.2f}] {f1_mean:>6.2f}")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
