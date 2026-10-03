#!/usr/bin/env python3
"""Dump Game 1 singleton ranks for unfiltered E* vs top-k prefilter cutoffs.

The PF Game 1 JSON files store the greedy solution, not the ranked pool.
This re-scores every factory candidate as a singleton (same utility as
prefilter_candidates) and reports, for each freeze leg and each k, whether
each unfiltered E* node would have entered the greedy pool.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from macag.factories.replacement_model import create_replacement_model_oracle
from macag.graph import CircuitGraph
from macag.scoring import derive_oracle_with_freeze
from macag.utils.metrics import compute_faithfulness_metrics, game1_utility

LEGS = (("frozen", True), ("unfrozen", False))
DEFAULT_KS = (10, 20, 50, 100, 500, 1000)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _estar(payload: dict[str, Any], leg: str) -> list[str]:
    return [str(n) for n in (payload.get(leg) or {}).get("evidence", {}).get("E_star") or []]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--oracle-kwargs", type=Path, required=True)
    parser.add_argument("--unfiltered-game1", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=0.02)
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="Prefilter cutoffs to test (default: 10 20 50 100 500 1000).",
    )
    args = parser.parse_args()

    kwargs = _load(args.oracle_kwargs)
    base = _load(args.unfiltered_game1)
    graph = CircuitGraph.from_json(str(args.graph))
    built = create_replacement_model_oracle(**kwargs)
    oracle, cands = built.oracle, list(built.candidates)
    backend = getattr(oracle, "backend", None)
    if backend is not None and hasattr(backend, "restrict_universe"):
        backend.restrict_universe(set(cands))
        oracle.clear_cache()

    payload: dict[str, Any] = {
        "graph": str(args.graph),
        "oracle_kwargs": str(args.oracle_kwargs),
        "unfiltered_game1": str(args.unfiltered_game1),
        "alpha": args.alpha,
        "lambda": args.lam,
        "score_kind": kwargs.get("score_kind"),
        "n_factory_candidates": len(cands),
        "ks": list(args.ks),
        "prefilter_rule": (
            "Rank by singleton Game-1 utility "
            "u({n}) = F({n}) - lambda, keep top-k. "
            "F is alpha-mixed logit-gap faithfulness of that one node."
        ),
        "legs": {},
    }

    for leg, freeze in LEGS:
        print(f">>> ranking leg={leg} freeze_attention={freeze} n={len(cands)}", flush=True)
        leg_oracle = derive_oracle_with_freeze(oracle, freeze_attention=freeze)
        ranking: list[tuple[float, str, float]] = []
        for node in tqdm(cands, desc=f"prefilter-{leg}"):
            metrics = compute_faithfulness_metrics(
                oracle=leg_oracle, target="y", nodes={node}, alpha=args.alpha
            )
            utility = game1_utility(metrics.faithfulness_delta, size=1, lam=args.lam)
            ranking.append((float(utility), str(node), float(metrics.faithfulness_delta)))
        ranking.sort(key=lambda t: (-t[0], t[1]))
        rank_of = {n: i + 1 for i, (_, n, _) in enumerate(ranking)}
        util_of = {n: u for u, n, _ in ranking}
        f_of = {n: f for _, n, f in ranking}
        baseline = _estar(base, leg)
        cutoffs: dict[str, Any] = {}
        for k in args.ks:
            k_use = min(int(k), len(ranking))
            top_ids = {n for _, n, _ in ranking[:k_use]}
            kept = [n for n in baseline if n in top_ids]
            dropped = [n for n in baseline if n not in top_ids]
            cutoffs[str(k)] = {
                "k": k_use,
                "cutoff_utility": ranking[k_use - 1][0] if k_use else None,
                "cutoff_node": ranking[k_use - 1][1] if k_use else None,
                "baseline_in_pool": kept,
                "baseline_prefiltered_out": dropped,
                "n_in_pool": len(kept),
                "n_out": len(dropped),
                "recall": (len(kept) / len(baseline) if baseline else 1.0),
            }
            print(
                f">>> {leg} k={k}: in={len(kept)}/{len(baseline)} out={dropped}",
                flush=True,
            )
        payload["legs"][leg] = {
            "n_candidates": len(cands),
            "baseline_E_star": baseline,
            "baseline_singleton_ranks": {n: rank_of.get(n) for n in baseline},
            "baseline_singleton_utilities": {n: util_of.get(n) for n in baseline},
            "baseline_singleton_F": {n: f_of.get(n) for n in baseline},
            "cutoffs": cutoffs,
            "top_20": [
                {"rank": i + 1, "node": n, "singleton_utility": u, "singleton_F": f}
                for i, (u, n, f) in enumerate(ranking[:20])
            ],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(">>> wrote", args.output_json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
