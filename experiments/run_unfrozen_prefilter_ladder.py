#!/usr/bin/env python3
"""Unfrozen Game 1 ladder: rank singletons once, then greedy at several k.

Does not restrict the intervention universe to the top-k pool. Prefilter only
limits which nodes greedy may add. Isolated outputs; does not overwrite the
dual-freeze PF10–1000 sweep.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from macag.cli.run_macag import _game1_leg_payload
from macag.factories.replacement_model import create_replacement_model_oracle
from macag.games.game1_min_faithful import solve_game1
from macag.graph import CircuitGraph, NodeId
from macag.kl_rescore import rescore_run_dir
from macag.scoring import TargetId, derive_oracle_with_freeze
from macag.utils.metrics import compute_faithfulness_metrics, game1_utility

DEFAULT_KS = (1500, 2000, 2500, 3000)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _estar(payload: dict[str, Any], leg: str) -> list[str]:
    block = payload.get(leg) if payload.get("freeze_mode") == "both" else payload
    return [str(n) for n in (block.get("evidence") or {}).get("E_star") or []]


def _rank_unfrozen(
    oracle: Any,
    candidates: Sequence[NodeId],
    *,
    alpha: float,
    lam: float,
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for node in tqdm(candidates, desc="unfrozen singleton rank"):
        metrics = compute_faithfulness_metrics(
            oracle=oracle, target="y", nodes={node}, alpha=alpha
        )
        utility = game1_utility(metrics.faithfulness_delta, size=1, lam=lam)
        ranking.append(
            {
                "node": str(node),
                "utility": float(utility),
                "F": float(metrics.faithfulness_delta),
            }
        )
    ranking.sort(key=lambda row: (-row["utility"], row["node"]))
    for i, row in enumerate(ranking, start=1):
        row["rank"] = i
    return ranking


def _prefilter_fn(ranked_nodes: list[NodeId]):
    def _fn(
        graph: CircuitGraph,
        oracle: Any,
        target: TargetId,
        candidates: Sequence[NodeId],
        alpha: float,
        lam: float,
        top_k: int,
    ) -> list[NodeId]:
        allowed = {str(n) for n in candidates if graph.has_node(n)}
        kept: list[NodeId] = []
        for node in ranked_nodes:
            if str(node) in allowed:
                kept.append(node)
            if len(kept) >= top_k:
                break
        return kept

    return _fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--oracle-kwargs", type=Path, required=True)
    parser.add_argument("--unfiltered-game1", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=0.02)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--ranking-json", type=Path, default=None)
    args = parser.parse_args()

    kwargs = _load(args.oracle_kwargs)
    base = _load(args.unfiltered_game1)
    baseline_e = _estar(base, "unfrozen")
    baseline_f = float((base.get("unfrozen") or base).get("scores", {}).get("faithfulness") or 0.0)
    graph = CircuitGraph.from_json(str(args.graph))
    built = create_replacement_model_oracle(**kwargs)
    oracle, cands = built.oracle, list(built.candidates)
    backend = getattr(oracle, "backend", None)
    if backend is not None and hasattr(backend, "restrict_universe"):
        backend.restrict_universe(set(cands))
        oracle.clear_cache()
    unfrozen = derive_oracle_with_freeze(oracle, freeze_attention=False)

    ranking_path = args.ranking_json or (args.outdir / "unfrozen_singleton_ranking.json")
    if ranking_path.is_file():
        print(f">>> reusing ranking {ranking_path}", flush=True)
        ranking = _load(ranking_path)["ranking"]
    else:
        print(f">>> ranking {len(cands)} unfrozen singletons", flush=True)
        ranking = _rank_unfrozen(unfrozen, cands, alpha=args.alpha, lam=args.lam)
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        ranking_path.write_text(
            json.dumps(
                {
                    "n_candidates": len(cands),
                    "alpha": args.alpha,
                    "lambda": args.lam,
                    "ranking": ranking,
                }
            )
            + "\n"
        )
        print(f">>> wrote {ranking_path}", flush=True)

    ranked_nodes = [row["node"] for row in ranking]
    rank_of = {row["node"]: row["rank"] for row in ranking}
    cutoff_by_k = {}
    for k in args.ks:
        k_use = min(int(k), len(ranking))
        cutoff_by_k[int(k)] = {
            "k": k_use,
            "cutoff_utility": ranking[k_use - 1]["utility"] if k_use else None,
            "n_baseline_in_pool": sum(1 for n in baseline_e if rank_of.get(n, 10**9) <= k_use),
            "baseline_in_pool": [n for n in baseline_e if rank_of.get(n, 10**9) <= k_use],
            "baseline_out": [n for n in baseline_e if rank_of.get(n, 10**9) > k_use],
            "baseline_ranks": {n: rank_of.get(n) for n in baseline_e},
        }

    prefilter = _prefilter_fn(ranked_nodes)
    for k in args.ks:
        kind_dir = args.outdir / f"unfrozen_pf{k}" / "logit_gap"
        out_path = kind_dir / "macag_game1.json"
        kind_dir.mkdir(parents=True, exist_ok=True)
        kwargs_out = dict(kwargs)
        kwargs_out["freeze_attention"] = False
        (kind_dir / "oracle_kwargs.json").write_text(json.dumps(kwargs_out, indent=2) + "\n")
        if out_path.is_file():
            print(f">>> skip existing {out_path}", flush=True)
            continue
        print(f">>> unfrozen Game 1 prefilter_top_k={k}", flush=True)
        start = time.time()
        result = solve_game1(
            graph=graph,
            oracle=unfrozen,
            target="y",
            candidates=cands,
            alpha=args.alpha,
            lam=args.lam,
            budget=None,
            faithfulness_eps=args.eps,
            stop_metric="raw_relative",
            prefilter_top_k=int(k),
            prefilter_fn=prefilter,
            connected=False,
            progress=True,
        )
        payload = {
            "input_id": f"{args.slug}-unfrozen-pf{k}",
            "target": "y",
            "foil": None,
            "game": "game1",
            "freeze_mode": "unfrozen",
            **_game1_leg_payload(result, freeze_attention=False),
        }
        payload["params"]["prefilter_top_k"] = int(k)
        payload["pool_vs_unfiltered"] = cutoff_by_k[int(k)]
        payload["wall_seconds"] = int(time.time() - start)
        selected = [str(n) for n in (payload["evidence"]["E_star"] or [])]
        base_set = set(baseline_e)
        sel_set = set(selected)
        payload["vs_unfiltered_unfrozen"] = {
            "unfiltered_F": baseline_f,
            "filtered_F": payload["scores"].get("faithfulness"),
            "retention": (
                float(payload["scores"]["faithfulness"]) / baseline_f if baseline_f else None
            ),
            "shared": sorted(base_set & sel_set),
            "dropped": sorted(base_set - sel_set),
            "new": sorted(sel_set - base_set),
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f">>> wrote {out_path} in {payload['wall_seconds']}s", flush=True)
        rescore_run_dir(kind_dir, force=True)
        print(f">>> KL rescored {kind_dir}", flush=True)

    summary = {
        "slug": args.slug,
        "n_candidates": len(cands),
        "ks": list(args.ks),
        "unfiltered_unfrozen_E_star": baseline_e,
        "unfiltered_unfrozen_F": baseline_f,
        "baseline_ranks": {n: rank_of.get(n) for n in baseline_e},
        "cutoffs": cutoff_by_k,
        "ranking_json": str(ranking_path),
    }
    summary_path = args.outdir / "unfrozen_ladder_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f">>> wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
