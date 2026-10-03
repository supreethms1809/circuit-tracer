#!/usr/bin/env python3
"""B3.2 exact best size-k subset on the Dallas logit-gap PF20 pools.

Recomputes the Game 1 singleton top-20 at each α (same prefilter as the
saved PF20 Game 1 runs), then brute-forces every size-|E*| subset.
Writes a tagged sidecar; does not overwrite Game 1 JSON.

Default: frozen legs only (the application-split claim). Unfrozen k>8 is
skipped (C(20,10) exceeds the default 100k cap unless --max-evaluations is
raised).

Usage:
  python experiments/dallas_bruteforce_pf20.py \\
    --outdir /gscratch/$USER/macag_dallas_austin_llama/llama32-524k/dallas-austin
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from macag.baselines.bruteforce import best_subset_bruteforce
from macag.factories.replacement_model import create_replacement_model_oracle
from macag.games.game1_min_faithful import prefilter_candidates
from macag.graph import CircuitGraph
from macag.scoring import derive_oracle_with_freeze
from macag.utils.metrics import compute_faithfulness_metrics, game1_utility, metrics_to_dict

LOGGER = logging.getLogger(__name__)

GAME1_BY_ALPHA = {
    0.0: "macag_game1_prefilter20_alpha0.json",
    0.5: "macag_game1_prefilter20.json",
    1.0: "macag_game1_prefilter20_alpha1.json",
}


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--score-kind", default="logit_gap")
    parser.add_argument("--prefilter-top-k", type=int, default=20)
    parser.add_argument("--lam", type=float, default=0.02)
    parser.add_argument("--max-evaluations", type=int, default=100_000)
    parser.add_argument("--unfrozen", action="store_true", help="Also brute-force unfrozen legs.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kind_dir = args.outdir / args.score_kind
    graph_path = args.outdir / "graphs" / "dallas-austin.json"
    kwargs_path = kind_dir / "oracle_kwargs.json"
    out_path = args.output_json or (
        kind_dir / f"macag_bruteforce_prefilter{args.prefilter_top_k}.json"
    )
    if out_path.is_file() and not args.force:
        LOGGER.info("skip (exists %s); pass --force to overwrite", out_path)
        return 0
    if not graph_path.is_file() or not kwargs_path.is_file():
        raise SystemExit(f"missing graph/kwargs under {args.outdir}")

    kwargs = json.loads(kwargs_path.read_text())
    graph = CircuitGraph.from_json(graph_path)
    built = create_replacement_model_oracle(**kwargs)
    oracle, cands = built.oracle, list(built.candidates or [])
    backend = getattr(oracle, "backend", None)
    if backend is not None and hasattr(backend, "restrict_universe"):
        backend.restrict_universe(set(cands))
        oracle.clear_cache()

    legs = ("frozen", "unfrozen") if args.unfrozen else ("frozen",)
    payload: dict[str, Any] = {
        "prefilter_top_k": args.prefilter_top_k,
        "score_kind": args.score_kind,
        "max_evaluations": args.max_evaluations,
        "legs_run": list(legs),
        "results": {},
    }

    for alpha, fname in GAME1_BY_ALPHA.items():
        g1_path = kind_dir / fname
        if not g1_path.is_file():
            LOGGER.warning("skip alpha=%s: missing %s", alpha, g1_path)
            continue
        g1 = json.loads(g1_path.read_text())
        for leg in legs:
            freeze = leg == "frozen"
            block = g1.get(leg) or {}
            e_star = [str(n) for n in (block.get("evidence") or {}).get("E_star") or []]
            k = len(e_star)
            if k <= 0:
                LOGGER.warning("skip alpha=%s %s: empty E*", alpha, leg)
                continue
            LOGGER.info(">>> alpha=%s %s |E*|=%d dump PF%d then brute", alpha, leg, k, args.prefilter_top_k)
            leg_oracle = derive_oracle_with_freeze(oracle, freeze_attention=freeze)
            pool = prefilter_candidates(
                graph,
                leg_oracle,
                "y",
                cands,
                alpha,
                args.lam,
                args.prefilter_top_k,
            )
            pool_ids = [str(n) for n in pool]
            game1_metrics = compute_faithfulness_metrics(
                oracle=leg_oracle, target="y", nodes=set(e_star), alpha=alpha
            )
            game1_v = float(game1_metrics.faithfulness_delta)
            game1_u = game1_utility(game1_v, size=k, lam=args.lam)
            try:
                brute = best_subset_bruteforce(
                    leg_oracle,
                    "y",
                    pool,
                    k=k,
                    alpha=alpha,
                    max_evaluations=args.max_evaluations,
                )
            except ValueError as exc:
                LOGGER.warning("bruteforce refused alpha=%s %s k=%d: %s", alpha, leg, k, exc)
                payload["results"][f"alpha{alpha}_{leg}"] = {
                    "alpha": alpha,
                    "leg": leg,
                    "k": k,
                    "pool": pool_ids,
                    "game1_E_star": e_star,
                    "game1_v": game1_v,
                    "game1_utility": game1_u,
                    "game1_scores": metrics_to_dict(game1_metrics),
                    "refused": str(exc),
                }
                continue
            best = [str(n) for n in brute.best_set]
            best_metrics = compute_faithfulness_metrics(
                oracle=leg_oracle, target="y", nodes=set(best), alpha=alpha
            )
            entry = {
                "alpha": alpha,
                "leg": leg,
                "k": k,
                "pool": pool_ids,
                "game1_source": str(g1_path),
                "game1_E_star": sorted(e_star),
                "game1_in_pool": [n for n in e_star if n in set(pool_ids)],
                "game1_v": game1_v,
                "game1_utility": game1_u,
                "game1_scores": metrics_to_dict(game1_metrics),
                "bruteforce_best_set": best,
                "bruteforce_v": brute.best_value,
                "bruteforce_utility": game1_utility(brute.best_value, size=k, lam=args.lam),
                "bruteforce_scores": metrics_to_dict(best_metrics),
                "bruteforce_evaluations": brute.evaluations,
                "bruteforce_ties": brute.ties,
                "optimality_gap_v": brute.best_value - game1_v,
                "jaccard_game1_vs_best": _jaccard(e_star, best),
                "game1_is_exact": set(e_star) == set(best),
            }
            payload["results"][f"alpha{alpha}_{leg}"] = entry
            LOGGER.info(
                "    Game1 v=%.4f brute v=%.4f gap=%.4f Jaccard=%.3f exact=%s",
                game1_v,
                brute.best_value,
                entry["optimality_gap_v"],
                entry["jaccard_game1_vs_best"],
                entry["game1_is_exact"],
            )

    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info(">>> wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
