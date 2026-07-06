"""B2.0 — head-to-head baseline harness over a shared graph + oracle.

Runs every selection method on the SAME candidate node set under the SAME
intervention oracle, so only the selection rule differs (macag.md §9.3 /
Appendix A). For each method it emits nested evidence sets for k = 1..budget
scored with the same FaithfulnessMetrics as the games, plus the comparison
block the paper's core table is built from: faithfulness@matched-k, AUC of the
faithfulness-vs-size curve, oracle-call counts per method, precision@k /
Jaccard against Shapley-gold, and the Spearman linearity diagnostics
(EAP-score vs Game-1 marginal gain).

Example (toy oracle):
    python -m macag.cli.run_baselines \
        --graph-json graph.json --target y --budget 8 \
        --toy-oracle-json toy.json --output-json baselines.json

Example (real interventions):
    python -m macag.cli.run_baselines \
        --graph-json graph.json --target Denver --budget 8 \
        --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle \
        --oracle-kwargs-file oracle_kwargs.json \
        --methods influence,eap,shapley,game1,acdc \
        --output-json baselines.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from macag.baselines.acdc_prune import acdc_target_size, acdc_tau_sweep
from macag.baselines.bruteforce import best_subset_bruteforce
from macag.baselines.common import (
    SelectionResult,
    jaccard,
    precision_at_k,
    spearman_rank_correlation,
)
from macag.baselines.eap import select_top_eap
from macag.baselines.influence import select_top_influence
from macag.baselines.shapley_select import select_top_shapley
from macag.cli.run_macag import _build_oracle, _load_candidates, _load_json, _sort_nodes
from macag.games.game1_min_faithful import solve_game1
from macag.graph import CircuitGraph, NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.utils.metrics import (
    compute_faithfulness_metrics,
    dedupe_preserve_order,
    game1_utility,
    metrics_to_dict,
)

LOGGER = logging.getLogger(__name__)

KNOWN_METHODS = ("influence", "eap", "shapley", "banzhaf", "game1", "acdc")
DEFAULT_METHODS = "influence,eap,shapley,game1,acdc"
DEFAULT_ACDC_TAUS = "0.001,0.01,0.05,0.1,0.2,0.5"
_FEATURE_NODE_TYPE = "cross layer transcoder"


def _parse_methods(spec: str) -> list[str]:
    methods = [method.strip().lower() for method in spec.split(",") if method.strip()]
    unknown = [method for method in methods if method not in KNOWN_METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; choose from {KNOWN_METHODS}.")
    if not methods:
        raise ValueError("Provide at least one method.")
    return dedupe_preserve_order(methods)  # type: ignore[arg-type]


def _parse_float_list(spec: str) -> list[float]:
    return [float(part) for part in spec.split(",") if part.strip()]


def _parse_int_list(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


def _default_candidates(graph: CircuitGraph) -> list[NodeId]:
    """Fall back to the graph's CLT feature nodes (the games' convention)."""
    feature_nodes = [
        node
        for node in graph.nodes()
        if str(graph.metadata(node).get("feature_type", "")).strip().lower() == _FEATURE_NODE_TYPE
    ]
    if feature_nodes:
        return feature_nodes
    LOGGER.warning(
        "No '%s' nodes found; using all %d graph nodes as candidates.",
        _FEATURE_NODE_TYPE,
        len(graph.nodes()),
    )
    return graph.nodes()


def _resolve_candidates(
    args: argparse.Namespace,
    graph: CircuitGraph,
    factory_candidates: list[NodeId] | None,
) -> list[NodeId]:
    candidates = _load_candidates(args.candidates_file) if args.candidates_file else None
    if candidates is None:
        candidates = list(factory_candidates) if factory_candidates is not None else _default_candidates(graph)
    elif factory_candidates is not None:
        valid = set(factory_candidates)
        candidates = [candidate for candidate in candidates if candidate in valid]
        if not candidates:
            raise ValueError(
                "No provided candidates are supported by the oracle backend after filtering."
            )
    candidates = [node for node in dedupe_preserve_order(candidates) if graph.has_node(node)]
    if not candidates:
        raise ValueError("Candidate pool is empty after graph filtering.")
    return candidates


def _run_selection(
    method: str,
    args: argparse.Namespace,
    graph: CircuitGraph,
    payload: dict[str, Any],
    oracle: ScoringOracle,
    target: TargetId,
    candidates: list[NodeId],
) -> tuple[SelectionResult, dict[str, int]]:
    """Run one selector with fresh cache + stats so per-method costs are honest."""
    oracle.clear_cache()
    oracle.reset_stats()

    if method == "influence":
        result = select_top_influence(graph, candidates, use_absolute=not args.influence_signed)
    elif method == "eap":
        result = select_top_eap(
            payload,
            candidates,
            target_match=args.eap_target_match,
            foil_match=args.eap_foil_match,
            use_absolute=not args.eap_signed,
        )
    elif method in ("shapley", "banzhaf"):
        permutations = args.shapley_permutations if method == "shapley" else args.banzhaf_samples
        result = select_top_shapley(
            oracle,
            target,
            candidates,
            alpha=args.alpha,
            permutations=permutations,
            seed=args.shapley_seed,
            antithetic=not args.no_antithetic,
            estimator=method,
            progress=args.progress,
        )
    elif method == "game1":
        game1 = solve_game1(
            graph=graph,
            oracle=oracle,
            target=target,
            candidates=candidates,
            alpha=args.alpha,
            lam=args.lam,
            budget=args.budget,
            prefilter_top_k=args.prefilter_top_k,
            min_gain=args.min_gain,
            connected=args.connected,
            progress=args.progress,
        )
        # solve_game1 resets stats internally; its counters are this method's cost.
        result = SelectionResult(
            method="game1",
            ranking=list(game1.selected_order),
            scores=None,
            params=dict(game1.params),
            extras={
                "iterations": game1.iterations,
                "stopped_early": len(game1.selected_order) < args.budget,
            },
        )
    else:
        raise ValueError(f"_run_selection does not handle method '{method}'.")

    stats = oracle.cache_stats()
    return result, {
        "oracle_calls": stats["oracle_calls"],
        "cache_hits": stats["cache_hits"],
    }


def _evaluate_prefixes(
    oracle: ScoringOracle,
    target: TargetId,
    ranking: Sequence[NodeId],
    budget: int,
    alpha: float,
    lam: float,
) -> dict[int, dict[str, Any]]:
    """Score each k-prefix of a ranking with the games' FaithfulnessMetrics."""
    results: dict[int, dict[str, Any]] = {}
    for k in range(1, min(budget, len(ranking)) + 1):
        evidence = set(ranking[:k])
        metrics = compute_faithfulness_metrics(oracle=oracle, target=target, nodes=evidence, alpha=alpha)
        results[k] = {
            "evidence": _sort_nodes(evidence),
            "scores": metrics_to_dict(metrics)
            | {"utility": game1_utility(metrics.faithfulness_delta, size=k, lam=lam)},
        }
    return results


def _comparison_block(
    methods: dict[str, dict[str, Any]],
    selections: dict[str, SelectionResult],
    budget: int,
    game1_marginals: dict[NodeId, float] | None,
) -> dict[str, Any]:
    faithfulness_at_k: dict[str, dict[str, float]] = {}
    auc_raw: dict[str, float] = {}
    for method, entry in methods.items():
        per_k = {
            str(k): payload["scores"]["faithfulness"] for k, payload in entry["results"].items()
        }
        faithfulness_at_k[method] = per_k
        if per_k:
            # Mean over k = 1..budget with missing sizes treated as the last
            # available prefix would overstate early-stopping methods; average
            # only over realized sizes and report the count.
            auc_raw[method] = sum(per_k.values()) / len(per_k)

    comparison: dict[str, Any] = {
        "faithfulness_at_k": faithfulness_at_k,
        "auc_raw_faithfulness": auc_raw,
    }

    gold = selections.get("shapley") or selections.get("banzhaf")
    if gold is not None:
        agreement: dict[str, dict[str, dict[str, float]]] = {}
        for method, selection in selections.items():
            if method == gold.method:
                continue
            per_k: dict[str, dict[str, float]] = {}
            for k in range(1, budget + 1):
                if k > len(selection.ranking) or k > len(gold.ranking):
                    break
                per_k[str(k)] = {
                    "precision_at_k": precision_at_k(selection.ranking, gold.ranking, k),
                    "jaccard": jaccard(set(selection.ranking[:k]), set(gold.ranking[:k])),
                }
            agreement[method] = per_k
        comparison[f"agreement_vs_{gold.method}"] = agreement

    # Pairwise Jaccard of the budget-size evidence sets.
    pairwise: dict[str, float] = {}
    names = sorted(selections)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            set_a = set(selections[name_a].ranking[:budget])
            set_b = set(selections[name_b].ranking[:budget])
            pairwise[f"{name_a}|{name_b}"] = jaccard(set_a, set_b)
    comparison["pairwise_jaccard_at_budget"] = pairwise

    # Spearman diagnostics over per-node score maps (macag.md §A.5): the
    # EAP-vs-greedy-marginal correlation localizes where local linearity breaks,
    # and method-vs-gold correlations measure ranking agreement beyond top-k.
    spearman: dict[str, float | None] = {}
    scored = {name: sel.scores for name, sel in selections.items() if sel.scores}
    names_scored = sorted(scored)
    for i, name_a in enumerate(names_scored):
        for name_b in names_scored[i + 1 :]:
            spearman[f"{name_a}|{name_b}"] = spearman_rank_correlation(
                scored[name_a], scored[name_b]
            )
    if game1_marginals:
        for name, score_map in scored.items():
            spearman[f"{name}|game1_marginal_gain"] = spearman_rank_correlation(
                score_map, game1_marginals
            )
    comparison["spearman"] = spearman
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline selectors head-to-head against MACAG Game 1 on a shared graph + oracle."
    )
    parser.add_argument("--graph-json", required=True, help="Path to circuit graph JSON.")
    parser.add_argument("--target", required=True, help="Target class/label.")
    parser.add_argument("--input-id", default="unknown", help="Input identifier for output JSON.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Faithfulness mix weight.")
    parser.add_argument("--lam", type=float, default=0.01, help="Sparsity lambda (Game 1 + reported utility).")
    parser.add_argument("--budget", type=int, required=True, help="Max evidence size k.")
    parser.add_argument("--candidates-file", default=None, help="Optional candidate node list (.json or text).")
    parser.add_argument("--output-json", required=True, help="Path to write result JSON.")
    parser.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help=f"Comma list from {KNOWN_METHODS} (default: {DEFAULT_METHODS}).",
    )

    parser.add_argument(
        "--oracle-factory",
        default=None,
        help="Factory path 'module.submodule:function' returning backend scorer or ScoringOracle.",
    )
    parser.add_argument("--oracle-kwargs-json", default=None, help="Inline JSON kwargs for the factory.")
    parser.add_argument("--oracle-kwargs-file", default=None, help="JSON file with factory kwargs.")
    parser.add_argument("--toy-oracle-json", default=None, help="Toy additive oracle JSON (tests).")
    parser.add_argument(
        "--include-error-nodes",
        action="store_true",
        help="Opt-in (C1): forwarded to the oracle factory; see run_macag.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable oracle memoization (inflates per-method oracle-call counts).",
    )

    parser.add_argument("--shapley-permutations", type=int, default=64, help="MC permutations for Shapley.")
    parser.add_argument("--banzhaf-samples", type=int, default=64, help="MC samples for Banzhaf.")
    parser.add_argument("--shapley-seed", type=int, default=0, help="Seed for Shapley/Banzhaf sampling.")
    parser.add_argument("--no-antithetic", action="store_true", help="Disable antithetic permutation pairing.")

    parser.add_argument("--prefilter-top-k", type=int, default=None, help="Game 1 singleton prefilter size.")
    parser.add_argument("--min-gain", type=float, default=0.0, help="Game 1 minimum positive gain.")
    parser.add_argument(
        "--no-connected",
        action="store_false",
        dest="connected",
        help="Let the game1 selector pick disconnected evidence sets. Default keeps the "
        "connectivity constraint so the head-to-head 'game1' is the same method the "
        "run_macag CLI reports (both default to connected).",
    )

    parser.add_argument(
        "--acdc-taus",
        default=DEFAULT_ACDC_TAUS,
        help=f"Comma list of ACDC thresholds to sweep (default: {DEFAULT_ACDC_TAUS}).",
    )
    parser.add_argument(
        "--acdc-order",
        choices=("top_down", "given"),
        default="top_down",
        help="ACDC traversal order (top_down = output side first).",
    )
    parser.add_argument(
        "--acdc-target-k",
        type=int,
        default=None,
        help="Bisect tau so ACDC keeps ~k nodes (budget-matched ACDC); -1 uses --budget. "
        "Adds methods.acdc.matched_k and an acdc entry in comparison.faithfulness_at_k.",
    )

    parser.add_argument(
        "--eap-target-match",
        default=None,
        help="Case-insensitive clerp substring selecting the target logit seed "
        "(default: the graph's is_target_logit flag).",
    )
    parser.add_argument("--eap-foil-match", default=None, help="Clerp substring selecting a -1 foil logit seed.")
    parser.add_argument("--eap-signed", action="store_true", help="Rank by signed EAP score instead of |score|.")
    parser.add_argument("--influence-signed", action="store_true", help="Rank by signed influence instead of |influence|.")

    parser.add_argument(
        "--bruteforce-k",
        default=None,
        help="Optional comma list of sizes for the exact best-subset search (B3.2).",
    )
    parser.add_argument(
        "--bruteforce-max-evals",
        type=int,
        default=100_000,
        help="Refuse brute force beyond this many subsets.",
    )

    parser.set_defaults(progress=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress", help="Disable progress logs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.progress:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    if args.budget < 1:
        raise ValueError("--budget must be >= 1.")
    methods = _parse_methods(args.methods)

    payload = _load_json(args.graph_json)
    graph = CircuitGraph.from_dict(payload)
    oracle, factory_candidates = _build_oracle(args)
    candidates = _resolve_candidates(args, graph, factory_candidates)

    backend = getattr(oracle, "backend", None)
    if backend is not None and hasattr(backend, "restrict_universe"):
        backend.restrict_universe(set(candidates))
        oracle.clear_cache()

    selections: dict[str, SelectionResult] = {}
    method_outputs: dict[str, dict[str, Any]] = {}
    acdc_output: dict[str, Any] | None = None

    for method in methods:
        if method == "acdc":
            oracle.clear_cache()
            oracle.reset_stats()
            sweep = acdc_tau_sweep(
                graph,
                oracle,
                args.target,
                candidates,
                taus=_parse_float_list(args.acdc_taus),
                alpha=args.alpha,
                order=args.acdc_order,
                progress=args.progress,
            )
            stats = oracle.cache_stats()
            acdc_output = {
                "selection_stats": {
                    "oracle_calls": stats["oracle_calls"],
                    "cache_hits": stats["cache_hits"],
                },
                "sweep": [
                    {
                        "tau": result.tau,
                        "size": len(result.kept),
                        "kept": _sort_nodes(set(result.kept)),
                        "value": result.value,
                        "removed_order": [str(node) for node in result.removed_order],
                    }
                    for result in sweep
                ],
            }
            continue

        selection, stats = _run_selection(method, args, graph, payload, oracle, args.target, candidates)
        selections[method] = selection
        method_outputs[method] = {
            "ranking": [str(node) for node in selection.ranking],
            "scores": (
                {str(node): score for node, score in selection.scores.items()}
                if selection.scores
                else None
            ),
            "params": selection.params,
            "extras": selection.extras,
            "selection_stats": stats,
        }

    # Evaluation pass: same FaithfulnessMetrics for every method's prefixes.
    # One shared warm cache — evaluation cost is not a comparison axis.
    oracle.clear_cache()
    oracle.reset_stats()
    for method, selection in selections.items():
        method_outputs[method]["results"] = _evaluate_prefixes(
            oracle, args.target, selection.ranking, args.budget, args.alpha, args.lam
        )
    if acdc_output is not None:
        # ACDC produces one set per tau, not nested prefixes; score each kept
        # set as-is and bucket the best entry per size for matched-|E| reads.
        best_by_size: dict[int, dict[str, Any]] = {}
        for entry in acdc_output["sweep"]:
            kept = set(entry["kept"])
            if not kept:
                continue
            metrics = compute_faithfulness_metrics(
                oracle=oracle, target=args.target, nodes=kept, alpha=args.alpha
            )
            entry["scores"] = metrics_to_dict(metrics) | {
                "utility": game1_utility(metrics.faithfulness_delta, size=len(kept), lam=args.lam)
            }
            size = len(kept)
            current = best_by_size.get(size)
            if current is None or entry["scores"]["faithfulness"] > current["scores"]["faithfulness"]:
                best_by_size[size] = {"evidence": entry["kept"], "scores": entry["scores"], "tau": entry["tau"]}
        acdc_output["best_by_size"] = {str(size): best_by_size[size] for size in sorted(best_by_size)}

        if args.acdc_target_k is not None:
            # Budget-matched ACDC: bisect tau to k on the warm evaluation cache.
            target_k = args.budget if args.acdc_target_k == -1 else args.acdc_target_k
            matched = acdc_target_size(
                graph,
                oracle,
                args.target,
                candidates,
                target_k=target_k,
                alpha=args.alpha,
                order=args.acdc_order,
                seed_results=sweep,
                progress=args.progress,
            )
            matched_metrics = compute_faithfulness_metrics(
                oracle=oracle, target=args.target, nodes=set(matched.kept), alpha=args.alpha
            )
            acdc_output["matched_k"] = {
                "target_k": target_k,
                "achieved_k": matched.params["achieved_k"],
                "exact": matched.params["exact"],
                "tau": matched.tau,
                "bisection_iters": matched.params["bisection_iters"],
                "evidence": _sort_nodes(set(matched.kept)),
                "scores": metrics_to_dict(matched_metrics)
                | {
                    "utility": game1_utility(
                        matched_metrics.faithfulness_delta, size=len(matched.kept), lam=args.lam
                    )
                },
            }

    bruteforce_output: dict[str, Any] | None = None
    if args.bruteforce_k:
        bruteforce_output = {}
        for k in _parse_int_list(args.bruteforce_k):
            result = best_subset_bruteforce(
                oracle,
                args.target,
                candidates,
                k=k,
                alpha=args.alpha,
                max_evaluations=args.bruteforce_max_evals,
            )
            entry: dict[str, Any] = {
                "best_set": _sort_nodes(set(result.best_set)),
                "best_value": result.best_value,
                "evaluations": result.evaluations,
                "ties": result.ties,
            }
            # Optimality gap vs each ranked method's size-k prefix (B3.2).
            gaps: dict[str, float] = {}
            for method, output in method_outputs.items():
                prefix = output.get("results", {}).get(k)
                if prefix is not None:
                    gaps[method] = result.best_value - prefix["scores"]["faithfulness"]
            entry["optimality_gap"] = gaps
            bruteforce_output[str(k)] = entry

    evaluation_stats = oracle.cache_stats()

    # Game 1's per-step marginal faithfulness gains, for the §A.5 Spearman
    # linearity diagnostic against EAP/influence/Shapley scores.
    game1_marginals: dict[NodeId, float] | None = None
    if "game1" in selections and method_outputs["game1"].get("results"):
        game1_marginals = {}
        previous = 0.0
        results = method_outputs["game1"]["results"]
        for k in sorted(results):
            faith = results[k]["scores"]["faithfulness"]
            game1_marginals[selections["game1"].ranking[k - 1]] = faith - previous
            previous = faith

    comparison = _comparison_block(
        {name: output for name, output in method_outputs.items() if output.get("results")},
        selections,
        args.budget,
        game1_marginals,
    )
    if acdc_output is not None and acdc_output.get("best_by_size"):
        # ACDC has no ranked prefixes, so _comparison_block skips it; mirror its
        # per-size faithfulness (and the budget-matched point when computed) into
        # faithfulness_at_k so curve/at-k consumers see every method.
        acdc_at_k = {
            size: block["scores"]["faithfulness"]
            for size, block in acdc_output["best_by_size"].items()
        }
        matched = acdc_output.get("matched_k")
        if matched is not None:
            acdc_at_k[str(matched["achieved_k"])] = matched["scores"]["faithfulness"]
        comparison["faithfulness_at_k"]["acdc"] = acdc_at_k

    output: dict[str, Any] = {
        "input_id": args.input_id,
        "target": args.target,
        "game": "baselines",
        "params": {
            "alpha": args.alpha,
            "lambda": args.lam,
            "budget": args.budget,
            "methods": methods,
            "shapley_permutations": args.shapley_permutations,
            "banzhaf_samples": args.banzhaf_samples,
            "shapley_seed": args.shapley_seed,
            "antithetic": not args.no_antithetic,
            "acdc_taus": _parse_float_list(args.acdc_taus),
            "acdc_order": args.acdc_order,
            "acdc_target_k": args.acdc_target_k,
            "game1_connected": args.connected,
            "prefilter_top_k": args.prefilter_top_k,
            "candidate_count": len(candidates),
        },
        "candidates": [str(node) for node in candidates],
        "methods": {
            name: {
                **output,
                "results": {str(k): payload for k, payload in output.get("results", {}).items()},
            }
            for name, output in method_outputs.items()
        },
        "comparison": comparison,
        "stats": {"evaluation_oracle_calls": evaluation_stats["oracle_calls"]},
    }
    if acdc_output is not None:
        output["methods"]["acdc"] = acdc_output
    if bruteforce_output is not None:
        output["bruteforce"] = bruteforce_output

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    if args.progress:
        LOGGER.info("Wrote baseline head-to-head results to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
