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
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from macag.baselines.acdc_prune import (
    ACDCBudgetUnreachableError,
    acdc_target_size,
    acdc_tau_sweep,
)
from macag.baselines.acdc_native import run_acdc_native
from macag.baselines.bruteforce import best_subset_bruteforce
from macag.baselines.common import (
    SelectionResult,
    jaccard,
    precision_at_k,
    precision_at_k_uncertainty_bounds,
    spearman_rank_correlation,
    tie_aware_precision_at_k,
)
from macag.baselines.eap import EAPUnavailableError, select_top_eap
from macag.baselines.eap_syed import select_top_eap_syed
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

# Canonical IDs. Aliases resolve before dispatch so historical JSON keys
# (`eap`, `acdc`) stay stable — see macag/docs/baseline_method_map.md.
METHOD_ALIASES: dict[str, str] = {
    "eap_graph": "eap",
    "acdc_ported": "acdc",
    "eap_ap": "eap_syed",
    "attribution_patching": "eap_syed",
}
CANONICAL_METHODS = (
    "influence",
    "eap",
    "eap_syed",
    "shapley",
    "banzhaf",
    "game1",
    "acdc",
    "acdc_native",
)
KNOWN_METHODS = CANONICAL_METHODS + tuple(METHOD_ALIASES.keys())
DEFAULT_METHODS = "influence,eap,shapley,game1,acdc"
DEFAULT_ACDC_TAUS = "0.001,0.01,0.05,0.1,0.2,0.5"
_FEATURE_NODE_TYPE = "cross layer transcoder"


def _canonicalize_method(name: str) -> str:
    return METHOD_ALIASES.get(name, name)


def _parse_methods(spec: str) -> list[str]:
    raw = [method.strip().lower() for method in spec.split(",") if method.strip()]
    unknown = [method for method in raw if method not in KNOWN_METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; choose from {KNOWN_METHODS}.")
    if not raw:
        raise ValueError("Provide at least one method.")
    # Resolve aliases, preserve first-seen canonical order.
    return dedupe_preserve_order([_canonicalize_method(method) for method in raw])  # type: ignore[arg-type]


def _parse_float_list(spec: str) -> list[float]:
    return [float(part) for part in spec.split(",") if part.strip()]


def _parse_int_list(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


def _oracle_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if getattr(args, "oracle_kwargs_json", None):
        kwargs.update(json.loads(args.oracle_kwargs_json))
    if getattr(args, "oracle_kwargs_file", None):
        kwargs.update(_load_json(args.oracle_kwargs_file))
    return kwargs


def _sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _experiment_identity(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    candidates: Sequence[NodeId],
) -> dict[str, Any]:
    """Immutable identity used to reject incompatible deferred baseline merges."""
    kwargs = _oracle_kwargs(args)
    graph_hash = _sha256_json(payload)
    candidate_ids = [str(node) for node in candidates]
    return {
        "schema_version": 1,
        "input_id": args.input_id,
        "target": args.target,
        "graph_sha256": graph_hash,
        "oracle_kwargs_sha256": _sha256_json(kwargs),
        "candidates_sha256": _sha256_json(candidate_ids),
        "score_kind": kwargs.get("score_kind"),
        "model_name": kwargs.get("model_name"),
        "local_clt_path": kwargs.get("local_clt_path"),
        "transcoder_set": kwargs.get("transcoder_set"),
        "clt_scan": kwargs.get("clt_scan"),
        "freeze_attention": kwargs.get("freeze_attention"),
        "ablation_mode": kwargs.get("ablation_mode", kwargs.get("ablation_kind")),
    }


def _replacement_backend(oracle: ScoringOracle) -> Any:
    backend = getattr(oracle, "backend", None)
    if backend is None or not hasattr(backend, "model") or not hasattr(backend, "node_to_intervention"):
        raise ValueError(
            "eap_syed requires a ReplacementModelInterventionScorer backend "
            "(--oracle-factory with a real ReplacementModel)."
        )
    return backend


def _resolve_corrupted_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "eap_corrupted_prompt", None):
        return str(args.eap_corrupted_prompt)
    kwargs = _oracle_kwargs(args)
    corrupted = kwargs.get("corrupted_prompt")
    if corrupted:
        return str(corrupted)
    raise ValueError(
        "eap_syed needs a corrupted prompt: pass --eap-corrupted-prompt or set "
        "corrupted_prompt in --oracle-kwargs-file."
    )


def _resolve_eap_logit_matches(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Resolve graph-EAP target/foil clerp substrings from CLI or oracle kwargs.

    Explicit ``--eap-target-match`` / ``--eap-foil-match`` win. Otherwise derive
    token strings from ``target_token_by_label`` + ``foil_by_target`` so EAP
    optimizes the same target–foil gap as the evaluation oracle, not merely the
    graph's top predicted logit (``is_target_logit``).
    """
    target_match = getattr(args, "eap_target_match", None)
    foil_match = getattr(args, "eap_foil_match", None)
    if target_match is not None and foil_match is not None:
        return target_match, foil_match

    kwargs = _oracle_kwargs(args)
    token_by_label = kwargs.get("target_token_by_label") or {}
    foil_by_target = kwargs.get("foil_by_target") or {}
    if not isinstance(token_by_label, Mapping):
        token_by_label = {}
    if not isinstance(foil_by_target, Mapping):
        foil_by_target = {}

    if target_match is None:
        token = token_by_label.get(args.target)
        # Preserve leading tokenizer spaces (e.g. " A"); EAP matches the quoted
        # logit token, not a stripped substring of the full clerp.
        if token is not None and str(token).strip():
            target_match = str(token)

    if foil_match is None:
        foil_label = foil_by_target.get(args.target)
        if foil_label is not None:
            foil_token = token_by_label.get(foil_label)
            if foil_token is not None and str(foil_token).strip():
                foil_match = str(foil_token)

    return target_match, foil_match


def _load_native_hooked_transformer(model_name: str, model_kwargs: Mapping[str, Any] | None) -> Any:
    """Load a HookedTransformer for acdc_native (separate from the CLT ReplacementModel)."""
    from transformer_lens import HookedTransformer

    kwargs = dict(model_kwargs or {})
    # TransformerLens uses `dtype` / `device` in from_pretrained.
    model = HookedTransformer.from_pretrained(model_name, **kwargs)
    model.cfg.use_attn_result = True
    return model


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
        target_match, foil_match = _resolve_eap_logit_matches(args)
        result = select_top_eap(
            payload,
            candidates,
            target_match=target_match,
            foil_match=foil_match,
            use_absolute=not args.eap_signed,
        )
    elif method == "eap_syed":
        backend = _replacement_backend(oracle)
        kwargs = _oracle_kwargs(args)
        foil_map = getattr(backend, "foil_by_target", None) or {}
        foil_label = foil_map.get(target)
        target_logit = int(backend.target_to_logit_idx[target])
        foil_logit = (
            int(backend.target_to_logit_idx[foil_label])
            if foil_label is not None and foil_label in backend.target_to_logit_idx
            else None
        )
        result = select_top_eap_syed(
            backend.model,
            prompt=str(kwargs.get("prompt") or backend.prompt),
            corrupted_prompt=_resolve_corrupted_prompt(args),
            node_to_intervention=backend.node_to_intervention,
            candidates=candidates,
            target_logit_idx=target_logit,
            foil_logit_idx=foil_logit,
            freeze_attention=bool(getattr(backend, "freeze_attention", True)),
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
            checkpoint_path=(
                str(Path(args.output_json).with_name("macag_baselines_shapley.ckpt.json"))
                if method == "shapley" and args.output_json
                else None
            ),
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
            cap_sufficiency=args.cap_sufficiency,
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
    cap_sufficiency: bool = False,
) -> dict[int, dict[str, Any]]:
    """Score each k-prefix of a ranking with the games' FaithfulnessMetrics.

    Always includes k=0 (empty evidence) so early-stop / empty selections remain
    visible in comparisons instead of disappearing from the method table.
    """
    results: dict[int, dict[str, Any]] = {}
    empty_metrics = compute_faithfulness_metrics(
        oracle=oracle, target=target, nodes=set(), alpha=alpha, cap_sufficiency=cap_sufficiency
    )
    results[0] = {
        "evidence": [],
        "scores": metrics_to_dict(empty_metrics)
        | {"utility": game1_utility(empty_metrics.faithfulness_delta, size=0, lam=lam)},
    }
    max_k = min(budget, len(ranking))
    for k in range(1, max_k + 1):
        evidence = set(ranking[:k])
        metrics = compute_faithfulness_metrics(
            oracle=oracle,
            target=target,
            nodes=evidence,
            alpha=alpha,
            cap_sufficiency=cap_sufficiency,
        )
        results[k] = {
            "evidence": _sort_nodes(evidence),
            "scores": metrics_to_dict(metrics)
            | {"utility": game1_utility(metrics.faithfulness_delta, size=k, lam=lam)},
        }
    return results


def _faithfulness_curve_on_budget(
    per_k_raw: Mapping[str, float],
    budget: int,
) -> dict[str, float]:
    """Carry-forward curve on the common domain k=0..budget.

    Early-stopping methods keep their last realized faithfulness for larger k
    (the selected set does not grow). Missing everything yields an empty map.
    """
    if not per_k_raw and budget < 0:
        return {}
    realized = {int(k): float(v) for k, v in per_k_raw.items()}
    if 0 not in realized:
        realized[0] = 0.0
    curve: dict[str, float] = {}
    last = realized.get(0, 0.0)
    for k in range(0, budget + 1):
        if k in realized:
            last = realized[k]
        curve[str(k)] = last
    return curve


def _trapezoidal_auc(curve: Mapping[str, float]) -> float:
    """Trapezoidal AUC over integer k keys, normalized by the k-span."""
    if not curve:
        return 0.0
    ks = sorted(int(k) for k in curve)
    if len(ks) == 1:
        return float(curve[str(ks[0])])
    area = 0.0
    for left, right in zip(ks, ks[1:]):
        area += 0.5 * (float(curve[str(left)]) + float(curve[str(right)])) * (right - left)
    span = ks[-1] - ks[0]
    return area / span if span > 0 else float(curve[str(ks[0])])


def _comparison_block(
    methods: dict[str, dict[str, Any]],
    selections: dict[str, SelectionResult],
    budget: int,
    game1_marginals: dict[NodeId, float] | None,
) -> dict[str, Any]:
    faithfulness_at_k: dict[str, dict[str, float]] = {}
    auc_raw: dict[str, float] = {}
    for method, entry in methods.items():
        per_k_raw = {
            str(k): payload["scores"]["faithfulness"] for k, payload in entry["results"].items()
        }
        curve = _faithfulness_curve_on_budget(per_k_raw, budget)
        faithfulness_at_k[method] = curve
        if curve:
            auc_raw[method] = _trapezoidal_auc(curve)

    comparison: dict[str, Any] = {
        "faithfulness_at_k": faithfulness_at_k,
        "auc_raw_faithfulness": auc_raw,
        "auc_definition": "trapezoidal_mean_over_k_0_to_budget_with_carry_forward",
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
                if gold.scores:
                    per_k[str(k)]["precision_at_k_tie_aware"] = tie_aware_precision_at_k(
                        selection.ranking,
                        gold.scores,
                        k,
                    )
                    raw_se = gold.extras.get("std_errors") or {}
                    if isinstance(raw_se, Mapping):
                        std_errors = {
                            node: float(raw_se.get(str(node), float("nan")))
                            for node in gold.scores
                        }
                        lower, upper = precision_at_k_uncertainty_bounds(
                            selection.ranking,
                            gold.scores,
                            std_errors,
                            k,
                        )
                        per_k[str(k)]["precision_at_k_uncertainty_lower"] = lower
                        per_k[str(k)]["precision_at_k_uncertainty_upper"] = upper
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
    parser.add_argument(
        "--cap-sufficiency",
        action="store_true",
        help=(
            "Evaluate and search under capped sufficiency, min(keep_only, all) - empty. "
            "Must match Game 1's --cap-sufficiency when comparing methods."
        ),
    )
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
        help="Search tau so ACDC keeps ≤k nodes (hard budget-matched ACDC); -1 uses "
        "--budget. Never returns an oversize set. Adds methods.acdc.matched_k and an "
        "acdc entry in comparison.faithfulness_at_k.",
    )

    parser.add_argument(
        "--eap-target-match",
        default=None,
        help="Quoted logit-token match for the +1 EAP seed (e.g. ' A' or 'A'). "
        "Default: token from oracle kwargs target_token_by_label[target], else "
        "the graph's is_target_logit flag.",
    )
    parser.add_argument(
        "--eap-foil-match",
        default=None,
        help="Quoted logit-token match for the -1 foil seed. Default: foil token "
        "from oracle kwargs (foil_by_target + target_token_by_label). Missing "
        "foil logits are skipped with a warning (target-only seeding).",
    )
    parser.add_argument("--eap-signed", action="store_true", help="Rank by signed EAP score instead of |score|.")
    parser.add_argument(
        "--eap-corrupted-prompt",
        default=None,
        help="Corrupted prompt for eap_syed (overrides corrupted_prompt in oracle kwargs).",
    )
    parser.add_argument("--influence-signed", action="store_true", help="Rank by signed influence instead of |influence|.")
    parser.add_argument(
        "--native-model-name",
        default=None,
        help="Optional HookedTransformer name for acdc_native (default: model_name from oracle kwargs).",
    )

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
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1].")
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
    acdc_native_output: dict[str, Any] | None = None
    acdc_sweep_results = None

    for method in methods:
        if method == "acdc":
            oracle.clear_cache()
            oracle.reset_stats()
            acdc_sweep_results = acdc_tau_sweep(
                graph,
                oracle,
                args.target,
                candidates,
                taus=_parse_float_list(args.acdc_taus),
                alpha=args.alpha,
                order=args.acdc_order,
                progress=args.progress,
                cap_sufficiency=args.cap_sufficiency,
            )
            acdc_output = {
                "sweep": [
                    {
                        "tau": result.tau,
                        "size": len(result.kept),
                        "kept": _sort_nodes(set(result.kept)),
                        "value": result.value,
                        "removed_order": [str(node) for node in result.removed_order],
                    }
                    for result in acdc_sweep_results
                ],
            }
            if args.acdc_target_k is not None:
                target_k = args.budget if args.acdc_target_k == -1 else args.acdc_target_k
                try:
                    matched = acdc_target_size(
                        graph,
                        oracle,
                        args.target,
                        candidates,
                        target_k=target_k,
                        alpha=args.alpha,
                        order=args.acdc_order,
                        seed_results=acdc_sweep_results,
                        progress=args.progress,
                        cap_sufficiency=args.cap_sufficiency,
                    )
                    acdc_output["_matched_pending"] = matched
                except ACDCBudgetUnreachableError as exc:
                    acdc_output["_matched_unavailable"] = {
                        "target_k": target_k,
                        "reason": str(exc),
                    }
                    LOGGER.warning(
                        "ACDC matched_k unavailable for %s: %s", args.input_id, exc
                    )
            stats = oracle.cache_stats()
            acdc_output["selection_stats"] = {
                "oracle_calls": stats["oracle_calls"],
                "cache_hits": stats["cache_hits"],
            }
            continue

        if method == "acdc_native":
            kwargs = _oracle_kwargs(args)
            model_name = args.native_model_name or kwargs.get("model_name")
            if not model_name:
                raise ValueError(
                    "acdc_native needs --native-model-name or model_name in oracle kwargs."
                )
            prompt = kwargs.get("prompt")
            if not prompt:
                raise ValueError("acdc_native needs prompt in oracle kwargs.")
            target_map = kwargs.get("target_to_logit_idx")
            if target_map is None and hasattr(oracle.backend, "target_to_logit_idx"):
                target_map = dict(oracle.backend.target_to_logit_idx)
            if not target_map:
                raise ValueError(
                    "acdc_native needs target_to_logit_idx (oracle kwargs or ReplacementModel backend)."
                )
            foil_map = kwargs.get("foil_by_target")
            if foil_map is None and hasattr(oracle.backend, "foil_by_target"):
                foil_map = getattr(oracle.backend, "foil_by_target", None)
            native_model = _load_native_hooked_transformer(
                str(model_name), kwargs.get("model_kwargs")
            )
            target_k = None
            if args.acdc_target_k is not None:
                target_k = args.budget if args.acdc_target_k == -1 else args.acdc_target_k
            acdc_native_output = run_acdc_native(
                native_model,
                prompt=str(prompt),
                target=args.target,
                target_to_logit_idx=target_map,
                taus=_parse_float_list(args.acdc_taus),
                alpha=args.alpha,
                order=args.acdc_order,
                score_kind=str(kwargs.get("score_kind", "logit_gap")),
                foil_by_target=foil_map,
                target_k=target_k,
                progress=args.progress,
            )
            continue

        try:
            selection, stats = _run_selection(
                method, args, graph, payload, oracle, args.target, candidates
            )
        except EAPUnavailableError as exc:
            # A graph may omit the requested target/foil from its exported logit
            # nodes. Target-only fallback would silently change the game; retain
            # an explicit unavailable method block and continue other selectors.
            stats = oracle.cache_stats()
            method_outputs[method] = {
                "status": "unavailable",
                "reason": str(exc),
                "ranking": None,
                "scores": None,
                "params": {
                    "target_match": _resolve_eap_logit_matches(args)[0],
                    "foil_match": _resolve_eap_logit_matches(args)[1],
                },
                "extras": {},
                "selection_stats": {
                    "oracle_calls": stats["oracle_calls"],
                    "cache_hits": stats["cache_hits"],
                },
                "results": {},
            }
            LOGGER.warning("Skipping graph EAP for %s: %s", args.input_id, exc)
            continue
        if method == "eap_syed":
            # Attribution patching bypasses ScoringOracle; record model work explicitly.
            stats = {
                **stats,
                "model_forwards": 3,  # clean acts, corrupted acts, intervention forward
                "model_backwards": 1,
                "oracle_calls_note": "eap_syed uses ReplacementModel directly; oracle_calls stay 0",
            }
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
            oracle,
            args.target,
            selection.ranking,
            args.budget,
            args.alpha,
            args.lam,
            cap_sufficiency=args.cap_sufficiency,
        )
    if acdc_output is not None:
        # ACDC produces one set per tau, not nested prefixes; score each kept
        # set as-is and bucket the best entry per size for matched-|E| reads.
        best_by_size: dict[int, dict[str, Any]] = {}
        for entry in acdc_output["sweep"]:
            kept = set(entry["kept"])
            metrics = compute_faithfulness_metrics(
                oracle=oracle,
                target=args.target,
                nodes=kept,
                alpha=args.alpha,
                cap_sufficiency=args.cap_sufficiency,
            )
            entry["scores"] = metrics_to_dict(metrics) | {
                "utility": game1_utility(metrics.faithfulness_delta, size=len(kept), lam=args.lam)
            }
            size = len(kept)
            current = best_by_size.get(size)
            if current is None or entry["scores"]["faithfulness"] > current["scores"]["faithfulness"]:
                best_by_size[size] = {"evidence": entry["kept"], "scores": entry["scores"], "tau": entry["tau"]}
        acdc_output["best_by_size"] = {str(size): best_by_size[size] for size in sorted(best_by_size)}

        matched_pending = acdc_output.pop("_matched_pending", None)
        matched_unavailable = acdc_output.pop("_matched_unavailable", None)
        if matched_pending is not None:
            matched = matched_pending
            matched_metrics = compute_faithfulness_metrics(
                oracle=oracle,
                target=args.target,
                nodes=set(matched.kept),
                alpha=args.alpha,
                cap_sufficiency=args.cap_sufficiency,
            )
            acdc_output["matched_k"] = {
                "target_k": matched.params["target_k"],
                "achieved_k": matched.params["achieved_k"],
                "exact": matched.params["exact"],
                "tau": matched.tau,
                "bisection_iters": matched.params["bisection_iters"],
                "search_evals": matched.params.get("search_evals"),
                "budget_capped": bool(matched.params.get("budget_capped", True)),
                "evidence": _sort_nodes(set(matched.kept)),
                "scores": metrics_to_dict(matched_metrics)
                | {
                    "utility": game1_utility(
                        matched_metrics.faithfulness_delta, size=len(matched.kept), lam=args.lam
                    )
                },
            }
        elif matched_unavailable is not None:
            acdc_output["matched_k"] = {
                "status": "unavailable",
                "target_k": matched_unavailable["target_k"],
                "achieved_k": None,
                "exact": False,
                "budget_capped": True,
                "reason": matched_unavailable["reason"],
                "evidence": None,
                "scores": None,
            }

    # Prefix/ACDC evaluation is one cost phase. Exact search is measured
    # separately so it cannot inflate ``evaluation_oracle_calls``.
    evaluation_stats = oracle.cache_stats()
    oracle.clear_cache()
    oracle.reset_stats()

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
            # Optimality gap vs each ranked method's size-k evidence (B3.2).
            # Early-stopping methods carry forward their last realized faithfulness
            # so a stall before k still reports a gap against the exact optimum.
            gaps: dict[str, float] = {}
            for method, output in method_outputs.items():
                results = output.get("results") or {}
                if k in results:
                    method_faith = results[k]["scores"]["faithfulness"]
                elif results:
                    last_k = max(int(key) for key in results if int(key) <= k)
                    method_faith = results[last_k]["scores"]["faithfulness"]
                else:
                    continue
                gaps[method] = result.best_value - method_faith
            entry["optimality_gap"] = gaps
            bruteforce_output[str(k)] = entry

    bruteforce_stats = oracle.cache_stats()

    # Game 1's per-step marginal faithfulness gains, for the §A.5 Spearman
    # linearity diagnostic against EAP/influence/Shapley scores.
    game1_marginals: dict[NodeId, float] | None = None
    if "game1" in selections and method_outputs["game1"].get("results"):
        game1_marginals = {}
        previous = 0.0
        results = method_outputs["game1"]["results"]
        for k in sorted(results):
            if k == 0:
                previous = results[k]["scores"]["faithfulness"]
                continue
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
        if (
            isinstance(matched, dict)
            and matched.get("status") != "unavailable"
            and matched.get("achieved_k") is not None
            and int(matched["achieved_k"]) <= args.budget
            and isinstance(matched.get("scores"), dict)
            and "faithfulness" in matched["scores"]
        ):
            acdc_at_k[str(matched["achieved_k"])] = matched["scores"]["faithfulness"]
        curve = _faithfulness_curve_on_budget(acdc_at_k, args.budget)
        comparison["faithfulness_at_k"]["acdc"] = curve
        comparison.setdefault("auc_raw_faithfulness", {})["acdc"] = _trapezoidal_auc(curve)

    if acdc_native_output is not None and acdc_native_output.get("best_by_size"):
        # Behavioral track: faithfulness is under the native component oracle,
        # already stored on each sweep entry / best_by_size block.
        native_at_k = {
            size: block["scores"]["faithfulness"]
            for size, block in acdc_native_output["best_by_size"].items()
        }
        matched_native = acdc_native_output.get("matched_k")
        if (
            isinstance(matched_native, dict)
            and matched_native.get("status") != "unavailable"
            and matched_native.get("achieved_k") is not None
            and int(matched_native["achieved_k"]) <= args.budget
            and matched_native.get("value") is not None
        ):
            native_at_k[str(matched_native["achieved_k"])] = matched_native["value"]
        curve = _faithfulness_curve_on_budget(native_at_k, args.budget)
        comparison["faithfulness_at_k"]["acdc_native"] = curve
        comparison.setdefault("auc_raw_faithfulness", {})["acdc_native"] = _trapezoidal_auc(curve)
        comparison.setdefault("notes", {})
        comparison["notes"]["acdc_native"] = (
            "Scored under native head/MLP oracle; not Jaccard-comparable to CLT feature methods. "
            "See macag/docs/baseline_method_map.md."
        )

    method_selection_costs: dict[str, dict[str, Any]] = {}
    cost_entries = dict(method_outputs)
    if acdc_output is not None:
        cost_entries["acdc"] = acdc_output
    if acdc_native_output is not None:
        cost_entries["acdc_native"] = acdc_native_output
    for method, entry in cost_entries.items():
        stats = entry.get("selection_stats") or {}
        forwards = int(stats.get("model_forwards", 0))
        backwards = int(stats.get("model_backwards", 0))
        method_selection_costs[method] = {
            "oracle_intervention_calls": int(stats.get("oracle_calls", 0)),
            "direct_model_forwards": forwards,
            "direct_model_backwards": backwards,
            "cost_basis": (
                "direct_model_passes"
                if forwards or backwards
                else "oracle_intervention_calls"
            ),
        }

    output: dict[str, Any] = {
        "input_id": args.input_id,
        "target": args.target,
        "game": "baselines",
        "experiment_identity": _experiment_identity(args, payload, candidates),
        "params": {
            "alpha": args.alpha,
            "lambda": args.lam,
            "budget": args.budget,
            "score_kind": _oracle_kwargs(args).get("score_kind"),
            "freeze_attention": _oracle_kwargs(args).get("freeze_attention"),
            "ablation_mode": _oracle_kwargs(args).get(
                "ablation_mode", _oracle_kwargs(args).get("ablation_kind")
            ),
            "methods": methods,
            "method_map": "macag/docs/baseline_method_map.md",
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
        "stats": {
            "method_selection_costs": method_selection_costs,
            "evaluation_oracle_calls": evaluation_stats["oracle_calls"],
            "evaluation_cache_hits": evaluation_stats["cache_hits"],
            "bruteforce_oracle_calls": bruteforce_stats["oracle_calls"],
            "bruteforce_cache_hits": bruteforce_stats["cache_hits"],
        },
    }
    if acdc_output is not None:
        output["methods"]["acdc"] = acdc_output
    if acdc_native_output is not None:
        output["methods"]["acdc_native"] = acdc_native_output
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
