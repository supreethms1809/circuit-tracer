"""Run MACAG Game 1 / Game 2 from the command line."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any

from macag.games.game1_attention_probe import solve_game1_dual
from macag.games.game1_min_faithful import EvidenceSetResult, solve_game1
from macag.games.game2_contrastive import solve_game2
from macag.graph import CircuitGraph, NodeId
from macag.scoring import ScoringOracle, ToyAdditiveInterventionScorer, derive_oracle_with_freeze
from macag.utils.metrics import metrics_to_dict

LOGGER = logging.getLogger(__name__)


def _sort_nodes(nodes: set[NodeId]) -> list[str]:
    return sorted([str(node) for node in nodes])


def _load_json(path: str | Path) -> Any:
    resolved = Path(path)
    try:
        text = resolved.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved}: {exc}") from exc


def _load_candidates(path: str | Path) -> list[NodeId]:
    candidate_path = Path(path)
    if candidate_path.suffix == ".json":
        payload = _load_json(candidate_path)
        if not isinstance(payload, list):
            raise ValueError("Candidate JSON must contain a list of node IDs.")
        return payload
    return [line.strip() for line in candidate_path.read_text().splitlines() if line.strip()]


_ALLOWED_FACTORY_MODULE_PREFIXES = (
    "macag.",
    "circuit_tracer.",
)


def _factory_accepts_kwarg(factory: Any, name: str) -> bool:
    import inspect

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return True  # signature unavailable; let the call itself surface errors
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


def _load_factory(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError("Factory import path must use 'module_path:factory_name' format.")
    module_name, factory_name = import_path.split(":", 1)
    if not any(module_name.startswith(prefix) for prefix in _ALLOWED_FACTORY_MODULE_PREFIXES):
        raise ValueError(
            f"Factory module '{module_name}' is not in the allowed prefixes: "
            f"{_ALLOWED_FACTORY_MODULE_PREFIXES}. "
            "Update _ALLOWED_FACTORY_MODULE_PREFIXES if you need a custom factory module."
        )
    module = importlib.import_module(module_name)
    return getattr(module, factory_name)


def _normalize_factory_output(
    built: Any,
    no_cache: bool,
) -> tuple[ScoringOracle, list[NodeId] | None]:
    if isinstance(built, ScoringOracle):
        built.cache_enabled = not no_cache
        candidates = None
        backend = getattr(built, "backend", None)
        if backend is not None and hasattr(backend, "supported_nodes"):
            candidates = sorted([str(node) for node in backend.supported_nodes()])
        return built, candidates

    if hasattr(built, "oracle"):
        oracle = built.oracle
        if not isinstance(oracle, ScoringOracle):
            raise ValueError("Factory output `.oracle` must be a ScoringOracle.")
        oracle.cache_enabled = not no_cache
        candidates = getattr(built, "candidates", None)
        return oracle, candidates

    if isinstance(built, dict) and "oracle" in built:
        oracle = built["oracle"]
        if not isinstance(oracle, ScoringOracle):
            raise ValueError("Factory output dict['oracle'] must be a ScoringOracle.")
        oracle.cache_enabled = not no_cache
        return oracle, built.get("candidates")

    oracle = ScoringOracle(backend=built, cache_enabled=not no_cache)
    candidates = None
    if hasattr(built, "supported_nodes"):
        candidates = sorted([str(node) for node in built.supported_nodes()])
    return oracle, candidates


def _build_oracle(args: argparse.Namespace) -> tuple[ScoringOracle, list[NodeId] | None]:
    if args.toy_oracle_json:
        backend = ToyAdditiveInterventionScorer.from_json_file(args.toy_oracle_json)
        return ScoringOracle(backend=backend, cache_enabled=not args.no_cache), None

    if not args.oracle_factory:
        raise ValueError("Provide --oracle-factory or --toy-oracle-json.")

    kwargs: dict[str, Any] = {}
    if args.oracle_kwargs_json:
        kwargs.update(json.loads(args.oracle_kwargs_json))
    if args.oracle_kwargs_file:
        kwargs.update(_load_json(args.oracle_kwargs_file))

    factory = _load_factory(args.oracle_factory)
    if getattr(args, "include_error_nodes", False):
        if _factory_accepts_kwarg(factory, "include_error_nodes"):
            kwargs["include_error_nodes"] = True
        else:
            raise ValueError(
                f"--include-error-nodes was set, but factory '{args.oracle_factory}' does "
                "not accept an 'include_error_nodes' keyword argument."
            )
    built = factory(**kwargs)
    return _normalize_factory_output(built=built, no_cache=args.no_cache)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph-json", required=True, help="Path to circuit graph JSON.")
    parser.add_argument("--target", required=True, help="Target class/label.")
    parser.add_argument("--input-id", default="unknown", help="Input identifier for output JSON.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Faithfulness mix weight.")
    parser.add_argument("--lam", type=float, default=0.01, help="Sparsity penalty lambda.")
    parser.add_argument("--budget", type=int, default=None, help="Optional |E| budget.")
    parser.add_argument(
        "--prefilter-top-k",
        type=int,
        default=None,
        help="Optional top-k singleton prefilter before greedy search.",
    )
    parser.add_argument(
        "--connected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require evidence connectedness (default: on; disable with --no-connected).",
    )
    parser.add_argument("--min-gain", type=float, default=0.0, help="Minimum positive gain to add nodes.")
    parser.add_argument("--candidates-file", default=None, help="Optional candidate node list (.json or text).")
    parser.add_argument("--output-json", required=True, help="Path to write result JSON.")

    parser.add_argument(
        "--oracle-factory",
        default=None,
        help="Factory path 'module.submodule:function' returning backend scorer or ScoringOracle.",
    )
    parser.add_argument(
        "--oracle-kwargs-json",
        default=None,
        help="Inline JSON object passed as kwargs to --oracle-factory.",
    )
    parser.add_argument(
        "--oracle-kwargs-file",
        default=None,
        help="JSON file with kwargs passed to --oracle-factory.",
    )
    parser.add_argument(
        "--toy-oracle-json",
        default=None,
        help="Path to toy additive oracle JSON (quick test harness).",
    )
    parser.add_argument(
        "--include-error-nodes",
        action="store_true",
        help=(
            "Opt-in (C1): admit MLP reconstruction-error nodes as ablation candidates so "
            "'empty' is a true empty baseline. Not supported by the feature-index "
            "intervention path; raises a clear error if error nodes are present. The "
            "default report-only normalized metrics de-bias faithfulness without this."
        ),
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable oracle memoization.")
    parser.set_defaults(progress=True)
    parser.add_argument(
        "--progress",
        action="store_true",
        dest="progress",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable solver progress logs/progress bars (enabled by default).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Fallback logging frequency when tqdm is unavailable.",
    )


def _resolve_game1_stop_metric(stop_metric: str | None, freeze_mode: str) -> str:
    """Resolve the --stop-metric sentinel against --freeze-mode.

    Unset defaults to 'normalized' for frozen runs (today's behavior) and
    'raw_relative' otherwise. 'normalized' + 'both' is an error: the dual run
    exists to produce a matched comparison, and the normalized stop divides by
    recoverable_range, which is degenerate on the unfrozen leg — the two legs
    would stop under non-comparable rules.
    """
    if freeze_mode == "both":
        if stop_metric == "normalized":
            raise ValueError(
                "--stop-metric normalized is incompatible with --freeze-mode both: "
                "recoverable_range is degenerate on the unfrozen leg, so the legs "
                "would stop under non-comparable rules. Use raw_relative (the "
                "default for --freeze-mode both)."
            )
        return "raw_relative"
    if stop_metric is None:
        if freeze_mode == "unfrozen":
            LOGGER.info(
                "--stop-metric not set; using raw_relative for the unfrozen run "
                "(recommended when recoverable_range is unreliable)."
            )
            return "raw_relative"
        return "normalized"
    if stop_metric == "normalized" and freeze_mode == "unfrozen":
        LOGGER.warning(
            "--stop-metric normalized with --freeze-mode unfrozen: recoverable_range "
            "is often unreliable when attention is unfrozen; consider raw_relative."
        )
    return stop_metric


def _game1_leg_payload(
    result: EvidenceSetResult, freeze_attention: bool | None = None
) -> dict[str, Any]:
    """Serialize one Game 1 solve. This is the single-mode output minus the
    envelope (input_id/target/foil/game), shared by both freeze-mode paths so
    the schemas cannot drift."""
    params = dict(result.params)
    if freeze_attention is not None:
        params["freeze_attention"] = freeze_attention
    return {
        "params": params,
        "evidence": {
            "E_star": _sort_nodes(result.evidence),
            "E_y": _sort_nodes(result.evidence),
            "E_foil": [],
            "shared": [],
            "unique_y": _sort_nodes(result.evidence),
            "unique_foil": [],
        },
        "scores": metrics_to_dict(result.metrics) | {"utility": result.utility},
        "stats": {
            "oracle_calls": result.oracle_calls,
            "cache_hits": result.cache_hits,
            "cache_size": result.cache_size,
            "iterations": result.iterations,
            "total_candidates": result.total_candidates,
            "prefiltered_candidate_count": result.candidate_count,
            "sparsity": result.sparsity,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MACAG evidence allocation games.")
    subparsers = parser.add_subparsers(dest="game", required=True)

    game1 = subparsers.add_parser("game1", help="Run Game 1: minimal faithful subset.")
    _add_common_args(game1)
    game1.add_argument(
        "--faithfulness-eps",
        type=float,
        default=None,
        help=(
            "Optional stop condition; interpretation depends on --stop-metric. "
            "eps is in [0, 1]."
        ),
    )
    game1.add_argument(
        "--stop-metric",
        choices=("normalized", "raw_relative"),
        default=None,
        help=(
            "How --faithfulness-eps stops the greedy. 'normalized': stop when "
            "faithfulness_normalized >= 1 - eps (divides by recoverable_range; degenerate "
            "when that range collapses, e.g. unfrozen attention). 'raw_relative': "
            "denominator-free diminishing-returns stop (stop when the best marginal raw "
            "faithfulness gain < eps * the first feature's gain). Default: 'normalized' "
            "for --freeze-mode frozen, 'raw_relative' for unfrozen/both. 'normalized' is "
            "rejected with --freeze-mode both (the legs would stop under non-comparable "
            "rules)."
        ),
    )
    game1.add_argument(
        "--freeze-mode",
        choices=("frozen", "unfrozen", "both"),
        default="frozen",
        help=(
            "Attention-freezing protocol. 'frozen' (default): use the factory-built "
            "oracle as-is (freeze_attention defaults to true). 'unfrozen': derive an "
            "unfrozen oracle from the factory-built one. 'both': run a fully matched "
            "frozen+unfrozen pair (same budget/prefilter/eps, raw_relative stop on both "
            "legs) and emit the per-prompt attention-mediation diagnostic. "
            "unfrozen/both require a ReplacementModel-backed oracle "
            "(not --toy-oracle-json)."
        ),
    )

    game2 = subparsers.add_parser("game2", help="Run Game 2: contrastive allocation.")
    _add_common_args(game2)
    game2.add_argument("--foil", required=True, help="Foil class/label.")
    game2.add_argument("--beta", type=float, default=0.1, help="Overlap penalty beta.")
    game2.add_argument(
        "--abr-iters",
        type=int,
        default=10,
        help="Max solver iterations (ABR rounds or fictitious-play rounds).",
    )
    game2.add_argument(
        "--solver",
        choices=("abr", "fp"),
        default="abr",
        help=(
            "Update rule: 'abr' (default) best-responds to the opponent's last "
            "evidence set; 'fp' (fictitious play) best-responds to the opponent's "
            "empirical history of sets via an expected-overlap penalty, which "
            "dampens ABR cycling."
        ),
    )
    game2.add_argument(
        "--fp-tol",
        type=float,
        default=1e-3,
        help=(
            "Fictitious play only: stop when the max change in empirical node "
            "frequencies across both agents falls below this tolerance."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.progress:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    if args.game == "game1":
        # Fail fast, before the (possibly expensive) oracle build.
        stop_metric = _resolve_game1_stop_metric(args.stop_metric, args.freeze_mode)
        if args.freeze_mode != "frozen" and args.toy_oracle_json:
            raise ValueError(
                "--freeze-mode unfrozen/both requires a ReplacementModel-backed oracle "
                "(--oracle-factory); toy oracles have no attention to freeze."
            )

    graph = CircuitGraph.from_json(args.graph_json)
    oracle, factory_candidates = _build_oracle(args)
    candidates = _load_candidates(args.candidates_file) if args.candidates_file else None
    if candidates is None and factory_candidates is not None:
        candidates = list(factory_candidates)
    elif candidates is not None and factory_candidates is not None:
        valid = set(factory_candidates)
        candidates = [candidate for candidate in candidates if candidate in valid]
        if not candidates:
            raise ValueError(
                "No provided candidates are supported by the oracle backend after filtering."
            )

    backend = getattr(oracle, "backend", None)
    if candidates is not None and backend is not None and hasattr(backend, "restrict_universe"):
        backend.restrict_universe(set(candidates))
        # Universe changed; drop any universe-dependent cache entries scored against
        # the pre-restriction universe (I1).
        oracle.clear_cache()

    output: dict[str, Any]
    if args.game == "game1" and args.freeze_mode == "both":
        # Derive AFTER restrict_universe so both legs inherit the restriction.
        # The helper short-circuits when freeze already matches, so exactly one
        # dataclasses.replace happens whatever the oracle kwargs said.
        frozen_oracle = derive_oracle_with_freeze(oracle, True)
        unfrozen_oracle = derive_oracle_with_freeze(oracle, False)
        dual = solve_game1_dual(
            graph=graph,
            frozen_oracle=frozen_oracle,
            unfrozen_oracle=unfrozen_oracle,
            target=args.target,
            candidates=candidates,
            alpha=args.alpha,
            lam=args.lam,
            budget=args.budget,
            faithfulness_eps=args.faithfulness_eps,
            prefilter_top_k=args.prefilter_top_k,
            connected=args.connected,
            min_gain=args.min_gain,
            progress=args.progress,
            log_every=args.log_every,
        )
        output = {
            "input_id": args.input_id,
            "target": args.target,
            "foil": None,
            "game": "game1",
            "freeze_mode": "both",
            "params": dual.params,
            "frozen": _game1_leg_payload(dual.frozen, freeze_attention=True),
            "unfrozen": _game1_leg_payload(dual.unfrozen, freeze_attention=False),
            "attention_mediation": dual.diagnostic.to_dict(),
        }
    elif args.game == "game1":
        if args.freeze_mode == "unfrozen":
            oracle = derive_oracle_with_freeze(oracle, False)
        elif getattr(getattr(oracle, "backend", None), "freeze_attention", None) is False:
            LOGGER.warning(
                "--freeze-mode frozen (the default) is honoring the kwargs-built "
                "UNFROZEN oracle (freeze_attention=false); pass --freeze-mode "
                "unfrozen to make this explicit."
            )
        result = solve_game1(
            graph=graph,
            oracle=oracle,
            target=args.target,
            candidates=candidates,
            alpha=args.alpha,
            lam=args.lam,
            budget=args.budget,
            faithfulness_eps=args.faithfulness_eps,
            stop_metric=stop_metric,
            prefilter_top_k=args.prefilter_top_k,
            connected=args.connected,
            min_gain=args.min_gain,
            progress=args.progress,
            log_every=args.log_every,
        )
        output = {
            "input_id": args.input_id,
            "target": args.target,
            "foil": None,
            "game": "game1",
            **_game1_leg_payload(result),
        }
    else:
        result = solve_game2(
            graph=graph,
            oracle=oracle,
            y=args.target,
            y_foil=args.foil,
            candidates=candidates,
            alpha=args.alpha,
            lam=args.lam,
            beta=args.beta,
            abr_iters=args.abr_iters,
            budget=args.budget,
            connected=args.connected,
            min_gain=args.min_gain,
            prefilter_top_k=args.prefilter_top_k,
            solver=args.solver,
            fp_tol=args.fp_tol,
            progress=args.progress,
            log_every=args.log_every,
        )
        output = {
            "input_id": args.input_id,
            "target": args.target,
            "foil": args.foil,
            "game": "game2",
            "params": result.params,
            "evidence": {
                "E_y": _sort_nodes(result.evidence_y),
                "E_foil": _sort_nodes(result.evidence_foil),
                "shared": _sort_nodes(result.shared),
                "unique_y": _sort_nodes(result.unique_y),
                "unique_foil": _sort_nodes(result.unique_foil),
            },
            "scores": {
                "target": metrics_to_dict(result.metrics_y),
                "foil": metrics_to_dict(result.metrics_foil),
                "utility_target": result.utility_y,
                "utility_foil": result.utility_foil,
                "overlap_rate": result.overlap_rate,
            },
            "stats": {
                "oracle_calls": result.oracle_calls,
                "cache_hits": result.cache_hits,
                "cache_size": result.cache_size,
                "iterations": result.iterations,
                "converged": result.converged,
                "best_iteration": result.best_iteration,
                "total_candidates": result.total_candidates,
                "sparsity_y": result.sparsity_y,
                "sparsity_foil": result.sparsity_foil,
            },
        }
        if args.solver == "fp":
            output["fp"] = {
                "node_frequencies_y": result.node_frequencies_y,
                "node_frequencies_foil": result.node_frequencies_foil,
            }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
