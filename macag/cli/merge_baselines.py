"""CLI: merge a separately-run selector into a saved ``macag_baselines.json``.

Enables the fast/slow split for expensive selectors (MC Shapley-gold is ~90% of
a prompt's baseline cost): run the sweep with
``BASELINE_METHODS=influence,eap,game1,acdc`` first, later run
``run_baselines --methods shapley`` per prompt into a sidecar JSON, then merge:

    python -m macag.cli.merge_baselines \
        --main  <run_dir>/macag_baselines.json \
        --extra <run_dir>/macag_baselines_shapley.json

The merge copies the extra payload's ``methods`` blocks into the main payload
and **recomputes the whole comparison block** (faithfulness_at_k, AUC,
agreement-vs-gold precision@k/Jaccard, pairwise Jaccard, Spearman incl. the
game1-marginal diagnostic) from the stored rankings/results — pure JSON math,
no oracle calls, so it is safe to run on CPU after the fact.

Incompatible experiments are rejected: input_id, target, candidates, budget,
alpha, lambda, and score kind must match.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from macag.baselines.common import SelectionResult
from macag.cli.run_baselines import (
    _comparison_block,
    _faithfulness_curve_on_budget,
    _trapezoidal_auc,
)
from macag.graph import NodeId

LOGGER = logging.getLogger(__name__)

_IDENTITY_PARAM_KEYS = (
    "alpha",
    "lambda",
    "budget",
    "score_kind",
    "freeze_attention",
    "ablation_mode",
)


def _selection_from_payload(method: str, entry: dict[str, Any]) -> SelectionResult | None:
    ranking = entry.get("ranking")
    if ranking is None:
        return None
    scores = entry.get("scores")
    return SelectionResult(
        method=method,
        ranking=[str(node) for node in ranking],
        scores={str(node): float(score) for node, score in scores.items()} if scores else None,
        params=dict(entry.get("params") or {}),
        extras=dict(entry.get("extras") or {}),
    )


def _game1_marginals(methods: dict[str, Any]) -> dict[NodeId, float] | None:
    entry = methods.get("game1") or {}
    results = entry.get("results") or {}
    ranking = entry.get("ranking") or []
    if not results or not ranking:
        return None
    marginals: dict[NodeId, float] = {}
    previous = 0.0
    for k in sorted(int(key) for key in results):
        faith = results[str(k)]["scores"]["faithfulness"]
        if k == 0:
            previous = faith
            continue
        if k - 1 < len(ranking):
            marginals[ranking[k - 1]] = faith - previous
        previous = faith
    return marginals


def _block_complete(method: str, entry: dict[str, Any]) -> bool:
    """True when ``entry`` already has a usable result for ``method``."""
    if entry.get("status") == "unavailable":
        return True
    if entry.get("results"):
        return True
    # τ-sweep methods have no ranked prefixes; treat sweep / matched_k / best_by_size as done.
    if method in ("acdc", "acdc_native"):
        return bool(entry.get("matched_k") or entry.get("sweep") or entry.get("best_by_size"))
    return False


def _inject_tau_faithfulness(
    comparison: dict[str, Any], method: str, entry: dict[str, Any], budget: int
) -> None:
    """Mirror τ-sweep best-by-size faithfulness into ``comparison.faithfulness_at_k``."""
    if not entry.get("best_by_size") and not entry.get("matched_k"):
        return
    at_k: dict[str, float] = {}
    for size, block in (entry.get("best_by_size") or {}).items():
        if isinstance(block.get("scores"), dict) and "faithfulness" in block["scores"]:
            at_k[size] = block["scores"]["faithfulness"]
    matched = entry.get("matched_k")
    if matched is not None and matched.get("status") != "unavailable":
        ach = matched.get("achieved_k")
        # Hard budget: never inject an oversize matched point into the ≤budget curve.
        if ach is not None and int(ach) <= budget:
            # Ported ACDC stores scores.faithfulness; native may store value.
            if isinstance(matched.get("scores"), dict) and "faithfulness" in matched["scores"]:
                at_k[str(ach)] = matched["scores"]["faithfulness"]
            elif "value" in matched and matched["value"] is not None:
                at_k[str(ach)] = matched["value"]
    curve = _faithfulness_curve_on_budget(at_k, budget)
    comparison.setdefault("faithfulness_at_k", {})[method] = curve
    comparison.setdefault("auc_raw_faithfulness", {})[method] = _trapezoidal_auc(curve)


def _require_compatible(
    main: dict[str, Any],
    extra: dict[str, Any],
    *,
    allow_missing_main_identity: bool = False,
) -> None:
    """Reject merges that would combine incompatible experiments."""
    mismatches: list[str] = []
    main_identity = main.get("experiment_identity")
    extra_identity = extra.get("experiment_identity")
    if isinstance(main_identity, dict) and isinstance(extra_identity, dict):
        if main_identity != extra_identity:
            identity_keys = sorted(set(main_identity) | set(extra_identity))
            for key in identity_keys:
                if main_identity.get(key) != extra_identity.get(key):
                    mismatches.append(
                        "experiment_identity."
                        f"{key}: main={main_identity.get(key)!r} "
                        f"extra={extra_identity.get(key)!r}"
                    )
    elif isinstance(extra_identity, dict) and main_identity is None:
        if not allow_missing_main_identity:
            mismatches.append(
                "experiment_identity missing from main payload "
                "(explicit adoption is required for legacy artifacts)"
            )
    elif isinstance(main_identity, dict) != isinstance(extra_identity, dict):
        mismatches.append("experiment_identity missing from one payload")
    elif main_identity is None and extra_identity is None:
        mismatches.append("experiment_identity missing from both payloads")

    for key in ("input_id", "target"):
        main_val = main.get(key)
        extra_val = extra.get(key)
        if main_val is not None and extra_val is not None and main_val != extra_val:
            mismatches.append(f"{key}: main={main_val!r} extra={extra_val!r}")

    main_params = main.get("params") or {}
    extra_params = extra.get("params") or {}
    for key in _IDENTITY_PARAM_KEYS:
        if key in main_params and key in extra_params and main_params[key] != extra_params[key]:
            mismatches.append(f"params.{key}: main={main_params[key]!r} extra={extra_params[key]!r}")

    main_cands = [str(c) for c in (main.get("candidates") or [])]
    extra_cands = [str(c) for c in (extra.get("candidates") or [])]
    if main_cands != extra_cands:
        mismatches.append(
            f"candidates differ (main={len(main_cands)} nodes, extra={len(extra_cands)} nodes)"
        )

    if mismatches:
        raise ValueError(
            "Refusing to merge incompatible baseline payloads:\n  - "
            + "\n  - ".join(mismatches)
        )


def merge_payloads(
    main: dict[str, Any],
    extra: dict[str, Any],
    *,
    adopt_extra_identity: bool = False,
) -> dict[str, Any]:
    """Merge ``extra``'s methods into ``main`` and rebuild the comparison block."""
    _require_compatible(
        main,
        extra,
        allow_missing_main_identity=adopt_extra_identity,
    )
    if adopt_extra_identity and main.get("experiment_identity") is None:
        main["experiment_identity"] = dict(extra["experiment_identity"])
    methods: dict[str, Any] = dict(main.get("methods") or {})
    added: list[str] = []
    for method, entry in (extra.get("methods") or {}).items():
        existing = methods.get(method)
        if existing is not None and _block_complete(method, existing):
            LOGGER.info("keeping existing '%s' block (already present in main)", method)
            continue
        methods[method] = entry
        added.append(method)
    main["methods"] = methods

    ranked = {
        name: entry for name, entry in methods.items() if entry.get("results")
    }
    selections: dict[str, SelectionResult] = {}
    for name, entry in ranked.items():
        selection = _selection_from_payload(name, entry)
        if selection is not None:
            selections[name] = selection

    budget = int(main.get("params", {}).get("budget", 8))
    comparison = _comparison_block(ranked, selections, budget, _game1_marginals(methods))

    # Preserve τ-sweep injections (no ranked prefixes -> _comparison_block skips them).
    _inject_tau_faithfulness(comparison, "acdc", methods.get("acdc") or {}, budget)
    _inject_tau_faithfulness(comparison, "acdc_native", methods.get("acdc_native") or {}, budget)
    # Keep honesty notes from either payload (extra wins on key collision).
    notes: dict[str, Any] = {}
    notes.update((main.get("comparison") or {}).get("notes") or {})
    notes.update((extra.get("comparison") or {}).get("notes") or {})
    if notes:
        comparison["notes"] = notes
    main["comparison"] = comparison

    params = dict(main.get("params") or {})
    merged_methods = params.get("methods") or []
    params["methods"] = sorted(set(merged_methods) | set(added))
    for key in ("shapley_permutations", "banzhaf_samples", "shapley_seed", "antithetic"):
        if key in (extra.get("params") or {}):
            params[key] = extra["params"][key]
    main["params"] = params

    main_stats = dict(main.get("stats") or {})
    extra_stats = dict(extra.get("stats") or {})
    method_costs = dict(main_stats.get("method_selection_costs") or {})
    method_costs.update(extra_stats.get("method_selection_costs") or {})
    main_stats["method_selection_costs"] = method_costs
    for key in (
        "evaluation_oracle_calls",
        "evaluation_cache_hits",
        "bruteforce_oracle_calls",
        "bruteforce_cache_hits",
    ):
        main_stats[key] = int(main_stats.get(key, 0)) + int(extra_stats.get(key, 0))
    main_stats["merged_run_count"] = int(main_stats.get("merged_run_count", 1)) + int(
        extra_stats.get("merged_run_count", 1)
    )
    main["stats"] = main_stats
    return main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True,
                        help="macag_baselines.json to merge into (rewritten in place).")
    parser.add_argument("--extra", type=Path, required=True,
                        help="Sidecar baselines JSON with the separately-run method(s).")
    parser.add_argument(
        "--adopt-extra-identity",
        action="store_true",
        help=(
            "For a legacy main payload without experiment_identity, adopt the "
            "sidecar identity after legacy field/candidate checks pass."
        ),
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.progress:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    main_payload = json.loads(args.main.read_text())
    extra_payload = json.loads(args.extra.read_text())
    merged = merge_payloads(
        main_payload,
        extra_payload,
        adopt_extra_identity=args.adopt_extra_identity,
    )
    args.main.write_text(json.dumps(merged, indent=2))
    print(f"merged {sorted((extra_payload.get('methods') or {}).keys())} into {args.main}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
