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
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from macag.baselines.common import SelectionResult
from macag.cli.run_baselines import _comparison_block
from macag.graph import NodeId

LOGGER = logging.getLogger(__name__)


def _selection_from_payload(method: str, entry: dict[str, Any]) -> SelectionResult | None:
    ranking = entry.get("ranking")
    if not ranking:
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
        if k - 1 < len(ranking):
            marginals[ranking[k - 1]] = faith - previous
        previous = faith
    return marginals


def merge_payloads(main: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge ``extra``'s methods into ``main`` and rebuild the comparison block."""
    methods: dict[str, Any] = dict(main.get("methods") or {})
    added: list[str] = []
    for method, entry in (extra.get("methods") or {}).items():
        if method in methods and (methods[method].get("results") or method == "acdc"):
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

    # Preserve the ACDC injection (no ranked prefixes -> _comparison_block skips it).
    acdc = methods.get("acdc") or {}
    if acdc.get("best_by_size"):
        acdc_at_k = {
            size: block["scores"]["faithfulness"]
            for size, block in acdc["best_by_size"].items()
        }
        matched = acdc.get("matched_k")
        if matched is not None:
            acdc_at_k[str(matched["achieved_k"])] = matched["scores"]["faithfulness"]
        comparison["faithfulness_at_k"]["acdc"] = acdc_at_k
    main["comparison"] = comparison

    params = dict(main.get("params") or {})
    merged_methods = params.get("methods") or []
    params["methods"] = sorted(set(merged_methods) | set(added))
    for key in ("shapley_permutations", "banzhaf_samples", "shapley_seed", "antithetic"):
        if key in (extra.get("params") or {}):
            params[key] = extra["params"][key]
    main["params"] = params
    return main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True,
                        help="macag_baselines.json to merge into (rewritten in place).")
    parser.add_argument("--extra", type=Path, required=True,
                        help="Sidecar baselines JSON with the separately-run method(s).")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.progress:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    main_payload = json.loads(args.main.read_text())
    extra_payload = json.loads(args.extra.read_text())
    merged = merge_payloads(main_payload, extra_payload)
    args.main.write_text(json.dumps(merged, indent=2))
    print(f"merged {sorted((extra_payload.get('methods') or {}).keys())} into {args.main}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
