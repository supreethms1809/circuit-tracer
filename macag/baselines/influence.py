"""B2.1 — top-k influence: rank candidates by the graph's own `influence` metric.

The cheap "is search needed?" floor (macag.md §A.3): no interventions, no
interaction modeling — pure magnitude ranking read off the attribution graph.
"""

from __future__ import annotations

import logging
from typing import Sequence

from macag.baselines.common import SelectionResult
from macag.graph import CircuitGraph, NodeId
from macag.utils.metrics import dedupe_preserve_order

LOGGER = logging.getLogger(__name__)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def select_top_influence(
    graph: CircuitGraph,
    candidates: Sequence[NodeId],
    use_absolute: bool = True,
) -> SelectionResult:
    """Rank candidates by their `influence` node metadata, best first.

    Candidates missing an `influence` value are ranked last (deterministically by
    node string) and excluded from the score map. Raises if no candidate carries
    an influence value at all — that means the graph JSON was exported without
    pruning-time influence and this baseline cannot run on it.
    """
    pool = [node for node in dedupe_preserve_order(candidates) if graph.has_node(node)]
    scores: dict[NodeId, float] = {}
    missing: list[NodeId] = []
    for node in pool:
        influence = _to_float(graph.metadata(node).get("influence"))
        if influence is None:
            missing.append(node)
            continue
        scores[node] = abs(influence) if use_absolute else influence

    if not scores:
        raise ValueError(
            "No candidate node carries an 'influence' metadata value; the top-k "
            "influence baseline needs a graph JSON exported with node influence."
        )
    if missing:
        LOGGER.warning(
            "%d candidate(s) have no 'influence' metadata and are ranked last.", len(missing)
        )

    ranking = sorted(scores, key=lambda node: (-scores[node], str(node)))
    ranking.extend(sorted(missing, key=str))
    return SelectionResult(
        method="influence",
        ranking=ranking,
        scores=scores,
        params={"use_absolute": use_absolute},
        extras={"missing_influence_count": len(missing)},
    )
