"""B2.1 — top-k influence: rank candidates by the graph's own influence metric.

The cheap "is search needed?" floor (macag.md §A.3): no interventions, no
interaction modeling — pure magnitude ranking read off the attribution graph.

Circuit-tracer exports two related fields:
- ``influence_raw``: raw node influence magnitude (larger = stronger). Prefer this.
- ``influence``: cumulative coverage used by the frontend prune slider. When ranked
  descending this field is *reversed* (strongest nodes get the smallest values).
  Legacy graphs without ``influence_raw`` are handled by ranking cumulative values
  ascending when they look like coverage scores in (0, 1].
"""

from __future__ import annotations

import logging
import math
from typing import Mapping, Sequence

from macag.baselines.common import SelectionResult
from macag.graph import CircuitGraph, NodeId
from macag.utils.metrics import dedupe_preserve_order

LOGGER = logging.getLogger(__name__)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def influence_magnitude(metadata: Mapping[str, object]) -> tuple[float | None, str]:
    """Return (score, source) where larger score means a stronger node.

    Prefers ``influence_raw``. Falls back to ``influence``: if all callers later
    mark the pool as cumulative coverage, the selector inverts the ranking.
    """
    raw = _to_float(metadata.get("influence_raw"))
    if raw is not None:
        return raw, "influence_raw"
    cum = _to_float(metadata.get("influence"))
    if cum is not None:
        return cum, "influence"
    return None, "missing"


def _looks_like_cumulative_coverage(values: Sequence[float]) -> bool:
    """Heuristic for legacy graphs that stored cumulative coverage in ``influence``."""
    if not values:
        return False
    return max(values) <= 1.0 + 1e-9 and min(values) >= 0.0 and max(values) >= 0.5


def select_top_influence(
    graph: CircuitGraph,
    candidates: Sequence[NodeId],
    use_absolute: bool = True,
) -> SelectionResult:
    """Rank candidates by influence magnitude, strongest first.

    Candidates missing an influence value are ranked last (deterministically by
    node string) and excluded from the score map. Raises if no candidate carries
    an influence value at all — that means the graph JSON was exported without
    pruning-time influence and this baseline cannot run on it.
    """
    pool = [node for node in dedupe_preserve_order(candidates) if graph.has_node(node)]
    scores: dict[NodeId, float] = {}
    sources: dict[NodeId, str] = {}
    missing: list[NodeId] = []
    for node in pool:
        magnitude, source = influence_magnitude(graph.metadata(node))
        if magnitude is None:
            missing.append(node)
            continue
        scores[node] = abs(magnitude) if use_absolute else magnitude
        sources[node] = source

    if not scores:
        raise ValueError(
            "No candidate node carries an 'influence' / 'influence_raw' metadata "
            "value; the top-k influence baseline needs a graph JSON exported with "
            "node influence."
        )
    if missing:
        LOGGER.warning(
            "%d candidate(s) have no influence metadata and are ranked last.", len(missing)
        )

    used_raw = any(src == "influence_raw" for src in sources.values())
    used_only_legacy = not used_raw and all(src == "influence" for src in sources.values())
    reverse_legacy = used_only_legacy and _looks_like_cumulative_coverage(list(scores.values()))
    if reverse_legacy:
        # Cumulative coverage: smallest value = strongest node.
        ranking = sorted(scores, key=lambda node: (scores[node], str(node)))
        score_semantics = "legacy_cumulative_ascending"
    else:
        ranking = sorted(scores, key=lambda node: (-scores[node], str(node)))
        score_semantics = "raw_descending" if used_raw else "influence_descending"
    ranking.extend(sorted(missing, key=str))
    return SelectionResult(
        method="influence",
        ranking=ranking,
        scores=scores,
        params={"use_absolute": use_absolute, "score_semantics": score_semantics},
        extras={
            "missing_influence_count": len(missing),
            "used_influence_raw": used_raw,
            "legacy_cumulative_inverted": reverse_legacy,
        },
    )
