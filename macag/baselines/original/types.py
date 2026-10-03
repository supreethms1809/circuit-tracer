"""Shared result types for the original-pipeline baseline track.

These methods operate on **native edges / component writers**, not CLT feature
nodes. Their outputs live in ``macag_original_baselines.json`` and must never be
merged into rematch ``macag_baselines.json`` Jaccard tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OriginalCircuitResult:
    """Finished circuit from an original-pipeline selector."""

    method: str
    edges_kept: list[str]
    size: int
    scores: dict[str, float | None]
    selection_stats: dict[str, int | float]
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    reason: str | None = None
    ranking: list[str] | None = None
    edge_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "method": self.method,
            "edges_kept": list(self.edges_kept),
            "size": int(self.size),
            "scores": dict(self.scores),
            "selection_stats": dict(self.selection_stats),
            "params": dict(self.params),
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.ranking is not None:
            payload["ranking"] = list(self.ranking)
        if self.edge_scores is not None:
            payload["edge_scores"] = dict(self.edge_scores)
        return payload


def unavailable_result(method: str, reason: str, **params: Any) -> OriginalCircuitResult:
    return OriginalCircuitResult(
        method=method,
        edges_kept=[],
        size=0,
        scores={"logit_gap": None, "kl": None},
        selection_stats={},
        params=dict(params),
        status="unavailable",
        reason=reason,
    )
