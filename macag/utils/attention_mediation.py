"""Attention-mediation diagnostic from paired frozen/unfrozen Game 1 runs.

The headline signal (macag.md §10.4) is the sign of ``recoverable_range`` under
each attention-freezing convention: a range that is negative with attention
frozen but non-negative once unfrozen fingerprints an attention-mediated
behavior — ablating features barely moves the score while the frozen pattern
carries the answer, and the features only become load-bearing when attention is
allowed to recompute. Secondary signals compare the two minimal evidence sets:
overlap, and how many upstream / early-layer features the unfrozen run recruits
that the frozen run dropped as redundant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from macag.graph import CircuitGraph, NodeId
from macag.utils.metrics import FaithfulnessMetrics, overlap_rate

VERDICT_ATTENTION_MEDIATED = "attention_mediated"
VERDICT_FEATURE_MEDIATED = "feature_mediated"
VERDICT_INDETERMINATE = "indeterminate"

# Node feature_types whose layer index counts toward model depth. Logit and
# embedding nodes carry sentinel layers ("E", model depth) and are excluded,
# matching the convention in experiments/analyze_macag_sweep.py.
_DEPTH_FEATURE_TYPES = frozenset({"cross layer transcoder", "mlp reconstruction error"})


@dataclass(frozen=True)
class AttentionMediationDiagnostic:
    """Paired frozen/unfrozen Game 1 comparison for one prompt.

    Count fields are ``None`` when the graph carries no resolvable layer/position
    information; keys are always present so serialized output has a stable schema.
    """

    range_frozen: float
    range_unfrozen: float
    range_flip: bool
    reverse_flip: bool
    verdict: str
    evidence_size_frozen: int
    evidence_size_unfrozen: int
    evidence_jaccard: float
    evidence_shared: list[str]
    evidence_only_frozen: list[str]
    evidence_only_unfrozen: list[str]
    upstream_count_frozen: int | None
    upstream_count_unfrozen: int | None
    early_count_frozen: int | None
    early_count_unfrozen: int | None
    n_layers: int | None
    final_ctx_idx: int | None
    # P1-5: dead-band half-width used for the verdict (0.0 = legacy strict-zero).
    margin: float = 0.0
    frozen_zone: str = "unknown"
    unfrozen_zone: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _node_layer_pos(graph: CircuitGraph, node: NodeId) -> tuple[int | None, int | None]:
    """Resolve a node's (layer, ctx position), metadata first, then the
    ``{layer}_{feature}_{pos}`` node-ID convention, else (None, None)."""
    layer: int | None = None
    pos: int | None = None
    meta = graph.metadata(node) if graph.has_node(node) else {}
    raw_layer = meta.get("layer")
    if raw_layer is not None:
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError):
            layer = None  # e.g. embedding nodes use layer "E"
    raw_pos = meta.get("ctx_idx")
    if raw_pos is not None:
        try:
            pos = int(raw_pos)
        except (TypeError, ValueError):
            pos = None
    if layer is not None and pos is not None:
        return layer, pos

    parts = str(node).split("_")
    if len(parts) == 3:
        try:
            parsed_layer, parsed_pos = int(parts[0]), int(parts[2])
        except ValueError:
            return layer, pos
        return (layer if layer is not None else parsed_layer, pos if pos is not None else parsed_pos)
    return layer, pos


def _infer_depth_and_final_pos(graph: CircuitGraph) -> tuple[int | None, int | None]:
    """Infer (n_layers, final_ctx_idx) from graph node metadata / node IDs.

    n_layers comes from feature/error nodes only (logit/embedding layers are
    sentinels); final_ctx_idx from all nodes — logit nodes sit at the final
    prompt position, so the max over every resolvable node is the prediction
    position without needing the prompt token list.
    """
    max_layer: int | None = None
    max_pos: int | None = None
    for node in graph.nodes():
        layer, pos = _node_layer_pos(graph, node)
        if pos is not None and (max_pos is None or pos > max_pos):
            max_pos = pos
        if layer is None:
            continue
        feature_type = str(graph.metadata(node).get("feature_type", "")).strip().lower()
        if feature_type and feature_type not in _DEPTH_FEATURE_TYPES:
            continue
        if max_layer is None or layer > max_layer:
            max_layer = layer
    n_layers = max_layer + 1 if max_layer is not None else None
    return n_layers, max_pos


def _upstream_early_counts(
    graph: CircuitGraph,
    evidence: set[NodeId],
    n_layers: int | None,
    final_ctx_idx: int | None,
) -> tuple[int | None, int | None]:
    """Count evidence nodes upstream of the prediction position and in early
    layers (< n_layers / 3). Nodes with unresolvable layer/pos are skipped."""
    upstream: int | None = 0 if final_ctx_idx is not None else None
    early: int | None = 0 if n_layers is not None else None
    for node in evidence:
        layer, pos = _node_layer_pos(graph, node)
        if upstream is not None and pos is not None and pos < final_ctx_idx:
            upstream += 1
        if early is not None and layer is not None and layer < n_layers / 3.0:
            early += 1
    return upstream, early


def _zone(value: float, margin: float) -> str:
    """Classify a recoverable range against the dead-band ``[-margin, +margin]``."""
    if value < -margin:
        return "negative"
    if value > margin:
        return "positive"
    return "band"


def compute_attention_mediation_diagnostic(
    graph: CircuitGraph,
    frozen_metrics: FaithfulnessMetrics,
    unfrozen_metrics: FaithfulnessMetrics,
    frozen_evidence: set[NodeId],
    unfrozen_evidence: set[NodeId],
    margin: float = 0.0,
) -> AttentionMediationDiagnostic:
    """Build the per-prompt diagnostic from matched frozen/unfrozen Game 1 legs.

    Verdict rule with dead-band ``margin`` (P1-5; default 0.0 = legacy
    strict-zero rule, except the exact (0, 0) corner which is now
    ``indeterminate`` rather than ``feature_mediated`` — a zero recoverable
    range under both conventions means nothing was recovered):

      * frozen robustly negative, unfrozen not robustly negative
        -> attention_mediated (the §10.4 flip)
      * neither robustly negative, at least one robustly positive
        -> feature_mediated
      * frozen not robustly negative, unfrozen robustly negative
        -> indeterminate, reverse_flip=True (unexpected inversion:
        recomputed attention destroyed recoverability)
      * otherwise (both robustly negative, or both inside the band)
        -> indeterminate

    Raw ranges are always reported alongside the verdict so confidence bands
    can be applied downstream; ``frozen_zone``/``unfrozen_zone`` record the
    per-leg ``negative``/``band``/``positive`` classification.
    """
    if margin < 0.0:
        raise ValueError(f"Verdict margin must be non-negative, got {margin}.")
    range_frozen = frozen_metrics.recoverable_range
    range_unfrozen = unfrozen_metrics.recoverable_range
    frozen_zone = _zone(range_frozen, margin)
    unfrozen_zone = _zone(range_unfrozen, margin)
    frozen_neg = frozen_zone == "negative"
    unfrozen_neg = unfrozen_zone == "negative"
    range_flip = frozen_neg and not unfrozen_neg
    reverse_flip = not frozen_neg and unfrozen_neg
    if range_flip:
        verdict = VERDICT_ATTENTION_MEDIATED
    elif not frozen_neg and not unfrozen_neg and (
        frozen_zone == "positive" or unfrozen_zone == "positive"
    ):
        verdict = VERDICT_FEATURE_MEDIATED
    else:
        verdict = VERDICT_INDETERMINATE

    if not frozen_evidence and not unfrozen_evidence:
        jaccard = 1.0
    else:
        jaccard = overlap_rate(frozen_evidence, unfrozen_evidence)

    n_layers, final_ctx_idx = _infer_depth_and_final_pos(graph)
    upstream_frozen, early_frozen = _upstream_early_counts(
        graph, frozen_evidence, n_layers, final_ctx_idx
    )
    upstream_unfrozen, early_unfrozen = _upstream_early_counts(
        graph, unfrozen_evidence, n_layers, final_ctx_idx
    )

    return AttentionMediationDiagnostic(
        range_frozen=range_frozen,
        range_unfrozen=range_unfrozen,
        range_flip=range_flip,
        reverse_flip=reverse_flip,
        verdict=verdict,
        evidence_size_frozen=len(frozen_evidence),
        evidence_size_unfrozen=len(unfrozen_evidence),
        evidence_jaccard=jaccard,
        evidence_shared=sorted(str(node) for node in frozen_evidence & unfrozen_evidence),
        evidence_only_frozen=sorted(str(node) for node in frozen_evidence - unfrozen_evidence),
        evidence_only_unfrozen=sorted(str(node) for node in unfrozen_evidence - frozen_evidence),
        upstream_count_frozen=upstream_frozen,
        upstream_count_unfrozen=upstream_unfrozen,
        early_count_frozen=early_frozen,
        early_count_unfrozen=early_unfrozen,
        n_layers=n_layers,
        final_ctx_idx=final_ctx_idx,
        margin=margin,
        frozen_zone=frozen_zone,
        unfrozen_zone=unfrozen_zone,
    )
