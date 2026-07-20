"""Automatic supernode discovery utilities for MACAG."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import logging
import math
from typing import Any, Iterable, Sequence

from macag.graph import CircuitGraph, NodeId

LOGGER = logging.getLogger(__name__)


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    # NaN is truthy and poisons salience sums and sort keys; treat as missing.
    if math.isnan(parsed):
        return None
    return parsed


def _normalize_feature_types(feature_types: Sequence[str] | None) -> set[str] | None:
    if feature_types is None:
        return None
    normalized = {feature_type.strip().lower() for feature_type in feature_types if feature_type.strip()}
    return normalized or None


def node_salience(
    metadata: dict[str, Any],
    influence_weight: float = 1.0,
    activation_weight: float = 0.2,
    token_prob_weight: float = 0.0,
) -> float:
    """Compute a deterministic salience score from available node metrics."""
    influence = _to_float(metadata.get("influence")) or 0.0
    activation = _to_float(metadata.get("activation")) or 0.0
    token_prob = _to_float(metadata.get("token_prob")) or 0.0
    return (
        influence_weight * abs(influence)
        + activation_weight * abs(activation)
        + token_prob_weight * abs(token_prob)
    )


def rank_nodes_by_salience(
    graph: CircuitGraph,
    feature_types: Sequence[str] | None = ("cross layer transcoder",),
    top_k: int | None = 200,
    min_salience: float = 0.0,
    influence_weight: float = 1.0,
    activation_weight: float = 0.2,
    token_prob_weight: float = 0.0,
) -> list[tuple[NodeId, float]]:
    """Return nodes ranked by salience descending."""
    allowed_feature_types = _normalize_feature_types(feature_types)
    ranked: list[tuple[NodeId, float]] = []
    for node in graph.nodes():
        metadata = graph.metadata(node)
        if allowed_feature_types is not None:
            feature_type = str(metadata.get("feature_type", "")).strip().lower()
            if feature_type not in allowed_feature_types:
                continue
        score = node_salience(
            metadata=metadata,
            influence_weight=influence_weight,
            activation_weight=activation_weight,
            token_prob_weight=token_prob_weight,
        )
        if score < min_salience:
            continue
        ranked.append((node, score))

    ranked.sort(key=lambda item: (-item[1], str(item[0])))
    if top_k is not None and top_k >= 0:
        return ranked[:top_k]
    return ranked


def _connected_components(graph: CircuitGraph, nodes: Iterable[NodeId]) -> list[list[NodeId]]:
    node_set = set(nodes)
    visited: set[NodeId] = set()
    components: list[list[NodeId]] = []

    for start in sorted(node_set, key=str):
        if start in visited:
            continue
        queue: deque[NodeId] = deque([start])
        visited.add(start)
        component: list[NodeId] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for nxt in graph.neighbors(current):
                if nxt in node_set and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        components.append(component)
    return components


def _ctx_bucket(graph: CircuitGraph, node: NodeId) -> tuple[int, int]:
    """Sort key for ctx_idx buckets; unknown goes last."""
    metadata = graph.metadata(node)
    ctx = metadata.get("ctx_idx")
    if isinstance(ctx, int):
        return (0, ctx)
    parsed = _to_float(ctx)
    if parsed is not None and float(parsed).is_integer():
        return (0, int(parsed))
    return (1, 0)


def _split_component_by_ctx(graph: CircuitGraph, component: list[NodeId]) -> list[list[NodeId]]:
    by_ctx: dict[tuple[int, int], list[NodeId]] = defaultdict(list)
    for node in component:
        by_ctx[_ctx_bucket(graph, node)].append(node)
    if len(by_ctx) <= 1:
        return [component]
    return [by_ctx[key] for key in sorted(by_ctx.keys())]


def _chunk_group(
    group: list[NodeId],
    scores: dict[NodeId, float],
    max_group_size: int,
) -> list[list[NodeId]]:
    if max_group_size <= 0 or len(group) <= max_group_size:
        return [group]
    ranked = sorted(group, key=lambda node: (-scores.get(node, 0.0), str(node)))
    return [ranked[i : i + max_group_size] for i in range(0, len(ranked), max_group_size)]


def _common_label_snippet(graph: CircuitGraph, nodes: list[NodeId]) -> str:
    clerps = [str(graph.metadata(node).get("clerp", "")).strip() for node in nodes]
    clerps = [clerp for clerp in clerps if clerp]
    if not clerps:
        return ""
    top = Counter(clerps).most_common(1)[0][0]
    compact = "_".join(top.split())[:24]
    return compact


def _group_ctx_suffix(graph: CircuitGraph, nodes: list[NodeId]) -> str:
    ctx_values: set[int] = set()
    for node in nodes:
        metadata = graph.metadata(node)
        ctx_raw = metadata.get("ctx_idx")
        if isinstance(ctx_raw, int):
            ctx_values.add(ctx_raw)
            continue
        parsed = _to_float(ctx_raw)
        if parsed is not None and float(parsed).is_integer():
            ctx_values.add(int(parsed))
    if len(ctx_values) == 1:
        return f"pos{next(iter(ctx_values))}"
    return ""


def propose_supernodes(
    graph: CircuitGraph,
    top_k: int = 200,
    min_group_size: int = 3,
    max_group_size: int = 12,
    min_salience: float = 0.0,
    label_prefix: str = "AUTO",
    feature_types: Sequence[str] | None = ("cross layer transcoder",),
    split_by_ctx: bool = True,
    influence_weight: float = 1.0,
    activation_weight: float = 0.2,
    token_prob_weight: float = 0.0,
) -> list[list[str]]:
    """
    Propose supernodes from graph metrics and connectivity.

    Returns the same shape circuit-tracer expects in `qParams.supernodes`:
    [[label, node_id_1, node_id_2, ...], ...]
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive.")
    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive.")
    if min_group_size > max_group_size:
        raise ValueError("min_group_size cannot exceed max_group_size.")
    if min_salience < 0.0:
        raise ValueError("min_salience must be non-negative.")

    ranked = rank_nodes_by_salience(
        graph=graph,
        feature_types=feature_types,
        top_k=top_k,
        min_salience=min_salience,
        influence_weight=influence_weight,
        activation_weight=activation_weight,
        token_prob_weight=token_prob_weight,
    )
    if not ranked:
        return []

    scores = {node: score for node, score in ranked}
    selected_nodes = [node for node, _ in ranked]
    components = _connected_components(graph, selected_nodes)
    components.sort(
        key=lambda component: (
            -sum(scores.get(node, 0.0) for node in component),
            min(str(node) for node in component),
        )
    )

    groups: list[list[NodeId]] = []
    dropped_below_min = 0
    for component in components:
        split_components = _split_component_by_ctx(graph, component) if split_by_ctx else [component]
        for split_component in split_components:
            for chunk in _chunk_group(split_component, scores=scores, max_group_size=max_group_size):
                if len(chunk) >= min_group_size:
                    groups.append(chunk)
                else:
                    dropped_below_min += len(chunk)
    if dropped_below_min:
        LOGGER.info(
            "propose_supernodes dropped %d salient node(s) in groups smaller than "
            "min_group_size=%d (component fragments / chunk tails)",
            dropped_below_min,
            min_group_size,
        )

    if not groups and len(selected_nodes) >= min_group_size:
        fallback = selected_nodes[:max_group_size]
        groups.append(fallback)

    supernodes: list[list[str]] = []
    for idx, group in enumerate(groups, start=1):
        ordered_nodes = sorted(group, key=lambda node: (-scores.get(node, 0.0), str(node)))
        snippet = _common_label_snippet(graph, ordered_nodes)
        ctx_suffix = _group_ctx_suffix(graph, ordered_nodes)
        label_parts = [f"{label_prefix}:{idx:02d}"]
        if snippet:
            label_parts.append(snippet)
        if ctx_suffix:
            label_parts.append(ctx_suffix)
        label = ":".join(label_parts)
        supernodes.append([label, *[str(node) for node in ordered_nodes]])

    return supernodes


def supernode_candidates(supernodes: Sequence[Sequence[str]]) -> list[str]:
    """Flatten supernodes into sorted unique node IDs."""
    nodes: set[str] = set()
    for supernode in supernodes:
        for node in supernode[1:]:
            nodes.add(str(node))
    return sorted(nodes)


def merge_supernodes_into_payload(
    graph_payload: dict[str, Any],
    supernodes: Sequence[Sequence[str]],
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Merge proposed supernodes into graph payload qParams."""
    qparams = graph_payload.get("qParams")
    if not isinstance(qparams, dict):
        qparams = {}
        graph_payload["qParams"] = qparams
    existing = qparams.get("supernodes")
    if not isinstance(existing, list):
        existing = []
    new_groups = [[str(item) for item in group] for group in supernodes]
    qparams["supernodes"] = new_groups if replace_existing else list(existing) + new_groups
    return graph_payload
