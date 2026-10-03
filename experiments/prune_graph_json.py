#!/usr/bin/env python3
"""Prune a circuit-tracer frontend graph JSON at Neuronpedia thresholds.

This is an offline analogue of ``circuit_tracer.graph.prune_graph`` applied to
an already-exported attribution JSON, so the v4 unpruned graph is not rewritten.

Node cutoff uses stored ``influence_raw`` mass (default 0.8). Edge cutoff uses
absolute adjacency ``weight`` mass among surviving nodes (default 0.98). Isolated
feature / error nodes are then dropped. Embedding and logit nodes are always kept.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ALWAYS_KEEP_TYPES = frozenset({"embedding", "logit"})
FEATURE_TYPE = "cross layer transcoder"
ERROR_TYPE = "mlp reconstruction error"


def _node_id(node: Mapping[str, Any]) -> str:
    node_id = node.get("node_id", node.get("id"))
    if node_id is None:
        raise ValueError("graph node is missing both 'node_id' and 'id'")
    return str(node_id)


def _feature_type(node: Mapping[str, Any]) -> str:
    return str(node.get("feature_type") or "").strip().lower()


def _raw_influence(node: Mapping[str, Any]) -> float:
    value = node.get("influence_raw", node.get("influence"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _abs_weight(edge: Mapping[str, Any]) -> float:
    try:
        return abs(float(edge.get("weight") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _mass_cutoff(scores: list[float], threshold: float) -> float:
    """Smallest score that still covers ``threshold`` of total mass when ranked desc."""
    if threshold >= 1.0:
        return min(scores) if scores else 0.0
    total = sum(scores)
    if total <= 0.0 or not scores:
        return 0.0
    acc = 0.0
    cutoff = scores[0]
    for score in sorted(scores, reverse=True):
        acc += score
        cutoff = score
        if acc / total >= threshold:
            break
    return cutoff


def select_nodes_by_influence(
    nodes: Iterable[Mapping[str, Any]],
    *,
    node_threshold: float,
) -> set[str]:
    node_list = list(nodes)
    scores = [_raw_influence(node) for node in node_list]
    cutoff = _mass_cutoff(scores, node_threshold)
    keep: set[str] = set()
    for node, score in zip(node_list, scores):
        if _feature_type(node) in ALWAYS_KEEP_TYPES or score >= cutoff:
            keep.add(_node_id(node))
    return keep


def select_edges_by_weight(
    edges: Iterable[Mapping[str, Any]],
    keep_nodes: set[str],
    *,
    edge_threshold: float,
) -> list[dict[str, Any]]:
    candidates = [
        dict(edge)
        for edge in edges
        if str(edge.get("source")) in keep_nodes and str(edge.get("target")) in keep_nodes
    ]
    weights = [_abs_weight(edge) for edge in candidates]
    cutoff = _mass_cutoff(weights, edge_threshold)
    return [edge for edge, weight in zip(candidates, weights) if weight >= cutoff]


def drop_disconnected(
    nodes: Iterable[Mapping[str, Any]],
    edges: list[dict[str, Any]],
    keep_nodes: set[str],
) -> tuple[set[str], list[dict[str, Any]]]:
    types = {_node_id(node): _feature_type(node) for node in nodes}
    remaining_nodes = set(keep_nodes)
    remaining_edges = list(edges)
    changed = True
    while changed:
        successors: dict[str, set[str]] = defaultdict(set)
        predecessors: dict[str, set[str]] = defaultdict(set)
        for edge in remaining_edges:
            source = str(edge["source"])
            target = str(edge["target"])
            successors[source].add(target)
            predecessors[target].add(source)
        drop: set[str] = set()
        for node_id in remaining_nodes:
            kind = types.get(node_id, "")
            if kind in ALWAYS_KEEP_TYPES:
                continue
            if kind == FEATURE_TYPE and (
                not predecessors[node_id] or not successors[node_id]
            ):
                drop.add(node_id)
            elif kind == ERROR_TYPE and not successors[node_id]:
                drop.add(node_id)
        if not drop:
            changed = False
            continue
        remaining_nodes -= drop
        remaining_edges = [
            edge
            for edge in remaining_edges
            if str(edge["source"]) in remaining_nodes
            and str(edge["target"]) in remaining_nodes
        ]
    return remaining_nodes, remaining_edges


def prune_graph_payload(
    payload: Mapping[str, Any],
    *,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = list(payload.get("nodes") or [])
    edges = list(payload.get("links") or payload.get("edges") or [])
    before_types = Counter(_feature_type(node) for node in nodes)
    keep = select_nodes_by_influence(nodes, node_threshold=node_threshold)
    after_node = Counter(
        _feature_type(node) for node in nodes if _node_id(node) in keep
    )
    pruned_edges = select_edges_by_weight(
        edges, keep, edge_threshold=edge_threshold
    )
    keep, pruned_edges = drop_disconnected(nodes, pruned_edges, keep)
    pruned_nodes = [dict(node) for node in nodes if _node_id(node) in keep]
    after_types = Counter(_feature_type(node) for node in pruned_nodes)

    metadata = dict(payload.get("metadata") or {})
    metadata["node_threshold"] = float(node_threshold)
    metadata["edge_threshold"] = float(edge_threshold)
    metadata["pruned_from_unpruned"] = True

    pruned: dict[str, Any] = {
        "metadata": metadata,
        "qParams": payload.get("qParams") or {},
        "nodes": pruned_nodes,
        "links": pruned_edges,
    }
    stats = {
        "node_threshold": float(node_threshold),
        "edge_threshold": float(edge_threshold),
        "n_nodes_before": len(nodes),
        "n_nodes_after": len(pruned_nodes),
        "n_edges_before": len(edges),
        "n_edges_after": len(pruned_edges),
        "n_features_before": int(before_types.get(FEATURE_TYPE, 0)),
        "n_features_after": int(after_types.get(FEATURE_TYPE, 0)),
        "types_before": dict(before_types),
        "types_after_node_cutoff": dict(after_node),
        "types_after": dict(after_types),
        "feature_node_ids": sorted(
            _node_id(node)
            for node in pruned_nodes
            if _feature_type(node) == FEATURE_TYPE
        ),
    }
    return pruned, stats


def prune_graph_file(
    source: Path,
    dest: Path,
    *,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(source.read_text())
    pruned, stats = prune_graph_payload(
        payload,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    stats["source_graph"] = str(source)
    stats["pruned_graph"] = str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(pruned) + "\n")
    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--stats-json", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    stats = prune_graph_file(
        args.input,
        args.output,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        stats_path=args.stats_json,
    )
    print(
        "pruned "
        f"features {stats['n_features_before']} -> {stats['n_features_after']} "
        f"edges {stats['n_edges_before']} -> {stats['n_edges_after']} "
        f"wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
