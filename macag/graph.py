"""Graph utilities for MACAG evidence-allocation games."""

from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

NodeId = Hashable


def _node_sort_key(node: NodeId) -> str:
    return str(node)


def grow_connected_frontier(
    graph: "CircuitGraph",
    ranked_nodes: Sequence[NodeId],
    top_k: int,
) -> list[NodeId]:
    """Pick up to `top_k` nodes that form a connected subcircuit, biased by rank.

    Seeds with the top-ranked node, then walks the remaining nodes in rank order
    keeping any that are connected (through the full graph) to the seed's
    component. If the seed's component is exhausted before `top_k` is reached, the
    remaining slots are filled by raw rank. Used by the connectivity-aware
    prefilters so a singleton-gain top-k truncation does not hand the connected
    greedy a mutually-disconnected pool (L2).
    """
    if top_k <= 0 or not ranked_nodes:
        return []
    seed = ranked_nodes[0]
    kept: list[NodeId] = [seed]
    kept_set: set[NodeId] = {seed}
    for node in ranked_nodes[1:]:
        if len(kept) >= top_k:
            break
        if graph.connected_through({seed, node}):
            kept.append(node)
            kept_set.add(node)
    if len(kept) < top_k:
        for node in ranked_nodes:
            if len(kept) >= top_k:
                break
            if node not in kept_set:
                kept.append(node)
                kept_set.add(node)
    return kept


class CircuitGraph:
    """A lightweight directed graph wrapper with node metadata."""

    def __init__(
        self,
        nodes: list[NodeId] | None = None,
        edges: list[tuple[NodeId, NodeId]] | None = None,
        node_metadata: Mapping[NodeId, Mapping[str, Any]] | None = None,
    ) -> None:
        self._node_metadata: dict[NodeId, dict[str, Any]] = {}
        self._succ: dict[NodeId, set[NodeId]] = defaultdict(set)
        self._pred: dict[NodeId, set[NodeId]] = defaultdict(set)

        if nodes:
            for node in nodes:
                metadata = dict(node_metadata.get(node, {})) if node_metadata else {}
                self.add_node(node, metadata=metadata)
        if edges:
            for source, target in edges:
                self.add_edge(source, target)

    def add_node(self, node: NodeId, metadata: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        if node not in self._node_metadata:
            self._node_metadata[node] = {}
        if metadata:
            self._node_metadata[node].update(dict(metadata))
        if kwargs:
            self._node_metadata[node].update(kwargs)
        _ = self._succ[node]
        _ = self._pred[node]

    def add_edge(self, source: NodeId, target: NodeId) -> None:
        if source not in self._node_metadata:
            self.add_node(source)
        if target not in self._node_metadata:
            self.add_node(target)
        self._succ[source].add(target)
        self._pred[target].add(source)

    def nodes(self) -> list[NodeId]:
        return list(self._node_metadata.keys())

    def edges(self) -> list[tuple[NodeId, NodeId]]:
        edges: list[tuple[NodeId, NodeId]] = []
        for source in self.nodes():
            for target in sorted(self._succ[source], key=_node_sort_key):
                edges.append((source, target))
        return edges

    def has_node(self, node: NodeId) -> bool:
        return node in self._node_metadata

    def has_edge(self, source: NodeId, target: NodeId) -> bool:
        return target in self._succ.get(source, set())

    def metadata(self, node: NodeId) -> dict[str, Any]:
        return dict(self._node_metadata[node])

    def neighbors(self, node: NodeId) -> set[NodeId]:
        return set(self._succ.get(node, set())) | set(self._pred.get(node, set()))

    def subgraph(self, node_subset: set[NodeId] | list[NodeId] | tuple[NodeId, ...]) -> "CircuitGraph":
        selected = set(node_subset)
        metadata = {node: self._node_metadata[node] for node in selected if node in self._node_metadata}
        edges = [
            (source, target)
            for source in selected
            for target in self._succ.get(source, set())
            if target in selected
        ]
        return CircuitGraph(nodes=list(selected), edges=edges, node_metadata=metadata)

    def induced_subgraph(self, node_subset: set[NodeId] | list[NodeId] | tuple[NodeId, ...]) -> "CircuitGraph":
        return self.subgraph(node_subset)

    def is_weakly_connected(self, node_subset: set[NodeId] | None = None) -> bool:
        selected = set(node_subset) if node_subset is not None else set(self.nodes())
        if len(selected) <= 1:
            return True
        if not selected:
            return True

        start = next(iter(selected))
        queue: deque[NodeId] = deque([start])
        visited: set[NodeId] = {start}

        while queue:
            current = queue.popleft()
            for nxt in self.neighbors(current):
                if nxt in selected and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        return visited == selected

    def connected_through(
        self,
        node_subset: set[NodeId] | list[NodeId] | tuple[NodeId, ...],
        *,
        allow_intermediate: bool = True,
    ) -> bool:
        """Return True iff every node in `node_subset` lies in one weakly-connected
        component of the *full* attribution graph.

        Unlike :meth:`is_weakly_connected`, which only counts edges whose endpoints
        are both inside the subset, this routes connectivity *through* intermediate
        nodes that are not part of the subset (error / embedding / other feature
        nodes). This matches the circuit intuition that two transcoder features are
        "connected" when they belong to the same sub-circuit even if they only
        interact via an intermediate node (L1). Setting ``allow_intermediate=False``
        recovers the stricter subset-internal behavior.
        """
        selected = set(node_subset)
        if len(selected) <= 1:
            return True
        if not allow_intermediate:
            return self.is_weakly_connected(selected)

        missing = [node for node in selected if node not in self._node_metadata]
        if missing:
            return False

        # BFS over the full graph starting from one subset node; succeed as soon as
        # all subset nodes are reached.
        start = next(iter(selected))
        queue: deque[NodeId] = deque([start])
        visited: set[NodeId] = {start}
        remaining = selected - {start}

        while queue and remaining:
            current = queue.popleft()
            for nxt in self.neighbors(current):
                if nxt not in visited:
                    visited.add(nxt)
                    remaining.discard(nxt)
                    queue.append(nxt)

        return not remaining

    def to_dict(self) -> dict[str, Any]:
        node_entries = []
        for node in self.nodes():
            node_entries.append({"id": node, **self.metadata(node)})
        edge_entries = []
        for source, target in self.edges():
            edge_entries.append({"source": source, "target": target})
        return {"nodes": node_entries, "edges": edge_entries}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CircuitGraph":
        nodes_raw = payload.get("nodes", [])
        # Circuit tracer JSON uses "links"; fallback to "edges".
        edges_raw = payload.get("edges", payload.get("links", []))

        node_ids: list[NodeId] = []
        node_metadata: dict[NodeId, dict[str, Any]] = {}

        for node in nodes_raw:
            if isinstance(node, Mapping):
                node_id = node.get("id", node.get("node_id"))
                if node_id is None:
                    raise ValueError("Node entry is missing both 'id' and 'node_id'.")
                node_ids.append(node_id)
                meta = {
                    k: v
                    for k, v in node.items()
                    if k not in {"id", "node_id"}
                }
                node_metadata[node_id] = meta
            else:
                node_ids.append(node)
                node_metadata[node] = {}

        edges: list[tuple[NodeId, NodeId]] = []
        for edge in edges_raw:
            if isinstance(edge, Mapping):
                source = edge.get("source")
                target = edge.get("target")
                if source is None or target is None:
                    raise ValueError("Edge mapping must include 'source' and 'target'.")
                edges.append((source, target))
            else:
                if len(edge) != 2:
                    raise ValueError("Edge tuple/list must contain exactly two entries.")
                edges.append((edge[0], edge[1]))

        graph = cls(nodes=node_ids, node_metadata=node_metadata)
        for source, target in edges:
            graph.add_edge(source, target)
        return graph

    @classmethod
    def from_json(cls, path: str | Path) -> "CircuitGraph":
        resolved = Path(path)
        try:
            text = resolved.read_text()
        except FileNotFoundError:
            raise FileNotFoundError(f"Graph JSON file not found: {resolved}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in graph file {resolved}: {exc}") from exc
        return cls.from_dict(payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_networkx(cls, nx_graph: Any) -> "CircuitGraph":
        graph = cls()
        for node, metadata in nx_graph.nodes(data=True):
            graph.add_node(node, metadata=metadata)
        for source, target in nx_graph.edges():
            graph.add_edge(source, target)
        return graph
