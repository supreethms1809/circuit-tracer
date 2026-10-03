"""Native output-edge universe for original ACDC / Syed EAP.

We use the **component output-edge graph**: each attention head and MLP has one
outgoing edge into the residual stream. Edge IDs match InterpBench / MIB
component names (``a{l}.h{h}``, ``m{l}``).

This is the practical Syed/Conmy reduction used when circuits are defined by
which writers remain in the residual path. It is **not** the full pairwise
path-edge DAG (head→MLP vs head→logits distinguished). See
``macag/docs/baseline_original_track.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from macag.scoring_components import component_universe, parse_component


@dataclass(frozen=True)
class OutputEdge:
    """One writer→residual edge."""

    edge_id: str
    kind: str  # "head" | "mlp"
    layer: int
    head: int | None
    hook_name: str

    def topdown_key(self) -> tuple[int, int, str]:
        # Output-side first: higher layer first, MLPs after heads at same layer.
        head_rank = -1 if self.head is None else self.head
        return (-self.layer, head_rank, self.edge_id)


def build_output_edges(model: Any) -> list[OutputEdge]:
    """All head + MLP output edges for a HookedTransformer."""
    cfg = model.cfg
    n_layers = int(cfg.n_layers)
    n_heads = int(cfg.n_heads)
    if not getattr(cfg, "use_attn_result", False):
        raise ValueError(
            "Output-edge baselines require model.cfg.use_attn_result=True "
            "(per-head hook_result)."
        )
    edges: list[OutputEdge] = []
    for edge_id in component_universe(n_layers, n_heads):
        kind, layer, head = parse_component(edge_id)
        if kind == "head":
            hook = f"blocks.{layer}.attn.hook_result"
        else:
            hook = f"blocks.{layer}.hook_mlp_out"
        edges.append(
            OutputEdge(
                edge_id=edge_id,
                kind=kind,
                layer=layer,
                head=head,
                hook_name=hook,
            )
        )
    return edges


def edges_by_id(edges: Sequence[OutputEdge]) -> dict[str, OutputEdge]:
    return {edge.edge_id: edge for edge in edges}


def topdown_order(edges: Sequence[OutputEdge]) -> list[OutputEdge]:
    return sorted(edges, key=lambda edge: edge.topdown_key())
