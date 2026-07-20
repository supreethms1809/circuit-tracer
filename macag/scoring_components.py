"""Native-component (attention-head / MLP) intervention scorer for MACAG.

MACAG's oracle contract only needs the four intervention modes plus the
universe protocol, so it is not tied to transcoder features. This backend
ablates a TransformerLens ``HookedTransformer``'s **native components** —
attention heads ``a{l}.h{h}`` and MLPs ``m{l}`` (the InterpBench / MIB node
naming) — via forward hooks, enabling MACAG games on models with *known
ground-truth circuits* (InterpBench, roadmap B4.1 part b) where CLT features
don't exist.

Requires ``model.cfg.use_attn_result = True`` (head ablation hooks
``blocks.{l}.attn.hook_result``; the MIB InterpBench loader sets it).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from macag.graph import NodeId
from macag.scoring import ScoreKind, TargetId, compute_kl_score, compute_scalar_score

LOGGER = logging.getLogger(__name__)

_HEAD_RE = re.compile(r"^a(\d+)\.h(\d+)$")
_MLP_RE = re.compile(r"^m(\d+)$")


def component_universe(n_layers: int, n_heads: int) -> list[str]:
    """All head + MLP node IDs in InterpBench naming, sorted for determinism."""
    nodes = [f"a{layer}.h{head}" for layer in range(n_layers) for head in range(n_heads)]
    nodes += [f"m{layer}" for layer in range(n_layers)]
    return sorted(nodes)


def parse_component(node: str) -> tuple[str, int, int | None]:
    """``a3.h1`` -> ("head", 3, 1); ``m2`` -> ("mlp", 2, None)."""
    head_match = _HEAD_RE.match(node)
    if head_match:
        return "head", int(head_match.group(1)), int(head_match.group(2))
    mlp_match = _MLP_RE.match(node)
    if mlp_match:
        return "mlp", int(mlp_match.group(1)), None
    raise ValueError(f"Unrecognized component node ID: {node!r} (expected a{{l}}.h{{h}} or m{{l}})")


@dataclass
class HookedComponentInterventionScorer:
    """Four-mode scorer that zero-ablates native components via TL hooks."""

    model: Any
    prompt: str
    target_to_logit_idx: Mapping[TargetId, int]
    score_kind: ScoreKind = "logit_gap"
    foil_by_target: Mapping[TargetId, TargetId] | None = None
    default_foil: TargetId | None = None
    node_universe: set[NodeId] | None = None
    _all_nodes: set[NodeId] = field(init=False, repr=False)
    _tokens: Any = field(init=False, default=None, repr=False)
    _ref_logits: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        cfg = self.model.cfg
        supported = set(component_universe(int(cfg.n_layers), int(cfg.n_heads)))
        if self.node_universe is None:
            self._all_nodes = supported
        else:
            self._all_nodes = set(self.node_universe) & supported
            if not self._all_nodes:
                raise ValueError(
                    "node_universe has no overlap with this model's components "
                    f"(n_layers={cfg.n_layers}, n_heads={cfg.n_heads})."
                )
        if not getattr(cfg, "use_attn_result", False):
            raise ValueError(
                "HookedComponentInterventionScorer needs model.cfg.use_attn_result=True "
                "(head ablation hooks attn.hook_result)."
            )

    # ---- universe protocol (mirrors ReplacementModelInterventionScorer) ----
    def supported_nodes(self) -> set[NodeId]:
        cfg = self.model.cfg
        return set(component_universe(int(cfg.n_layers), int(cfg.n_heads)))

    def intervention_universe(self) -> set[NodeId]:
        return set(self._all_nodes)

    def universe_fingerprint(self) -> frozenset[NodeId]:
        return frozenset(self._all_nodes)

    def restrict_universe(self, nodes: set[NodeId] | list[NodeId] | tuple[NodeId, ...]) -> None:
        restricted = set(nodes) & self.supported_nodes()
        if not restricted:
            raise ValueError(
                "restrict_universe produced an empty component universe; "
                "check node IDs (a{l}.h{h} / m{l})."
            )
        self._all_nodes = restricted

    # ------------------------------------------------------------- scoring
    def _prompt_tokens(self) -> Any:
        if self._tokens is None:
            self._tokens = self.model.to_tokens(self.prompt)
        return self._tokens

    def _hooks_for(self, nodes_to_ablate: set[NodeId]) -> list[tuple[str, Any]]:
        heads_by_layer: dict[int, list[int]] = {}
        mlp_layers: list[int] = []
        for node in nodes_to_ablate:
            kind, layer, head = parse_component(str(node))
            if kind == "head" and head is not None:
                heads_by_layer.setdefault(layer, []).append(head)
            else:
                mlp_layers.append(layer)

        def zero_heads(heads: list[int]):
            def hook(value: Any, hook: Any = None) -> Any:  # [batch, pos, head, d]
                value[:, :, heads, :] = 0.0
                return value

            return hook

        def zero_mlp(value: Any, hook: Any = None) -> Any:  # [batch, pos, d]
            value[:] = 0.0
            return value

        fwd_hooks: list[tuple[str, Any]] = []
        for layer, heads in sorted(heads_by_layer.items()):
            fwd_hooks.append((f"blocks.{layer}.attn.hook_result", zero_heads(sorted(heads))))
        for layer in sorted(set(mlp_layers)):
            fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", zero_mlp))
        return fwd_hooks

    def _run_logits(self, nodes_to_ablate: set[NodeId]) -> Any:
        import torch

        tokens = self._prompt_tokens()
        with torch.inference_mode():
            return self.model.run_with_hooks(
                tokens, fwd_hooks=self._hooks_for(nodes_to_ablate)
            )

    def _resolve_foil(self, target: TargetId) -> TargetId | None:
        if self.foil_by_target and target in self.foil_by_target:
            return self.foil_by_target[target]
        return self.default_foil

    def _score_logits(self, logits: Any, target: TargetId) -> float:
        if self.score_kind == "kl_divergence":
            if self._ref_logits is None:
                self._ref_logits = self._run_logits(set())
            return compute_kl_score(self._ref_logits, logits)
        foil_idx: int | None = None
        if self.score_kind == "logit_gap":
            foil = self._resolve_foil(target)
            if foil is None:
                raise ValueError(
                    "score_kind='logit_gap' requires foil_by_target[target] or default_foil."
                )
            foil_idx = self.target_to_logit_idx[foil]
        return compute_scalar_score(
            logits=logits,
            target_index=self.target_to_logit_idx.get(target, 0),
            score_kind=self.score_kind,
            foil_index=foil_idx,
        )

    def score_all(self, target: TargetId) -> float:
        logits = self._run_logits(set())
        if self.score_kind == "kl_divergence":
            self._ref_logits = logits
            return 0.0
        return self._score_logits(logits, target)

    def score_empty(self, target: TargetId) -> float:
        return self._score_logits(self._run_logits(set(self._all_nodes)), target)

    def score_keep_only(self, nodes: set[NodeId], target: TargetId) -> float:
        to_ablate = self._all_nodes - set(nodes)
        return self._score_logits(self._run_logits(to_ablate), target)

    def score_remove(self, nodes: set[NodeId], target: TargetId) -> float:
        to_ablate = set(nodes) & self._all_nodes
        dropped = len(nodes) - len(to_ablate)
        if dropped:
            LOGGER.debug(
                "score_remove ignoring %d node(s) outside the component universe", dropped
            )
        return self._score_logits(self._run_logits(to_ablate), target)


def load_gold_component_nodes(graph_payload: Mapping[str, Any]) -> set[str]:
    """Gold component set from an InterpBench-style circuit JSON.

    Prefers node-level ``in_graph`` flags; falls back to nodes incident to any
    ``in_graph`` edge. ``input``/``logits`` sentinels are excluded (they are not
    ablatable components).
    """
    nodes = graph_payload.get("nodes") or {}
    gold = {name for name, spec in nodes.items() if spec.get("in_graph")}
    if not gold:
        for edge_name, spec in (graph_payload.get("edges") or {}).items():
            if not spec.get("in_graph"):
                continue
            source, _, rest = edge_name.partition("->")
            destination = rest.split("<")[0]
            gold.update({source, destination})
    return {name for name in gold if name not in ("input", "logits")}
