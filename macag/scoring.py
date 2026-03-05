"""Scoring oracles and intervention adapters for MACAG."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol

from macag.graph import NodeId

LOGGER = logging.getLogger(__name__)

TargetId = Any
ScoreKind = Literal["logit", "prob", "logit_gap", "negative_loss"]
Intervention = tuple[int, Any, int, Any]


class InterventionScorer(Protocol):
    """Protocol for intervention scoring backends."""

    def score_all(self, target: TargetId) -> float:
        ...

    def score_empty(self, target: TargetId) -> float:
        ...

    def score_keep_only(self, nodes: set[NodeId], target: TargetId) -> float:
        ...

    def score_remove(self, nodes: set[NodeId], target: TargetId) -> float:
        ...


@dataclass
class CallbackInterventionScorer:
    """Callable-based intervention scorer."""

    score_all_fn: Callable[[TargetId], float]
    score_empty_fn: Callable[[TargetId], float]
    score_keep_only_fn: Callable[[set[NodeId], TargetId], float]
    score_remove_fn: Callable[[set[NodeId], TargetId], float]

    def score_all(self, target: TargetId) -> float:
        return float(self.score_all_fn(target))

    def score_empty(self, target: TargetId) -> float:
        return float(self.score_empty_fn(target))

    def score_keep_only(self, nodes: set[NodeId], target: TargetId) -> float:
        return float(self.score_keep_only_fn(nodes, target))

    def score_remove(self, nodes: set[NodeId], target: TargetId) -> float:
        return float(self.score_remove_fn(nodes, target))


class ScoringOracle:
    """Memoized scoring oracle keyed by (mode, target, frozenset(nodes))."""

    def __init__(self, backend: InterventionScorer, cache_enabled: bool = True) -> None:
        self.backend = backend
        self.cache_enabled = cache_enabled
        self._cache: dict[tuple[str, type, str, frozenset[NodeId]], float] = {}
        self._oracle_calls = 0
        self._cache_hits = 0

    def _cache_key(self, mode: str, target: TargetId, nodes: set[NodeId]) -> tuple[str, type, str, frozenset[NodeId]]:
        target_key = str(target)
        return (mode, type(target), target_key, frozenset(nodes))

    def _score(self, mode: str, target: TargetId, nodes: set[NodeId] | None = None) -> float:
        nodes = nodes or set()
        key = self._cache_key(mode, target, nodes)
        if self.cache_enabled and key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        if mode == "all":
            score = float(self.backend.score_all(target))
        elif mode == "empty":
            score = float(self.backend.score_empty(target))
        elif mode == "keep_only":
            score = float(self.backend.score_keep_only(nodes, target))
        elif mode == "remove":
            score = float(self.backend.score_remove(nodes, target))
        else:
            raise ValueError(f"Unknown scoring mode: {mode}")

        self._oracle_calls += 1
        if self.cache_enabled:
            self._cache[key] = score
        return score

    def all(self, target: TargetId) -> float:
        return self._score("all", target)

    def empty(self, target: TargetId) -> float:
        return self._score("empty", target)

    def keep_only(self, nodes: set[NodeId], target: TargetId) -> float:
        return self._score("keep_only", target, nodes)

    def remove(self, nodes: set[NodeId], target: TargetId) -> float:
        return self._score("remove", target, nodes)

    def keep_all_except(self, nodes: set[NodeId], target: TargetId) -> float:
        return self.remove(nodes, target)

    def cache_stats(self) -> dict[str, int]:
        return {
            "oracle_calls": self._oracle_calls,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    def reset_stats(self) -> None:
        self._oracle_calls = 0
        self._cache_hits = 0


def _last_token_logits(logits: Any) -> Any:
    if logits.ndim == 3:
        return logits[0, -1]
    if logits.ndim == 2:
        return logits[-1]
    if logits.ndim == 1:
        return logits
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def compute_scalar_score(
    logits: Any,
    target_index: int,
    score_kind: ScoreKind = "logit_gap",
    foil_index: int | None = None,
) -> float:
    import torch

    token_logits = _last_token_logits(logits)
    if score_kind == "logit":
        return float(token_logits[target_index].item())
    if score_kind == "prob":
        probs = torch.softmax(token_logits, dim=-1)
        return float(probs[target_index].item())
    if score_kind == "negative_loss":
        log_probs = torch.log_softmax(token_logits, dim=-1)
        return float(log_probs[target_index].item())
    if score_kind == "logit_gap":
        if foil_index is None:
            raise ValueError("score_kind='logit_gap' requires a foil index.")
        return float((token_logits[target_index] - token_logits[foil_index]).item())
    raise ValueError(f"Unknown score kind: {score_kind}")


@dataclass
class ReplacementModelInterventionScorer:
    """Adapter that scores subsets by ablation interventions on circuit nodes."""

    model: Any
    prompt: str | Any
    node_to_intervention: Mapping[NodeId, tuple[int, Any, int] | Intervention]
    target_to_logit_idx: Mapping[TargetId, int]
    score_kind: ScoreKind = "logit_gap"
    foil_by_target: Mapping[TargetId, TargetId] | None = None
    default_foil: TargetId | None = None
    ablation_value: float = 0.0
    constrained_layers: range | None = None
    freeze_attention: bool = True
    node_universe: set[NodeId] | None = None
    _all_nodes: set[NodeId] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        supported = set(self.node_to_intervention.keys())
        if self.node_universe is None:
            self._all_nodes = supported
        else:
            self._all_nodes = set(self.node_universe) & supported
            if not self._all_nodes:
                LOGGER.warning(
                    "node_universe has no overlap with supported intervention nodes; "
                    "scoring will behave as if all nodes are ablated."
                )

    def supported_nodes(self) -> set[NodeId]:
        return set(self.node_to_intervention.keys())

    def intervention_universe(self) -> set[NodeId]:
        return set(self._all_nodes)

    def restrict_universe(self, nodes: set[NodeId] | list[NodeId] | tuple[NodeId, ...]) -> None:
        supported = set(self.node_to_intervention.keys())
        self._all_nodes = set(nodes) & supported

    def _normalize_intervention(
        self,
        spec: tuple[int, Any, int] | Intervention,
    ) -> Intervention:
        if len(spec) == 4:
            layer, pos, feature_idx, _ = spec
        elif len(spec) == 3:
            layer, pos, feature_idx = spec  # type: ignore[misc]
        else:
            raise ValueError(
                "Each intervention spec must be (layer, pos, feature_idx) or "
                "(layer, pos, feature_idx, value)."
            )
        return (layer, pos, feature_idx, self.ablation_value)

    def _ablation_interventions(self, nodes_to_ablate: set[NodeId]) -> list[Intervention]:
        interventions: list[Intervention] = []
        for node in nodes_to_ablate:
            spec = self.node_to_intervention[node]
            interventions.append(self._normalize_intervention(spec))
        return interventions

    def _target_indices(self, target: TargetId) -> tuple[int, int | None]:
        target_idx = self.target_to_logit_idx[target]
        if self.score_kind != "logit_gap":
            return target_idx, None

        foil_target = None
        if self.foil_by_target and target in self.foil_by_target:
            foil_target = self.foil_by_target[target]
        elif self.default_foil is not None:
            foil_target = self.default_foil

        if foil_target is None:
            raise ValueError(
                "score_kind='logit_gap' requires foil_by_target[target] or default_foil."
            )
        return target_idx, self.target_to_logit_idx[foil_target]

    def _run_logits(self, interventions: list[Intervention]) -> Any:
        logits, _ = self.model.feature_intervention(
            self.prompt,
            interventions,
            constrained_layers=self.constrained_layers,
            freeze_attention=self.freeze_attention,
            return_activations=False,
        )
        return logits

    def _score_logits(self, logits: Any, target: TargetId) -> float:
        target_idx, foil_idx = self._target_indices(target)
        return compute_scalar_score(
            logits=logits,
            target_index=target_idx,
            score_kind=self.score_kind,
            foil_index=foil_idx,
        )

    def score_all(self, target: TargetId) -> float:
        logits = self._run_logits(interventions=[])
        return self._score_logits(logits, target)

    def score_empty(self, target: TargetId) -> float:
        interventions = self._ablation_interventions(self._all_nodes)
        logits = self._run_logits(interventions=interventions)
        return self._score_logits(logits, target)

    def score_keep_only(self, nodes: set[NodeId], target: TargetId) -> float:
        to_ablate = self._all_nodes - set(nodes)
        interventions = self._ablation_interventions(to_ablate)
        logits = self._run_logits(interventions=interventions)
        return self._score_logits(logits, target)

    def score_remove(self, nodes: set[NodeId], target: TargetId) -> float:
        interventions = self._ablation_interventions(set(nodes))
        logits = self._run_logits(interventions=interventions)
        return self._score_logits(logits, target)


@dataclass
class ToyAdditiveInterventionScorer:
    """Simple additive scorer for fast validation and tests."""

    weights_by_target: Mapping[TargetId, Mapping[NodeId, float]]
    base_by_target: Mapping[TargetId, float] | None = None

    def _weights(self, target: TargetId) -> Mapping[NodeId, float]:
        if target not in self.weights_by_target:
            raise KeyError(f"Unknown target: {target}")
        return self.weights_by_target[target]

    def _base(self, target: TargetId) -> float:
        if not self.base_by_target:
            return 0.0
        return float(self.base_by_target.get(target, 0.0))

    def score_all(self, target: TargetId) -> float:
        weights = self._weights(target)
        return self._base(target) + float(sum(weights.values()))

    def score_empty(self, target: TargetId) -> float:
        return self._base(target)

    def score_keep_only(self, nodes: set[NodeId], target: TargetId) -> float:
        weights = self._weights(target)
        subtotal = sum(weights.get(node, 0.0) for node in nodes)
        return self._base(target) + float(subtotal)

    def score_remove(self, nodes: set[NodeId], target: TargetId) -> float:
        weights = self._weights(target)
        removed = sum(weights.get(node, 0.0) for node in nodes)
        return self.score_all(target) - float(removed)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ToyAdditiveInterventionScorer":
        resolved = Path(path)
        try:
            text = resolved.read_text()
        except FileNotFoundError:
            raise FileNotFoundError(f"Toy oracle JSON file not found: {resolved}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in toy oracle file {resolved}: {exc}") from exc
        return cls(
            weights_by_target=payload["weights_by_target"],
            base_by_target=payload.get("base_by_target"),
        )
