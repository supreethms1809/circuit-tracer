"""Scoring oracles and intervention adapters for MACAG."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol

from macag.graph import NodeId

LOGGER = logging.getLogger(__name__)

TargetId = Any
ScoreKind = Literal[
    "logit", "prob", "logit_gap", "negative_loss", "kl_divergence", "answer_span"
]
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

    def universe_fingerprint(self) -> Any:
        """Identity of the current intervention universe.

        Returned value must change whenever ``score_empty`` / ``score_keep_only``
        results would change due to the universe being mutated (e.g. via
        ``restrict_universe``). Backends without a mutable universe may return
        ``None``. Used by :class:`ScoringOracle` to avoid serving stale cached
        universe-dependent scores (I1).
        """
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

    # Modes whose result depends on the backend's intervention universe, so the
    # cache key must include the universe fingerprint to avoid stale hits after
    # restrict_universe (I1). "empty"/"keep_only" ablate the universe or its
    # complement; "remove" intersects the requested nodes with the universe.
    _UNIVERSE_DEPENDENT_MODES = frozenset({"empty", "keep_only", "remove"})

    def __init__(self, backend: InterventionScorer, cache_enabled: bool = True) -> None:
        self.backend = backend
        self.cache_enabled = cache_enabled
        self._cache: dict[tuple[str, type, str, frozenset[NodeId], Any], float] = {}
        self._oracle_calls = 0
        self._cache_hits = 0

    def _universe_fingerprint(self, mode: str) -> Any:
        if mode not in self._UNIVERSE_DEPENDENT_MODES:
            return None
        fingerprint_fn = getattr(self.backend, "universe_fingerprint", None)
        if fingerprint_fn is None:
            return None
        return fingerprint_fn()

    def _cache_key(
        self, mode: str, target: TargetId, nodes: set[NodeId]
    ) -> tuple[str, type, str, frozenset[NodeId], Any]:
        target_key = str(target)
        return (mode, type(target), target_key, frozenset(nodes), self._universe_fingerprint(mode))

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


def derive_oracle_with_freeze(oracle: ScoringOracle, freeze_attention: bool) -> ScoringOracle:
    """Return a new ScoringOracle whose backend has ``freeze_attention`` as requested.

    The derived oracle gets a fresh cache and fresh stats so frozen/unfrozen legs of a
    dual Game 1 run report independent oracle counts. The underlying model reference is
    shared shallowly — the model is never reloaded. Raises ``TypeError`` for backends
    without a freeze concept (toy oracles, callback scorers).
    """
    backend = oracle.backend
    if not (dataclasses.is_dataclass(backend) and hasattr(backend, "freeze_attention")):
        raise TypeError(
            f"Oracle backend {type(backend).__name__} has no freeze_attention field; "
            "frozen/unfrozen derivation requires a ReplacementModel-backed scorer "
            "(toy oracles have no attention to freeze)."
        )
    if backend.freeze_attention == freeze_attention:
        return ScoringOracle(backend=backend, cache_enabled=oracle.cache_enabled)

    derived = dataclasses.replace(backend, freeze_attention=freeze_attention)
    # dataclasses.replace re-runs __post_init__, which rebuilds _all_nodes from the
    # node_universe FIELD — discarding any post-construction restrict_universe().
    # Re-apply the source's live universe so both legs ablate the same node set.
    if hasattr(derived, "restrict_universe") and hasattr(backend, "intervention_universe"):
        derived.restrict_universe(backend.intervention_universe())
    return ScoringOracle(backend=derived, cache_enabled=oracle.cache_enabled)


def _last_token_logits(logits: Any) -> Any:
    if logits.ndim == 3:
        return logits[0, -1]
    if logits.ndim == 2:
        return logits[-1]
    if logits.ndim == 1:
        return logits
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def compute_kl_score(ref_logits: Any, intervened_logits: Any) -> float:
    """Return ``-KL(P_ref || P_int)`` at the last token (higher = more faithful).

    ``P_ref`` is the reference (no-intervention) distribution; ``P_int`` is the
    distribution after ablation. Matches the MIB/ACDC convention of comparing the
    intervened circuit to the full model at the scored position.
    """
    import torch
    import torch.nn.functional as F

    ref = _last_token_logits(ref_logits)
    intr = _last_token_logits(intervened_logits)
    ref_probs = torch.softmax(ref, dim=-1)
    int_log_probs = torch.log_softmax(intr, dim=-1)
    kl = F.kl_div(
        int_log_probs.unsqueeze(0),
        ref_probs.unsqueeze(0),
        reduction="batchmean",
    )
    return float(-kl.item())


def compute_span_logprob(logits: Any, span_ids: Any, prompt_len: int) -> float:
    """Summed teacher-forced log-prob of ``span_ids`` continuing a prompt.

    ``logits`` must cover the full ``prompt + span`` sequence; span token ``i``
    is predicted at sequence position ``prompt_len - 1 + i``. For a length-1
    span this is the last-prompt-position log-prob of the answer token, so the
    ``answer_span`` gap reduces exactly to ``logit_gap`` (the shared log-softmax
    normalizer cancels between target and foil scored at the same position).
    """
    import torch

    if logits.ndim == 3:
        seq_logits = logits[0]
    elif logits.ndim == 2:
        seq_logits = logits
    else:
        raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")
    total = 0.0
    for i, token_id in enumerate(span_ids):
        log_probs = torch.log_softmax(seq_logits[prompt_len - 1 + i], dim=-1)
        total += float(log_probs[int(token_id)].item())
    return total


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
    if score_kind == "kl_divergence":
        raise ValueError(
            "score_kind='kl_divergence' requires full logits via compute_kl_score(); "
            "it is not a single-token scalar."
        )
    if score_kind == "answer_span":
        raise ValueError(
            "score_kind='answer_span' requires per-continuation forwards via "
            "compute_span_logprob(); it is not a single-token scalar."
        )
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
    # answer_span only: full token-id list per target label (multi-token answers).
    target_span_ids_by_label: Mapping[TargetId, list[int]] | None = None
    _all_nodes: set[NodeId] = field(init=False, repr=False)
    _ref_logits: Any = field(init=False, default=None, repr=False)
    _prompt_token_ids: Any = field(init=False, default=None, repr=False)
    # Real model forwards (answer_span runs 2 per logical oracle score; the
    # oracle's oracle_calls counter stays 1-per-coalition so cost comparisons
    # across score kinds remain per-score).
    forward_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        supported = set(self.node_to_intervention.keys())
        if self.node_universe is None:
            self._all_nodes = supported
        else:
            self._all_nodes = set(self.node_universe) & supported
            if not self._all_nodes:
                # An empty universe means score_empty ablates NOTHING, so
                # empty == all, recoverable_range == 0, and every normalized
                # metric silently degenerates to 0. Fail loudly instead.
                raise ValueError(
                    "node_universe has no overlap with supported intervention nodes; "
                    "with an empty ablation universe 'empty' equals 'all' and all "
                    "faithfulness metrics degenerate. Check that candidate node IDs "
                    "match the graph's node_id convention."
                )
        if self.score_kind == "answer_span" and not self.target_span_ids_by_label:
            raise ValueError(
                "score_kind='answer_span' requires target_span_ids_by_label "
                "(the full token-id span per target label)."
            )

    def supported_nodes(self) -> set[NodeId]:
        return set(self.node_to_intervention.keys())

    def intervention_universe(self) -> set[NodeId]:
        return set(self._all_nodes)

    def universe_fingerprint(self) -> tuple[frozenset[NodeId], bool]:
        """Fingerprint the current ablation universe so ScoringOracle invalidates
        empty/keep_only cache entries when ``restrict_universe`` changes it (I1).

        Includes ``freeze_attention``: flipping it changes every intervention-backed
        score (empty/keep_only/remove), which is exactly the universe-dependent mode
        set, so the fingerprint is the natural invalidation point. ``score_all`` is
        freeze-invariant (no interventions, so frozen-at-clean patterns equal the
        recomputed ones) and correctly keeps its cache entries.
        """
        return (frozenset(self._all_nodes), self.freeze_attention)

    def restrict_universe(self, nodes: set[NodeId] | list[NodeId] | tuple[NodeId, ...]) -> None:
        supported = set(self.node_to_intervention.keys())
        restricted = set(nodes) & supported
        if not restricted and supported:
            raise ValueError(
                "restrict_universe produced an empty ablation universe: none of the "
                f"{len(set(nodes))} requested nodes are supported intervention nodes. "
                "With an empty universe 'empty' equals 'all' and all faithfulness "
                "metrics degenerate. Check node ID conventions."
            )
        self._all_nodes = restricted

    def _normalize_intervention(
        self,
        spec: tuple[int, Any, int] | Intervention,
    ) -> Intervention:
        # Honor an explicit per-node value when the spec provides one (4-tuple);
        # only fall back to the global ablation_value for bare (layer, pos, feat)
        # specs (I3). Graph-loaded specs are 3-tuples, so the default ablation
        # path is unchanged.
        if len(spec) == 4:
            layer, pos, feature_idx, value = spec
        elif len(spec) == 3:
            layer, pos, feature_idx = spec  # type: ignore[misc]
            value = self.ablation_value
        else:
            raise ValueError(
                "Each intervention spec must be (layer, pos, feature_idx) or "
                "(layer, pos, feature_idx, value)."
            )
        return (layer, pos, feature_idx, value)

    def _ablation_interventions(self, nodes_to_ablate: set[NodeId]) -> list[Intervention]:
        interventions: list[Intervention] = []
        for node in nodes_to_ablate:
            spec = self.node_to_intervention[node]
            interventions.append(self._normalize_intervention(spec))
        return interventions

    def _resolve_foil(self, target: TargetId) -> TargetId | None:
        if self.foil_by_target and target in self.foil_by_target:
            return self.foil_by_target[target]
        return self.default_foil

    def _target_indices(self, target: TargetId) -> tuple[int, int | None]:
        if self.score_kind in ("kl_divergence", "answer_span"):
            return 0, None
        target_idx = self.target_to_logit_idx[target]
        if self.score_kind != "logit_gap":
            return target_idx, None

        foil_target = self._resolve_foil(target)
        if foil_target is None:
            raise ValueError(
                "score_kind='logit_gap' requires foil_by_target[target] or default_foil."
            )
        return target_idx, self.target_to_logit_idx[foil_target]

    def _run_logits(self, interventions: list[Intervention], inputs: Any = None) -> Any:
        self.forward_count += 1
        logits, _ = self.model.feature_intervention(
            self.prompt if inputs is None else inputs,
            interventions,
            constrained_layers=self.constrained_layers,
            freeze_attention=self.freeze_attention,
            return_activations=False,
        )
        return logits

    def _prompt_ids(self) -> Any:
        """Prompt token ids on the model's own tokenization path (BOS-prepended).

        Uses ``model.ensure_tokenized`` when available so span positions line up
        with the graph's ``ctx_idx`` convention; it is idempotent for tensors that
        already start with a special token, so ``cat(prompt_ids, span)`` passed
        back into ``feature_intervention`` keeps prompt-position interventions
        valid.
        """
        if self._prompt_token_ids is None:
            ensure = getattr(self.model, "ensure_tokenized", None)
            if ensure is not None:
                self._prompt_token_ids = ensure(self.prompt)
            else:
                import torch

                tokenizer = self.model.tokenizer
                self._prompt_token_ids = tokenizer(
                    self.prompt, return_tensors="pt", add_special_tokens=False
                ).input_ids.squeeze(0).to(torch.long)
        return self._prompt_token_ids

    def _span_ids(self, label: TargetId) -> list[int]:
        spans = self.target_span_ids_by_label or {}
        if label not in spans:
            raise ValueError(f"No answer span registered for target label {label!r}.")
        return list(spans[label])

    def _score_span(self, interventions: list[Intervention], target: TargetId) -> float:
        """answer_span score: teacher-forced summed log-prob gap, target − foil.

        Two forwards per logical score (prompt+target span, prompt+foil span);
        interventions apply at absolute prompt positions, which appending answer
        tokens does not disturb.
        """
        import torch

        foil = self._resolve_foil(target)
        if foil is None:
            raise ValueError(
                "score_kind='answer_span' requires foil_by_target[target] or default_foil."
            )
        prompt_ids = self._prompt_ids()
        prompt_len = int(prompt_ids.shape[0])
        gap = 0.0
        for sign, label in ((1.0, target), (-1.0, foil)):
            span = self._span_ids(label)
            span_tensor = torch.tensor(
                span, dtype=prompt_ids.dtype, device=prompt_ids.device
            )
            inputs = torch.cat([prompt_ids, span_tensor])
            logits = self._run_logits(interventions, inputs=inputs)
            gap += sign * compute_span_logprob(logits, span, prompt_len)
        return gap

    def _reference_logits(self) -> Any:
        if self._ref_logits is None:
            self._ref_logits = self._run_logits(interventions=[])
        return self._ref_logits

    def _score_logits(self, logits: Any, target: TargetId) -> float:
        if self.score_kind == "kl_divergence":
            return compute_kl_score(self._reference_logits(), logits)
        target_idx, foil_idx = self._target_indices(target)
        return compute_scalar_score(
            logits=logits,
            target_index=target_idx,
            score_kind=self.score_kind,
            foil_index=foil_idx,
        )

    def _score_with_interventions(
        self, interventions: list[Intervention], target: TargetId
    ) -> float:
        if self.score_kind == "answer_span":
            return self._score_span(interventions, target)
        logits = self._run_logits(interventions=interventions)
        return self._score_logits(logits, target)

    def score_all(self, target: TargetId) -> float:
        if self.score_kind == "kl_divergence":
            self._ref_logits = self._run_logits(interventions=[])
            return 0.0
        return self._score_with_interventions([], target)

    def score_empty(self, target: TargetId) -> float:
        interventions = self._ablation_interventions(self._all_nodes)
        return self._score_with_interventions(interventions, target)

    def score_keep_only(self, nodes: set[NodeId], target: TargetId) -> float:
        to_ablate = self._all_nodes - set(nodes)
        return self._score_with_interventions(self._ablation_interventions(to_ablate), target)

    def score_remove(self, nodes: set[NodeId], target: TargetId) -> float:
        # Intersect with the universe so remove() treats out-of-universe nodes as
        # immutable background, symmetric with score_keep_only (which never ablates
        # outside _all_nodes). Also avoids KeyError on unsupported node IDs.
        to_ablate = set(nodes) & self._all_nodes
        dropped = len(nodes) - len(to_ablate)
        if dropped:
            LOGGER.debug(
                "score_remove ignoring %d node(s) outside the intervention universe", dropped
            )
        return self._score_with_interventions(self._ablation_interventions(to_ablate), target)


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
