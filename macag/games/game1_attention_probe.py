"""Dual frozen/unfrozen Game 1 run with an attention-mediation diagnostic.

This is orchestration, not a new search: it runs :func:`solve_game1` twice on
the same graph — once against a frozen-attention oracle, once against an
unfrozen one — under fully matched parameters, then pairs the two results into
the per-prompt attention-mediation diagnostic (macag.md §10.4). Matched means
the solver hyperparameters are identical; each leg's prefilter still ranks
singletons under its own oracle, so the retained pools may differ — that is the
intended protocol (same k, mode-specific gains).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

from macag.games.game1_min_faithful import CandidatePrefilter, EvidenceSetResult, solve_game1
from macag.graph import CircuitGraph, NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.utils.attention_mediation import (
    AttentionMediationDiagnostic,
    compute_attention_mediation_diagnostic,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DualGame1Result:
    """Paired Game 1 results plus the attention-mediation diagnostic."""

    frozen: EvidenceSetResult
    unfrozen: EvidenceSetResult
    diagnostic: AttentionMediationDiagnostic
    params: dict[str, Any]


def _check_freeze_orientation(frozen_oracle: ScoringOracle, unfrozen_oracle: ScoringOracle) -> None:
    """Guard against swapped arguments when both backends expose freeze_attention.

    Backends without the attribute (toy/callback scorers used in tests) pass
    silently — the dual solver itself is freeze-agnostic.
    """
    frozen_flag = getattr(frozen_oracle.backend, "freeze_attention", None)
    unfrozen_flag = getattr(unfrozen_oracle.backend, "freeze_attention", None)
    if frozen_flag is None or unfrozen_flag is None:
        return
    if frozen_flag is not True or unfrozen_flag is not False:
        raise ValueError(
            "solve_game1_dual oracle orientation mismatch: expected "
            "frozen_oracle.backend.freeze_attention=True and "
            f"unfrozen_oracle.backend.freeze_attention=False, got {frozen_flag!r} "
            f"and {unfrozen_flag!r}. The arguments are likely swapped."
        )


def solve_game1_dual(
    graph: CircuitGraph,
    frozen_oracle: ScoringOracle,
    unfrozen_oracle: ScoringOracle,
    target: TargetId,
    candidates: Sequence[NodeId] | None = None,
    alpha: float = 0.5,
    lam: float = 0.01,
    budget: int | None = None,
    faithfulness_eps: float | None = None,
    prefilter_top_k: int | None = None,
    prefilter_fn: CandidatePrefilter | None = None,
    connected: bool = False,
    min_gain: float = 0.0,
    progress: bool = True,
    log_every: int = 50,
) -> DualGame1Result:
    """Run matched frozen + unfrozen Game 1 legs and diagnose attention mediation.

    There is deliberately no ``stop_metric`` parameter: both legs use
    ``"raw_relative"``. The normalized stop divides by ``recoverable_range``,
    which is documented-degenerate on the unfrozen leg (macag.md §2.3), so
    mixing stop rules would make the evidence-size and upstream-recruitment
    comparison incomparable — the very thing the dual run exists to measure.

    Each leg runs against its own oracle (separate caches: every intervention
    score depends on the freeze convention) and reports independent oracle
    stats. Pass two oracles over the same underlying model — see
    :func:`macag.scoring.derive_oracle_with_freeze`.
    """
    _check_freeze_orientation(frozen_oracle, unfrozen_oracle)

    shared_kwargs: dict[str, Any] = dict(
        graph=graph,
        target=target,
        candidates=candidates,
        alpha=alpha,
        lam=lam,
        budget=budget,
        faithfulness_eps=faithfulness_eps,
        stop_metric="raw_relative",
        prefilter_top_k=prefilter_top_k,
        prefilter_fn=prefilter_fn,
        connected=connected,
        min_gain=min_gain,
        progress=progress,
        log_every=log_every,
    )

    if progress:
        LOGGER.info("Game1 dual: frozen leg starting")
    frozen_result = solve_game1(oracle=frozen_oracle, **shared_kwargs)
    if progress:
        LOGGER.info("Game1 dual: unfrozen leg starting")
    unfrozen_result = solve_game1(oracle=unfrozen_oracle, **shared_kwargs)

    diagnostic = compute_attention_mediation_diagnostic(
        graph=graph,
        frozen_metrics=frozen_result.metrics,
        unfrozen_metrics=unfrozen_result.metrics,
        frozen_evidence=frozen_result.evidence,
        unfrozen_evidence=unfrozen_result.evidence,
    )
    if progress:
        LOGGER.info(
            "Game1 dual finished: verdict=%s range_frozen=%.6f range_unfrozen=%.6f |E|=%d->%d",
            diagnostic.verdict,
            diagnostic.range_frozen,
            diagnostic.range_unfrozen,
            diagnostic.evidence_size_frozen,
            diagnostic.evidence_size_unfrozen,
        )

    params = {
        "alpha": alpha,
        "lambda": lam,
        "budget": budget,
        "faithfulness_eps": faithfulness_eps,
        "stop_metric": "raw_relative",
        "prefilter_top_k": prefilter_top_k,
        "connected": connected,
        "min_gain": min_gain,
        "matched": True,
    }
    return DualGame1Result(
        frozen=frozen_result,
        unfrozen=unfrozen_result,
        diagnostic=diagnostic,
        params=params,
    )
