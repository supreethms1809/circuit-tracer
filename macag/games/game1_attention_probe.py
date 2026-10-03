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
from pathlib import Path
from typing import Any, Sequence

from macag.games.game1_min_faithful import (
    CandidatePrefilter,
    EvidenceSetResult,
    GAME1_CKPT_SCHEMA,
    _atomic_write_json,
    _load_game1_checkpoint,
    result_from_selected_order,
    solve_game1,
)
from macag.graph import CircuitGraph, NodeId
from macag.scoring import ScoringOracle, TargetId
from macag.utils.attention_mediation import (
    AttentionMediationDiagnostic,
    compute_attention_mediation_diagnostic,
)
from macag.nvtx import nvtx_range

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
    checkpoint_path: str | Path | None = None,
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
        checkpoint_path=checkpoint_path,
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
    pool = list(candidates) if candidates is not None else list(graph.nodes())
    pool = [node for node in pool if graph.has_node(node)]
    total_candidates = len(pool)

    def persist_dual(*, leg: str, frozen_done: bool, frozen_order: Sequence[NodeId], selected_order: Sequence[NodeId]) -> None:
        if checkpoint_path is None:
            return
        _atomic_write_json(
            checkpoint_path,
            {
                "schema": GAME1_CKPT_SCHEMA,
                "freeze_mode": "both",
                "leg": leg,
                "frozen_done": frozen_done,
                "frozen_order": list(frozen_order),
                "selected_order": list(selected_order),
                "first_faith_gain": None,
                "params": {k: v for k, v in params.items() if k != "matched"},
            },
        )

    def reconstruct(oracle: ScoringOracle, order: Sequence[NodeId]) -> EvidenceSetResult:
        return result_from_selected_order(
            graph=graph,
            oracle=oracle,
            target=target,
            selected_order=order,
            alpha=alpha,
            lam=lam,
            total_candidates=total_candidates,
            candidate_count=total_candidates,
            params={k: v for k, v in params.items() if k != "matched"},
        )

    ckpt = _load_game1_checkpoint(checkpoint_path)
    frozen_done = bool(ckpt and ckpt.get("frozen_done"))
    unfrozen_resume: list[NodeId] = []

    if frozen_done:
        frozen_order = list(ckpt.get("frozen_order") or [])
        if not frozen_order and str(ckpt.get("leg") or "") != "unfrozen":
            frozen_order = list(ckpt.get("selected_order") or [])
        if progress:
            LOGGER.info("Game1 dual: skipping completed frozen leg |E|=%d", len(frozen_order))
        frozen_result = reconstruct(frozen_oracle, frozen_order)
        if str(ckpt.get("leg") or "") == "unfrozen":
            unfrozen_resume = list(ckpt.get("selected_order") or [])
    else:
        frozen_resume = list((ckpt or {}).get("selected_order") or (ckpt or {}).get("frozen_order") or [])
        if progress:
            LOGGER.info("Game1 dual: frozen leg starting")
        with nvtx_range("game1.frozen"):
            frozen_result = solve_game1(
                oracle=frozen_oracle,
                resume_selected_order=frozen_resume,
                checkpoint_meta={
                    "freeze_mode": "both",
                    "leg": "frozen",
                    "frozen_done": False,
                    "frozen_order": frozen_resume,
                },
                **shared_kwargs,
            )
        persist_dual(
            leg="unfrozen",
            frozen_done=True,
            frozen_order=frozen_result.selected_order,
            selected_order=[],
        )

    if progress:
        LOGGER.info("Game1 dual: unfrozen leg starting")
    with nvtx_range("game1.unfrozen"):
        unfrozen_result = solve_game1(
            oracle=unfrozen_oracle,
            resume_selected_order=unfrozen_resume,
            checkpoint_meta={
                "freeze_mode": "both",
                "leg": "unfrozen",
                "frozen_done": True,
                "frozen_order": frozen_result.selected_order,
            },
            **shared_kwargs,
        )

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

    return DualGame1Result(
        frozen=frozen_result,
        unfrozen=unfrozen_result,
        diagnostic=diagnostic,
        params=params,
    )
