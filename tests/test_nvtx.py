"""NVTX helper is a no-op unless MACAG_NVTX is set."""

from __future__ import annotations

from macag.games.game1_min_faithful import solve_game1
from macag.graph import CircuitGraph
from macag.scoring import ScoringOracle, ToyAdditiveInterventionScorer
from macag.nvtx import nvtx_enabled, nvtx_range


def test_nvtx_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MACAG_NVTX", raising=False)
    assert nvtx_enabled() is False
    with nvtx_range("should-be-noop"):
        pass


def test_nvtx_enabled_flag(monkeypatch) -> None:
    monkeypatch.setenv("MACAG_NVTX", "1")
    assert nvtx_enabled() is True
    with nvtx_range("enabled-path-must-not-raise"):
        pass


def test_game1_nvtx_does_not_change_toy_result(monkeypatch) -> None:
    graph = CircuitGraph(nodes=["a", "b"], edges=[])
    backend = ToyAdditiveInterventionScorer(
        weights_by_target={"y": {"a": 1.0, "b": 0.2}},
        base_by_target={"y": 0.0},
    )
    oracle = ScoringOracle(backend)
    monkeypatch.setenv("MACAG_NVTX", "1")
    result = solve_game1(
        graph=graph,
        oracle=oracle,
        target="y",
        candidates=["a", "b"],
        alpha=1.0,
        lam=0.0,
        progress=False,
    )
    assert "a" in result.evidence
