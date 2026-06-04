"""Regression tests for MACAG bug fixes (C1-C3, L1-L3, I1-I4).

These use the dependency-free ToyAdditiveInterventionScorer / small custom
backends so they run without a model or GPU.
"""

from __future__ import annotations

from typing import Any

import pytest

from macag.graph import CircuitGraph, grow_connected_frontier
from macag.scoring import (
    ReplacementModelInterventionScorer,
    ScoringOracle,
    ToyAdditiveInterventionScorer,
)
from macag.games.game1_min_faithful import prefilter_candidates, solve_game1
from macag.games.game2_contrastive import solve_game2
from macag.utils.metrics import compute_faithfulness_metrics, metrics_to_dict
from macag.factories.replacement_model import (
    load_feature_node_interventions_from_graph_json,
)


def _oracle(weights_by_target: dict[Any, dict[str, float]], base: dict[Any, float] | None = None):
    backend = ToyAdditiveInterventionScorer(
        weights_by_target=weights_by_target,
        base_by_target=base,
    )
    return ScoringOracle(backend=backend, cache_enabled=True)


def _full_graph(nodes: list[str], edges: list[tuple[str, str]] | None = None) -> CircuitGraph:
    return CircuitGraph(nodes=nodes, edges=edges or [])


# --------------------------------------------------------------------------- C1
def test_c1_normalized_metrics_and_error_floor() -> None:
    # base acts as an error floor; recoverable range is sum(weights).
    oracle = _oracle({"y": {"a": 6.0, "b": 4.0}}, base={"y": 100.0})
    metrics = compute_faithfulness_metrics(oracle, target="y", nodes={"a"}, alpha=0.5)

    assert metrics.empty_score == pytest.approx(100.0)  # empty score carries the floor
    assert metrics.recoverable_range == pytest.approx(10.0)  # all - empty
    # sufficiency_normalized = sufficiency / range = 6 / 10 (floor-independent).
    assert metrics.sufficiency_normalized == pytest.approx(0.6)
    assert metrics.necessity_normalized == pytest.approx(0.6)
    assert metrics.faithfulness_delta_normalized == pytest.approx(0.6)

    # Report dict surfaces the error floor and normalized view (C1).
    as_dict = metrics_to_dict(metrics)
    assert as_dict["error_floor"] == pytest.approx(100.0)
    assert as_dict["recoverable_range"] == pytest.approx(10.0)
    assert as_dict["faithfulness_normalized"] == pytest.approx(0.6)


def test_c1_zero_range_guard() -> None:
    oracle = _oracle({"y": {"a": 0.0, "b": 0.0}}, base={"y": 5.0})
    metrics = compute_faithfulness_metrics(oracle, target="y", nodes={"a"}, alpha=0.5)
    assert metrics.recoverable_range == pytest.approx(0.0)
    assert metrics.sufficiency_normalized == 0.0
    assert metrics.necessity_normalized == 0.0
    assert metrics.faithfulness_delta_normalized == 0.0


def test_c1_include_error_nodes_raises_on_error_graph(tmp_path) -> None:
    import json

    graph = {
        "nodes": [
            {"node_id": "0_12_1", "feature_type": "cross layer transcoder", "layer": "0", "ctx_idx": 1},
            {"node_id": "0_0_2", "feature_type": "mlp reconstruction error", "layer": "0", "ctx_idx": 2},
        ]
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(graph))

    # default path ignores error nodes and succeeds
    interventions = load_feature_node_interventions_from_graph_json(str(path))
    assert set(interventions) == {"0_12_1"}

    # opt-in raises loudly rather than mapping the error node to a bogus feature idx
    with pytest.raises(NotImplementedError):
        load_feature_node_interventions_from_graph_json(str(path), include_error_nodes=True)


# --------------------------------------------------------------------------- C2
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_c2_eps_stop_uses_normalized_objective(alpha: float) -> None:
    # A=9, B=1; range=10. Selecting A reaches normalized 0.9 >= 1 - 0.2 = 0.8,
    # so the solver stops at {A}. The OLD condition (all - keep_only = 1 <= 0.2)
    # would NOT have stopped here, and at alpha=0 it tested the wrong metric.
    graph = _full_graph(["a", "b"])
    oracle = _oracle({"y": {"a": 9.0, "b": 1.0}})
    result = solve_game1(
        graph=graph,
        oracle=oracle,
        target="y",
        candidates=["a", "b"],
        alpha=alpha,
        lam=0.0,
        faithfulness_eps=0.2,
        progress=False,
    )
    assert result.evidence == {"a"}
    assert result.metrics.faithfulness_delta_normalized >= 0.8


# --------------------------------------------------------------------------- C3
def test_c3_best_iterate_returned_over_oscillating_abr() -> None:
    # Symmetric weights, budget=1, beta high → ABR oscillates {A},{A} <-> {B},{B}.
    # The {A},{A} allocation has the higher combined utility; best-iterate must
    # return it regardless of how many ABR iterations run (and their parity).
    graph = _full_graph(["A", "B"])
    oracle = _oracle({"y": {"A": 1.0, "B": 0.9}, "f": {"A": 1.0, "B": 0.9}})
    result = solve_game2(
        graph=graph,
        oracle=oracle,
        y="y",
        y_foil="f",
        candidates=["A", "B"],
        alpha=0.5,
        lam=0.0,
        beta=0.3,
        budget=1,
        abr_iters=4,
        progress=False,
    )
    assert result.evidence_y == {"A"}
    assert result.evidence_foil == {"A"}
    # combined utility of the best (A,A) iterate: (1.0 - 0.3) * 2
    assert result.utility_y + result.utility_foil == pytest.approx(1.4)


def test_c3_symmetric_allocation_for_symmetric_oracle() -> None:
    graph = _full_graph(["A", "B", "C", "D"])
    w = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4}
    oracle = _oracle({"y": dict(w), "f": dict(w)})
    result = solve_game2(
        graph=graph,
        oracle=oracle,
        y="y",
        y_foil="f",
        candidates=["A", "B", "C", "D"],
        alpha=0.5,
        lam=0.05,
        beta=0.2,
        abr_iters=8,
        progress=False,
    )
    # Jacobi symmetry: both agents best-respond to the same frozen opponent.
    assert len(result.evidence_y) == len(result.evidence_foil)
    assert result.utility_y == pytest.approx(result.utility_foil)


# ------------------------------------------------------------------------- L1/L2
def test_l1_connected_through_intermediate_node() -> None:
    # a - mid - b ; features a,b connect only through the (non-candidate) mid node.
    graph = _full_graph(["a", "mid", "b"], edges=[("a", "mid"), ("mid", "b")])
    assert graph.connected_through({"a", "b"}) is True
    assert graph.is_weakly_connected({"a", "b"}) is False


def test_l1_connected_greedy_grows_past_singleton() -> None:
    graph = _full_graph(["a", "mid", "b"], edges=[("a", "mid"), ("mid", "b")])
    oracle = _oracle({"y": {"a": 5.0, "b": 4.0}})
    result = solve_game1(
        graph=graph,
        oracle=oracle,
        target="y",
        candidates=["a", "b"],  # mid is an intermediate, not a candidate
        alpha=0.5,
        lam=0.0,
        connected=True,
        progress=False,
    )
    # With the topology-aware check both features are reachable through mid.
    assert result.evidence == {"a", "b"}


def test_l2_connected_frontier_keeps_connected_pool() -> None:
    # Two components: {a,mid,b} connected, {x} isolated. Top-ranked is the isolated
    # x, but a connected frontier of size 2 must stay within one component.
    graph = _full_graph(["a", "mid", "b", "x"], edges=[("a", "mid"), ("mid", "b")])
    frontier = grow_connected_frontier(graph, ranked_nodes=["x", "a", "b"], top_k=2)
    assert "x" in frontier  # seed is the top-ranked node
    assert len(frontier) == 2
    # the second slot is filled from x's (empty) component → falls back by rank to a
    assert frontier == ["x", "a"]


# --------------------------------------------------------------------------- L3
def test_l3_prefilter_is_ranked_when_topk_ge_len() -> None:
    graph = _full_graph(["a", "b", "c"])
    oracle = _oracle({"y": {"a": 1.0, "b": 5.0, "c": 3.0}})
    out = prefilter_candidates(
        graph, oracle, target="y", candidates=["a", "b", "c"], alpha=0.5, lam=0.0, top_k=3
    )
    # Ranked by singleton gain desc, not returned in raw input order.
    assert out == ["b", "c", "a"]


# --------------------------------------------------------------------------- I1
class _MutableUniverseBackend:
    """Minimal backend whose empty/keep_only scores depend on a mutable universe."""

    def __init__(self) -> None:
        self._universe = {"a", "b"}

    def restrict_universe(self, nodes: set[str]) -> None:
        self._universe = set(nodes)

    def universe_fingerprint(self):
        return frozenset(self._universe)

    def score_all(self, target):  # universe-independent
        return 10.0

    def score_empty(self, target):  # depends on universe size
        return float(len(self._universe))

    def score_keep_only(self, nodes, target):
        return float(len(self._universe) + len(nodes))

    def score_remove(self, nodes, target):  # universe-independent
        return 10.0 - float(len(nodes))


def test_i1_cache_invalidates_on_universe_change() -> None:
    backend = _MutableUniverseBackend()
    oracle = ScoringOracle(backend=backend, cache_enabled=True)

    assert oracle.empty("y") == 2.0  # |universe| = 2
    remove_first = oracle.remove({"a"}, "y")

    backend.restrict_universe({"a"})  # universe now size 1
    # empty is universe-dependent → must reflect the new universe, not the stale 2.0
    assert oracle.empty("y") == 1.0
    # remove is universe-independent → identical and may be served from cache
    assert oracle.remove({"a"}, "y") == remove_first


# --------------------------------------------------------------------------- I3
def test_i3_per_node_value_honored() -> None:
    scorer = ReplacementModelInterventionScorer(
        model=None,
        prompt="",
        node_to_intervention={"n3": (0, 0, 5), "n4": (1, 2, 7, 3.5)},
        target_to_logit_idx={},
        ablation_value=0.0,
    )
    # 3-tuple → falls back to ablation_value
    assert scorer._normalize_intervention((0, 0, 5)) == (0, 0, 5, 0.0)
    # 4-tuple → honors the explicit value
    assert scorer._normalize_intervention((1, 2, 7, 3.5)) == (1, 2, 7, 3.5)


# --------------------------------------------------------------------------- I4
def test_i4_sparsity_against_full_pool_not_prefiltered() -> None:
    nodes = [f"n{i}" for i in range(6)]
    graph = _full_graph(nodes)
    oracle = _oracle({"y": {n: float(6 - i) for i, n in enumerate(nodes)}})
    result = solve_game1(
        graph=graph,
        oracle=oracle,
        target="y",
        candidates=nodes,
        alpha=0.5,
        lam=0.0,
        budget=2,
        prefilter_top_k=2,
        progress=False,
    )
    assert result.total_candidates == 6
    assert result.candidate_count == 2  # prefiltered pool
    # sparsity is reported against the true 6-node pool, not the prefiltered 2.
    assert result.sparsity == pytest.approx(1.0 - len(result.evidence) / 6)
