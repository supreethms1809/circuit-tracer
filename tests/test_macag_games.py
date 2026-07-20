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


def test_raw_relative_stop_is_denominator_free() -> None:
    # Diminishing per-feature gains a=10, b=9, c=0.5. With raw_relative the first
    # gain is 10; c's marginal gain 0.5 < eps*first_gain = 0.1*10 = 1.0, so the
    # greedy stops BEFORE adding c -> {a, b}. This uses no recoverable_range
    # denominator, so it is stable when that range collapses (unfrozen attention).
    graph = _full_graph(["a", "b", "c"])
    oracle = _oracle({"y": {"a": 10.0, "b": 9.0, "c": 0.5}})
    result = solve_game1(
        graph=graph,
        oracle=oracle,
        target="y",
        candidates=["a", "b", "c"],
        lam=0.0,
        faithfulness_eps=0.1,
        stop_metric="raw_relative",
        progress=False,
    )
    assert result.evidence == {"a", "b"}
    assert "c" not in result.evidence
    assert result.iterations == 2


def test_raw_relative_stop_ignores_lam_penalty() -> None:
    # The stop is defined on RAW faithfulness gains. With lam=0.9, c's utility
    # gain is 1.5 - 0.9 = 0.6 (a utility-gain stop against eps*(10 - 0.9) = 0.91
    # would WRONGLY stop before c), but its faithfulness gain 1.5 >= 0.1*10 = 1.0,
    # so the greedy must keep going and add c.
    graph = _full_graph(["a", "b", "c"])
    oracle = _oracle({"y": {"a": 10.0, "b": 9.0, "c": 1.5}})
    result = solve_game1(
        graph=graph,
        oracle=oracle,
        target="y",
        candidates=["a", "b", "c"],
        lam=0.9,
        faithfulness_eps=0.1,
        stop_metric="raw_relative",
        progress=False,
    )
    assert result.evidence == {"a", "b", "c"}


def test_invalid_stop_metric_raises() -> None:
    graph = _full_graph(["a"])
    oracle = _oracle({"y": {"a": 1.0}})
    with pytest.raises(ValueError):
        solve_game1(graph=graph, oracle=oracle, target="y", candidates=["a"],
                    stop_metric="bogus", progress=False)


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


# ------------------------------------------------------------------ Fictitious play
def test_fp_invalid_solver_raises() -> None:
    graph = _full_graph(["A"])
    oracle = _oracle({"y": {"A": 1.0}, "f": {"A": 1.0}})
    with pytest.raises(ValueError):
        solve_game2(graph=graph, oracle=oracle, y="y", y_foil="f",
                    candidates=["A"], solver="bogus", progress=False)


def test_fp_round1_matches_abr_round1() -> None:
    # With a single round both solvers best-respond to an empty opponent history,
    # so the first-round responses must be identical.
    graph = _full_graph(["A", "B", "C"])
    weights = {"y": {"A": 1.0, "B": 0.7, "C": 0.2}, "f": {"A": 0.3, "B": 0.9, "C": 0.5}}
    common = dict(graph=graph, oracle=_oracle(weights), y="y", y_foil="f",
                  candidates=["A", "B", "C"], alpha=0.5, lam=0.05, beta=0.2,
                  abr_iters=1, progress=False)
    abr = solve_game2(solver="abr", **common)
    fp = solve_game2(solver="fp", **common)
    assert fp.evidence_y == abr.evidence_y
    assert fp.evidence_foil == abr.evidence_foil


def test_fp_converges_when_best_responses_repeat() -> None:
    # Agents prefer disjoint nodes, so FP best responses repeat at round 2 and
    # the solver reports convergence with frequency 1.0 on each agent's node.
    graph = _full_graph(["A", "B"])
    oracle = _oracle({"y": {"A": 1.0, "B": 0.1}, "f": {"A": 0.1, "B": 1.0}})
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
        abr_iters=10,
        solver="fp",
        progress=False,
    )
    assert result.converged is True
    assert result.iterations == 2
    assert result.evidence_y == {"A"}
    assert result.evidence_foil == {"B"}
    assert result.node_frequencies_y == pytest.approx({"A": 1.0})
    assert result.node_frequencies_foil == pytest.approx({"B": 1.0})
    assert result.params["solver"] == "fp"
    assert result.params["fp_tol"] is not None


def test_fp_dampens_oscillation_and_reports_frequencies() -> None:
    # Same setup as the C3 ABR oscillation test: symmetric weights, budget=1,
    # high beta. FP mixes over {A, B} instead of hard-cycling; the empirical
    # frequencies are valid probabilities over the candidates and best-iterate
    # tracking still returns the (A, A) allocation by hard-overlap utility.
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
        abr_iters=6,
        solver="fp",
        progress=False,
    )
    assert result.evidence_y == {"A"}
    assert result.evidence_foil == {"A"}
    assert result.utility_y + result.utility_foil == pytest.approx(1.4)
    for freqs in (result.node_frequencies_y, result.node_frequencies_foil):
        assert set(freqs) <= {"A", "B"}
        assert all(0.0 <= value <= 1.0 for value in freqs.values())
        # budget=1 → exactly one node per round → frequencies sum to 1.
        assert sum(freqs.values()) == pytest.approx(1.0)


def test_fp_deterministic_across_runs() -> None:
    graph = _full_graph(["A", "B", "C", "D"])
    weights = {"y": {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4},
               "f": {"A": 0.9, "B": 0.7, "C": 0.8, "D": 0.5}}
    common = dict(graph=graph, y="y", y_foil="f", candidates=["A", "B", "C", "D"],
                  alpha=0.5, lam=0.05, beta=0.2, abr_iters=5, solver="fp",
                  progress=False)
    first = solve_game2(oracle=_oracle(weights), **common)
    second = solve_game2(oracle=_oracle(weights), **common)
    assert first.evidence_y == second.evidence_y
    assert first.evidence_foil == second.evidence_foil
    assert first.node_frequencies_y == second.node_frequencies_y
    assert first.node_frequencies_foil == second.node_frequencies_foil
    assert first.utility_y == pytest.approx(second.utility_y)
    assert first.utility_foil == pytest.approx(second.utility_foil)


def test_abr_default_has_no_frequencies() -> None:
    graph = _full_graph(["A", "B"])
    oracle = _oracle({"y": {"A": 1.0, "B": 0.5}, "f": {"A": 0.5, "B": 1.0}})
    result = solve_game2(graph=graph, oracle=oracle, y="y", y_foil="f",
                         candidates=["A", "B"], abr_iters=3, progress=False)
    assert result.params["solver"] == "abr"
    assert result.params["fp_tol"] is None
    assert result.node_frequencies_y == {}
    assert result.node_frequencies_foil == {}


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


def test_connected_through_excludes_hub_intermediates() -> None:
    # In pruned attribution graphs nearly every feature has an edge to a logit
    # node, so routing connectivity through logits/embeddings made `connected`
    # vacuous. a and b only meet through a logit hub → NOT connected by default.
    graph = CircuitGraph(
        nodes=["a", "L", "b"],
        edges=[("a", "L"), ("b", "L")],
        node_metadata={
            "a": {"feature_type": "cross layer transcoder"},
            "b": {"feature_type": "cross layer transcoder"},
            "L": {"feature_type": "logit"},
        },
    )
    assert graph.connected_through({"a", "b"}) is False
    # Opting out of the exclusion restores the old permissive behavior.
    assert graph.connected_through({"a", "b"}, exclude_intermediate_types=None) is True
    # A subset node is always traversable even if its type is excluded.
    assert graph.connected_through({"a", "L", "b"}) is True


def test_connected_through_allows_error_node_intermediates() -> None:
    graph = CircuitGraph(
        nodes=["a", "err", "b"],
        edges=[("a", "err"), ("err", "b")],
        node_metadata={
            "a": {"feature_type": "cross layer transcoder"},
            "b": {"feature_type": "cross layer transcoder"},
            "err": {"feature_type": "mlp reconstruction error"},
        },
    )
    assert graph.connected_through({"a", "b"}) is True


def test_connected_greedy_does_not_bridge_through_logits() -> None:
    # Both features are valuable, but they only meet through the logit hub, so
    # the connected greedy must keep a singleton instead of a fake sub-circuit.
    graph = CircuitGraph(
        nodes=["a", "L", "b"],
        edges=[("a", "L"), ("b", "L")],
        node_metadata={
            "a": {"feature_type": "cross layer transcoder"},
            "b": {"feature_type": "cross layer transcoder"},
            "L": {"feature_type": "logit"},
        },
    )
    oracle = _oracle({"y": {"a": 5.0, "b": 4.0}})
    result = solve_game1(
        graph=graph,
        oracle=oracle,
        target="y",
        candidates=["a", "b"],
        alpha=0.5,
        lam=0.0,
        connected=True,
        progress=False,
    )
    assert result.evidence == {"a"}


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
    # remove is also keyed by universe fingerprint (backends may intersect with the
    # universe); this backend's remove is universe-independent so the value matches.
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


# ------------------------------------------------------------------ universe safety
class _CapturingModel:
    """Fake ReplacementModel that records the interventions it was asked to run."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def feature_intervention(self, prompt, interventions, **kwargs):
        import numpy as np

        self.calls.append(sorted(interventions))
        return np.zeros(4), None


def test_score_remove_is_restricted_to_universe() -> None:
    model = _CapturingModel()
    scorer = ReplacementModelInterventionScorer(
        model=model,
        prompt="p",
        node_to_intervention={"a": (0, 0, 1), "b": (0, 1, 2)},
        target_to_logit_idx={"y": 0},
        score_kind="logit",
        node_universe={"a"},
    )
    # b is supported but outside the universe; z is entirely unsupported. Neither
    # may be ablated (and z must not raise KeyError).
    scorer.score_remove({"a", "b", "z"}, "y")
    assert model.calls[-1] == [(0, 0, 1, 0.0)]


def test_empty_universe_raises_instead_of_degenerating() -> None:
    with pytest.raises(ValueError):
        ReplacementModelInterventionScorer(
            model=None,
            prompt="p",
            node_to_intervention={"a": (0, 0, 1)},
            target_to_logit_idx={"y": 0},
            node_universe={"not_a_node"},
        )

    scorer = ReplacementModelInterventionScorer(
        model=None,
        prompt="p",
        node_to_intervention={"a": (0, 0, 1)},
        target_to_logit_idx={"y": 0},
    )
    with pytest.raises(ValueError):
        scorer.restrict_universe({"not_a_node"})


# ------------------------------------------------------------------ determinism
def test_subgraph_nodes_and_edges_are_sorted() -> None:
    graph = _full_graph(["c", "a", "b"], edges=[("c", "a"), ("b", "c"), ("a", "b")])
    sub = graph.subgraph({"c", "b", "a"})
    assert sub.nodes() == ["a", "b", "c"]
    assert sub.edges() == [("a", "b"), ("b", "c"), ("c", "a")]


def test_from_dict_prefers_node_id_over_id() -> None:
    # Must match the intervention loader's precedence (node_id first) so graph
    # nodes and oracle candidates can never silently diverge.
    payload = {
        "nodes": [{"node_id": "canonical", "id": "legacy", "feature_type": "cross layer transcoder"}],
        "edges": [],
    }
    graph = CircuitGraph.from_dict(payload)
    assert graph.has_node("canonical")
    assert not graph.has_node("legacy")


# ------------------------------------------------------------------ per-solve stats
def test_oracle_stats_are_per_solve() -> None:
    graph = _full_graph(["a", "b"])
    oracle = _oracle({"y": {"a": 2.0, "b": 1.0}})
    first = solve_game1(graph=graph, oracle=oracle, target="y",
                        candidates=["a", "b"], lam=0.0, progress=False)
    assert first.oracle_calls > 0
    # Same solve again on the same (cached) oracle: everything is a cache hit, and
    # the reported stats must reflect THIS solve, not the cumulative history.
    second = solve_game1(graph=graph, oracle=oracle, target="y",
                         candidates=["a", "b"], lam=0.0, progress=False)
    assert second.oracle_calls == 0
    assert second.cache_hits > 0


def test_game2_reports_best_iteration() -> None:
    graph = _full_graph(["A", "B"])
    oracle = _oracle({"y": {"A": 1.0, "B": 0.9}, "f": {"A": 1.0, "B": 0.9}})
    result = solve_game2(
        graph=graph, oracle=oracle, y="y", y_foil="f", candidates=["A", "B"],
        alpha=0.5, lam=0.0, beta=0.3, budget=1, abr_iters=4, progress=False,
    )
    # A real solver round (not the initial empty allocation) produced the result.
    assert result.best_iteration >= 1
    assert result.evidence_y == {"A"}


# ------------------------------------------------------------------ salience guards
def test_node_salience_ignores_nan() -> None:
    from macag.utils.supernodes import node_salience

    assert node_salience({"influence": float("nan"), "activation": 2.0}) == pytest.approx(0.4)


# ------------------------------------------------------- dual freeze / attention probe
def _metrics_with_range(range_value: float):
    """FaithfulnessMetrics stub where only recoverable_range matters."""
    from macag.utils.metrics import FaithfulnessMetrics

    return FaithfulnessMetrics(
        all_score=range_value,
        empty_score=0.0,
        keep_only_score=0.0,
        remove_score=0.0,
        sufficiency=0.0,
        necessity=0.0,
        faithfulness_delta=0.0,
        recoverable_range=range_value,
        sufficiency_normalized=0.0,
        necessity_normalized=0.0,
        faithfulness_delta_normalized=0.0,
    )


def _diagnostic(range_frozen: float, range_unfrozen: float,
                frozen_evidence: set[str] | None = None,
                unfrozen_evidence: set[str] | None = None,
                graph: CircuitGraph | None = None):
    from macag.utils.attention_mediation import compute_attention_mediation_diagnostic

    return compute_attention_mediation_diagnostic(
        graph=graph if graph is not None else _full_graph(["a", "b"]),
        frozen_metrics=_metrics_with_range(range_frozen),
        unfrozen_metrics=_metrics_with_range(range_unfrozen),
        frozen_evidence=frozen_evidence if frozen_evidence is not None else set(),
        unfrozen_evidence=unfrozen_evidence if unfrozen_evidence is not None else set(),
    )


def test_dual_solver_returns_both_results_under_matched_params() -> None:
    from macag.games.game1_attention_probe import solve_game1_dual

    graph = _full_graph(["a", "b", "c"])
    frozen_oracle = _oracle({"y": {"a": 10.0, "b": 1.0, "c": 0.1}})
    unfrozen_oracle = _oracle({"y": {"a": 2.0, "b": 5.0, "c": 4.0}})
    result = solve_game1_dual(
        graph=graph,
        frozen_oracle=frozen_oracle,
        unfrozen_oracle=unfrozen_oracle,
        target="y",
        candidates=["a", "b", "c"],
        alpha=0.5,
        lam=0.0,
        budget=2,
        faithfulness_eps=0.2,
        progress=False,
    )
    # Frozen leg: a (gain 10) then raw_relative stop (b's gain 1 < 0.2*10).
    assert result.frozen.evidence == {"a"}
    # Unfrozen leg: b (5) then c (4 >= 0.2*5), then budget cap.
    assert result.unfrozen.evidence == {"b", "c"}
    # Both legs ran under identical matched params with the forced stop metric.
    for leg in (result.frozen, result.unfrozen):
        assert leg.params["stop_metric"] == "raw_relative"
        assert leg.params["budget"] == 2
        assert leg.params["alpha"] == 0.5
        assert leg.oracle_calls > 0  # independent per-leg stats
    assert result.params["matched"] is True
    assert result.params["stop_metric"] == "raw_relative"
    # Both ranges positive -> feature-mediated; disjoint evidence -> jaccard 0.
    assert result.diagnostic.verdict == "feature_mediated"
    assert result.diagnostic.evidence_jaccard == pytest.approx(0.0)


def test_dual_diagnostic_flip_is_attention_mediated() -> None:
    diag = _diagnostic(range_frozen=-3.0, range_unfrozen=5.0)
    assert diag.range_flip is True
    assert diag.reverse_flip is False
    assert diag.verdict == "attention_mediated"


def test_dual_diagnostic_both_positive_is_feature_mediated() -> None:
    diag = _diagnostic(range_frozen=4.0, range_unfrozen=5.0)
    assert diag.range_flip is False
    assert diag.verdict == "feature_mediated"


def test_dual_diagnostic_both_negative_is_indeterminate() -> None:
    diag = _diagnostic(range_frozen=-4.0, range_unfrozen=-5.0)
    assert diag.range_flip is False
    assert diag.reverse_flip is False
    assert diag.verdict == "indeterminate"


def test_dual_diagnostic_reverse_flip_is_indeterminate_and_flagged() -> None:
    diag = _diagnostic(range_frozen=4.0, range_unfrozen=-5.0)
    assert diag.range_flip is False
    assert diag.reverse_flip is True
    assert diag.verdict == "indeterminate"


def test_dual_diagnostic_jaccard_and_set_decomposition() -> None:
    diag = _diagnostic(
        range_frozen=1.0,
        range_unfrozen=1.0,
        frozen_evidence={"a", "b"},
        unfrozen_evidence={"b", "c"},
        graph=_full_graph(["a", "b", "c"]),
    )
    assert diag.evidence_jaccard == pytest.approx(1.0 / 3.0)
    assert diag.evidence_shared == ["b"]
    assert diag.evidence_only_frozen == ["a"]
    assert diag.evidence_only_unfrozen == ["c"]
    assert diag.evidence_size_frozen == 2
    assert diag.evidence_size_unfrozen == 2

    # Two empty evidence sets are identical, not zero-overlap.
    empty = _diagnostic(range_frozen=1.0, range_unfrozen=1.0)
    assert empty.evidence_jaccard == pytest.approx(1.0)


def test_dual_diagnostic_upstream_early_counts_from_metadata() -> None:
    graph = CircuitGraph(
        nodes=["f_early", "f_late", "L"],
        node_metadata={
            "f_early": {"feature_type": "cross layer transcoder", "layer": "0", "ctx_idx": 1},
            "f_late": {"feature_type": "cross layer transcoder", "layer": "5", "ctx_idx": 2},
            # Logit node: excluded from depth, but its position marks the
            # prediction token.
            "L": {"feature_type": "logit", "layer": "8", "ctx_idx": 2},
        },
    )
    diag = _diagnostic(
        range_frozen=1.0,
        range_unfrozen=1.0,
        frozen_evidence={"f_late"},
        unfrozen_evidence={"f_early", "f_late"},
        graph=graph,
    )
    assert diag.n_layers == 6  # max feature layer 5 + 1; logit layer 8 excluded
    assert diag.final_ctx_idx == 2
    assert diag.upstream_count_frozen == 0  # f_late sits at the prediction token
    assert diag.upstream_count_unfrozen == 1  # f_early at ctx 1 < 2
    assert diag.early_count_frozen == 0  # layer 5 >= 6/3
    assert diag.early_count_unfrozen == 1  # layer 0 < 2


def test_dual_diagnostic_counts_from_node_id_fallback() -> None:
    # No metadata: layer/pos parse from the "{layer}_{feature}_{pos}" convention.
    graph = _full_graph(["0_7_1", "5_3_4"])
    diag = _diagnostic(
        range_frozen=1.0,
        range_unfrozen=1.0,
        frozen_evidence={"0_7_1"},
        unfrozen_evidence={"5_3_4"},
        graph=graph,
    )
    assert diag.n_layers == 6
    assert diag.final_ctx_idx == 4
    assert diag.upstream_count_frozen == 1  # pos 1 < 4
    assert diag.upstream_count_unfrozen == 0  # pos 4 == final
    assert diag.early_count_frozen == 1  # layer 0 < 2
    assert diag.early_count_unfrozen == 0  # layer 5 >= 2


def test_dual_diagnostic_counts_none_when_unresolvable() -> None:
    diag = _diagnostic(
        range_frozen=1.0,
        range_unfrozen=1.0,
        frozen_evidence={"a"},
        unfrozen_evidence={"b"},
        graph=_full_graph(["a", "b"]),
    )
    assert diag.n_layers is None
    assert diag.final_ctx_idx is None
    assert diag.upstream_count_frozen is None
    assert diag.upstream_count_unfrozen is None
    assert diag.early_count_frozen is None
    assert diag.early_count_unfrozen is None
    # Keys stay present (None-valued) in the serialized form: stable schema.
    as_dict = diag.to_dict()
    assert "upstream_count_frozen" in as_dict and as_dict["upstream_count_frozen"] is None


def test_dual_solver_rejects_swapped_freeze_oracles() -> None:
    from macag.games.game1_attention_probe import solve_game1_dual

    def _rm_oracle(freeze: bool) -> ScoringOracle:
        scorer = ReplacementModelInterventionScorer(
            model=_CapturingModel(),
            prompt="p",
            node_to_intervention={"a": (0, 0, 1)},
            target_to_logit_idx={"y": 0},
            score_kind="logit",
            freeze_attention=freeze,
        )
        return ScoringOracle(backend=scorer, cache_enabled=True)

    with pytest.raises(ValueError, match="swapped"):
        solve_game1_dual(
            graph=_full_graph(["a"]),
            frozen_oracle=_rm_oracle(False),  # wired backwards
            unfrozen_oracle=_rm_oracle(True),
            target="y",
            candidates=["a"],
            progress=False,
        )


# ------------------------------------------------------------------ freeze derivation
def test_derive_oracle_with_freeze_shares_model_and_resets_cache() -> None:
    from macag.scoring import derive_oracle_with_freeze

    model = _CapturingModel()
    scorer = ReplacementModelInterventionScorer(
        model=model,
        prompt="p",
        node_to_intervention={"a": (0, 0, 1)},
        target_to_logit_idx={"y": 0},
        score_kind="logit",
    )
    oracle = ScoringOracle(backend=scorer, cache_enabled=True)
    oracle.empty("y")  # populate the source cache

    derived = derive_oracle_with_freeze(oracle, False)
    assert derived.backend is not scorer
    assert derived.backend.freeze_attention is False
    assert derived.backend.model is model  # shared shallowly, never reloaded
    assert derived.cache_stats() == {"oracle_calls": 0, "cache_hits": 0, "cache_size": 0}
    # Source untouched.
    assert scorer.freeze_attention is True
    assert oracle.cache_stats()["cache_size"] >= 1


def test_derive_oracle_preserves_restricted_universe() -> None:
    from macag.scoring import derive_oracle_with_freeze

    scorer = ReplacementModelInterventionScorer(
        model=_CapturingModel(),
        prompt="p",
        node_to_intervention={"a": (0, 0, 1), "b": (0, 1, 2)},
        target_to_logit_idx={"y": 0},
        score_kind="logit",
    )
    oracle = ScoringOracle(backend=scorer, cache_enabled=True)
    scorer.restrict_universe({"a"})  # post-construction restriction (the CLI path)

    derived = derive_oracle_with_freeze(oracle, False)
    # dataclasses.replace re-runs __post_init__, which would silently rebuild the
    # universe from the node_universe FIELD; the helper must re-apply the live one.
    assert derived.backend.intervention_universe() == {"a"}


def test_derive_oracle_rejects_toy_backend() -> None:
    from macag.scoring import derive_oracle_with_freeze

    oracle = _oracle({"y": {"a": 1.0}})
    with pytest.raises(TypeError, match="freeze_attention"):
        derive_oracle_with_freeze(oracle, False)


def test_universe_fingerprint_includes_freeze_attention() -> None:
    model = _CapturingModel()
    scorer = ReplacementModelInterventionScorer(
        model=model,
        prompt="p",
        node_to_intervention={"a": (0, 0, 1)},
        target_to_logit_idx={"y": 0},
        score_kind="logit",
    )
    oracle = ScoringOracle(backend=scorer, cache_enabled=True)

    oracle.empty("y")
    assert len(model.calls) == 1
    oracle.empty("y")  # cache hit
    assert len(model.calls) == 1

    fingerprint_frozen = scorer.universe_fingerprint()
    scorer.freeze_attention = False  # in-place flip must bust the memoized entry
    assert scorer.universe_fingerprint() != fingerprint_frozen
    oracle.empty("y")
    assert len(model.calls) == 2

    scorer.freeze_attention = True  # flipping back re-hits the original entry
    oracle.empty("y")
    assert len(model.calls) == 2


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
