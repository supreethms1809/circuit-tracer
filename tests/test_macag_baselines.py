"""Tests for the MACAG baseline selectors and head-to-head harness (Phase 2).

All tests run on the dependency-free toy backends (no model / GPU):
- ToyAdditiveInterventionScorer for exact-value checks (Shapley == weights),
- a CallbackInterventionScorer synergy game for the greedy-stall / optimality
  gap construction of macag.md §3.2.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from macag.baselines.acdc_prune import (
    ACDCBudgetUnreachableError,
    acdc_prune,
    acdc_target_size,
    acdc_tau_sweep,
)
from macag.baselines.bruteforce import best_subset_bruteforce
from macag.baselines.common import (
    coalition_value,
    jaccard,
    precision_at_k,
    precision_at_k_uncertainty_bounds,
    ranking_from_scores,
    spearman_rank_correlation,
    tie_aware_precision_at_k,
)
from macag.baselines.eap import (
    EAPUnavailableError,
    compute_eap_node_scores,
    select_top_eap,
)
from macag.baselines.influence import select_top_influence
from macag.baselines.shapley_select import (
    estimate_banzhaf,
    estimate_shapley,
    select_top_shapley,
)
from macag.cli.run_baselines import main as run_baselines_main
from macag.games.game1_min_faithful import solve_game1
from macag.graph import CircuitGraph
from macag.scoring import (
    CallbackInterventionScorer,
    ScoringOracle,
    ToyAdditiveInterventionScorer,
)

WEIGHTS = {"a": 6.0, "b": 4.0, "c": 1.0}


def _identity(tag: str = "same") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input_id": "p1",
        "target": "y",
        "graph_sha256": f"graph-{tag}",
        "oracle_kwargs_sha256": f"oracle-{tag}",
        "candidates_sha256": f"candidates-{tag}",
        "score_kind": "logit_gap",
        "model_name": "toy",
        "local_clt_path": None,
        "transcoder_set": None,
        "clt_scan": "toy",
        "freeze_attention": True,
        "ablation_mode": "zero",
    }


def _additive_oracle(base: float = 100.0) -> ScoringOracle:
    backend = ToyAdditiveInterventionScorer(
        weights_by_target={"y": dict(WEIGHTS)},
        base_by_target={"y": base},
    )
    return ScoringOracle(backend=backend, cache_enabled=True)


def _synergy_oracle() -> ScoringOracle:
    """keep(S) = 10 if {a,b} <= S else 0, plus 1 if c in S (macag.md §3.2 synergy)."""

    def keep(nodes: set[Any]) -> float:
        value = 10.0 if {"a", "b"} <= set(nodes) else 0.0
        if "c" in nodes:
            value += 1.0
        return value

    universe = {"a", "b", "c"}
    backend = CallbackInterventionScorer(
        score_all_fn=lambda target: keep(universe),
        score_empty_fn=lambda target: keep(set()),
        score_keep_only_fn=lambda nodes, target: keep(set(nodes)),
        score_remove_fn=lambda nodes, target: keep(universe - set(nodes)),
    )
    return ScoringOracle(backend=backend, cache_enabled=True)


def _graph() -> CircuitGraph:
    return CircuitGraph(
        nodes=["a", "b", "c"],
        node_metadata={
            "a": {"feature_type": "cross layer transcoder", "layer": "0", "ctx_idx": 1, "influence": 6.0},
            "b": {"feature_type": "cross layer transcoder", "layer": "1", "ctx_idx": 1, "influence": 4.0},
            "c": {"feature_type": "cross layer transcoder", "layer": "2", "ctx_idx": 1, "influence": 1.0},
        },
    )


# ------------------------------------------------------------------ rank stats
def test_spearman_perfect_inverse_and_ties() -> None:
    a = {"x": 1.0, "y": 2.0, "z": 3.0}
    assert spearman_rank_correlation(a, {"x": 10.0, "y": 20.0, "z": 30.0}) == pytest.approx(1.0)
    assert spearman_rank_correlation(a, {"x": 3.0, "y": 2.0, "z": 1.0}) == pytest.approx(-1.0)
    # Constant side -> undefined.
    assert spearman_rank_correlation(a, {"x": 5.0, "y": 5.0, "z": 5.0}) is None
    # Fewer than two common keys -> undefined.
    assert spearman_rank_correlation({"x": 1.0}, {"x": 2.0}) is None
    # Ties get average ranks and still correlate positively with the untied order.
    tied = spearman_rank_correlation(a, {"x": 1.0, "y": 1.0, "z": 2.0})
    assert tied is not None and 0.0 < tied < 1.0


def test_precision_and_jaccard() -> None:
    assert precision_at_k(["a", "b", "c"], ["a", "c", "b"], 2) == pytest.approx(0.5)
    assert precision_at_k(["a", "b"], ["a", "b"], 0) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1.0 / 3.0)
    assert jaccard(set(), set()) == 0.0


# ------------------------------------------------------------------- influence
def test_influence_ranking_and_missing_metadata() -> None:
    graph = _graph()
    graph.add_node("d", metadata={"feature_type": "cross layer transcoder"})  # no influence
    result = select_top_influence(graph, ["d", "c", "b", "a"])
    assert result.ranking == ["a", "b", "c", "d"]
    assert result.scores == {"a": 6.0, "b": 4.0, "c": 1.0}
    assert result.extras["missing_influence_count"] == 1


def test_influence_signed_vs_absolute() -> None:
    graph = CircuitGraph(
        nodes=["p", "n"],
        node_metadata={"p": {"influence": 2.0}, "n": {"influence": -5.0}},
    )
    assert select_top_influence(graph, ["p", "n"]).ranking == ["n", "p"]
    assert select_top_influence(graph, ["p", "n"], use_absolute=False).ranking == ["p", "n"]


def test_influence_requires_some_influence() -> None:
    graph = CircuitGraph(nodes=["a"], node_metadata={"a": {}})
    with pytest.raises(ValueError, match="influence"):
        select_top_influence(graph, ["a"])


# ------------------------------------------------------------------------- eap
def _eap_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {"node_id": "f1", "feature_type": "cross layer transcoder", "layer": "0", "ctx_idx": 1},
            {"node_id": "f2", "feature_type": "cross layer transcoder", "layer": "1", "ctx_idx": 1},
            {
                "node_id": "L_t",
                "feature_type": "logit",
                "clerp": 'Output " target"',
                "is_target_logit": True,
            },
            {
                "node_id": "L_f",
                "feature_type": "logit",
                "clerp": 'Output " foil"',
                "is_target_logit": False,
            },
        ],
        "links": [
            {"source": "f1", "target": "f2", "weight": 2.0},
            {"source": "f2", "target": "L_t", "weight": 3.0},
            {"source": "f1", "target": "L_t", "weight": 0.5},
            {"source": "f2", "target": "L_f", "weight": 1.0},
        ],
    }


def test_eap_path_effects_target_only() -> None:
    effects, info = compute_eap_node_scores(_eap_payload())
    assert info["converged"] is True
    assert effects["L_t"] == pytest.approx(1.0)
    assert effects["L_f"] == pytest.approx(0.0)
    assert effects["f2"] == pytest.approx(3.0)  # 3*1 + 1*0
    assert effects["f1"] == pytest.approx(6.5)  # 2*3 + 0.5*1


def test_eap_path_effects_with_foil() -> None:
    effects, _ = compute_eap_node_scores(_eap_payload(), foil_match=" foil")
    assert effects["f2"] == pytest.approx(2.0)  # 3*1 + 1*(-1)
    assert effects["f1"] == pytest.approx(4.5)  # 2*2 + 0.5*1


def test_eap_target_match_overrides_flag_and_errors() -> None:
    payload = _eap_payload()
    effects, info = compute_eap_node_scores(payload, target_match=" foil")
    assert effects["f2"] == pytest.approx(1.0)
    assert "L_f" in info["seeds"]

    with pytest.raises(ValueError, match="No target logit seed"):
        compute_eap_node_scores(payload, target_match="no such clerp")

    with pytest.raises(EAPUnavailableError, match="different objective"):
        compute_eap_node_scores(payload, foil_match="no such clerp")

    unweighted = {
        "nodes": payload["nodes"],
        "links": [{"source": "f1", "target": "L_t"}],
    }
    with pytest.raises(ValueError, match="weight"):
        compute_eap_node_scores(unweighted)


def test_eap_matches_quoted_logit_token_not_clerp_substring() -> None:
    """Regression: stripped 'A'/'1' must not hit 'cellular' or '(p=0.010)'."""
    payload = {
        "nodes": [
            {"node_id": "f1", "feature_type": "cross layer transcoder"},
            {
                "node_id": "L_a",
                "feature_type": "logit",
                "clerp": 'Output " A" (p=0.668)',
                "is_target_logit": True,
            },
            {
                "node_id": "L_cell",
                "feature_type": "logit",
                "clerp": 'Output " cellular" (p=0.010)',
                "is_target_logit": False,
            },
            {
                "node_id": "L_b",
                "feature_type": "logit",
                "clerp": 'Output " B" (p=0.062)',
                "is_target_logit": False,
            },
        ],
        "links": [
            {"source": "f1", "target": "L_a", "weight": 1.0},
            {"source": "f1", "target": "L_cell", "weight": 1.0},
            {"source": "f1", "target": "L_b", "weight": 1.0},
        ],
    }
    effects, info = compute_eap_node_scores(payload, target_match=" A")
    assert info["seeds"] == {"L_a": 1.0}
    assert effects["L_a"] == pytest.approx(1.0)
    assert effects["L_cell"] == pytest.approx(0.0)
    with pytest.raises(EAPUnavailableError, match="different objective"):
        compute_eap_node_scores(payload, target_match=" A", foil_match="1")

    # Leading-space and stripped letter forms both select the A logit only.
    _, info_stripped = compute_eap_node_scores(payload, target_match="A", foil_match="B")
    assert info_stripped["seeds"] == {"L_a": 1.0, "L_b": -1.0}

    # Case-sensitive: " D" must not also seed lowercase " d".
    payload_case = {
        "nodes": [
            {"node_id": "f1", "feature_type": "cross layer transcoder"},
            {
                "node_id": "L_D",
                "feature_type": "logit",
                "clerp": 'Output " D" (p=0.562)',
                "is_target_logit": True,
            },
            {
                "node_id": "L_d",
                "feature_type": "logit",
                "clerp": 'Output " d" (p=0.012)',
                "is_target_logit": False,
            },
        ],
        "links": [
            {"source": "f1", "target": "L_D", "weight": 1.0},
            {"source": "f1", "target": "L_d", "weight": 1.0},
        ],
    }
    _, info_case = compute_eap_node_scores(payload_case, target_match=" D")
    assert info_case["seeds"] == {"L_D": 1.0}


def test_select_top_eap_ranks_candidates_only() -> None:
    result = select_top_eap(_eap_payload(), ["f1", "f2"], foil_match=" foil")
    assert result.ranking == ["f1", "f2"]
    assert result.scores == {"f1": pytest.approx(4.5), "f2": pytest.approx(2.0)}


# --------------------------------------------------------------------- shapley
def test_shapley_exact_on_additive_game() -> None:
    oracle = _additive_oracle()
    estimate = estimate_shapley(oracle, "y", ["a", "b", "c"], alpha=0.5, permutations=4, seed=0)
    for node, weight in WEIGHTS.items():
        assert estimate.values[node] == pytest.approx(weight)
        assert estimate.std_errors[node] == pytest.approx(0.0, abs=1e-9)
    assert estimate.efficiency_gap == pytest.approx(0.0, abs=1e-9)
    assert estimate.ranking() == ["a", "b", "c"]


def test_shapley_deterministic_and_efficient_on_synergy_game() -> None:
    oracle = _synergy_oracle()
    first = estimate_shapley(oracle, "y", ["a", "b", "c"], alpha=1.0, permutations=16, seed=7)
    second = estimate_shapley(oracle, "y", ["a", "b", "c"], alpha=1.0, permutations=16, seed=7)
    assert first.values == second.values

    # Permutation marginals telescope, so MC Shapley is exactly efficient.
    assert first.efficiency_gap == pytest.approx(0.0, abs=1e-9)
    assert first.grand_value == pytest.approx(11.0)
    # c contributes +1 in every ordering; a and b split the synergy pair.
    assert first.values["c"] == pytest.approx(1.0)
    assert first.values["a"] + first.values["b"] == pytest.approx(10.0)


def test_shapley_checkpoint_resume_matches_full_run(tmp_path) -> None:
    ckpt = tmp_path / "shapley.ckpt.json"
    estimate_shapley(
        _synergy_oracle(), "y", ["a", "b", "c"], alpha=1.0, permutations=4, seed=3, checkpoint_path=ckpt
    )
    resumed = estimate_shapley(
        _synergy_oracle(), "y", ["a", "b", "c"], alpha=1.0, permutations=8, seed=3, checkpoint_path=ckpt
    )
    full = estimate_shapley(_synergy_oracle(), "y", ["a", "b", "c"], alpha=1.0, permutations=8, seed=3)
    assert resumed.values == full.values
    assert resumed.efficiency_gap == pytest.approx(0.0, abs=1e-9)


def test_shapley_antithetic_standard_error_uses_pair_means() -> None:
    estimate = estimate_shapley(
        _synergy_oracle(),
        "y",
        ["a", "b"],
        alpha=1.0,
        permutations=4,
        seed=3,
        antithetic=True,
    )
    # Within each permutation/reverse pair, a and b have marginals {0, 10};
    # the independent pair mean is always 5, so its SE is exactly zero.
    assert estimate.values == {"a": pytest.approx(5.0), "b": pytest.approx(5.0)}
    assert estimate.std_errors["a"] == pytest.approx(0.0)
    assert estimate.std_errors["b"] == pytest.approx(0.0)
    assert estimate.params["independent_draws"] == 2
    assert estimate.params["uncertainty_unit"] == "antithetic_pair_mean"
    selected = select_top_shapley(
        _additive_oracle(), "y", ["a", "b", "c"], permutations=4
    )
    assert selected.extras["rank_stability_95"]["1"]["stable"] is True


def test_shapley_one_draw_uncertainty_is_undefined() -> None:
    estimate = estimate_shapley(
        _synergy_oracle(),
        "y",
        ["a", "b"],
        alpha=1.0,
        permutations=1,
        antithetic=False,
    )
    assert all(value != value for value in estimate.std_errors.values())
    with pytest.raises(ValueError, match="even"):
        estimate_shapley(
            _synergy_oracle(),
            "y",
            ["a", "b"],
            permutations=3,
            antithetic=True,
        )


def test_banzhaf_exact_on_additive_game() -> None:
    oracle = _additive_oracle()
    estimate = estimate_banzhaf(oracle, "y", ["a", "b", "c"], alpha=0.5, samples=4, seed=0)
    for node, weight in WEIGHTS.items():
        assert estimate.values[node] == pytest.approx(weight)
    assert estimate.estimator == "banzhaf"
    assert estimate.efficiency_gap is None


def test_tie_and_uncertainty_aware_gold_agreement() -> None:
    gold = {"a": 2.0, "b": 1.0, "c": 1.0}
    # At k=2, b/c split one boundary slot, so selecting a+c earns 1 + 1/2.
    assert tie_aware_precision_at_k(["a", "c"], gold, 2) == pytest.approx(0.75)
    lower, upper = precision_at_k_uncertainty_bounds(
        ["a", "c"],
        gold,
        {"a": 0.01, "b": 1.0, "c": 1.0},
        2,
    )
    assert 0.0 <= lower <= upper <= 1.0
    assert upper == pytest.approx(1.0)


def test_score_validation_rejects_nonfinite_and_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="finite"):
        ranking_from_scores({"a": float("nan"), "b": 1.0})
    with pytest.raises(ValueError, match="alpha"):
        estimate_shapley(_additive_oracle(), "y", ["a"], alpha=1.1)
    with pytest.raises(ValueError, match="alpha"):
        best_subset_bruteforce(_additive_oracle(), "y", ["a"], k=1, alpha=-0.1)



# ------------------------------------------------------------------------ acdc
def test_acdc_prunes_below_threshold_weights() -> None:
    oracle = _additive_oracle()
    result = acdc_prune(_graph(), oracle, "y", ["a", "b", "c"], tau=2.0, alpha=0.5)
    assert result.kept == ["a", "b"]
    assert result.removed_order == ["c"]
    assert result.value == pytest.approx(10.0)
    # top_down order: highest layer first -> c (layer 2) tested first.
    assert [d["node"] for d in result.decisions] == ["c", "b", "a"]


def test_acdc_given_order_and_sweep() -> None:
    oracle = _additive_oracle()
    result = acdc_prune(_graph(), oracle, "y", ["a", "b", "c"], tau=2.0, alpha=0.5, order="given")
    assert [d["node"] for d in result.decisions] == ["a", "b", "c"]

    sweep = acdc_tau_sweep(_graph(), oracle, "y", ["a", "b", "c"], taus=[5.0, 0.5], alpha=0.5)
    assert [r.tau for r in sweep] == [0.5, 5.0]
    assert set(sweep[0].kept) == {"a", "b", "c"}  # tau below every weight keeps all
    assert set(sweep[1].kept) == {"a"}  # only a's degradation (6) clears tau=5


def test_acdc_target_size_hits_exact_sizes_on_additive_game() -> None:
    oracle = _additive_oracle()
    for target_k, expected in ((2, {"a", "b"}), (1, {"a"}), (3, {"a", "b", "c"})):
        result = acdc_target_size(_graph(), oracle, "y", ["a", "b", "c"], target_k=target_k)
        assert set(result.kept) == expected, target_k
        assert result.params["achieved_k"] == target_k
        assert result.params["exact"] is True
        assert result.params["target_k"] == target_k


def test_acdc_target_size_unreachable_returns_nearest_under_budget() -> None:
    """Synergy game only realizes sizes {3, 2, 0}; k=1 is a plateau gap.

    Hard budget forbids overshoot: size 2 is nearer than 0 by abs-distance but
    exceeds target_k, so matched_k must return the empty under-budget set.
    """
    oracle = _synergy_oracle()
    graph = CircuitGraph(nodes=["a", "b", "c"])
    result = acdc_target_size(graph, oracle, "y", ["a", "b", "c"], target_k=1, alpha=1.0)
    assert set(result.kept) == set()
    assert result.params["achieved_k"] == 0
    assert result.params["exact"] is False
    assert result.params["budget_capped"] is True
    assert result.params["bisection_iters"] >= 1


def test_acdc_target_size_never_returns_overshoot_when_under_budget_exists() -> None:
    """Prefer size 7 over size 9 when targeting 8 (both distance 1)."""
    # Additive weights: prune order drops smallest degradation first under large tau.
    # With distinct positive weights, exact prefixes of the top-down order are
    # achievable; construct so size 7 and 9 both appear and 8 does not.
    backend = ToyAdditiveInterventionScorer(
        weights_by_target={
            "y": {f"n{i}": float(10 - i) for i in range(10)}
        },
        base_by_target={"y": 0.0},
    )
    oracle = ScoringOracle(backend=backend, cache_enabled=True)
    nodes = [f"n{i}" for i in range(10)]
    graph = CircuitGraph(
        nodes=nodes,
        node_metadata={
            n: {"feature_type": "cross layer transcoder", "layer": str(i), "ctx_idx": 0}
            for i, n in enumerate(nodes)
        },
    )
    # Seed only sizes that skip 8: tau=0 keeps 10; a large tau that leaves 7;
    # and a mid tau that leaves 9. The search may discover 8 — if so exact is fine;
    # the invariant under test is never returning >8 when ≤8 exists.
    seed = [
        acdc_prune(graph, oracle, "y", nodes, tau=0.0, alpha=1.0),
        acdc_prune(graph, oracle, "y", nodes, tau=1.5, alpha=1.0),
        acdc_prune(graph, oracle, "y", nodes, tau=3.5, alpha=1.0),
    ]
    result = acdc_target_size(
        graph,
        oracle,
        "y",
        nodes,
        target_k=8,
        alpha=1.0,
        seed_results=seed,
        max_iters=48,
    )
    assert result.params["achieved_k"] <= 8
    assert result.params["budget_capped"] is True
    if result.params["exact"]:
        assert result.params["achieved_k"] == 8
    else:
        # Closest under-budget among evaluated feasible sets.
        assert result.params["achieved_k"] < 8


def test_acdc_target_size_seed_midpoint_collision() -> None:
    """Symmetric degradations make the first bisection midpoint hit the tau=0 seed.

    Weights {a:5, b:4, c:-5} give lo=-6, hi=6, mid=0.0 — already evaluated. The
    search must reuse that result to narrow the bracket (not break), eventually
    reaching tau≈4.5 where exactly {a} survives for target_k=1.
    """
    backend = ToyAdditiveInterventionScorer(
        weights_by_target={"y": {"a": 5.0, "b": 4.0, "c": -5.0}},
        base_by_target={"y": 100.0},
    )
    oracle = ScoringOracle(backend=backend, cache_enabled=True)
    graph = CircuitGraph(nodes=["a", "b", "c"])
    result = acdc_target_size(graph, oracle, "y", ["a", "b", "c"], target_k=1)
    assert set(result.kept) == {"a"}
    assert result.params["achieved_k"] == 1
    assert result.params["exact"] is True


def test_acdc_target_size_validates_inputs() -> None:
    oracle = _additive_oracle()
    with pytest.raises(ValueError, match="target_k"):
        acdc_target_size(_graph(), oracle, "y", ["a", "b", "c"], target_k=0)
    with pytest.raises(ValueError, match="non-empty"):
        acdc_target_size(_graph(), oracle, "y", ["zz"], target_k=1)


# ------------------------------------------------------------------ bruteforce
def test_bruteforce_finds_top_pair_on_additive_game() -> None:
    oracle = _additive_oracle()
    result = best_subset_bruteforce(oracle, "y", ["a", "b", "c"], k=2, alpha=0.5)
    assert set(result.best_set) == {"a", "b"}
    assert result.best_value == pytest.approx(10.0)
    assert result.evaluations == 3


def test_bruteforce_eval_guard() -> None:
    oracle = _additive_oracle()
    with pytest.raises(ValueError, match="max_evaluations"):
        best_subset_bruteforce(oracle, "y", [f"n{i}" for i in range(30)], k=3, max_evaluations=100)


def test_near_ties_use_tolerance_and_deterministic_node_order() -> None:
    scores = {"b": 1.0 + 5e-13, "a": 1.0}
    assert ranking_from_scores(scores, tie_tol=1e-12) == ["a", "b"]
    oracle = ScoringOracle(
        backend=ToyAdditiveInterventionScorer(
            weights_by_target={"y": scores},
            base_by_target={"y": 0.0},
        ),
        cache_enabled=True,
    )
    result = best_subset_bruteforce(
        oracle, "y", ["a", "b"], k=1, alpha=1.0, tie_tol=1e-12
    )
    assert result.best_set == ["a"]
    assert result.ties == 1


def test_greedy_stalls_on_synergy_but_bruteforce_finds_pair() -> None:
    """The §3.2 super-modular spike: no singleton gain, big pair value."""
    oracle = _synergy_oracle()
    graph = CircuitGraph(nodes=["a", "b", "c"])
    greedy = solve_game1(
        graph, oracle, "y", candidates=["a", "b", "c"], alpha=1.0, lam=0.0, budget=2, progress=False
    )
    assert greedy.evidence == {"c"}  # greedy stalls after the singleton

    oracle.clear_cache()
    brute = best_subset_bruteforce(oracle, "y", ["a", "b", "c"], k=2, alpha=1.0)
    assert set(brute.best_set) == {"a", "b"}
    assert brute.best_value == pytest.approx(10.0)
    greedy_value = coalition_value(oracle, "y", greedy.evidence, alpha=1.0)
    assert brute.best_value - greedy_value == pytest.approx(9.0)  # the optimality gap


# --------------------------------------------------------------------- harness
def test_run_baselines_harness_end_to_end(tmp_path) -> None:
    graph_payload = {
        "nodes": [
            {"node_id": "a", "feature_type": "cross layer transcoder", "layer": "0", "ctx_idx": 1, "influence": 6.0},
            {"node_id": "b", "feature_type": "cross layer transcoder", "layer": "1", "ctx_idx": 1, "influence": 4.0},
            {"node_id": "c", "feature_type": "cross layer transcoder", "layer": "2", "ctx_idx": 1, "influence": 1.0},
            {"node_id": "L", "feature_type": "logit", "clerp": 'Output " y"', "is_target_logit": True},
        ],
        "links": [
            {"source": "a", "target": "L", "weight": 6.0},
            {"source": "b", "target": "L", "weight": 4.0},
            {"source": "c", "target": "L", "weight": 1.0},
        ],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph_payload))

    toy_path = tmp_path / "toy.json"
    toy_path.write_text(
        json.dumps({"weights_by_target": {"y": WEIGHTS}, "base_by_target": {"y": 100.0}})
    )
    output_path = tmp_path / "baselines.json"

    exit_code = run_baselines_main(
        [
            "--graph-json", str(graph_path),
            "--target", "y",
            "--budget", "3",
            "--toy-oracle-json", str(toy_path),
            "--methods", "influence,eap,shapley,game1,acdc",
            "--shapley-permutations", "4",
            "--acdc-taus", "2.0",
            "--acdc-target-k", "2",
            "--bruteforce-k", "2",
            "--no-connected",  # toy features only touch via the logit hub
            "--no-progress",
            "--output-json", str(output_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text())

    # Candidates default to the CLT feature nodes; the logit node is excluded.
    assert payload["candidates"] == ["a", "b", "c"]

    # Every ranked method agrees on this additive game.
    for method in ("influence", "eap", "shapley", "game1"):
        assert payload["methods"][method]["ranking"] == ["a", "b", "c"], method
        results = payload["methods"][method]["results"]
        assert results["1"]["evidence"] == ["a"]
        assert results["1"]["scores"]["faithfulness"] == pytest.approx(6.0)
        assert results["3"]["scores"]["faithfulness"] == pytest.approx(11.0)

    # Zero-intervention selectors pay zero oracle calls; Shapley pays real ones.
    assert payload["methods"]["influence"]["selection_stats"]["oracle_calls"] == 0
    assert payload["methods"]["eap"]["selection_stats"]["oracle_calls"] == 0
    assert payload["methods"]["shapley"]["selection_stats"]["oracle_calls"] > 0
    assert payload["methods"]["shapley"]["extras"]["efficiency_gap"] == pytest.approx(0.0, abs=1e-9)

    # ACDC at tau=2 prunes c and keeps {a, b}.
    acdc = payload["methods"]["acdc"]
    assert acdc["sweep"][0]["kept"] == ["a", "b"]
    assert acdc["best_by_size"]["2"]["scores"]["faithfulness"] == pytest.approx(10.0)

    # Budget-matched ACDC (--acdc-target-k 2) bisects tau to exactly k=2 and is
    # mirrored into the comparison map alongside the ranked methods.
    matched = acdc["matched_k"]
    assert matched["target_k"] == 2 and matched["achieved_k"] == 2
    assert matched["exact"] is True
    assert matched["evidence"] == ["a", "b"]
    assert matched["scores"]["faithfulness"] == pytest.approx(10.0)
    assert payload["comparison"]["faithfulness_at_k"]["acdc"]["2"] == pytest.approx(10.0)
    assert payload["params"]["acdc_target_k"] == 2

    # Brute force at k=2 matches every method's 2-prefix -> zero optimality gap.
    brute = payload["bruteforce"]["2"]
    assert brute["best_set"] == ["a", "b"]
    assert brute["best_value"] == pytest.approx(10.0)
    for method, gap in brute["optimality_gap"].items():
        assert gap == pytest.approx(0.0, abs=1e-9), method

    comparison = payload["comparison"]
    assert comparison["agreement_vs_shapley"]["influence"]["2"]["precision_at_k"] == pytest.approx(1.0)
    assert comparison["spearman"]["eap|influence"] == pytest.approx(1.0)
    assert comparison["spearman"]["eap|game1_marginal_gain"] == pytest.approx(1.0)
    # Trapezoidal AUC on k=0..3 with values 0,6,10,11 → 21.5 / 3.
    assert comparison["auc_raw_faithfulness"]["game1"] == pytest.approx(21.5 / 3.0)
    assert comparison["auc_definition"].startswith("trapezoidal")
    assert "0" in comparison["faithfulness_at_k"]["game1"]
    assert acdc["selection_stats"]["oracle_calls"] > 0
    assert payload["stats"]["evaluation_oracle_calls"] > 0
    assert payload["stats"]["bruteforce_oracle_calls"] > 0
    assert payload["stats"]["bruteforce_oracle_calls"] >= brute["evaluations"]


def test_run_baselines_marks_unrepresentable_eap_unavailable(tmp_path) -> None:
    graph_payload = {
        "nodes": [
            {
                "node_id": "a",
                "feature_type": "cross layer transcoder",
                "influence": 1.0,
            },
            {
                "node_id": "L_a",
                "feature_type": "logit",
                "clerp": 'Output " A" (p=0.8)',
                "is_target_logit": True,
            },
        ],
        "links": [{"source": "a", "target": "L_a", "weight": 1.0}],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph_payload))
    toy_path = tmp_path / "toy.json"
    toy_path.write_text(
        json.dumps({"weights_by_target": {"y": {"a": 1.0}}, "base_by_target": {"y": 0.0}})
    )
    output_path = tmp_path / "baselines.json"
    oracle_kwargs = {
        "target_token_by_label": {"y": " A", "y_foil": "1"},
        "foil_by_target": {"y": "y_foil"},
        "score_kind": "logit_gap",
    }

    assert run_baselines_main(
        [
            "--graph-json",
            str(graph_path),
            "--target",
            "y",
            "--input-id",
            "p-eap-unavailable",
            "--toy-oracle-json",
            str(toy_path),
            "--oracle-kwargs-json",
            json.dumps(oracle_kwargs),
            "--methods",
            "eap,influence",
            "--budget",
            "1",
            "--no-progress",
            "--output-json",
            str(output_path),
        ]
    ) == 0
    payload = json.loads(output_path.read_text())
    assert payload["methods"]["eap"]["status"] == "unavailable"
    assert "different objective" in payload["methods"]["eap"]["reason"]
    assert payload["methods"]["influence"]["results"]["1"]["evidence"] == ["a"]
    assert "eap" not in payload["comparison"]["faithfulness_at_k"]
    assert payload["experiment_identity"]["score_kind"] == "logit_gap"


def test_run_baselines_game1_connected_default(tmp_path) -> None:
    """game1 defaults to the connectivity constraint (same method as run_macag).

    On a graph whose features only meet at the logit hub, connected greedy
    cannot extend past its seed node — pinning both the default and the
    hub-exclusion connectivity semantics.
    """
    graph_payload = {
        "nodes": [
            {"node_id": "a", "feature_type": "cross layer transcoder", "layer": "0", "ctx_idx": 1, "influence": 6.0},
            {"node_id": "b", "feature_type": "cross layer transcoder", "layer": "1", "ctx_idx": 1, "influence": 4.0},
            {"node_id": "L", "feature_type": "logit", "clerp": 'Output " y"', "is_target_logit": True},
        ],
        "links": [
            {"source": "a", "target": "L", "weight": 6.0},
            {"source": "b", "target": "L", "weight": 4.0},
        ],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph_payload))
    toy_path = tmp_path / "toy.json"
    toy_path.write_text(
        json.dumps({"weights_by_target": {"y": WEIGHTS}, "base_by_target": {"y": 100.0}})
    )
    output_path = tmp_path / "baselines.json"

    exit_code = run_baselines_main(
        [
            "--graph-json", str(graph_path),
            "--target", "y",
            "--budget", "2",
            "--toy-oracle-json", str(toy_path),
            "--methods", "game1",
            "--no-progress",
            "--output-json", str(output_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text())

    assert payload["params"]["game1_connected"] is True
    game1 = payload["methods"]["game1"]
    assert game1["params"]["connected"] is True
    # b is only reachable from a through the logit hub, so the connected
    # greedy stops at the seed node despite budget=2.
    assert game1["ranking"] == ["a"]
    assert game1["extras"]["stopped_early"] is True
    assert set(game1["results"].keys()) == {"0", "1"}
    # Carry-forward puts the size-1 faithfulness onto budget k=2 in the comparison.
    assert payload["comparison"]["faithfulness_at_k"]["game1"]["2"] == pytest.approx(
        game1["results"]["1"]["scores"]["faithfulness"]
    )


def test_merge_baselines_deferred_shapley(tmp_path) -> None:
    """Fast pass without shapley + shapley-only sidecar == full-run comparison."""
    from macag.cli.merge_baselines import main as merge_main

    graph_payload = {
        "nodes": [
            {"node_id": "a", "feature_type": "cross layer transcoder", "layer": "0", "ctx_idx": 1, "influence": 6.0},
            {"node_id": "b", "feature_type": "cross layer transcoder", "layer": "1", "ctx_idx": 1, "influence": 4.0},
            {"node_id": "c", "feature_type": "cross layer transcoder", "layer": "2", "ctx_idx": 1, "influence": 1.0},
        ],
        "links": [],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph_payload))
    toy_path = tmp_path / "toy.json"
    toy_path.write_text(
        json.dumps({"weights_by_target": {"y": WEIGHTS}, "base_by_target": {"y": 100.0}})
    )

    common = [
        "--graph-json", str(graph_path), "--target", "y", "--budget", "3",
        "--toy-oracle-json", str(toy_path), "--no-progress",
        "--no-connected",  # edgeless toy graph: connected game1 would cap at 1 node
    ]
    fast_path = tmp_path / "macag_baselines.json"
    run_baselines_main(common + ["--methods", "influence,game1",
                                 "--output-json", str(fast_path)])
    sidecar_path = tmp_path / "macag_baselines_shapley.json"
    run_baselines_main(common + ["--methods", "shapley", "--shapley-permutations", "4",
                                 "--output-json", str(sidecar_path)])

    fast = json.loads(fast_path.read_text())
    assert "shapley" not in fast["methods"]
    assert "agreement_vs_shapley" not in fast["comparison"]

    assert merge_main(["--main", str(fast_path), "--extra", str(sidecar_path)]) == 0
    merged = json.loads(fast_path.read_text())
    assert merged["methods"]["shapley"]["ranking"] == ["a", "b", "c"]
    assert merged["comparison"]["faithfulness_at_k"]["shapley"]["3"] == pytest.approx(11.0)
    # gold-agreement recomputed: influence's 2-prefix matches gold exactly
    agreement = merged["comparison"]["agreement_vs_shapley"]
    assert agreement["influence"]["2"]["precision_at_k"] == pytest.approx(1.0)
    assert agreement["game1"]["2"]["jaccard"] == pytest.approx(1.0)
    assert merged["comparison"]["spearman"]["shapley|game1_marginal_gain"] == pytest.approx(1.0)
    assert "shapley" in merged["params"]["methods"]


def test_run_baselines_rejects_unknown_method(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        run_baselines_main(
            [
                "--graph-json", "unused.json",
                "--target", "y",
                "--budget", "2",
                "--methods", "influence,frobnicate",
                "--output-json", str(tmp_path / "out.json"),
            ]
        )


def test_method_aliases_resolve_without_renaming_legacy_ids(tmp_path) -> None:
    """eap_graph / acdc_ported resolve to eap / acdc so old JSON keys stay stable."""
    from macag.cli.run_baselines import METHOD_ALIASES, _parse_methods

    assert METHOD_ALIASES["eap_graph"] == "eap"
    assert METHOD_ALIASES["acdc_ported"] == "acdc"
    assert METHOD_ALIASES["eap_ap"] == "eap_syed"
    assert _parse_methods("influence,eap_graph,acdc_ported") == ["influence", "eap", "acdc"]
    # Dedup after aliasing
    assert _parse_methods("eap,eap_graph") == ["eap"]


def test_syed_eap_formula_on_fake_replacement_model() -> None:
    """(a_corr - a_clean) * dL/da_clean ranks the larger |Δa·grad| feature first."""
    import torch

    from macag.baselines.eap_syed import select_top_eap_syed

    class _FakeModel:
        def __init__(self) -> None:
            self._call = 0
            self.training = False

        def ensure_tokenized(self, prompt: str):
            return torch.arange(3, dtype=torch.long)

        def eval(self):
            return self

        def train(self):
            return self

        def zero_grad(self, set_to_none: bool = False):
            return None

        def get_activations(self, tokens, sparse: bool = False):
            self._call += 1
            acts = torch.zeros(1, 3, 2)
            if self._call == 1:  # clean
                acts[0, 0, 0] = 1.0
                acts[0, 0, 1] = 0.0
            else:  # corrupt
                acts[0, 0, 0] = 3.0
                acts[0, 0, 1] = 4.0
            return torch.zeros(1, 3, 5), acts

        def feature_intervention(
            self, tokens, interventions, freeze_attention=True, return_activations=False
        ):
            # L = 2*a0 + 5*a1 so grads are 2 and 5
            acc = None
            for _layer, _pos, feat, value in interventions:
                weight = 2.0 if int(feat) == 0 else 5.0
                term = weight * value
                acc = term if acc is None else acc + term
            logits = torch.zeros(1, 3, 5)
            logits = logits.clone()
            logits[0, -1, 0] = acc
            return logits, None

    result = select_top_eap_syed(
        _FakeModel(),
        prompt="clean",
        corrupted_prompt="corrupt",
        node_to_intervention={"f0": (0, 0, 0), "f1": (0, 0, 1)},
        candidates=["f0", "f1"],
        target_logit_idx=0,
        foil_logit_idx=None,
        use_absolute=True,
    )
    # Δa0=2, grad=2 → 4; Δa1=4, grad=5 → 20
    assert result.ranking == ["f1", "f0"]
    assert result.scores["f1"] == pytest.approx(20.0)
    assert result.scores["f0"] == pytest.approx(4.0)
    assert result.params.get("grad_path") == "leaf_feature_intervention"


def test_syed_eap_leaf_path_rejects_detached_logits() -> None:
    """Mirrors ReplacementModel.feature_intervention (@torch.no_grad) failure mode."""
    import torch

    from macag.baselines.eap_syed import compute_syed_eap_node_scores

    class _DetachedModel:
        training = False

        def ensure_tokenized(self, prompt: str):
            return torch.arange(2, dtype=torch.long)

        def eval(self):
            return self

        def train(self):
            return self

        def get_activations(self, tokens, sparse: bool = False):
            return torch.zeros(1, 2, 5), torch.zeros(1, 2, 1)

        def feature_intervention(
            self, tokens, interventions, freeze_attention=True, return_activations=False
        ):
            return torch.zeros(1, 2, 5), None  # no grad_fn

    with pytest.raises(RuntimeError, match="without grad_fn"):
        compute_syed_eap_node_scores(
            _DetachedModel(),
            prompt="a",
            corrupted_prompt="b",
            node_to_intervention={"f0": (0, 0, 0)},
            candidates=["f0"],
            target_logit_idx=0,
        )


def test_syed_eap_decoder_contract_single_and_cross_layer() -> None:
    import torch

    from macag.baselines.eap_syed import _decoder_grad_contract

    class _Transcoders:
        def __init__(self, W: torch.Tensor) -> None:
            self._W = W

        def _get_decoder_vectors(self, layer_id, feat_ids):
            return self._W

    class _Model:
        def __init__(self, W: torch.Tensor) -> None:
            self.transcoders = _Transcoders(W)

    # Single-layer: W·g = 1*3 + 2*4 = 11
    resid = torch.tensor([[[3.0, 4.0, 5.0]]])  # (1, 1, 3)
    single = _Model(torch.tensor([[1.0, 2.0, 0.0]]))
    assert _decoder_grad_contract(
        single, layer=0, pos=0, feat=0, resid_grads=resid
    ) == pytest.approx(11.0)

    # Cross-layer writes to layer and layer+1
    resid2 = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ]
    )
    cross = _Model(torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]))  # [1, 2, 3]
    assert _decoder_grad_contract(
        cross, layer=0, pos=0, feat=0, resid_grads=resid2
    ) == pytest.approx(2.0 * 1.0 + 3.0 * 1.0)


def test_syed_eap_squeezes_batched_residual_grads_for_nonzero_pos() -> None:
    """Regression: ReplacementModel hook grads are (batch, pos, d_model).

    The llama rematch failure indexed ``resid_grads.shape[1]`` as position while
    that axis was still the batch dim (size 1), so every node with ctx_idx>0
    looked out of range.
    """
    import torch

    from macag.baselines.eap_syed import select_top_eap_syed

    class _Cfg:
        n_layers = 1

    class _Transcoders:
        def _get_decoder_vectors(self, layer_id, feat_ids):
            # Identity write into a 2-d residual stream.
            return torch.tensor([[1.0, 0.0]])

    class _Hooks:
        def __init__(self, model: "_DecoderModel", hooks):
            self.model = model
            self.hooks = hooks

        def __enter__(self):
            self.model._active_hooks = list(self.hooks)
            return self

        def __exit__(self, *args):
            self.model._active_hooks = []
            return False

    class _DecoderModel:
        def __init__(self) -> None:
            self.cfg = _Cfg()
            self.feature_output_hook = "hook_resid_post"
            self.transcoders = _Transcoders()
            self.training = False
            self._active_hooks: list = []
            self._call = 0

        def ensure_tokenized(self, prompt: str):
            return torch.arange(3, dtype=torch.long)

        def eval(self):
            return self

        def train(self):
            return self

        def zero_grad(self, set_to_none: bool = False):
            return None

        def hooks(self, hooks):
            return _Hooks(self, hooks)

        def get_activations(self, tokens, sparse: bool = False):
            # (n_layers, n_pos, n_feat) — mirrors real ReplacementModel CLT acts.
            self._call += 1
            acts = torch.zeros(1, 3, 1)
            if self._call == 1:  # clean
                acts[0, 2, 0] = 1.0
            else:  # corrupt
                acts[0, 2, 0] = 4.0
            return torch.zeros(1, 3, 2), acts

        def __call__(self, tokens):
            # Emit a batched residual activation (1, pos, d_model) through the hook
            # so grads keep the production batch axis.
            resid = torch.zeros(1, 3, 2, requires_grad=True)
            for name, hook_fn in self._active_hooks:
                assert "hook_resid_post" in name
                resid = hook_fn(resid, None)
            # L = resid[0, pos=2, 0] so ∂L/∂resid has mass only at pos 2.
            logits = torch.zeros(1, 3, 2)
            logits = logits.clone()
            logits[0, -1, 0] = resid[0, 2, 0]
            return logits

    result = select_top_eap_syed(
        _DecoderModel(),
        prompt="clean",
        corrupted_prompt="corrupt",
        # Intentionally at pos=2 — the production crash case.
        node_to_intervention={"f_pos2": (0, 2, 0)},
        candidates=["f_pos2"],
        target_logit_idx=0,
        foil_logit_idx=None,
        use_absolute=True,
    )
    # Δa = 3, decoder·grad = 1 → score 3
    assert result.ranking == ["f_pos2"]
    assert result.scores["f_pos2"] == pytest.approx(3.0)
    assert result.params.get("grad_path") == "decoder_dot_residual_grad"
    assert result.extras.get("seq_len") == 3


def test_native_component_graph_orders_top_down() -> None:
    from macag.baselines.acdc_native import build_native_component_graph
    from macag.baselines.acdc_prune import _topdown_order

    graph = build_native_component_graph(n_layers=2, n_heads=2)
    ordered = _topdown_order(graph, ["a0.h0", "a1.h0", "m0", "m1"])
    # Highest layer first
    assert ordered[0] in ("a1.h0", "m1")
    assert ordered[-1] in ("a0.h0", "m0")


# ----------------------------------------------------------- audit regressions
def test_influence_prefers_raw_over_reversed_cumulative() -> None:
    """Exported cumulative coverage ranks weakest-first if sorted descending."""
    graph = CircuitGraph(
        nodes=["a", "b", "c"],
        node_metadata={
            # True raw influences 6,4,1 → cumulative 6/11, 10/11, 1.0
            "a": {"influence": 6 / 11, "influence_raw": 6.0},
            "b": {"influence": 10 / 11, "influence_raw": 4.0},
            "c": {"influence": 1.0, "influence_raw": 1.0},
        },
    )
    assert select_top_influence(graph, ["a", "b", "c"]).ranking == ["a", "b", "c"]


def test_influence_legacy_cumulative_sorted_ascending() -> None:
    graph = CircuitGraph(
        nodes=["a", "b", "c"],
        node_metadata={
            "a": {"influence": 6 / 11},
            "b": {"influence": 10 / 11},
            "c": {"influence": 1.0},
        },
    )
    result = select_top_influence(graph, ["a", "b", "c"])
    assert result.ranking == ["a", "b", "c"]
    assert result.extras["legacy_cumulative_inverted"] is True


def test_acdc_target_size_finds_nonmonotone_exact_k() -> None:
    """Path-dependent prune where exact k=1 exists outside a tau=0 bisection bracket."""

    class TableBackend:
        def __init__(self) -> None:
            self.table = {
                frozenset(): 0.0,
                frozenset({"a"}): 12.0,
                frozenset({"b"}): 1.0,
                frozenset({"c"}): -20.0,
                frozenset({"a", "b"}): 29.0,
                frozenset({"a", "c"}): 26.0,
                frozenset({"b", "c"}): 20.0,
                frozenset({"a", "b", "c"}): 25.0,
            }
            self.universe = {"a", "b", "c"}

        def score_all(self, target: str) -> float:
            return self.table[frozenset(self.universe)]

        def score_empty(self, target: str) -> float:
            return self.table[frozenset()]

        def score_keep_only(self, nodes: set[str], target: str) -> float:
            return self.table[frozenset(nodes)]

        def score_remove(self, nodes: set[str], target: str) -> float:
            return self.table[frozenset(self.universe - set(nodes))]

    oracle = ScoringOracle(backend=TableBackend(), cache_enabled=True)
    graph = CircuitGraph(nodes=["a", "b", "c"])
    assert set(
        acdc_prune(graph, oracle, "y", ["a", "b", "c"], tau=20, alpha=1.0, order="given").kept
    ) == {"b"}
    matched = acdc_target_size(
        graph, oracle, "y", ["a", "b", "c"], target_k=1, alpha=1.0, order="given"
    )
    assert matched.params["exact"] is True
    assert matched.params["achieved_k"] == 1
    assert len(matched.kept) == 1


def test_merge_baselines_rejects_candidate_mismatch(tmp_path) -> None:
    from macag.cli.merge_baselines import merge_payloads

    main = {
        "input_id": "p1",
        "target": "y",
        "params": {"budget": 2, "alpha": 0.5, "lambda": 0.01, "methods": []},
        "candidates": ["a", "b"],
        "methods": {},
        "comparison": {},
    }
    extra = {
        "input_id": "p1",
        "target": "y",
        "params": {"budget": 2, "alpha": 0.5, "lambda": 0.01, "methods": []},
        "candidates": ["a", "b", "c"],
        "methods": {},
        "comparison": {},
    }
    with pytest.raises(ValueError, match="candidates"):
        merge_payloads(main, extra)


def test_merge_baselines_rejects_graph_or_oracle_identity_mismatch() -> None:
    from macag.cli.merge_baselines import merge_payloads

    base = {
        "input_id": "p1",
        "target": "y",
        "params": {
            "budget": 2,
            "alpha": 0.5,
            "lambda": 0.01,
            "score_kind": "logit_gap",
            "freeze_attention": True,
            "ablation_mode": "zero",
            "methods": [],
        },
        "candidates": ["a", "b"],
        "methods": {},
        "comparison": {},
    }
    main = dict(base, experiment_identity=_identity("main"))
    extra = dict(base, experiment_identity=_identity("extra"))
    with pytest.raises(ValueError, match="graph_sha256"):
        merge_payloads(main, extra)


def test_merge_baselines_requires_explicit_legacy_identity_adoption() -> None:
    from macag.cli.merge_baselines import merge_payloads

    params = {
        "budget": 2,
        "alpha": 0.5,
        "lambda": 0.01,
        "score_kind": "logit_gap",
        "freeze_attention": True,
        "ablation_mode": "zero",
        "methods": [],
    }
    main = {
        "input_id": "p1",
        "target": "y",
        "params": dict(params),
        "candidates": ["a", "b"],
        "methods": {},
        "comparison": {},
    }
    extra = dict(main, params=dict(params), experiment_identity=_identity())
    with pytest.raises(ValueError, match="explicit adoption"):
        merge_payloads(dict(main), extra)
    adopted = merge_payloads(dict(main), extra, adopt_extra_identity=True)
    assert adopted["experiment_identity"] == _identity()


def test_kl_rescore_prefers_matched_k() -> None:
    from macag.kl_rescore import _acdc_evidence_for_budget

    entry = {
        "matched_k": {
            "evidence": ["a", "b"],
            "achieved_k": 2,
            "exact": True,
            "scores": {"faithfulness": 10.0},
        },
        "best_by_size": {
            "8": {"evidence": ["a", "b", "c", "d"], "scores": {"faithfulness": 99.0}},
            "193": {
                "evidence": [f"n{i}" for i in range(193)],
                "scores": {"faithfulness": 100.0},
            },
        },
    }
    evidence, faith, meta = _acdc_evidence_for_budget(entry, budget=8)
    assert evidence == ["a", "b"]
    assert faith == pytest.approx(10.0)
    assert meta["source"] == "matched_k"
