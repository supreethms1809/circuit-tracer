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

from macag.baselines.acdc_prune import acdc_prune, acdc_target_size, acdc_tau_sweep
from macag.baselines.bruteforce import best_subset_bruteforce
from macag.baselines.common import (
    coalition_value,
    jaccard,
    precision_at_k,
    spearman_rank_correlation,
)
from macag.baselines.eap import compute_eap_node_scores, select_top_eap
from macag.baselines.influence import select_top_influence
from macag.baselines.shapley_select import estimate_banzhaf, estimate_shapley
from macag.cli.run_baselines import main as run_baselines_main
from macag.games.game1_min_faithful import solve_game1
from macag.graph import CircuitGraph
from macag.scoring import (
    CallbackInterventionScorer,
    ScoringOracle,
    ToyAdditiveInterventionScorer,
)

WEIGHTS = {"a": 6.0, "b": 4.0, "c": 1.0}


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
    with pytest.raises(ValueError, match="foil_match"):
        compute_eap_node_scores(payload, foil_match="no such clerp")

    unweighted = {
        "nodes": payload["nodes"],
        "links": [{"source": "f1", "target": "L_t"}],
    }
    with pytest.raises(ValueError, match="weight"):
        compute_eap_node_scores(unweighted)


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


def test_banzhaf_exact_on_additive_game() -> None:
    oracle = _additive_oracle()
    estimate = estimate_banzhaf(oracle, "y", ["a", "b", "c"], alpha=0.5, samples=4, seed=0)
    for node, weight in WEIGHTS.items():
        assert estimate.values[node] == pytest.approx(weight)
    assert estimate.estimator == "banzhaf"


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


def test_acdc_target_size_unreachable_returns_nearest_by_value() -> None:
    """Synergy game only realizes sizes {3, 2, 0}; k=1 is a plateau gap.

    Nearest achievable sizes are 0 and 2 (distance 1 each); the value tie-break
    must pick {a, b} (v=10) over the empty set (v=0) and flag exact=False.
    """
    oracle = _synergy_oracle()
    graph = CircuitGraph(nodes=["a", "b", "c"])
    result = acdc_target_size(graph, oracle, "y", ["a", "b", "c"], target_k=1, alpha=1.0)
    assert set(result.kept) == {"a", "b"}
    assert result.params["achieved_k"] == 2
    assert result.params["exact"] is False
    assert result.params["bisection_iters"] >= 1


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
    assert comparison["auc_raw_faithfulness"]["game1"] == pytest.approx((6.0 + 10.0 + 11.0) / 3.0)


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
    assert list(game1["results"].keys()) == ["1"]


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
