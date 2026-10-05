"""P1/P2 audit follow-up regression tests (published baselines untouched).

P1-1: harness --faithfulness-eps/--stop-metric parity (defaults preserve legacy).
P1-2: per-method selection_pool_size reporting (game1 = post-prefilter count).
P1-6: singleton + random-k floors are opt-in (not in defaults), deterministic.
P1-5: attention-mediation verdict dead-band margin (0.0 = legacy strict-zero).
P1-7: KL/alt-foil rescore outputs record cap provenance.
P2:   derive clears the KL ref cache; gap same-first-subtoken fails fast;
      forward counts surfaced in cache stats; ACDC tau-unit docs (doc-only).
"""

from __future__ import annotations

import json

import pytest

from macag.baselines.floors import select_random, select_top_singleton
from macag.cli.run_baselines import DEFAULT_METHODS, KNOWN_METHODS
from macag.factories.replacement_model import _validate_gap_indices
from macag.graph import CircuitGraph
from macag.kl_rescore import (
    merge_kl_into_outputs,
    rescore_baselines,
    rescore_game1_leg,
    rescore_game2,
)
from macag.scoring import (
    CallbackInterventionScorer,
    ReplacementModelInterventionScorer,
    ScoringOracle,
    derive_oracle_with_freeze,
)
from macag.utils.attention_mediation import compute_attention_mediation_diagnostic
from macag.utils.metrics import FaithfulnessMetrics


def _graph(nodes: list[str]) -> CircuitGraph:
    return CircuitGraph(nodes=nodes, edges=[])


def _metrics_with_range(range_value: float) -> FaithfulnessMetrics:
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


def _diagnostic(rf: float, ru: float, margin: float = 0.0):
    return compute_attention_mediation_diagnostic(
        graph=_graph(["a", "b"]),
        frozen_metrics=_metrics_with_range(rf),
        unfrozen_metrics=_metrics_with_range(ru),
        frozen_evidence=set(),
        unfrozen_evidence=set(),
        margin=margin,
    )


def _toy_oracle(weights: dict[str, float]):
    from macag.scoring import ToyAdditiveInterventionScorer

    return ScoringOracle(
        backend=ToyAdditiveInterventionScorer(weights_by_target={"y": weights})
    )


# ---------------------------------------------------------------- P1-6 floors
def test_p16_floors_opt_in_not_default() -> None:
    assert "singleton" in KNOWN_METHODS
    assert "random" in KNOWN_METHODS
    defaults = [m.strip() for m in DEFAULT_METHODS.split(",")]
    assert "singleton" not in defaults
    assert "random" not in defaults


def test_p16_singleton_ranks_by_singleton_value() -> None:
    oracle = _toy_oracle({"a": 1.0, "b": 5.0, "c": 3.0})
    result = select_top_singleton(oracle, "y", ["a", "b", "c"])
    assert [str(n) for n in result.ranking] == ["b", "c", "a"]
    assert result.scores is not None
    assert result.scores["b"] == pytest.approx(5.0)
    assert result.params["cap_sufficiency"] is True


def test_p16_random_is_seeded_and_covers_pool() -> None:
    pool = ["a", "b", "c", "d", "e"]
    first = select_random(pool, seed=0)
    again = select_random(pool, seed=0)
    other = select_random(pool, seed=1)
    assert [str(n) for n in first.ranking] == [str(n) for n in again.ranking]
    assert sorted(str(n) for n in first.ranking) == sorted(pool)
    assert [str(n) for n in first.ranking] != [str(n) for n in other.ranking]
    assert first.scores is None
    assert first.params == {"seed": 0}


def test_p16_floors_reject_empty_pool() -> None:
    oracle = _toy_oracle({})
    with pytest.raises(ValueError, match="non-empty"):
        select_top_singleton(oracle, "y", [])
    with pytest.raises(ValueError, match="non-empty"):
        select_random([], seed=0)


# ---------------------------------------------------------------- P1-5 margin
def test_p15_legacy_verdicts_at_zero_margin() -> None:
    assert _diagnostic(-2.0, 1.0).verdict == "attention_mediated"
    assert _diagnostic(2.0, 1.0).verdict == "feature_mediated"
    assert _diagnostic(-2.0, -1.0).verdict == "indeterminate"
    assert _diagnostic(0.0, 3.0).verdict == "feature_mediated"
    flipped = _diagnostic(2.0, -1.0)
    assert flipped.verdict == "indeterminate" and flipped.reverse_flip is True


def test_p15_exact_double_zero_is_indeterminate() -> None:
    diag = _diagnostic(0.0, 0.0)
    assert diag.verdict == "indeterminate"
    assert diag.frozen_zone == "band" and diag.unfrozen_zone == "band"


def test_p15_margin_deadens_near_zero_signals() -> None:
    # Strict-zero calls this a flip; with a margin it is noise -> indeterminate.
    assert _diagnostic(-0.001, 0.001).verdict == "attention_mediated"
    margined = _diagnostic(-0.001, 0.001, margin=0.01)
    assert margined.verdict == "indeterminate"
    # Robust signals survive the same margin.
    assert _diagnostic(-2.0, 1.0, margin=0.01).verdict == "attention_mediated"
    assert _diagnostic(2.0, 1.0, margin=0.01).verdict == "feature_mediated"
    # Margin + zones are recorded for downstream confidence bands.
    assert margined.margin == pytest.approx(0.01)
    assert margined.to_dict()["frozen_zone"] == "band"


def test_p15_negative_margin_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _diagnostic(1.0, 1.0, margin=-0.5)


def test_p15_dual_solver_threads_verdict_margin() -> None:
    from macag.games.game1_attention_probe import solve_game1_dual
    from macag.scoring import ToyAdditiveInterventionScorer

    graph = _graph(["a", "b"])
    frozen = ScoringOracle(
        backend=ToyAdditiveInterventionScorer(weights_by_target={"y": {"a": 1.0}})
    )
    unfrozen = ScoringOracle(
        backend=ToyAdditiveInterventionScorer(weights_by_target={"y": {"a": 1.0}})
    )
    result = solve_game1_dual(
        graph=graph,
        frozen_oracle=frozen,
        unfrozen_oracle=unfrozen,
        target="y",
        verdict_margin=0.25,
    )
    assert result.params["verdict_margin"] == pytest.approx(0.25)
    assert result.diagnostic.margin == pytest.approx(0.25)


# ---------------------------------------------------------------- P1-7 caps
def _callback_oracle() -> ScoringOracle:
    backend = CallbackInterventionScorer(
        score_all_fn=lambda _t: 0.0,
        score_empty_fn=lambda _t: -8.0,
        score_keep_only_fn=lambda nodes, _t: -8.0 + 2.0 * len(nodes),
        score_remove_fn=lambda nodes, _t: -2.0 * len(nodes),
    )
    return ScoringOracle(backend=backend)


def test_p17_rescore_blocks_record_cap_flags() -> None:
    oracle = _callback_oracle()
    leg = {
        "params": {"alpha": 0.5},
        "evidence": {"E_star": ["a"]},
        "scores": {},
    }
    out = rescore_game1_leg(leg, oracle, target="y")
    assert out["cap_sufficiency"] is True and out["cap_necessity"] is True
    out_nc = rescore_game1_leg(
        leg, oracle, target="y", cap_sufficiency=False, cap_necessity=False
    )
    assert out_nc["cap_sufficiency"] is False and out_nc["cap_necessity"] is False

    g2 = {
        "params": {"alpha": 0.5},
        "evidence": {"E_y": ["a"], "E_foil": ["b"]},
        "scores": {},
    }
    assert rescore_game2(g2, oracle, target="y")["cap_sufficiency"] is True

    payload = {
        "params": {"alpha": 0.5, "budget": 2},
        "methods": {
            "influence": {
                "results": {
                    "1": {"evidence": ["a"], "scores": {"faithfulness": 1.0}},
                    "2": {"evidence": ["a", "b"], "scores": {"faithfulness": 2.0}},
                }
            }
        },
    }
    bl = rescore_baselines(payload, oracle, target="y")
    assert bl["cap_sufficiency"] is True and bl["cap_necessity"] is True
    bl_nc = rescore_baselines(
        payload, oracle, target="y", cap_sufficiency=False, cap_necessity=False
    )
    assert bl_nc["cap_sufficiency"] is False and bl_nc["cap_necessity"] is False


def test_p17_merge_embeds_cap_siblings(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "macag_game1.json").write_text(
        json.dumps(
            {
                "freeze_mode": "both",
                "frozen": {"evidence": {"E_star": ["a"]}},
                "unfrozen": {"evidence": {"E_star": ["a"]}},
            }
        )
    )
    (run_dir / "macag_baselines.json").write_text(
        json.dumps({"methods": {"influence": {"ranking": ["a"]}}})
    )
    kl_payload = {
        "cap_sufficiency": True,
        "cap_necessity": False,
        "game1": {
            "frozen": {
                "kl_divergence": {"faithfulness": 1.0},
                "cap_sufficiency": True,
                "cap_necessity": False,
            },
            "unfrozen": {"kl_divergence": {"faithfulness": 0.5}},
        },
        "baselines": {
            "cap_sufficiency": True,
            "cap_necessity": False,
            "methods": {"influence": {"kl_divergence": {"faithfulness": 1.0}}},
        },
    }
    merge_kl_into_outputs(run_dir, kl_payload)
    g1 = json.loads((run_dir / "macag_game1.json").read_text())
    # Block-level flags win; missing block flags fall back to the sidecar top level.
    assert g1["frozen"]["kl_faithfulness_caps"] == {
        "cap_sufficiency": True,
        "cap_necessity": False,
    }
    assert g1["unfrozen"]["kl_faithfulness_caps"] == {
        "cap_sufficiency": True,
        "cap_necessity": False,
    }
    bl = json.loads((run_dir / "macag_baselines.json").read_text())
    assert bl["methods"]["influence"]["kl_faithfulness_caps"] == {
        "cap_sufficiency": True,
        "cap_necessity": False,
    }


# ---------------------------------------------------------------- P2
def test_p2_gap_same_token_fails_fast() -> None:
    with pytest.raises(ValueError, match="identically zero"):
        _validate_gap_indices(
            {"y": 5, "y_foil": 5}, {"y": "y_foil"}, None, "logit_gap"
        )
    # Distinct indices pass; non-gap kinds skip validation entirely.
    _validate_gap_indices({"y": 5, "y_foil": 7}, {"y": "y_foil"}, None, "logit_gap")
    _validate_gap_indices({"y": 5, "y_foil": 5}, {"y": "y_foil"}, None, "answer_span")


def test_p2_scorer_gap_same_token_fails_fast() -> None:
    scorer = ReplacementModelInterventionScorer(
        model=None,
        prompt="x",
        node_to_intervention={},
        target_to_logit_idx={"y": 5, "y_foil": 5},
        score_kind="logit_gap",
        foil_by_target={"y": "y_foil"},
    )
    with pytest.raises(ValueError, match="identically zero"):
        scorer._target_indices("y")


def test_p2_derive_clears_kl_ref_cache() -> None:
    backend = ReplacementModelInterventionScorer(
        model=None,
        prompt="x",
        node_to_intervention={"a": (0, 0, 0)},
        target_to_logit_idx={"y": 1},
        score_kind="logit",
        freeze_attention=True,
    )
    backend._ref_logits = object()  # simulate a cached frozen-convention ref
    oracle = ScoringOracle(backend=backend)
    derived = derive_oracle_with_freeze(oracle, False)
    assert derived.backend is not backend
    assert derived.backend._ref_logits is None
    assert derived.backend.intervention_universe() == {"a"}


def test_p2_forwards_surfaced_and_reset() -> None:
    oracle = _callback_oracle()
    oracle.all("y")
    stats = oracle.cache_stats()
    assert stats["forwards"] == stats["oracle_calls"]  # no forward_count: fallback

    backend = ReplacementModelInterventionScorer(
        model=None,
        prompt="x",
        node_to_intervention={},
        target_to_logit_idx={"y": 1},
        score_kind="logit",
    )
    backend.forward_count = 7
    oracle2 = ScoringOracle(backend=backend)
    assert oracle2.cache_stats()["forwards"] == 7
    oracle2.reset_stats()
    assert backend.forward_count == 0
    assert oracle2.cache_stats()["forwards"] == 0


def test_p2_acdc_tau_units_documented() -> None:
    import importlib

    # NB: import_module (not `import ... as`), because macag.baselines
    # re-exports the acdc_prune FUNCTION under the same attribute name.
    mod = importlib.import_module("macag.baselines.acdc_prune")

    assert "raw v units" in mod.__doc__
    assert "raw v units" in mod.acdc_prune.__doc__
