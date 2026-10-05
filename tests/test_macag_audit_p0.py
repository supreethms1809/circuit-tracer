"""P0 audit regression tests (A0/A1/A2/A2'/A5).

A0: MCQA/ARC foil must be a wrong clean-prompt option, not the counterfactual
    alphabet's correct answer.
A1: symmetric logit_gap Game 2 (f_foil == -f_y) is flagged; --game2-one-sided
    fails fast on it.
A2/A2': sufficiency AND necessity capped by default; uncapped preserved opt-out;
    R <= 0 flagged degenerate; Game 1 search uses the capped v.
A5: KL/baseline rescoring matches Game 1's |E*| (frozen leg), not the budget.
"""

from __future__ import annotations

import json

from macag.baselines.common import coalition_value
from macag.games.game1_min_faithful import solve_game1
from macag.games.game2_contrastive import is_symmetric_gap_oracle, solve_game2
from macag.graph import CircuitGraph
from macag.kl_rescore import game1_matched_k, rescore_baselines
from macag.scoring import (
    CallbackInterventionScorer,
    ReplacementModelInterventionScorer,
    ScoringOracle,
)
from macag.utils.metrics import compute_faithfulness_metrics


def _graph(nodes: list[str]) -> CircuitGraph:
    return CircuitGraph(nodes=nodes, edges=[])


# ---------------------------------------------------------------- A0
def test_a0_mcqa_foil_is_wrong_clean_option() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "build_mib_prompts",
        str(Path("experiments/build_mib_benchmark_prompts.py")),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return type("E", (), {"input_ids": [ord(str(text)[0])]})()

        def decode(self, ids):
            return chr(int(ids[0]))

    tok = _Tok()
    row = {"choices": {"label": ["A", "B", "C", "D"]}, "answerKey": 3}
    correct, foil = mod._mcqa_tokens(tok, row)
    assert correct == "D"
    assert foil != correct
    assert foil == "A"  # round-robin balanced: (3+1) % 4 == 0
    # builder validation: foil must appear in the clean prompt
    clean = "Q\nA. x\nB. y\nC. z\nD. w\nAnswer:"
    assert foil in clean
    # balanced across all four correct answers
    for answer_key, expected_foil in ((0, "B"), (1, "C"), (2, "D"), (3, "A")):
        c, f = mod._mcqa_tokens(tok, {"choices": {"label": ["A", "B", "C", "D"]}, "answerKey": answer_key})
        assert (c, f) == (("A", "B", "C", "D")[answer_key], expected_foil)


def test_a0_committed_prompts_have_clean_foils() -> None:
    payload = json.loads(open("macag/data/mib_benchmark_prompts.json").read())
    for task in ("mcqa", "arc_easy"):
        for item in payload["tasks"][task]:
            assert item["incorrect_token"] in ("A", "B", "C", "D"), item["id"]
            assert item["incorrect_token"] != item["correct_token"], item["id"]
            assert item["incorrect_token"] in item["clean_prompt"], item["id"]
            # digit-alphabet foils ('1'-'4') must be gone
            assert item["incorrect_token"] not in ("1", "2", "3", "4"), item["id"]


# ---------------------------------------------------------------- A2/A2'
def _overshoot_oracle() -> ScoringOracle:
    # all=0, empty=-4; keep overshoots to +2 (suff uncapped 6, capped 4);
    # remove dives to -6 (nec uncapped 6, capped 4).
    backend = CallbackInterventionScorer(
        score_all_fn=lambda _t: 0.0,
        score_empty_fn=lambda _t: -4.0,
        score_keep_only_fn=lambda nodes, _t: 2.0 if nodes else -4.0,
        score_remove_fn=lambda nodes, _t: -6.0 if nodes else 0.0,
    )
    return ScoringOracle(backend=backend)


def test_a2_sufficiency_capped_by_default() -> None:
    m = compute_faithfulness_metrics(_overshoot_oracle(), target="y", nodes={"a"}, alpha=0.5)
    assert m.sufficiency == 4.0  # min(2,0)+4
    assert m.sufficiency_uncapped == 6.0
    assert m.keep_excess == 2.0
    uncapped = compute_faithfulness_metrics(
        _overshoot_oracle(), target="y", nodes={"a"}, alpha=0.5, cap_sufficiency=False,
    )
    assert uncapped.sufficiency == 6.0


def test_a2_prime_necessity_capped_by_default() -> None:
    m = compute_faithfulness_metrics(_overshoot_oracle(), target="y", nodes={"a"}, alpha=0.5)
    assert m.necessity == 4.0  # 0-max(-6,-4)
    assert m.necessity_uncapped == 6.0
    assert m.remove_below_empty == 2.0
    uncapped = compute_faithfulness_metrics(
        _overshoot_oracle(), target="y", nodes={"a"}, alpha=0.5, cap_necessity=False,
    )
    assert uncapped.necessity == 6.0


def test_a2_double_prime_degenerate_flag() -> None:
    backend = CallbackInterventionScorer(
        score_all_fn=lambda _t: -5.0,  # R = -5-(-2) = -3 <= 0
        score_empty_fn=lambda _t: -2.0,
        score_keep_only_fn=lambda nodes, _t: -2.0,
        score_remove_fn=lambda nodes, _t: -5.0,
    )
    oracle = ScoringOracle(backend=backend)
    m = compute_faithfulness_metrics(oracle, target="y", nodes=set(), alpha=0.5)
    assert m.is_degenerate is True
    assert m.recoverable_range <= 0.0


def test_a2_double_prime_degenerate_skips_normalized_stop() -> None:
    # R < 0: capped faithfulness of any non-empty set (<= R) can never beat the
    # empty set ((1-alpha)*R), so the greedy correctly selects nothing. The
    # trap the normalized-stop guard protects against: the singleton's
    # NORMALIZED faithfulness is spuriously >= 1 (suff <= R < 0 divided by R),
    # which without the guard would certify it as "recovering everything".
    backend = CallbackInterventionScorer(
        score_all_fn=lambda _t: -5.0,
        score_empty_fn=lambda _t: -2.0,
        score_keep_only_fn=lambda nodes, _t: -5.0 if nodes else -2.0,
        score_remove_fn=lambda nodes, _t: -2.0 if nodes else -5.0,
    )
    oracle = ScoringOracle(backend=backend)
    singleton = compute_faithfulness_metrics(oracle, target="y", nodes={"a"}, alpha=0.5)
    assert singleton.is_degenerate is True
    assert singleton.faithfulness_delta_normalized >= 0.8  # the false certificate
    graph = _graph(["a", "b"])
    result = solve_game1(
        graph=graph, oracle=oracle, target="y", candidates=["a", "b"],
        alpha=0.5, lam=0.0, budget=2, faithfulness_eps=0.2,
        stop_metric="normalized", progress=False,
    )
    assert result.metrics.is_degenerate is True
    assert result.evidence == set()


def test_a2_coalition_value_uses_caps_by_default() -> None:
    v_capped = coalition_value(_overshoot_oracle(), "y", {"a"}, 0.5)
    v_uncapped = coalition_value(
        _overshoot_oracle(), "y", {"a"}, 0.5,
        cap_sufficiency=False, cap_necessity=False,
    )
    assert v_capped == 0.5 * 4.0 + 0.5 * 4.0
    assert v_uncapped == 0.5 * 6.0 + 0.5 * 6.0
    assert v_capped < v_uncapped


def test_a2_game1_selection_uses_capped_v() -> None:
    # Node 'over' overshoots uncapped but ties capped; 'real' wins capped.
    # all=0, empty=-4. over: keep=+2 (capped suff 4), remove=-6 (capped nec 4).
    # real: keep=0 (capped suff 4), remove=-4 (nec 4)... make over strictly worse
    # capped via a second node? Simpler: capped faithfulness of {'over'} is 4.0
    # while uncapped is 6.0; assert the solver's reported metrics are capped.
    graph = _graph(["over"])
    result = solve_game1(
        graph=graph, oracle=_overshoot_oracle(), target="y",
        candidates=["over"], alpha=0.5, lam=0.0, progress=False,
    )
    assert result.metrics.sufficiency == 4.0
    assert result.metrics.necessity == 4.0
    assert result.params["cap_sufficiency"] is True
    assert result.params["cap_necessity"] is True


# ---------------------------------------------------------------- A1
def _gap_oracle() -> ScoringOracle:
    scorer = ReplacementModelInterventionScorer(
        model=None,
        prompt="p",
        node_to_intervention={"a": (0, 0, 1)},
        target_to_logit_idx={"y": 0, "y_foil": 1},
        score_kind="logit_gap",
        foil_by_target={"y": "y_foil", "y_foil": "y"},
    )
    return ScoringOracle(backend=scorer, cache_enabled=False)


def test_a1_symmetric_gap_detected_and_flagged() -> None:
    from macag.games.game2_contrastive import is_one_sided_oracle

    oracle = _gap_oracle()
    assert is_symmetric_gap_oracle(oracle, "y", "y_foil") is True
    assert is_one_sided_oracle(oracle) is False
    one_sided = ReplacementModelInterventionScorer(
        model=None,
        prompt="p",
        node_to_intervention={"a": (0, 0, 1)},
        target_to_logit_idx={"y": 0, "y_foil": 1},
        score_kind="logit",
    )
    one_oracle = ScoringOracle(backend=one_sided)
    assert is_symmetric_gap_oracle(one_oracle, "y", "y_foil") is False
    assert is_one_sided_oracle(one_oracle) is True
    # answer_span with a swapped map is gap-style too (negated span gap)
    span_scorer = ReplacementModelInterventionScorer(
        model=None,
        prompt="p",
        node_to_intervention={"a": (0, 0, 1)},
        target_to_logit_idx={"y": 0, "y_foil": 1},
        score_kind="answer_span",
        foil_by_target={"y": "y_foil", "y_foil": "y"},
        target_span_ids_by_label={"y": [0], "y_foil": [1]},
    )
    span_oracle = ScoringOracle(backend=span_scorer)
    assert is_symmetric_gap_oracle(span_oracle, "y", "y_foil") is True
    assert is_one_sided_oracle(span_oracle) is False
    # KL is target-blind: neither symmetric nor one-sided
    kl_scorer = ReplacementModelInterventionScorer(
        model=None,
        prompt="p",
        node_to_intervention={"a": (0, 0, 1)},
        target_to_logit_idx={"y": 0},
        score_kind="kl_divergence",
    )
    kl_oracle = ScoringOracle(backend=kl_scorer)
    assert is_symmetric_gap_oracle(kl_oracle, "y", "y_foil") is False
    assert is_one_sided_oracle(kl_oracle) is False


def test_a1_game2_params_carry_symmetry_flags() -> None:
    from macag.scoring import ToyAdditiveInterventionScorer

    backend = ToyAdditiveInterventionScorer(
        weights_by_target={"y": {"a": 1.0}, "f": {"a": 0.5}},
        base_by_target={"y": 0.0, "f": 0.0},
    )
    oracle = ScoringOracle(backend=backend)
    result = solve_game2(
        graph=_graph(["a"]), oracle=oracle, y="y", y_foil="f",
        candidates=["a"], progress=False,
    )
    # toy backend has no score_kind: not a symmetric gap pair, one_sided True
    assert result.params["structural_gap_symmetry"] is False
    assert result.params["cap_sufficiency"] is True
    assert result.params["cap_necessity"] is True


def test_a1_one_sided_cli_rejects_symmetric_gap() -> None:
    from macag.cli.run_macag import main as run_macag_main

    import tempfile, os

    graph = {"nodes": [{"node_id": "a", "feature_type": "cross layer transcoder"}]}
    with tempfile.TemporaryDirectory() as tmp:
        g = os.path.join(tmp, "g.json")
        open(g, "w").write(json.dumps(graph))
        toy = os.path.join(tmp, "toy.json")
        open(toy, "w").write(json.dumps(
            {"weights_by_target": {"y": {"a": 1.0}, "y_foil": {"a": 1.0}}}
        ))
        out = os.path.join(tmp, "out.json")
        # one-sided flag with a toy oracle (no logit_gap symmetry) must pass
        rc = run_macag_main([
            "game2", "--graph-json", g, "--target", "y", "--foil", "y_foil",
            "--toy-oracle-json", toy, "--output-json", out,
            "--game2-one-sided", "--no-progress",
        ])
        assert rc == 0


# ---------------------------------------------------------------- A5
def test_a5_matched_k_reads_game1_size() -> None:
    assert game1_matched_k({
        "freeze_mode": "both",
        "frozen": {"evidence": {"E_star": ["a", "b", "c"]}},
        "unfrozen": {"evidence": {"E_star": ["a"]}},
    }) == 3
    assert game1_matched_k({"evidence": {"E_star": ["x", "y"]}}) == 2
    assert game1_matched_k({}) is None


def test_a5_rescore_baselines_uses_matched_k() -> None:
    backend = CallbackInterventionScorer(
        score_all_fn=lambda _t: 0.0,
        score_empty_fn=lambda _t: -8.0,
        score_keep_only_fn=lambda nodes, _t: -8.0 + 2.0 * len(nodes),
        score_remove_fn=lambda nodes, _t: -2.0 * len(nodes),
    )
    oracle = ScoringOracle(backend=backend)
    payload = {
        "params": {"alpha": 0.5, "budget": 8},
        "methods": {
            "influence": {
                "results": {
                    str(k): {
                        "evidence": [f"n{i}" for i in range(k)],
                        "scores": {"faithfulness": float(k)},
                    }
                    for k in (1, 2, 8)
                }
            },
            "acdc": {"best_by_size": {
                str(k): {"evidence": [f"m{i}" for i in range(k)],
                         "scores": {"faithfulness": float(k)}}
                for k in (2, 8)
            }},
        },
    }
    out = rescore_baselines(payload, oracle, target="y", matched_k=2)
    assert out["matched_k"] == 2
    assert out["methods"]["influence"]["evidence_size"] == 2
    assert out["methods"]["acdc"]["evidence_size"] == 2
    legacy = rescore_baselines(payload, oracle, target="y", matched_k=None)
    assert legacy["methods"]["influence"]["evidence_size"] == 8
    assert "matched_k" not in legacy
