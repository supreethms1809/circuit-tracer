"""Tests for KL-divergence scoring and post-hoc rescoring."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from macag.kl_rescore import rescore_game1_leg
from macag.scoring import (
    CallbackInterventionScorer,
    ReplacementModelInterventionScorer,
    ScoringOracle,
    compute_kl_score,
)
from macag.utils.metrics import compute_faithfulness_metrics


def test_compute_kl_score_identical_distributions() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0])
    assert compute_kl_score(logits, logits) == pytest.approx(0.0, abs=1e-6)


def test_compute_kl_score_higher_when_closer_to_reference() -> None:
    ref = torch.tensor([0.0, 2.0, 0.0])
    close = torch.tensor([0.1, 1.9, 0.0])
    far = torch.tensor([2.0, 0.0, 0.0])
    assert compute_kl_score(ref, close) > compute_kl_score(ref, far)


class _MockLogitsModel:
    """Returns logits keyed by how many nodes are ablated."""

    def __init__(
        self,
        ref: torch.Tensor,
        empty: torch.Tensor,
        partial: torch.Tensor,
    ) -> None:
        self.ref = ref
        self.empty = empty
        self.partial = partial

    def feature_intervention(
        self,
        prompt: Any,
        interventions: list[Any],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del prompt, kwargs
        if not interventions:
            logits = self.ref
        elif len(interventions) == 1:
            logits = self.partial
        else:
            logits = self.empty
        return logits.unsqueeze(0).unsqueeze(0), None


def test_replacement_scorer_kl_divergence_modes() -> None:
    ref = torch.tensor([0.0, 3.0, 0.0])
    empty = torch.tensor([2.0, 0.0, 0.0])
    partial = torch.tensor([0.1, 2.9, 0.0])
    model = _MockLogitsModel(ref=ref, empty=empty, partial=partial)
    scorer = ReplacementModelInterventionScorer(
        model=model,
        prompt="test",
        node_to_intervention={"a": (0, 0, 1), "b": (0, 0, 2)},
        target_to_logit_idx={"y": 1},
        score_kind="kl_divergence",
    )
    oracle = ScoringOracle(backend=scorer, cache_enabled=True)

    assert oracle.all("y") == pytest.approx(0.0, abs=1e-6)
    empty = oracle.empty("y")
    keep = oracle.keep_only({"a"}, "y")
    assert keep > empty
    assert oracle.all("y") == pytest.approx(0.0, abs=1e-6)


def test_kl_faithfulness_metrics_signs() -> None:
    ref = torch.tensor([0.0, 4.0, 0.0])
    empty_logits = torch.tensor([2.0, 0.0, 0.0])
    keep_logits = torch.tensor([0.1, 3.9, 0.0])

    def logits_for(mode: str, nodes: set[str]) -> torch.Tensor:
        if mode == "all":
            return ref
        if mode == "empty":
            return empty_logits
        if mode == "keep_only":
            return keep_logits if nodes == {"a"} else empty_logits
        if mode == "remove":
            return empty_logits if nodes == {"a"} else ref
        raise AssertionError(mode)

    def score_all(target: str) -> float:
        del target
        return compute_kl_score(ref, logits_for("all", set()))

    def score_empty(target: str) -> float:
        del target
        return compute_kl_score(ref, logits_for("empty", set()))

    def score_keep_only(nodes: set[str], target: str) -> float:
        del target
        return compute_kl_score(ref, logits_for("keep_only", nodes))

    def score_remove(nodes: set[str], target: str) -> float:
        del target
        return compute_kl_score(ref, logits_for("remove", nodes))

    backend = CallbackInterventionScorer(
        score_all_fn=score_all,
        score_empty_fn=score_empty,
        score_keep_only_fn=score_keep_only,
        score_remove_fn=score_remove,
    )
    oracle = ScoringOracle(backend=backend)
    metrics = compute_faithfulness_metrics(oracle, target="y", nodes={"a"}, alpha=0.5)
    assert metrics.sufficiency > 0.0
    assert metrics.faithfulness_delta > 0.0


def test_rescore_game1_leg_preserves_logit_gap_and_adds_kl() -> None:
    backend = CallbackInterventionScorer(
        score_all_fn=lambda _t: 0.0,
        score_empty_fn=lambda _t: -4.0,
        score_keep_only_fn=lambda nodes, _t: -1.0 if nodes else -4.0,
        score_remove_fn=lambda nodes, _t: -4.0 if nodes else 0.0,
    )
    oracle = ScoringOracle(backend=backend)
    leg = {
        "params": {"alpha": 0.5},
        "evidence": {"E_star": ["a", "b"]},
        "scores": {
            "faithfulness": 2.5,
            "sufficiency": 3.0,
            "recoverable_range": 4.0,
        },
    }
    out = rescore_game1_leg(leg, oracle, target="y")
    assert out["logit_gap"]["faithfulness"] == 2.5
    assert out["kl_divergence"]["faithfulness"] == pytest.approx(3.5)
    assert out["evidence_size"] == 2


def test_merge_kl_into_outputs_embeds_game1(tmp_path) -> None:
    from macag.kl_rescore import merge_kl_into_outputs

    g1 = {
        "freeze_mode": "both",
        "frozen": {"scores": {"faithfulness": 1.0}, "evidence": {"E_star": ["a"]}},
        "unfrozen": {"scores": {"faithfulness": 2.0}, "evidence": {"E_star": ["a", "b"]}},
    }
    g1_path = tmp_path / "macag_game1.json"
    g1_path.write_text(json.dumps(g1))

    kl_payload = {
        "game1": {
            "frozen": {"kl_divergence": {"faithfulness": 0.5, "sufficiency": 0.4}},
            "unfrozen": {"kl_divergence": {"faithfulness": 0.6, "sufficiency": 0.5}},
        }
    }
    merge_kl_into_outputs(tmp_path, kl_payload)
    merged = json.loads(g1_path.read_text())
    assert merged["frozen"]["kl_faithfulness"]["faithfulness"] == 0.5
    assert merged["unfrozen"]["kl_faithfulness"]["faithfulness"] == 0.6


def test_rescore_game1_leg_custom_score_label() -> None:
    backend = CallbackInterventionScorer(
        score_all_fn=lambda _t: 0.0,
        score_empty_fn=lambda _t: -2.0,
        score_keep_only_fn=lambda nodes, _t: -1.0 if nodes else -2.0,
        score_remove_fn=lambda nodes, _t: -2.0 if nodes else 0.0,
    )
    oracle = ScoringOracle(backend=backend)
    leg = {"params": {"alpha": 0.5}, "evidence": {"E_star": ["a"]}, "scores": {}}
    out = rescore_game1_leg(leg, oracle, target="y", score_label="altfoil")
    assert "altfoil" in out and "kl_divergence" not in out
    assert out["altfoil"]["faithfulness"] == pytest.approx(1.5)


def test_altfoil_spec_substitutes_foil_without_mutating_kwargs() -> None:
    from macag.kl_rescore import altfoil_spec

    spec = altfoil_spec(" David")
    stored = {
        "score_kind": "answer_span",
        "target_token_by_label": {"y": " Mary", "y_foil": " John"},
        "freeze_attention": True,
    }
    original = json.loads(json.dumps(stored))
    assert spec.transform is not None
    transformed = spec.transform(dict(stored))
    assert transformed["target_token_by_label"] == {"y": " Mary", "y_foil": " David"}
    assert transformed["score_kind"] == "answer_span"  # stored score_kind preserved
    assert stored == original  # input not mutated
    assert spec.output_name == "macag_altfoil_faithfulness.json"
    assert spec.embed_key == "altfoil_faithfulness"

    with pytest.raises(ValueError, match="y_foil"):
        spec.transform({"target_token_by_label": {"y": " Mary"}})


def test_merge_with_altfoil_spec_embeds_under_altfoil_key(tmp_path) -> None:
    from macag.kl_rescore import altfoil_spec, merge_kl_into_outputs

    g1 = {"scores": {"faithfulness": 1.0}, "evidence": {"E_star": ["a"]}}
    g1_path = tmp_path / "macag_game1.json"
    g1_path.write_text(json.dumps(g1))

    payload = {"game1": {"single": {"altfoil": {"faithfulness": 0.7}}}}
    merge_kl_into_outputs(tmp_path, payload, spec=altfoil_spec(" David"))
    merged = json.loads(g1_path.read_text())
    assert merged["altfoil_faithfulness"]["faithfulness"] == 0.7
    assert "kl_faithfulness" not in merged
