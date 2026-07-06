"""Tests for mean/corrupted ablation values (per-node 4-tuple intervention values)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from macag.factories.replacement_model import compute_mean_ablation_values
from macag.scoring import ReplacementModelInterventionScorer

N_LAYERS, SEQ, D = 2, 3, 4


class _ActivationsMockModel:
    """get_activations returns a fixed cache per prompt; records interventions."""

    def __init__(self, cache_by_prompt: dict[str, torch.Tensor]) -> None:
        self.cache_by_prompt = cache_by_prompt
        self.received_interventions: list[list[Any]] = []

    def ensure_tokenized(self, prompt: str) -> str:
        return prompt  # identity: caches are keyed by prompt text

    def get_activations(self, inputs: str, sparse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        del sparse
        cache = self.cache_by_prompt[inputs]
        return torch.zeros(1, 1, 5), cache

    def feature_intervention(
        self, prompt: Any, interventions: list[Any], **kwargs: Any
    ) -> tuple[torch.Tensor, None]:
        del prompt, kwargs
        self.received_interventions.append(list(interventions))
        logits = torch.tensor([0.0, float(len(interventions)), 0.0])
        return logits.unsqueeze(0).unsqueeze(0), None


def _clean_cache() -> torch.Tensor:
    cache = torch.zeros(N_LAYERS, SEQ, D)
    cache[0, :, 1] = torch.tensor([1.0, 2.0, 3.0])  # mean 2.0
    cache[1, :, 2] = torch.tensor([4.0, 0.0, 2.0])  # mean 2.0
    return cache


NODES = {"n0": (0, 1, 1), "n1": (1, 2, 2)}


def test_prompt_positions_mean_values() -> None:
    model = _ActivationsMockModel({"clean": _clean_cache()})
    values = compute_mean_ablation_values(model, NODES, mode="prompt_positions", prompt="clean")
    assert values["n0"] == pytest.approx(2.0)
    assert values["n1"] == pytest.approx(2.0)


def test_corrupted_prompt_position_matched_values() -> None:
    corrupted = _clean_cache() * 10.0
    model = _ActivationsMockModel({"clean": _clean_cache(), "corr": corrupted})
    values = compute_mean_ablation_values(
        model, NODES, mode="corrupted_prompt", prompt="clean", corrupted_prompt="corr"
    )
    # position-matched: cache[layer, pos, feat] of the corrupted pass
    assert values["n0"] == pytest.approx(float(corrupted[0, 1, 1]))
    assert values["n1"] == pytest.approx(float(corrupted[1, 2, 2]))


def test_corrupted_length_mismatch_falls_back_to_mean() -> None:
    short = torch.zeros(N_LAYERS, SEQ - 1, D)
    model = _ActivationsMockModel({"clean": _clean_cache(), "corr": short})
    values = compute_mean_ablation_values(
        model, NODES, mode="corrupted_prompt", prompt="clean", corrupted_prompt="corr"
    )
    assert values["n0"] == pytest.approx(2.0)  # mean fallback, not corrupted read


def test_mode_and_prompt_validation() -> None:
    model = _ActivationsMockModel({"clean": _clean_cache()})
    with pytest.raises(ValueError, match="mode"):
        compute_mean_ablation_values(model, NODES, mode="resample", prompt="clean")
    with pytest.raises(ValueError, match="clean prompt"):
        compute_mean_ablation_values(model, NODES, mode="prompt_positions")
    with pytest.raises(ValueError, match="corrupted_prompt"):
        compute_mean_ablation_values(model, NODES, mode="corrupted_prompt", prompt="clean")


def test_scorer_sends_per_node_values() -> None:
    """4-tuple specs (from mean-ablation rewriting) reach the model unchanged."""
    model = _ActivationsMockModel({"clean": _clean_cache()})
    values = compute_mean_ablation_values(model, NODES, mode="prompt_positions", prompt="clean")
    node_to_intervention = {
        node: (spec[0], spec[1], spec[2], values[node]) for node, spec in NODES.items()
    }
    scorer = ReplacementModelInterventionScorer(
        model=model,
        prompt="clean",
        node_to_intervention=node_to_intervention,
        target_to_logit_idx={"y": 1},
        score_kind="logit",
    )
    scorer.score_empty("y")
    sent = {tuple(iv) for iv in model.received_interventions[-1]}
    assert sent == {(0, 1, 1, 2.0), (1, 2, 2, 2.0)}
