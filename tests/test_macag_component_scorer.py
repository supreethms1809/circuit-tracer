"""Tests for the native-component (head/MLP) scorer + InterpBench gold parsing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from macag.scoring import ScoringOracle
from macag.scoring_components import (
    HookedComponentInterventionScorer,
    component_universe,
    load_gold_component_nodes,
    parse_component,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CIRCUIT_JSON = (
    REPO_ROOT
    / "external"
    / "MIB-circuit-track"
    / "circuits"
    / "EAP-IG-inputs_patching_edge"
    / "ioi_interpbench"
    / "importances.json"
)


class _HookedStub:
    """Records the hook names of every run; logits scale with ablation count."""

    def __init__(self, n_layers: int = 2, n_heads: int = 2, vocab: int = 4) -> None:
        self.cfg = SimpleNamespace(n_layers=n_layers, n_heads=n_heads, use_attn_result=True)
        self.vocab = vocab
        self.runs: list[list[str]] = []

    def to_tokens(self, prompt: str) -> torch.Tensor:
        del prompt
        return torch.tensor([[0, 1, 2]])

    def run_with_hooks(self, tokens: torch.Tensor, fwd_hooks: list[tuple[str, Any]]) -> torch.Tensor:
        self.runs.append([name for name, _fn in fwd_hooks])
        # target logit (idx 1) decays with the number of hooked components
        logits = torch.zeros(1, tokens.shape[1], self.vocab)
        logits[0, -1, 1] = 4.0 - float(len(fwd_hooks))
        logits[0, -1, 2] = 1.0
        return logits


def _scorer(stub: _HookedStub | None = None) -> tuple[HookedComponentInterventionScorer, _HookedStub]:
    stub = stub or _HookedStub()
    scorer = HookedComponentInterventionScorer(
        model=stub,
        prompt="test",
        target_to_logit_idx={"y": 1, "y_foil": 2},
        score_kind="logit_gap",
        foil_by_target={"y": "y_foil", "y_foil": "y"},
    )
    return scorer, stub


def test_component_universe_and_parse() -> None:
    universe = component_universe(2, 2)
    assert universe == sorted(["a0.h0", "a0.h1", "a1.h0", "a1.h1", "m0", "m1"])
    assert parse_component("a3.h1") == ("head", 3, 1)
    assert parse_component("m2") == ("mlp", 2, None)
    with pytest.raises(ValueError, match="Unrecognized"):
        parse_component("logits")


def test_hook_spec_construction() -> None:
    scorer, stub = _scorer()
    scorer.score_remove({"a1.h1", "a1.h0", "m0"}, "y")
    hook_names = stub.runs[-1]
    assert hook_names == ["blocks.1.attn.hook_result", "blocks.0.hook_mlp_out"]


def test_head_hook_zeroes_only_named_heads() -> None:
    scorer, _stub = _scorer()
    hooks = scorer._hooks_for({"a0.h1"})
    assert hooks[0][0] == "blocks.0.attn.hook_result"
    value = torch.ones(1, 3, 2, 5)  # [batch, pos, head, d]
    out = hooks[0][1](value)
    assert torch.all(out[:, :, 1, :] == 0.0)
    assert torch.all(out[:, :, 0, :] == 1.0)


def test_keep_remove_complementarity_and_universe() -> None:
    scorer, stub = _scorer()
    oracle = ScoringOracle(backend=scorer, cache_enabled=True)
    all_score = oracle.all("y")
    empty = oracle.empty("y")
    assert all_score > empty  # ablating everything hurts the target
    # keep_only(universe) == all-behavior (no hooks); remove(universe) == empty
    assert oracle.keep_only(set(component_universe(2, 2)), "y") == pytest.approx(all_score)
    assert oracle.remove(set(component_universe(2, 2)), "y") == pytest.approx(empty)

    fp_before = scorer.universe_fingerprint()
    scorer.restrict_universe({"a0.h0", "m1"})
    assert scorer.universe_fingerprint() != fp_before
    with pytest.raises(ValueError, match="empty component universe"):
        scorer.restrict_universe({"not_a_node"})


def test_requires_attn_result() -> None:
    stub = _HookedStub()
    stub.cfg.use_attn_result = False
    with pytest.raises(ValueError, match="use_attn_result"):
        HookedComponentInterventionScorer(
            model=stub, prompt="p", target_to_logit_idx={"y": 1}, score_kind="logit"
        )


def test_load_gold_component_nodes_from_node_flags() -> None:
    payload = {
        "nodes": {
            "input": {"in_graph": True},
            "m0": {"in_graph": True},
            "a1.h1": {"in_graph": True},
            "a0.h0": {"in_graph": False},
            "logits": {"in_graph": True},
        },
        "edges": {},
    }
    assert load_gold_component_nodes(payload) == {"m0", "a1.h1"}


def test_load_gold_component_nodes_edge_fallback() -> None:
    payload = {
        "nodes": {"m0": {"in_graph": False}, "a1.h1": {"in_graph": False}},
        "edges": {
            "input->m0": {"in_graph": True},
            "m0->a1.h1<v>": {"in_graph": True},
            "a0.h0->logits": {"in_graph": False},
        },
    }
    assert load_gold_component_nodes(payload) == {"m0", "a1.h1"}


@pytest.mark.skipif(not LOCAL_CIRCUIT_JSON.is_file(), reason="MIB repo not vendored")
def test_gold_parser_on_local_interpbench_example() -> None:
    payload = json.loads(LOCAL_CIRCUIT_JSON.read_text())
    assert payload["cfg"]["n_layers"] == 6 and payload["cfg"]["n_heads"] == 4
    universe = set(component_universe(6, 4))
    gold = load_gold_component_nodes(payload)
    # every parsed gold node must be an ablatable component of this model
    assert gold <= universe
