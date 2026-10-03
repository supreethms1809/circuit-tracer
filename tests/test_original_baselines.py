"""Tests for true AutoCircuit-backed original baselines."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("auto_circuit")

from macag.baselines.original.acdc_edge import run_acdc_edge
from macag.baselines.original.eap_edge import run_eap_edge
from macag.baselines.original.tl_compat import enable_autocircuit_tl3_compat


@pytest.fixture(scope="module")
def tiny_tl_model():
    enable_autocircuit_tl3_compat()
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        n_ctx=32,
        d_head=8,
        n_heads=4,
        d_mlp=64,
        act_fn="gelu",
        tokenizer_name="gpt2",
        device="cpu",
    )
    model = HookedTransformer(cfg)
    return model


@pytest.fixture(scope="module")
def tiny_gqa_tl_model():
    """Minimal GQA model (n_heads > n_key_value_heads) — mirrors Gemma/Llama."""
    enable_autocircuit_tl3_compat()
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        n_ctx=32,
        d_head=8,
        n_heads=4,
        n_key_value_heads=2,
        d_mlp=64,
        act_fn="gelu",
        tokenizer_name="gpt2",
        device="cpu",
    )
    return HookedTransformer(cfg)


def test_eap_edge_true_autocircuit(tiny_tl_model) -> None:
    clean = "Hello world"
    corrupt = "Hello there"
    clean_tok = tiny_tl_model.to_tokens(clean)
    corrupt_tok = tiny_tl_model.to_tokens(corrupt)
    L = min(int(clean_tok.shape[-1]), int(corrupt_tok.shape[-1]))
    target_idx = int(clean_tok[0, L - 1].item())
    foil_idx = int(corrupt_tok[0, L - 1].item())
    result = run_eap_edge(
        tiny_tl_model,
        clean_prompt=clean,
        corrupted_prompt=corrupt,
        target_token="unused",
        foil_token="unused",
        k=8,
        target_idx=target_idx,
        foil_idx=foil_idx,
    )
    assert result["status"] == "ok"
    assert result["size"] == 8
    assert len(result["edges_kept"]) == 8
    assert "->" in result["edges_kept"][0]
    assert result["params"]["implementation"].startswith("auto_circuit")
    assert result["scores"]["kl"] is not None
    assert result["sweep"]


def test_eap_edge_native_has_sweep_not_budget(tiny_tl_model) -> None:
    clean = "Hello world"
    corrupt = "Hello there"
    clean_tok = tiny_tl_model.to_tokens(clean)
    corrupt_tok = tiny_tl_model.to_tokens(corrupt)
    L = min(int(clean_tok.shape[-1]), int(corrupt_tok.shape[-1]))
    payload = run_eap_edge(
        tiny_tl_model,
        clean_prompt=clean,
        corrupted_prompt=corrupt,
        target_token="unused",
        foil_token="unused",
        k=None,
        target_idx=int(clean_tok[0, L - 1].item()),
        foil_idx=int(corrupt_tok[0, L - 1].item()),
    )
    assert payload["status"] == "ok"
    assert payload["params"]["budget_capped"] is False
    assert payload["params"]["mode"] == "native_ranking"
    assert payload["size"] == 0
    assert payload["sweep"]
    assert payload["ranking"]


def test_acdc_edge_true_autocircuit(tiny_tl_model) -> None:
    clean = "Hello world"
    corrupt = "Hello there"
    clean_tok = tiny_tl_model.to_tokens(clean)
    corrupt_tok = tiny_tl_model.to_tokens(corrupt)
    L = min(int(clean_tok.shape[-1]), int(corrupt_tok.shape[-1]))
    target_idx = int(clean_tok[0, L - 1].item())
    foil_idx = int(corrupt_tok[0, L - 1].item())
    payload = run_acdc_edge(
        tiny_tl_model,
        clean_prompt=clean,
        corrupted_prompt=corrupt,
        target_token="unused",
        foil_token="unused",
        taus=[0.01, 0.1],
        metric="kl",
        target_k=8,
        target_idx=target_idx,
        foil_idx=foil_idx,
    )
    assert payload["status"] == "ok"
    assert payload["params"]["implementation"].startswith("auto_circuit")
    matched = payload["matched_k"]
    assert matched["status"] == "ok"
    assert matched["achieved_k"] <= 8
    assert matched["budget_capped"] is True


def test_eap_edge_gqa_does_not_indexerror(tiny_gqa_tl_model) -> None:
    clean = "Hello world"
    corrupt = "Hello there"
    clean_tok = tiny_gqa_tl_model.to_tokens(clean)
    corrupt_tok = tiny_gqa_tl_model.to_tokens(corrupt)
    L = min(int(clean_tok.shape[-1]), int(corrupt_tok.shape[-1]))
    target_idx = int(clean_tok[0, L - 1].item())
    foil_idx = int(corrupt_tok[0, L - 1].item())
    result = run_eap_edge(
        tiny_gqa_tl_model,
        clean_prompt=clean,
        corrupted_prompt=corrupt,
        target_token="unused",
        foil_token="unused",
        k=4,
        target_idx=target_idx,
        foil_idx=foil_idx,
    )
    assert result["status"] == "ok"
    assert result["size"] == 4
