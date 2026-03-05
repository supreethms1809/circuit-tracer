from __future__ import annotations

from types import SimpleNamespace

import torch

from circuit_tracer.replacement_model import (
    replacement_model_nnsight as nnsight_rm,
)
from circuit_tracer.replacement_model import (
    replacement_model_transformerlens as tl_rm,
)


def test_transformerlens_intervention_buffers_use_ensure_tokenized_length(monkeypatch) -> None:
    captured: dict[str, tuple[int, ...]] = {}
    real_zeros = tl_rm.torch.zeros

    def capture_zeros(shape, *args, **kwargs):
        if "shape" not in captured:
            captured["shape"] = tuple(shape) if isinstance(shape, (list, tuple)) else (shape, *args)
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(tl_rm.torch, "zeros", capture_zeros)

    fake_model = SimpleNamespace(
        cfg=SimpleNamespace(
            n_layers=3,
            d_model=7,
            dtype=torch.float32,
            device=torch.device("cpu"),
            output_logits_soft_cap=0.0,
        ),
        ensure_tokenized=lambda _: torch.tensor([101, 11, 12, 13, 14]),
        _get_activation_caching_hooks=lambda **_: ([], []),
        feature_output_hook="hook_mlp_out",
    )

    tl_rm.TransformerLensReplacementModel._get_feature_intervention_hooks(
        fake_model,
        inputs="plain prompt without explicit bos",
        interventions=[],
        freeze_attention=False,
        return_activations=False,
    )

    assert captured["shape"] == (3, 5, 7)


def test_nnsight_intervention_buffers_use_ensure_tokenized_length(monkeypatch) -> None:
    captured: dict[str, tuple[int, ...]] = {}
    real_zeros = nnsight_rm.torch.zeros

    def capture_zeros(shape, *args, **kwargs):
        if "shape" not in captured:
            captured["shape"] = tuple(shape) if isinstance(shape, (list, tuple)) else (shape, *args)
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(nnsight_rm.torch, "zeros", capture_zeros)
    monkeypatch.setattr(nnsight_rm, "save", lambda value: value)

    fake_model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=2, d_model=5),
        dtype=torch.float32,
        device=torch.device("cpu"),
        ensure_tokenized=lambda _: torch.tensor([101, 21, 22, 23]),
        output=SimpleNamespace(logits=torch.tensor([0.0])),
    )

    result = nnsight_rm.NNSightReplacementModel._perform_feature_intervention(
        fake_model,
        inputs="plain prompt without explicit bos",
        interventions=[],
        activation_matrix=torch.empty(2, 1, 1),
        original_activations=None,
        activation_barrier=None,
        direct_effects_barrier=None,
        constrained_layers=range(0),
    )

    assert captured["shape"] == (2, 4, 5)
    assert torch.equal(result, torch.tensor([0.0]))
