"""Regression tests for optimizer groups used by CLT training."""

from __future__ import annotations

import pytest
import torch

from spline_clt.kan_transcoder import KANCrossLayerTranscoder
from spline_clt.paper.config import TrainingSettings
from spline_clt.training.data import ActivationDataset
from spline_clt.training.train import (
    TrainConfig,
    _build_optimizers,
    _threshold_metrics,
    create_optimizer,
    initialize_thresholds_from_data,
)


def _linear_jumprelu_model() -> KANCrossLayerTranscoder:
    return KANCrossLayerTranscoder(
        n_layers=2,
        d_model=8,
        d_transcoder=16,
        encoder_type="linear",
        activation_function="jump_relu",
        threshold_init=0.01,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_threshold_has_dedicated_adamw_group() -> None:
    model = _linear_jumprelu_model()
    config = TrainConfig(
        encoder_type="linear",
        threshold_weight_decay=0.0,
        threshold_adam_eps=1e-15,
    )

    optimizer, optimizer_local = _build_optimizers(
        model, model, config, lr=config.learning_rate
    )

    assert optimizer_local is None
    assert len(optimizer.param_groups) == 2
    main_group, threshold_group = optimizer.param_groups
    assert main_group["weight_decay"] == config.weight_decay
    assert main_group["eps"] == 1e-8
    assert threshold_group["weight_decay"] == config.threshold_weight_decay
    assert threshold_group["eps"] == config.threshold_adam_eps
    assert threshold_group["params"] == [model.activation_function.threshold]


def test_threshold_metrics_match_effective_threshold_without_fsdp() -> None:
    model = _linear_jumprelu_model()
    expected = model.effective_threshold.detach()

    metrics = _threshold_metrics(
        model,
        distributed=False,
        device=torch.device("cpu"),
    )

    assert metrics["stats/threshold_mean"] == pytest.approx(
        expected.mean().item()
    )
    assert metrics["stats/threshold_median"] == pytest.approx(
        expected.median().item()
    )
    assert metrics["stats/threshold_max"] == pytest.approx(
        expected.max().item()
    )


def test_optimizer_rejects_threshold_lost_to_parameter_flattening() -> None:
    threshold = torch.nn.Parameter(torch.tensor(0.0))
    flattened = torch.nn.Parameter(torch.zeros(4))
    config = TrainConfig(encoder_type="linear")

    with pytest.raises(RuntimeError, match="use_orig_params=True"):
        create_optimizer(
            [flattened],
            config,
            lr=config.learning_rate,
            threshold_params=[threshold],
        )


def test_paper_training_settings_expose_threshold_optimizer_controls() -> None:
    settings = TrainingSettings(
        d_transcoder=16,
        encoder_type="linear",
        threshold_init=0.2,
        jumprelu_bandwidth=0.05,
        threshold_weight_decay=0.0,
        threshold_adam_eps=1e-12,
        threshold_init_strategy="data_quantile",
        threshold_init_target_l0=4.0,
        threshold_calibration_samples=8,
        threshold_calibration_values_per_sample=32,
        normalize_inputs=False,
        normalization_samples=64,
        weight_decay=0.02,
        adam_beta1=0.8,
        adam_beta2=0.99,
    )

    assert settings.threshold_init == 0.2
    assert settings.jumprelu_bandwidth == 0.05
    assert settings.threshold_adam_eps == 1e-12
    assert settings.threshold_init_strategy == "data_quantile"
    assert settings.threshold_init_target_l0 == 4.0
    assert settings.threshold_calibration_samples == 8
    assert settings.threshold_calibration_values_per_sample == 32
    assert settings.normalize_inputs is False
    assert settings.normalization_samples == 64
    assert settings.weight_decay == 0.02
    assert settings.adam_beta1 == 0.8
    assert settings.adam_beta2 == 0.99


def test_data_quantile_threshold_initialization_targets_layer_l0() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(24, 2, 8, 8)
    dataset = ActivationDataset(mlp_inputs=inputs, mlp_outputs=inputs.clone())
    model = _linear_jumprelu_model()

    layer_thresholds = initialize_thresholds_from_data(
        model,
        dataset,
        target_l0=4.0,
        n_sequences=24,
        values_per_sample=128,
        seed=11,
    )

    assert layer_thresholds.shape == (model.n_layers,)
    assert (layer_thresholds > 0).all()
    assert torch.allclose(
        model.effective_threshold,
        layer_thresholds[:, None].expand_as(model.effective_threshold),
    )
    with torch.no_grad():
        activations = torch.cat(
            [model.encode(sample) for sample in inputs.unbind(0)], dim=1
        )
    l0_per_layer = (activations > 0).float().sum(dim=-1).mean(dim=-1)
    assert torch.allclose(
        l0_per_layer,
        torch.full_like(l0_per_layer, 4.0),
        atol=0.75,
    )
