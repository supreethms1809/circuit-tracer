"""Training-loss and scheduler invariants for both CLT encoder arms."""

from __future__ import annotations

import torch

from spline_clt.kan_transcoder import KANCrossLayerTranscoder
from spline_clt.training.loss import (
    compute_decoder_norms,
    reconstruction_loss,
    sparsity_loss,
)
from spline_clt.training.train import TrainConfig, create_optimizer, get_lr


def test_nmse_zero_baseline_is_one_for_all_layer_weightings() -> None:
    target = torch.randn(3, 7, 5)
    prediction = torch.zeros_like(target)

    assert torch.allclose(
        reconstruction_loss(prediction, target, per_layer=False),
        torch.tensor(1.0),
    )
    for beta in (0.0, 0.5, 1.0):
        assert torch.allclose(
            reconstruction_loss(
                prediction,
                target,
                per_layer=True,
                layer_energy_beta=beta,
            ),
            torch.tensor(1.0),
        )


def test_per_layer_beta_one_matches_global_nmse() -> None:
    target = torch.randn(4, 9, 6)
    prediction = torch.randn_like(target)

    global_loss = reconstruction_loss(prediction, target, per_layer=False)
    beta_one_loss = reconstruction_loss(
        prediction,
        target,
        per_layer=True,
        layer_energy_beta=1.0,
    )
    assert torch.allclose(global_loss, beta_one_loss, rtol=1e-6, atol=1e-7)


def test_sparsity_mean_normalization_divides_by_layer_count() -> None:
    activations = torch.rand(3, 11, 7)
    decoder_norms = [torch.rand(7) for _ in range(3)]

    summed = sparsity_loss(
        activations,
        decoder_norms,
        lambda_=0.2,
        per_layer_mean=False,
    )
    meaned = sparsity_loss(
        activations,
        decoder_norms,
        lambda_=0.2,
        per_layer_mean=True,
    )
    assert torch.allclose(meaned * activations.shape[0], summed)


def test_sparsity_gradient_and_threshold_optimizer_raise_gate() -> None:
    model = KANCrossLayerTranscoder(
        n_layers=1,
        d_model=4,
        d_transcoder=8,
        encoder_type="linear",
        activation_function="jump_relu",
        threshold_init=0.01,
        jumprelu_bandwidth=0.001,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.no_grad():
        model.encoders[0].W_enc.zero_()
        model.b_enc.fill_(0.01025)
    config = TrainConfig(
        n_layers=1,
        d_model=4,
        d_transcoder=8,
        encoder_type="linear",
    )
    threshold = model.activation_function.threshold
    optimizer = create_optimizer(
        model.parameters(),
        config,
        lr=1e-4,
        threshold_params=[threshold],
    )
    before = model.effective_threshold.detach().clone()

    activations = model.encode(torch.zeros(1, 16, 4))
    loss = sparsity_loss(
        activations,
        compute_decoder_norms(model),
        lambda_=0.2,
        per_layer_mean=True,
    )
    loss.backward()

    assert threshold.grad is not None
    assert (threshold.grad < 0).all()
    optimizer.step()
    assert (model.effective_threshold > before).all()


def test_warmup_cosine_schedule_boundaries() -> None:
    config = TrainConfig(learning_rate=1e-3, warmup_steps=10, total_steps=100)

    assert get_lr(0, config) == 0.0
    assert get_lr(10, config) == config.learning_rate
    assert 0.0 < get_lr(50, config) < config.learning_rate
    assert abs(get_lr(100, config)) < 1e-12


def test_lambda_sparsity_constant_by_default() -> None:
    from spline_clt.training.train import get_lambda_sparsity, sparsity_lambda_at_step

    config = TrainConfig(lambda_sparsity=1e-5, total_steps=100)
    assert get_lambda_sparsity(0, config) == 1e-5
    assert get_lambda_sparsity(50, config) == 1e-5
    assert get_lambda_sparsity(100, config) == 1e-5


def test_lambda_sparsity_warmup_hold_cosine_decay() -> None:
    from spline_clt.training.train import get_lambda_sparsity

    peak = 1e-5
    floor = 3e-6
    config = TrainConfig(
        lambda_sparsity=peak,
        lambda_sparsity_final=floor,
        sparsity_warmup_steps=10,
        sparsity_decay_start=70,
        total_steps=100,
    )
    assert get_lambda_sparsity(0, config) == 0.0
    assert abs(get_lambda_sparsity(5, config) - 0.5 * peak) < 1e-15
    assert get_lambda_sparsity(10, config) == peak
    assert get_lambda_sparsity(50, config) == peak
    assert get_lambda_sparsity(70, config) == peak
    mid = get_lambda_sparsity(85, config)
    assert floor < mid < peak
    assert abs(get_lambda_sparsity(100, config) - floor) < 1e-15


def test_lambda_sparsity_decay_disabled_when_start_zero() -> None:
    from spline_clt.training.train import get_lambda_sparsity

    config = TrainConfig(
        lambda_sparsity=1e-5,
        lambda_sparsity_final=1e-6,
        sparsity_warmup_steps=10,
        sparsity_decay_start=0,
        total_steps=100,
    )
    assert get_lambda_sparsity(10, config) == 1e-5
    assert get_lambda_sparsity(99, config) == 1e-5
