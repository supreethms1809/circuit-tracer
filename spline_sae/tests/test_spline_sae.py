"""CPU smoke test for SplineSAE forward + losses."""

import math

import torch
import torch.nn.functional as F

from spline_sae.loss import compute_sae_losses
from spline_sae.model import JumpReLU, SplineSAE


def test_linear_sae_forward_shapes() -> None:
    m = SplineSAE(d_model=32, d_sae=64, encoder_type="linear", activation="jumprelu")
    x = torch.randn(8, 16, 32)
    y_hat, a, pre = m(x)
    assert y_hat.shape == x.shape
    assert a.shape == (8, 16, 64)
    assert pre.shape == a.shape


def test_kan_sae_gap_metrics() -> None:
    m = SplineSAE(
        d_model=32,
        d_sae=48,
        encoder_type="kan",
        activation="jumprelu",
        grid_size=3,
        spline_order=2,
    )
    x = torch.randn(4, 8, 32)
    loss, metrics = compute_sae_losses(m, x, lambda_sparsity=1e-3, lambda_nl_gap=0.1)
    assert torch.isfinite(loss)
    assert "stats/l0" in metrics
    assert "stats/spline_contribution_frac" in metrics


def test_topk_sae() -> None:
    m = SplineSAE(d_model=32, d_sae=64, encoder_type="linear", activation="topk", topk_k=5)
    x = torch.randn(2, 10, 32)
    _y, a, _ = m(x)
    assert int((a > 0).sum(dim=-1).max().item()) <= 5


def test_base_jump_nested_and_metrics() -> None:
    m = SplineSAE(
        d_model=32,
        d_sae=48,
        encoder_type="kan",
        activation="base_jump",
        grid_size=3,
        spline_order=2,
        jumprelu_bandwidth=0.01,
    )
    # Zero spline path → BaseJump ≡ JumpReLU(base + b_enc)
    with torch.no_grad():
        kl = m.encoder.kan_linear
        kl.spline_weight.zero_()
        if hasattr(kl, "spline_scaler"):
            kl.spline_scaler.zero_()
    x = torch.randn(4, 8, 32)
    a_full, pre = m.encode(x)
    a_base, score = m.encode_base_only(x)
    assert torch.allclose(a_full, a_base, atol=1e-5)
    assert torch.allclose(pre, score, atol=1e-5)

    loss, metrics = compute_sae_losses(m, x, lambda_sparsity=1e-3)
    assert torch.isfinite(loss)
    assert "stats/act_frac" in metrics
    assert metrics["stats/recon_gap"] == 0.0 or abs(metrics["stats/recon_gap"]) < 1e-4


def test_jumprelu_ste_threshold_grad() -> None:
    jr = JumpReLU(d_sae=4, threshold_init=0.5, bandwidth=0.1)
    score = torch.tensor([[0.45, 0.55, 0.0, 1.0]], requires_grad=True)
    out = jr(score)
    out.sum().backward()
    assert jr.log_threshold.grad is not None
    assert torch.isfinite(jr.log_threshold.grad).all()


def test_forward_split_sums_to_forward() -> None:
    m = SplineSAE(
        d_model=16,
        d_sae=24,
        encoder_type="kan",
        activation="base_jump",
        grid_size=3,
        spline_order=2,
    )
    x = torch.randn(2, 5, 16)
    base, spline = m.encoder.forward_split(x)
    full = m.encoder(x)
    assert torch.allclose(base + spline, full, atol=1e-5)


def test_mlp_decoder_shapes() -> None:
    m = SplineSAE(
        d_model=32,
        d_sae=64,
        encoder_type="linear",
        activation="jumprelu",
        decoder_type="mlp",
        decoder_hidden=48,
    )
    x = torch.randn(2, 10, 32)
    y_hat, a, _ = m(x)
    assert y_hat.shape == x.shape
    loss, metrics = compute_sae_losses(m, x, lambda_sparsity=1e-3)
    assert torch.isfinite(loss)
    assert metrics["stats/l0"] >= 0


def test_linear_mlp_nested_zero_mlp() -> None:
    m = SplineSAE(
        d_model=32,
        d_sae=64,
        encoder_type="linear",
        activation="jumprelu",
        decoder_type="linear_mlp",
        decoder_hidden=32,
    )
    with torch.no_grad():
        assert m.mlp_dec is not None
        m.mlp_dec.fc1.weight.zero_()
        m.mlp_dec.fc1.bias.zero_()
        m.mlp_dec.fc2.weight.zero_()
        m.mlp_dec.fc2.bias.zero_()
    a = torch.randn(4, 64).relu()
    y_lin = F.linear(a, m.W_dec.T, m.b_dec)
    y = m.decode(a)
    assert torch.allclose(y, y_lin, atol=1e-5)


def test_kan_decoder_shapes() -> None:
    m = SplineSAE(
        d_model=32,
        d_sae=64,
        encoder_type="kan",
        activation="base_jump",
        decoder_type="kan",
        decoder_bot=16,
        grid_size=3,
        spline_order=2,
    )
    x = torch.randn(2, 8, 32)
    y_hat, a, _ = m(x)
    assert y_hat.shape == x.shape
    loss, _ = compute_sae_losses(m, x, lambda_sparsity=1e-3)
    assert torch.isfinite(loss)


def test_transcoder_target_y() -> None:
    """Encode x, reconstruct a different y (attention-transcoder style)."""
    m = SplineSAE(d_model=32, d_sae=64, encoder_type="linear", activation="jumprelu")
    x = torch.randn(4, 8, 32)
    y = torch.randn(4, 8, 32) * 3.0
    _, m_y = compute_sae_losses(m, x, y=y, lambda_sparsity=0.0, compute_metrics=True)
    _, m_x = compute_sae_losses(m, x, y=x, lambda_sparsity=0.0, compute_metrics=True)
    assert m_y["loss/nmse"] != m_x["loss/nmse"]
    assert math.isfinite(m_y["loss/nmse"])

