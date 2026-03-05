"""Tests for KAN encoder module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from spline_clt.kan_encoder import KANEncoder


@pytest.fixture
def encoder():
    """Create a small KAN encoder for testing."""
    return KANEncoder(d_model=64, n_features=128, grid_size=3, spline_order=3)


@pytest.fixture
def encoder_gpt2():
    """Create a GPT-2 scale KAN encoder."""
    return KANEncoder(d_model=768, n_features=4096, grid_size=5, spline_order=3)


class TestKANEncoderShapes:
    def test_forward_2d(self, encoder):
        """Test forward pass with (batch, d_model) input."""
        x = torch.randn(32, 64)
        out = encoder(x)
        assert out.shape == (32, 128)

    def test_forward_1d(self, encoder):
        """Test forward pass with single sample."""
        x = torch.randn(1, 64)
        out = encoder(x)
        assert out.shape == (1, 128)

    def test_basis_expansion_dim(self, encoder):
        """Test basis expansion dimensionality property."""
        # d_model * (grid_size + spline_order) = 64 * (3 + 3) = 384
        assert encoder.basis_expansion_dim == 64 * (3 + 3)


class TestKANEncoderGradients:
    def test_gradient_flow(self, encoder):
        """Test that gradients flow through the KAN encoder."""
        x = torch.randn(8, 64, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (8, 64)
        assert not torch.all(x.grad == 0)

    def test_parameter_gradients(self, encoder):
        """Test that all parameters receive gradients."""
        x = torch.randn(8, 64)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


class TestKANEncoderJacobian:
    def test_encoder_vectors_shape(self, encoder):
        """Test that encoder vectors have correct shape."""
        x = torch.randn(4, 64)
        out = encoder(x)

        # Create a sparse mask with some active features
        active = (out > 0).float().to_sparse()
        if active._nnz() > 0:
            enc_vecs = encoder.get_encoder_vectors(x, active)
            assert enc_vecs.shape[1] == 64  # d_model
            assert enc_vecs.shape[0] == active._nnz()

    def test_encoder_vectors_empty(self, encoder):
        """Test encoder vectors with no active features."""
        x = torch.randn(4, 64)
        # Create an all-zero sparse mask
        active = torch.zeros(4, 128).to_sparse()
        enc_vecs = encoder.get_encoder_vectors(x, active)
        assert enc_vecs.shape == (0, 64)

    def test_jacobian_correctness(self, encoder):
        """Verify Jacobian via finite differences."""
        x = torch.randn(1, 64)
        out = encoder(x)

        # Pick a feature that's nonzero
        feat_idx = out.abs().argmax(dim=-1).item()

        # Compute Jacobian row via our method
        active_mask = torch.zeros(1, 128)
        active_mask[0, feat_idx] = 1.0
        active_mask = active_mask.to_sparse()
        jac_row = encoder.get_encoder_vectors(x, active_mask)  # (1, 64)

        # Finite difference verification
        eps = 1e-4
        fd_jac = torch.zeros(64)
        for d in range(64):
            x_plus = x.clone()
            x_plus[0, d] += eps
            x_minus = x.clone()
            x_minus[0, d] -= eps
            fd_jac[d] = (encoder(x_plus)[0, feat_idx] - encoder(x_minus)[0, feat_idx]) / (2 * eps)

        assert torch.allclose(jac_row.squeeze(0), fd_jac, atol=1e-2), (
            f"Jacobian mismatch: max diff = {(jac_row.squeeze(0) - fd_jac).abs().max():.4e}"
        )

    def test_encoder_vectors_fast(self, encoder):
        """Test fast Jacobian computation matches standard."""
        x = torch.randn(4, 64)
        out = encoder(x)
        active = (out > out.median()).float().to_sparse()

        if active._nnz() > 0:
            enc_slow = encoder.get_encoder_vectors(x, active)
            enc_fast = encoder.get_encoder_vectors_fast(x, active)
            assert torch.allclose(enc_slow, enc_fast, atol=1e-4), (
                f"Fast vs slow mismatch: max diff = {(enc_slow - enc_fast).abs().max():.4e}"
            )


class TestKANEncoderParameters:
    def test_parameter_count(self, encoder):
        """Test parameter count is reasonable."""
        n_params = encoder.n_parameters
        # KANLinear params: base_weight(128, 64) + spline_weight(128, 64, 6) + spline_scaler(128, 64)
        # = 8192 + 49152 + 8192 = 65536
        assert n_params > 0
        assert n_params > 64 * 128  # More than linear encoder

    def test_parameter_count_vs_linear(self, encoder):
        """KAN encoder should have more parameters than linear for same dims."""
        linear_params = 64 * 128  # d_model * n_features
        assert encoder.n_parameters > linear_params

    def test_grid_update(self, encoder):
        """Test that grid update runs without error."""
        x = torch.randn(100, 64)
        encoder.update_grid(x)
        # Should still work after update
        out = encoder(x)
        assert out.shape == (100, 128)


class TestKANEncoderGPT2Scale:
    def test_forward_gpt2(self, encoder_gpt2):
        """Test forward pass at GPT-2 scale."""
        x = torch.randn(16, 768)
        out = encoder_gpt2(x)
        assert out.shape == (16, 4096)

    def test_forward_backward_gpt2(self, encoder_gpt2):
        """Test forward + backward at GPT-2 scale completes."""
        x = torch.randn(16, 768, requires_grad=True)
        out = encoder_gpt2(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
