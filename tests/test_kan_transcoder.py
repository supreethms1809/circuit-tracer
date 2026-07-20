"""Tests for Spline Cross-Layer Transcoder."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from spline_clt.kan_transcoder import KANCrossLayerTranscoder, load_spline_clt


@pytest.fixture
def small_model():
    """Create a small Spline-CLT for testing."""
    return KANCrossLayerTranscoder(
        n_layers=3,
        d_transcoder=32,
        d_model=16,
        grid_size=3,
        spline_order=3,
        activation_function="relu",
        device=torch.device("cpu"),
    )


@pytest.fixture
def jumprelu_model():
    """Create a Spline-CLT with JumpReLU activation."""
    return KANCrossLayerTranscoder(
        n_layers=3,
        d_transcoder=32,
        d_model=16,
        grid_size=3,
        spline_order=3,
        activation_function="jump_relu",
        device=torch.device("cpu"),
    )


class TestKANTranscoderShapes:
    def test_encode_layer(self, small_model):
        """Test single-layer encoding produces correct shapes."""
        x = torch.randn(8, 16)  # (n_pos, d_model)
        out = small_model.encode_layer(x, layer_id=0)
        assert out.shape == (8, 32)  # (n_pos, d_transcoder)

    def test_encode_all_layers(self, small_model):
        """Test encoding all layers at once."""
        x = torch.randn(3, 8, 16)  # (n_layers, n_pos, d_model)
        out = small_model.encode(x)
        assert out.shape == (3, 8, 32)  # (n_layers, n_pos, d_transcoder)

    def test_encode_sparse(self, small_model):
        """Test sparse encoding returns correct types."""
        x = torch.randn(3, 8, 16)
        sparse_feats, enc_vecs = small_model.encode_sparse(x)
        assert sparse_feats.is_sparse
        assert sparse_feats.shape == (3, 8, 32)
        assert enc_vecs.shape[1] == 16  # d_model

    def test_decode_shape(self, small_model):
        """Test that decoding produces correct output shape."""
        x = torch.randn(3, 8, 16)
        features = small_model.encode(x).to_sparse()
        recon = small_model.decode(features)
        assert recon.shape == (3, 8, 16)  # (n_layers, n_pos, d_model)

    def test_forward_shape(self, small_model):
        """Test full forward pass shape."""
        x = torch.randn(3, 8, 16)
        _, y_hat, dec_norms = small_model.forward(x)
        assert y_hat.shape == (3, 8, 16)
        assert len(dec_norms) == 3


class TestCrossLayerDecoding:
    def test_cross_layer_decoder_shapes(self, small_model):
        """Test decoder weight shapes for cross-layer writing."""
        # Layer 0 writes to layers 0, 1, 2 → (d_transcoder, 3, d_model)
        assert small_model.W_dec[0].shape == (32, 3, 16)
        # Layer 1 writes to layers 1, 2 → (d_transcoder, 2, d_model)
        assert small_model.W_dec[1].shape == (32, 2, 16)
        # Layer 2 writes to layer 2 only → (d_transcoder, 1, d_model)
        assert small_model.W_dec[2].shape == (32, 1, 16)

    def test_cross_layer_reconstruction(self, small_model):
        """Test that features from early layers affect later layers' reconstruction."""
        x = torch.randn(3, 4, 16)

        # Encode only layer 0 (set other layers to zero input)
        x_zeroed = x.clone()
        x_zeroed[1:] = 0

        features_full = small_model.encode(x).to_sparse()
        features_zeroed = small_model.encode(x_zeroed).to_sparse()

        recon_full = small_model.decode(features_full)
        recon_zeroed = small_model.decode(features_zeroed)

        # Layer 0 features should still contribute to later layers
        # (via cross-layer decoder), so reconstruction should differ
        layer2_diff = (recon_full[2] - recon_zeroed[2]).norm()
        assert layer2_diff > 0, "Cross-layer decoding not working"


class TestActivationFunction:
    def test_relu_activation(self, small_model):
        """Test ReLU activation produces non-negative outputs."""
        x = torch.randn(3, 8, 16)
        out = small_model.encode(x)
        assert (out >= 0).all()

    def test_jumprelu_activation(self, jumprelu_model):
        """Test JumpReLU activation produces sparse outputs."""
        x = torch.randn(3, 8, 16)
        out = jumprelu_model.encode(x)
        assert (out >= 0).all()
        # JumpReLU should produce sparser outputs than ReLU
        density = (out > 0).float().mean()
        assert density < 1.0  # Not everything is active

    def test_effective_threshold_is_positive(self, jumprelu_model):
        """Effective JumpReLU gates stay strictly positive in log-parametrization."""
        jumprelu_model.activation_function.threshold.data.fill_(-10.0)
        eff = jumprelu_model.effective_threshold
        assert eff.shape == (jumprelu_model.n_layers, jumprelu_model.d_transcoder)
        assert (eff > 0).all()

    def test_activations_nonnegative_under_extreme_log_threshold(self, jumprelu_model):
        """Very negative log-thresholds must not admit negative feature activations."""
        jumprelu_model.activation_function.threshold.data.fill_(-8.0)
        x = torch.randn(3, 8, 16)
        out = jumprelu_model.encode(x)
        assert (out >= 0).all()

    def test_encode_without_activation(self, small_model):
        """Test encoding without activation function."""
        x = torch.randn(8, 16)
        pre_act = small_model.encode_layer(x, 0, apply_activation_function=False)
        # Pre-activations can be negative
        assert (pre_act < 0).any()


class TestAttributionComponents:
    def test_compute_attribution_components(self, small_model):
        """Test that attribution components have correct format."""
        x = torch.randn(3, 8, 16)
        result = small_model.compute_attribution_components(x)

        assert "activation_matrix" in result
        assert "reconstruction" in result
        assert "encoder_vecs" in result
        assert "decoder_vecs" in result
        assert "encoder_to_decoder_map" in result
        assert "decoder_locations" in result

        assert result["activation_matrix"].is_sparse
        assert result["reconstruction"].shape == (3, 8, 16)
        assert result["encoder_vecs"].shape[1] == 16  # d_model
        assert result["decoder_vecs"].shape[1] == 16  # d_model

    def test_select_decoder_vectors(self, small_model):
        """Test decoder vector selection."""
        x = torch.randn(3, 8, 16)
        features = small_model.encode(x).to_sparse()

        if features._nnz() > 0:
            pos_ids, layer_ids, feat_ids, dec_vecs, enc_map = (
                small_model.select_decoder_vectors(features)
            )

            assert pos_ids.shape == layer_ids.shape
            assert pos_ids.shape == feat_ids.shape
            assert dec_vecs.shape[0] == pos_ids.shape[0]
            assert dec_vecs.shape[1] == 16  # d_model


class TestSaveLoad:
    def test_save_and_load(self, jumprelu_model):
        """Test that save/load preserves model behavior."""
        x = torch.randn(3, 8, 16)
        _, original_out, _ = jumprelu_model.forward(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            jumprelu_model.to_safetensors(tmpdir)
            loaded = load_spline_clt(tmpdir, device=torch.device("cpu"))

            _, loaded_out, _ = loaded.forward(x)
            assert torch.allclose(original_out, loaded_out, atol=1e-5), (
                f"Save/load mismatch: max diff = {(original_out - loaded_out).abs().max():.4e}"
            )

    def test_save_load_preserves_threshold(self, jumprelu_model):
        """Test that JumpReLU log-thresholds round-trip in memory."""
        from circuit_tracer.transcoder.activation_functions import JumpReLU

        original_threshold = jumprelu_model.activation_function.threshold.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            jumprelu_model.to_safetensors(tmpdir)
            loaded = load_spline_clt(tmpdir, device=torch.device("cpu"))

            assert isinstance(loaded.activation_function, JumpReLU)
            assert torch.allclose(
                loaded.activation_function.threshold,
                original_threshold,
                atol=1e-6,
            )

    def test_save_load_preserves_effective_threshold_on_disk(self, jumprelu_model):
        """On-disk threshold_i must store literal effective θ, not log θ."""
        from safetensors.torch import load_file

        jumprelu_model.activation_function.threshold.data.uniform_(-2.0, 1.0)
        expected = jumprelu_model.effective_threshold.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            jumprelu_model.to_safetensors(tmpdir)
            loaded = load_spline_clt(tmpdir, device=torch.device("cpu"))

            on_disk = load_file(os.path.join(tmpdir, "encoder_0.safetensors"))["threshold_0"]
            assert torch.allclose(on_disk, expected[0], atol=1e-6)
            assert torch.allclose(loaded.effective_threshold, expected, atol=1e-6)
            assert (loaded.effective_threshold > 0).all()

    def test_legacy_negative_threshold_on_disk_loads_positive(self, jumprelu_model):
        """Legacy checkpoints with θ < 0 on disk are clamped to a positive gate."""
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            jumprelu_model.to_safetensors(tmpdir)
            enc_path = os.path.join(tmpdir, "encoder_0.safetensors")
            enc_data = dict(load_file(enc_path))
            enc_data["threshold_0"] = torch.full_like(enc_data["threshold_0"], -2.2)
            save_file(enc_data, enc_path)

            loaded = load_spline_clt(tmpdir, device=torch.device("cpu"))
            assert (loaded.effective_threshold > 0).all()

    def test_save_and_load_linear_encoder(self):
        """Test that linear encoder save/load round-trip preserves behavior."""
        model = KANCrossLayerTranscoder(
            n_layers=2,
            d_transcoder=16,
            d_model=8,
            encoder_type="linear",
            activation_function="relu",
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 4, 8)
        _, original_out, _ = model.forward(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.to_safetensors(tmpdir)
            loaded = load_spline_clt(tmpdir, device=torch.device("cpu"))

            assert loaded.encoder_type == "linear"
            _, loaded_out, _ = loaded.forward(x)
            assert torch.allclose(original_out, loaded_out, atol=1e-5), (
                f"Linear save/load mismatch: max diff = {(original_out - loaded_out).abs().max():.4e}"
            )


class TestGradientFlow:
    def test_end_to_end_gradient(self, small_model):
        """Test gradient flows through full encode-decode pipeline."""
        x = torch.randn(3, 8, 16, requires_grad=True)
        _, out, _ = small_model.forward(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_encoder_parameters_get_gradients(self, small_model):
        """Test that KAN encoder parameters receive gradients."""
        x = torch.randn(3, 8, 16)
        _, out, _ = small_model.forward(x)
        loss = out.sum()
        loss.backward()

        for i, encoder in enumerate(small_model.encoders):
            for name, param in encoder.named_parameters():
                assert param.grad is not None, (
                    f"No gradient for encoder {i}, param {name}"
                )

    def test_decoder_parameters_get_gradients(self, small_model):
        """Test that decoder parameters receive gradients."""
        x = torch.randn(3, 8, 16)
        _, out, _ = small_model.forward(x)
        loss = out.sum()
        loss.backward()

        for i, w_dec in enumerate(small_model.W_dec):
            assert w_dec.grad is not None, f"No gradient for W_dec[{i}]"


class TestLossComputation:
    def test_total_loss_runs(self, small_model):
        """Test that total loss computation runs without error."""
        from spline_clt.training.loss import total_loss

        x = torch.randn(3, 8, 16)
        y = torch.randn(3, 8, 16)

        loss, metrics = total_loss(small_model, x, y)
        assert loss.requires_grad
        assert "loss/total" in metrics
        assert "loss/reconstruction" in metrics
        assert "loss/sparsity" in metrics

    def test_loss_decreases_with_training(self, small_model):
        """Test that loss decreases after a few optimization steps."""
        from spline_clt.training.loss import total_loss

        x = torch.randn(3, 8, 16)
        y = torch.randn(3, 8, 16)
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        initial_loss = None
        for step in range(20):
            optimizer.zero_grad()
            loss, metrics = total_loss(small_model, x, y, lambda_sparsity=0.0)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_sparsity_loss_nonnegative_under_threshold_pressure(self, jumprelu_model):
        """Sparsity penalty cannot invert when log-thresholds are pushed very low."""
        from spline_clt.training.loss import compute_decoder_norms, sparsity_loss

        jumprelu_model.activation_function.threshold.data.fill_(-6.0)
        x = torch.randn(3, 8, 16)
        activations = jumprelu_model.encode(x)
        assert (activations >= 0).all()

        dec_norms = compute_decoder_norms(jumprelu_model)
        loss = sparsity_loss(activations, dec_norms, lambda_=0.05)
        assert loss.item() >= 0.0

    def test_total_loss_nonnegative_for_linear_jumprelu(self):
        """Linear JumpReLU training loss components stay non-negative."""
        from spline_clt.training.loss import total_loss

        model = KANCrossLayerTranscoder(
            n_layers=3,
            d_transcoder=32,
            d_model=16,
            encoder_type="linear",
            activation_function="jump_relu",
            device=torch.device("cpu"),
        )
        x = torch.randn(3, 8, 16)
        y = torch.randn(3, 8, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for _ in range(30):
            optimizer.zero_grad()
            loss, metrics = total_loss(model, x, y, lambda_sparsity=0.05)
            assert metrics["loss/sparsity"] >= 0.0
            assert metrics["loss/reconstruction"] >= 0.0
            assert metrics["loss/kan_regularization"] >= 0.0
            assert metrics["loss/total"] >= 0.0
            loss.backward()
            optimizer.step()
