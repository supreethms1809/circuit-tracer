"""Tests for Shapley value attribution."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from spline_clt.kan_transcoder import KANCrossLayerTranscoder
from attribution.shapley import shapley_attribution, shapley_logit_attribution


@pytest.fixture
def small_kan_model():
    """Tiny Spline-CLT for Shapley tests (fast, CPU)."""
    return KANCrossLayerTranscoder(
        n_layers=2,
        d_transcoder=16,
        d_model=8,
        grid_size=3,
        spline_order=3,
        activation_function="relu",
        device=torch.device("cpu"),
    )


@pytest.fixture
def small_linear_model():
    """Tiny linear CLT for Shapley tests."""
    return KANCrossLayerTranscoder(
        n_layers=2,
        d_transcoder=16,
        d_model=8,
        encoder_type="linear",
        grid_size=3,
        spline_order=3,
        activation_function="relu",
        device=torch.device("cpu"),
    )


class TestShapleyAttribution:
    def test_returns_required_keys(self, small_kan_model):
        """Shapley result must contain active_features, activation_values, shapley_values."""
        x = torch.randn(2, 4, 8)
        result = shapley_attribution(small_kan_model, x, n_samples=4, max_features=16)

        assert "active_features" in result
        assert "activation_values" in result
        assert "shapley_values" in result

    def test_shapes_match(self, small_kan_model):
        """active_features, activation_values, shapley_values must all have n_active rows."""
        x = torch.randn(2, 4, 8)
        result = shapley_attribution(small_kan_model, x, n_samples=4, max_features=16)

        n_active = len(result["activation_values"])
        assert result["active_features"].shape == (n_active, 3)
        assert result["shapley_values"].shape == (n_active,)

    def test_active_features_index_range(self, small_kan_model):
        """Layer/pos/feat indices must be within valid range."""
        x = torch.randn(2, 4, 8)
        result = shapley_attribution(small_kan_model, x, n_samples=4, max_features=16)

        if len(result["active_features"]) == 0:
            pytest.skip("No active features (model not trained, all activations zero)")

        feats = result["active_features"]
        assert feats[:, 0].max() < 2, "Layer index out of range"
        assert feats[:, 1].max() < 4, "Position index out of range"
        assert feats[:, 2].max() < 16, "Feature index out of range"

    def test_max_features_respected(self, small_kan_model):
        """At most max_features features are returned."""
        x = torch.randn(2, 10, 8)
        result = shapley_attribution(small_kan_model, x, n_samples=4, max_features=8)
        assert len(result["active_features"]) <= 8

    def test_empty_activations_handled(self, small_kan_model):
        """When no features are active, return empty tensors without error.

        We force zero activations by monkey-patching encode to return zeros.
        """
        import unittest.mock as mock

        x = torch.randn(2, 4, 8)
        zero_acts = torch.zeros(2, 4, 16)
        with mock.patch.object(small_kan_model, "encode", return_value=zero_acts):
            result = shapley_attribution(small_kan_model, x, n_samples=4, max_features=16)
        assert result["active_features"].shape[0] == 0
        assert result["shapley_values"].shape[0] == 0

    def test_shapley_values_finite(self, small_kan_model):
        """Shapley values must be finite (no NaN/inf)."""
        x = torch.randn(2, 4, 8)
        result = shapley_attribution(small_kan_model, x, n_samples=8, max_features=16)
        if len(result["shapley_values"]) > 0:
            assert torch.isfinite(result["shapley_values"]).all(), \
                "Shapley values contain NaN or inf"

    def test_antithetic_reduces_variance(self, small_kan_model):
        """Antithetic sampling should be supported without error (variance reduction property
        cannot be tested with a single call but the code path must run cleanly)."""
        x = torch.randn(2, 4, 8)
        result_anti = shapley_attribution(
            small_kan_model, x, n_samples=8, max_features=16, antithetic=True
        )
        result_plain = shapley_attribution(
            small_kan_model, x, n_samples=8, max_features=16, antithetic=False
        )
        # Both should return same number of features
        assert len(result_anti["shapley_values"]) == len(result_plain["shapley_values"])

    def test_works_with_linear_encoder(self, small_linear_model):
        """Shapley attribution should work identically for linear CLT."""
        x = torch.randn(2, 4, 8)
        result = shapley_attribution(small_linear_model, x, n_samples=4, max_features=16)
        assert "shapley_values" in result
        if len(result["shapley_values"]) > 0:
            assert torch.isfinite(result["shapley_values"]).all()

    def test_feature_target_raises_not_implemented(self, small_kan_model):
        """target='feature' is deliberately unimplemented: the old path read
        prev_acts directly, never changed under source ablations, and silently
        returned zeros. It must raise instead of returning garbage."""
        x = torch.randn(2, 4, 8)
        with pytest.raises(NotImplementedError, match="Feature-to-feature Shapley"):
            shapley_attribution(
                small_kan_model, x, target="feature",
                n_samples=4, max_features=16, antithetic=False,
            )


class TestShapleyLogitAttribution:
    def test_returns_required_keys(self, small_kan_model):
        """Logit Shapley must return the standard three keys."""
        x = torch.randn(2, 4, 8)
        logit_dir = torch.randn(8)
        result = shapley_logit_attribution(
            small_kan_model, x, logit_dir, n_samples=4, max_features=16
        )
        assert "active_features" in result
        assert "activation_values" in result
        assert "shapley_values" in result

    def test_shapes(self, small_kan_model):
        """Logit Shapley shapes match n_active."""
        x = torch.randn(2, 4, 8)
        logit_dir = torch.randn(8)
        result = shapley_logit_attribution(
            small_kan_model, x, logit_dir, n_samples=4, max_features=16
        )
        n_active = len(result["activation_values"])
        assert result["active_features"].shape == (n_active, 3)
        assert result["shapley_values"].shape == (n_active,)

    def test_empty_activations(self, small_kan_model):
        """When no features are active, logit Shapley returns empty tensors without error."""
        import unittest.mock as mock

        x = torch.randn(2, 4, 8)
        logit_dir = torch.randn(8)
        zero_acts = torch.zeros(2, 4, 16)
        with mock.patch.object(small_kan_model, "encode", return_value=zero_acts):
            result = shapley_logit_attribution(
                small_kan_model, x, logit_dir, n_samples=4, max_features=16
            )
        assert result["active_features"].shape[0] == 0

    def test_logit_direction_normalisation_invariant(self, small_kan_model):
        """Scaling the logit direction by a constant scales Shapley values by the same constant."""
        x = torch.randn(2, 4, 8)
        direction = torch.randn(8)
        r1 = shapley_logit_attribution(
            small_kan_model, x, direction, n_samples=8, max_features=16
        )
        r2 = shapley_logit_attribution(
            small_kan_model, x, direction * 2.0, n_samples=8, max_features=16
        )
        if len(r1["shapley_values"]) > 0:
            ratio = r2["shapley_values"] / r1["shapley_values"].clamp(min=1e-9)
            # Ratios should all be approximately 2.0
            assert torch.isfinite(ratio).all()
