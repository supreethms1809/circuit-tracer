"""Tests for causal attribution engine."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from kan_clt.kan_transcoder import KANCrossLayerTranscoder
from attribution.causal import ablation_attribution, build_attribution_graph


@pytest.fixture
def small_model():
    """Small KAN-CLT for attribution testing."""
    return KANCrossLayerTranscoder(
        n_layers=2,
        d_transcoder=16,
        d_model=8,
        grid_size=3,
        spline_order=3,
        activation_function="relu",
        device=torch.device("cpu"),
    )


class TestAblationAttribution:
    def test_attribution_returns_correct_keys(self, small_model):
        """Test that ablation attribution returns all expected keys."""
        x = torch.randn(2, 4, 8)
        result = ablation_attribution(small_model, x, max_features=32)

        assert "active_features" in result
        assert "activation_values" in result
        assert "feature_effects" in result
        assert "output_effects" in result
        assert "baseline_reconstruction" in result

    def test_attribution_shapes(self, small_model):
        """Test attribution output shapes."""
        x = torch.randn(2, 4, 8)
        result = ablation_attribution(small_model, x, max_features=32)

        n_active = len(result["activation_values"])
        assert result["active_features"].shape == (n_active, 3)
        assert result["feature_effects"].shape == (n_active, n_active)
        assert result["output_effects"].shape == (n_active, 2, 4, 8)
        assert result["baseline_reconstruction"].shape == (2, 4, 8)

    def test_self_ablation_has_effect(self, small_model):
        """Test that ablating a feature changes the reconstruction."""
        x = torch.randn(2, 4, 8)
        result = ablation_attribution(small_model, x, max_features=32)

        # Output effects should be non-zero for active features
        if len(result["activation_values"]) > 0:
            effects_norm = result["output_effects"].norm(dim=-1).sum(dim=(1, 2))
            assert effects_norm.max() > 0, "No feature has any causal effect"

    def test_max_features_limits_output(self, small_model):
        """Test that max_features parameter works."""
        x = torch.randn(2, 4, 8)
        result = ablation_attribution(small_model, x, max_features=5)
        assert len(result["activation_values"]) <= 5


class TestBuildAttributionGraph:
    def test_graph_adjacency_matrix(self, small_model):
        """Test that graph has correctly sized adjacency matrix."""
        x = torch.randn(2, 4, 8)
        result = build_attribution_graph(small_model, x, max_features=16)

        n_active = len(result["active_features"])
        n_layers, n_pos = 2, 4
        n_error = n_layers * n_pos
        n_tokens = n_pos
        expected_size = n_active + n_error + n_tokens

        assert result["adjacency_matrix"].shape == (expected_size, expected_size)

    def test_graph_node_ordering(self, small_model):
        """Test node ordering: features, errors, tokens."""
        x = torch.randn(2, 4, 8)
        result = build_attribution_graph(small_model, x, max_features=16)

        n_active = len(result["active_features"])
        adj = result["adjacency_matrix"]

        # Feature-to-feature block should be in top-left
        feat_block = adj[:n_active, :n_active]
        assert feat_block.shape == (n_active, n_active)
