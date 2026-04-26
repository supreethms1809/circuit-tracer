"""Tests for attribution graph construction via the original circuit_tracer
backward-pass pipeline, applied to both Spline-CLT (KAN encoder) and
linear-CLT transcoders.

These tests verify that:
1. The Spline-CLT / linear-CLT ``compute_attribution_components`` output is
   compatible with the circuit_tracer ``AttributionContext``.
2. The full ``attribute()`` pipeline produces a well-formed ``Graph`` with the
   correct node layout, adjacency matrix dimensions, and edge properties
   (cross-position effects, layer ordering, error/token/logit nodes).
3. ``build_prompt_graph`` in the paper evaluation module delegates to the
   original circuit_tracer pipeline and returns the expected metadata.
"""

import pytest
import torch

from circuit_tracer.attribution.attribute_transformerlens import attribute
from circuit_tracer.graph import Graph, compute_graph_scores
from circuit_tracer.replacement_model.replacement_model import ReplacementModel
from spline_clt.kan_transcoder import KANCrossLayerTranscoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# GPT-2 small: 12 layers, 768 d_model.  We use a tiny d_transcoder to keep
# the test fast while matching the transformer's actual dimensions.
_GPT2_N_LAYERS = 12
_GPT2_D_MODEL = 768
_GPT2_MODEL_NAME = "gpt2"


def _make_clt(encoder_type: str = "kan") -> KANCrossLayerTranscoder:
    """Build a CLT that matches GPT-2's dimensions with a tiny feature count."""
    return KANCrossLayerTranscoder(
        n_layers=_GPT2_N_LAYERS,
        d_transcoder=32,  # small for speed
        d_model=_GPT2_D_MODEL,
        encoder_type=encoder_type,
        grid_size=3,
        spline_order=3,
        activation_function="relu",
        device=torch.device("cpu"),
    )


def _make_replacement_model(clt: KANCrossLayerTranscoder):
    """Create a ReplacementModel wrapping GPT-2 + CLT."""
    return ReplacementModel.from_pretrained_and_transcoders(
        model_name=_GPT2_MODEL_NAME,
        transcoders=clt,
        backend="transformerlens",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Tests: compute_attribution_components compatibility
# ---------------------------------------------------------------------------

class TestAttributionComponents:
    """Verify that Spline-CLT returns attribution components in the format
    expected by the circuit_tracer ``AttributionContext``."""

    @staticmethod
    def _small_clt(encoder_type: str) -> KANCrossLayerTranscoder:
        """Standalone small CLT for component-level tests (no transformer needed)."""
        return KANCrossLayerTranscoder(
            n_layers=2, d_transcoder=16, d_model=8,
            encoder_type=encoder_type, grid_size=3, spline_order=3,
            activation_function="relu", device=torch.device("cpu"),
        )

    @pytest.mark.parametrize("encoder_type", ["kan", "linear"])
    def test_component_keys(self, encoder_type):
        clt = self._small_clt(encoder_type)
        x = torch.randn(clt.n_layers, 4, clt.d_model)
        components = clt.compute_attribution_components(x)
        expected_keys = {
            "activation_matrix",
            "reconstruction",
            "encoder_vecs",
            "decoder_vecs",
            "encoder_to_decoder_map",
            "decoder_locations",
        }
        assert expected_keys == set(components.keys())

    @pytest.mark.parametrize("encoder_type", ["kan", "linear"])
    def test_encoder_vecs_shape(self, encoder_type):
        """encoder_vecs should have shape (total_active, d_model)."""
        clt = self._small_clt(encoder_type)
        x = torch.randn(clt.n_layers, 4, clt.d_model)
        components = clt.compute_attribution_components(x)
        n_active = components["activation_matrix"]._nnz()
        if n_active > 0:
            assert components["encoder_vecs"].shape == (n_active, clt.d_model)

    @pytest.mark.parametrize("encoder_type", ["kan", "linear"])
    def test_decoder_vecs_d_model(self, encoder_type):
        """decoder_vecs last dim should be d_model."""
        clt = self._small_clt(encoder_type)
        x = torch.randn(clt.n_layers, 4, clt.d_model)
        components = clt.compute_attribution_components(x)
        if components["decoder_vecs"].numel() > 0:
            assert components["decoder_vecs"].shape[-1] == clt.d_model

    @pytest.mark.parametrize("encoder_type", ["kan", "linear"])
    def test_decoder_locations_shape(self, encoder_type):
        """decoder_locations should be (2, n_decoder_entries)."""
        clt = self._small_clt(encoder_type)
        x = torch.randn(clt.n_layers, 4, clt.d_model)
        components = clt.compute_attribution_components(x)
        assert components["decoder_locations"].ndim == 2
        assert components["decoder_locations"].shape[0] == 2


# ---------------------------------------------------------------------------
# Tests: full attribution graph via ReplacementModel + attribute()
# ---------------------------------------------------------------------------

class TestFullAttributionGraph:
    """End-to-end tests that run the original circuit_tracer ``attribute()``
    with a Spline-CLT (and linear CLT) plugged in, verifying graph structure."""

    @pytest.fixture(params=["kan", "linear"])
    def replacement_model_and_clt(self, request):
        clt = _make_clt(encoder_type=request.param)
        model = _make_replacement_model(clt)
        return model, clt, request.param

    def test_attribute_returns_graph(self, replacement_model_and_clt):
        model, clt, _ = replacement_model_and_clt
        graph = attribute(prompt="Hello world", model=model, max_n_logits=3)
        assert isinstance(graph, Graph)

    def test_graph_node_layout(self, replacement_model_and_clt):
        """Verify adjacency matrix shape matches:
        [n_selected_features + n_error + n_tokens + n_logits]^2"""
        model, clt, _ = replacement_model_and_clt
        graph = attribute(prompt="Hello world", model=model, max_n_logits=3)

        n_selected = len(graph.selected_features)
        n_pos = graph.n_pos
        n_layers = model.cfg.n_layers
        n_logits = len(graph.logit_tokens)
        n_error = n_layers * n_pos
        n_tokens = n_pos

        expected_size = n_selected + n_error + n_tokens + n_logits
        assert graph.adjacency_matrix.shape == (expected_size, expected_size), (
            f"Expected ({expected_size}, {expected_size}), "
            f"got {graph.adjacency_matrix.shape} "
            f"(n_selected={n_selected}, n_error={n_error}, "
            f"n_tokens={n_tokens}, n_logits={n_logits})"
        )

    def test_feature_rows_populated(self, replacement_model_and_clt):
        """Feature rows (target = feature) should have non-zero entries."""
        model, clt, _ = replacement_model_and_clt
        graph = attribute(prompt="Hello world", model=model, max_n_logits=3)
        n_selected = len(graph.selected_features)
        if n_selected > 0:
            feature_rows = graph.adjacency_matrix[:n_selected]
            assert feature_rows.abs().sum() > 0, \
                "Feature rows should have non-zero attribution edges"

    def test_logit_rows_populated(self, replacement_model_and_clt):
        """Logit rows should have non-zero entries."""
        model, clt, _ = replacement_model_and_clt
        graph = attribute(prompt="Hello world", model=model, max_n_logits=3)
        n_logits = len(graph.logit_tokens)
        if n_logits > 0:
            logit_rows = graph.adjacency_matrix[-n_logits:]
            assert logit_rows.abs().sum() > 0, \
                "Logit rows should have non-zero edges"

    def test_error_and_token_rows_are_zero(self, replacement_model_and_clt):
        """Error and token nodes are source-only — their rows should be all zeros."""
        model, clt, _ = replacement_model_and_clt
        graph = attribute(prompt="Hello world", model=model, max_n_logits=3)
        n_selected = len(graph.selected_features)
        n_logits = len(graph.logit_tokens)
        n_pos = graph.n_pos
        n_error = model.cfg.n_layers * n_pos

        error_token_start = n_selected
        error_token_end = n_selected + n_error + n_pos
        error_token_rows = graph.adjacency_matrix[error_token_start:error_token_end]
        assert error_token_rows.abs().sum() == 0, \
            "Error and token rows should be zero (source-only nodes)"

    def test_cross_position_edges_exist(self, replacement_model_and_clt):
        """The backward-pass attribution should capture cross-position effects
        mediated by attention.  At least some feature-to-feature edges should
        connect different token positions."""
        model, clt, _ = replacement_model_and_clt
        # Use a longer prompt to increase the chance of cross-position attention effects
        graph = attribute(
            prompt="The capital of France is Paris and the capital of Germany is",
            model=model,
            max_n_logits=3,
        )
        n_selected = len(graph.selected_features)
        if n_selected < 2:
            pytest.skip("Too few selected features for cross-position check")

        # Get the selected features' positions
        selected_feats = graph.active_features[graph.selected_features]  # (n_sel, 3)
        positions = selected_feats[:, 1]  # position column

        # Check feature-to-feature block for edges between different positions
        ff_block = graph.adjacency_matrix[:n_selected, :n_selected]
        cross_pos_found = False
        for tgt_idx in range(n_selected):
            for src_idx in range(n_selected):
                if positions[tgt_idx] != positions[src_idx] and ff_block[tgt_idx, src_idx].abs() > 1e-8:
                    cross_pos_found = True
                    break
            if cross_pos_found:
                break

        assert cross_pos_found, (
            "Expected cross-position feature-to-feature edges from attention-mediated "
            "attribution, but found none.  This was the key bug in the CLT-only "
            "ablation approach."
        )

    def test_layer_ordering_in_feature_edges(self, replacement_model_and_clt):
        """Feature-to-feature edges should flow from earlier layers to later layers
        (source layer < target layer) in the original circuit_tracer convention."""
        model, clt, _ = replacement_model_and_clt
        graph = attribute(prompt="Hello world", model=model, max_n_logits=3)
        n_selected = len(graph.selected_features)
        if n_selected < 2:
            pytest.skip("Too few selected features for layer ordering check")

        selected_feats = graph.active_features[graph.selected_features]
        layers = selected_feats[:, 0]

        ff_block = graph.adjacency_matrix[:n_selected, :n_selected]
        # adj[target, source]: for non-zero entries, source layer should be < target layer
        for tgt_idx in range(n_selected):
            for src_idx in range(n_selected):
                if ff_block[tgt_idx, src_idx].abs() > 1e-8:
                    assert layers[src_idx] < layers[tgt_idx], (
                        f"Feature edge from layer {layers[src_idx]} to layer "
                        f"{layers[tgt_idx]} violates layer ordering (source must "
                        f"be at a strictly earlier layer than target)"
                    )

    def test_graph_scores_computable(self, replacement_model_and_clt):
        """compute_graph_scores should run without error on the produced graph."""
        model, clt, _ = replacement_model_and_clt
        graph = attribute(prompt="Hello world", model=model, max_n_logits=3)
        replacement_score, completeness_score = compute_graph_scores(graph)
        assert 0.0 <= replacement_score <= 1.0
        assert 0.0 <= completeness_score <= 1.0

    def test_selected_features_are_valid_indices(self, replacement_model_and_clt):
        """selected_features should be valid indices into active_features."""
        model, clt, _ = replacement_model_and_clt
        graph = attribute(prompt="Hello world", model=model, max_n_logits=3)
        n_total_active = graph.active_features.shape[0]
        assert (graph.selected_features >= 0).all()
        assert (graph.selected_features < n_total_active).all()

    def test_activation_values_match_selected_count(self, replacement_model_and_clt):
        """activation_values should have one entry per active feature."""
        model, clt, _ = replacement_model_and_clt
        graph = attribute(prompt="Hello world", model=model, max_n_logits=3)
        assert graph.activation_values.shape[0] == graph.active_features.shape[0]


# ---------------------------------------------------------------------------
# Tests: Spline-CLT branched graph construction (Jacobian-based ablation)
# ---------------------------------------------------------------------------


class TestSplineGraphBranch:
    """Verify the Spline-only graph construction path (Jacobian-based causal
    ablation) produces a well-formed circuit_tracer ``Graph``.
    """

    @pytest.fixture(scope="class")
    def lm(self):
        from transformer_lens import HookedTransformer
        return HookedTransformer.from_pretrained(
            _GPT2_MODEL_NAME,
            device="cpu",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )

    @pytest.fixture(scope="class")
    def spline_clt(self):
        return _make_clt(encoder_type="kan")

    def _build(self, lm, spline_clt, prompt: str = "Hello world", max_features: int = 16):
        from attribution.spline_graph import build_spline_graph
        from spline_clt.paper.evaluate import collect_prompt_cache

        cache = collect_prompt_cache(
            lm=lm,
            prompt=prompt,
            n_layers=spline_clt.n_layers,
            feature_input_hook=spline_clt.feature_input_hook,
            feature_output_hook=spline_clt.feature_output_hook,
        )
        return build_spline_graph(
            model=spline_clt,
            lm=lm,
            prompt=prompt,
            input_tokens=cache.token_ids,
            mlp_inputs=cache.mlp_inputs,
            mlp_outputs=cache.mlp_outputs,
            final_logits=cache.logits[-1],
            max_features=max_features,
            max_n_logits=3,
            desired_logit_prob=0.9,
            method="jacobian_ablation",
            scan="test-spline",
        )

    def test_returns_graph(self, lm, spline_clt):
        graph = self._build(lm, spline_clt)
        assert isinstance(graph, Graph)

    def test_node_layout(self, lm, spline_clt):
        graph = self._build(lm, spline_clt)
        n_selected = len(graph.selected_features)
        n_pos = graph.n_pos
        n_layers = lm.cfg.n_layers
        n_logits = len(graph.logit_tokens)
        expected = n_selected + n_layers * n_pos + n_pos + n_logits
        assert graph.adjacency_matrix.shape == (expected, expected)

    def test_error_token_rows_zero(self, lm, spline_clt):
        graph = self._build(lm, spline_clt)
        n_sel = len(graph.selected_features)
        n_pos = graph.n_pos
        n_err = lm.cfg.n_layers * n_pos
        block = graph.adjacency_matrix[n_sel : n_sel + n_err + n_pos]
        assert block.abs().sum().item() == 0.0

    def test_layer_ordering(self, lm, spline_clt):
        graph = self._build(lm, spline_clt)
        n_sel = len(graph.selected_features)
        if n_sel < 2:
            pytest.skip("Too few features for ordering check")
        layers = graph.active_features[graph.selected_features][:, 0]
        ff = graph.adjacency_matrix[:n_sel, :n_sel]
        nz = ff.abs() > 1e-8
        for t in range(n_sel):
            for s in range(n_sel):
                if nz[t, s]:
                    assert layers[s] < layers[t]

    def test_compute_graph_scores(self, lm, spline_clt):
        graph = self._build(lm, spline_clt)
        rep, comp = compute_graph_scores(graph)
        assert 0.0 <= rep <= 1.0
        assert 0.0 <= comp <= 1.0

    def test_shapley_method_not_implemented(self, lm, spline_clt):
        from attribution.spline_graph import build_spline_graph
        from spline_clt.paper.evaluate import collect_prompt_cache

        cache = collect_prompt_cache(
            lm=lm,
            prompt="Hello world",
            n_layers=spline_clt.n_layers,
            feature_input_hook=spline_clt.feature_input_hook,
            feature_output_hook=spline_clt.feature_output_hook,
        )
        with pytest.raises(NotImplementedError):
            build_spline_graph(
                model=spline_clt, lm=lm, prompt="Hello world",
                input_tokens=cache.token_ids,
                mlp_inputs=cache.mlp_inputs,
                mlp_outputs=cache.mlp_outputs,
                final_logits=cache.logits[-1],
                max_features=8, max_n_logits=3,
                desired_logit_prob=0.9, method="shapley", scan="test",
            )
