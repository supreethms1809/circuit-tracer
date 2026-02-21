"""Graph adapter for converting KAN-CLT attribution results to circuit-tracer format.

Bridges KAN-CLT's attribution output to circuit_tracer.graph.Graph so we can reuse
the existing pruning, visualization, and evaluation infrastructure.
"""

import torch

from circuit_tracer.graph import Graph


def create_graph_from_attribution(
    attribution_result: dict[str, torch.Tensor],
    input_string: str,
    input_tokens: torch.Tensor,
    logit_tokens: torch.Tensor,
    logit_probabilities: torch.Tensor,
    cfg,
    scan: str | list[str] | None = None,
) -> Graph:
    """Convert KAN-CLT attribution results to a circuit-tracer Graph.

    Args:
        attribution_result: Output from attribution.causal.build_attribution_graph().
        input_string: The input prompt string.
        input_tokens: Token IDs, shape (n_pos,).
        logit_tokens: Top logit token IDs, shape (n_logits,).
        logit_probabilities: Probabilities for top logits, shape (n_logits,).
        cfg: HookedTransformerConfig or compatible config object.
        scan: Transcoder identifier for visualization.

    Returns:
        circuit_tracer.graph.Graph instance compatible with pruning and visualization.
    """
    active_features = attribution_result["active_features"]
    activation_values = attribution_result["activation_values"]
    adjacency_matrix = attribution_result["adjacency_matrix"]

    # selected_features is a boolean mask over all possible features
    # For KAN-CLT, we just mark all returned features as selected
    n_active = len(active_features)
    selected_features = torch.ones(n_active, dtype=torch.bool)

    return Graph(
        input_string=input_string,
        input_tokens=input_tokens,
        active_features=active_features,
        adjacency_matrix=adjacency_matrix,
        cfg=cfg,
        logit_tokens=logit_tokens,
        logit_probabilities=logit_probabilities,
        selected_features=selected_features,
        activation_values=activation_values,
        scan=scan,
    )
