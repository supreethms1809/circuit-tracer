"""Graph adapter for converting Spline-CLT attribution results to circuit-tracer format.

Bridges Spline-CLT's attribution output to circuit_tracer.graph.Graph so we can reuse
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
    """Convert Spline-CLT attribution results to a circuit-tracer Graph.

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
    active_features = attribution_result["active_features"].detach()
    activation_values = attribution_result["activation_values"].detach()
    adjacency_matrix = attribution_result["adjacency_matrix"].detach()

    # circuit_tracer.Graph expects selected_features to be an index tensor into
    # active_features, not a boolean mask. Since active_features already contains
    # only the selected nodes, this is the identity index map.
    n_active = len(active_features)
    selected_features = torch.arange(n_active, dtype=torch.long)

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
