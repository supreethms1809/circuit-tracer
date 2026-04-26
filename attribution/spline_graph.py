"""Spline-CLT graph construction via Jacobian-based causal ablation.

The linear-CLT path uses ``circuit_tracer.attribute()`` (matmul through a
frozen-attention local replacement model). For Spline-CLT, the encoder is
nonlinear and that matmul shortcut no longer corresponds to the encoder's
true response. Instead, we run causal ablations on the CLT and project the
resulting reconstruction deltas onto each downstream feature's *local*
encoder direction (the Jacobian J_KAN(x)). Same-position error and
embedding edges are also projected through the Jacobian. This is a single
combined method: ablation provides the causal delta, the Jacobian makes
that delta projectable into feature space.
"""

from __future__ import annotations

from typing import Any

import torch

from attribution.causal import build_attribution_graph
from circuit_tracer.graph import Graph
from circuit_tracer.utils.salient_logits import compute_salient_logits
from spline_clt.kan_transcoder import KANCrossLayerTranscoder


def build_spline_graph(
    *,
    model: KANCrossLayerTranscoder,
    lm: Any,
    prompt: str,
    input_tokens: torch.Tensor,
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    final_logits: torch.Tensor,
    max_features: int = 256,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    method: str = "jacobian_ablation",
    scan: str | None = None,
) -> Graph:
    """Construct a circuit-tracer ``Graph`` from a Spline-CLT.

    Args:
        model: Trained Spline-CLT (KAN encoder).
        lm: HookedTransformer used to source the unembedding and config.
        prompt: Original input string.
        input_tokens: ``(n_pos,)`` token ids (BOS prepended), matching mlp_inputs.
        mlp_inputs: ``(n_layers, n_pos, d_model)`` residual-stream inputs.
        mlp_outputs: ``(n_layers, n_pos, d_model)`` true MLP outputs.
        final_logits: ``(d_vocab,)`` logits at the final position.
        max_features: Cap on feature nodes.
        max_n_logits / desired_logit_prob: Logit selection knobs.
        method: Currently only ``"jacobian_ablation"`` is implemented.
        scan: Transcoder scan id for the graph header.
    """
    if method != "jacobian_ablation":
        raise NotImplementedError(
            f"Spline graph method {method!r} not implemented. "
            "Only 'jacobian_ablation' is supported."
        )

    device = mlp_inputs.device
    n_pos = mlp_inputs.shape[1]

    W_U = lm.W_U  # (d_model, d_vocab)
    logit_idx, logit_p, _ = compute_salient_logits(
        final_logits.to(device).float(),
        W_U.to(device).float(),
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )

    # Token embedding vectors at each position (W_E rows).
    token_vectors = lm.W_E[input_tokens.to(W_U.device)].to(device=device, dtype=mlp_inputs.dtype)

    raw = build_attribution_graph(
        model=model,
        x_in=mlp_inputs,
        max_features=max_features,
        W_U=W_U.to(device).float(),
        logit_token_ids=logit_idx.to(device),
        y_true=mlp_outputs,
        token_vectors=token_vectors,
    )

    active_features = raw["active_features"].cpu()  # (n_active, 3) [layer, pos, feat]
    activation_values = raw["activation_values"].cpu()
    adjacency = raw["adjacency_matrix"].cpu().float()
    n_active = active_features.shape[0]

    # circuit_tracer's Graph expects ``selected_features`` to index into
    # ``active_features``. We attribute every kept feature, so the selected set
    # is the full range.
    selected_features = torch.arange(n_active, dtype=torch.int64)

    return Graph(
        input_string=prompt,
        input_tokens=input_tokens.cpu(),
        active_features=active_features,
        adjacency_matrix=adjacency,
        cfg=lm.cfg,
        logit_tokens=logit_idx.cpu(),
        logit_probabilities=logit_p.cpu(),
        selected_features=selected_features,
        activation_values=activation_values,
        scan=scan,
    )
