"""Shapley value attribution for Spline-CLT.

Since the KAN encoder is nonlinear, edge weights cannot be computed as
a · w_{s→t}. Instead we use game-theoretic Shapley values:

For each target feature t, the Shapley value of source feature s is:
    φ_s(t) = (1/n!) Σ_{π ∈ S_n} [v(S_π^s ∪ {s}, t) - v(S_π^s, t)]

where S_π^s is the set of features preceding s in permutation π, and
v(S, t) is the value function (activation of t when only features in S
are active).

We approximate this with Monte Carlo sampling + antithetic pairs for
variance reduction. For output logits the value function is the logit.

Reference: Shapley (1953), Lundberg & Lee (2017, SHAP).
"""

import torch
from tqdm import tqdm

from spline_clt.kan_transcoder import KANCrossLayerTranscoder
from spline_clt.seed import seed_everything


@torch.no_grad()
def shapley_attribution(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    target: str = "reconstruction",
    n_samples: int = 256,
    max_features: int = 64,
    antithetic: bool = True,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute Shapley value attribution for active features.

    Args:
        model: Trained Spline-CLT model.
        x_in: Residual stream inputs, shape (n_layers, n_pos, d_model).
        target: "reconstruction" (MSE improvement) or "feature" (feature-to-feature).
        n_samples: Number of permutation samples per feature. More → lower variance.
        max_features: Maximum active features to attribute (top by magnitude).
        antithetic: Use antithetic sampling (doubles effective samples, halves variance).

    Returns:
        Dict with:
            active_features: (n_active, 3) tensor of (layer, pos, feat_idx).
            activation_values: (n_active,) baseline activations.
            shapley_values: (n_active,) Shapley values — feature contribution to target.
            feature_shapley: (n_active, n_active) feature-to-feature Shapley matrix (if target="feature").
    """
    n_layers, n_pos, d_model = x_in.shape
    device = x_in.device
    dtype = x_in.dtype
    if seed is not None:
        seed_everything(seed)

    # --- Baseline activations ---
    baseline_acts = model.encode(x_in)  # (n_layers, n_pos, d_transcoder)
    sparse = baseline_acts.to_sparse().coalesce()
    layer_idx, pos_idx, feat_idx = sparse.indices()
    act_values = sparse.values()

    n_active = len(act_values)
    if n_active == 0:
        return {
            "active_features": torch.zeros(0, 3, dtype=torch.long, device=device),
            "activation_values": torch.zeros(0, device=device),
            "shapley_values": torch.zeros(0, device=device),
        }

    # Limit to top features by activation magnitude
    if n_active > max_features:
        top_k = act_values.abs().topk(max_features).indices
        layer_idx = layer_idx[top_k]
        pos_idx = pos_idx[top_k]
        feat_idx = feat_idx[top_k]
        act_values = act_values[top_k]
        n_active = max_features

    active_features = torch.stack([layer_idx, pos_idx, feat_idx], dim=1)

    # --- Value function ---
    def value_fn(included_mask: torch.Tensor) -> torch.Tensor:
        """Compute value with a subset of active features included.

        Args:
            included_mask: Boolean tensor of shape (n_active,).

        Returns:
            Scalar value (reconstruction quality).
        """
        masked_acts = baseline_acts.clone()
        # Zero out features NOT in the included set
        for i in range(n_active):
            if not included_mask[i]:
                l, p, f = int(layer_idx[i]), int(pos_idx[i]), int(feat_idx[i])
                masked_acts[l, p, f] = 0.0

        y_hat = model.decode_dense(masked_acts, input_acts=x_in)
        return y_hat  # (n_layers, n_pos, d_model)

    # Baseline (all features included) and empty (no features)
    full_output = value_fn(torch.ones(n_active, dtype=torch.bool, device=device))
    empty_output = value_fn(torch.zeros(n_active, dtype=torch.bool, device=device))

    # --- Monte Carlo Shapley ---
    # For "reconstruction" target: Shapley value = contribution to ||y_hat - y_empty||^2 reduction
    shapley_vals = torch.zeros(n_active, device=device, dtype=torch.float32)

    # Effective number of permutations (antithetic doubles them)
    n_perm = n_samples // 2 if antithetic else n_samples

    for _ in tqdm(range(n_perm), desc="Shapley sampling", leave=False):
        perm = torch.randperm(n_active, device=device)

        for use_antithetic in ([False, True] if antithetic else [False]):
            order = perm.flip(0) if use_antithetic else perm

            prev_mask = torch.zeros(n_active, dtype=torch.bool, device=device)
            prev_output = value_fn(prev_mask)

            for i in order:
                curr_mask = prev_mask.clone()
                curr_mask[i] = True
                curr_output = value_fn(curr_mask)

                # Marginal contribution: improvement in output closeness to full_output
                prev_dist = (full_output.float() - prev_output.float()).pow(2).mean()
                curr_dist = (full_output.float() - curr_output.float()).pow(2).mean()
                shapley_vals[i] += (prev_dist - curr_dist).item()

                prev_mask = curr_mask
                prev_output = curr_output

    n_total_perms = n_perm * (2 if antithetic else 1)
    shapley_vals /= n_total_perms

    # --- Feature-to-feature Shapley (optional, only if n_active is small) ---
    feature_shapley = None
    if target == "feature" and n_active <= 32:
        feature_shapley = _feature_to_feature_shapley(
            model, x_in, baseline_acts,
            layer_idx, pos_idx, feat_idx, act_values,
            n_active, n_samples=min(n_samples, 64),
        )

    result = {
        "active_features": active_features,
        "activation_values": act_values,
        "shapley_values": shapley_vals,
        "full_output": full_output,
        "empty_output": empty_output,
    }
    if feature_shapley is not None:
        result["feature_shapley"] = feature_shapley
    return result


@torch.no_grad()
def _feature_to_feature_shapley(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    baseline_acts: torch.Tensor,
    layer_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    feat_idx: torch.Tensor,
    act_values: torch.Tensor,
    n_active: int,
    n_samples: int = 64,
) -> torch.Tensor:
    """Compute (n_active, n_active) Shapley values for feature → feature influence.

    φ_{s→t} = Shapley value of source s for target t's activation level.
    """
    device = x_in.device
    shapley_mat = torch.zeros(n_active, n_active, device=device, dtype=torch.float32)

    for t_idx in range(n_active):
        t_layer = int(layer_idx[t_idx])
        t_pos = int(pos_idx[t_idx])
        t_feat = int(feat_idx[t_idx])
        # Only features at earlier/same layers can influence t
        sources = [s for s in range(n_active)
                   if int(layer_idx[s]) <= t_layer and s != t_idx]
        if not sources:
            continue

        n_sources = len(sources)
        vals = torch.zeros(n_sources, device=device, dtype=torch.float32)

        for _ in range(n_samples):
            perm = torch.randperm(n_sources)
            prev_mask = torch.zeros(n_active, dtype=torch.bool, device=device)
            prev_acts = baseline_acts.clone()
            # zero out all source features initially
            for s in sources:
                prev_acts[int(layer_idx[s]), int(pos_idx[s]), int(feat_idx[s])] = 0.0
            prev_val = prev_acts[t_layer, t_pos, t_feat].float().item()

            for rank, s_local in enumerate(perm):
                s_global = sources[s_local]
                curr_acts = prev_acts.clone()
                curr_acts[
                    int(layer_idx[s_global]), int(pos_idx[s_global]), int(feat_idx[s_global])
                ] = act_values[s_global]
                curr_val = curr_acts[t_layer, t_pos, t_feat].float().item()
                vals[s_local] += curr_val - prev_val
                prev_acts = curr_acts
                prev_val = curr_val

        vals /= n_samples
        for i, s in enumerate(sources):
            shapley_mat[s, t_idx] = vals[i]

    return shapley_mat


@torch.no_grad()
def shapley_logit_attribution(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    logit_target: torch.Tensor,
    n_samples: int = 128,
    max_features: int = 64,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute Shapley values for feature contributions to a target logit direction.

    Args:
        model: Trained Spline-CLT.
        x_in: Residual stream inputs, shape (n_layers, n_pos, d_model).
        logit_target: Direction in residual stream space (e.g., W_U[:, token_id]),
            shape (d_model,).
        n_samples: Monte Carlo samples.
        max_features: Max active features to attribute.

    Returns:
        Dict with active_features, activation_values, shapley_values (contribution
        to target logit from each feature).
    """
    n_layers, n_pos, d_model = x_in.shape
    if seed is not None:
        seed_everything(seed)

    baseline_acts = model.encode(x_in)
    sparse = baseline_acts.to_sparse().coalesce()
    layer_idx, pos_idx, feat_idx = sparse.indices()
    act_values = sparse.values()
    n_active = len(act_values)

    if n_active == 0:
        return {
            "active_features": torch.zeros(0, 3, dtype=torch.long, device=x_in.device),
            "activation_values": torch.zeros(0, device=x_in.device),
            "shapley_values": torch.zeros(0, device=x_in.device),
        }

    if n_active > max_features:
        top_k = act_values.abs().topk(max_features).indices
        layer_idx = layer_idx[top_k]
        pos_idx = pos_idx[top_k]
        feat_idx = feat_idx[top_k]
        act_values = act_values[top_k]
        n_active = max_features

    logit_target = logit_target.to(x_in.device, dtype=torch.float32)

    def logit_value_fn(included_mask: torch.Tensor) -> float:
        masked_acts = baseline_acts.clone()
        for i in range(n_active):
            if not included_mask[i]:
                masked_acts[int(layer_idx[i]), int(pos_idx[i]), int(feat_idx[i])] = 0.0
        y_hat = model.decode_dense(masked_acts, input_acts=x_in)
        # Project final-layer output onto logit direction at last position
        return (y_hat[-1, -1].float() @ logit_target).item()

    shapley_vals = torch.zeros(n_active, dtype=torch.float32, device=x_in.device)
    n_perm = n_samples // 2

    for _ in range(n_perm):
        perm = torch.randperm(n_active, device=x_in.device)
        for use_flip in [False, True]:
            order = perm.flip(0) if use_flip else perm
            prev_mask = torch.zeros(n_active, dtype=torch.bool, device=x_in.device)
            prev_val = logit_value_fn(prev_mask)
            for i in order:
                curr_mask = prev_mask.clone()
                curr_mask[i] = True
                curr_val = logit_value_fn(curr_mask)
                shapley_vals[i] += curr_val - prev_val
                prev_mask = curr_mask
                prev_val = curr_val

    shapley_vals /= n_perm * 2

    return {
        "active_features": torch.stack([layer_idx, pos_idx, feat_idx], dim=1),
        "activation_values": act_values,
        "shapley_values": shapley_vals,
    }
