"""Loss functions for Spline-CLT training.

Training objective:
    L_total = L_NMSE + L_sparsity [+ L_kan_reg]

Where:
    L_NMSE = Σ_l ||y_hat^l - y^l||^2 / Σ_l ||y^l||^2   (scale-free reconstruction)
    L_sparsity = λ Σ tanh(c · ||W_dec_i|| · a_i)       (sparsity regularization)

Anthropic's CLT uses the unnormalized L_MSE = Σ_l ||y_hat^l - y^l||^2. We divide
by Σ_l ||y^l||^2 so that a single λ transfers across base models. The unnormalized
form makes the reconstruction gradient scale with the base model's activation
magnitude, which varies ~33x across the models we train on (mean(y^2) is 0.75 for
gpt2-large, 7.88 for gpt2-small, 24.7 for qwen3-0.6b). Under a shared λ that made
the sparsity term 15x larger than reconstruction at init for gpt2-large but only
3x for qwen3-0.6b: gpt2-large drove every feature dead (L0 91816 -> 0.9 by step
1856) and never recovered, while qwen3-0.6b trained normally. Normalizing puts
L_NMSE at ~1.0 at the zero-output baseline for every model, so the loss balance
is a property of the objective rather than of the base model's activation scale.

Identical for encoder_type="kan" and "linear" — the spline-vs-linear comparison
must not be confounded by a differing objective.
"""

import torch

from spline_clt.kan_transcoder import KANCrossLayerTranscoder

_EPS = 1e-8


def reconstruction_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = _EPS,
    per_layer: bool = False,
    layer_energy_beta: float = 0.0,
) -> torch.Tensor:
    """Scale-free (normalized) reconstruction loss between predicted and true MLP outputs.

    All variants equal ~1.0 when y_hat == 0 and 0.0 on perfect reconstruction, for
    any base model, so ``lambda_sparsity`` means the same thing under each.

    ``per_layer=False`` (global): Σ(y_hat - y)^2 / Σ y^2 over *all* layers at once.
    This weights each layer by its share of ||y||^2. On GPT-2 small that share is
    59% for layer 2 alone — of which 99.5% is the single massive-activation
    dimension #447 — while layers 3-9 hold 2.6% between them. The optimizer duly
    fits dim 447 and abandons the middle of the network (measured: layer 2 rel_fro
    0.062, layers 5-8 rel_fro 0.86-0.91 with <2 active features per token).

    ``per_layer=True``: weighted mean over layers of each layer's own FVU
    ``Σ_pos,dim (y_hat_l - y_l)^2 / Σ_pos,dim y_l^2``, with weights
    ``w_l ∝ ||y_l||^(2·layer_energy_beta)`` normalized to sum 1:

    * ``layer_energy_beta = 0`` (default): uniform weights — every layer counts
      equally regardless of magnitude. Rescues the middle of the network, but on a
      model with extreme per-layer variance (Qwen3-0.6B spans 0.018 to 378, a
      21000x range) it *under*-weights the few layers that carry most of the energy:
      Qwen layers 2 & 27 hold ~87% of ||y||^2 but get 2/28 of the loss, so the model
      half-abandons them (measured: layer 2 rel_fro 0.987, ||y_hat||/||y|| = 0.066).
    * ``layer_energy_beta = 1``: weights ∝ ||y_l||^2 — algebraically identical to the
      global objective.
    * ``0 < beta < 1`` (e.g. 0.5 → weight ∝ ||y_l||): a middle ground that keeps the
      middle layers meaningfully weighted while restoring weight to the dominant ones.

    Because the weights sum to 1 and each layer's FVU is 1.0 at zero output, the whole
    loss is 1.0 at zero output for any beta — so lambda_sparsity transfers.

    Args:
        y_hat: Predicted outputs, shape (n_layers, n_pos, d_model).
        y_true: True MLP outputs, shape (n_layers, n_pos, d_model).
        eps: Floor on the denominator.
        per_layer: Use per-layer FVU (optionally energy-tempered) instead of global.
        layer_energy_beta: Energy-tempering exponent for the per-layer weights.
            Ignored when ``per_layer`` is False.

    Returns:
        Scalar normalized reconstruction loss.
    """
    yf = y_hat.float()
    y = y_true.float()
    if per_layer:
        num = ((yf - y) ** 2).sum(dim=(1, 2))
        den = (y**2).sum(dim=(1, 2)).clamp_min(eps)  # per-layer ||y_l||^2
        fvu = num / den
        if layer_energy_beta == 0.0:
            return fvu.mean()
        w = den.pow(layer_energy_beta)
        w = w / w.sum().clamp_min(eps)
        return (w * fvu).sum()
    return ((yf - y) ** 2).sum() / (y**2).sum().clamp_min(eps)


def paper_style_reconstruction_sparsity_metrics(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    activations: torch.Tensor,
    eps: float = _EPS,
) -> dict[str, float]:
    """Scalars for comparing runs to CLT reporting (e.g. attribution-graph papers).

    Anthropic quote *normalized mean reconstruction error* and *L0* (features
    active per input token) without always specifying the exact formula in-line.
    We log several unambiguous quantities so you can align with their appendix
    or reproduce their definition once pinned down.

    Args:
        y_hat: Predicted MLP outputs, (n_layers, n_pos, d_model).
        y_true: Ground-truth MLP outputs, same shape.
        activations: Post–JumpReLU features, (n_layers, n_pos, d_transcoder).
            Kept on whatever device it already lives on; the L0 mask is reduced
            in place rather than copied to host.

    Returns:
        Flat dict suitable for tqdm / W&B (all plain floats).
    """
    with torch.no_grad():
        yf = y_hat.detach().float()
        y = y_true.detach().float()
        n_elem = y.numel()
        mse_mean = ((yf - y) ** 2).mean()
        diff = yf - y
        rel_fro = diff.norm() / (y.norm() + eps)
        # NMSE: ratio of mean squared error to mean squared target (scale-invariant).
        mean_y2 = (y**2).mean().clamp_min(eps)
        nmse_mean = mse_mean / mean_y2

        # One bool mask, reduced on-device. Summing a bool tensor yields int64
        # directly, so we never materialize the float32 copy (4x the activation
        # bytes — 3 GB at gpt2-large shapes) that the old CPU path built twice.
        ac = activations.detach()
        active = ac > 0
        n_layers, n_pos, _ = ac.shape
        n_active = active.sum(dtype=torch.int64)
        l0_per_token = n_active / n_pos
        active_per_layer_pos = n_active / (n_layers * n_pos)

        # The aggregate rel_fro is dominated by whichever layer carries the most
        # ||y||^2 (59% is layer 2 on gpt2-small, itself 99.5% dim #447). Report the
        # unweighted per-layer mean and the worst layer so a network whose middle
        # is unmodelled cannot hide behind a good aggregate.
        pl_fvu = ((yf - y) ** 2).sum(dim=(1, 2)) / (y**2).sum(dim=(1, 2)).clamp_min(eps)
        pl_rel_fro = pl_fvu.sqrt()
        pl_l0 = active.sum(dim=(1, 2), dtype=torch.int64) / n_pos

    return {
        "reconstruction/mse_sum": float((mse_mean * n_elem).item()),
        "reconstruction/rel_fro_error": float(rel_fro.item()),
        "reconstruction/nmse_mean": float(nmse_mean.item()),
        "reconstruction/rel_fro_per_layer_mean": float(pl_rel_fro.mean().item()),
        "reconstruction/rel_fro_worst_layer": float(pl_rel_fro.max().item()),
        "stats/l0_active_features_per_token": float(l0_per_token.item()),
        "stats/active_features_per_pos": float(active_per_layer_pos.item()),
        "stats/l0_min_layer": float(pl_l0.min().item()),
    }


def sparsity_loss(
    activations: torch.Tensor,
    decoder_norms: list[torch.Tensor],
    lambda_: float = 0.05,
    c: float = 1.0,
    per_layer_mean: bool = False,
) -> torch.Tensor:
    """Sparsity regularization loss matching Anthropic's formulation.

    L_sparsity = λ Σ tanh(c · ||W_dec_i|| · a_i)

    This encourages features to be sparse (few active per token) while accounting
    for the magnitude of decoder vectors (larger decoders penalized more).

    Args:
        activations: Feature activations, shape (n_layers, n_pos, d_transcoder).
            Should be post-activation (after JumpReLU/ReLU).
        decoder_norms: List of decoder weight norms per layer. Each element has
            shape (d_transcoder,) giving the L2 norm of each feature's decoder.
        lambda_: Sparsity coefficient.
        c: Scaling factor inside tanh.
        per_layer_mean: When True, divide the summed penalty by ``n_layers`` as well,
            so it is a **mean** over layers rather than a sum. Paired with per-layer
            reconstruction normalization this makes λ mean "cost per active feature
            per layer per token" — n_layers-independent, so one λ transfers across
            models of different depth. When False (default), the penalty is the
            per-token sum over all layers (deeper models feel more pressure at fixed λ).

    Returns:
        Scalar sparsity loss.
    """
    total = torch.tensor(0.0, device=activations.device, dtype=torch.float32)

    for layer_id in range(activations.shape[0]):
        # decoder_norms[layer_id]: (d_transcoder,)
        # activations[layer_id]: (n_pos, d_transcoder)
        weighted = c * decoder_norms[layer_id].unsqueeze(0) * activations[layer_id].float()
        total = total + torch.tanh(weighted).sum()

    total = total / activations.shape[1]  # normalize by n_pos
    if per_layer_mean:
        total = total / activations.shape[0]  # mean over layers
    return lambda_ * total


def compute_decoder_norms(model: KANCrossLayerTranscoder) -> list[torch.Tensor]:
    """Compute L2 norms of decoder vectors for each feature.

    For cross-layer decoders, uses the maximum norm across target layers.

    Args:
        model: The Spline-CLT model.

    Returns:
        List of tensors, one per layer, each of shape (d_transcoder,).
    """
    norms = []
    for layer_id in range(model.n_layers):
        # W_dec[layer_id]: (d_transcoder, n_remaining_layers, d_model)
        w = model.W_dec[layer_id]
        # Norm across d_model, max across target layers
        layer_norms = w.norm(dim=-1).max(dim=-1).values  # (d_transcoder,)
        norms.append(layer_norms.detach())
    return norms


def compute_losses(
    activations: torch.Tensor,
    y_hat: torch.Tensor,
    dec_norms: list[torch.Tensor],
    model: KANCrossLayerTranscoder,
    y_true: torch.Tensor,
    lambda_sparsity: float = 0.05,
    c_sparsity: float = 1.0,
    lambda_kan_reg: float = 0.001,
    compute_metrics: bool = True,
    recon_per_layer: bool = False,
    sparsity_per_layer_mean: bool = False,
    recon_layer_energy_beta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute losses from pre-computed activations and reconstructions.

    Used by the FSDP training path where encode/decode_dense/W_dec are all
    called inside model_for_train.forward() so FSDP's unshard hooks fire.
    KAN encoder spline weights are in FSDP's ignored_modules (never sharded)
    and can be accessed safely outside the FSDP forward context.

    Args:
        activations: Post-activation features, (n_layers, n_pos, d_transcoder).
        y_hat: Reconstructed MLP outputs, (n_layers, n_pos, d_model).
        dec_norms: Decoder norms from forward(), one (d_transcoder,) per layer.
        model: Unwrapped KANCrossLayerTranscoder for encoder_type / encoders.
        y_true: True MLP outputs, (n_layers, n_pos, d_model).
        lambda_sparsity: Sparsity loss coefficient.
        c_sparsity: Sparsity scaling factor.
        lambda_kan_reg: L1 coefficient on spline weights (KAN encoder only).
        compute_metrics: Populate the diagnostic metrics dict. Each metric forces
            a device sync, so callers that only need the loss (non-logging steps)
            should pass False and get ``{}`` back.

    Returns:
        ``(l_total, metrics)``. ``l_total = l_recon + l_sparse + l_kan_reg`` — note
        the third term, which is reported as ``loss/kan_regularization``.
    """
    l_recon = reconstruction_loss(
        y_hat, y_true, per_layer=recon_per_layer,
        layer_energy_beta=recon_layer_energy_beta,
    )
    l_sparse = sparsity_loss(
        activations, dec_norms, lambda_sparsity, c_sparsity,
        per_layer_mean=sparsity_per_layer_mean,
    )

    # KAN regularization: L1 on spline weights (KAN encoder only).
    # We do NOT use KANLinear.regularization_loss() because its entropy branch
    # computes p * log(p) which evaluates to 0 * -inf = NaN when spline_weight is all
    # zeros (which happens after update_grid when the old grid didn't cover the data).
    # Even with regularize_entropy=0.0, Python's 0.0 * nan = nan in IEEE 754.
    # Linear encoder has no spline weights, so this term is skipped.
    l_kan_reg = torch.zeros((), device=activations.device, dtype=torch.float32)
    if model.encoder_type == "kan" and lambda_kan_reg > 0.0:
        for encoder in model.encoders:
            l_kan_reg = l_kan_reg + encoder.kan_linear.spline_weight.abs().mean()
        l_kan_reg = lambda_kan_reg * l_kan_reg

    l_total = l_recon + l_sparse + l_kan_reg

    if not compute_metrics:
        return l_total, {}

    metrics = {
        "loss/total": l_total.item(),
        "loss/reconstruction": l_recon.item(),
        "loss/sparsity": l_sparse.item(),
        "loss/kan_regularization": l_kan_reg.item(),
        **paper_style_reconstruction_sparsity_metrics(y_hat, y_true, activations),
    }
    return l_total, metrics


def total_loss(
    model: KANCrossLayerTranscoder,
    x_in: torch.Tensor,
    y_true: torch.Tensor,
    lambda_sparsity: float = 0.05,
    c_sparsity: float = 1.0,
    lambda_kan_reg: float = 0.001,
    compute_metrics: bool = True,
    recon_per_layer: bool = False,
    sparsity_per_layer_mean: bool = False,
    recon_layer_energy_beta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total training loss (non-FSDP path).

    Calls encode/decode_dense/W_dec directly on the unwrapped model. Use
    compute_losses() with model_for_train(x_in) for the FSDP path instead.

    Args:
        model: The Spline-CLT model (unwrapped or DDP-wrapped).
        x_in: Residual stream inputs, shape (n_layers, n_pos, d_model).
        y_true: True MLP outputs, shape (n_layers, n_pos, d_model).
        lambda_sparsity: Sparsity loss coefficient.
        c_sparsity: Sparsity scaling factor.
        lambda_kan_reg: L1 coefficient on spline weights (KAN encoder only).
        compute_metrics: See :func:`compute_losses`.
    """
    # Forward pass — use dense decode to avoid OOM from materializing
    # (n_active * n_layers * d_model) during early training when sparsity is low
    activations = model.encode(x_in)
    y_hat = model.decode_dense(activations, input_acts=x_in)
    dec_norms = compute_decoder_norms(model)
    return compute_losses(activations, y_hat, dec_norms, model, y_true,
                          lambda_sparsity, c_sparsity, lambda_kan_reg,
                          compute_metrics, recon_per_layer, sparsity_per_layer_mean,
                          recon_layer_energy_beta)
