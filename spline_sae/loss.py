"""Losses for Spline-SAE / sparse transcoder training.

Primary objective (shared by linear and KAN arms):

    L = NMSE(y_hat, y) + λ_s * mean(tanh(c * ||W_dec_i|| * a_i))

where ``y_hat = decode(encode(x))``. SAE mode uses ``y = x``; transcoder mode
uses a separate target activation (e.g. resid_pre → attn_out).

KAN diagnostics / optional pressures:

    recon_gap = NMSE(ŷ_base_only, y) − NMSE(ŷ_full, y)
        >0 means splines improve recon over the SiLU/base path alone.

    If lambda_frac_hinge > 0:
        L += λ_frac * relu(f_target − spline_contribution_frac)

    freeze_base_after (in train.py) freezes base_weight so further recon gains
    must come through the spline path — that is the practical "gap" curriculum.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spline_sae.model import SplineSAE

_EPS = 1e-8


def nmse(y_hat: torch.Tensor, y: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Normalized MSE: ||y_hat - y||^2 / ||y||^2."""
    yf = y_hat.float()
    yt = y.float()
    return ((yf - yt) ** 2).sum() / yt.pow(2).sum().clamp_min(eps)


def sparsity_loss(
    activations: torch.Tensor,
    feature_scales: torch.Tensor,
    lambda_sparsity: float,
    c_sparsity: float = 1.0,
) -> torch.Tensor:
    """Feature-scale weighted tanh sparsity (CLT/SAE style).

    ``feature_scales`` is ``||W_dec_i||`` for linear decoders, or the MLP first
    layer column norms for pure MLP decoders.
    """
    scales = feature_scales.float()
    a = activations.float()
    per_feat = torch.tanh(c_sparsity * scales * a).mean(dim=tuple(range(a.ndim - 1)))
    return lambda_sparsity * per_feat.sum()


def _diffable_spline_frac(model: SplineSAE, x: torch.Tensor, max_tokens: int = 2048) -> torch.Tensor:
    """Differentiable ‖spline‖ / ‖base+spline‖ (KAN only)."""
    kl = model.encoder.kan_linear  # type: ignore[attr-defined]
    flat = x.reshape(-1, model.d_model)
    if flat.shape[0] > max_tokens:
        flat = flat[:max_tokens]
    xb = flat.float()
    base = F.linear(kl.base_activation(xb), kl.base_weight)
    spline = F.linear(
        kl.b_splines(xb).view(xb.size(0), -1),
        kl.scaled_spline_weight.view(kl.out_features, -1),
    )
    denom = (base + spline).norm().clamp_min(_EPS)
    return spline.norm() / denom


@torch.no_grad()
def recon_gap_metric(
    model: SplineSAE,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
) -> dict[str, float]:
    """NMSE full vs base-only against target ``y`` (defaults to ``x``)."""
    target = x if y is None else y
    y_full, _, _ = model(x)
    nmse_full = float(nmse(y_full, target).item())
    out: dict[str, float] = {"nmse_full": nmse_full}
    if model.encoder_type != "kan":
        out["nmse_base_only"] = nmse_full
        out["recon_gap"] = 0.0
        return out
    a_base, _ = model.encode_base_only(x)
    y_base = model.decode(a_base)
    nmse_base = float(nmse(y_base, target).item())
    out["nmse_base_only"] = nmse_base
    out["recon_gap"] = nmse_base - nmse_full
    return out


def compute_sae_losses(
    model: SplineSAE,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    lambda_sparsity: float = 1e-3,
    c_sparsity: float = 1.0,
    lambda_nl_gap: float = 0.0,
    lambda_frac_hinge: float = 0.0,
    frac_target: float = 0.35,
    compute_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Forward + losses. Encode ``x``; reconstruct ``y`` (defaults to ``x`` for SAE)."""
    target = x if y is None else y
    y_hat, acts, _pre = model(x)
    l_recon = nmse(y_hat, target)
    l_sparse = sparsity_loss(acts, model.feature_scales(), lambda_sparsity, c_sparsity)

    l_extra = torch.zeros((), device=x.device, dtype=torch.float32)
    frac_t: torch.Tensor | None = None
    if model.encoder_type == "kan" and lambda_frac_hinge > 0.0:
        frac_t = _diffable_spline_frac(model, x)
        l_extra = l_extra + lambda_frac_hinge * torch.relu(
            torch.as_tensor(frac_target, device=x.device) - frac_t
        )

    if lambda_nl_gap > 0.0 and model.encoder_type == "kan":
        with torch.no_grad():
            a_base, _ = model.encode_base_only(x)
            y_base = model.decode(a_base)
            nmse_base = nmse(y_base, target)
        gap_t = (nmse_base - l_recon).clamp_min(0.0)
        l_extra = l_extra - lambda_nl_gap * gap_t

    l_total = l_recon + l_sparse + l_extra

    if not compute_metrics:
        return l_total, {}

    with torch.no_grad():
        l0 = (acts > 0).float().sum(dim=-1).mean().item()
        metrics = {
            "loss/total": float(l_total.item()),
            "loss/nmse": float(l_recon.item()),
            "loss/sparsity": float(l_sparse.item()),
            "loss/extra": float(l_extra.item()),
            "stats/l0": float(l0),
            "stats/explained_variance": float(
                1.0 - ((y_hat.float() - target.float()).pow(2).sum()
                       / target.float().pow(2).sum().clamp_min(_EPS)).item()
            ),
        }
        if model.encoder_type == "kan":
            metrics["stats/spline_contribution_frac"] = model.spline_contribution_fraction(x)
            gap_stats = recon_gap_metric(model, x, target)
            metrics["stats/nmse_base_only"] = gap_stats["nmse_base_only"]
            metrics["stats/recon_gap"] = gap_stats["recon_gap"]
            if model.activation_name == "base_jump":
                metrics["stats/act_frac"] = model.activation_spline_fraction(x)
            if frac_t is not None:
                metrics["stats/frac_hinge_value"] = float(frac_t.detach().item())
    return l_total, metrics
