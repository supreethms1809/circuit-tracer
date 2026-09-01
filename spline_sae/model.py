"""Per-layer Sparse Autoencoder with linear or KAN encoder.

Intentionally same-layer only (SAE, not CLT):

    a = act(encoder(x) + b_enc)   # JumpReLU / TopK / BaseJump
    y_hat = decode(a)             # linear, MLP, or linear+MLP residual

Default decoder is linear (steerable feature directions for circuit tracing).
``decoder_type='mlp'|'linear_mlp'`` is an experimental NMSE ablation — nonlinear
decoders break clean per-feature residual directions.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from spline_clt.kan_encoder import KANEncoder
from spline_clt.linear_encoder import LinearEncoder


ActName = Literal["jumprelu", "topk", "relu", "base_jump"]
EncoderName = Literal["kan", "linear"]
DecoderName = Literal["linear", "mlp", "linear_mlp", "kan"]


class JumpReLU(nn.Module):
    """ReLU with a learned positive threshold (log-parametrized).

    Forward is hard ``x * 1[x > θ]``. Backward uses a bandwidth-ε STE on the
    Heaviside so ``θ`` (and the score) receive gradient near the threshold.
    """

    def __init__(self, d_sae: int, threshold_init: float = 0.01, bandwidth: float = 0.001):
        super().__init__()
        if threshold_init <= 0:
            raise ValueError("threshold_init must be > 0")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")
        self.bandwidth = float(bandwidth)
        self.log_threshold = nn.Parameter(
            torch.full((d_sae,), float(torch.log(torch.tensor(threshold_init))))
        )

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def _broadcast_theta(self, like: torch.Tensor) -> torch.Tensor:
        theta = self.threshold
        while theta.ndim < like.ndim:
            theta = theta.view(*([1] * (like.ndim - theta.ndim)), *theta.shape)
        return theta

    def _ste_mask(self, score: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Hard gate forward; soft ramp backward on ``score`` / ``θ``."""
        hard = (score > theta).to(dtype=score.dtype)
        soft = ((score - theta) / self.bandwidth + 0.5).clamp(0.0, 1.0)
        return hard + soft - soft.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self._broadcast_theta(x)
        return self._ste_mask(x, theta) * x

    def gated(self, score: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """BaseJump: mask from ``score``, magnitude from ``relu(value)``."""
        theta = self._broadcast_theta(score)
        return self._ste_mask(score, theta) * F.relu(value)


class TopK(nn.Module):
    """Keep the top-k preactivations per token; zero elsewhere."""

    def __init__(self, k: int):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = min(self.k, x.shape[-1])
        values, indices = torch.topk(x, k=k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, indices, F.relu(values))
        return out


class MLPDecoder(nn.Module):
    """Two-layer MLP: d_sae → d_hidden → d_model."""

    def __init__(self, d_sae: int, d_model: int, d_hidden: int | None = None):
        super().__init__()
        h = int(d_hidden) if d_hidden is not None else int(d_model)
        self.fc1 = nn.Linear(d_sae, h, bias=True)
        self.fc2 = nn.Linear(h, d_model, bias=True)
        nn.init.kaiming_uniform_(self.fc1.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.fc2.weight, a=5**0.5)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(a)))

    def feature_scales(self) -> torch.Tensor:
        """Per-feature scale proxy for sparsity: ‖fc1 column_i‖."""
        return self.fc1.weight.float().norm(dim=0)


class KANDecoder(nn.Module):
    """Bottlenecked KAN decoder matching the encoder's B-spline family.

    Full ``KANLinear(d_sae → d_model)`` is intractable at SAE width (16k×8×d_model).
    Instead:

        h = W_down @ a          # d_sae → d_bot
        y = KANLinear(h)        # d_bot → d_model  (same grid/order as encoder)
    """

    def __init__(
        self,
        d_sae: int,
        d_model: int,
        d_bot: int = 512,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ) -> None:
        super().__init__()
        if d_bot < 1:
            raise ValueError("d_bot must be >= 1")
        self.d_sae = d_sae
        self.d_model = d_model
        self.d_bot = int(d_bot)
        self.down = nn.Linear(d_sae, self.d_bot, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        self.kan = KANEncoder(
            d_model=self.d_bot,
            n_features=d_model,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_base=scale_base,
            scale_spline=scale_spline,
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        h = self.down(a.float())
        return self.kan(h).to(dtype=a.dtype)

    def feature_scales(self) -> torch.Tensor:
        """Per-feature scale proxy: ‖down column_i‖."""
        return self.down.weight.float().norm(dim=0)


class SplineSAE(nn.Module):
    """Single-layer SAE with pluggable encoder, sparsifier, and decoder."""

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        encoder_type: EncoderName = "kan",
        activation: ActName = "jumprelu",
        decoder_type: DecoderName = "linear",
        decoder_hidden: int | None = None,
        topk_k: int = 32,
        threshold_init: float = 0.01,
        jumprelu_bandwidth: float = 0.001,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        decoder_bot: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.encoder_type = encoder_type
        self.activation_name = activation
        self.decoder_type = decoder_type

        if activation == "base_jump" and encoder_type != "kan":
            raise ValueError("activation='base_jump' requires encoder_type='kan'")

        if encoder_type == "kan":
            self.encoder: nn.Module = KANEncoder(
                d_model=d_model,
                n_features=d_sae,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_base=scale_base,
                scale_spline=scale_spline,
            )
        elif encoder_type == "linear":
            self.encoder = LinearEncoder(d_model=d_model, n_features=d_sae)
        else:
            raise ValueError(f"unknown encoder_type={encoder_type!r}")

        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Linear dictionary for linear / linear_mlp; also kept (unused in forward)
        # when decoder is kan/mlp so checkpoints share a W_dec slot if needed.
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)

        self.mlp_dec: MLPDecoder | None = None
        self.kan_dec: KANDecoder | None = None
        if decoder_type in ("mlp", "linear_mlp"):
            self.mlp_dec = MLPDecoder(d_sae, d_model, decoder_hidden)
        elif decoder_type == "kan":
            bot = int(decoder_hidden) if decoder_hidden is not None else int(decoder_bot)
            self.kan_dec = KANDecoder(
                d_sae=d_sae,
                d_model=d_model,
                d_bot=bot,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_base=scale_base,
                scale_spline=scale_spline,
            )
        elif decoder_type != "linear":
            raise ValueError(f"unknown decoder_type={decoder_type!r}")

        if activation in ("jumprelu", "base_jump"):
            self.act: nn.Module = JumpReLU(d_sae, threshold_init, jumprelu_bandwidth)
        elif activation == "topk":
            self.act = TopK(topk_k)
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"unknown activation={activation!r}")

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (activations, preactivations). x: (..., d_model)."""
        if self.activation_name == "base_jump":
            base, spline = self.encoder.forward_split(x)  # type: ignore[attr-defined]
            score = base + self.b_enc
            value = score + spline
            assert isinstance(self.act, JumpReLU)
            return self.act.gated(score, value), value
        pre = self.encoder(x) + self.b_enc
        return self.act(pre), pre

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        if self.decoder_type == "linear":
            return F.linear(a, self.W_dec.T, self.b_dec)
        if self.decoder_type == "kan":
            assert self.kan_dec is not None
            return self.kan_dec(a)
        assert self.mlp_dec is not None
        if self.decoder_type == "mlp":
            return self.mlp_dec(a)
        return F.linear(a, self.W_dec.T, self.b_dec) + self.mlp_dec(a)

    def feature_scales(self) -> torch.Tensor:
        """Per-feature scales for sparsity weighting."""
        if self.decoder_type == "linear":
            return self.W_dec.float().norm(dim=-1)
        if self.decoder_type == "linear_mlp":
            return self.W_dec.float().norm(dim=-1)
        if self.decoder_type == "kan":
            assert self.kan_dec is not None
            return self.kan_dec.feature_scales()
        assert self.mlp_dec is not None
        return self.mlp_dec.feature_scales()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (y_hat, activations, preactivations)."""
        a, pre = self.encode(x)
        return self.decode(a), a, pre

    def encode_base_only(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """KAN base/SiLU path only (spline zeroed). For recon-gap loss."""
        if self.encoder_type != "kan":
            return self.encode(x)
        if self.activation_name == "base_jump":
            base, _spline = self.encoder.forward_split(x)  # type: ignore[attr-defined]
            score = base + self.b_enc
            assert isinstance(self.act, JumpReLU)
            return self.act.gated(score, score), score
        kl = self.encoder.kan_linear  # type: ignore[attr-defined]
        base = F.linear(kl.base_activation(x.float()), kl.base_weight).to(x.dtype)
        pre = base + self.b_enc
        return self.act(pre), pre

    @torch.no_grad()
    def activation_spline_fraction(self, x: torch.Tensor, max_tokens: int = 4096) -> float:
        """‖mask ⊙ relu(spline)‖ / ‖a‖ for BaseJump; NaN otherwise."""
        if self.activation_name != "base_jump":
            return float("nan")
        flat = x.reshape(-1, self.d_model)
        if flat.shape[0] > max_tokens:
            flat = flat[:max_tokens]
        base, spline = self.encoder.forward_split(flat)  # type: ignore[attr-defined]
        score = base + self.b_enc
        value = score + spline
        assert isinstance(self.act, JumpReLU)
        a = self.act.gated(score, value)
        theta = self.act._broadcast_theta(score)
        mask = (score > theta).to(a.dtype)
        num = (mask * F.relu(spline)).norm()
        denom = a.norm()
        if denom <= 0:
            return float("nan")
        return float((num / denom).item())

    @torch.no_grad()
    def spline_contribution_fraction(self, x: torch.Tensor, max_tokens: int = 4096) -> float:
        """‖spline‖ / ‖base+spline‖ for KAN encoders; NaN for linear."""
        if self.encoder_type != "kan":
            return float("nan")
        flat = x.reshape(-1, self.d_model)
        if flat.shape[0] > max_tokens:
            flat = flat[:max_tokens]
        kl = self.encoder.kan_linear  # type: ignore[attr-defined]
        xb = flat.float()
        base = F.linear(kl.base_activation(xb), kl.base_weight)
        spline = F.linear(
            kl.b_splines(xb).view(xb.size(0), -1),
            kl.scaled_spline_weight.view(kl.out_features, -1),
        )
        denom = (base + spline).norm()
        if denom <= 0:
            return float("nan")
        return float((spline.norm() / denom).item())
