"""Metric helpers for original-pipeline circuits."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def tokenize_pair(model: Any, clean_prompt: str, corrupted_prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize clean/corrupt prompts; require equal length for patching."""
    clean = model.to_tokens(clean_prompt)
    corrupt = model.to_tokens(corrupted_prompt)
    if int(clean.shape[-1]) != int(corrupt.shape[-1]):
        raise ValueError(
            "clean/corrupted prompts must tokenize to the same length for edge "
            f"patching (got {int(clean.shape[-1])} vs {int(corrupt.shape[-1])})."
        )
    return clean, corrupt


def logit_gap_from_logits(
    logits: torch.Tensor, target_idx: int, foil_idx: int | None
) -> float:
    if logits.ndim == 3:
        last = logits[0, -1]
    elif logits.ndim == 2:
        last = logits[-1]
    else:
        raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}")
    if foil_idx is None:
        return float(last[target_idx].detach().item())
    return float((last[target_idx] - last[foil_idx]).detach().item())


def logit_gap_tensor(
    logits: torch.Tensor, target_idx: int, foil_idx: int | None
) -> torch.Tensor:
    if logits.ndim == 3:
        last = logits[0, -1]
    elif logits.ndim == 2:
        last = logits[-1]
    else:
        raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}")
    if foil_idx is None:
        return last[target_idx]
    return last[target_idx] - last[foil_idx]


def kl_full_vs_circuit(full_logits: torch.Tensor, circuit_logits: torch.Tensor) -> float:
    """KL(full ‖ circuit) on the final-position token distribution.

    Matches the common ACDC reporting convention of comparing the full model's
    next-token distribution to the circuit's.
    """
    if full_logits.ndim == 3:
        full_last = full_logits[0, -1]
        circ_last = circuit_logits[0, -1]
    else:
        full_last = full_logits[-1]
        circ_last = circuit_logits[-1]
    full_log_probs = F.log_softmax(full_last.float(), dim=-1)
    circ_log_probs = F.log_softmax(circ_last.float(), dim=-1)
    full_probs = full_log_probs.exp()
    return float(torch.sum(full_probs * (full_log_probs - circ_log_probs)).item())


def resolve_token_index(model: Any, token: str) -> int:
    """Map a single-token string to a vocab id (first piece if multi-token)."""
    ids = model.to_tokens(token, prepend_bos=False)
    flat = ids.view(-1)
    if flat.numel() < 1:
        raise ValueError(f"Token {token!r} produced no ids.")
    return int(flat[0].item())
