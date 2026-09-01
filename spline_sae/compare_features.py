"""Compare linear vs Spline-SAE/transcoder features at matched L0.

Identity is decoder direction (circuit-tracer ontology): unit W_dec cosine.
Reports:
  - greedy mutual nearest-neighbor matches above a cosine threshold
  - unmatched high-energy features on each side
  - per-matched-pair activation correlation on a shared holdout
  - recon-energy attribution explaining NMSE gap (||W_dec_i|| * mean a_i)

Usage:
  python -m spline_sae.compare_features \\
    --linear-dir .../probe_gemma_layer_tc_l24_linear_basejump_ctrl \\
    --kan-dir .../probe_gemma_layer_tc_l24_kan_basejump \\
    --output-dir .../feature_compare_layer_tc_l24
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from spline_sae.loss import nmse
from spline_sae.model import SplineSAE
from spline_sae.train import _activation_batches, _load_lm, _text_stream


def _load_sae(ckpt_dir: Path, device: str) -> tuple[SplineSAE, dict[str, Any], dict[str, Any]]:
    cfg_path = ckpt_dir / "resolved_config.json"
    cfg_raw = json.loads(cfg_path.read_text())
    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    sae = SplineSAE(
        d_model=int(cfg_raw["d_model"]),
        d_sae=int(cfg_raw["d_sae"]),
        encoder_type=cfg_raw["encoder_type"],
        activation=cfg_raw.get("activation", "jumprelu"),
        decoder_type=cfg_raw.get("decoder_type", "linear"),
        decoder_hidden=(cfg_raw.get("decoder_hidden") or None),
        decoder_bot=int(cfg_raw.get("decoder_bot", 512)),
        topk_k=int(cfg_raw.get("topk_k", 32)),
        threshold_init=float(cfg_raw.get("threshold_init", 0.01)),
        jumprelu_bandwidth=float(cfg_raw.get("jumprelu_bandwidth", 0.001)),
        grid_size=int(cfg_raw.get("grid_size", 5)),
        spline_order=int(cfg_raw.get("spline_order", 3)),
        scale_base=float(cfg_raw.get("scale_base", 1.0)),
        scale_spline=float(cfg_raw.get("scale_spline", 1.0)),
    )
    sae.load_state_dict(state)
    sae.to(device=device, dtype=torch.float32)
    sae.eval()
    return sae, cfg_raw, ckpt.get("eval", {})


@torch.no_grad()
def _decoder_unit(sae: SplineSAE) -> torch.Tensor:
    w = sae.W_dec.float()
    return F.normalize(w, dim=-1)


@torch.no_grad()
def match_decoders(
    w_lin: torch.Tensor,
    w_kan: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    """Greedy mutual nearest neighbors on decoder cosine."""
    # (d_lin, d_kan)
    sim = w_lin @ w_kan.T
    best_for_lin = sim.argmax(dim=1)
    best_for_kan = sim.argmax(dim=0)
    lin_idx = []
    kan_idx = []
    cos = []
    for i in range(sim.shape[0]):
        j = int(best_for_lin[i].item())
        if int(best_for_kan[j].item()) != i:
            continue
        c = float(sim[i, j].item())
        if c < threshold:
            continue
        lin_idx.append(i)
        kan_idx.append(j)
        cos.append(c)
    matched_lin = set(lin_idx)
    matched_kan = set(kan_idx)
    return {
        "n_matched": len(lin_idx),
        "threshold": threshold,
        "mean_cosine": float(sum(cos) / len(cos)) if cos else float("nan"),
        "median_cosine": float(sorted(cos)[len(cos) // 2]) if cos else float("nan"),
        "lin_idx": lin_idx,
        "kan_idx": kan_idx,
        "cosine": cos,
        "unmatched_lin": [i for i in range(sim.shape[0]) if i not in matched_lin],
        "unmatched_kan": [j for j in range(sim.shape[1]) if j not in matched_kan],
        "max_sim_lin": sim.max(dim=1).values.cpu(),
        "max_sim_kan": sim.max(dim=0).values.cpu(),
    }


@torch.no_grad()
def feature_energy(sae: SplineSAE, acts: torch.Tensor) -> torch.Tensor:
    """Per-feature recon proxy: ||W_dec_i|| * mean_token(a_i)."""
    dec_norm = sae.W_dec.float().norm(dim=-1)
    mean_a = acts.float().mean(dim=0)
    return dec_norm * mean_a


@torch.no_grad()
def collect_acts(
    sae: SplineSAE,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (acts [N,d], y_hat error energy contribution prep, nmse)."""
    act_chunks: list[torch.Tensor] = []
    y_hats: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in batches:
        y_hat, a, _ = sae(x)
        act_chunks.append(a.detach())
        y_hats.append(y_hat.detach())
        ys.append(y.detach())
    acts = torch.cat(act_chunks, dim=0)
    y_hat = torch.cat(y_hats, dim=0)
    y = torch.cat(ys, dim=0)
    return acts, y_hat, float(nmse(y_hat, y).item())


@torch.no_grad()
def nmse_keep_features(
    sae: SplineSAE,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    keep: torch.Tensor,
) -> float:
    """NMSE when only ``keep`` features (bool [d_sae]) are allowed to fire."""
    errs = []
    norms = []
    for x, y in batches:
        y_hat, a, _ = sae(x)
        a = a * keep.to(device=a.device, dtype=a.dtype)
        y_hat = sae.decode(a)
        diff = (y_hat.float() - y.float()).pow(2).sum()
        denom = y.float().pow(2).sum().clamp_min(1e-8)
        errs.append(diff)
        norms.append(denom)
    return float((sum(errs) / sum(norms)).item())


def main() -> None:
    p = argparse.ArgumentParser(description="Compare linear vs spline SAE features")
    p.add_argument("--linear-dir", type=str, required=True)
    p.add_argument("--kan-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--match-threshold", type=float, default=0.7)
    p.add_argument("--n-tokens", type=int, default=16384)
    p.add_argument("--batch-tokens", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    lin, lin_cfg, lin_ev = _load_sae(Path(args.linear_dir), device)
    kan, kan_cfg, kan_ev = _load_sae(Path(args.kan_dir), device)
    assert lin_cfg["hook_name"] == kan_cfg["hook_name"]
    assert lin_cfg.get("target_hook_name", "") == kan_cfg.get("target_hook_name", "")

    w_lin = _decoder_unit(lin)
    w_kan = _decoder_unit(kan)
    match = match_decoders(w_lin, w_kan, args.match_threshold)

    # Shared activation stream from kan resolved config (same hooks / delta).
    model_name = kan_cfg["model_name"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        kan_cfg.get("dtype", "bfloat16")
    ]
    print(f"[compare] loading LM {model_name}", flush=True)
    lm = _load_lm(model_name, device=device, dtype=dtype)
    texts = _text_stream(
        kan_cfg.get("dataset", "monology/pile-uncopyrighted"),
        kan_cfg.get("dataset_config"),
        kan_cfg.get("text_column", "text"),
        args.seed,
    )
    act_iter = _activation_batches(
        lm,
        kan_cfg["hook_name"],
        texts,
        int(kan_cfg.get("seq_len", 1024)),
        args.batch_tokens,
        device,
        target_hook_name=kan_cfg.get("target_hook_name", "") or "",
        reconstruct_delta=bool(kan_cfg.get("reconstruct_delta", False)),
    )
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    n = 0
    while n < args.n_tokens:
        x, y = next(act_iter)
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        batches.append((x, y))
        n += x.shape[0]
    print(f"[compare] holdout tokens={n} batches={len(batches)}", flush=True)

    acts_lin, _, nmse_lin = collect_acts(lin, batches)
    acts_kan, _, nmse_kan = collect_acts(kan, batches)
    e_lin = feature_energy(lin, acts_lin)
    e_kan = feature_energy(kan, acts_kan)

    # Matched activation correlations
    corrs = []
    for i, j in zip(match["lin_idx"], match["kan_idx"]):
        a = acts_lin[:, i]
        b = acts_kan[:, j]
        if a.std() < 1e-8 or b.std() < 1e-8:
            corrs.append(float("nan"))
            continue
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()
        corrs.append(float((a * b).mean().item()))

    # Energy partitions
    matched_lin_energy = float(e_lin[match["lin_idx"]].sum().item()) if match["lin_idx"] else 0.0
    matched_kan_energy = float(e_kan[match["kan_idx"]].sum().item()) if match["kan_idx"] else 0.0
    uniq_lin_energy = float(e_lin[match["unmatched_lin"]].sum().item()) if match["unmatched_lin"] else 0.0
    uniq_kan_energy = float(e_kan[match["unmatched_kan"]].sum().item()) if match["unmatched_kan"] else 0.0
    tot_lin = float(e_lin.sum().item()) + 1e-12
    tot_kan = float(e_kan.sum().item()) + 1e-12

    # Top unmatched by energy
    def top_unmatched(energy: torch.Tensor, idxs: list[int], k: int = 20) -> list[dict[str, float]]:
        if not idxs:
            return []
        sub = energy[idxs]
        order = torch.argsort(sub, descending=True)[:k]
        return [
            {"feat": int(idxs[int(o)]), "energy": float(sub[int(o)].item())}
            for o in order
        ]

    # NMSE with only matched / only unique features
    keep_lin_matched = torch.zeros(lin.d_sae, dtype=torch.bool)
    keep_lin_matched[match["lin_idx"]] = True
    keep_kan_matched = torch.zeros(kan.d_sae, dtype=torch.bool)
    keep_kan_matched[match["kan_idx"]] = True
    keep_lin_uniq = ~keep_lin_matched
    keep_kan_uniq = ~keep_kan_matched

    print("[compare] ablation NMSE (matched vs unique)...", flush=True)
    nmse_lin_matched = nmse_keep_features(lin, batches, keep_lin_matched)
    nmse_kan_matched = nmse_keep_features(kan, batches, keep_kan_matched)
    nmse_lin_uniq = nmse_keep_features(lin, batches, keep_lin_uniq)
    nmse_kan_uniq = nmse_keep_features(kan, batches, keep_kan_uniq)

    finite_corrs = [c for c in corrs if c == c]
    summary = {
        "linear_dir": args.linear_dir,
        "kan_dir": args.kan_dir,
        "hook_name": kan_cfg["hook_name"],
        "target_hook_name": kan_cfg.get("target_hook_name", ""),
        "reconstruct_delta": bool(kan_cfg.get("reconstruct_delta", False)),
        "n_tokens": n,
        "match_threshold": args.match_threshold,
        "ckpt_eval": {"linear": lin_ev, "kan": kan_ev},
        "holdout_nmse": {"linear": nmse_lin, "kan": nmse_kan, "delta": nmse_kan - nmse_lin},
        "decoder_match": {
            "n_matched": match["n_matched"],
            "frac_of_dict": match["n_matched"] / lin.d_sae,
            "mean_cosine": match["mean_cosine"],
            "median_cosine": match["median_cosine"],
            "frac_lin_with_any_nn_ge_thresh": float(
                (match["max_sim_lin"] >= args.match_threshold).float().mean().item()
            ),
            "frac_kan_with_any_nn_ge_thresh": float(
                (match["max_sim_kan"] >= args.match_threshold).float().mean().item()
            ),
            "mean_best_cosine_lin": float(match["max_sim_lin"].mean().item()),
            "mean_best_cosine_kan": float(match["max_sim_kan"].mean().item()),
        },
        "activation_corr_matched": {
            "mean": float(sum(finite_corrs) / len(finite_corrs)) if finite_corrs else float("nan"),
            "frac_finite": len(finite_corrs) / max(1, len(corrs)),
            "n": len(corrs),
        },
        "energy_fraction": {
            "linear_matched": matched_lin_energy / tot_lin,
            "linear_unique": uniq_lin_energy / tot_lin,
            "kan_matched": matched_kan_energy / tot_kan,
            "kan_unique": uniq_kan_energy / tot_kan,
        },
        "nmse_ablation": {
            "linear_matched_only": nmse_lin_matched,
            "linear_unique_only": nmse_lin_uniq,
            "kan_matched_only": nmse_kan_matched,
            "kan_unique_only": nmse_kan_uniq,
            "note": "Higher NMSE = worse; matched_only uses only MNN-matched features.",
        },
        "top_unmatched_linear": top_unmatched(e_lin, match["unmatched_lin"]),
        "top_unmatched_kan": top_unmatched(e_kan, match["unmatched_kan"]),
        "l0_holdout": {
            "linear": float((acts_lin > 0).float().sum(dim=-1).mean().item()),
            "kan": float((acts_kan > 0).float().sum(dim=-1).mean().item()),
        },
    }

    (out / "feature_compare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Save match table (compact)
    pairs = [
        {
            "lin": int(i),
            "kan": int(j),
            "cosine": float(c),
            "corr": float(r) if r == r else None,
            "e_lin": float(e_lin[i].item()),
            "e_kan": float(e_kan[j].item()),
        }
        for i, j, c, r in zip(match["lin_idx"], match["kan_idx"], match["cosine"], corrs)
    ]
    pairs.sort(key=lambda r: -r["cosine"])
    (out / "matched_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pairs))

    print(json.dumps({k: summary[k] for k in [
        "holdout_nmse", "decoder_match", "activation_corr_matched",
        "energy_fraction", "nmse_ablation", "l0_holdout",
    ]}, indent=2))
    print(f"[compare] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
