#!/usr/bin/env python3
"""CLI for from-scratch spline debug probes (paper_probes/).

Examples:
  conda run -n ct python -m experiments.probes.run_probes all --device cuda
  conda run -n ct python -m experiments.probes.run_probes A --steps 2000
  conda run -n ct python -m experiments.probes.run_probes E --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from experiments.probes import PROBES_ROOT
from experiments.probes.common import (
    ProbeTrainConfig,
    apply_shared_init,
    build_model,
    calibrate_reference,
    ensure_dir,
    eval_metrics,
    load_probe_dataset,
    set_trainable,
    train_arm,
    zero_spline_path_eval,
    restore_spline_path,
)
from experiments.probes.synthetic import (
    feature_recovery_score,
    make_synthetic_nonlinear_dataset,
)
from spline_clt.seed import seed_everything
from spline_clt.training.loss import compute_losses


def _base_cfg(args: argparse.Namespace) -> ProbeTrainConfig:
    return ProbeTrainConfig(
        total_steps=args.steps,
        d_transcoder=args.d_transcoder,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        seed=args.seed,
        data_dir=args.data_dir,
        log_every=max(1, args.steps // 40),
        warmup_steps=max(1, args.steps // 10),
        lambda_sparsity=args.lambda_sparsity,
        lambda_kan_reg=args.lambda_kan_reg,
    )


def run_A(args: argparse.Namespace) -> dict:
    """Phase A: locked-scale linear vs full KAN."""
    out = ensure_dir(Path(PROBES_ROOT) / "A_locked_scale")
    cfg = _base_cfg(args)
    # Calibrate linear as reference, then train both under shared locks.
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_ds, _ = load_probe_dataset(cfg)
    ref = build_model("linear", cfg, device)
    init_info = calibrate_reference(ref, train_ds, cfg)
    bw = float((0.1 * ref.effective_threshold.mean()).clamp_min(1e-4).item())
    ref.activation_function.bandwidth = bw
    init_info["locked_bandwidth"] = bw
    (out / "reference_init.json").write_text(json.dumps(init_info, indent=2) + "\n")

    # Train linear under locked θ/decoder (already calibrated on itself).
    lin_cfg = ProbeTrainConfig(**{**cfg.__dict__})
    lin_cfg.lock_threshold = True
    lin_cfg.lock_decoder = True
    lin = train_arm("linear", lin_cfg, out / "linear", reference_model=ref)

    kan_cfg = ProbeTrainConfig(**{**cfg.__dict__})
    kan_cfg.lock_threshold = True
    kan_cfg.lock_decoder = True
    kan_cfg.lambda_kan_reg = args.lambda_kan_reg
    kan = train_arm("kan", kan_cfg, out / "kan", reference_model=ref)

    summary = {
        "probe": "A_locked_scale",
        "question": "Does the gap survive when θ and W_dec are forced identical?",
        "linear_final_val": lin["summary"]["final_val"],
        "kan_final_val": kan["summary"]["final_val"],
        "kan_base_only_val": kan["summary"]["base_only_val"],
        "init": init_info,
        "gap_rel_fro": (
            kan["summary"]["final_val"].get("reconstruction/rel_fro_error", float("nan"))
            - lin["summary"]["final_val"].get("reconstruction/rel_fro_error", float("nan"))
        ),
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary("A_locked_scale", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_B(args: argparse.Namespace) -> dict:
    """Phase B: linear / SiLU-base / full KAN ladder."""
    out = ensure_dir(Path(PROBES_ROOT) / "B_component_ladder")
    cfg = _base_cfg(args)
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_ds, _ = load_probe_dataset(cfg)
    ref = build_model("linear", cfg, device)
    init_info = calibrate_reference(ref, train_ds, cfg)
    ref.activation_function.bandwidth = float(
        (0.1 * ref.effective_threshold.mean()).clamp_min(1e-4).item()
    )

    results = {}
    for arm in ("linear", "silu_base", "kan"):
        arm_cfg = ProbeTrainConfig(**{**cfg.__dict__})
        arm_cfg.lock_threshold = True
        arm_cfg.lock_decoder = True
        arm_cfg.lambda_kan_reg = 0.0 if arm != "kan" else args.lambda_kan_reg
        results[arm] = train_arm(arm, arm_cfg, out / arm, reference_model=ref)["summary"]

    summary = {
        "probe": "B_component_ladder",
        "question": "Is the gap from SiLU, B-spline path, or their interaction?",
        "init": init_info,
        "arms": {
            arm: {
                "rel_fro": results[arm]["final_val"].get("reconstruction/rel_fro_error"),
                "act_lp": results[arm]["final_val"].get("stats/active_features_per_pos"),
                "nmse": results[arm]["final_val"].get("reconstruction/nmse_mean"),
            }
            for arm in results
        },
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary("B_component_ladder", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_C(args: argparse.Namespace) -> dict:
    """Phase C: freeze encoder/decoder; base-only eval on a short KAN train."""
    out = ensure_dir(Path(PROBES_ROOT) / "C_freeze_path")
    cfg = _base_cfg(args)
    # Shorter default for freeze variants if user didn't override heavily
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = load_probe_dataset(cfg)
    ref = build_model("linear", cfg, device)
    calibrate_reference(ref, train_ds, cfg)
    ref.activation_function.bandwidth = float(
        (0.1 * ref.effective_threshold.mean()).clamp_min(1e-4).item()
    )

    variants = {
        "kan_full": {"freeze_encoder": False, "freeze_decoder": False, "lock_decoder": True},
        "encoder_only": {"freeze_encoder": False, "freeze_decoder": True, "lock_decoder": True},
        "decoder_only": {"freeze_encoder": True, "freeze_decoder": False, "lock_decoder": False},
    }
    results = {}
    for name, flags in variants.items():
        arm_cfg = ProbeTrainConfig(**{**cfg.__dict__})
        arm_cfg.lock_threshold = True
        arm_cfg.freeze_encoder = flags["freeze_encoder"]
        arm_cfg.freeze_decoder = flags["freeze_decoder"]
        arm_cfg.lock_decoder = flags["lock_decoder"]
        arm_cfg.lambda_kan_reg = 0.0
        results[name] = train_arm("kan", arm_cfg, out / name, reference_model=ref)["summary"]

    # Base-only already recorded on kan_full via train_arm
    summary = {
        "probe": "C_freeze_path",
        "question": "Encoder vs decoder vs spline-path blame?",
        "variants": {
            k: {
                "rel_fro": v["final_val"].get("reconstruction/rel_fro_error"),
                "act_lp": v["final_val"].get("stats/active_features_per_pos"),
                "base_only_rel_fro": (v.get("base_only_val") or {}).get(
                    "reconstruction/rel_fro_error"
                ),
            }
            for k, v in results.items()
        },
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary("C_freeze_path", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_D(args: argparse.Namespace) -> dict:
    """Phase D: extract grad-flow series from A records (or re-run short)."""
    out = ensure_dir(Path(PROBES_ROOT) / "D_grad_flow")
    a_root = Path(PROBES_ROOT) / "A_locked_scale"
    series = {}
    for arm in ("linear", "kan"):
        rec = a_root / arm / "training_records.jsonl"
        if not rec.exists():
            print(f"[D] missing {rec}; running A first", flush=True)
            run_A(args)
            break
    for arm in ("linear", "kan"):
        rec = a_root / arm / "training_records.jsonl"
        rows = []
        with rec.open() as f:
            for ln in f:
                r = json.loads(ln)
                rows.append(
                    {
                        "step": r["step"],
                        "grad/base_weight": r.get("grad/base_weight"),
                        "grad/spline_weight": r.get("grad/spline_weight"),
                        "grad/encoder_other": r.get("grad/encoder_other"),
                        "grad/decoder": r.get("grad/decoder"),
                        "grad/threshold": r.get("grad/threshold"),
                        "stats/spline_contribution_frac": r.get(
                            "stats/spline_contribution_frac"
                        ),
                        "preact/frac_above_theta": r.get("preact/frac_above_theta"),
                        "reconstruction/rel_fro_error": r.get(
                            "reconstruction/rel_fro_error"
                        ),
                    }
                )
        series[arm] = rows
        (out / f"{arm}_grad_series.json").write_text(json.dumps(rows, indent=2) + "\n")

    # Ratio summary at mid/late
    def _near(rows, step):
        return min(rows, key=lambda r: abs(r["step"] - step))

    summary = {"probe": "D_grad_flow", "question": "Where do gradients die?", "snapshots": {}}
    for arm, rows in series.items():
        if not rows:
            continue
        mid = _near(rows, args.steps // 2)
        late = rows[-1]
        summary["snapshots"][arm] = {"mid": mid, "late": late}
        if arm == "kan" and mid.get("grad/base_weight"):
            bw = mid["grad/base_weight"] or 0.0
            sw = mid["grad/spline_weight"] or 0.0
            summary["snapshots"][arm]["spline_over_base_grad_mid"] = (
                sw / bw if bw > 0 else None
            )

    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary("D_grad_flow", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_E(args: argparse.Namespace) -> dict:
    """Phase E: synthetic nonlinear recovery."""
    out = ensure_dir(Path(PROBES_ROOT) / "E_synthetic")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    d_model = 64
    n_layers = 4
    d_trans = 128
    ds = make_synthetic_nonlinear_dataset(
        n_samples=args.max_samples,
        n_layers=n_layers,
        seq_len=16,
        d_model=d_model,
        n_features=8,
        seed=args.seed,
    )
    # Split
    n_train = int(0.8 * len(ds))
    train_ds = type(ds)(ds.mlp_inputs[:n_train], ds.mlp_outputs[:n_train])
    val_ds = type(ds)(ds.mlp_inputs[n_train:], ds.mlp_outputs[n_train:])

    from experiments.probes.common import ProbeTrainConfig, batch_to_device, lr_at_step
    from torch.utils.data import DataLoader

    results = {}
    for arm in ("linear", "kan"):
        seed_everything(args.seed)
        model = build_model(
            arm,
            ProbeTrainConfig(
                n_layers=n_layers,
                d_model=d_model,
                d_transcoder=d_trans,
                jumprelu_bandwidth=0.05,
            ),
            device,
        )
        # Simple constant threshold for toy
        with torch.no_grad():
            model.activation_function.threshold.fill_(
                float(torch.tensor(0.05).log())
            )
        # Do NOT lock decoder on synthetic — both arms learn freely; same seed init differ by arch
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
        loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True, drop_last=True)
        steps = max(args.steps, 500)
        data_iter = iter(loader)
        model.train()
        for step in range(steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            x, y = batch_to_device(batch, device, torch.float32)
            opt.zero_grad(set_to_none=True)
            acts, y_hat, dec_norms = model(x)
            loss, metrics = compute_losses(
                acts,
                y_hat,
                dec_norms,
                model,
                y,
                lambda_sparsity=1e-4,
                c_sparsity=1.0,
                lambda_kan_reg=0.0,
                compute_metrics=(step % 100 == 0),
                recon_per_layer=True,
                sparsity_per_layer_mean=True,
            )
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(
                    f"[E:{arm}] step {step} loss={float(loss.detach()):.4f} "
                    f"rel_fro={metrics.get('reconstruction/rel_fro_error', float('nan')):.3f}",
                    flush=True,
                )
        val = eval_metrics(model, val_ds, device, torch.float32, batch_size=16)
        recovery = feature_recovery_score(model, val_ds, device=device)
        ckpt = ensure_dir(out / arm / "checkpoint")
        model.to_safetensors(str(ckpt))
        results[arm] = {"final_val": val, "recovery": recovery, "checkpoint": str(ckpt)}

    summary = {
        "probe": "E_synthetic",
        "question": "Can Spline-CLT recover known nonlinear features when linear cannot?",
        "linear_unexplained_frac": results["linear"]["recovery"]["linear_unexplained_frac_mean"],
        "arms": {
            arm: {
                "rel_fro": results[arm]["final_val"].get("reconstruction/rel_fro_error"),
                "recovery_rel_fro": results[arm]["recovery"]["rel_fro_mean"],
            }
            for arm in results
        },
        "interpretation_hint": (
            "If kan rel_fro << linear rel_fro on this toy, architecture works and GPT-2 regime is the issue. "
            "If both fail, training/JumpReLU pipeline is broken for nonlinear features. "
            "If linear matches kan, toy is too weak / still linearly separable."
        ),
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary("E_synthetic", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _write_summary(name: str, payload: dict) -> None:
    path = ensure_dir(Path(PROBES_ROOT) / "summaries") / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _lam_tag(lam: float) -> str:
    """Filesystem-safe λ tag, e.g. 0.005 → lam5em3."""
    return "lam" + f"{lam:.0e}".replace("e-0", "em").replace("e-", "em").replace("e+0", "e").replace(".", "p")


def run_F(args: argparse.Namespace) -> dict:
    """Phase F: capacity + L0 retarget (unlocked decoder/θ).

    Claim test: Spline/SiLU at smaller d_t can match linear at larger d_t once
    L0 is healthy. Each SLURM job runs one (arm, d_t, λ) cell; aggregate with
    ``F_agg``.
    """
    arm = args.f_arm
    if arm is None:
        raise SystemExit("Probe F requires --f-arm {linear,silu_base,kan}")

    d_t = int(args.d_transcoder)
    lam = float(args.lambda_sparsity)
    cell = f"{arm}_dt{d_t}_{_lam_tag(lam)}"
    out = ensure_dir(Path(PROBES_ROOT) / "F_capacity_l0" / cell)

    cfg = _base_cfg(args)
    cfg.d_transcoder = d_t
    cfg.lambda_sparsity = lam
    # Unlocked — real training setup; no linear-dictionary lock confound.
    cfg.lock_threshold = False
    cfg.lock_decoder = False
    cfg.freeze_encoder = False
    cfg.freeze_decoder = False
    cfg.lambda_kan_reg = args.lambda_kan_reg if arm == "kan" else 0.0

    print(
        f"[F] cell={cell} arm={arm} d_t={d_t} λ={lam} steps={cfg.total_steps} "
        f"lock_θ=False lock_dec=False",
        flush=True,
    )
    result = train_arm(arm, cfg, out, reference_model=None)
    fv = result["summary"]["final_val"]
    cell_summary = {
        "probe": "F_capacity_l0",
        "cell": cell,
        "arm": arm,
        "d_transcoder": d_t,
        "lambda_sparsity": lam,
        "steps": cfg.total_steps,
        "lock_threshold": False,
        "lock_decoder": False,
        "rel_fro": fv.get("reconstruction/rel_fro_error"),
        "nmse": fv.get("reconstruction/nmse_mean"),
        "l0": fv.get("stats/l0_active_features_per_token"),
        "act_lp": fv.get("stats/active_features_per_pos"),
        "base_only_rel_fro": (result["summary"].get("base_only_val") or {}).get(
            "reconstruction/rel_fro_error"
        ),
        "checkpoint": result["summary"]["checkpoint"],
    }
    (out / "cell.json").write_text(json.dumps(cell_summary, indent=2) + "\n")
    print(json.dumps(cell_summary, indent=2), flush=True)
    return cell_summary


def run_F_agg(args: argparse.Namespace) -> dict:
    """Aggregate F cells: pick λ closest to linear@2048 L0; test capacity claim."""
    root = Path(PROBES_ROOT) / "F_capacity_l0"
    cells = []
    for p in sorted(root.glob("*/cell.json")):
        cells.append(json.loads(p.read_text()))
    if not cells:
        raise SystemExit(f"No F cells under {root}")

    by_key: dict[tuple, list] = {}
    for c in cells:
        by_key.setdefault((c["arm"], c["d_transcoder"]), []).append(c)

    # Reference: linear at largest available d_t (prefer 2048).
    lin_refs = [c for c in cells if c["arm"] == "linear"]
    if not lin_refs:
        raise SystemExit("F_agg needs at least one linear cell")
    lin_ref = sorted(lin_refs, key=lambda c: (c["d_transcoder"], -c["lambda_sparsity"]))[-1]
    # Prefer exact 2048 if present
    for c in lin_refs:
        if c["d_transcoder"] == 2048:
            lin_ref = c
            break
    target_l0 = float(lin_ref["l0"])

    def _pick_l0(cands: list[dict]) -> dict:
        return min(cands, key=lambda c: abs(float(c["l0"]) - target_l0))

    picked = {}
    for key, cands in sorted(by_key.items()):
        best = _pick_l0(cands)
        picked[f"{key[0]}_dt{key[1]}"] = {
            "cell": best["cell"],
            "lambda_sparsity": best["lambda_sparsity"],
            "l0": best["l0"],
            "rel_fro": best["rel_fro"],
            "l0_abs_err_vs_linear_ref": abs(float(best["l0"]) - target_l0),
            "n_lambda_tried": len(cands),
        }

    # Claim checks
    lin_2048 = picked.get("linear_dt2048")
    lin_512 = picked.get("linear_dt512")
    claim = {
        "linear_ref_cell": lin_ref["cell"],
        "target_l0": target_l0,
        "silu_base_dt512_beats_or_ties_linear_dt2048": None,
        "kan_dt512_beats_or_ties_linear_dt2048": None,
        "matched_l0_closes_gap_at_2048": None,
        "notes": [],
    }
    if lin_2048:
        for key in ("silu_base_dt512", "kan_dt512"):
            row = picked.get(key)
            if row and row["rel_fro"] is not None:
                claim[f"{key}_beats_or_ties_linear_dt2048"] = float(row["rel_fro"]) <= float(
                    lin_2048["rel_fro"]
                )
        silu_2048 = picked.get("silu_base_dt2048")
        if silu_2048 and lin_2048:
            # Did L0-retargeted SiLU close the ~0.034 locked-probe gap?
            gap = float(silu_2048["rel_fro"]) - float(lin_2048["rel_fro"])
            claim["matched_l0_closes_gap_at_2048"] = gap < 0.01
            claim["silu_vs_linear_gap_at_2048"] = gap
            claim["notes"].append(
                "matched_l0_closes_gap_at_2048: best-λ silu@2048 vs linear@2048; "
                "True if rel_fro gap < 0.01"
            )
    if lin_512 and lin_2048:
        claim["notes"].append(
            f"linear@512 rel_fro={lin_512['rel_fro']:.4f} vs linear@2048 "
            f"{lin_2048['rel_fro']:.4f} (capacity baseline)"
        )

    summary = {
        "probe": "F_capacity_l0",
        "question": (
            "Does SiLU/KAN at smaller d_t match linear at larger d_t once L0 is retargeted?"
        ),
        "n_cells": len(cells),
        "picked_by_l0": picked,
        "all_cells": cells,
        "claim": claim,
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary("F_capacity_l0", summary)
    print(json.dumps({"picked_by_l0": picked, "claim": claim}, indent=2), flush=True)
    return summary


def run_all(args: argparse.Namespace) -> None:
    run_A(args)
    run_B(args)
    run_C(args)
    run_D(args)
    run_E(args)
    # Aggregate
    root = Path(PROBES_ROOT) / "summaries"
    agg = {}
    for p in sorted(root.glob("*.json")):
        if p.name == "ALL.json":
            continue
        agg[p.stem] = json.loads(p.read_text())
    (root / "ALL.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(f"Wrote {root / 'ALL.json'}", flush=True)


G_VARIANTS = {
    # F-matched control but with correct θ Adam group (wd=0, eps=1e-15).
    "baseline": {
        "target_l0": 32.0,
        "bandwidth_frac": 0.1,
        "theta_delay_frac": 0.0,
        "activation_function": "jump_relu",
    },
    # Open the gate at init toward linear's observed L0 (~95–110).
    "target128": {
        "target_l0": 128.0,
        "bandwidth_frac": 0.1,
        "theta_delay_frac": 0.0,
        "activation_function": "jump_relu",
    },
    "target256": {
        "target_l0": 256.0,
        "bandwidth_frac": 0.1,
        "theta_delay_frac": 0.0,
        "activation_function": "jump_relu",
    },
    # Wider STE so dead features can revive.
    "bw1x": {
        "target_l0": 32.0,
        "bandwidth_frac": 1.0,
        "theta_delay_frac": 0.0,
        "activation_function": "jump_relu",
    },
    # Learn features with θ≈0 for 30% of steps, then restore calibrated θ.
    "delay30": {
        "target_l0": 128.0,
        "bandwidth_frac": 0.1,
        "theta_delay_frac": 0.3,
        "activation_function": "jump_relu",
    },
    # Decisive ablation: no JumpReLU gate at all.
    "relu": {
        "target_l0": 32.0,
        "bandwidth_frac": 0.1,
        "theta_delay_frac": 0.0,
        "activation_function": "relu",
    },
}


def run_G(args: argparse.Namespace) -> dict:
    """Phase G: JumpReLU / θ schedule levers (unlocked decoder, λ=1e-6)."""
    arm = args.f_arm
    variant = args.g_variant
    if arm is None or variant is None:
        raise SystemExit("Probe G requires --f-arm and --g-variant")
    if variant not in G_VARIANTS:
        raise SystemExit(f"Unknown --g-variant {variant}; choose from {list(G_VARIANTS)}")

    spec = G_VARIANTS[variant]
    d_t = int(args.d_transcoder)
    lam = float(args.lambda_sparsity)
    cell = f"{arm}_dt{d_t}_{variant}"
    out = ensure_dir(Path(PROBES_ROOT) / "G_jumprelu_theta" / cell)

    cfg = _base_cfg(args)
    cfg.d_transcoder = d_t
    cfg.lambda_sparsity = lam
    cfg.lock_threshold = False
    cfg.lock_decoder = False
    cfg.freeze_encoder = False
    cfg.freeze_decoder = False
    cfg.use_threshold_adam_group = True
    cfg.target_l0 = float(spec["target_l0"])
    cfg.bandwidth_frac = float(spec["bandwidth_frac"])
    cfg.theta_delay_frac = float(spec["theta_delay_frac"])
    cfg.activation_function = str(spec["activation_function"])
    cfg.lambda_kan_reg = args.lambda_kan_reg if arm == "kan" else 0.0
    if cfg.activation_function == "relu":
        cfg.lock_threshold = True  # no-op; no threshold param

    print(
        f"[G] cell={cell} arm={arm} variant={variant} d_t={d_t} λ={lam} "
        f"target_l0={cfg.target_l0} bw_frac={cfg.bandwidth_frac} "
        f"delay={cfg.theta_delay_frac} act={cfg.activation_function}",
        flush=True,
    )
    result = train_arm(arm, cfg, out, reference_model=None)
    fv = result["summary"]["final_val"]
    cell_summary = {
        "probe": "G_jumprelu_theta",
        "cell": cell,
        "arm": arm,
        "variant": variant,
        "d_transcoder": d_t,
        "lambda_sparsity": lam,
        "target_l0": cfg.target_l0,
        "bandwidth_frac": cfg.bandwidth_frac,
        "theta_delay_frac": cfg.theta_delay_frac,
        "activation_function": cfg.activation_function,
        "steps": cfg.total_steps,
        "rel_fro": fv.get("reconstruction/rel_fro_error"),
        "nmse": fv.get("reconstruction/nmse_mean"),
        "l0": fv.get("stats/l0_active_features_per_token"),
        "act_lp": fv.get("stats/active_features_per_pos"),
        "base_only_rel_fro": (result["summary"].get("base_only_val") or {}).get(
            "reconstruction/rel_fro_error"
        ),
        "checkpoint": result["summary"]["checkpoint"],
        "init": result["summary"].get("init"),
    }
    (out / "cell.json").write_text(json.dumps(cell_summary, indent=2) + "\n")
    print(json.dumps(cell_summary, indent=2), flush=True)
    return cell_summary


def run_G_agg(args: argparse.Namespace) -> dict:
    """Aggregate G: did any JumpReLU lever raise SiLU L0 and close the gap?"""
    root = Path(PROBES_ROOT) / "G_jumprelu_theta"
    cells = [json.loads(p.read_text()) for p in sorted(root.glob("*/cell.json"))]
    if not cells:
        raise SystemExit(f"No G cells under {root}")

    table: dict[str, dict] = {}
    for c in cells:
        table.setdefault(c["arm"], {}).setdefault(c["variant"], {})[c["d_transcoder"]] = c

    lin_base = (table.get("linear") or {}).get("baseline", {}).get(2048)
    if lin_base is None:
        for _v, by_dt in (table.get("linear") or {}).items():
            if 2048 in by_dt:
                lin_base = by_dt[2048]
                break
    claim: dict = {"linear_baseline_2048": lin_base, "variants": {}}
    if lin_base:
        target = float(lin_base["l0"])
        for variant in G_VARIANTS:
            silu = (table.get("silu_base") or {}).get(variant, {}).get(2048)
            kan = (table.get("kan") or {}).get(variant, {}).get(2048)
            silu512 = (table.get("silu_base") or {}).get(variant, {}).get(512)
            row: dict = {"silu_2048": silu, "kan_2048": kan, "silu_512": silu512}
            if silu:
                row["silu_l0_vs_linear"] = float(silu["l0"]) - target
                row["silu_gap_rel_fro"] = float(silu["rel_fro"]) - float(lin_base["rel_fro"])
                row["closes_gap"] = row["silu_gap_rel_fro"] < 0.01
                row["l0_matched"] = abs(float(silu["l0"]) - target) < 0.15 * target
            if silu512 and lin_base:
                row["capacity_silu512_vs_lin2048"] = float(silu512["rel_fro"]) <= float(
                    lin_base["rel_fro"]
                )
            claim["variants"][variant] = row

    summary = {
        "probe": "G_jumprelu_theta",
        "question": "Can JumpReLU/θ schedule levers raise SiLU/KAN L0 and close the recon gap?",
        "n_cells": len(cells),
        "cells": cells,
        "claim": claim,
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary("G_jumprelu_theta", summary)
    print(json.dumps({"claim": claim}, indent=2), flush=True)
    return summary


def run_H(args: argparse.Namespace) -> dict:
    """Phase H: matched-L0 bakeoff under open-gate JumpReLU.

    Calibrate θ to ``--target-l0``, then **lock θ** (so L0 stays near the
    operating point). Unlocked decoder, λ=1e-6. Sweep target_l0 across jobs;
    ``H_agg`` picks cells nearest L0 bands 100/200.
    """
    arm = args.f_arm
    if arm is None:
        raise SystemExit("Probe H requires --f-arm")
    if args.target_l0 is None:
        raise SystemExit("Probe H requires --target-l0")

    d_t = int(args.d_transcoder)
    lam = float(args.lambda_sparsity)
    target = float(args.target_l0)
    t_tag = f"t{int(round(target))}"
    cell = f"{arm}_dt{d_t}_{t_tag}"
    out = ensure_dir(Path(PROBES_ROOT) / "H_matched_l0" / cell)

    cfg = _base_cfg(args)
    cfg.d_transcoder = d_t
    cfg.lambda_sparsity = lam
    cfg.target_l0 = target
    cfg.bandwidth_frac = 0.1
    cfg.theta_delay_frac = 0.0
    cfg.activation_function = "jump_relu"
    cfg.lock_threshold = True
    cfg.lock_decoder = False
    cfg.freeze_encoder = False
    cfg.freeze_decoder = False
    cfg.use_threshold_adam_group = False
    cfg.lambda_kan_reg = args.lambda_kan_reg if arm == "kan" else 0.0

    print(
        f"[H] cell={cell} arm={arm} d_t={d_t} target_l0={target} λ={lam} "
        f"lock_θ=True lock_dec=False steps={cfg.total_steps}",
        flush=True,
    )
    result = train_arm(arm, cfg, out, reference_model=None)
    fv = result["summary"]["final_val"]
    cell_summary = {
        "probe": "H_matched_l0",
        "cell": cell,
        "arm": arm,
        "d_transcoder": d_t,
        "target_l0": target,
        "lambda_sparsity": lam,
        "lock_threshold": True,
        "lock_decoder": False,
        "steps": cfg.total_steps,
        "rel_fro": fv.get("reconstruction/rel_fro_error"),
        "nmse": fv.get("reconstruction/nmse_mean"),
        "l0": fv.get("stats/l0_active_features_per_token"),
        "act_lp": fv.get("stats/active_features_per_pos"),
        "base_only_rel_fro": (result["summary"].get("base_only_val") or {}).get(
            "reconstruction/rel_fro_error"
        ),
        "checkpoint": result["summary"]["checkpoint"],
        "init": result["summary"].get("init"),
    }
    (out / "cell.json").write_text(json.dumps(cell_summary, indent=2) + "\n")
    print(json.dumps(cell_summary, indent=2), flush=True)
    return cell_summary


def run_H_agg(args: argparse.Namespace) -> dict:
    """Pick cells nearest L0 bands; test capacity + spline-vs-SiLU at matched L0."""
    root = Path(PROBES_ROOT) / "H_matched_l0"
    cells = [json.loads(p.read_text()) for p in sorted(root.glob("*/cell.json"))]
    if not cells:
        raise SystemExit(f"No H cells under {root}")

    bands = [100.0, 200.0]
    tol = 0.20

    by_key: dict[tuple, list] = {}
    for c in cells:
        by_key.setdefault((c["arm"], c["d_transcoder"]), []).append(c)

    picked: dict[str, dict] = {}
    for band in bands:
        for (arm, d_t), cands in sorted(by_key.items()):
            best = min(cands, key=lambda c: abs(float(c["l0"]) - band))
            key = f"{arm}_dt{d_t}_L{int(band)}"
            err = abs(float(best["l0"]) - band)
            picked[key] = {
                "cell": best["cell"],
                "target_l0": best["target_l0"],
                "l0": best["l0"],
                "rel_fro": best["rel_fro"],
                "l0_err": err,
                "matched": err <= tol * band,
            }

    claim: dict = {"bands": bands, "tol": tol, "picked": picked, "tests": {}}
    for band in bands:
        b = int(band)
        lin = picked.get(f"linear_dt2048_L{b}")
        silu512 = picked.get(f"silu_base_dt512_L{b}")
        silu2048 = picked.get(f"silu_base_dt2048_L{b}")
        kan512 = picked.get(f"kan_dt512_L{b}")
        kan2048 = picked.get(f"kan_dt2048_L{b}")
        tests: dict = {"band": band}
        if lin and silu2048 and lin["matched"] and silu2048["matched"]:
            tests["silu_beats_linear_at_2048"] = float(silu2048["rel_fro"]) <= float(
                lin["rel_fro"]
            )
            tests["silu_gap"] = float(silu2048["rel_fro"]) - float(lin["rel_fro"])
        if lin and silu512 and lin["matched"] and silu512["matched"]:
            tests["capacity_silu512_vs_lin2048"] = float(silu512["rel_fro"]) <= float(
                lin["rel_fro"]
            )
            tests["capacity_gap"] = float(silu512["rel_fro"]) - float(lin["rel_fro"])
        if lin and kan512 and lin["matched"] and kan512["matched"]:
            tests["capacity_kan512_vs_lin2048"] = float(kan512["rel_fro"]) <= float(
                lin["rel_fro"]
            )
        if silu2048 and kan2048 and silu2048["matched"] and kan2048["matched"]:
            tests["kan_beats_silu_at_2048"] = float(kan2048["rel_fro"]) < float(
                silu2048["rel_fro"]
            ) - 0.005
            tests["kan_minus_silu"] = float(kan2048["rel_fro"]) - float(
                silu2048["rel_fro"]
            )
        tests["raw"] = {
            "linear_2048": lin,
            "silu_512": silu512,
            "silu_2048": silu2048,
            "kan_512": kan512,
            "kan_2048": kan2048,
        }
        claim["tests"][f"L{b}"] = tests

    summary = {
        "probe": "H_matched_l0",
        "question": (
            "At matched L0 (~100 and ~200), does SiLU/KAN@512 beat linear@2048, "
            "and does full KAN beat SiLU-base?"
        ),
        "n_cells": len(cells),
        "claim": claim,
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary("H_matched_l0", summary)
    print(json.dumps(claim, indent=2), flush=True)
    return summary



def run_I(args: argparse.Namespace) -> dict:
    """Phase I: spline-only encoder (no SiLU base) vs silu_base / full KAN / linear.

    Recipe from H: unlocked decoder, θ calibrated then locked, λ=1e-6.
    ``spline_only`` enables periodic ``update_grid``; ``spline_only_nogrid`` does not.
    """
    arm = args.f_arm
    if arm is None:
        raise SystemExit("Probe I requires --f-arm")
    if args.target_l0 is None:
        raise SystemExit("Probe I requires --target-l0")
    if arm not in ("linear", "silu_base", "kan", "spline_only", "spline_only_nogrid"):
        raise SystemExit(f"Unsupported I arm: {arm}")

    d_t = int(args.d_transcoder)
    lam = float(args.lambda_sparsity)
    target = float(args.target_l0)
    t_tag = f"t{int(round(target))}"
    cell = f"{arm}_dt{d_t}_{t_tag}"
    out = ensure_dir(Path(PROBES_ROOT) / "I_spline_only" / cell)

    cfg = _base_cfg(args)
    cfg.d_transcoder = d_t
    cfg.lambda_sparsity = lam
    cfg.target_l0 = target
    cfg.bandwidth_frac = 0.1
    cfg.theta_delay_frac = 0.0
    cfg.activation_function = "jump_relu"
    cfg.lock_threshold = True
    cfg.lock_decoder = False
    cfg.freeze_encoder = False
    cfg.freeze_decoder = False
    cfg.use_threshold_adam_group = False
    cfg.lambda_kan_reg = 0.0
    if arm == "spline_only":
        cfg.update_grid_every = 500
        cfg.update_grid_from = 200
    else:
        cfg.update_grid_every = 0

    print(
        f"[I] cell={cell} arm={arm} d_t={d_t} target_l0={target} λ={lam} "
        f"update_grid_every={cfg.update_grid_every} lock_θ=True lock_dec=False",
        flush=True,
    )
    result = train_arm(arm, cfg, out, reference_model=None)
    fv = result["summary"]["final_val"]
    # Path diagnostics from last training record if present
    last_spline_frac = None
    last_grad_base = None
    last_grad_spline = None
    rec_path = Path(result["summary"]["records"])
    if rec_path.exists():
        lines = rec_path.read_text().strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
            last_spline_frac = last.get("stats/spline_contribution_frac")
            last_grad_base = last.get("grad/base_weight")
            last_grad_spline = last.get("grad/spline_weight")

    cell_summary = {
        "probe": "I_spline_only",
        "cell": cell,
        "arm": arm,
        "d_transcoder": d_t,
        "target_l0": target,
        "lambda_sparsity": lam,
        "update_grid_every": cfg.update_grid_every,
        "lock_threshold": True,
        "lock_decoder": False,
        "steps": cfg.total_steps,
        "rel_fro": fv.get("reconstruction/rel_fro_error"),
        "nmse": fv.get("reconstruction/nmse_mean"),
        "l0": fv.get("stats/l0_active_features_per_token"),
        "act_lp": fv.get("stats/active_features_per_pos"),
        "spline_contribution_frac_late": last_spline_frac,
        "grad_base_late": last_grad_base,
        "grad_spline_late": last_grad_spline,
        "base_only_rel_fro": (result["summary"].get("base_only_val") or {}).get(
            "reconstruction/rel_fro_error"
        ),
        "checkpoint": result["summary"]["checkpoint"],
        "init": result["summary"].get("init"),
    }
    (out / "cell.json").write_text(json.dumps(cell_summary, indent=2) + "\n")
    print(json.dumps(cell_summary, indent=2), flush=True)
    return cell_summary


def run_I_agg(args: argparse.Namespace) -> dict:
    """Aggregate I: does forced spline-only beat SiLU-base / enable capacity?"""
    root = Path(PROBES_ROOT) / "I_spline_only"
    cells = [json.loads(p.read_text()) for p in sorted(root.glob("*/cell.json"))]
    if not cells:
        raise SystemExit(f"No I cells under {root}")

    bands = [100.0, 200.0]
    tol = 0.20
    by_key: dict[tuple, list] = {}
    for c in cells:
        by_key.setdefault((c["arm"], c["d_transcoder"]), []).append(c)

    picked: dict[str, dict] = {}
    for band in bands:
        for (arm, d_t), cands in sorted(by_key.items()):
            best = min(cands, key=lambda c: abs(float(c["l0"]) - band))
            key = f"{arm}_dt{d_t}_L{int(band)}"
            err = abs(float(best["l0"]) - band)
            picked[key] = {
                "cell": best["cell"],
                "target_l0": best["target_l0"],
                "l0": best["l0"],
                "rel_fro": best["rel_fro"],
                "spline_frac": best.get("spline_contribution_frac_late"),
                "l0_err": err,
                "matched": err <= tol * band,
            }

    claim: dict = {"bands": bands, "tol": tol, "picked": picked, "tests": {}}
    for band in bands:
        b = int(band)
        lin = picked.get(f"linear_dt2048_L{b}")
        silu = picked.get(f"silu_base_dt2048_L{b}")
        kan = picked.get(f"kan_dt2048_L{b}")
        spl = picked.get(f"spline_only_dt2048_L{b}")
        spl_ng = picked.get(f"spline_only_nogrid_dt2048_L{b}")
        spl512 = picked.get(f"spline_only_dt512_L{b}")
        tests: dict = {"band": band, "raw": {
            "linear_2048": lin, "silu_2048": silu, "kan_2048": kan,
            "spline_only_2048": spl, "spline_only_nogrid_2048": spl_ng,
            "spline_only_512": spl512,
        }}
        if lin and spl and lin["matched"] and spl["matched"]:
            tests["spline_only_beats_linear_2048"] = float(spl["rel_fro"]) <= float(lin["rel_fro"])
            tests["spline_only_gap_vs_lin"] = float(spl["rel_fro"]) - float(lin["rel_fro"])
        if silu and spl and silu["matched"] and spl["matched"]:
            tests["spline_only_beats_silu_2048"] = float(spl["rel_fro"]) < float(silu["rel_fro"]) - 0.005
            tests["spline_only_gap_vs_silu"] = float(spl["rel_fro"]) - float(silu["rel_fro"])
        if kan and spl and kan["matched"] and spl["matched"]:
            tests["spline_only_vs_full_kan"] = float(spl["rel_fro"]) - float(kan["rel_fro"])
        if spl and spl_ng and spl["matched"] and spl_ng["matched"]:
            tests["update_grid_helps"] = float(spl["rel_fro"]) < float(spl_ng["rel_fro"]) - 0.005
            tests["grid_gap"] = float(spl["rel_fro"]) - float(spl_ng["rel_fro"])
        if lin and spl512 and lin["matched"] and spl512["matched"]:
            tests["capacity_spline512_vs_lin2048"] = float(spl512["rel_fro"]) <= float(lin["rel_fro"])
            tests["capacity_gap"] = float(spl512["rel_fro"]) - float(lin["rel_fro"])
        if spl:
            tests["spline_frac_near_one"] = (
                spl.get("spline_frac") is not None and float(spl["spline_frac"]) > 0.9
            )
        claim["tests"][f"L{b}"] = tests

    summary = {
        "probe": "I_spline_only",
        "question": (
            "With SiLU base removed, does the B-spline path do real work and "
            "beat SiLU-base / enable smaller-dictionary capacity?"
        ),
        "n_cells": len(cells),
        "claim": claim,
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary("I_spline_only", summary)
    print(json.dumps(claim, indent=2), flush=True)
    return summary



def run_J(args: argparse.Namespace) -> dict:
    """Probe J: Tier-1 routing (init scale / spline LR) on full KAN.

    GPT-2 r5b sparsity recipe: peak λ=1e-5, warmup 2000, decay from 4500 → 3e-6,
    absolute ``jumprelu_bandwidth=0.12``. Unlocked decoder, locked θ.
    """
    variant = args.j_variant
    if variant is None:
        raise SystemExit("Probe J requires --j-variant {B2,B3}")
    if args.target_l0 is None:
        raise SystemExit("Probe J requires --target-l0")
    if variant not in ("B2", "B3"):
        raise SystemExit(f"Unsupported J variant: {variant}")

    d_t = int(args.d_transcoder)
    lam = float(args.lambda_sparsity)
    target = float(args.target_l0)
    t_tag = f"t{int(round(target))}"
    cell = f"kan_{variant}_dt{d_t}_{t_tag}"
    out = ensure_dir(Path(PROBES_ROOT) / "J_tier1_routing" / cell)

    cfg = _base_cfg(args)
    cfg.d_transcoder = d_t
    cfg.lambda_sparsity = lam
    cfg.target_l0 = target
    # GPT-2 r5b JumpReLU / λ schedule
    cfg.absolute_jumprelu_bandwidth = 0.12
    cfg.jumprelu_bandwidth = 0.12
    cfg.sparsity_warmup_steps = 2000
    cfg.sparsity_decay_start = 4500
    cfg.lambda_sparsity_final = 3e-6
    cfg.theta_delay_frac = 0.0
    cfg.activation_function = "jump_relu"
    cfg.lock_threshold = True
    cfg.lock_decoder = False
    cfg.freeze_encoder = False
    cfg.freeze_decoder = False
    cfg.use_threshold_adam_group = False
    cfg.lambda_kan_reg = 0.0
    cfg.update_grid_every = 0
    cfg.scale_base = float(args.scale_base)
    cfg.scale_spline = float(args.scale_spline)
    cfg.lr_spline_mult = 5.0 if variant == "B3" else float(args.lr_spline_mult)

    print(
        f"[J] cell={cell} variant={variant} d_t={d_t} target_l0={target} "
        f"λ_peak={lam} warmup={cfg.sparsity_warmup_steps} "
        f"decay_start={cfg.sparsity_decay_start} λ_final={cfg.lambda_sparsity_final} "
        f"bw={cfg.absolute_jumprelu_bandwidth} "
        f"scale_base={cfg.scale_base} scale_spline={cfg.scale_spline} "
        f"lr_spline_mult={cfg.lr_spline_mult} lock_θ=True lock_dec=False",
        flush=True,
    )
    result = train_arm("kan", cfg, out, reference_model=None)
    fv = result["summary"]["final_val"]
    last_spline_frac = None
    last_grad_base = None
    last_grad_spline = None
    rec_path = Path(result["summary"]["records"])
    if rec_path.exists():
        lines = rec_path.read_text().strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
            last_spline_frac = last.get("stats/spline_contribution_frac")
            last_grad_base = last.get("grad/base_weight")
            last_grad_spline = last.get("grad/spline_weight")

    cell_summary = {
        "probe": "J_tier1_routing",
        "cell": cell,
        "variant": variant,
        "arm": "kan",
        "d_transcoder": d_t,
        "target_l0": target,
        "lambda_sparsity": lam,
        "sparsity_warmup_steps": cfg.sparsity_warmup_steps,
        "sparsity_decay_start": cfg.sparsity_decay_start,
        "lambda_sparsity_final": cfg.lambda_sparsity_final,
        "absolute_jumprelu_bandwidth": cfg.absolute_jumprelu_bandwidth,
        "lambda_kan_reg": cfg.lambda_kan_reg,
        "scale_base": cfg.scale_base,
        "scale_spline": cfg.scale_spline,
        "lr_spline_mult": cfg.lr_spline_mult,
        "lock_threshold": True,
        "lock_decoder": False,
        "steps": cfg.total_steps,
        "rel_fro": fv.get("reconstruction/rel_fro_error"),
        "nmse": fv.get("reconstruction/nmse_mean"),
        "l0": fv.get("stats/l0_active_features_per_token"),
        "act_lp": fv.get("stats/active_features_per_pos"),
        "spline_contribution_frac_late": last_spline_frac,
        "grad_base_late": last_grad_base,
        "grad_spline_late": last_grad_spline,
        "base_only_rel_fro": (result["summary"].get("base_only_val") or {}).get(
            "reconstruction/rel_fro_error"
        ),
        "checkpoint": result["summary"]["checkpoint"],
        "init": result["summary"].get("init"),
        "compare_to": {
            "kan": f"I_spline_only/kan_dt{d_t}_{t_tag}",
            "silu_base": f"I_spline_only/silu_base_dt{d_t}_{t_tag}",
        },
    }
    (out / "cell.json").write_text(json.dumps(cell_summary, indent=2) + "\n")
    print(json.dumps(cell_summary, indent=2), flush=True)
    return cell_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="From-scratch spline debug probes")
    parser.add_argument(
        "probe",
        choices=(
            "A", "B", "C", "D", "E", "F", "F_agg", "G", "G_agg",
            "H", "H_agg", "I", "I_agg", "J", "all",
        ),
        help="Which probe to run",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--d-transcoder", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--lambda-sparsity", type=float, default=0.005)
    parser.add_argument("--lambda-kan-reg", type=float, default=0.0)
    parser.add_argument("--scale-base", type=float, default=0.2)
    parser.add_argument("--scale-spline", type=float, default=1.0)
    parser.add_argument("--lr-spline-mult", type=float, default=1.0)
    parser.add_argument(
        "--j-variant",
        choices=("B2", "B3"),
        default=None,
        help="Probe J: B2=scale tilt; B3=scale tilt + 5× spline LR",
    )
    parser.add_argument(
        "--f-arm",
        choices=("linear", "silu_base", "kan", "spline_only", "spline_only_nogrid"),
        default=None,
        help="Probe F/G/H/I: which encoder arm for this cell",
    )
    parser.add_argument(
        "--g-variant",
        choices=tuple(G_VARIANTS.keys()),
        default=None,
        help="Probe G: JumpReLU/θ lever variant",
    )
    parser.add_argument(
        "--target-l0",
        type=float,
        default=None,
        help="Probe H/I/J: JumpReLU calibration target L0 (θ locked after)",
    )
    parser.add_argument(
        "--data-dir",
        default=(
            "/gscratch/ssuresh/shared/activations/"
            "paper_r3/gpt2_small_linear_feature_match/val/gpt2"
        ),
    )
    args = parser.parse_args(argv)
    ensure_dir(PROBES_ROOT)
    dispatch = {
        "A": run_A,
        "B": run_B,
        "C": run_C,
        "D": run_D,
        "E": run_E,
        "F": run_F,
        "F_agg": run_F_agg,
        "G": run_G,
        "G_agg": run_G_agg,
        "H": run_H,
        "H_agg": run_H_agg,
        "I": run_I,
        "I_agg": run_I_agg,
        "J": run_J,
        "all": run_all,
    }
    dispatch[args.probe](args)


if __name__ == "__main__":
    main()
