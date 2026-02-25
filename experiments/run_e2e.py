#!/usr/bin/env python3
"""Full end-to-end KAN-CLT pipeline: collect → train → evaluate → compare.

Runs every step from scratch and prints a final side-by-side comparison
between KAN-CLT and the matched linear CLT baseline.

Pipeline stages:
  1. collect   — run GPT-2 on text data, cache MLP activations to disk
  2. train-kan — train KAN-CLT on cached activations
  3. train-lin — train linear CLT baseline (same config, encoder_type=linear)
  4. evaluate  — reconstruct metrics, circuit tracing, spline analysis, monosemanticity
  5. compare   — print final KAN vs Linear comparison table

Each stage is skipped automatically if its output already exists.
Use --force-recollect / --force-retrain to rerun even if outputs exist.

Usage:
    # Full run from scratch
    conda run -n ct python experiments/run_e2e.py --device cuda

    # Quick smoke-test (small dataset, few steps) — only reconstruction & compare
    conda run -n ct python experiments/run_e2e.py --device cpu \\
        --n-tokens 5000 --total-steps 20 --d-transcoder 64 --batch-size 2

    # Full eval including circuit tracing, splines, monosemanticity
    conda run -n ct python experiments/run_e2e.py --device cuda \\
        --n-tokens 50000 --total-steps 500 --d-transcoder 256 --batch-size 4 \\
        --run-circuits --run-splines --run-monosemanticity

    # Use existing data, retrain from scratch
    conda run -n ct python experiments/run_e2e.py --device cuda --force-retrain

    # Use existing checkpoints, only re-run evaluation
    conda run -n ct python experiments/run_e2e.py --device cuda \\
        --skip-collect --skip-train
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _elapsed(t0: float) -> str:
    s = time.time() - t0
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


def _section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def _data_exists(data_dir: str) -> bool:
    return (
        os.path.isfile(os.path.join(data_dir, "mlp_inputs.pt"))
        and os.path.getsize(os.path.join(data_dir, "mlp_inputs.pt")) > 0
        and os.path.isfile(os.path.join(data_dir, "mlp_outputs.pt"))
        and os.path.getsize(os.path.join(data_dir, "mlp_outputs.pt")) > 0
    )


def _best_checkpoint(ckpt_dir: str, run_name: str) -> str:
    """Return path to the best checkpoint inside ckpt_dir."""
    return os.path.join(ckpt_dir, f"{run_name}_best")


def _checkpoint_exists(ckpt_dir: str, run_name: str) -> bool:
    path = _best_checkpoint(ckpt_dir, run_name)
    return os.path.isdir(path) and len(os.listdir(path)) > 0


# --------------------------------------------------------------------------- #
# Stage 1 — Data collection                                                   #
# --------------------------------------------------------------------------- #

def run_collect(args) -> None:
    """Collect GPT-2 activations and cache to disk."""
    from kan_clt.training.data import DataConfig, collect_activations

    _section("Stage 1 / 5 — Collect Activations")
    t0 = time.time()

    if _data_exists(args.data_dir) and not args.force_recollect:
        print(f"  Data already exists at {args.data_dir!r} — skipping.")
        print("  (use --force-recollect to re-run)")
        return

    print(f"  Model     : {args.lm_model}")
    print(f"  n_tokens  : {args.n_tokens:,}")
    print(f"  save_dir  : {args.data_dir}")

    cfg = DataConfig(
        model_name=args.lm_model,
        n_tokens=args.n_tokens,
        save_dir=args.data_dir,
        device=args.device,
    )
    dataset = collect_activations(cfg)
    print(f"\n  Collected {len(dataset)} sequences → {args.data_dir}")
    print(f"  [{_elapsed(t0)}]")


# --------------------------------------------------------------------------- #
# Stage 2 & 3 — Training                                                      #
# --------------------------------------------------------------------------- #

def _build_train_config(args, encoder_type: str):
    """Build a TrainConfig from CLI args, overriding key fields."""
    from kan_clt.training.train import TrainConfig

    if encoder_type == "kan":
        ckpt_dir = args.kan_checkpoint_dir
        run_name = "kan_clt_gpt2"
    else:
        ckpt_dir = args.linear_checkpoint_dir
        run_name = "linear_clt_gpt2"

    cfg = TrainConfig(
        encoder_type=encoder_type,
        n_layers=12,
        d_model=768,
        d_transcoder=args.d_transcoder,
        grid_size=args.grid_size,
        spline_order=3,
        learning_rate=args.learning_rate,
        warmup_steps=min(1000, args.total_steps // 10),
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        lambda_sparsity=args.lambda_sparsity,
        c_sparsity=1.0,
        grad_clip=1.0,
        log_every=max(1, args.total_steps // 200),
        eval_every=max(1, args.total_steps // 10),
        save_every=max(1, args.total_steps // 10),
        checkpoint_dir=ckpt_dir,
        run_name=run_name,
        update_grid_every=max(0, args.total_steps // 5) if encoder_type == "kan" else 0,
        update_grid_from=min(2000, args.total_steps // 20),
        data_dir=args.data_dir,
        device=args.device,
        dtype=args.dtype,
        val_fraction=0.05,
    )
    return cfg


def run_train(args, encoder_type: str, dataset=None):
    """Train one model (KAN or linear). Returns the trained model."""
    from kan_clt.training.train import train

    label = "KAN-CLT" if encoder_type == "kan" else "Linear CLT"
    ckpt_dir = args.kan_checkpoint_dir if encoder_type == "kan" else args.linear_checkpoint_dir
    run_name = "kan_clt_gpt2" if encoder_type == "kan" else "linear_clt_gpt2"

    _section(f"Stage {'2' if encoder_type == 'kan' else '3'} / 5 — Train {label}")
    t0 = time.time()

    if _checkpoint_exists(ckpt_dir, run_name) and not args.force_retrain:
        print(f"  Checkpoint exists at {_best_checkpoint(ckpt_dir, run_name)!r} — skipping.")
        print("  (use --force-retrain to re-run)")
        return None

    cfg = _build_train_config(args, encoder_type)
    print(f"  encoder_type  : {encoder_type}")
    print(f"  d_transcoder  : {cfg.d_transcoder}")
    print(f"  total_steps   : {cfg.total_steps:,}")
    print(f"  batch_size    : {cfg.batch_size}")
    print(f"  lambda_sparsity: {cfg.lambda_sparsity}")
    print(f"  checkpoint_dir: {cfg.checkpoint_dir}")

    model = train(cfg, dataset=dataset)
    print(f"\n  Training complete  [{_elapsed(t0)}]")
    print(f"  Best checkpoint: {_best_checkpoint(ckpt_dir, run_name)}")
    return model


# --------------------------------------------------------------------------- #
# Stage 4 — Evaluation                                                        #
# --------------------------------------------------------------------------- #

def run_evaluate(args) -> dict:
    """Run the four-stage evaluation pipeline. Returns combined metrics dict."""
    from experiments.run_pipeline import (
        run_reconstruction,
        run_circuits,
        run_splines,
        run_monosemanticity,
        run_weight_norms,
        run_checkpoint_scan,
        write_report,
    )

    _section("Stage 4 / 5 — Evaluation")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    kan_ckpt = _best_checkpoint(args.kan_checkpoint_dir, "kan_clt_gpt2")
    lin_ckpt = _best_checkpoint(args.linear_checkpoint_dir, "linear_clt_gpt2")

    if not os.path.isdir(kan_ckpt):
        kan_ckpt = None
    if not os.path.isdir(lin_ckpt):
        lin_ckpt = None

    if kan_ckpt is None and lin_ckpt is None:
        print("  No checkpoints found — cannot evaluate.")
        return {}

    os.makedirs(args.output_dir, exist_ok=True)

    recon = run_reconstruction(
        kan_ckpt, lin_ckpt,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_eval_samples,
        device=device,
        dtype=dtype,
    )

    # Diagnostic: per-layer encoder weight norms (collapse detection)
    if not args.skip_weight_norms:
        run_weight_norms(
            kan_ckpt, lin_ckpt,
            output_dir=args.output_dir,
            device=device,
            dtype=dtype,
        )

    # Diagnostic: loss curve across saved step checkpoints
    if not args.skip_checkpoint_scan:
        scan_samples = min(50, args.n_eval_samples)
        for ckpt_dir, run_name in [
            (args.kan_checkpoint_dir,    "kan_clt_gpt2"),
            (args.linear_checkpoint_dir, "linear_clt_gpt2"),
        ]:
            if os.path.isdir(ckpt_dir):
                run_checkpoint_scan(
                    checkpoint_dir=ckpt_dir,
                    run_name=run_name,
                    data_dir=args.data_dir,
                    n_samples=scan_samples,
                    output_dir=args.output_dir,
                    device=device,
                    dtype=dtype,
                )

    circuits = {}
    if not args.skip_circuits:
        circuits = run_circuits(
            kan_ckpt, lin_ckpt,
            output_dir=args.output_dir,
            lm_name=args.lm_model,
            max_features=args.max_features,
            run_shapley=args.shapley,
            shapley_samples=args.shapley_samples,
            device=device,
            dtype=dtype,
        )

    if not args.skip_splines and kan_ckpt:
        run_splines(
            kan_ckpt,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_features=args.n_spline_features,
            n_samples=args.n_eval_samples,
            device=device,
            no_plot=args.no_plot,
        )

    mono = {}
    if not args.skip_monosemanticity:
        mono = run_monosemanticity(
            kan_ckpt, lin_ckpt,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_features=args.n_mono_features,
            n_samples=args.n_eval_samples,
            device=device,
            dtype=dtype,
        )

    # Build a namespace-like object for write_report (class bodies can't access
    # enclosing function-scope variables, so we use SimpleNamespace instead)
    import types
    report_args = types.SimpleNamespace(
        kan_checkpoint=kan_ckpt,
        linear_checkpoint=lin_ckpt,
        device=str(device),
        dtype=args.dtype,
    )
    report_path = write_report(args.output_dir, recon, circuits, mono, report_args)
    print(f"\nReport: {report_path}")
    return {"reconstruction": recon, "circuits": circuits, "monosemanticity": mono}


# --------------------------------------------------------------------------- #
# Stage 5 — Final comparison printout                                         #
# --------------------------------------------------------------------------- #

def print_final_comparison(metrics: dict) -> None:
    """Print a clean final summary comparing KAN-CLT vs Linear CLT."""
    _section("Stage 5 / 5 — Final Comparison: KAN-CLT vs Linear CLT")

    recon = metrics.get("reconstruction", {})
    mono  = metrics.get("monosemanticity", {})

    if not recon:
        print("  No reconstruction metrics available.")
        return

    # ---- Reconstruction table ----
    models = list(recon.keys())
    col = 16

    print("\n── Reconstruction Quality ─────────────────────────────────────")
    fmt_hdr = f"  {'Metric':<26}" + "".join(f"{m:>{col}}" for m in models)
    print(fmt_hdr)
    print("  " + "-" * (26 + col * len(models)))

    rows = [
        ("MSE (total)",          "mse_total",          6),
        ("Cosine similarity",     "cosine_similarity",  4),
        ("Relative error",        "relative_error",     4),
        ("Active features / pos", "active_per_pos",     2),
        ("Activation density",    "activation_density", 6),
        ("Parameters",            "n_params",           0),
    ]

    for label, key, dec in rows:
        vals = []
        for m in models:
            v = recon[m].get(key, float("nan"))
            vals.append(f"{v:>{col},}" if key == "n_params" else f"{v:>{col}.{dec}f}")
        print(f"  {label:<26}" + "".join(vals))

    # ---- Per-layer MSE ----
    if all("mse_per_layer" in recon.get(m, {}) for m in models):
        n_layers = len(recon[models[0]]["mse_per_layer"])
        print("\n── Per-layer MSE ───────────────────────────────────────────────")
        print(f"  {'Layer':<8}" + "".join(f"{m:>{col}}" for m in models), end="")
        if len(models) == 2:
            print(f"  {'Δ (KAN−Lin)':>{col}}")
        else:
            print()
        for l in range(n_layers):
            row_vals = [recon[m]["mse_per_layer"][l] for m in models]
            line = f"  {l:<8}" + "".join(f"{v:>{col}.6f}" for v in row_vals)
            if len(models) == 2:
                delta = row_vals[0] - row_vals[1]
                line += f"  {delta:>+{col}.6f}"
            print(line)

    # ---- Monosemanticity ----
    if mono:
        print("\n── Feature Monosemanticity (Gini Coefficient) ──────────────────")
        print(f"  {'Model':<16} {'Mean Gini':>12} {'Gini>0.8':>10} {'Features':>10}")
        print("  " + "-" * 50)
        for label, m in mono.items():
            print(f"  {label:<16} {m['mean_gini']:>12.4f} "
                  f"{m['high_gini_count']:>10} {m['n_features']:>10}")

    # ---- Verdict ----
    if len(models) == 2 and "KAN-CLT" in recon and "Linear CLT" in recon:
        kan_mse = recon["KAN-CLT"]["mse_total"]
        lin_mse = recon["Linear CLT"]["mse_total"]
        pct = (lin_mse - kan_mse) / lin_mse * 100
        verdict = f"KAN-CLT {'better' if pct > 0 else 'worse'} by {abs(pct):.1f}% MSE"

        kan_cos = recon["KAN-CLT"]["cosine_similarity"]
        lin_cos = recon["Linear CLT"]["cosine_similarity"]
        verdict_cos = f"cosine sim {'better' if kan_cos > lin_cos else 'worse'} by {abs(kan_cos-lin_cos):.4f}"

        print(f"\n  ► {verdict}, {verdict_cos}")

        if mono and "kan" in mono and "linear" in mono:
            kg, lg = mono["kan"]["mean_gini"], mono["linear"]["mean_gini"]
            gpct = (kg - lg) / max(lg, 1e-9) * 100
            print(f"  ► Monosemanticity: KAN Gini {'higher' if gpct > 0 else 'lower'} "
                  f"by {abs(gpct):.1f}% (KAN={kg:.4f}, Linear={lg:.4f})")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KAN-CLT full end-to-end pipeline: collect → train → evaluate → compare",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Compute ----
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])

    # ---- Data collection ----
    parser.add_argument("--lm-model", default="gpt2")
    parser.add_argument("--data-dir", default="data/activations")
    parser.add_argument("--n-tokens", type=int, default=2_000_000,
                        help="Tokens to collect from wikitext-103. "
                             "2M ≈ 15.6K sequences at seq_len=128 (~73 GB bfloat16 on disk). "
                             "Previous default (10M on wikitext-2) was capped at ~8.5K sequences "
                             "by dataset exhaustion; wikitext-103 has enough text to fill any budget.")

    # ---- Training ----
    parser.add_argument("--kan-checkpoint-dir",    default="checkpoints/gpt2_small")
    parser.add_argument("--linear-checkpoint-dir", default="checkpoints/gpt2_small_linear")
    parser.add_argument("--d-transcoder",    type=int,   default=4096)
    parser.add_argument("--grid-size",       type=int,   default=5)
    parser.add_argument("--total-steps",     type=int,   default=200_000,
                        help="Training steps (increased from 50K; more steps needed for good reconstruction)")
    parser.add_argument("--batch-size",      type=int,   default=16,
                        help="Sequences per batch (increased from 8 for better GPU utilization on GH200)")
    parser.add_argument("--learning-rate",   type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=8000,
                        help="Max sequences loaded into RAM for training "
                             "(increased from implicit 3000; GH200 handles ~38 GB easily)")
    parser.add_argument("--lambda-sparsity", type=float, default=0.01,
                        help="Sparsity penalty weight. Default lowered to 0.01 "
                             "(0.05 over-sparsifies KAN, killing reconstruction).")

    # ---- Evaluation ----
    parser.add_argument("--output-dir",         default="results/e2e")
    parser.add_argument("--n-eval-samples",      type=int,  default=500,
                        help="Sequences for reconstruction/monosemanticity eval "
                             "(increased from 200 for more reliable Gini estimates)")
    parser.add_argument("--max-features",        type=int,  default=64)
    parser.add_argument("--shapley",             action="store_true")
    parser.add_argument("--shapley-samples",     type=int,  default=64)
    parser.add_argument("--n-spline-features",   type=int,  default=20)
    parser.add_argument("--n-mono-features",     type=int,  default=200,
                        help="Features to analyse for monosemanticity (doubled for broader coverage)")
    parser.add_argument("--no-plot",             action="store_true")

    # ---- Skip flags ----
    # Circuits, splines, and monosemanticity require TransformerLens + GPT-2 download
    # and are slow on CPU.  They are OFF by default; pass --run-* to enable.
    parser.add_argument("--skip-collect",          action="store_true")
    parser.add_argument("--skip-train",            action="store_true",
                        help="Skip both training runs")
    parser.add_argument("--skip-circuits",         action="store_true", default=True)
    parser.add_argument("--skip-splines",          action="store_true", default=True)
    parser.add_argument("--skip-monosemanticity",  action="store_true", default=True)
    # Opt-in flags to enable the slow evaluation stages
    parser.add_argument("--run-circuits",         action="store_true",
                        help="Enable circuit tracing (requires TransformerLens + GPT-2)")
    parser.add_argument("--run-splines",          action="store_true",
                        help="Enable B-spline transfer function extraction")
    parser.add_argument("--run-monosemanticity",  action="store_true",
                        help="Enable monosemanticity (Gini) evaluation")
    # Diagnostic stages (off by default, slow but useful for debugging)
    parser.add_argument("--skip-weight-norms",    action="store_true", default=True)
    parser.add_argument("--skip-checkpoint-scan", action="store_true", default=True)
    parser.add_argument("--run-weight-norms",     action="store_true",
                        help="Per-layer encoder L2 norms — detects feature collapse "
                             "(layer 1 dominating >50%% of norm = collapse)")
    parser.add_argument("--run-checkpoint-scan",  action="store_true",
                        help="Evaluate every step checkpoint to show loss progression. "
                             "Flat curve = undertrained or lambda_sparsity too high.")

    # ---- Force flags ----
    parser.add_argument("--force-recollect", action="store_true",
                        help="Re-run data collection even if data already exists")
    parser.add_argument("--force-retrain",   action="store_true",
                        help="Re-train even if checkpoints already exist")

    args = parser.parse_args()

    # --run-* flags override the default skip=True
    if args.run_circuits:
        args.skip_circuits = False
    if args.run_splines:
        args.skip_splines = False
    if args.run_monosemanticity:
        args.skip_monosemanticity = False
    if args.run_weight_norms:
        args.skip_weight_norms = False
    if args.run_checkpoint_scan:
        args.skip_checkpoint_scan = False

    t_total = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  KAN-CLT End-to-End Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*64}")
    print(f"  device: {args.device}  |  dtype: {args.dtype}")
    print(f"  d_transcoder={args.d_transcoder}  grid_size={args.grid_size}  "
          f"total_steps={args.total_steps:,}")

    # Save config for reproducibility
    config_path = os.path.join(args.output_dir, "pipeline_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Stage 1: Collect ----
    if not args.skip_collect:
        run_collect(args)
    else:
        print("\n[Stage 1/5] Data collection skipped (--skip-collect)")

    # ---- Stage 2: Train KAN-CLT ----
    # Keep dataset in memory to share between both training runs
    dataset = None
    if not args.skip_train:
        # Load dataset once, reuse for both models
        if _data_exists(args.data_dir):
            from kan_clt.training.data import ActivationDataset
            print("\nLoading dataset into memory for training...")
            t_load = time.time()
            dataset = ActivationDataset.load(args.data_dir, max_samples=args.max_train_samples)
            print(f"  {len(dataset)} sequences loaded (max_train_samples={args.max_train_samples})  [{_elapsed(t_load)}]")

        run_train(args, encoder_type="kan",    dataset=dataset)
        run_train(args, encoder_type="linear", dataset=dataset)

        # Free dataset to reclaim RAM before evaluation
        del dataset
        dataset = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print("\n[Stage 2/5] KAN-CLT training skipped (--skip-train)")
        print("[Stage 3/5] Linear CLT training skipped (--skip-train)")

    # ---- Stage 4: Evaluate ----
    metrics = run_evaluate(args)

    # ---- Stage 5: Final comparison ----
    print_final_comparison(metrics)

    # Summary
    _section("Done")
    print(f"  Total time  : {_elapsed(t_total)}")
    print(f"  Results dir : {args.output_dir}/")
    print(f"    report.md                   ← Markdown comparison report")
    print(f"    reconstruction_metrics.json ← MSE, cosine sim, sparsity")
    print(f"    circuit_summary.json        ← Active features per prompt")
    print(f"    monosemanticity/            ← Gini coefficients, max-activating examples")
    print(f"    splines/                    ← B-spline curves (KAN only)")


if __name__ == "__main__":
    main()
