#!/usr/bin/env python3
"""Full end-to-end Spline-CLT evaluation pipeline.

Runs all evaluation phases in sequence and produces a self-contained results
directory with a Markdown summary report.

Stages (each skippable via --skip-*):
  1. reconstruct  — reconstruction MSE, cosine sim, sparsity (compare_models)
  2. circuits     — causal attribution + optional Shapley on benchmark prompts
  3. splines      — B-spline transfer function extraction (KAN only)
  4. monosemantic — Gini coefficient, max-activating examples

Usage:
    # Full run with both checkpoints
    python experiments/run_pipeline.py \\
        --kan-checkpoint  checkpoints/gpt2_small/spline_clt_gpt2_best \\
        --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \\
        --data-dir data/activations \\
        --output-dir results/eval_run1

    # Spline-CLT only (no baseline comparison)
    python experiments/run_pipeline.py \\
        --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \\
        --data-dir data/activations

    # Skip expensive stages
    python experiments/run_pipeline.py \\
        --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \\
        --skip-shapley --skip-splines
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
# Benchmark prompts for circuit tracing                                        #
# --------------------------------------------------------------------------- #
CIRCUIT_PROMPTS = [
    # Indirect object identification (IOI)
    {
        "name": "ioi_john_mary",
        "prompt": "John gave Mary the book. She gave it to",
        "task": "ioi",
    },
    {
        "name": "ioi_alice_bob",
        "prompt": "Alice told Bob the secret. He whispered it to",
        "task": "ioi",
    },
    # Arithmetic
    {
        "name": "addition_25_37",
        "prompt": "25 + 37 =",
        "task": "arithmetic",
    },
    {
        "name": "addition_14_28",
        "prompt": "14 + 28 =",
        "task": "arithmetic",
    },
    # Factual recall
    {
        "name": "factual_eiffel",
        "prompt": "The Eiffel Tower is located in",
        "task": "factual",
    },
    {
        "name": "factual_capital_france",
        "prompt": "The capital of France is",
        "task": "factual",
    },
    {
        "name": "factual_capital_germany",
        "prompt": "The capital of Germany is",
        "task": "factual",
    },
]


# --------------------------------------------------------------------------- #
# Stage helpers                                                                #
# --------------------------------------------------------------------------- #

def _checkpoint_exists(path: str | None) -> bool:
    return path is not None and os.path.isdir(path)


def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}m"


# --------------------------------------------------------------------------- #
# Stage 1: Reconstruction evaluation                                           #
# --------------------------------------------------------------------------- #

def run_reconstruction(
    kan_ckpt: str | None,
    lin_ckpt: str | None,
    data_dir: str,
    output_dir: str,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Run compare_models evaluation. Returns dict of metrics keyed by model label."""
    from spline_clt.kan_transcoder import load_spline_clt
    from spline_clt.training.data import ActivationDataset
    from experiments.compare_models import evaluate_model, print_comparison

    _section("Stage 1: Reconstruction & Sparsity")
    t0 = time.time()

    print(f"Loading dataset ({n_samples} samples)...")
    dataset = ActivationDataset.load(data_dir, max_samples=n_samples + 50)
    print(f"  {len(dataset)} samples loaded")

    models_to_eval = []
    if _checkpoint_exists(kan_ckpt):
        models_to_eval.append(("Spline-CLT", kan_ckpt))
    if _checkpoint_exists(lin_ckpt):
        models_to_eval.append(("Linear CLT", lin_ckpt))

    if not models_to_eval:
        print("  No checkpoints found — skipping reconstruction stage.")
        return {}

    all_metrics = {}
    for label, ckpt in models_to_eval:
        print(f"\nLoading {label} from {ckpt}...")
        model = load_spline_clt(ckpt, device=device, dtype=dtype)
        model.eval()
        n_p = sum(p.numel() for p in model.parameters())
        print(f"  encoder_type={model.encoder_type}, {n_p:,} parameters")

        print(f"Evaluating {label}...")
        m = evaluate_model(model, dataset, device, dtype, n_samples=n_samples)
        all_metrics[label] = m
        print(f"  MSE={m['mse_total']:.6f}  cos_sim={m['cosine_similarity']:.4f}  "
              f"active/pos={m['active_per_pos']:.2f}")

    if "Spline-CLT" in all_metrics and "Linear CLT" in all_metrics:
        print_comparison(all_metrics["Spline-CLT"], all_metrics["Linear CLT"])

    # Save as JSON (mse_per_layer is a list, rest are scalars)
    out_path = os.path.join(output_dir, "reconstruction_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved to {out_path}  [{_elapsed(t0)}]")
    return all_metrics


# --------------------------------------------------------------------------- #
# Stage 2: Circuit tracing                                                     #
# --------------------------------------------------------------------------- #

def run_circuits(
    kan_ckpt: str | None,
    lin_ckpt: str | None,
    output_dir: str,
    lm_name: str,
    max_features: int,
    run_shapley: bool,
    shapley_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Run circuit tracing on benchmark prompts for each checkpoint.

    Uses the original circuit_tracer backward-pass attribution pipeline,
    which captures attention-mediated cross-position effects and proper
    multi-layer gradient flow.
    """
    from spline_clt.kan_transcoder import load_spline_clt
    from circuit_tracer.attribution.attribute_transformerlens import attribute
    from circuit_tracer.replacement_model.replacement_model import ReplacementModel
    from experiments.run_circuit import print_circuit_summary

    _section("Stage 2: Circuit Tracing")
    t0 = time.time()

    checkpoints = {}
    if _checkpoint_exists(kan_ckpt):
        checkpoints["kan"] = kan_ckpt
    if _checkpoint_exists(lin_ckpt):
        checkpoints["linear"] = lin_ckpt

    if not checkpoints:
        print("  No checkpoints found — skipping circuit tracing stage.")
        return {}

    circuit_dir = os.path.join(output_dir, "circuits")
    os.makedirs(circuit_dir, exist_ok=True)

    summary = {}

    for ckpt_label, ckpt_path in checkpoints.items():
        print(f"\nLoading {ckpt_label} checkpoint from {ckpt_path}...")
        clt = load_spline_clt(ckpt_path, device=device, dtype=dtype)
        clt.eval()

        # Create a ReplacementModel once per checkpoint (wraps transformer + CLT)
        replacement_model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=lm_name,
            transcoders=clt,
            backend="transformerlens",
            device=device,
            dtype=dtype,
        )

        for prompt_cfg in CIRCUIT_PROMPTS:
            pname = prompt_cfg["name"]
            prompt = prompt_cfg["prompt"]
            out_key = f"{ckpt_label}/{pname}"
            print(f"\n  [{ckpt_label}] {pname}: {prompt!r}")

            try:
                tokens = replacement_model.to_str_tokens(prompt)
                graph = attribute(
                    prompt=prompt,
                    model=replacement_model,
                    max_feature_nodes=max_features,
                )
                selected = graph.selected_features
                n_active = len(selected)
                print(f"    {n_active} selected features")
                print_circuit_summary(graph, tokens, clt, top_k=10)

                result = {
                    "prompt": prompt,
                    "task": prompt_cfg["task"],
                    "n_active_features": n_active,
                    "tokens": tokens,
                    "active_features": graph.active_features[selected].tolist(),
                    "activation_values": graph.activation_values[selected].tolist(),
                }

                if run_shapley:
                    from attribution.shapley import shapley_attribution
                    print(f"    Running Shapley ({shapley_samples} samples)...")
                    # Collect MLP inputs for Shapley via the replacement model
                    input_ids = replacement_model.to_tokens(prompt).squeeze(0)
                    hook_in_names = [
                        f"blocks.{i}.{clt.feature_input_hook}"
                        for i in range(clt.n_layers)
                    ]
                    with torch.no_grad():
                        _, cache = replacement_model.run_with_cache(
                            input_ids, names_filter=hook_in_names
                        )
                    mlp_inputs = torch.stack(
                        [cache[name].squeeze(0) for name in hook_in_names]
                    ).to(device=device, dtype=dtype)

                    shap = shapley_attribution(
                        clt, mlp_inputs,
                        n_samples=shapley_samples,
                        max_features=min(max_features, 32),
                        antithetic=True,
                    )
                    result["shapley_values"] = shap["shapley_values"].tolist()
                    print(f"    Shapley done. Top value: {max(result['shapley_values']):.4f}")

                summary[out_key] = result
                torch.save(
                    result,
                    os.path.join(circuit_dir, f"{ckpt_label}_{pname}.pt"),
                )

            except Exception as e:
                print(f"    ERROR: {e}")
                summary[out_key] = {"error": str(e)}

        # Free VRAM between checkpoints
        del clt, replacement_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Save summary JSON (convert lists but not tensors)
    summary_path = os.path.join(output_dir, "circuit_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}  [{_elapsed(t0)}]")
    return summary


# --------------------------------------------------------------------------- #
# Stage 3: Spline analysis (KAN only)                                         #
# --------------------------------------------------------------------------- #

def run_splines(
    kan_ckpt: str | None,
    data_dir: str,
    output_dir: str,
    n_features: int,
    n_samples: int,
    device: torch.device,
    no_plot: bool,
) -> None:
    """Extract and save spline transfer functions for the top KAN features."""
    from spline_clt.kan_transcoder import load_spline_clt
    from spline_clt.training.data import ActivationDataset
    from experiments.analyze_splines import (
        collect_feature_stats,
        extract_spline_curve,
        top_input_dims,
        save_curves_csv,
    )

    _section("Stage 3: Spline Analysis")
    t0 = time.time()

    if not _checkpoint_exists(kan_ckpt):
        print("  No KAN checkpoint found — skipping spline analysis.")
        return

    model = load_spline_clt(kan_ckpt, device=device, dtype=torch.float32)
    model.eval()

    if model.encoder_type != "kan":
        print(f"  encoder_type={model.encoder_type!r} — spline analysis only for KAN. Skipping.")
        return

    print(f"Loading dataset ({n_samples} samples)...")
    dataset = ActivationDataset.load(data_dir, max_samples=n_samples + 50)

    spline_dir = os.path.join(output_dir, "splines")
    os.makedirs(spline_dir, exist_ok=True)

    stats = collect_feature_stats(model, dataset, device, torch.float32, n_samples=n_samples)
    freq = stats["activation_frequency"]
    flat_freq = freq.reshape(-1)
    top_indices = flat_freq.topk(n_features).indices
    top_features = [
        (int(i // model.d_transcoder), int(i % model.d_transcoder)) for i in top_indices
    ]

    HAS_MPL = False
    if not no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            HAS_MPL = True
        except ImportError:
            pass

    print(f"Extracting spline curves for top {n_features} features...")
    for rank, (layer_id, feature_id) in enumerate(top_features):
        f_freq = freq[layer_id, feature_id].item()
        try:
            dims = top_input_dims(model, layer_id, feature_id, top_k=3)
            curves = {}
            for dim in dims:
                t_vals, curve = extract_spline_curve(model, layer_id, feature_id, dim, device=device)
                curves[dim] = curve
            save_curves_csv(spline_dir, layer_id, feature_id, t_vals, curves)

            if HAS_MPL:
                fig, ax = plt.subplots(figsize=(7, 3))
                for dim, curve in curves.items():
                    ax.plot(t_vals, curve, label=f"dim {dim}")
                ax.axhline(0, color="k", linewidth=0.5)
                ax.set_title(f"L{layer_id} F{feature_id}  freq={f_freq:.3f}")
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(os.path.join(spline_dir, f"layer{layer_id}_feat{feature_id}.png"), dpi=90)
                plt.close(fig)

            if (rank + 1) % 5 == 0:
                print(f"  {rank+1}/{n_features} done...")
        except Exception as e:
            print(f"  L{layer_id} F{feature_id}: error — {e}")

    print(f"Saved to {spline_dir}/  [{_elapsed(t0)}]")


# --------------------------------------------------------------------------- #
# Stage 4: Monosemanticity                                                    #
# --------------------------------------------------------------------------- #

def run_monosemanticity(
    kan_ckpt: str | None,
    lin_ckpt: str | None,
    data_dir: str,
    output_dir: str,
    n_features: int,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Compute Gini coefficients and collect max-activating examples."""
    from spline_clt.kan_transcoder import load_spline_clt
    from spline_clt.training.data import ActivationDataset
    from eval.monosemanticity import collect_max_activating_examples, save_reports, print_summary

    _section("Stage 4: Monosemanticity")
    t0 = time.time()

    checkpoints = {}
    if _checkpoint_exists(kan_ckpt):
        checkpoints["kan"] = kan_ckpt
    if _checkpoint_exists(lin_ckpt):
        checkpoints["linear"] = lin_ckpt

    if not checkpoints:
        print("  No checkpoints found — skipping monosemanticity stage.")
        return {}

    print(f"Loading dataset ({n_samples} samples)...")
    dataset = ActivationDataset.load(data_dir, max_samples=n_samples + 50)

    mono_dir = os.path.join(output_dir, "monosemanticity")
    os.makedirs(mono_dir, exist_ok=True)

    comparison = {}
    for ckpt_label, ckpt_path in checkpoints.items():
        print(f"\nLoading {ckpt_label} from {ckpt_path}...")
        model = load_spline_clt(ckpt_path, device=device, dtype=dtype)
        model.eval()

        reports = collect_max_activating_examples(
            model, dataset,
            top_n_features=n_features,
            top_k_examples=10,
            n_samples=n_samples,
            device=device,
            dtype=dtype,
        )
        print_summary(reports, top_n=15)

        json_path = os.path.join(mono_dir, f"{ckpt_label}_features.json")
        save_reports(reports, json_path)

        ginis = [r.gini_coefficient for r in reports]
        comparison[ckpt_label] = {
            "mean_gini": sum(ginis) / len(ginis),
            "high_gini_count": sum(1 for g in ginis if g > 0.8),
            "n_features": len(reports),
        }

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if len(comparison) == 2:
        kan_g = comparison["kan"]["mean_gini"]
        lin_g = comparison["linear"]["mean_gini"]
        print(f"\nGini comparison: KAN={kan_g:.4f}  Linear={lin_g:.4f}  Δ={kan_g-lin_g:+.4f}")

    print(f"\nSaved to {mono_dir}/  [{_elapsed(t0)}]")
    return comparison


# --------------------------------------------------------------------------- #
# Diagnostic: Encoder weight norms                                             #
# --------------------------------------------------------------------------- #

def run_weight_norms(
    kan_ckpt: str | None,
    lin_ckpt: str | None,
    output_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Print per-layer encoder L2 norms to diagnose feature collapse.

    If a single layer dominates (>50% of total norm), the encoder has likely
    collapsed — all features are being driven by one layer's activations.
    Common fix: lower lambda_sparsity or learning rate.
    """
    from spline_clt.kan_transcoder import load_spline_clt

    _section("Diagnostic: Encoder Weight Norms per Layer")
    t0 = time.time()

    checkpoints = {}
    if _checkpoint_exists(kan_ckpt):
        checkpoints["Spline-CLT"] = kan_ckpt
    if _checkpoint_exists(lin_ckpt):
        checkpoints["Linear CLT"] = lin_ckpt

    if not checkpoints:
        print("  No checkpoints found — skipping weight norm diagnostic.")
        return

    norms_report = {}
    for label, ckpt in checkpoints.items():
        print(f"\nLoading {label} from {ckpt}...")
        model = load_spline_clt(ckpt, device=device, dtype=dtype)
        model.eval()

        print(f"\n  {label} — encoder L2 norm per layer")
        print(f"  {'Layer':<8} {'Encoder norm':>14}")
        print(f"  {'─'*24}")

        with torch.no_grad():
            layer_norms = []
            for layer_id in range(model.n_layers):
                enc = model.encoders[layer_id]
                norm = sum(p.norm(2).item() ** 2 for p in enc.parameters()) ** 0.5
                layer_norms.append(norm)
                print(f"  {layer_id:<8} {norm:>14.4f}")

        norms_report[label] = layer_norms
        total = sum(layer_norms) + 1e-9
        max_layer = max(range(len(layer_norms)), key=lambda i: layer_norms[i])
        max_frac = layer_norms[max_layer] / total
        if max_frac > 0.5:
            print(f"\n  WARNING: Layer {max_layer} dominates ({max_frac*100:.1f}% of total norm).")
            print(f"  Feature collapse suspected — try lowering --lambda-sparsity.")
        else:
            print(f"\n  OK: Layer {max_layer} is largest at {max_frac*100:.1f}% of total norm.")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, "weight_norms.json")
    with open(out_path, "w") as f:
        json.dump(norms_report, f, indent=2)
    print(f"\nSaved to {out_path}  [{_elapsed(t0)}]")


# --------------------------------------------------------------------------- #
# Diagnostic: Checkpoint loss progression                                      #
# --------------------------------------------------------------------------- #

def run_checkpoint_scan(
    checkpoint_dir: str,
    run_name: str,
    data_dir: str,
    n_samples: int,
    output_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Evaluate all saved step checkpoints to show whether loss is still decreasing.

    Loads every *_step<N> checkpoint in checkpoint_dir (plus _best), evaluates
    reconstruction MSE and cosine similarity on a small sample, and prints a
    table sorted by step number.  If MSE plateaued early, the model is
    undertrained or lambda_sparsity is too high.
    """
    import glob as _glob
    from spline_clt.kan_transcoder import load_spline_clt
    from spline_clt.training.data import ActivationDataset
    from experiments.compare_models import evaluate_model

    _section(f"Diagnostic: Loss Progression — {run_name}")
    t0 = time.time()

    # Collect step checkpoints sorted by step number, then append _best
    pattern = os.path.join(checkpoint_dir, f"{run_name}_step*")
    def _step_num(p: str) -> int:
        try:
            return int(p.rsplit("step", 1)[-1])
        except ValueError:
            return -1

    step_dirs = sorted(_glob.glob(pattern), key=_step_num)
    best_dir = os.path.join(checkpoint_dir, f"{run_name}_best")
    if os.path.isdir(best_dir):
        step_dirs.append(best_dir)

    if not step_dirs:
        print(f"  No step checkpoints found in {checkpoint_dir!r} — skipping.")
        return

    print(f"  Found {len(step_dirs)} checkpoint(s). Loading dataset ({n_samples} samples)...")
    dataset = ActivationDataset.load(data_dir, max_samples=n_samples + 50)

    results = []
    print(f"\n  {'Checkpoint':<38} {'MSE':>10} {'Cos Sim':>10} {'Active/pos':>12}")
    print(f"  {'─'*72}")

    for ckpt_path in step_dirs:
        name = os.path.basename(ckpt_path)
        try:
            model = load_spline_clt(ckpt_path, device=device, dtype=dtype)
            model.eval()
            m = evaluate_model(model, dataset, device, dtype, n_samples=n_samples)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  {name:<38} {m['mse_total']:>10.4f} "
                  f"{m['cosine_similarity']:>10.4f} {m['active_per_pos']:>12.2f}")
            results.append({
                "checkpoint": name,
                **{k: v for k, v in m.items() if k != "mse_per_layer"},
            })
        except Exception as e:
            print(f"  {name:<38} ERROR: {e}")

    if len(results) >= 2:
        first_mse = results[0]["mse_total"]
        last_mse = results[-2]["mse_total"] if results[-1]["checkpoint"].endswith("best") \
            else results[-1]["mse_total"]
        if last_mse >= first_mse * 0.99:
            print(f"\n  WARNING: MSE did not improve significantly ({first_mse:.4f} → {last_mse:.4f}).")
            print(f"  Model may be undertrained or lambda_sparsity is too high.")
        else:
            pct = (first_mse - last_mse) / first_mse * 100
            print(f"\n  MSE improved by {pct:.1f}% over training ({first_mse:.4f} → {last_mse:.4f}).")

    out_path = os.path.join(output_dir, f"checkpoint_scan_{run_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}  [{_elapsed(t0)}]")


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #

def write_report(
    output_dir: str,
    recon_metrics: dict,
    circuit_summary: dict,
    mono_comparison: dict,
    args,
) -> str:
    """Write a Markdown summary report."""
    report_path = os.path.join(output_dir, "report.md")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Spline-CLT Evaluation Report",
        f"\nGenerated: {now}",
        f"Device: {args.device}  |  dtype: {args.dtype}",
        "",
    ]

    if args.kan_checkpoint:
        lines.append(f"**Spline-CLT**: `{args.kan_checkpoint}`")
    if args.linear_checkpoint:
        lines.append(f"**Linear CLT**: `{args.linear_checkpoint}`")
    lines.append("")

    # Reconstruction
    if recon_metrics:
        lines += ["## Reconstruction Metrics", ""]
        header = f"| Model | MSE | Cosine Sim | Rel. Error | Active/pos | Params |"
        sep    = f"|-------|-----|-----------|------------|------------|--------|"
        lines += [header, sep]
        for label, m in recon_metrics.items():
            lines.append(
                f"| {label} | {m['mse_total']:.6f} | {m['cosine_similarity']:.4f} "
                f"| {m['relative_error']:.4f} | {m['active_per_pos']:.2f} | {m['n_params']:,} |"
            )
        lines.append("")

    # Circuit tracing summary
    if circuit_summary:
        lines += ["## Circuit Tracing Results", ""]
        header = f"| Model | Prompt | Task | Active Features |"
        sep    = f"|-------|--------|------|----------------|"
        lines += [header, sep]
        for key, v in circuit_summary.items():
            if "error" in v:
                lines.append(f"| {key} | — | — | ERROR: {v['error']} |")
            else:
                ckpt_label, pname = key.split("/", 1)
                lines.append(
                    f"| {ckpt_label} | `{v['prompt']}` | {v['task']} | {v['n_active_features']} |"
                )
        lines.append("")

    # Monosemanticity
    if mono_comparison:
        lines += ["## Monosemanticity (Gini Coefficient)", ""]
        header = f"| Model | Mean Gini | Features with Gini > 0.8 |"
        sep    = f"|-------|-----------|--------------------------|"
        lines += [header, sep]
        for label, m in mono_comparison.items():
            lines.append(
                f"| {label} | {m['mean_gini']:.4f} | {m['high_gini_count']}/{m['n_features']} |"
            )
        lines.append("")

    lines += [
        "## Files",
        "```",
        f"{output_dir}/",
        "  report.md                   # this file",
        "  reconstruction_metrics.json # per-model MSE, cosine sim, sparsity",
        "  circuit_summary.json        # active features per prompt",
        "  circuits/                   # .pt files for each (model, prompt) pair",
        "  splines/                    # spline curve CSVs and PNGs (KAN only)",
        "  monosemanticity/            # feature JSON reports per model",
        "```",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spline-CLT full evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoints
    parser.add_argument("--kan-checkpoint", type=str, default=None,
                        help="Spline-CLT checkpoint directory")
    parser.add_argument("--linear-checkpoint", type=str, default=None,
                        help="Linear CLT checkpoint directory")

    # Data & output
    parser.add_argument("--data-dir", type=str, default="data/activations")
    parser.add_argument("--output-dir", type=str, default="results/pipeline_run")
    parser.add_argument("--lm-model", type=str, default="gpt2",
                        help="TransformerLens model name (for circuit tracing)")

    # Compute
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16"])

    # Stage controls
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--skip-circuits", action="store_true")
    parser.add_argument("--skip-splines", action="store_true")
    parser.add_argument("--skip-monosemanticity", action="store_true")

    # Stage-specific options
    parser.add_argument("--n-eval-samples", type=int, default=200,
                        help="Sequences for reconstruction/mono eval")
    parser.add_argument("--max-features", type=int, default=64,
                        help="Max active features for circuit tracing")
    parser.add_argument("--shapley", action="store_true",
                        help="Run Shapley attribution during circuit tracing (slow)")
    parser.add_argument("--shapley-samples", type=int, default=64)
    parser.add_argument("--n-spline-features", type=int, default=20,
                        help="Number of top features for spline analysis")
    parser.add_argument("--n-mono-features", type=int, default=100,
                        help="Number of top features for monosemanticity analysis")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib (headless / no display)")

    args = parser.parse_args()

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # At least one checkpoint required
    if not _checkpoint_exists(args.kan_checkpoint) and not _checkpoint_exists(args.linear_checkpoint):
        print("ERROR: provide at least one of --kan-checkpoint or --linear-checkpoint")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    t_total = time.time()

    print(f"\nSpline-CLT Evaluation Pipeline")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Device     : {device}  |  dtype: {dtype}")
    if args.kan_checkpoint:
        print(f"  KAN ckpt   : {args.kan_checkpoint}")
    if args.linear_checkpoint:
        print(f"  Linear ckpt: {args.linear_checkpoint}")

    # --- Stage 1: Reconstruction ---
    recon_metrics = {}
    if not args.skip_reconstruction:
        recon_metrics = run_reconstruction(
            kan_ckpt=args.kan_checkpoint,
            lin_ckpt=args.linear_checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_samples=args.n_eval_samples,
            device=device,
            dtype=dtype,
        )

    # --- Stage 2: Circuit tracing ---
    circuit_summary = {}
    if not args.skip_circuits:
        circuit_summary = run_circuits(
            kan_ckpt=args.kan_checkpoint,
            lin_ckpt=args.linear_checkpoint,
            output_dir=args.output_dir,
            lm_name=args.lm_model,
            max_features=args.max_features,
            run_shapley=args.shapley,
            shapley_samples=args.shapley_samples,
            device=device,
            dtype=dtype,
        )

    # --- Stage 3: Spline analysis ---
    if not args.skip_splines:
        run_splines(
            kan_ckpt=args.kan_checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_features=args.n_spline_features,
            n_samples=args.n_eval_samples,
            device=device,
            no_plot=args.no_plot,
        )

    # --- Stage 4: Monosemanticity ---
    mono_comparison = {}
    if not args.skip_monosemanticity:
        mono_comparison = run_monosemanticity(
            kan_ckpt=args.kan_checkpoint,
            lin_ckpt=args.linear_checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_features=args.n_mono_features,
            n_samples=args.n_eval_samples,
            device=device,
            dtype=dtype,
        )

    # --- Write report ---
    _section("Summary Report")
    report_path = write_report(
        args.output_dir, recon_metrics, circuit_summary, mono_comparison, args
    )
    print(f"Report written to {report_path}")
    print(f"\nTotal pipeline time: {_elapsed(t_total)}")


if __name__ == "__main__":
    main()
