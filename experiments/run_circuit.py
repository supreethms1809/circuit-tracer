#!/usr/bin/env python3
"""End-to-end circuit tracing pipeline using Spline-CLT.

Runs:
  1. Load a trained Spline-CLT checkpoint
  2. Collect residual stream activations for a given prompt via TransformerLens
  3. Run causal ablation attribution to build the feature graph
  4. (Optional) Run Shapley attribution for selected features
  5. Print circuit summary and save results

Usage:
    python experiments/run_circuit.py \
        --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
        --prompt "The Eiffel Tower is located in" \
        --model gpt2 \
        --max-features 64 \
        [--shapley]       # run Shapley in addition to ablation
        [--output results/circuits/run1.pt]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spline_clt.kan_transcoder import KANCrossLayerTranscoder, load_spline_clt
from attribution.causal import build_attribution_graph


def collect_activations_for_prompt(
    model_name: str,
    prompt: str,
    n_layers: int,
    device: torch.device,
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
):
    """Run the language model and collect MLP inputs/outputs via TransformerLens.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2").
        prompt: Text prompt to run.
        n_layers: Number of layers to collect from.
        device: Compute device.
        feature_input_hook: Hook point name for MLP input.
        feature_output_hook: Hook point name for MLP output.

    Returns:
        Tuple of:
            mlp_inputs: (n_layers, seq_len, d_model)
            mlp_outputs: (n_layers, seq_len, d_model)
            tokens: List of string tokens
    """
    try:
        import transformer_lens
        from transformer_lens import HookedTransformer
    except ImportError:
        raise ImportError(
            "transformer_lens is required for circuit tracing. "
            "Install with: pip install transformer-lens"
        )

    print(f"Loading {model_name} with TransformerLens...")
    lm = HookedTransformer.from_pretrained(model_name, device=device)
    lm.eval()

    tokens = lm.to_str_tokens(prompt)
    hook_in_names = [f"blocks.{i}.{feature_input_hook}" for i in range(n_layers)]
    hook_out_names = [f"blocks.{i}.{feature_output_hook}" for i in range(n_layers)]

    print(f"Running forward pass: {len(tokens)} tokens...")
    with torch.no_grad():
        _, cache = lm.run_with_cache(
            prompt, names_filter=hook_in_names + hook_out_names
        )

    mlp_inputs = torch.stack(
        [cache[name].squeeze(0) for name in hook_in_names]
    ).to(device)   # (n_layers, seq_len, d_model)
    mlp_outputs = torch.stack(
        [cache[name].squeeze(0) for name in hook_out_names]
    ).to(device)

    del cache
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return mlp_inputs, mlp_outputs, tokens, lm


def print_circuit_summary(
    attribution_result: dict,
    tokens: list[str],
    model: KANCrossLayerTranscoder,
    top_k: int = 20,
) -> None:
    """Print a human-readable summary of the top features in the circuit."""
    active = attribution_result["active_features"]  # (n_active, 3)
    values = attribution_result["activation_values"]  # (n_active,)
    effects = attribution_result.get("feature_effects")  # (n_active, n_active) or None

    n_active = len(active)
    print(f"\n{'='*60}")
    print(f"Circuit summary: {n_active} active features")
    print(f"{'='*60}")

    # Sort by activation magnitude
    sorted_idx = values.abs().argsort(descending=True)

    print(f"\nTop {min(top_k, n_active)} active features (by activation magnitude):")
    print(f"  {'Rank':<5} {'Layer':<6} {'Pos':<5} {'FeatIdx':<8} {'Activation':>12} {'Token':>12}")
    print(f"  {'-'*55}")
    for rank in range(min(top_k, n_active)):
        i = sorted_idx[rank]
        layer, pos, feat = active[i, 0].item(), active[i, 1].item(), active[i, 2].item()
        act_val = values[i].item()
        tok = tokens[pos] if pos < len(tokens) else "?"
        print(f"  {rank+1:<5} {layer:<6} {pos:<5} {feat:<8} {act_val:>12.4f} {tok!r:>12}")

    if effects is not None:
        # Show top feature-to-feature connections by effect magnitude
        flat_effects = effects.abs().flatten()
        top_connections = flat_effects.topk(min(10, len(flat_effects))).indices
        print(f"\nTop {min(10, len(flat_effects))} feature-to-feature edges:")
        print(f"  {'Source':>8} {'Target':>8} {'Effect':>12}")
        print(f"  {'-'*30}")
        seen = set()
        for flat_idx in top_connections:
            src = int(flat_idx // n_active)
            tgt = int(flat_idx % n_active)
            if src == tgt or (src, tgt) in seen:
                continue
            seen.add((src, tgt))
            sl, sp, sf = active[src, 0].item(), active[src, 1].item(), active[src, 2].item()
            tl, tp, tf = active[tgt, 0].item(), active[tgt, 1].item(), active[tgt, 2].item()
            effect = effects[src, tgt].item()
            print(f"  L{sl}F{sf}@{sp} → L{tl}F{tf}@{tp}  {effect:>12.4f}")

    # Reconstruction quality
    baseline = attribution_result.get("baseline_reconstruction")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spline-CLT end-to-end circuit tracing")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Spline-CLT checkpoint directory")
    parser.add_argument(
        "--prompt", type=str,
        default="The Eiffel Tower is located in",
        help="Prompt to trace the circuit for",
    )
    parser.add_argument("--model", type=str, default="gpt2",
                        help="TransformerLens model name")
    parser.add_argument("--max-features", type=int, default=64,
                        help="Max features to include in attribution graph")
    parser.add_argument("--shapley", action="store_true",
                        help="Also run Shapley attribution (slow)")
    parser.add_argument("--shapley-samples", type=int, default=64,
                        help="Monte Carlo samples for Shapley (if --shapley)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results dict (.pt file)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16"])
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # --- Load Spline-CLT ---
    print(f"Loading Spline-CLT from {args.checkpoint}...")
    clt = load_spline_clt(args.checkpoint, device=device, dtype=dtype)
    clt.eval()
    print(f"  encoder_type={clt.encoder_type}, n_layers={clt.n_layers}, "
          f"d_transcoder={clt.d_transcoder}")

    # --- Collect activations ---
    mlp_inputs, mlp_outputs, tokens, lm = collect_activations_for_prompt(
        model_name=args.model,
        prompt=args.prompt,
        n_layers=clt.n_layers,
        device=device,
        feature_input_hook=clt.feature_input_hook,
        feature_output_hook=clt.feature_output_hook,
    )
    mlp_inputs = mlp_inputs.to(dtype=dtype)
    mlp_outputs = mlp_outputs.to(dtype=dtype)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"Activation shape: {list(mlp_inputs.shape)}")

    # --- Causal attribution ---
    print(f"\nRunning causal attribution (max_features={args.max_features})...")
    # Select top logit tokens for feature→logit edges
    logits = lm(args.prompt).squeeze(0)
    probs = torch.softmax(logits[-1].float(), dim=-1)
    top_logit_ids = probs.topk(8).indices
    attribution = build_attribution_graph(
        clt, mlp_inputs, max_features=args.max_features,
        W_U=lm.W_U, logit_token_ids=top_logit_ids,
    )
    print_circuit_summary(attribution, tokens, clt)

    results = {
        "prompt": args.prompt,
        "tokens": tokens,
        "attribution": attribution,
        "mlp_inputs": mlp_inputs.cpu(),
        "mlp_outputs": mlp_outputs.cpu(),
    }

    # --- Shapley attribution (optional) ---
    if args.shapley:
        print(f"\nRunning Shapley attribution ({args.shapley_samples} samples)...")
        from attribution.shapley import shapley_attribution
        shapley = shapley_attribution(
            clt,
            mlp_inputs,
            target="reconstruction",
            n_samples=args.shapley_samples,
            max_features=min(args.max_features, 32),  # Shapley is O(n^2 * samples)
            antithetic=True,
        )
        results["shapley"] = shapley

        # Print top Shapley features
        shap_vals = shapley["shapley_values"]
        active = shapley["active_features"]
        sorted_idx = shap_vals.abs().argsort(descending=True)
        print(f"\nTop Shapley features (contribution to reconstruction):")
        print(f"  {'Rank':<5} {'Layer':<6} {'Pos':<5} {'FeatIdx':<8} {'Shapley':>12} {'Token':>12}")
        print(f"  {'-'*55}")
        for rank in range(min(10, len(sorted_idx))):
            i = sorted_idx[rank]
            layer, pos, feat = active[i, 0].item(), active[i, 1].item(), active[i, 2].item()
            sv = shap_vals[i].item()
            tok = tokens[pos] if pos < len(tokens) else "?"
            print(f"  {rank+1:<5} {layer:<6} {pos:<5} {feat:<8} {sv:>12.6f} {tok!r:>12}")

    # --- Save results ---
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save(results, args.output)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n(Use --output path.pt to save results for downstream analysis.)")


if __name__ == "__main__":
    main()
