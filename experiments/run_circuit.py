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

from circuit_tracer.attribution.attribute_transformerlens import attribute
from circuit_tracer.replacement_model.replacement_model import ReplacementModel
from spline_clt.kan_transcoder import KANCrossLayerTranscoder, load_spline_clt


def print_circuit_summary(
    graph,
    tokens: list[str],
    model: KANCrossLayerTranscoder,
    top_k: int = 20,
) -> None:
    """Print a human-readable summary of the top features in the circuit.

    Args:
        graph: A ``circuit_tracer.graph.Graph`` object.
        tokens: List of string tokens for the prompt.
        model: The Spline-CLT (used only for metadata).
        top_k: Number of top features to display.
    """
    # Graph stores all active features; selected_features indexes the subset
    # included in the adjacency matrix.
    all_active = graph.active_features  # (total_active, 3)
    all_values = graph.activation_values  # (total_active,)
    selected_idx = graph.selected_features  # indices into all_active

    active = all_active[selected_idx]  # (n_selected, 3)
    values = all_values[selected_idx]  # (n_selected,)
    n_active = len(active)

    print(f"\n{'='*60}")
    print(f"Circuit summary: {n_active} selected features "
          f"(of {len(all_active)} total active)")
    print(f"{'='*60}")

    # Sort by activation magnitude
    sorted_idx = values.abs().argsort(descending=True)

    print(f"\nTop {min(top_k, n_active)} features (by activation magnitude):")
    print(f"  {'Rank':<5} {'Layer':<6} {'Pos':<5} {'FeatIdx':<8} {'Activation':>12} {'Token':>12}")
    print(f"  {'-'*55}")
    for rank in range(min(top_k, n_active)):
        i = sorted_idx[rank]
        layer, pos, feat = active[i, 0].item(), active[i, 1].item(), active[i, 2].item()
        act_val = values[i].item()
        tok = tokens[pos] if pos < len(tokens) else "?"
        print(f"  {rank+1:<5} {layer:<6} {pos:<5} {feat:<8} {act_val:>12.4f} {tok!r:>12}")

    # Show top feature-to-feature edges from the adjacency matrix
    adj = graph.adjacency_matrix
    ff_block = adj[:n_active, :n_active]
    if ff_block.numel() > 0:
        flat_effects = ff_block.abs().flatten()
        k = min(10, len(flat_effects))
        top_connections = flat_effects.topk(k).indices
        print(f"\nTop {k} feature-to-feature edges:")
        print(f"  {'Source':>8} {'Target':>8} {'Effect':>12}")
        print(f"  {'-'*30}")
        seen = set()
        for flat_idx in top_connections:
            tgt = int(flat_idx // n_active)  # adj is [target, source]
            src = int(flat_idx % n_active)
            if src == tgt or (src, tgt) in seen:
                continue
            seen.add((src, tgt))
            sl, sp, sf = active[src, 0].item(), active[src, 1].item(), active[src, 2].item()
            tl, tp, tf = active[tgt, 0].item(), active[tgt, 1].item(), active[tgt, 2].item()
            effect = ff_block[tgt, src].item()
            print(f"  L{sl}F{sf}@{sp} -> L{tl}F{tf}@{tp}  {effect:>12.4f}")
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

    # --- Create ReplacementModel (wraps transformer + CLT for full backward-pass attribution) ---
    print(f"Creating ReplacementModel with {args.model}...")
    replacement_model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=args.model,
        transcoders=clt,
        backend="transformerlens",
        device=device,
        dtype=dtype,
    )

    print(f"\nPrompt: {args.prompt!r}")
    tokens = replacement_model.to_str_tokens(args.prompt)
    print(f"Tokens ({len(tokens)}): {tokens}")

    # --- Attribution via original circuit_tracer backward pass ---
    print(f"\nRunning attribution (max_features={args.max_features})...")
    graph = attribute(
        prompt=args.prompt,
        model=replacement_model,
        max_feature_nodes=args.max_features,
        verbose=True,
    )
    print_circuit_summary(graph, tokens, clt)

    results = {
        "prompt": args.prompt,
        "tokens": tokens,
        "graph": graph,
    }

    # --- Shapley attribution (optional) ---
    if args.shapley:
        print(f"\nRunning Shapley attribution ({args.shapley_samples} samples)...")
        # Collect activations for Shapley (which operates on the CLT directly)
        input_ids = replacement_model.to_tokens(args.prompt).squeeze(0)
        hook_in_names = [
            f"blocks.{i}.{clt.feature_input_hook}" for i in range(clt.n_layers)
        ]
        with torch.no_grad():
            _, cache = replacement_model.run_with_cache(
                input_ids, names_filter=hook_in_names
            )
        mlp_inputs = torch.stack(
            [cache[name].squeeze(0) for name in hook_in_names]
        ).to(device=device, dtype=dtype)

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
