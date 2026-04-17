#!/usr/bin/env python3
"""Main training script for Spline-CLT.

Usage:
    # Train with default config
    python experiments/train_spline_clt.py

    # Train with custom config
    python experiments/train_spline_clt.py --config experiments/configs/gpt2_small.yaml

    # Collect activations first
    python experiments/train_spline_clt.py --collect-data --model gpt2

    # Train with specific hyperparameters
    python experiments/train_spline_clt.py --config experiments/configs/gpt2_small.yaml \\
        --lambda-sparsity 0.1 --grid-size 3
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spline_clt.training.train import TrainConfig, train, load_config
from spline_clt.training.data import DataConfig, collect_activations, ActivationDataset


def main():
    parser = argparse.ArgumentParser(description="Train Spline-CLT")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file",
    )
    parser.add_argument("--collect-data", action="store_true", help="Collect activations first")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name for data collection")
    parser.add_argument("--n-tokens", type=int, default=None, help="Number of tokens to collect")
    parser.add_argument("--encoder-type", type=str, default=None,
                        choices=["kan", "linear"], help="Encoder type (overrides config)")
    parser.add_argument("--lambda-sparsity", type=float, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainConfig()

    # Override with CLI args
    if args.encoder_type is not None:
        config.encoder_type = args.encoder_type
    if args.lambda_sparsity is not None:
        config.lambda_sparsity = args.lambda_sparsity
    if args.grid_size is not None:
        config.grid_size = args.grid_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.total_steps is not None:
        config.total_steps = args.total_steps
    if args.device is not None:
        config.device = args.device

    # Collect activations if requested
    if args.collect_data:
        print(f"Collecting activations from {args.model}...")
        data_config = DataConfig(
            model_name=args.model,
            save_dir=config.data_dir,
            device=config.device,
        )
        if args.n_tokens is not None:
            data_config.n_tokens = args.n_tokens

        dataset = collect_activations(data_config)
        if dataset is None:
            print(f"Collected activations and saved to {config.data_dir}")
            if not args.config:
                raise ValueError(
                    "Config is required when load_after_collect=False because "
                    "model dimensions cannot be inferred from an in-memory dataset."
                )
        else:
            print(f"Collected {len(dataset)} samples, saved to {config.data_dir}")

            if not args.config:
                # Infer model dims from data
                config.n_layers = dataset.mlp_inputs.shape[1]
                config.d_model = dataset.mlp_inputs.shape[3]
    else:
        dataset = None

    # Train
    print(f"Training Spline-CLT: {config.d_transcoder} features, "
          f"grid_size={config.grid_size}, λ={config.lambda_sparsity}")
    print(f"Device: {config.device}, Steps: {config.total_steps}")

    model = train(config, dataset=dataset)
    print(f"Training complete. Model saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
