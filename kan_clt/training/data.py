"""Activation dataset collection for KAN-CLT training.

Runs a transformer model on text data and caches residual stream activations
and MLP outputs at all layers. These cached activations are used to train
the KAN-CLT to reconstruct MLP outputs.
"""

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class DataConfig:
    """Configuration for activation dataset collection."""

    model_name: str = "gpt2"
    dataset_name: str = "Salesforce/wikitext"  # uses Parquet, no deprecated scripts
    dataset_config: str = "wikitext-103-raw-v1"  # full corpus (~1.8M rows, enough for 100K+ sequences)
    n_tokens: int = 10_000_000
    seq_len: int = 128
    batch_size: int = 32
    save_dir: str = "data/activations"
    feature_input_hook: str = "hook_resid_mid"
    feature_output_hook: str = "hook_mlp_out"
    device: str = "cuda"


class ActivationDataset(Dataset):
    """Dataset of cached transformer activations for KAN-CLT training.

    Each item is a dict with:
        - mlp_inputs: (n_layers, seq_len, d_model) — residual stream before MLP
        - mlp_outputs: (n_layers, seq_len, d_model) — MLP outputs

    Can be created by collecting activations from a model, or loaded from disk.

    Args:
        mlp_inputs: Tensor of shape (n_samples, n_layers, seq_len, d_model).
        mlp_outputs: Tensor of shape (n_samples, n_layers, seq_len, d_model).
    """

    def __init__(
        self,
        mlp_inputs: torch.Tensor,
        mlp_outputs: torch.Tensor,
    ):
        assert mlp_inputs.shape == mlp_outputs.shape
        self.mlp_inputs = mlp_inputs
        self.mlp_outputs = mlp_outputs

    def __len__(self) -> int:
        return self.mlp_inputs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "mlp_inputs": self.mlp_inputs[idx],
            "mlp_outputs": self.mlp_outputs[idx],
        }

    def save(self, path: str) -> None:
        """Save dataset to disk as memory-mapped tensors."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.mlp_inputs, os.path.join(path, "mlp_inputs.pt"))
        torch.save(self.mlp_outputs, os.path.join(path, "mlp_outputs.pt"))

    @classmethod
    def load(cls, path: str, max_samples: int | None = None) -> "ActivationDataset":
        """Load dataset from disk.

        Uses mmap to avoid touching 40 GB at once, then copies a subset into
        contiguous RAM. mmap=True on WSL2 causes SIGBUS when Hyper-V reclaims
        pages under memory pressure; copying a bounded subset avoids this.

        Args:
            path: Directory containing mlp_inputs.pt and mlp_outputs.pt.
            max_samples: If given, only the first max_samples sequences are
                loaded into RAM. Default: 3000 (~14 GB total for GPT-2 small).
        """
        if max_samples is None:
            max_samples = 3000

        # mmap to avoid touching all 40 GB at once during the slice copy
        mlp_inputs_mm = torch.load(
            os.path.join(path, "mlp_inputs.pt"), map_location="cpu", weights_only=True,
            mmap=True,
        )
        mlp_outputs_mm = torch.load(
            os.path.join(path, "mlp_outputs.pt"), map_location="cpu", weights_only=True,
            mmap=True,
        )
        n = min(max_samples, mlp_inputs_mm.shape[0])
        # .clone() copies the slice into a plain contiguous tensor — no mmap backing,
        # so no SIGBUS risk if the OS reclaims pages later.
        mlp_inputs = mlp_inputs_mm[:n].clone()
        mlp_outputs = mlp_outputs_mm[:n].clone()
        return cls(mlp_inputs, mlp_outputs)


@torch.no_grad()
def collect_activations(config: DataConfig) -> ActivationDataset:
    """Run a transformer on text data and collect MLP input/output activations.

    Uses TransformerLens HookedTransformer for hook-based activation extraction.

    Args:
        config: Data collection configuration.

    Returns:
        ActivationDataset with cached activations.
    """
    from transformer_lens import HookedTransformer
    from datasets import load_dataset

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = HookedTransformer.from_pretrained(
        config.model_name,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Load dataset (use Parquet-based datasets; script-based ones like stas/openwebtext-10k
    # are no longer supported by Hugging Face)
    load_kwargs = {"path": config.dataset_name, "split": "train"}
    if config.dataset_config:
        load_kwargs["name"] = config.dataset_config
    dataset = load_dataset(**load_kwargs)
    tokenizer = model.tokenizer
    assert tokenizer is not None

    # Tokenize
    n_sequences = config.n_tokens // config.seq_len
    all_tokens = []

    for item in dataset:
        tokens = tokenizer(
            item["text"],
            truncation=True,
            max_length=config.seq_len,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        if len(tokens) == config.seq_len:
            all_tokens.append(tokens)
        if len(all_tokens) >= n_sequences:
            break

    all_tokens = torch.stack(all_tokens[:n_sequences]).to(device)
    n_sequences = len(all_tokens)  # actual count (may be < config.n_tokens // seq_len)

    # Pre-compute hook names once
    hook_names_in = [
        f"blocks.{i}.{config.feature_input_hook}" for i in range(n_layers)
    ]
    hook_names_out = [
        f"blocks.{i}.{config.feature_output_hook}" for i in range(n_layers)
    ]

    # Preallocate output tensors in bfloat16 on CPU.
    # Avoids accumulating a list of float32 tensors and the subsequent torch.cat,
    # which would require 2× peak RAM (~160 GB for 8500 wikitext-2 sequences at float32).
    # bfloat16 + in-place fill keeps peak usage to ~20 GB for the same dataset.
    mlp_inputs = torch.empty(
        n_sequences, n_layers, config.seq_len, d_model, dtype=torch.bfloat16
    )
    mlp_outputs = torch.empty(
        n_sequences, n_layers, config.seq_len, d_model, dtype=torch.bfloat16
    )

    sample_idx = 0
    for batch_start in tqdm(
        range(0, n_sequences, config.batch_size),
        desc="Collecting activations",
    ):
        batch_tokens = all_tokens[batch_start : batch_start + config.batch_size]
        actual_bs = len(batch_tokens)

        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter=hook_names_in + hook_names_out,
        )

        # Stack per-layer activations: (batch, n_layers, seq_len, d_model)
        mlp_in = torch.stack([cache[name] for name in hook_names_in], dim=1)
        mlp_out = torch.stack([cache[name] for name in hook_names_out], dim=1)

        mlp_inputs[sample_idx : sample_idx + actual_bs] = mlp_in.cpu().bfloat16()
        mlp_outputs[sample_idx : sample_idx + actual_bs] = mlp_out.cpu().bfloat16()
        sample_idx += actual_bs

    dataset = ActivationDataset(mlp_inputs, mlp_outputs)

    # Save to disk
    os.makedirs(config.save_dir, exist_ok=True)
    dataset.save(config.save_dir)

    return dataset
