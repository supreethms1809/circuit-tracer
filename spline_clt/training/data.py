"""Activation dataset collection for Spline-CLT training.

Runs a transformer model on text data and caches residual stream activations
and MLP outputs at all layers. These cached activations are used to train
the Spline-CLT to reconstruct MLP outputs.
"""

import os
from dataclasses import dataclass

import numpy as np
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
    seed: int = 0


class ActivationDataset(Dataset):
    """Dataset of cached transformer activations for Spline-CLT training.

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

        Supports two on-disk formats:
        - **numpy format** (new, preferred): ``mlp_inputs.npy`` / ``mlp_outputs.npy``
          stored as int16 (bfloat16 bits reinterpreted).  Data is memory-mapped
          from disk — no RAM pre-allocation required regardless of dataset size.
        - **torch format** (legacy): ``mlp_inputs.pt`` / ``mlp_outputs.pt``
          loaded with ``mmap=True`` to avoid touching the full file at once.

        In both cases a bounded ``max_samples`` slice is copied into contiguous
        RAM to avoid OS-reclaim SIGBUS on WSL2 / Hyper-V.

        Args:
            path: Directory containing the activation files.
            max_samples: Number of sequences to load into RAM.
                Default: 3000 (~14 GB for GPT-2 small).
        """
        if max_samples is None:
            max_samples = 3000

        inputs_npy = os.path.join(path, "mlp_inputs.npy")
        outputs_npy = os.path.join(path, "mlp_outputs.npy")

        if os.path.exists(inputs_npy) and os.path.exists(outputs_npy):
            # Numpy format: int16 on disk, reinterpreted as bfloat16.
            inputs_mm = np.load(inputs_npy, mmap_mode="r")
            outputs_mm = np.load(outputs_npy, mmap_mode="r")
            n = min(max_samples, inputs_mm.shape[0])
            # Convert mmap slice → contiguous int16 → bfloat16 torch tensor.
            mlp_inputs = torch.from_numpy(inputs_mm[:n].copy()).view(torch.bfloat16)
            mlp_outputs = torch.from_numpy(outputs_mm[:n].copy()).view(torch.bfloat16)
        else:
            # Legacy torch format: mmap to avoid touching all bytes at once,
            # then copy a bounded subset into contiguous RAM.
            mlp_inputs_mm = torch.load(
                os.path.join(path, "mlp_inputs.pt"),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            mlp_outputs_mm = torch.load(
                os.path.join(path, "mlp_outputs.pt"),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            n = min(max_samples, mlp_inputs_mm.shape[0])
            mlp_inputs = mlp_inputs_mm[:n].clone()
            mlp_outputs = mlp_outputs_mm[:n].clone()

        return cls(mlp_inputs, mlp_outputs)


@torch.no_grad()
def collect_activations(config: DataConfig) -> ActivationDataset:
    """Run a transformer on text data and collect MLP input/output activations.

    Uses TransformerLens HookedTransformer for hook-based activation extraction.
    Activations are written directly to memory-mapped numpy files on disk so that
    the pre-allocation RAM cost is O(batch) rather than O(dataset), avoiding OOM
    on large models (e.g. Qwen2.5-0.5B with 24 layers where a full in-RAM
    pre-allocation would exceed 170 GB).

    Args:
        config: Data collection configuration.

    Returns:
        ActivationDataset with cached activations (backed by the saved files).
    """
    from transformer_lens import HookedTransformer
    from datasets import load_dataset

    if config.device:
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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

    # Load dataset from HuggingFace Hub (supports any Hub repo or Parquet-based dataset).
    load_kwargs = {"path": config.dataset_name, "split": "train"}
    if config.dataset_config:
        load_kwargs["name"] = config.dataset_config
    dataset = load_dataset(**load_kwargs)

    dataset = dataset.shuffle(seed=config.seed)
    tokenizer = model.tokenizer
    assert tokenizer is not None

    # Tokenize — chunk each sample into non-overlapping seq_len windows to use
    # all tokens, not just the first seq_len.  Codelion samples average ~1K tokens;
    # truncating to 128 would discard ~88% of the data.
    # Batches are formed on the fly to avoid holding all tokens in RAM at once.
    n_sequences = config.n_tokens // config.seq_len
    token_batches = []
    current_batch = []
    collected = 0

    for item in dataset:
        tokens = tokenizer(
            item["text"],
            truncation=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        # Chunk into non-overlapping windows of seq_len
        for start in range(0, len(tokens) - config.seq_len + 1, config.seq_len):
            current_batch.append(tokens[start : start + config.seq_len])
            if len(current_batch) == config.batch_size:
                token_batches.append(torch.stack(current_batch))
                current_batch = []
                collected += config.batch_size
            if collected >= n_sequences:
                break
        if collected >= n_sequences:
            break

    # Flush remaining partial batch
    if current_batch and collected < n_sequences:
        token_batches.append(torch.stack(current_batch))

    n_sequences = sum(b.shape[0] for b in token_batches)

    # Pre-compute hook names once
    hook_names_in = [
        f"blocks.{i}.{config.feature_input_hook}" for i in range(n_layers)
    ]
    hook_names_out = [
        f"blocks.{i}.{config.feature_output_hook}" for i in range(n_layers)
    ]

    # Write activations directly to memory-mapped numpy files (int16 = bfloat16 bits).
    # This eliminates the in-RAM pre-allocation that caused OOM for large models:
    #   GPT-2 small (12 L, 768 d):  2 × ~18 GB = ~36 GB in RAM  — historically fine
    #   Qwen2.5-0.5B (24 L, 896 d): 2 × ~86 GB = ~172 GB in RAM — OOM on 121 GB node
    # With mmap the RAM cost per step is only the current batch cache (~1–2 GB).
    os.makedirs(config.save_dir, exist_ok=True)
    inputs_npy = os.path.join(config.save_dir, "mlp_inputs.npy")
    outputs_npy = os.path.join(config.save_dir, "mlp_outputs.npy")
    shape = (n_sequences, n_layers, config.seq_len, d_model)

    # np.lib.format.open_memmap writes a .npy header (shape + dtype) then maps the
    # data region — readable later via np.load(path, mmap_mode='r').
    mlp_inputs_mm = np.lib.format.open_memmap(
        inputs_npy, dtype=np.int16, mode="w+", shape=shape
    )
    mlp_outputs_mm = np.lib.format.open_memmap(
        outputs_npy, dtype=np.int16, mode="w+", shape=shape
    )

    sample_idx = 0
    for batch_tokens in tqdm(token_batches, desc="Collecting activations"):
        batch_tokens = batch_tokens.to(device)
        actual_bs = batch_tokens.shape[0]

        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter=hook_names_in + hook_names_out,
        )

        # Stack per-layer activations: (batch, n_layers, seq_len, d_model)
        mlp_in = torch.stack([cache[name] for name in hook_names_in], dim=1)
        mlp_out = torch.stack([cache[name] for name in hook_names_out], dim=1)

        # Convert bfloat16 → int16 (same 16-bit bit pattern) for numpy storage.
        mlp_inputs_mm[sample_idx : sample_idx + actual_bs] = (
            mlp_in.cpu().bfloat16().contiguous().view(torch.int16).numpy()
        )
        mlp_outputs_mm[sample_idx : sample_idx + actual_bs] = (
            mlp_out.cpu().bfloat16().contiguous().view(torch.int16).numpy()
        )
        sample_idx += actual_bs

    # Flush mmap writes to disk before returning.
    mlp_inputs_mm.flush()
    mlp_outputs_mm.flush()
    del mlp_inputs_mm, mlp_outputs_mm

    # Load via the standard path so callers get a consistent ActivationDataset.
    # Uses max_samples=n_sequences to return all collected data.
    return ActivationDataset.load(config.save_dir, max_samples=n_sequences)
