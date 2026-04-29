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
    #: Directory for the train-split memmaps. On a per-node setup this should
    #: be a local NVMe path (e.g. ``/lscratch/<model>/activations``) since the
    #: train cache is the bulk of the data and is regenerable.
    save_dir: str = "data/activations"
    #: Directory for the val-split memmaps. If unset, falls back to
    #: ``save_dir`` (legacy single-dir layout). For paper runs this should
    #: live on shared/persistent storage so val is reproducible across nodes
    #: and survives ephemeral local disks.
    val_save_dir: str | None = None
    #: Fraction of collected sequences routed into the val split. The split
    #: is materialised at collection time using ``seed`` for a reproducible
    #: permutation, so every consumer of these files sees the same partition.
    val_fraction: float = 0.05
    feature_input_hook: str = "hook_resid_mid"
    feature_output_hook: str = "hook_mlp_out"
    device: str = "cuda"
    seed: int = 0
    #: If True (default), reload the saved mmap into RAM and return an
    #: ``ActivationDataset``. For large suites this duplicates the full cache
    #: (e.g. ~100s of GiB for Qwen2.5-0.5B at 2M tokens). Callers that only need
    #: files on disk (e.g. paper runs that load a bounded slice later) should set
    #: this to False.
    load_after_collect: bool = False


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
        mlp_inputs: torch.Tensor | np.ndarray,
        mlp_outputs: torch.Tensor | np.ndarray,
    ):
        assert mlp_inputs.shape == mlp_outputs.shape
        # Streaming mode: arrays are int16 numpy mmaps, reinterpreted as bfloat16
        # per-sample in __getitem__. Keeps RAM cost O(batch) instead of O(dataset).
        self._streaming = isinstance(mlp_inputs, np.ndarray)
        self.mlp_inputs = mlp_inputs
        self.mlp_outputs = mlp_outputs

    def __len__(self) -> int:
        return self.mlp_inputs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._streaming:
            # Copy a single sample out of the mmap into RAM, then reinterpret
            # int16 bits as bfloat16. np.array(..., copy=True) is essential:
            # indexing a numpy mmap returns a view, and OS page reclaim on
            # WSL2 / Hyper-V can SIGBUS those views mid-batch.
            in_np: np.ndarray = self.mlp_inputs  # type: ignore[assignment]
            out_np: np.ndarray = self.mlp_outputs  # type: ignore[assignment]
            return {
                "mlp_inputs": torch.from_numpy(
                    np.array(in_np[idx], copy=True)
                ).view(torch.bfloat16),
                "mlp_outputs": torch.from_numpy(
                    np.array(out_np[idx], copy=True)
                ).view(torch.bfloat16),
            }
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
    def load(
        cls,
        path: str,
        max_samples: int | None = None,
        split: str | None = None,
    ) -> "ActivationDataset":
        """Load dataset from disk.

        On-disk formats:
        - **split numpy format** (preferred, new): collection writes four
          memmaps named ``mlp_inputs_<split>.npy`` /
          ``mlp_outputs_<split>.npy`` for ``split in {"train", "val"}``.
          Pass ``split="train"`` or ``split="val"`` to select one. Train and
          val may live in different directories; pass each path separately.
        - **single numpy format** (legacy): ``mlp_inputs.npy`` /
          ``mlp_outputs.npy`` produced by older collection runs. Used when
          ``split`` is ``None`` (or when split-suffixed files are absent).
        - **torch format** (legacy): ``mlp_inputs.pt`` / ``mlp_outputs.pt``
          loaded with ``mmap=True`` to avoid touching the full file at once.

        Two loading modes:
        - ``max_samples=None`` (streaming, preferred for training): keep the
          mmap handles and copy one sample per ``__getitem__`` call. Lets
          training see every collected sequence at O(batch) RAM. Numpy format
          only.
        - ``max_samples=N``: copy an ``N``-sequence slice into contiguous RAM.
          Used by paper evaluation / analysis scripts that want a bounded
          deterministic subset.

        Args:
            path: Directory containing the activation files.
            max_samples: Number of sequences to load into RAM. ``None`` streams
                the full on-disk dataset from mmap.
            split: Optional ``"train"`` / ``"val"`` selector for the split-file
                layout. ``None`` falls back to the legacy single-file names.
        """
        if split is not None:
            if split not in {"train", "val"}:
                raise ValueError(f"split must be 'train' or 'val', got {split!r}")
            inputs_npy = os.path.join(path, f"mlp_inputs_{split}.npy")
            outputs_npy = os.path.join(path, f"mlp_outputs_{split}.npy")
        else:
            inputs_npy = os.path.join(path, "mlp_inputs.npy")
            outputs_npy = os.path.join(path, "mlp_outputs.npy")

        if max_samples is None:
            # Streaming path: return mmap-backed dataset covering every sample.
            if not (os.path.exists(inputs_npy) and os.path.exists(outputs_npy)):
                raise FileNotFoundError(
                    f"Streaming mode requires numpy-format activations at {path} "
                    "(mlp_inputs.npy / mlp_outputs.npy). Legacy .pt format is not "
                    "supported for streaming; pass max_samples=N to load a slice."
                )
            inputs_mm = np.load(inputs_npy, mmap_mode="r")
            outputs_mm = np.load(outputs_npy, mmap_mode="r")
            return cls(inputs_mm, outputs_mm)

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
def collect_activations(config: DataConfig) -> ActivationDataset | None:
    """Run a transformer on text data and collect MLP input/output activations.

    Uses TransformerLens HookedTransformer for hook-based activation extraction.
    Activations are written directly to memory-mapped numpy files on disk so that
    the pre-allocation RAM cost is O(batch) rather than O(dataset), avoiding OOM
    on large models (e.g. Qwen2.5-0.5B with 24 layers where a full in-RAM
    pre-allocation would exceed 170 GB).

    Args:
        config: Data collection configuration.

    Returns:
        ActivationDataset with cached activations when ``load_after_collect`` is
        True; otherwise ``None`` after files are written.
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

    # Compute the train/val split at collection time using a deterministic
    # permutation seeded by config.seed. Materialising the split here means
    # every consumer sees the same partition without coordinating seeds, and
    # the val files can live on a different (durable) filesystem from the
    # train files.
    val_fraction = float(config.val_fraction)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    n_val = max(1, int(n_sequences * val_fraction)) if val_fraction > 0 else 0
    n_train = n_sequences - n_val
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n_sequences)
    is_val = np.zeros(n_sequences, dtype=bool)
    if n_val > 0:
        is_val[perm[:n_val]] = True
    # Per-global-index slot in the destination memmap.
    train_pos = np.full(n_sequences, -1, dtype=np.int64)
    val_pos = np.full(n_sequences, -1, dtype=np.int64)
    ti = vi = 0
    for g in range(n_sequences):
        if is_val[g]:
            val_pos[g] = vi
            vi += 1
        else:
            train_pos[g] = ti
            ti += 1
    assert ti == n_train and vi == n_val

    # Write activations directly to memory-mapped numpy files (int16 = bfloat16 bits).
    # This eliminates the in-RAM pre-allocation that caused OOM for large models:
    #   GPT-2 small (12 L, 768 d):  2 × ~18 GB = ~36 GB in RAM  — historically fine
    #   Qwen2.5-0.5B (24 L, 896 d): 2 × ~86 GB = ~172 GB in RAM — OOM on 121 GB node
    # With mmap the RAM cost per step is only the current batch cache (~1–2 GB).
    train_dir = config.save_dir
    val_dir = config.val_save_dir or config.save_dir
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    train_in_path = os.path.join(train_dir, "mlp_inputs_train.npy")
    train_out_path = os.path.join(train_dir, "mlp_outputs_train.npy")
    val_in_path = os.path.join(val_dir, "mlp_inputs_val.npy")
    val_out_path = os.path.join(val_dir, "mlp_outputs_val.npy")
    train_shape = (n_train, n_layers, config.seq_len, d_model)
    val_shape = (n_val, n_layers, config.seq_len, d_model)

    # np.lib.format.open_memmap writes a .npy header (shape + dtype) then maps the
    # data region — readable later via np.load(path, mmap_mode='r').
    train_in_mm = np.lib.format.open_memmap(
        train_in_path, dtype=np.int16, mode="w+", shape=train_shape
    )
    train_out_mm = np.lib.format.open_memmap(
        train_out_path, dtype=np.int16, mode="w+", shape=train_shape
    )
    val_in_mm = (
        np.lib.format.open_memmap(
            val_in_path, dtype=np.int16, mode="w+", shape=val_shape
        )
        if n_val > 0
        else None
    )
    val_out_mm = (
        np.lib.format.open_memmap(
            val_out_path, dtype=np.int16, mode="w+", shape=val_shape
        )
        if n_val > 0
        else None
    )

    global_idx = 0
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
        mlp_in_np = mlp_in.cpu().bfloat16().contiguous().view(torch.int16).numpy()
        mlp_out_np = mlp_out.cpu().bfloat16().contiguous().view(torch.int16).numpy()

        for i in range(actual_bs):
            g = global_idx + i
            if is_val[g]:
                assert val_in_mm is not None and val_out_mm is not None
                val_in_mm[val_pos[g]] = mlp_in_np[i]
                val_out_mm[val_pos[g]] = mlp_out_np[i]
            else:
                train_in_mm[train_pos[g]] = mlp_in_np[i]
                train_out_mm[train_pos[g]] = mlp_out_np[i]
        global_idx += actual_bs

    # Flush mmap writes to disk before returning.
    train_in_mm.flush()
    train_out_mm.flush()
    del train_in_mm, train_out_mm
    if val_in_mm is not None:
        val_in_mm.flush()
        val_out_mm.flush()
        del val_in_mm, val_out_mm

    if not config.load_after_collect:
        return None

    # Load the train split via the standard path. Callers that need val too
    # can call ActivationDataset.load(val_dir, split="val") explicitly.
    return ActivationDataset.load(train_dir, split="train")
