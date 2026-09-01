"""Activation dataset collection for Spline-CLT training.

Runs a transformer model on text data and caches residual stream activations
and MLP outputs at all layers. These cached activations are used to train
the Spline-CLT to reconstruct MLP outputs.
"""

import gc
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass

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
    dtype: str = "float32"
    seed: int = 0
    #: If True (default), reload the saved mmap into RAM and return an
    #: ``ActivationDataset``. For large suites this duplicates the full cache
    #: (e.g. ~100s of GiB for Qwen2.5-0.5B at 2M tokens). Callers that only need
    #: files on disk (e.g. paper runs that load a bounded slice later) should set
    #: this to False.
    load_after_collect: bool = False
    #: Skip this many contiguous ``seq_len`` windows from the shuffled corpus before
    #: counting toward ``n_tokens``. Legacy window-level cursor; superseded by
    #: ``skip_items`` for chunked collection (window-level skip re-tokenizes the whole
    #: consumed prefix every chunk, which is O(chunks^2) over a run). Kept only as a
    #: fallback for callers that have not migrated to the item cursor.
    skip_sequences: int = 0
    #: Skip this many whole shuffled corpus *items* before collecting. Items are
    #: dropped with ``Dataset.select`` (no tokenization of the skipped prefix), so the
    #: per-chunk resume cost is O(items_in_this_chunk) instead of O(consumed_prefix).
    #: This is the preferred chunked-collection cursor; when > 0 it takes precedence
    #: over ``skip_sequences``.
    skip_items: int = 0
    #: Deduplicate the validation holdout and keep it disjoint from train by exact
    #: token-window content. When True: (1) the val split keeps only the FIRST
    #: occurrence of each distinct 128-token window (drops within-val duplicates —
    #: the corpus has degenerate windows repeated thousands of times); (2) every
    #: sequence whose window content matches a val window is EXCLUDED from train,
    #: across all chunks, so the model never trains on a held-out window. Val window
    #: hashes are persisted to ``val_hashes_path`` so later chunks (which collect
    #: train-only) can still exclude them. Off by default to preserve legacy behavior.
    dedup: bool = False
    #: When ``dedup`` is True, also drop duplicate windows *within* the train split
    #: of each chunk (keep one copy of each distinct window). Reduces wasted training
    #: on repeated boilerplate. Independent of the val disjointness above.
    dedup_train: bool = False
    #: File holding hex token-window hashes reserved for validation (one per line),
    #: accumulated across chunks. Defaults to ``<val_save_dir>/val_hashes.txt``.
    #: Only consulted when ``dedup`` is True.
    val_hashes_path: str | None = None


@dataclass
class ActivationCollectResult:
    """Return value from :func:`collect_activations`."""

    dataset: "ActivationDataset | None"
    #: Total sequences written to memmaps for this collect call (train + val slots).
    n_sequences: int
    #: Number of whole shuffled corpus items consumed to produce this chunk. The
    #: caller advances its item cursor by this amount so the next chunk resumes at the
    #: following item boundary (item-aligned resume).
    n_items_consumed: int = 0


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
def compute_input_normalization(
    dataset: "ActivationDataset",
    n_layers: int,
    d_model: int,
    n_sequences: int = 256,
    seed: int = 0,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-layer, per-dimension mean/std of the encoder input.

    Samples a deterministic subset of sequences and accumulates first/second
    moments in float64 (heavy outliers otherwise destroy precision). Used to
    standardize the residual-stream input before the encoder so the B-spline
    grid operates in a well-conditioned coordinate system.

    The computation is seeded and indexes the dataset directly (not via a
    DistributedSampler), so every rank derives identical statistics without any
    collective communication.

    Args:
        dataset: Activation dataset; ``dataset[i]["mlp_inputs"]`` must have shape
            (n_layers, seq_len, d_model).
        n_layers: Number of transformer layers.
        d_model: Residual-stream dimension.
        n_sequences: Number of sequences to sample for the estimate.
        seed: Seed for the deterministic sample.
        eps: Std floor; dims with std < eps get std=1 (no-op normalization).

    Returns:
        (mean, std), each of shape (n_layers, d_model), float32.
    """
    n = len(dataset)
    if n == 0:
        return (
            torch.zeros(n_layers, d_model),
            torch.ones(n_layers, d_model),
        )
    rng = np.random.default_rng(seed)
    count = min(n_sequences, n)
    idx = rng.choice(n, size=count, replace=False)

    sums = torch.zeros(n_layers, d_model, dtype=torch.float64)
    sqs = torch.zeros(n_layers, d_model, dtype=torch.float64)
    n_pos = 0
    for i in idx:
        x = dataset[int(i)]["mlp_inputs"].to(torch.float64)  # (n_layers, seq, d_model)
        x = x.reshape(n_layers, -1, d_model)
        sums += x.sum(dim=1)
        sqs += (x * x).sum(dim=1)
        n_pos += x.shape[1]

    mean = sums / n_pos
    var = (sqs / n_pos - mean**2).clamp_min(0.0)
    std = var.sqrt()
    std = torch.where(std < eps, torch.ones_like(std), std)
    return mean.float(), std.float()


@torch.no_grad()
def collect_activations(config: DataConfig) -> ActivationCollectResult:
    """Run a transformer on text data and collect MLP input/output activations.

    Uses TransformerLens HookedTransformer for hook-based activation extraction.
    Activations are written directly to memory-mapped numpy files on disk so that
    the pre-allocation RAM cost is O(batch) rather than O(dataset), avoiding OOM
    on large models (e.g. Qwen2.5-0.5B with 24 layers where a full in-RAM
    pre-allocation would exceed 170 GB).

    Args:
        config: Data collection configuration.

    Returns:
        :class:`ActivationCollectResult` with optional in-RAM dataset when
        ``load_after_collect`` is True and ``n_sequences`` written this call.
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

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]

    # Load model
    model = HookedTransformer.from_pretrained(
        config.model_name,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
    )
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Load dataset from HuggingFace Hub (supports any Hub repo or Parquet-based dataset).
    load_kwargs = {"path": config.dataset_name, "split": "train"}
    if config.dataset_config:
        load_kwargs["name"] = config.dataset_config
    dataset = load_dataset(**load_kwargs)

    dataset = dataset.shuffle(seed=config.seed)

    # Resume cursor. Prefer the item cursor (``skip_items``): dropping whole items
    # via ``select`` costs nothing per skipped item (no tokenization), so each chunk
    # is O(items_in_this_chunk) rather than re-tokenizing the entire consumed prefix.
    # ``skip_sequences`` is the legacy window-level fallback, used only when no item
    # cursor is supplied.
    skip_items = max(0, int(config.skip_items))
    if skip_items > 0:
        skip_items = min(skip_items, len(dataset))
        dataset = dataset.select(range(skip_items, len(dataset)))
    skip_sequences = 0 if skip_items > 0 else max(0, int(config.skip_sequences))

    tokenizer = model.tokenizer
    assert tokenizer is not None

    # Tokenize — chunk each sample into non-overlapping seq_len windows to use
    # all tokens, not just the first seq_len.  Codelion samples average ~1K tokens;
    # truncating to 128 would discard ~88% of the data.
    # Batches are formed on the fly to avoid holding all tokens in RAM at once.
    n_sequences_target = config.n_tokens // config.seq_len
    skipped = 0
    token_batches: list[torch.Tensor] = []
    current_batch: list[torch.Tensor] = []
    sequences_committed = 0
    # Items consumed from the (already item-skipped) view. Reported back so the
    # caller advances its item cursor to the next item boundary. We count the item
    # whose windows reach the target as fully consumed (item-aligned resume): any of
    # its trailing windows beyond the target are dropped, at most <1 item of corpus
    # per chunk, which is negligible at multi-billion-token budgets.
    items_consumed = 0

    for item in dataset:
        items_consumed += 1
        tokens = tokenizer(
            item["text"],
            truncation=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        # Chunk into non-overlapping windows of seq_len
        for start in range(0, len(tokens) - config.seq_len + 1, config.seq_len):
            if skipped < skip_sequences:
                skipped += 1
                continue
            current_batch.append(tokens[start : start + config.seq_len])
            if len(current_batch) == config.batch_size:
                token_batches.append(torch.stack(current_batch))
                current_batch = []
                sequences_committed += config.batch_size
            if sequences_committed >= n_sequences_target:
                break
        if sequences_committed >= n_sequences_target:
            break

    # Flush remaining partial batch
    if current_batch and sequences_committed < n_sequences_target:
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
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n_sequences)
    n_val_target = max(1, int(n_sequences * val_fraction)) if val_fraction > 0 else 0
    want_val = np.zeros(n_sequences, dtype=bool)
    if n_val_target > 0:
        want_val[perm[:n_val_target]] = True

    is_val = np.zeros(n_sequences, dtype=bool)
    is_train = np.zeros(n_sequences, dtype=bool)
    new_val_hashes: list[bytes] = []
    val_hash_path: str | None = None

    if not config.dedup:
        # Legacy behavior: every non-val sequence is train; no content filtering.
        is_val = want_val.copy()
        is_train = ~want_val
    else:
        import hashlib

        # Exact per-window content fingerprint from the raw tokens (available before
        # the model runs). token_batches rows are in global-index order.
        seq_hashes: list[bytes] = []
        for b in token_batches:
            bt = b.cpu().numpy()
            for row in range(bt.shape[0]):
                seq_hashes.append(
                    hashlib.blake2b(
                        np.ascontiguousarray(bt[row]).tobytes(), digest_size=16
                    ).digest()
                )
        assert len(seq_hashes) == n_sequences

        val_hash_path = config.val_hashes_path or os.path.join(
            config.val_save_dir or config.save_dir, "val_hashes.txt"
        )
        prior_val_hashes: set[bytes] = set()
        if os.path.exists(val_hash_path):
            with open(val_hash_path) as f:
                prior_val_hashes = {bytes.fromhex(line.strip()) for line in f if line.strip()}

        # Pass 1: dedupe val — keep the first occurrence of each distinct window not
        # already reserved by a prior chunk.
        seen_val: set[bytes] = set()
        for g in range(n_sequences):
            if want_val[g]:
                h = seq_hashes[g]
                if h in prior_val_hashes or h in seen_val:
                    continue
                seen_val.add(h)
                is_val[g] = True
        new_val_hashes = list(seen_val)
        val_hashes = prior_val_hashes | seen_val

        # Pass 2: train = non-val-designated windows whose content is NOT reserved
        # for val (disjointness), optionally self-deduped.
        seen_train: set[bytes] = set()
        for g in range(n_sequences):
            if want_val[g]:
                continue
            h = seq_hashes[g]
            if h in val_hashes:
                continue  # would leak a held-out window into train
            if config.dedup_train:
                if h in seen_train:
                    continue
                seen_train.add(h)
            is_train[g] = True

    n_val = int(is_val.sum())
    n_train = int(is_train.sum())

    # Per-global-index slot in the destination memmap (-1 = dropped, written nowhere).
    train_pos = np.full(n_sequences, -1, dtype=np.int64)
    val_pos = np.full(n_sequences, -1, dtype=np.int64)
    ti = vi = 0
    for g in range(n_sequences):
        if is_val[g]:
            val_pos[g] = vi
            vi += 1
        elif is_train[g]:
            train_pos[g] = ti
            ti += 1
    assert ti == n_train and vi == n_val
    if config.dedup:
        print(
            f"[collect/dedup] collected={n_sequences} -> train={n_train} val={n_val} "
            f"dropped={n_sequences - n_train - n_val} (val-dups + train-leak); "
            f"new_val_windows={len(new_val_hashes)}",
            flush=True,
        )

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
            elif is_train[g]:
                train_in_mm[train_pos[g]] = mlp_in_np[i]
                train_out_mm[train_pos[g]] = mlp_out_np[i]
            # else: dropped (val duplicate or train-leak) — written nowhere.
        global_idx += actual_bs

    # Flush mmap writes to disk before returning.
    train_in_mm.flush()
    train_out_mm.flush()
    del train_in_mm, train_out_mm
    if val_in_mm is not None:
        val_in_mm.flush()
        val_out_mm.flush()
        del val_in_mm, val_out_mm

    # Persist this chunk's new val window hashes so later chunks (which collect
    # train-only, val_fraction=0) still exclude those held-out windows from train.
    # Appended after the val data is safely flushed, so a crash never leaves
    # reserved hashes without their val rows.
    if config.dedup and val_hash_path is not None and new_val_hashes:
        os.makedirs(os.path.dirname(val_hash_path) or ".", exist_ok=True)
        with open(val_hash_path, "a") as f:
            for h in new_val_hashes:
                f.write(h.hex() + "\n")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if not config.load_after_collect:
        return ActivationCollectResult(
            dataset=None,
            n_sequences=n_sequences,
            n_items_consumed=items_consumed,
        )

    # Load the train split via the standard path. Callers that need val too
    # can call ActivationDataset.load(val_dir, split="val") explicitly.
    loaded = ActivationDataset.load(train_dir, split="train")
    return ActivationCollectResult(
        dataset=loaded,
        n_sequences=n_sequences,
        n_items_consumed=items_consumed,
    )


def collect_activations_isolated(
    config: DataConfig,
    *,
    cuda_visible_devices: str | None = None,
) -> ActivationCollectResult:
    """Run :func:`collect_activations` in a fresh subprocess.

    Loading Gemma (or any HookedTransformer) inside the FSDP training process
    allocates and frees multi-GB tensors through the *same* CUDA caching
    allocator that holds the ~69 GiB FSDP model. That fragments the allocator;
    the next training backward then fails to find a contiguous ~9.26 GiB slab
    for the full-decoder unshard even though steady-state ``memory_allocated``
    is flat across chunks.

    The child process gets its own allocator. When it exits, the driver frees
    Gemma's memory without punching holes in the parent's CUDA slabs.
    ``load_after_collect`` must be False (memmaps only); the parent never needs
    the in-process dataset handle.

    The parent must **release the FSDP training session from the GPUs** before
    calling this for mid-run chunks (:func:`spline_clt.training.train.release_session_gpu`).
    Collection loads the base LM only — it does not need FSDP — and co-resident
    FSDP + Gemma on the same card is what OOMs chunk N+1.

    Progress (tqdm / model-load logs) is inherited on stderr — do **not** pipe
    ``capture_output`` here: a long collect fills the pipe buffer and deadlocks
    the child while the parent waits forever with a silent ``.err`` log.
    """
    if config.load_after_collect:
        raise ValueError(
            "collect_activations_isolated requires load_after_collect=False "
            "(child returns counts only; parent reloads memmaps)."
        )

    cfg = dict(asdict(config))
    env = os.environ.copy()
    # Child must not inherit torchrun/elastic membership — those vars make
    # libraries assume it is still a distributed worker.
    _drop_keys = {
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "LOCAL_WORLD_SIZE",
        "GROUP_WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    }
    _drop_prefixes = ("TORCHELASTIC_", "PET_", "TORCH_DISTRIBUTED_")
    for key in list(env):
        if key in _drop_keys or key.startswith(_drop_prefixes):
            env.pop(key, None)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        # Child sees a single device as cuda:0.
        cfg["device"] = "cuda:0"

    result_path = None
    cfg_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_collect_cfg.json", delete=False
        ) as cf:
            json.dump(cfg, cf)
            cfg_path = cf.name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_collect_result.json", delete=False
        ) as rf:
            result_path = rf.name

        child = r"""
import json, sys
from spline_clt.training.data import DataConfig, collect_activations
cfg_path, result_path = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = DataConfig(**json.load(f))
result = collect_activations(cfg)
with open(result_path, "w") as f:
    json.dump({
        "n_sequences": int(result.n_sequences),
        "n_items_consumed": int(result.n_items_consumed),
    }, f)
"""
        proc = subprocess.run(
            [sys.executable, "-c", child, cfg_path, result_path],
            env=env,
            check=False,
            # Inherit stdout/stderr so Loading/tqdm appear in the Slurm .err
            # and the pipe cannot fill up and deadlock.
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Isolated activation collection failed "
                f"(exit {proc.returncode}). See child logs above."
            )
        with open(result_path) as f:
            payload = json.load(f)
    finally:
        for path in (cfg_path, result_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return ActivationCollectResult(
        dataset=None,
        n_sequences=int(payload["n_sequences"]),
        n_items_consumed=int(payload["n_items_consumed"]),
    )
