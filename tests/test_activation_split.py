"""Tests for the train/val split materialised at activation collection time."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from spline_clt.training.data import ActivationDataset


@dataclass
class _FakeCfg:
    seed: int
    val_fraction: float
    save_dir: str
    val_save_dir: str
    seq_len: int = 4
    n_layers: int = 2
    d_model: int = 3


def _make_split_files(cfg: _FakeCfg, n_sequences: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate the collection-time split logic and write 4 memmaps.

    Mirrors ``collect_activations`` so we can test the on-disk layout without
    spinning up a HookedTransformer. Returns the originating
    (sequences, is_val, perm) triple so tests can verify the routing.
    """
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_sequences)
    n_val = max(1, int(n_sequences * cfg.val_fraction))
    is_val = np.zeros(n_sequences, dtype=bool)
    is_val[perm[:n_val]] = True
    n_train = n_sequences - n_val

    # Synthetic per-sequence payload — distinct values so we can verify routing.
    # Stored as int16 (the bf16 bit-pattern slot).
    sequences = np.arange(
        n_sequences * cfg.n_layers * cfg.seq_len * cfg.d_model,
        dtype=np.int16,
    ).reshape(n_sequences, cfg.n_layers, cfg.seq_len, cfg.d_model)

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.val_save_dir, exist_ok=True)
    train_in = np.lib.format.open_memmap(
        os.path.join(cfg.save_dir, "mlp_inputs_train.npy"),
        dtype=np.int16, mode="w+",
        shape=(n_train, cfg.n_layers, cfg.seq_len, cfg.d_model),
    )
    train_out = np.lib.format.open_memmap(
        os.path.join(cfg.save_dir, "mlp_outputs_train.npy"),
        dtype=np.int16, mode="w+",
        shape=(n_train, cfg.n_layers, cfg.seq_len, cfg.d_model),
    )
    val_in = np.lib.format.open_memmap(
        os.path.join(cfg.val_save_dir, "mlp_inputs_val.npy"),
        dtype=np.int16, mode="w+",
        shape=(n_val, cfg.n_layers, cfg.seq_len, cfg.d_model),
    )
    val_out = np.lib.format.open_memmap(
        os.path.join(cfg.val_save_dir, "mlp_outputs_val.npy"),
        dtype=np.int16, mode="w+",
        shape=(n_val, cfg.n_layers, cfg.seq_len, cfg.d_model),
    )

    ti = vi = 0
    for g in range(n_sequences):
        if is_val[g]:
            val_in[vi] = sequences[g]
            val_out[vi] = sequences[g] + 1  # different content for outputs
            vi += 1
        else:
            train_in[ti] = sequences[g]
            train_out[ti] = sequences[g] + 1
            ti += 1
    train_in.flush(); train_out.flush(); val_in.flush(); val_out.flush()
    return sequences, is_val, perm


def test_split_layout_train_and_val_are_disjoint(tmp_path) -> None:
    cfg = _FakeCfg(seed=42, val_fraction=0.2,
                   save_dir=str(tmp_path / "train"),
                   val_save_dir=str(tmp_path / "val"))
    n_sequences = 50
    sequences, is_val, _ = _make_split_files(cfg, n_sequences)

    train_ds = ActivationDataset.load(cfg.save_dir, split="train")
    val_ds = ActivationDataset.load(cfg.val_save_dir, split="val")

    n_val_expected = max(1, int(n_sequences * cfg.val_fraction))
    assert len(val_ds) == n_val_expected
    assert len(train_ds) == n_sequences - n_val_expected

    # Reconstruct the originating sequence ids from the routed payloads;
    # the union must equal the full set, and they must be disjoint.
    def first_int(arr):
        # Each sequence is (n_layers, seq_len, d_model); first int16 encodes
        # its global index * n_layers * seq_len * d_model.
        return int(arr[0, 0, 0])
    train_starts = {first_int(np.array(train_ds.mlp_inputs[i])) for i in range(len(train_ds))}
    val_starts = {first_int(np.array(val_ds.mlp_inputs[i])) for i in range(len(val_ds))}
    stride = cfg.n_layers * cfg.seq_len * cfg.d_model
    expected_starts = {g * stride for g in range(n_sequences)}
    assert train_starts | val_starts == expected_starts
    assert train_starts.isdisjoint(val_starts)
    # Routing matches the deterministic permutation.
    val_ids = {s // stride for s in val_starts}
    assert val_ids == {g for g in range(n_sequences) if is_val[g]}


def test_split_is_seed_deterministic(tmp_path) -> None:
    n_sequences = 40
    cfg_a = _FakeCfg(seed=7, val_fraction=0.1,
                     save_dir=str(tmp_path / "a/train"),
                     val_save_dir=str(tmp_path / "a/val"))
    cfg_b = _FakeCfg(seed=7, val_fraction=0.1,
                     save_dir=str(tmp_path / "b/train"),
                     val_save_dir=str(tmp_path / "b/val"))
    cfg_c = _FakeCfg(seed=8, val_fraction=0.1,
                     save_dir=str(tmp_path / "c/train"),
                     val_save_dir=str(tmp_path / "c/val"))
    _, is_val_a, _ = _make_split_files(cfg_a, n_sequences)
    _, is_val_b, _ = _make_split_files(cfg_b, n_sequences)
    _, is_val_c, _ = _make_split_files(cfg_c, n_sequences)
    np.testing.assert_array_equal(is_val_a, is_val_b)
    assert not np.array_equal(is_val_a, is_val_c)


def test_load_split_rejects_invalid_split_name(tmp_path) -> None:
    cfg = _FakeCfg(seed=0, val_fraction=0.2,
                   save_dir=str(tmp_path / "train"),
                   val_save_dir=str(tmp_path / "val"))
    _make_split_files(cfg, n_sequences=20)
    with pytest.raises(ValueError, match="split must be"):
        ActivationDataset.load(cfg.save_dir, split="other")


def test_load_legacy_layout_still_works(tmp_path) -> None:
    """Old single-file layout (no split arg) still loads via the legacy path."""
    n, layers, seq, d = 8, 2, 4, 3
    arr = np.arange(n * layers * seq * d, dtype=np.int16).reshape(n, layers, seq, d)
    np.save(tmp_path / "mlp_inputs.npy", arr)
    np.save(tmp_path / "mlp_outputs.npy", arr + 1)

    ds = ActivationDataset.load(str(tmp_path), max_samples=n)
    assert len(ds) == n
    sample = ds[0]
    assert sample["mlp_inputs"].shape == (layers, seq, d)
    assert sample["mlp_outputs"].shape == (layers, seq, d)


def test_streaming_load_per_split_returns_int_pattern_intact(tmp_path) -> None:
    """Verify that __getitem__ in streaming mode reinterprets int16→bfloat16
    correctly without copying through python ints (regression for SIGBUS path)."""
    cfg = _FakeCfg(seed=0, val_fraction=0.25,
                   save_dir=str(tmp_path / "train"),
                   val_save_dir=str(tmp_path / "val"))
    _make_split_files(cfg, n_sequences=16)
    val_ds = ActivationDataset.load(cfg.val_save_dir, split="val")
    sample = val_ds[0]
    assert sample["mlp_inputs"].dtype == torch.bfloat16
    assert sample["mlp_outputs"].dtype == torch.bfloat16
