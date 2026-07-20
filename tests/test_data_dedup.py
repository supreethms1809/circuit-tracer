"""Regression tests for dedup + train/val disjointness in activation collection.

Mocks HookedTransformer + load_dataset so `collect_activations` runs on CPU with
no network/GPU. The fake model emits activations equal to the token id (broadcast
over d_model), so a written activation window can be mapped back to its exact token
window — letting us assert dedup/disjointness on the real on-disk output.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from spline_clt.training.data import DataConfig, collect_activations


class _FakeCfg:
    n_layers = 2
    d_model = 4


class _FakeTokenizerOut:
    def __init__(self, ids: list[int]):
        self.input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _fake_tokenizer(text, truncation=False, return_tensors=None):  # noqa: ANN001
    return _FakeTokenizerOut([int(t) for t in text.split()])


class _FakeModel:
    cfg = _FakeCfg()
    tokenizer = staticmethod(_fake_tokenizer)

    def eval(self):
        return self

    def run_with_cache(self, batch_tokens, names_filter=None):  # noqa: ANN001
        b, s = batch_tokens.shape
        act = batch_tokens.unsqueeze(-1).float().expand(b, s, _FakeCfg.d_model).contiguous()
        cache = {}
        for i in range(_FakeCfg.n_layers):
            cache[f"blocks.{i}.hook_resid_mid"] = act.clone()
            cache[f"blocks.{i}.hook_mlp_out"] = act.clone()
        return None, cache


class _FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def shuffle(self, seed=0):  # noqa: ANN001 keep order deterministic for the test
        return self

    def select(self, rng):  # noqa: ANN001
        return _FakeDataset([self._rows[i] for i in rng])

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


def _install_mocks(monkeypatch, rows):
    import datasets
    import transformer_lens

    monkeypatch.setattr(
        transformer_lens.HookedTransformer, "from_pretrained",
        staticmethod(lambda *a, **k: _FakeModel()),
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _FakeDataset(rows))


def _windows_from_file(path: str) -> list[tuple[int, ...]]:
    """Recover token windows (layer-0, seq positions) from written activations."""
    mm = np.load(path, mmap_mode="r")  # (n, n_layers, seq, d_model) int16
    out = []
    for i in range(mm.shape[0]):
        t = torch.from_numpy(np.ascontiguousarray(mm[i, 0])).view(torch.bfloat16).float()
        # each position broadcast over d_model -> take column 0, round to int token id
        out.append(tuple(int(round(v)) for v in t[:, 0].tolist()))
    return out


def _make_rows() -> list[dict]:
    # seq_len=4 windows. One degenerate window [50,50,50,50] repeated many times
    # (the corpus-duplication pathology), plus unique windows.
    rows = []
    for _ in range(12):
        rows.append({"text": "50 50 50 50"})          # duplicate window
    for k in range(12):
        base = 100 + 4 * k
        rows.append({"text": f"{base} {base+1} {base+2} {base+3}"})  # unique windows
    return rows


def test_dedup_val_unique_and_disjoint(tmp_path, monkeypatch):
    rows = _make_rows()
    _install_mocks(monkeypatch, rows)
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    cfg = DataConfig(
        model_name="gpt2", dataset_name="fake", dataset_config="",
        n_tokens=24 * 4, seq_len=4, batch_size=5,
        save_dir=str(train_dir), val_save_dir=str(val_dir),
        val_fraction=0.25, device="cpu", dtype="bfloat16", seed=0,
        dedup=True, dedup_train=False,
    )
    collect_activations(cfg)

    val_w = _windows_from_file(str(val_dir / "mlp_inputs_val.npy"))
    train_w = _windows_from_file(str(train_dir / "mlp_inputs_train.npy"))

    # 1) val has no duplicate windows
    assert len(val_w) == len(set(val_w)), f"val not deduped: {val_w}"
    # 2) train is disjoint from val by content
    assert set(train_w).isdisjoint(set(val_w)), "train leaks a held-out window"
    # 3) the degenerate window, if reserved for val, never appears in train
    if (50, 50, 50, 50) in set(val_w):
        assert (50, 50, 50, 50) not in set(train_w)
    # 4) val_hashes.txt lists exactly the distinct val window hashes
    hpath = val_dir / "val_hashes.txt"
    assert hpath.exists()
    persisted = {line.strip() for line in hpath.read_text().splitlines() if line.strip()}
    expected = {
        hashlib.blake2b(np.asarray(w, dtype=np.int64).tobytes(), digest_size=16).hexdigest()
        for w in set(val_w)
    }
    assert persisted == expected


def test_dedup_train_removes_within_train_duplicates(tmp_path, monkeypatch):
    rows = _make_rows()
    _install_mocks(monkeypatch, rows)
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    cfg = DataConfig(
        model_name="gpt2", dataset_name="fake", dataset_config="",
        n_tokens=24 * 4, seq_len=4, batch_size=5,
        save_dir=str(train_dir), val_save_dir=str(val_dir),
        val_fraction=0.0, device="cpu", dtype="bfloat16", seed=1,
        dedup=True, dedup_train=True,
    )
    collect_activations(cfg)
    train_w = _windows_from_file(str(train_dir / "mlp_inputs_train.npy"))
    assert len(train_w) == len(set(train_w)), "train self-dedup left duplicates"


def test_cross_chunk_val_excluded_from_later_train(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    vhash = str(val_dir / "val_hashes.txt")
    dup = (50, 50, 50, 50)
    dup_hash = hashlib.blake2b(
        np.asarray(dup, dtype=np.int64).tobytes(), digest_size=16
    ).hexdigest()

    # Chunk 0: dataset dominated by the duplicate window -> val certainly reserves it.
    # keep token ids < 256 so activations (token id as bf16) round-trip exactly.
    rows0 = [{"text": "50 50 50 50"} for _ in range(20)] + [
        {"text": f"{60 + 4 * k} {61 + 4 * k} {62 + 4 * k} {63 + 4 * k}"} for k in range(4)
    ]
    _install_mocks(monkeypatch, rows0)
    collect_activations(DataConfig(
        model_name="gpt2", dataset_name="fake", dataset_config="",
        n_tokens=24 * 4, seq_len=4, batch_size=5,
        save_dir=str(train_dir), val_save_dir=str(val_dir),
        val_fraction=0.25, device="cpu", dtype="bfloat16", seed=0,
        dedup=True, val_hashes_path=vhash,
    ))
    persisted = {line.strip() for line in (val_dir / "val_hashes.txt").read_text().splitlines()}
    assert dup_hash in persisted, "chunk-0 val should have reserved the dominant window"

    # Chunk 1: train-only (val_fraction=0), contains the reserved window + uniques.
    rows1 = [{"text": "50 50 50 50"} for _ in range(6)] + [
        {"text": f"{150 + 4 * k} {151 + 4 * k} {152 + 4 * k} {153 + 4 * k}"} for k in range(6)
    ]
    _install_mocks(monkeypatch, rows1)
    collect_activations(DataConfig(
        model_name="gpt2", dataset_name="fake", dataset_config="",
        n_tokens=12 * 4, seq_len=4, batch_size=4,
        save_dir=str(train_dir), val_save_dir=str(val_dir),
        val_fraction=0.0, device="cpu", dtype="bfloat16", seed=2,
        dedup=True, val_hashes_path=vhash,
    ))
    train_w = set(_windows_from_file(str(train_dir / "mlp_inputs_train.npy")))
    assert dup not in train_w, "later-chunk train leaked a window reserved for val"
    assert (150, 151, 152, 153) in train_w, "unique later-chunk windows should remain in train"


def test_legacy_no_dedup_keeps_all(tmp_path, monkeypatch):
    rows = _make_rows()
    _install_mocks(monkeypatch, rows)
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    cfg = DataConfig(
        model_name="gpt2", dataset_name="fake", dataset_config="",
        n_tokens=24 * 4, seq_len=4, batch_size=5,
        save_dir=str(train_dir), val_save_dir=str(val_dir),
        val_fraction=0.25, device="cpu", dtype="bfloat16", seed=0,
        dedup=False,
    )
    collect_activations(cfg)
    train_w = _windows_from_file(str(train_dir / "mlp_inputs_train.npy"))
    val_w = _windows_from_file(str(val_dir / "mlp_inputs_val.npy"))
    # legacy: no drops, so train+val == total collected, and duplicates are kept
    assert len(train_w) + len(val_w) == 24
    assert not (val_dir / "val_hashes.txt").exists()
