"""Rebuild a clean, de-duplicated validation holdout from a collected val cache.

Why: the `gpt2_2b_v2` fixed val (5% of chunk 0) is 64% a single 128-token window
that recurs verbatim across the corpus and is memorized from train, so the logged
val_loss (~0.24) is meaningless and every val-based eval metric is contaminated.
(Only 33.9% of the 7814 val sequences are unique.)

This utility content-hashes each val sequence (layer-0 input bytes = a deterministic
fingerprint of the token window), drops duplicates, and writes a new drop-in val
cache (`mlp_inputs_val.npy` / `mlp_outputs_val.npy`) plus a manifest.

Disjoint-from-train: true verification needs train-side content hashes, which live
on node-local /lscratch and are not reachable here. The available PROXY is
within-val multiplicity — a window duplicated inside a 5% sample is near-certainly
present in the 95% train split, so `--mode singletons` (default) drops EVERY
duplicated window, not just the extra copies. If you later have a train-hash file
(one hex digest per line, e.g. produced by hashing collected train shards or the
`kept_hashes` of this run over the train stream), pass `--train-hashes FILE` to
also exclude any val window whose hash appears in train.

Usage:
    python scripts/dedup_val.py \
        --in-dir  /gscratch/ssuresh/shared/activations/gpt2_2b_v2/val/gpt2 \
        --out-dir /gscratch/ssuresh/shared/activations/gpt2_2b_v2/val_clean/gpt2 \
        --mode singletons
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np


def _seq_hash(arr_l0: np.ndarray) -> bytes:
    """16-byte content fingerprint of one sequence's layer-0 input (int16 bytes)."""
    return hashlib.blake2b(np.ascontiguousarray(arr_l0).tobytes(), digest_size=16).digest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, help="dir with mlp_inputs_val.npy / mlp_outputs_val.npy")
    ap.add_argument("--out-dir", required=True, help="destination dir for the clean val cache")
    ap.add_argument(
        "--mode",
        choices=["singletons", "representatives"],
        default="singletons",
        help="singletons: keep only windows that appear exactly once in val "
        "(leakage-safe, default). representatives: keep one copy of each distinct "
        "window (larger, but keeps 1 copy of duplicated/likely-leaked windows).",
    )
    ap.add_argument(
        "--train-hashes",
        default=None,
        help="optional file of hex content-hashes present in TRAIN; matching val "
        "windows are dropped (true disjoint-from-train filter).",
    )
    ap.add_argument("--copy-batch", type=int, default=256, help="sequences copied per write batch")
    args = ap.parse_args()

    in_x = os.path.join(args.in_dir, "mlp_inputs_val.npy")
    in_y = os.path.join(args.in_dir, "mlp_outputs_val.npy")
    if not (os.path.exists(in_x) and os.path.exists(in_y)):
        raise FileNotFoundError(f"expected mlp_inputs_val.npy / mlp_outputs_val.npy under {args.in_dir}")

    xin = np.load(in_x, mmap_mode="r")
    yout = np.load(in_y, mmap_mode="r")
    assert xin.shape == yout.shape, (xin.shape, yout.shape)
    N, L, S, D = xin.shape
    print(f"[dedup] input val: {N} seqs, shape={xin.shape}, dtype={xin.dtype}", flush=True)

    # --- content hashes (layer 0) ---
    t0 = time.time()
    hashes: list[bytes] = []
    for i in range(N):
        hashes.append(_seq_hash(xin[i, 0]))
        if (i + 1) % 2000 == 0:
            print(f"[dedup] hashed {i + 1}/{N} ({time.time() - t0:.0f}s)", flush=True)
    from collections import Counter

    mult = Counter(hashes)
    n_distinct = len(mult)
    n_singletons = sum(1 for v in mult.values() if v == 1)
    dom_hash, dom_cnt = mult.most_common(1)[0]
    print(
        f"[dedup] distinct windows={n_distinct} ({100 * n_distinct / N:.1f}%), "
        f"singleton windows={n_singletons}, dominant window×{dom_cnt} "
        f"({100 * dom_cnt / N:.1f}% of val)",
        flush=True,
    )

    train_set: set[bytes] = set()
    if args.train_hashes:
        with open(args.train_hashes) as f:
            train_set = {bytes.fromhex(line.strip()) for line in f if line.strip()}
        print(f"[dedup] loaded {len(train_set)} train hashes for disjoint filter", flush=True)

    # --- select keep indices ---
    keep: list[int] = []
    seen: set[bytes] = set()
    for i, h in enumerate(hashes):
        if h in train_set:
            continue
        if args.mode == "singletons":
            if mult[h] == 1:
                keep.append(i)
        else:  # representatives
            if h not in seen:
                seen.add(h)
                keep.append(i)
    n_keep = len(keep)
    print(
        f"[dedup] mode={args.mode} -> keeping {n_keep}/{N} sequences "
        f"({100 * n_keep / N:.1f}%)",
        flush=True,
    )
    if n_keep == 0:
        raise SystemExit("[dedup] refusing to write an empty val set")

    # --- write clean cache (int16, drop-in shape) ---
    os.makedirs(args.out_dir, exist_ok=True)
    out_x = np.lib.format.open_memmap(
        os.path.join(args.out_dir, "mlp_inputs_val.npy"),
        dtype=np.int16, mode="w+", shape=(n_keep, L, S, D),
    )
    out_y = np.lib.format.open_memmap(
        os.path.join(args.out_dir, "mlp_outputs_val.npy"),
        dtype=np.int16, mode="w+", shape=(n_keep, L, S, D),
    )
    keep_arr = np.asarray(keep, dtype=np.int64)
    for start in range(0, n_keep, args.copy_batch):
        idx = keep_arr[start : start + args.copy_batch]
        out_x[start : start + len(idx)] = xin[idx]
        out_y[start : start + len(idx)] = yout[idx]
        if start % (args.copy_batch * 8) == 0:
            print(f"[dedup] wrote {start + len(idx)}/{n_keep}", flush=True)
    out_x.flush(); out_y.flush()

    manifest = {
        "source": os.path.abspath(args.in_dir),
        "mode": args.mode,
        "n_source": int(N),
        "n_distinct_windows": int(n_distinct),
        "n_singleton_windows": int(n_singletons),
        "dominant_window_count": int(dom_cnt),
        "dominant_window_frac": float(dom_cnt / N),
        "n_train_hashes": int(len(train_set)),
        "n_kept": int(n_keep),
        "shape": [int(n_keep), int(L), int(S), int(D)],
        "kept_hashes": [hashes[i].hex() for i in keep],
    }
    with open(os.path.join(args.out_dir, "dedup_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[dedup] wrote clean val -> {args.out_dir} ({n_keep} seqs) + dedup_manifest.json", flush=True)


if __name__ == "__main__":
    main()
