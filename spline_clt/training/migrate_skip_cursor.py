"""One-time migration: derive an item-level corpus cursor for a chunked run.

Older chunked-offline checkpoints persisted only ``corpus_skip_sequences`` (a
window-level counter). The resume path used to re-tokenize that entire window
prefix every chunk, which is O(chunks^2) over a run and eventually exceeds the
4 h NCCL collective timeout (rank 1 blocks on the post-collect broadcast while
rank 0 grinds through the skip). The fix tracks an *item* cursor
(``corpus_skip_items``) and drops skipped items with ``Dataset.select`` — no
tokenization of the consumed prefix.

This script walks the shuffled corpus exactly once, with *batched* tokenization,
counting non-overlapping ``seq_len`` windows per item until the cumulative count
reaches the recorded ``corpus_skip_sequences``. It records the item index at the
next item boundary (item-aligned resume: at most <1 item of corpus is dropped vs
the exact window cursor, which is negligible at billion-token budgets) and writes
``corpus_skip_items`` back into the training state.

Run it SINGLE-PROCESS (no torchrun): there is no second rank, so no NCCL timeout
ceiling applies to the one-time prefix walk. CPU-only is fine — it never runs the
model forward.

Usage:
    python -m spline_clt.training.migrate_skip_cursor \
        --state   /path/to/<run>_training_state.pt \
        --resolved-config /path/to/resolved_config.json
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def _count_windows(token_lengths, seq_len: int) -> int:
    # Number of non-overlapping seq_len windows in a sequence of length L is
    # len(range(0, L - seq_len + 1, seq_len)) == L // seq_len for L >= seq_len,
    # and 0 otherwise. L // seq_len already evaluates to 0 when L < seq_len.
    return sum(L // seq_len for L in token_lengths)


def derive_item_cursor(
    *,
    dataset_name: str,
    dataset_config: str,
    shuffle_seed: int,
    model_name: str,
    seq_len: int,
    target_windows: int,
    tok_batch_size: int = 2000,
    log_every: int = 200_000,
) -> tuple[int, int]:
    """Walk the shuffled corpus and return (item_cursor, windows_covered).

    The returned ``item_cursor`` is the smallest number of leading items whose
    cumulative window count is >= ``target_windows`` (rounding UP to the next item
    boundary, so we never resume into already-consumed data — at most a few
    trailing windows of one item are dropped).
    """
    load_kwargs = {"path": dataset_name, "split": "train"}
    if dataset_config:
        load_kwargs["name"] = dataset_config
    dataset = load_dataset(**load_kwargs)
    dataset = dataset.shuffle(seed=shuffle_seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    cumulative_windows = 0
    items_seen = 0
    t0 = time.time()
    last_log = 0

    for batch in dataset.iter(batch_size=tok_batch_size):
        texts = batch["text"]
        enc = tokenizer(texts, truncation=False)
        for ids in enc["input_ids"]:
            items_seen += 1
            cumulative_windows += len(ids) // seq_len
            if cumulative_windows >= target_windows:
                dt = time.time() - t0
                print(
                    f"[migrate] reached target: items={items_seen}, "
                    f"windows={cumulative_windows} (>= {target_windows}) in {dt:.1f}s",
                    flush=True,
                )
                return items_seen, cumulative_windows
        if items_seen - last_log >= log_every:
            last_log = items_seen
            dt = time.time() - t0
            rate = items_seen / max(dt, 1e-9)
            print(
                f"[migrate] items={items_seen} windows={cumulative_windows}/"
                f"{target_windows} ({rate:.0f} items/s)",
                flush=True,
            )

    # Exhausted the corpus before reaching the target — should not happen for a
    # valid resume, but surface it loudly rather than writing a bogus cursor.
    raise RuntimeError(
        f"Corpus exhausted after {items_seen} items / {cumulative_windows} windows "
        f"before reaching target_windows={target_windows}. The dataset, shuffle "
        f"seed, seq_len, or tokenizer does not match the original collection."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True, help="Path to <run>_training_state.pt")
    ap.add_argument(
        "--resolved-config",
        required=True,
        help="Path to the run's resolved_config.json (for dataset params).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the cursor but do not write the state file.",
    )
    args = ap.parse_args()

    state = torch.load(args.state, map_location="cpu")
    target_windows = int(state.get("corpus_skip_sequences", 0))
    existing = state.get("corpus_skip_items")
    print(
        f"[migrate] state={args.state}\n"
        f"[migrate]   chunk_index={state.get('chunk_index')}\n"
        f"[migrate]   corpus_skip_sequences={target_windows}\n"
        f"[migrate]   existing corpus_skip_items={existing}",
        flush=True,
    )
    if target_windows <= 0:
        print("[migrate] corpus_skip_sequences <= 0; nothing to migrate.", flush=True)
        return
    if existing:
        print(
            f"[migrate] corpus_skip_items already set to {existing}; refusing to "
            "overwrite. Delete the key first if you really want to recompute.",
            flush=True,
        )
        return

    cfg = json.load(open(args.resolved_config))
    ds = cfg["dataset"]
    item_cursor, windows = derive_item_cursor(
        dataset_name=ds["dataset_name"],
        dataset_config=ds.get("dataset_config", "") or "",
        shuffle_seed=int(ds["seed"]),
        model_name=ds["model_name"],
        seq_len=int(ds["seq_len"]),
        target_windows=target_windows,
    )

    print(
        f"[migrate] RESULT: corpus_skip_items={item_cursor} "
        f"(covers {windows} windows; target was {target_windows}; "
        f"dropping {windows - target_windows} trailing window(s) of the boundary item)",
        flush=True,
    )

    if args.dry_run:
        print("[migrate] --dry-run: not writing state file.", flush=True)
        return

    state["corpus_skip_items"] = int(item_cursor)
    torch.save(state, args.state)
    print(f"[migrate] wrote corpus_skip_items={item_cursor} -> {args.state}", flush=True)


if __name__ == "__main__":
    main()
