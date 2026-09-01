"""Collect max-activating contexts with decoded tokens for auto-interp.

Streams held-out corpus text (wikitext validation+test — disjoint from the
training split), runs the base model and the CLT encoder, and per sampled
feature stores:

- top-K max-activating windows (explanation shard) with per-token activations,
- a reservoir of activating windows from a disjoint scoring shard,
- a reservoir of non-activating windows (verified feature-silent),
- per-feature activation frequency, plus corpus-level dead-feature fraction.

Feature sampling is uniform random over alive features by default (the
mandatory reported control in spec §2.3/§3). Pass ``--feature-list`` to target
an explicit set of ``(layer, feature)`` IDs instead — used for naming features
that appear in RAVEL (or other) circuit graphs.

Usage:
  python -m rebuttal_eval.autointerp.collect --checkpoint <ckpt> \
      --model gpt2 --out-dir <dir> [--n-features 200] [--top-k 20] \
      [--n-stat-tokens 250000] [--n-example-tokens 1000000] [--window-len 64]

  # Targeted graph features:
  python -m rebuttal_eval.autointerp.collect --checkpoint <ckpt> \
      --model gpt2 --out-dir <dir> --feature-list feature_list.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import torch

from rebuttal_eval.common import git_sha, load_transcoder
from rebuttal_eval.autointerp.graph_features import load_feature_list

RESERVOIR_CAP = 50


def iter_token_windows(
    lm: Any, window_len: int, seed: int
) -> Iterator[torch.Tensor]:
    """Yield fixed-length token windows from held-out wikitext text."""
    from datasets import load_dataset

    texts: list[str] = []
    for split in ("validation", "test"):
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        texts.extend(row["text"] for row in dataset if row["text"].strip())
    rng = random.Random(seed)
    rng.shuffle(texts)

    buffer: list[int] = []
    for text in texts:
        buffer.extend(lm.to_tokens(text, prepend_bos=False).squeeze(0).tolist())
        while len(buffer) >= window_len:
            yield torch.tensor(buffer[:window_len], dtype=torch.long)
            buffer = buffer[window_len:]


class _FeatureCollector:
    """Per-feature example accumulators (top-K heap + reservoirs)."""

    def __init__(self, top_k: int, seed: int):
        self.top_k = top_k
        self.rng = random.Random(seed)
        self.top_heap: list[tuple[float, int, list[float]]] = []
        self.activating: list[tuple[int, float, list[float]]] = []
        self.non_activating: list[int] = []
        self._seen_activating = 0
        self._seen_non_activating = 0

    def add(
        self, window_id: int, max_act: float, token_acts: list[float], shard: str
    ) -> None:
        if max_act > 0:
            if shard == "explain":
                item = (max_act, window_id, token_acts)
                if len(self.top_heap) < self.top_k:
                    heapq.heappush(self.top_heap, item)
                else:
                    heapq.heappushpop(self.top_heap, item)
            else:
                self._seen_activating += 1
                entry = (window_id, max_act, token_acts)
                if len(self.activating) < RESERVOIR_CAP:
                    self.activating.append(entry)
                else:
                    slot = self.rng.randrange(self._seen_activating)
                    if slot < RESERVOIR_CAP:
                        self.activating[slot] = entry
        else:
            self._seen_non_activating += 1
            if len(self.non_activating) < RESERVOIR_CAP:
                self.non_activating.append(window_id)
            else:
                slot = self.rng.randrange(self._seen_non_activating)
                if slot < RESERVOIR_CAP:
                    self.non_activating[slot] = window_id


@torch.no_grad()
def collect(args: argparse.Namespace) -> dict[str, Any]:
    from transformer_lens import HookedTransformer

    device = torch.device(args.device)
    model = load_transcoder(args.checkpoint, device=device, dtype=torch.float32)
    lm = HookedTransformer.from_pretrained(
        args.model, device=device, fold_ln=False,
        center_writing_weights=False, center_unembed=False,
    )
    lm.eval()
    n_layers, d_transcoder = model.n_layers, model.d_transcoder
    hook_names = [f"blocks.{i}.{model.feature_input_hook}" for i in range(n_layers)]

    def encode_batch(token_batch: torch.Tensor) -> torch.Tensor:
        """(B, P) tokens -> (B, n_layers, P, d_transcoder) activations."""
        _, cache = lm.run_with_cache(token_batch, names_filter=hook_names)
        x = torch.stack([cache[name] for name in hook_names], dim=1)
        return torch.stack(
            [model.encode(x[b].to(torch.float32)) for b in range(x.shape[0])]
        )

    # Pass 1: activation frequency over the stat shard -> alive set + sampling.
    window_iter = iter_token_windows(lm, args.window_len, args.seed)
    fire_counts = torch.zeros(n_layers, d_transcoder, dtype=torch.float64, device=device)
    n_positions = 0
    n_stat_windows = max(1, args.n_stat_tokens // args.window_len)
    batch: list[torch.Tensor] = []
    for _ in range(n_stat_windows):
        try:
            batch.append(next(window_iter))
        except StopIteration:
            break
        if len(batch) == args.batch_size:
            acts = encode_batch(torch.stack(batch).to(device))
            fire_counts += (acts > 0).sum(dim=(0, 2)).double()
            n_positions += acts.shape[0] * acts.shape[2]
            batch = []
    if batch:
        acts = encode_batch(torch.stack(batch).to(device))
        fire_counts += (acts > 0).sum(dim=(0, 2)).double()
        n_positions += acts.shape[0] * acts.shape[2]

    frequency = (fire_counts / max(n_positions, 1)).cpu()
    alive = (frequency > 0).nonzero(as_tuple=False)
    dead_fraction = 1.0 - alive.shape[0] / (n_layers * d_transcoder)
    l0_estimate = float(frequency.sum().item())

    feature_list_path = getattr(args, "feature_list", "") or ""
    if feature_list_path:
        requested = load_feature_list(feature_list_path)
        sampled_pairs = []
        skipped_oob = 0
        for layer, feat in requested:
            if not (0 <= layer < n_layers and 0 <= feat < d_transcoder):
                skipped_oob += 1
                continue
            sampled_pairs.append((layer, feat))
        if not sampled_pairs:
            raise ValueError(
                f"feature-list {feature_list_path} produced no in-range "
                f"(layer, feature) pairs for n_layers={n_layers}, "
                f"d_transcoder={d_transcoder}"
            )
        n_dead_targeted = sum(
            1 for layer, feat in sampled_pairs if frequency[layer, feat].item() == 0
        )
        sampling = (
            f"explicit feature list from {feature_list_path}: "
            f"n={len(sampled_pairs)} targeted "
            f"({n_dead_targeted} dead on wiki corpus, freq=0; "
            f"{skipped_oob} out-of-range skipped); "
            f"alive pool size={alive.shape[0]}"
        )
    else:
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(alive.shape[0], generator=generator)
        sampled = alive[order[: args.n_features]]
        sampled_pairs = [(int(layer), int(feat)) for layer, feat in sampled.tolist()]
        sampling = (
            f"uniform random over alive features (freq>0), "
            f"n={len(sampled_pairs)} of {alive.shape[0]} alive; dead "
            f"features excluded from sampling by construction and the dead "
            f"fraction is reported alongside"
        )

    # Pass 2: example collection over a fresh stream, split into disjoint
    # explanation/scoring shards by window parity.
    collectors = {
        pair: _FeatureCollector(args.top_k, seed=args.seed + i)
        for i, pair in enumerate(sampled_pairs)
    }
    layer_ids = torch.tensor([p[0] for p in sampled_pairs], device=device)
    feat_ids = torch.tensor([p[1] for p in sampled_pairs], device=device)
    windows: list[list[str]] = []
    window_iter = iter_token_windows(lm, args.window_len, args.seed + 1)
    n_example_windows = max(1, args.n_example_tokens // args.window_len)

    def process_batch(token_stack: torch.Tensor) -> None:
        acts = encode_batch(token_stack.to(device))  # (B, L, P, F)
        # (B, P, n_sampled): each sampled feature's activation trace.
        sampled_acts = acts[:, layer_ids, :, feat_ids].permute(1, 2, 0)
        max_acts = sampled_acts.amax(dim=1)  # (B, n_sampled)
        for row in range(token_stack.shape[0]):
            window_id = len(windows)
            windows.append(
                [lm.tokenizer.decode([t]) for t in token_stack[row].tolist()]
            )
            shard = "explain" if window_id % 2 == 0 else "score"
            row_max = max_acts[row]
            fired = (row_max > 0).nonzero(as_tuple=False).squeeze(-1)
            fired_set = set(fired.tolist())
            for feature_index, pair in enumerate(sampled_pairs):
                if feature_index in fired_set:
                    collectors[pair].add(
                        window_id,
                        float(row_max[feature_index].item()),
                        [round(v, 4) for v in sampled_acts[row, :, feature_index].tolist()],
                        shard,
                    )
                else:
                    collectors[pair].add(window_id, 0.0, [], shard)

    batch = []
    for _ in range(n_example_windows):
        try:
            batch.append(next(window_iter))
        except StopIteration:
            break
        if len(batch) == args.batch_size:
            process_batch(torch.stack(batch))
            batch = []
    if batch:
        process_batch(torch.stack(batch))

    features = []
    for pair in sampled_pairs:
        collector = collectors[pair]
        layer, feat = pair
        features.append(
            {
                "layer": layer,
                "feature": feat,
                "activation_frequency": float(frequency[layer, feat].item()),
                "top_examples": [
                    {"window_id": wid, "max_act": val, "token_acts": acts_list}
                    for val, wid, acts_list in sorted(collector.top_heap, reverse=True)
                ],
                "scoring_activating": [
                    {"window_id": wid, "max_act": val, "token_acts": acts_list}
                    for wid, val, acts_list in collector.activating
                ],
                "scoring_non_activating": collector.non_activating,
            }
        )

    return {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "base_model": args.model,
            "encoder_type": getattr(model, "encoder_type", "linear"),
            "d_transcoder": d_transcoder,
            "n_layers": n_layers,
            "git_sha": git_sha(),
            "seed": args.seed,
            "window_len": args.window_len,
            "n_stat_positions": n_positions,
            "n_example_windows": len(windows),
            "corpus": "Salesforce/wikitext wikitext-2-raw-v1 validation+test "
                      "(held out from training split)",
            "sampling": sampling,
            "feature_list": feature_list_path or None,
            "n_features_targeted": len(sampled_pairs),
            "dead_feature_fraction": dead_fraction,
            "l0_estimate_active_per_pos": l0_estimate,
            "explain_score_shards": "disjoint by window parity",
        },
        "features": features,
        "windows": windows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-features", type=int, default=200,
                        help="random alive sample size (ignored with --feature-list)")
    parser.add_argument(
        "--feature-list",
        default="",
        help="JSON feature list of (layer, feature) pairs; skips random sampling",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-stat-tokens", type=int, default=250_000)
    parser.add_argument("--n-example-tokens", type=int, default=1_000_000)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    payload = collect(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "collection.json"
    out_path.write_text(json.dumps(payload, default=str))
    meta = payload["meta"]
    print(
        f"collected {len(payload['features'])} features "
        f"({meta['n_example_windows']} windows); dead fraction "
        f"{meta['dead_feature_fraction']:.3f}; L0 est "
        f"{meta['l0_estimate_active_per_pos']:.1f} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
