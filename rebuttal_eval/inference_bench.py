"""REQ-7: transcoder inference time and memory, spline vs linear.

Benchmarks the CLT itself (encode + decode_dense) on real val activation
windows, per checkpoint: latency mean ± std over timed iterations, tokens/s,
peak CUDA memory, L0, d_t. All checkpoints in one invocation must share a
base model (same val activation dir) so the comparison is on identical
inputs and hardware (spec §2.5).

Usage:
  python -m rebuttal_eval.inference_bench --checkpoints CKPT [CKPT ...] \
      --activation-dir <val_dir> --out-dir <dir> [--n-warmup 5] [--n-iters 30]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from rebuttal_eval.common import (
    Provenance,
    emit,
    fmt,
    git_sha,
    load_transcoder,
    load_val_dataset,
    sample_indices,
)


def bench_checkpoint(
    checkpoint: str,
    dataset,
    indices: list[int],
    device: torch.device,
    n_warmup: int,
    n_iters: int,
) -> dict[str, Any]:
    model = load_transcoder(checkpoint, device=device, dtype=torch.float32)
    samples = [
        {
            key: value.to(device=device, dtype=torch.float32)
            for key, value in dataset[idx].items()
        }
        for idx in indices
    ]
    n_pos = samples[0]["mlp_inputs"].shape[1]
    use_cuda = device.type == "cuda"

    @torch.no_grad()
    def forward(sample):
        activations = model.encode(sample["mlp_inputs"])
        y_hat = model.decode_dense(activations, input_acts=sample["mlp_inputs"])
        return activations, y_hat

    for i in range(n_warmup):
        forward(samples[i % len(samples)])
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies_ms: list[float] = []
    l0_values: list[float] = []
    for i in range(n_iters):
        sample = samples[i % len(samples)]
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        activations, _ = forward(sample)
        if use_cuda:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1e3)
        l0_values.append(float((activations > 0).float().sum(dim=-1).mean().item()))

    peak_mem_gib = (
        torch.cuda.max_memory_allocated() / 2**30 if use_cuda else None
    )
    latency_mean = statistics.fmean(latencies_ms)
    result = {
        "checkpoint": checkpoint,
        "encoder_type": model.encoder_type,
        "d_transcoder": model.d_transcoder,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "n_pos_per_window": n_pos,
        "latency_ms_mean": latency_mean,
        "latency_ms_std": statistics.stdev(latencies_ms) if n_iters > 1 else 0.0,
        "tokens_per_s": n_pos / (latency_mean / 1e3),
        "peak_mem_gib": peak_mem_gib,
        "l0_active_per_pos": statistics.fmean(l0_values),
        "n_iters": n_iters,
    }
    del model, samples
    if use_cuda:
        torch.cuda.empty_cache()
    return result


def _render_markdown(results: list[dict[str, Any]], device: str) -> str:
    lines = [
        "# Transcoder inference cost (REQ-7)",
        "",
        f"Identical inputs and hardware ({device}) for every row; fp32; "
        f"encode + decode_dense per {results[0]['n_pos_per_window']}-token "
        f"window. git: {git_sha()}",
        "",
        "| Encoder | d_t | L0 | latency ms (mean ± std) | tokens/s | peak mem GiB |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(results, key=lambda r: (r["encoder_type"], r["d_transcoder"])):
        lines.append(
            f"| {row['encoder_type']} | {row['d_transcoder']:,} "
            f"| {fmt(row['l0_active_per_pos'], 4)} "
            f"| {fmt(row['latency_ms_mean'], 4)} ± {fmt(row['latency_ms_std'], 3)} "
            f"| {fmt(row['tokens_per_s'], 4)} | {fmt(row['peak_mem_gib'], 3)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dataset = load_val_dataset(args.activation_dir, split=args.split)
    indices = sample_indices(len(dataset), args.n_windows, args.seed)

    results = []
    for checkpoint in args.checkpoints:
        result = bench_checkpoint(
            checkpoint, dataset, indices, device, args.n_warmup, args.n_iters
        )
        print(
            f"{Path(checkpoint).name}: {result['latency_ms_mean']:.2f} ms/window, "
            f"{result['tokens_per_s']:.0f} tok/s, peak "
            f"{fmt(result['peak_mem_gib'], 3)} GiB"
        )
        results.append(result)

    payload = {
        "git_sha": git_sha(),
        "device": args.device,
        "activation_dir": args.activation_dir,
        "seed": args.seed,
        "results": results,
    }
    name = f"inference_bench_{args.label}" if args.label else "inference_bench"
    emit(args.out_dir, name, payload, _render_markdown(results, args.device))

    provenance = Provenance(script="rebuttal_eval.inference_bench", seed=args.seed)
    for row in results:
        provenance.record(
            "2.5", "latency_ms_mean", row["latency_ms_mean"],
            checkpoint=row["checkpoint"],
        )
        provenance.record(
            "2.5", "peak_mem_gib", row["peak_mem_gib"], checkpoint=row["checkpoint"]
        )
    provenance.write(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
