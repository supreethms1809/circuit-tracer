"""REQ-7: attribution cost vs dictionary size d_t (and vs active features).

For each checkpoint, builds the circuit-tracer ReplacementModel and times
`attribute()` on a fixed prompt list: seconds/prompt, peak CUDA memory, and
active feature count. Emits the d_t-scaling table plus a linear fit of cost
against active features — the AC's "realistic dictionary sizes" question asks
whether cost tracks active features rather than d_t, and that must be
verified empirically, not asserted.

Usage:
  python -m rebuttal_eval.attr_scaling --checkpoints CKPT [CKPT ...] \
      --model gpt2 --out-dir <dir> [--n-prompts 5] \
      [--benchmark experiments/paper_configs/benchmarks/neurips_core.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from rebuttal_eval.common import Provenance, emit, fmt, git_sha, load_transcoder

DEFAULT_BENCHMARK = "experiments/paper_configs/benchmarks/neurips_core.json"


def load_prompts(benchmark_path: str, n_prompts: int) -> list[str]:
    payload = json.loads(Path(benchmark_path).read_text())
    entries = payload.get("benchmark_entries", payload if isinstance(payload, list) else [])
    prompts = [entry["prompt"] for entry in entries][:n_prompts]
    if not prompts:
        raise ValueError(f"no prompts found in {benchmark_path}")
    return prompts


def bench_attribution(
    checkpoint: str,
    model_name: str,
    prompts: list[str],
    device: torch.device,
    max_features: int,
    batch_size: int,
) -> dict[str, Any]:
    from circuit_tracer.attribution.attribute_transformerlens import attribute
    from spline_clt.paper.evaluate import load_replacement_model

    transcoder = load_transcoder(checkpoint, device=device, dtype=torch.float32)
    replacement_model = load_replacement_model(
        model_name, transcoder, device, dtype=torch.float32
    )
    use_cuda = device.type == "cuda"

    per_prompt: list[dict[str, Any]] = []
    for prompt in prompts:
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        graph = attribute(
            prompt=prompt,
            model=replacement_model,
            max_n_logits=10,
            desired_logit_prob=0.95,
            max_feature_nodes=max_features,
            batch_size=batch_size,
            verbose=False,
        )
        if use_cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        n_active = (
            int(graph.active_features.shape[0])
            if hasattr(graph, "active_features")
            else 0
        )
        per_prompt.append(
            {
                "prompt": prompt,
                "seconds": seconds,
                "peak_mem_gib": (
                    torch.cuda.max_memory_allocated() / 2**30 if use_cuda else None
                ),
                "active_features": n_active,
            }
        )
        del graph

    seconds_list = [p["seconds"] for p in per_prompt]
    result = {
        "checkpoint": checkpoint,
        "encoder_type": transcoder.encoder_type,
        "d_transcoder": transcoder.d_transcoder,
        "seconds_per_prompt_mean": statistics.fmean(seconds_list),
        "seconds_per_prompt_std": (
            statistics.stdev(seconds_list) if len(seconds_list) > 1 else 0.0
        ),
        "peak_mem_gib_max": max(
            (p["peak_mem_gib"] for p in per_prompt if p["peak_mem_gib"] is not None),
            default=None,
        ),
        "active_features_mean": statistics.fmean(
            p["active_features"] for p in per_prompt
        ),
        "per_prompt": per_prompt,
    }
    del transcoder, replacement_model
    if use_cuda:
        torch.cuda.empty_cache()
    return result


def _cost_vs_active_fit(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Least-squares fit seconds ~ a * active_features + b across all prompts."""
    points = [
        (p["active_features"], p["seconds"])
        for result in results
        for p in result["per_prompt"]
    ]
    if len(points) < 3:
        return None
    xs = torch.tensor([p[0] for p in points], dtype=torch.float64)
    ys = torch.tensor([p[1] for p in points], dtype=torch.float64)
    design = torch.stack([xs, torch.ones_like(xs)], dim=1)
    solution = torch.linalg.lstsq(design, ys.unsqueeze(1)).solution.squeeze(1)
    predictions = design @ solution
    ss_res = float(((ys - predictions) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    return {
        "slope_seconds_per_active_feature": float(solution[0]),
        "intercept_seconds": float(solution[1]),
        "r_squared": 1.0 - ss_res / max(ss_tot, 1e-16),
        "n_points": len(points),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    fit = payload["cost_vs_active_features_fit"]
    lines = [
        "# Attribution cost scaling (REQ-7)",
        "",
        f"model: {payload['model']}, {payload['n_prompts']} identical prompts "
        f"per row, device {payload['device']}, max_features "
        f"{payload['max_features']}. git: {git_sha()}",
        "",
        "| Encoder | d_t | s/prompt (mean ± std) | peak mem GiB | active feats |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(results, key=lambda r: (r["encoder_type"], r["d_transcoder"])):
        lines.append(
            f"| {row['encoder_type']} | {row['d_transcoder']:,} "
            f"| {fmt(row['seconds_per_prompt_mean'], 4)} ± "
            f"{fmt(row['seconds_per_prompt_std'], 3)} "
            f"| {fmt(row['peak_mem_gib_max'], 3)} "
            f"| {fmt(row['active_features_mean'], 4)} |"
        )
    if fit:
        lines += [
            "",
            "## Cost vs active features (pooled over all rows/prompts)",
            "",
            f"- seconds ≈ {fmt(fit['slope_seconds_per_active_feature'], 3)} × "
            f"active_features + {fmt(fit['intercept_seconds'], 3)} "
            f"(R² = {fmt(fit['r_squared'], 3)}, n = {fit['n_points']})",
            "- High R² supports 'cost tracks active features, not dictionary "
            "size'; low R² means that claim must NOT be made in the rebuttal.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=7500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    prompts = load_prompts(args.benchmark, args.n_prompts)
    results = []
    for checkpoint in args.checkpoints:
        result = bench_attribution(
            checkpoint, args.model, prompts, device, args.max_features, args.batch_size
        )
        print(
            f"{Path(checkpoint).name}: "
            f"{result['seconds_per_prompt_mean']:.1f} s/prompt, peak "
            f"{fmt(result['peak_mem_gib_max'], 3)} GiB, "
            f"{result['active_features_mean']:.0f} active feats"
        )
        results.append(result)

    payload = {
        "git_sha": git_sha(),
        "model": args.model,
        "device": args.device,
        "n_prompts": len(prompts),
        "max_features": args.max_features,
        "results": results,
        "cost_vs_active_features_fit": _cost_vs_active_fit(results),
    }
    name = f"attr_scaling_{args.label}" if args.label else "attr_scaling"
    emit(args.out_dir, name, payload, _render_markdown(payload))

    provenance = Provenance(script="rebuttal_eval.attr_scaling")
    for row in results:
        provenance.record(
            "2.5", "seconds_per_prompt_mean", row["seconds_per_prompt_mean"],
            checkpoint=row["checkpoint"],
        )
    provenance.write(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
