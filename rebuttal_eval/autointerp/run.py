"""End-to-end auto-interp: collect -> explain -> score -> aggregate (REQ-4).

Emits mean ± std detection/fuzzing accuracy across features, together with
the §2.3 controls block (L0, dead-feature fraction and exclusion policy,
sampling method, contexts per feature, explainer/scorer model names).

Usage:
  python -m rebuttal_eval.autointerp.run --checkpoint <ckpt> --model gpt2 \
      --out-dir <dir> --backend openai_compat \
      [--endpoint-file results/rebuttal/vllm_endpoint.json | --base-url URL] \
      [--llm-model MODEL] [--n-features 200] [--skip-collect]

  # Target features from RAVEL (or other) graphs:
  python -m rebuttal_eval.autointerp.run ... --feature-list feature_list.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from rebuttal_eval.common import Provenance, emit, fmt, git_sha
from rebuttal_eval.autointerp import collect as collect_mod
from rebuttal_eval.autointerp.explain import explain_features
from rebuttal_eval.autointerp.llm_backends import build_backend
from rebuttal_eval.autointerp.score import score_all


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate(
    results: list[dict[str, Any]], collection_meta: dict[str, Any], scorer: str
) -> dict[str, Any]:
    detection = [r["detection_accuracy"] for r in results if r["detection_accuracy"] is not None]
    fuzzing = [r["fuzzing_accuracy"] for r in results if r["fuzzing_accuracy"] is not None]
    det_mean, det_std = _mean_std(detection)
    fuz_mean, fuz_std = _mean_std(fuzzing)
    targeted = collection_meta.get("n_features_targeted")
    feature_list = collection_meta.get("feature_list")
    if feature_list:
        dead_policy = (
            "explicit feature list: dead-on-corpus features are retained and "
            "may yield empty example reservoirs (excluded from mean when "
            "detection/fuzzing accuracy is null); dead fraction is corpus-wide"
        )
    else:
        dead_policy = (
            "dead features are excluded from sampling (alive-only draw); "
            "dead fraction reported so scores can be discounted"
        )
    return {
        "n_features_targeted": targeted,
        "n_features_scored_detection": len(detection),
        "n_features_scored_fuzzing": len(fuzzing),
        "detection_accuracy_mean": det_mean,
        "detection_accuracy_std": det_std,
        "fuzzing_accuracy_mean": fuz_mean,
        "fuzzing_accuracy_std": fuz_std,
        "controls": {
            "l0_active_per_pos": collection_meta["l0_estimate_active_per_pos"],
            "dead_feature_fraction": collection_meta["dead_feature_fraction"],
            "dead_features_excluded": dead_policy,
            "sampling": collection_meta["sampling"],
            "corpus": collection_meta["corpus"],
            "feature_list": feature_list,
            "explainer_and_scorer_model": scorer,
            "explain_score_shards": collection_meta["explain_score_shards"],
        },
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["collection_meta"]
    agg = payload["aggregate"]
    lines = [
        f"# Auto-interp scores — {meta['base_model']} / {meta['encoder_type']}",
        "",
        f"- checkpoint: `{meta['checkpoint']}`",
        f"- explainer/scorer: {agg['controls']['explainer_and_scorer_model']}",
        f"- git: {payload['git_sha']}",
        "",
        "| Base model | Encoder | L0 | Dead frac | N feats | Detection | Fuzzing |",
        "|---|---|---|---|---|---|---|",
        (
            f"| {meta['base_model']} | {meta['encoder_type']} "
            f"| {fmt(agg['controls']['l0_active_per_pos'], 4)} "
            f"| {fmt(agg['controls']['dead_feature_fraction'], 3)} "
            f"| {agg['n_features_scored_detection']} "
            f"| {fmt(agg['detection_accuracy_mean'], 3)} ± {fmt(agg['detection_accuracy_std'], 2)} "
            f"| {fmt(agg['fuzzing_accuracy_mean'], 3)} ± {fmt(agg['fuzzing_accuracy_std'], 2)} |"
        ),
        "",
        "Controls:",
        f"- sampling: {agg['controls']['sampling']}",
        f"- corpus: {agg['controls']['corpus']}",
        f"- dead features: {agg['controls']['dead_features_excluded']}",
        f"- shards: {agg['controls']['explain_score_shards']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", default="openai_compat",
                        choices=["openai_compat", "anthropic"])
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--endpoint-file", default="")
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
    parser.add_argument("--n-detection", type=int, default=10)
    parser.add_argument("--n-fuzzing", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-collect", action="store_true",
                        help="Reuse an existing collection.json in --out-dir")
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Stop after explain (still writes explanations.jsonl)",
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collection_path = out_dir / "collection.json"

    if args.skip_collect and collection_path.exists():
        collection = json.loads(collection_path.read_text())
    else:
        import torch

        collect_args = argparse.Namespace(
            checkpoint=args.checkpoint,
            model=args.model,
            out_dir=args.out_dir,
            n_features=args.n_features,
            feature_list=args.feature_list,
            top_k=args.top_k,
            n_stat_tokens=args.n_stat_tokens,
            n_example_tokens=args.n_example_tokens,
            window_len=args.window_len,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device
            or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        collection = collect_mod.collect(collect_args)
        collection_path.write_text(json.dumps(collection, default=str))

    backend = build_backend(
        args.backend, args.llm_model, base_url=args.base_url,
        endpoint_file=args.endpoint_file,
    )
    explanations = explain_features(
        collection, backend, out_dir / "explanations.jsonl"
    )
    if args.skip_score:
        print(
            f"autointerp: explained {len(explanations)} features "
            f"(--skip-score) -> {out_dir}"
        )
        return 0

    results = score_all(
        collection,
        explanations,
        backend,
        out_dir / "feature_scores.jsonl",
        seed=args.seed,
        n_detection=args.n_detection,
        n_fuzzing=args.n_fuzzing,
    )

    payload = {
        "git_sha": git_sha(),
        "collection_meta": collection["meta"],
        "aggregate": aggregate(results, collection["meta"], backend.model),
    }
    name = f"autointerp_{args.label}" if args.label else "autointerp"
    emit(out_dir, name, payload, _render_markdown(payload))

    provenance = Provenance(
        script="rebuttal_eval.autointerp.run",
        checkpoint=str(args.checkpoint),
        seed=args.seed,
    )
    agg = payload["aggregate"]
    for cell in ("detection_accuracy_mean", "fuzzing_accuracy_mean"):
        provenance.record("2.3", cell, agg[cell])
    provenance.write(out_dir)

    print(
        f"autointerp: detection {fmt(agg['detection_accuracy_mean'], 3)} ± "
        f"{fmt(agg['detection_accuracy_std'], 2)}, fuzzing "
        f"{fmt(agg['fuzzing_accuracy_mean'], 3)} ± {fmt(agg['fuzzing_accuracy_std'], 2)} "
        f"over {agg['n_features_scored_detection']} features -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
