"""Name features in circuit graphs via targeted autointerp + clerp writeback.

Extracts unique ``(layer, feature)`` IDs from a suite's
``evaluation/graphs/*.json``, runs collect → explain → score on exactly those
IDs (wikitext-2 val+test corpus — same held-out stream as REQ-4), then writes
explanations into each graph node's ``clerp`` field (annotated copies).

Corpus choice: keep wikitext for max-activating contexts and scoring. RAVEL
prompts are short attribute templates and make a poor activation corpus /
scoring reservoir; targeting graph feature IDs is enough to ground labels in
the circuit, while wiki contexts keep detection/fuzzing comparable to REQ-4.

Usage:
  python -m rebuttal_eval.autointerp.name_graphs \\
      --graphs-dir <suite>/runs/.../evaluation/graphs \\
      --checkpoint <ckpt> --model gpt2 --out-dir <dir> \\
      --backend openai_compat \\
      --endpoint-file results/rebuttal/vllm_endpoint.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rebuttal_eval.autointerp.graph_features import (
    extract_features_from_graphs_dir,
    load_explanations,
    load_scores,
    write_clerps_to_graphs,
)
from rebuttal_eval.autointerp import run as run_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs-dir", required=True,
                        help="directory of circuit-tracer graph JSON files")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True,
                        help="autointerp artifacts + annotated graphs/")
    parser.add_argument("--backend", default="openai_compat",
                        choices=["openai_compat", "anthropic"])
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--endpoint-file", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-stat-tokens", type=int, default=250_000)
    parser.add_argument("--n-example-tokens", type=int, default=1_000_000)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-detection", type=int, default=10)
    parser.add_argument("--n-fuzzing", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default=None)
    parser.add_argument("--label", default="graph_features")
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="reuse existing collection.json / feature_list.json in --out-dir",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="explain + write clerps only (no detection/fuzzing)",
    )
    parser.add_argument(
        "--include-scores-in-clerp",
        action="store_true",
        help="append det/fuzz accuracies to the visible clerp string",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="only write feature_list.json and exit",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_list_path = out_dir / "feature_list.json"
    annotated_dir = out_dir / "annotated_graphs"

    if args.skip_collect and feature_list_path.exists():
        feature_payload = json.loads(feature_list_path.read_text())
    else:
        feature_payload = extract_features_from_graphs_dir(args.graphs_dir)
        feature_list_path.write_text(json.dumps(feature_payload, indent=2))
    print(
        f"feature list: {feature_payload['n_unique_features']} unique IDs from "
        f"{feature_payload['n_graphs']} graphs -> {feature_list_path}"
    )
    if args.extract_only:
        return 0

    run_argv = [
        "--checkpoint", args.checkpoint,
        "--model", args.model,
        "--out-dir", str(out_dir),
        "--backend", args.backend,
        "--llm-model", args.llm_model,
        "--base-url", args.base_url,
        "--endpoint-file", args.endpoint_file,
        "--feature-list", str(feature_list_path),
        "--top-k", str(args.top_k),
        "--n-stat-tokens", str(args.n_stat_tokens),
        "--n-example-tokens", str(args.n_example_tokens),
        "--window-len", str(args.window_len),
        "--batch-size", str(args.batch_size),
        "--n-detection", str(args.n_detection),
        "--n-fuzzing", str(args.n_fuzzing),
        "--seed", str(args.seed),
        "--label", args.label,
    ]
    if args.device:
        run_argv.extend(["--device", args.device])
    if args.skip_collect:
        run_argv.append("--skip-collect")
    if args.skip_score:
        run_argv.append("--skip-score")

    rc = run_mod.main(run_argv)
    if rc != 0:
        return rc

    explanations = load_explanations(out_dir / "explanations.jsonl")
    scores_path = out_dir / "feature_scores.jsonl"
    scores = load_scores(scores_path if scores_path.exists() else None)
    totals = write_clerps_to_graphs(
        args.graphs_dir,
        explanations,
        annotated_dir,
        scores,
        include_scores_in_clerp=args.include_scores_in_clerp,
    )
    summary_path = annotated_dir / "clerp_writeback_summary.json"
    summary_path.write_text(json.dumps(totals, indent=2))
    print(
        f"name_graphs: wrote clerps for {totals['named']}/{totals['feature_nodes']} "
        f"feature nodes across {totals['n_graphs']} graphs -> {annotated_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
