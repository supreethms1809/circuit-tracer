"""Suggest supernodes from graph metrics and optionally write candidates/output graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from macag.graph import CircuitGraph
from macag.utils.supernodes import (
    merge_supernodes_into_payload,
    propose_supernodes,
    supernode_candidates,
)


def _load_json(path: str | Path) -> Any:
    resolved = Path(path)
    try:
        text = resolved.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved}: {exc}") from exc


def _parse_feature_types(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suggest supernodes from graph metrics (influence/activation/connectivity)."
    )
    parser.add_argument("--graph-json", required=True, help="Path to source graph JSON.")
    parser.add_argument(
        "--output-supernodes-json",
        default=None,
        help="Optional path to write suggested supernodes JSON list.",
    )
    parser.add_argument(
        "--output-candidates-json",
        default=None,
        help="Optional path to write candidate node IDs (union of suggested supernodes).",
    )
    parser.add_argument(
        "--output-graph-json",
        default=None,
        help="Optional path to write graph JSON with updated qParams.supernodes.",
    )
    parser.add_argument(
        "--replace-existing-supernodes",
        action="store_true",
        help="When writing --output-graph-json, replace existing qParams.supernodes.",
    )
    parser.add_argument("--top-k", type=int, default=200, help="Top-k salient nodes considered.")
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=3,
        help="Minimum group size to emit as a supernode.",
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=12,
        help="Maximum group size; larger groups are chunked.",
    )
    parser.add_argument(
        "--min-salience",
        type=float,
        default=0.0,
        help="Minimum salience threshold for node inclusion.",
    )
    parser.add_argument(
        "--label-prefix",
        default="AUTO",
        help="Prefix used for suggested supernode labels.",
    )
    parser.add_argument(
        "--feature-types",
        default="cross layer transcoder",
        help="Comma-separated feature_type filter (default: cross layer transcoder).",
    )
    parser.add_argument(
        "--no-split-by-ctx",
        action="store_true",
        help="Disable splitting connected components by ctx_idx before chunking.",
    )
    parser.add_argument(
        "--influence-weight",
        type=float,
        default=1.0,
        help="Weight for |influence| in salience score.",
    )
    parser.add_argument(
        "--activation-weight",
        type=float,
        default=0.2,
        help="Weight for |activation| in salience score.",
    )
    parser.add_argument(
        "--token-prob-weight",
        type=float,
        default=0.0,
        help="Weight for |token_prob| in salience score.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not any((args.output_supernodes_json, args.output_candidates_json, args.output_graph_json)):
        parser.error(
            "Provide at least one output target: "
            "--output-supernodes-json, --output-candidates-json, or --output-graph-json."
        )

    payload = _load_json(args.graph_json)
    graph = CircuitGraph.from_dict(payload)
    supernodes = propose_supernodes(
        graph=graph,
        top_k=args.top_k,
        min_group_size=args.min_group_size,
        max_group_size=args.max_group_size,
        min_salience=args.min_salience,
        label_prefix=args.label_prefix,
        feature_types=_parse_feature_types(args.feature_types),
        split_by_ctx=not args.no_split_by_ctx,
        influence_weight=args.influence_weight,
        activation_weight=args.activation_weight,
        token_prob_weight=args.token_prob_weight,
    )
    candidates = supernode_candidates(supernodes)

    if args.output_supernodes_json:
        out = Path(args.output_supernodes_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(supernodes, indent=2))

    if args.output_candidates_json:
        out = Path(args.output_candidates_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(candidates, indent=2))

    if args.output_graph_json:
        merged = merge_supernodes_into_payload(
            graph_payload=payload,
            supernodes=supernodes,
            replace_existing=args.replace_existing_supernodes,
        )
        out = Path(args.output_graph_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=2))

    print(
        f"Suggested {len(supernodes)} supernodes covering {len(candidates)} unique node IDs "
        f"from graph={args.graph_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
