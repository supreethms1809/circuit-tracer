"""Extract graph feature IDs and write autointerp labels into graph clerps.

RAVEL (and other) circuit-tracer graphs store CLT feature nodes with
``feature_type == "cross layer transcoder"`` and
``node_id == "{layer}_{feat_idx}_{pos}"``. The integer ``feature`` field is a
Cantor pairing of ``(layer, feat_idx)`` — do **not** use it as the per-layer
feature index.

Usage:
  python -m rebuttal_eval.autointerp.graph_features extract \\
      --graphs-dir <suite>/runs/.../evaluation/graphs --out feature_list.json

  python -m rebuttal_eval.autointerp.graph_features write-clerps \\
      --graphs-dir <graphs> --explanations explanations.jsonl \\
      --scores feature_scores.jsonl --out-dir <annotated_graphs>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FEATURE_TYPE = "cross layer transcoder"


def parse_feature_node_id(node_id: str) -> tuple[int, int] | None:
    """Parse ``{layer}_{feat_idx}_{pos}`` from a CLT feature node_id."""
    parts = str(node_id).split("_")
    if len(parts) != 3:
        return None
    try:
        layer, feat_idx, _pos = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    return layer, feat_idx


def feature_key(layer: int, feature: int) -> str:
    return f"{layer}:{feature}"


def extract_features_from_graph(graph: dict[str, Any]) -> set[tuple[int, int]]:
    """Return unique ``(layer, feat_idx)`` pairs from one graph JSON."""
    pairs: set[tuple[int, int]] = set()
    for node in graph.get("nodes", []):
        if node.get("feature_type") != FEATURE_TYPE:
            continue
        parsed = parse_feature_node_id(node.get("node_id", ""))
        if parsed is not None:
            pairs.add(parsed)
    return pairs


def extract_features_from_graphs_dir(
    graphs_dir: str | Path,
) -> dict[str, Any]:
    """Scan ``*.json`` under ``graphs_dir`` and return a feature-list payload."""
    root = Path(graphs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"graphs dir not found: {root}")

    graphs = sorted(root.glob("*.json"))
    unique: set[tuple[int, int]] = set()
    per_graph: dict[str, int] = {}
    for path in graphs:
        payload = json.loads(path.read_text())
        pairs = extract_features_from_graph(payload)
        unique.update(pairs)
        per_graph[path.name] = len(pairs)

    features = [
        {"layer": layer, "feature": feat}
        for layer, feat in sorted(unique)
    ]
    return {
        "source_graphs_dir": str(root.resolve()),
        "n_graphs": len(graphs),
        "n_unique_features": len(features),
        "features_per_graph": per_graph,
        "features": features,
        "sampling": (
            "explicit feature list extracted from circuit-tracer graphs "
            f"({len(graphs)} files); unique (layer, feature) pairs via "
            "node_id, not Cantor-paired `feature` field"
        ),
    }


def load_feature_list(path: str | Path) -> list[tuple[int, int]]:
    """Load ``(layer, feature)`` pairs from a feature-list JSON.

    Accepts:
    - ``{"features": [{"layer": int, "feature": int}, ...]}``
    - ``[{"layer": int, "feature": int}, ...]``
    - ``[[layer, feature], ...]``
    """
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        rows = payload.get("features", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"unsupported feature-list format in {path}")

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if isinstance(row, dict):
            layer, feat = int(row["layer"]), int(row["feature"])
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            layer, feat = int(row[0]), int(row[1])
        else:
            raise ValueError(f"bad feature-list row: {row!r}")
        pair = (layer, feat)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def load_explanations(path: str | Path) -> dict[str, str]:
    """Load ``{layer:feature -> explanation}`` from explanations.jsonl or .json."""
    file = Path(path)
    text = file.read_text().strip()
    if not text:
        return {}
    if file.suffix == ".json" and text.lstrip().startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict) and "features" not in payload:
            # already a key->explanation map, or nested under explanations
            if all(isinstance(v, str) for v in payload.values()):
                return {str(k): v for k, v in payload.items()}
            if "explanations" in payload:
                return {str(k): str(v) for k, v in payload["explanations"].items()}
    out: dict[str, str] = {}
    for line in text.splitlines():
        row = json.loads(line)
        key = row.get("key") or feature_key(int(row["layer"]), int(row["feature"]))
        out[str(key)] = str(row["explanation"])
    return out


def load_scores(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load per-feature score rows keyed by ``layer:feature``."""
    if path is None:
        return {}
    file = Path(path)
    if not file.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = feature_key(int(row["layer"]), int(row["feature"]))
        out[key] = row
    return out


def format_clerp(
    explanation: str,
    score_row: dict[str, Any] | None = None,
    *,
    include_scores: bool = False,
) -> str:
    """Build the human-readable clerp string for a feature node."""
    text = explanation.strip()
    if not include_scores or not score_row:
        return text
    det = score_row.get("detection_accuracy")
    fuzz = score_row.get("fuzzing_accuracy")
    parts: list[str] = []
    if det is not None:
        parts.append(f"det={det:.2f}")
    if fuzz is not None:
        parts.append(f"fuzz={fuzz:.2f}")
    if not parts:
        return text
    return f"{text} ({', '.join(parts)})"


def annotate_graph_clerps(
    graph: dict[str, Any],
    explanations: dict[str, str],
    scores: dict[str, dict[str, Any]] | None = None,
    *,
    include_scores_in_clerp: bool = False,
) -> dict[str, int]:
    """Mutate ``graph`` nodes in place; return coverage counts."""
    scores = scores or {}
    named = 0
    missing = 0
    feature_nodes = 0
    for node in graph.get("nodes", []):
        if node.get("feature_type") != FEATURE_TYPE:
            continue
        feature_nodes += 1
        parsed = parse_feature_node_id(node.get("node_id", ""))
        if parsed is None:
            missing += 1
            continue
        key = feature_key(*parsed)
        explanation = explanations.get(key)
        if not explanation:
            missing += 1
            continue
        score_row = scores.get(key)
        node["clerp"] = format_clerp(
            explanation,
            score_row,
            include_scores=include_scores_in_clerp,
        )
        autointerp: dict[str, Any] = {
            "key": key,
            "explanation": explanation,
        }
        if score_row is not None:
            autointerp["detection_accuracy"] = score_row.get("detection_accuracy")
            autointerp["fuzzing_accuracy"] = score_row.get("fuzzing_accuracy")
            autointerp["detection_n"] = score_row.get("detection_n")
            autointerp["fuzzing_n"] = score_row.get("fuzzing_n")
        node["autointerp"] = autointerp
        named += 1
    return {
        "feature_nodes": feature_nodes,
        "named": named,
        "missing_explanation": missing,
    }


def write_clerps_to_graphs(
    graphs_dir: str | Path,
    explanations: dict[str, str],
    out_dir: str | Path,
    scores: dict[str, dict[str, Any]] | None = None,
    *,
    include_scores_in_clerp: bool = False,
) -> dict[str, Any]:
    """Copy each graph JSON into ``out_dir`` with feature clerps filled in."""
    src = Path(graphs_dir)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    totals = {
        "n_graphs": 0,
        "feature_nodes": 0,
        "named": 0,
        "missing_explanation": 0,
        "graphs": {},
    }
    for path in sorted(src.glob("*.json")):
        graph = json.loads(path.read_text())
        counts = annotate_graph_clerps(
            graph,
            explanations,
            scores,
            include_scores_in_clerp=include_scores_in_clerp,
        )
        (dst / path.name).write_text(json.dumps(graph))
        totals["n_graphs"] += 1
        for key in ("feature_nodes", "named", "missing_explanation"):
            totals[key] += counts[key]
        totals["graphs"][path.name] = counts
    return totals


def _cmd_extract(args: argparse.Namespace) -> int:
    payload = extract_features_from_graphs_dir(args.graphs_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"extracted {payload['n_unique_features']} unique features from "
        f"{payload['n_graphs']} graphs -> {out}"
    )
    return 0


def _cmd_write_clerps(args: argparse.Namespace) -> int:
    explanations = load_explanations(args.explanations)
    scores = load_scores(args.scores) if args.scores else {}
    totals = write_clerps_to_graphs(
        args.graphs_dir,
        explanations,
        args.out_dir,
        scores,
        include_scores_in_clerp=args.include_scores_in_clerp,
    )
    summary_path = Path(args.out_dir) / "clerp_writeback_summary.json"
    summary_path.write_text(json.dumps(totals, indent=2))
    print(
        f"wrote clerps for {totals['named']}/{totals['feature_nodes']} feature "
        f"nodes across {totals['n_graphs']} graphs -> {args.out_dir}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    extract_p = sub.add_parser("extract", help="Extract unique feature IDs")
    extract_p.add_argument("--graphs-dir", required=True)
    extract_p.add_argument("--out", required=True)
    extract_p.set_defaults(func=_cmd_extract)

    write_p = sub.add_parser("write-clerps", help="Write explanations into clerps")
    write_p.add_argument("--graphs-dir", required=True)
    write_p.add_argument("--explanations", required=True,
                         help="explanations.jsonl from autointerp.run")
    write_p.add_argument("--scores", default="",
                         help="optional feature_scores.jsonl")
    write_p.add_argument("--out-dir", required=True,
                         help="directory for annotated graph copies")
    write_p.add_argument(
        "--include-scores-in-clerp",
        action="store_true",
        help="append det/fuzz accuracies to the clerp string",
    )
    write_p.set_defaults(func=_cmd_write_clerps)

    args = parser.parse_args(argv)
    if args.command == "write-clerps" and not args.scores:
        args.scores = ""
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
