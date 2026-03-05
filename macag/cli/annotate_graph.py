"""Annotate a circuit-tracer graph JSON with MACAG evidence for visualization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


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


def _ensure_qparams(graph_payload: dict[str, Any]) -> dict[str, Any]:
    qparams = graph_payload.get("qParams")
    if not isinstance(qparams, dict):
        qparams = {}
        graph_payload["qParams"] = qparams
    qparams.setdefault("pinnedIds", [])
    qparams.setdefault("supernodes", [])
    qparams.setdefault("linkType", "both")
    qparams.setdefault("clickedId", "")
    qparams.setdefault("sg_pos", "")
    return qparams


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _build_groups(macag_result: dict[str, Any], label_prefix: str) -> tuple[list[str], list[list[str]]]:
    evidence = macag_result.get("evidence", {})
    if isinstance(evidence, list):
        evidence = {
            "E_star": evidence,
            "E_y": evidence,
        }
    game = macag_result.get("game")

    if game == "game2":
        shared = [str(node) for node in evidence.get("shared", [])]
        unique_y = [str(node) for node in evidence.get("unique_y", [])]
        unique_foil = [str(node) for node in evidence.get("unique_foil", [])]
        e_y = [str(node) for node in evidence.get("E_y", [])]
        e_foil = [str(node) for node in evidence.get("E_foil", [])]

        pinned = _dedupe_preserve_order(shared + unique_y + unique_foil + e_y + e_foil)
        supernodes: list[list[str]] = []
        if shared:
            supernodes.append([f"{label_prefix}:shared", *shared])
        if unique_y:
            supernodes.append([f"{label_prefix}:unique_y", *unique_y])
        if unique_foil:
            supernodes.append([f"{label_prefix}:unique_foil", *unique_foil])
        return pinned, supernodes

    e_star = [str(node) for node in evidence.get("E_star", evidence.get("E_y", []))]
    pinned = _dedupe_preserve_order(e_star)
    supernodes = [[f"{label_prefix}:E_star", *e_star]] if e_star else []
    return pinned, supernodes


def annotate_graph_with_macag(
    graph_json_path: str | Path,
    macag_result_json_path: str | Path,
    output_path: str | Path | None = None,
    label_prefix: str = "MACAG",
    replace_existing_supernodes: bool = False,
    replace_existing_pins: bool = False,
    update_metadata_index: bool = True,
) -> Path:
    """Annotate graph JSON qParams with MACAG evidence groups."""
    graph_payload = _load_json(graph_json_path)
    macag_result = _load_json(macag_result_json_path)
    qparams = _ensure_qparams(graph_payload)

    pinned, supernodes = _build_groups(macag_result=macag_result, label_prefix=label_prefix)

    current_pins = [str(node) for node in qparams.get("pinnedIds", [])]
    if replace_existing_pins:
        merged_pins = pinned
    else:
        merged_pins = _dedupe_preserve_order(current_pins + pinned)
    qparams["pinnedIds"] = merged_pins

    current_supernodes = qparams.get("supernodes", [])
    if not isinstance(current_supernodes, list):
        current_supernodes = []
    if replace_existing_supernodes:
        merged_supernodes = supernodes
    else:
        merged_supernodes = list(current_supernodes) + supernodes
    qparams["supernodes"] = merged_supernodes

    output = Path(output_path) if output_path else Path(graph_json_path)
    output.write_text(json.dumps(graph_payload, indent=2))

    if update_metadata_index and isinstance(graph_payload.get("metadata"), dict):
        try:
            from circuit_tracer.frontend.utils import add_graph_metadata

            add_graph_metadata(graph_payload["metadata"], str(output.parent))
        except Exception as exc:
            # Metadata indexing is a convenience for the local frontend; failure here
            # should not block writing the annotated graph payload.
            LOGGER.warning("Failed to update graph metadata index: %s", exc)

    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate a circuit graph JSON with MACAG evidence for frontend visualization."
    )
    parser.add_argument("--graph-json", required=True, help="Path to existing circuit graph JSON.")
    parser.add_argument("--macag-result-json", required=True, help="Path to MACAG output JSON.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to write annotated graph JSON (defaults to in-place update of --graph-json).",
    )
    parser.add_argument(
        "--label-prefix",
        default="MACAG",
        help="Prefix for generated supernode labels.",
    )
    parser.add_argument(
        "--replace-existing-supernodes",
        action="store_true",
        help="Replace existing supernodes instead of appending.",
    )
    parser.add_argument(
        "--replace-existing-pins",
        action="store_true",
        help="Replace existing pinned nodes instead of appending.",
    )
    parser.add_argument(
        "--no-metadata-index-update",
        action="store_true",
        help="Do not update graph-metadata.json in output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    annotate_graph_with_macag(
        graph_json_path=args.graph_json,
        macag_result_json_path=args.macag_result_json,
        output_path=args.output_json,
        label_prefix=args.label_prefix,
        replace_existing_supernodes=args.replace_existing_supernodes,
        replace_existing_pins=args.replace_existing_pins,
        update_metadata_index=not args.no_metadata_index_update,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
