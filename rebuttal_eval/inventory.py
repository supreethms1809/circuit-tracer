"""§0.4 inventory: checkpoints, evaluation outputs, and result files.

Consolidates the ad-hoc listing scripts (scripts/find_runs.py,
scripts/check_progress.py, scripts/find_correct_checkpoints.py) into one
read-only CLI. The spec requires this inventory to be emitted before any
result extraction.

Usage:
  python -m rebuttal_eval.inventory --out-dir <dir> \
      [--roots /gscratch/ssuresh/results/paper ...] [--max-depth 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rebuttal_eval.common import emit, git_sha

DEFAULT_ROOTS = ["/gscratch/ssuresh/results/paper"]

_SUITE_MARKERS = ("aggregate_metrics.json", "manifest.json", "resolved_config.json")


def _entry(path: Path, kind: str) -> dict[str, Any]:
    stat = path.stat()
    total_size = 0
    n_files = 0
    # Immediate files only: recursive sizes over multi-TB result trees are
    # slow and not needed for the inventory table.
    for root, _dirs, files in os.walk(path):
        for file_name in files:
            try:
                total_size += os.path.getsize(os.path.join(root, file_name))
                n_files += 1
            except OSError:
                continue
        break
    return {
        "kind": kind,
        "path": str(path),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": total_size,
        "n_files": n_files,
    }


def _checkpoint_entry(path: Path) -> dict[str, Any]:
    entry = _entry(path, "checkpoint")
    try:
        from safetensors import safe_open

        with safe_open(str(path / "metadata.safetensors"), framework="pt") as handle:
            keys = handle.keys()
            entry["arch"] = {
                "n_layers": int(handle.get_tensor("n_layers").item()),
                "d_transcoder": int(handle.get_tensor("d_transcoder").item()),
                "d_model": int(handle.get_tensor("d_model").item()),
                "encoder_type": (
                    "linear"
                    if "encoder_type_linear" in keys
                    and bool(handle.get_tensor("encoder_type_linear").item())
                    else "kan"
                ),
            }
    except Exception as error:  # unreadable metadata is itself worth reporting
        entry["arch_error"] = str(error)
    return entry


def _suite_entry(path: Path) -> dict[str, Any]:
    entry = _entry(path, "suite_output")
    entry["artifacts_present"] = {
        name: (path / name).exists()
        for name in (
            "aggregate_metrics.json",
            "manifest.json",
            "resolved_config.json",
            "per_example_metrics.jsonl",
            "report.md",
            "tables.csv",
        )
    }
    aggregate_path = path / "aggregate_metrics.json"
    if aggregate_path.exists():
        try:
            aggregate = json.loads(aggregate_path.read_text())
            entry["benchmark_size"] = aggregate.get("benchmark_size")
            entry["benchmark_manifest_version"] = aggregate.get(
                "benchmark_manifest_version"
            )
            entry["variants"] = sorted(aggregate.get("variants", {}))
        except (json.JSONDecodeError, OSError) as error:
            entry["aggregate_error"] = str(error)
    runs_dir = path / "runs"
    if runs_dir.is_dir():
        entry["runs"] = {
            variant.name: sorted(seed.name for seed in variant.iterdir() if seed.is_dir())
            for variant in sorted(runs_dir.iterdir())
            if variant.is_dir()
        }
    return entry


def build_inventory(roots: list[str], max_depth: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            entries.append({"kind": "missing_root", "path": root})
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            depth = len(Path(dirpath).relative_to(root_path).parts)
            if depth >= max_depth:
                dirnames[:] = []
                continue
            current = Path(dirpath)
            if "metadata.safetensors" in filenames:
                entries.append(_checkpoint_entry(current))
                dirnames[:] = []  # checkpoints have no nested results
            elif any(marker in filenames for marker in _SUITE_MARKERS) and (
                current.parent.name != "evaluation"
            ):
                entries.append(_suite_entry(current))
    return entries


def _render_markdown(entries: list[dict[str, Any]]) -> str:
    checkpoints = [e for e in entries if e["kind"] == "checkpoint"]
    suites = [e for e in entries if e["kind"] == "suite_output"]
    missing = [e for e in entries if e["kind"] == "missing_root"]
    lines = [
        "# Results inventory (spec §0.4)",
        "",
        f"git: {git_sha()}  |  checkpoints: {len(checkpoints)}  |  "
        f"suite outputs: {len(suites)}",
        "",
        "## Checkpoints",
        "",
        "| Path | Encoder | d_t | L | mtime | size |",
        "|---|---|---|---|---|---|",
    ]
    for entry in sorted(checkpoints, key=lambda e: e["path"]):
        arch = entry.get("arch", {})
        lines.append(
            f"| `{entry['path']}` | {arch.get('encoder_type', '?')} "
            f"| {arch.get('d_transcoder', '?')} | {arch.get('n_layers', '?')} "
            f"| {entry['mtime'][:10]} | {entry['size_bytes'] / 1e9:.1f} GB |"
        )
    lines += [
        "",
        "## Suite outputs",
        "",
        "| Path | bench N | variants x seeds | aggregate | per-example | mtime |",
        "|---|---|---|---|---|---|",
    ]
    for entry in sorted(suites, key=lambda e: e["path"]):
        runs = entry.get("runs", {})
        runs_txt = "; ".join(f"{v}:{','.join(s)}" for v, s in runs.items()) or "-"
        artifacts = entry.get("artifacts_present", {})
        lines.append(
            f"| `{entry['path']}` | {entry.get('benchmark_size', '?')} "
            f"| {runs_txt} "
            f"| {'y' if artifacts.get('aggregate_metrics.json') else 'n'} "
            f"| {'y' if artifacts.get('per_example_metrics.jsonl') else 'n'} "
            f"| {entry['mtime'][:10]} |"
        )
    if missing:
        lines += ["", "## Missing roots", ""]
        lines += [f"- `{entry['path']}`" for entry in missing]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    entries = build_inventory(args.roots, args.max_depth)
    payload = {"git_sha": git_sha(), "roots": args.roots, "entries": entries}
    emit(args.out_dir, "inventory", payload, _render_markdown(entries))
    print(
        f"inventory: {sum(e['kind'] == 'checkpoint' for e in entries)} checkpoints, "
        f"{sum(e['kind'] == 'suite_output' for e in entries)} suite outputs "
        f"-> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
