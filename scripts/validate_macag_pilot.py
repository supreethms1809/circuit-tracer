#!/usr/bin/env python3
"""Check one pilot cell against the final-campaign output contract.

Exits 0 only when the cell has the protocol marks the paper will trust:
code version, capped metrics, one-sided Game 2 with both betas, per-leg
baseline blocks, and matched-k status on each non-degenerate leg.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Score-kind directory (e.g. .../logit_gap).")
    parser.add_argument("--tag", default="macag-final-v1")
    args = parser.parse_args()
    root = Path(args.run_dir)
    errors: list[str] = []

    g1_path = root / "macag_game1.json"
    if not g1_path.is_file():
        _fail(errors, "missing macag_game1.json")
        g1: dict = {}
    else:
        g1 = _load(g1_path)
        version = (g1.get("code_version") or {}).get("git_commit") or g1.get("code_version")
        if not version:
            _fail(errors, "game1 missing code_version")
        params = g1.get("params") or {}
        if params.get("cap_sufficiency") is not True or params.get("cap_necessity") is not True:
            _fail(errors, f"game1 caps not on: {params.get('cap_sufficiency')}, {params.get('cap_necessity')}")
        if g1.get("freeze_mode") != "both":
            _fail(errors, "game1 freeze_mode is not both")

    for beta, name in (("0", "macag_game2_abr_b0.json"), ("0.2", "macag_game2_abr_b0.2.json")):
        path = root / name
        if not path.is_file():
            _fail(errors, f"missing {name}")
            continue
        g2 = _load(path)
        params = g2.get("params") or {}
        if params.get("one_sided") is not True:
            _fail(errors, f"{name} one_sided is not true")
        if params.get("structural_gap_symmetry") is True:
            _fail(errors, f"{name} structural_gap_symmetry is true")
        if "degenerate_target" not in g2 or "degenerate_foil" not in g2:
            _fail(errors, f"{name} missing degenerate_target/degenerate_foil")
        have = params.get("beta")
        if have is None or abs(float(have) - float(beta)) > 1e-9:
            _fail(errors, f"{name} beta is {have}, expected {beta}")
        if params.get("freeze_attention") is not False:
            _fail(errors, f"{name} freeze_attention is not false")

    bl_path = root / "macag_baselines.json"
    if not bl_path.is_file():
        _fail(errors, "missing macag_baselines.json")
    else:
        bl = _load(bl_path)
        stats = bl.get("stats") or {}
        if "wall_seconds" not in stats:
            _fail(errors, "baselines missing wall_seconds")
        for method, entry in (bl.get("methods") or {}).items():
            legs = entry.get("legs") or {}
            if "frozen" not in legs or "unfrozen" not in legs:
                _fail(errors, f"{method} missing frozen/unfrozen legs")
                continue
            for leg, block in legs.items():
                read = block.get("matched_k_read") or block.get("matched_k") or {}
                if read.get("status") not in ("ok", "degenerate_leg", "unavailable_at_k", "no_game1"):
                    _fail(errors, f"{method}.{leg} matched status {read.get('status')!r}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"pilot ok: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
