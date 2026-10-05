#!/usr/bin/env python3
"""Longest-first (cell, stage) assignment for the final MACAG campaign.

Reads a cost table JSON and writes one assignment file the MIB launcher
consumes via ASSIGNMENT_FILE. Stages stay ordered per cell: a later stage is
not scheduled to start before the previous stage of that cell finishes.

Cost table (missing costs default to 1):

    {"cells": [{"cell_id": "gemma2-426k/mib_gemma2_ioi_0007",
                "cost": {"graph": 1, "game1_baselines": 40,
                         "game2_b0": 20, "game2_b0p2": 20}}]}

``cell_id`` is ``<clt-tag>/<slug>``, matching the launcher's filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from macag.utils.final_protocol import assign_stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", required=True, help="Cost-table JSON.")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", required=True, help="Assignment JSON (list of worker queues).")
    args = parser.parse_args()
    payload = json.loads(Path(args.costs).read_text())
    cells = payload["cells"] if isinstance(payload, dict) else payload
    assigned = assign_stages(cells, args.workers)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(assigned, indent=2) + "\n")
    loads = [sum(item["cost"] for item in queue) for queue in assigned]
    print(f"wrote {out} workers={args.workers} max_load={max(loads) if loads else 0:.1f}")


if __name__ == "__main__":
    main()
