"""REQ-7: reconstruct training GPU-hours from W&B run durations.

Run ids are harvested from `*_training_state.pt` files (the trainer persists
`wandb_run_id` there) and/or given explicitly. A campaign that resumed
across multiple W&B runs must pass every run id in the chain — this script
sums whatever it is given and lists per-run durations so chains can be
audited. Peak training memory was never logged during training; that cell is
emitted as NOT FOUND rather than estimated (spec §0.3.1).

Usage:
  python -m rebuttal_eval.wandb_pull --entity <e> --project <p> \
      --out-dir <dir> [--run-ids ID ...] [--state-files F ...] \
      [--gpus-per-run N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rebuttal_eval.common import Provenance, emit, fmt, git_sha


def harvest_run_ids(state_files: list[str]) -> dict[str, str]:
    """{run_id: source state file} from training_state.pt payloads."""
    import torch

    found: dict[str, str] = {}
    for state_file in state_files:
        try:
            payload = torch.load(
                state_file, map_location="cpu", weights_only=False
            )
            run_id = payload.get("wandb_run_id")
            if run_id:
                found[str(run_id)] = state_file
        except Exception as error:
            print(f"warning: could not read {state_file}: {error}", file=sys.stderr)
    return found


def pull_runs(
    entity: str, project: str, run_ids: list[str], gpus_per_run: int | None
) -> list[dict[str, Any]]:
    import wandb

    api = wandb.Api()
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
        except Exception as error:
            rows.append({"run_id": run_id, "error": str(error)})
            continue
        runtime_s = run.summary.get("_runtime")
        n_gpus = gpus_per_run
        if n_gpus is None:
            # torchrun world size is the best in-band GPU count when present.
            n_gpus = run.config.get("world_size") or run.config.get("num_gpus")
        gpu_hours = (
            runtime_s / 3600.0 * n_gpus
            if runtime_s is not None and n_gpus
            else None
        )
        rows.append(
            {
                "run_id": run_id,
                "name": run.name,
                "state": run.state,
                "runtime_hours": runtime_s / 3600.0 if runtime_s else None,
                "n_gpus": n_gpus,
                "gpu_hours": gpu_hours,
                "final_step": run.summary.get("_step"),
            }
        )
    return rows


def _render_markdown(rows: list[dict[str, Any]], total_gpu_hours: float | None) -> str:
    lines = [
        "# Training compute from W&B (REQ-7)",
        "",
        f"git: {git_sha()}. Values are cached from W&B, not recomputed. "
        "Peak training memory: NOT FOUND (never logged during training).",
        "",
        "| Run | State | wall h | GPUs | GPU-h | final step |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['run_id']} | ERROR: {row['error']} | | | | |")
            continue
        lines.append(
            f"| {row['name']} ({row['run_id']}) | {row['state']} "
            f"| {fmt(row['runtime_hours'], 4)} | {row['n_gpus'] or 'NOT FOUND'} "
            f"| {fmt(row['gpu_hours'], 4)} | {row['final_step']} |"
        )
    lines += [
        "",
        f"**Total GPU-hours across listed runs: {fmt(total_gpu_hours, 5)}** "
        "(sum only complete resume chains — verify no run in the chain is missing).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--state-files", nargs="*", default=[])
    parser.add_argument("--gpus-per-run", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    harvested = harvest_run_ids(args.state_files)
    for run_id, source in harvested.items():
        print(f"harvested run id {run_id} from {Path(source).name}")
    run_ids = list(dict.fromkeys(args.run_ids + list(harvested)))
    if not run_ids:
        parser.error("no run ids (pass --run-ids and/or --state-files)")

    rows = pull_runs(args.entity, args.project, run_ids, args.gpus_per_run)
    gpu_hours = [r["gpu_hours"] for r in rows if r.get("gpu_hours") is not None]
    total = sum(gpu_hours) if gpu_hours else None

    payload = {
        "git_sha": git_sha(),
        "entity": args.entity,
        "project": args.project,
        "harvested_from": harvested,
        "runs": rows,
        "total_gpu_hours": total,
        "peak_train_memory": "NOT FOUND (never logged during training)",
    }
    emit(args.out_dir, "wandb_compute", payload, _render_markdown(rows, total))

    provenance = Provenance(script="rebuttal_eval.wandb_pull")
    for row in rows:
        if row.get("gpu_hours") is not None:
            provenance.record(
                "2.5", "gpu_hours", row["gpu_hours"], mode="cached",
                checkpoint=row["run_id"],
            )
    provenance.write(args.out_dir)
    print(f"total GPU-hours: {fmt(total, 5)} over {len(rows)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
