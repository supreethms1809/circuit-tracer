"""CLI entrypoint for config-driven paper evaluation suites."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from spline_clt.paper.runner import PaperSuiteRunner


def _clean_evaluations(runner: PaperSuiteRunner) -> list[str]:
    """Delete evaluation and MACAG outputs while preserving checkpoints.

    Returns a list of paths that were removed.
    """
    removed: list[str] = []

    # Per-variant/seed: remove evaluation/ and macag/ dirs
    all_pairs = [
        (vname, seed)
        for vname in sorted(runner.config.model_variants)
        for seed in runner.config.seeds
    ]
    for variant_name, seed in all_pairs:
        run_dir = runner._run_dir(variant_name, seed)
        for subdir in ("evaluation", "macag"):
            target = run_dir / subdir
            if target.exists():
                shutil.rmtree(target)
                removed.append(str(target))

    # Suite-level aggregate outputs
    for name in (
        "aggregate_metrics.json",
        "per_example_metrics.jsonl",
        "tables.csv",
        "report.md",
    ):
        p = runner.suite_root / name
        if p.exists():
            p.unlink()
            removed.append(str(p))
    if runner.figures_root.exists():
        shutil.rmtree(runner.figures_root)
        removed.append(str(runner.figures_root))

    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a config-driven paper evaluation suite.")
    parser.add_argument("--suite", required=True, help="Path to a suite JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Expand the job graph without executing it.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and resolve the suite config without executing it.",
    )
    parser.add_argument(
        "--re-evaluate",
        action="store_true",
        help="Delete evaluation/MACAG outputs (preserving checkpoints) before running.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Zero-based worker index for PBS/SLURM array jobs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of parallel workers.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["collect", "train", "evaluate", "macag", "report"],
        default=None,
        help="Restrict execution to these stages (intersects with suite-level stage flags).",
    )
    args = parser.parse_args(argv)

    if args.worker_id >= args.num_workers:
        print(
            f"Error: --worker-id {args.worker_id} is out of range for --num-workers {args.num_workers}",
            flush=True,
        )
        return 1

    # validate/dry-run paths: no sharding needed
    runner = PaperSuiteRunner(
        args.suite,
        worker_id=0 if (args.validate_only or args.dry_run) else args.worker_id,
        num_workers=1 if (args.validate_only or args.dry_run) else args.num_workers,
        stages_override=args.stages,
    )
    if args.validate_only:
        print(json.dumps(runner.validate(), indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(json.dumps(runner.dry_run(), indent=2, sort_keys=True))
        return 0

    if args.re_evaluate:
        removed = _clean_evaluations(runner)
        if removed:
            print(f"Cleaned {len(removed)} evaluation artifacts:", flush=True)
            for p in removed:
                print(f"  - {p}", flush=True)
        else:
            print("No evaluation artifacts to clean.", flush=True)

    print(json.dumps(runner.run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
