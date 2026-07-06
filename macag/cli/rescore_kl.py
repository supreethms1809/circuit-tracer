"""CLI: post-hoc KL faithfulness rescoring for saved MACAG runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from macag.kl_rescore import rescore_run_dir, rescore_tree

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score saved MACAG evidence sets under KL divergence (independent of "
            "the logit-gap selection objective). Writes macag_kl_faithfulness.json "
            "per run directory."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path, help="Single prompt output directory.")
    group.add_argument("--root", type=Path, help="Sweep root (e.g. results/macag_acdc).")
    parser.add_argument(
        "--clts",
        default="",
        help="Comma-separated CLT tags to include when using --root (default: all).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing KL JSON.")
    parser.add_argument("--progress", action="store_true", help="Log each rescored directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.progress:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.run_dir is not None:
        path = rescore_run_dir(args.run_dir, force=args.force)
        if path is None:
            LOGGER.error("rescore failed for %s", args.run_dir)
            return 1
        print(f"wrote {path}")
        return 0

    clts = [c.strip() for c in args.clts.split(",") if c.strip()] or None
    paths = rescore_tree(args.root, clt_tags=clts, force=args.force)
    print(f"rescored {len(paths)} run(s) under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
