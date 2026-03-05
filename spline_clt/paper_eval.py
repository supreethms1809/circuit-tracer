"""CLI entrypoint for config-driven paper evaluation suites."""

from __future__ import annotations

import argparse
import json

from spline_clt.paper.runner import PaperSuiteRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a config-driven paper evaluation suite.")
    parser.add_argument("--suite", required=True, help="Path to a suite JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Expand the job graph without executing it.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and resolve the suite config without executing it.",
    )
    args = parser.parse_args(argv)

    runner = PaperSuiteRunner(args.suite)
    if args.validate_only:
        print(json.dumps(runner.validate(), indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(json.dumps(runner.dry_run(), indent=2, sort_keys=True))
        return 0

    print(json.dumps(runner.run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
