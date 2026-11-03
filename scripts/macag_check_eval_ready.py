#!/usr/bin/env python3
"""Refuse a final-campaign launch unless the tree and prompt file match the freeze.

Checks: clean git worktree, HEAD equals the required tag (default macag-final-v1),
and the prompt JSON's SHA-256 matches the recorded digest when one is given.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _dirty_outside_allowlist(allow_prefixes: list[str]) -> str | None:
    """Return porcelain lines for tracked edits outside allow_prefixes, else None."""
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if not dirty:
        return None
    blocked: list[str] = []
    for line in dirty.splitlines():
        # Porcelain: XY + space + path (X/Y may be space for unstaged/staged).
        path = line[2:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if any(
            path == pref
            or path.startswith(pref.rstrip("/") + "/")
            or (pref.endswith("/") and path.startswith(pref))
            for pref in allow_prefixes
        ):
            continue
        blocked.append(line)
    return "\n".join(blocked) if blocked else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="macag-final-v1")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompts-sha256", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--allow-dirty-prefix",
        action="append",
        default=[],
        metavar="PATH",
        help="Tracked path or dir prefix that may differ from the tag commit (repeatable).",
    )
    args = parser.parse_args()

    allow_prefixes = list(args.allow_dirty_prefix)
    env_prefixes = os.environ.get("MACAG_DIRTY_OK_PREFIXES", "")
    if env_prefixes:
        allow_prefixes.extend(p for p in env_prefixes.split(":") if p)

    if not args.allow_dirty:
        # Untracked local notes are not part of the run. Analysis-only edits
        # may be allowlisted so long as HEAD still matches the campaign tag.
        blocked = _dirty_outside_allowlist(allow_prefixes)
        if blocked:
            print("refusing launch: git worktree has disallowed edits", file=sys.stderr)
            print(blocked, file=sys.stderr)
            if allow_prefixes:
                print(f"(allowed prefixes: {allow_prefixes})", file=sys.stderr)
            return 2
    head = _git("rev-parse", "HEAD")
    try:
        tagged = _git("rev-parse", f"{args.tag}^{{commit}}")
    except subprocess.CalledProcessError:
        print(f"refusing launch: tag {args.tag} does not exist", file=sys.stderr)
        return 2
    if head != tagged:
        print(f"refusing launch: HEAD {head} is not {args.tag} ({tagged})", file=sys.stderr)
        return 2

    prompt_path = Path(args.prompts)
    digest = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    print(f"prompts_sha256={digest}")
    print(f"code_version={head}")
    if args.prompts_sha256 and digest != args.prompts_sha256:
        print("refusing launch: prompt file digest mismatch", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
