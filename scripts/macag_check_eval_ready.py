#!/usr/bin/env python3
"""Refuse a final-campaign launch unless the tree and prompt file match the freeze.

Checks: clean git worktree, HEAD equals the required tag (default macag-final-v1),
and the prompt JSON's SHA-256 matches the recorded digest when one is given.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="macag-final-v1")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompts-sha256", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if not args.allow_dirty:
        # Untracked local notes are not part of the run. A modified tracked
        # file is: the job would not match the tag.
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        if dirty:
            print("refusing launch: git worktree is dirty", file=sys.stderr)
            print(dirty, file=sys.stderr)
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
