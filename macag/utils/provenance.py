"""Code provenance stamped into every MACAG output JSON (A6).

Records the git commit hash plus a dirty-worktree flag so any result file can
be traced back to the exact code that produced it. Never raises: outside a git
checkout (or without git on PATH) the fields are ``None`` rather than failing
a multi-hour run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def code_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return ``{"git_commit": <sha|None>, "git_dirty": <bool|None>}``.

    ``git_dirty`` is True when the worktree has staged/unstaged changes or
    untracked files beyond the run's own outputs. Untracked-file noise from
    result directories is expected on analysis machines, so callers should
    treat ``dirty=True`` as "inspect before citing", not "invalid".
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    commit = _git(["rev-parse", "HEAD"], root)
    if commit is None:
        return {"git_commit": None, "git_dirty": None}
    status = _git(["status", "--porcelain"], root)
    dirty: bool | None = None
    if status is not None:
        dirty = bool(status.strip())
    return {"git_commit": commit, "git_dirty": dirty}
