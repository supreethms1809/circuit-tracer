#!/usr/bin/env python3
"""Print training status for the current user's running Slurm jobs.

Queries ``squeue``, locates each job's stderr log under ``logs/slurm/``, and
parses the latest paper-eval ``Training:`` progress line (step, act_lp, rel_fro).

Usage:
  python scripts/slurm/job_training_status.py
  python scripts/slurm/job_training_status.py --user ssuresh
  python scripts/slurm/job_training_status.py --all-states   # include PD/CG etc.
  python scripts/slurm/job_training_status.py --job 10501 10454
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "slurm"

# tqdm + custom postfix from spline_clt/training/train.py.
# lr is optional and may be truncated mid-refresh (e.g. "2.79e" before "-05").
TRAINING_RE = re.compile(
    r"Training:.*?\|\s*(?P<step>\d+)/(?P<total>\d+).*?"
    r"rel_fro=(?P<rel>[0-9.]+),\s*"
    r"l0_tok=(?P<l0>[0-9.]+),\s*"
    r"act_lp=(?P<act>[0-9.]+)"
    r"(?:,\s*lr=(?P<lr>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))?"
)
SUITE_RE = re.compile(r"Suite name:\s+(\S+)")
SUITE_ALT_RE = re.compile(r"\[paper-eval\] Suite:\s+(\S+)")
COLLECT_RE = re.compile(r"Collecting activations:.*?\|\s*(?P<step>\d+)/(?P<total>\d+)")


@dataclass
class JobRow:
    job_id: str
    name: str
    state: str
    elapsed: str
    reason: str
    suite: str = ""
    stage: str = ""
    step: int | None = None
    total: int | None = None
    act: float | None = None
    rel: float | None = None
    l0: float | None = None
    lr: float | None = None
    log_path: Path | None = None
    note: str = ""


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{exc}") from exc


def list_jobs(
    user: str,
    job_ids: list[str] | None,
    *,
    include_pending: bool = False,
    all_states: bool = False,
) -> list[JobRow]:
    fmt = "%i|%j|%T|%M|%R"
    if job_ids:
        out = _run(["squeue", "-h", "-o", fmt, "-j", ",".join(job_ids)])
    else:
        out = _run(["squeue", "-h", "-u", user, "-o", fmt])

    if all_states:
        allow = None
    elif include_pending or job_ids:
        allow = {"RUNNING", "COMPLETING", "PENDING"}
    else:
        allow = {"RUNNING", "COMPLETING"}

    rows: list[JobRow] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        jid, name, state, elapsed, reason = parts
        if allow is not None and state not in allow:
            continue
        rows.append(
            JobRow(
                job_id=jid,
                name=name,
                state=state,
                elapsed=elapsed,
                reason=reason,
            )
        )
    return rows


def find_log(job_id: str, log_dir: Path) -> Path | None:
    """Prefer stderr logs; try common name prefixes used in this repo."""
    patterns = [
        f"r5_{job_id}.err",
        f"r2_{job_id}.err",
        f"paper_multinode_{job_id}.err",
        f"*_{job_id}.err",
        f"*{job_id}.err",
    ]
    for pat in patterns:
        hits = sorted(log_dir.glob(pat))
        if hits:
            # Prefer names that look like paper training launchers
            preferred = [h for h in hits if h.name.startswith(("r5_", "r2_", "paper_"))]
            return (preferred or hits)[0]
    return None


def _read_text(path: Path, max_bytes: int = 32_000_000) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.replace(b"\r", b"\n").decode("utf-8", errors="ignore")


def parse_status(job: JobRow, log_dir: Path) -> JobRow:
    log = find_log(job.job_id, log_dir)
    job.log_path = log
    if log is None:
        job.stage = "no-log"
        job.note = f"no *.err under {log_dir} for job {job.job_id}"
        return job

    # Suite often printed in .out as well
    suite = ""
    for side in (log, log.with_suffix(".out")):
        if not side.exists():
            continue
        head = _read_text(side)[:200_000]
        m = SUITE_RE.search(head) or SUITE_ALT_RE.search(head)
        if m:
            suite = m.group(1)
            break
    job.suite = suite

    text = _read_text(log)
    tail = text[-12_000:]

    if re.search(r"Traceback \(most recent call last\)", tail) or re.search(
        r"CUDA out of memory|SIGBUS|SIGKILL|NCCL error", tail, re.I
    ):
        # Still try to report last training point if present
        job.stage = "error"
        job.note = "error signature in log tail"

    last = None
    for m in TRAINING_RE.finditer(text):
        last = m
    if last is not None:
        job.stage = "train" if job.stage != "error" else "train+error"
        job.step = int(last.group("step"))
        job.total = int(last.group("total"))
        job.rel = float(last.group("rel"))
        job.l0 = float(last.group("l0"))
        job.act = float(last.group("act"))
        lr_s = last.group("lr")
        if lr_s:
            try:
                job.lr = float(lr_s)
            except ValueError:
                job.lr = None
        return job

    # Collection / startup fallbacks
    last_c = None
    for m in COLLECT_RE.finditer(text):
        last_c = m
    if last_c is not None:
        job.stage = "collect" if job.stage != "error" else "collect+error"
        job.step = int(last_c.group("step"))
        job.total = int(last_c.group("total"))
        return job

    if job.stage != "error":
        if "Calibrating" in tail or "JumpReLU" in tail:
            job.stage = "init"
        elif job.state == "PENDING":
            job.stage = "pending"
        else:
            job.stage = "starting"
    return job


def format_table(rows: list[JobRow]) -> str:
    headers = (
        "JOBID",
        "NAME",
        "STATE",
        "TIME",
        "SUITE",
        "STAGE",
        "PROG",
        "ACT",
        "REL",
        "L0",
    )
    table: list[list[str]] = []
    for r in rows:
        suite = r.suite or "—"
        # shorten suite for display
        if suite.startswith("paper_"):
            suite = suite[len("paper_") :]
        if len(suite) > 36:
            suite = suite[:33] + "..."
        if r.step is not None and r.total:
            pct = 100.0 * r.step / r.total
            prog = f"{r.step}/{r.total} ({pct:.1f}%)"
        elif r.step is not None and r.total is not None:
            prog = f"{r.step}/{r.total}"
        else:
            prog = "—"
        table.append(
            [
                r.job_id,
                r.name,
                r.state,
                r.elapsed,
                suite,
                r.stage,
                prog,
                f"{r.act:.1f}" if r.act is not None else "—",
                f"{r.rel:.3f}" if r.rel is not None else "—",
                f"{r.l0:.0f}" if r.l0 is not None else "—",
            ]
        )

    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cols: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    lines = [fmt_row(list(headers)), fmt_row(["-" * w for w in widths])]
    for row in table:
        lines.append(fmt_row(row))

    notes = [r for r in rows if r.note]
    if notes:
        lines.append("")
        lines.append("notes:")
        for r in notes:
            log = r.log_path.name if r.log_path else "?"
            lines.append(f"  {r.job_id} ({log}): {r.note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user",
        default=os.environ.get("USER", ""),
        help="Slurm user (default: $USER)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory with Slurm .err logs (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Include PENDING and other non-running states",
    )
    parser.add_argument(
        "--job",
        nargs="+",
        default=None,
        help="Only these job IDs",
    )
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Also show PENDING jobs (without requiring --all-states)",
    )
    args = parser.parse_args(argv)

    if not args.user and not args.job:
        print("error: --user is empty and no --job given", file=sys.stderr)
        return 2

    rows = list_jobs(
        args.user,
        args.job,
        include_pending=args.include_pending,
        all_states=args.all_states,
    )
    if not rows:
        print("No matching Slurm jobs.")
        return 0

    state_rank = {"RUNNING": 0, "COMPLETING": 1, "PENDING": 2}
    rows.sort(key=lambda r: (state_rank.get(r.state, 9), int(r.job_id)))

    for job in rows:
        parse_status(job, args.log_dir)

    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
