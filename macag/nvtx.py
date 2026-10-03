"""Optional NVTX ranges for Nsight Systems.

Enabled only when ``MACAG_NVTX=1`` (or true/yes). Production Game 1 jobs leave
this unset so the ranges are a no-op. This module lives at the ``macag``
package root so scoring/metrics can import it without a circular import through
``macag.utils``.

Prefers the standalone ``nvtx`` package (installed in the conda env) because
Nsight Systems traces those markers reliably. Falls back to ``torch.cuda.nvtx``
if the package is missing.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Iterator


def nvtx_enabled() -> bool:
    return os.environ.get("MACAG_NVTX", "").strip().lower() in {"1", "true", "yes"}


def _range_hooks() -> tuple[Any, Any] | None:
    try:
        import nvtx as nvtx_pkg

        return nvtx_pkg.push_range, nvtx_pkg.pop_range
    except Exception:
        pass
    try:
        import torch

        return torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
    except Exception:
        return None


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Push an NVTX range when profiling; otherwise a no-op."""
    if not nvtx_enabled():
        yield
        return
    hooks = _range_hooks()
    if hooks is None:
        yield
        return
    push, pop = hooks
    try:
        push(name)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            pop()
        except Exception:
            pass
