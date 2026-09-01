"""Shared infrastructure for rebuttal evaluation scripts.

Provides checkpoint/dataset loading wrappers, provenance recording, path
scrubbing for OpenReview-bound text, and the dual JSON+markdown output
convention used by every script in this package.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from spline_clt.kan_transcoder import KANCrossLayerTranscoder, load_spline_clt
from spline_clt.training.data import ActivationDataset

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Ordered (pattern, replacement) pairs applied to any text destined for
#: OpenReview. Longest / most specific first so the generic username rule
#: does not pre-empt the path rules.
_SCRUB_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/gscratch/[A-Za-z0-9_.-]+"), "$RESULTS"),
    (re.compile(r"/cluster/[A-Za-z0-9_.-]+/home/[A-Za-z0-9_.-]+"), "$HOME"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+"), "$HOME"),
    (re.compile(r"\bai4wy-?\d*\b"), "nodeX"),
    (re.compile(r"\bssuresh\b"), "$USER"),
    (re.compile(r"\buwyo-\d+\b"), "$ACCOUNT"),
]


def scrub(text: str) -> str:
    """Remove identifying paths, usernames, and hostnames from text."""
    for pattern, replacement in _SCRUB_RULES:
        text = pattern.sub(replacement, text)
    return text


def git_sha() -> str:
    """Current repo commit (short), or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _skip_kan_spline_init() -> None:
    """Zero-init KANLinear splines at construction instead of the lstsq fit.

    KANLinear.reset_parameters runs curve2coeff (an lstsq) per layer at
    construction; every one of those weights is immediately overwritten by
    the checkpoint load, so the fit is pure startup cost (minutes at
    d_t=12288). Same pattern as scripts/rescore_clean_val.py.
    """
    import math

    import efficient_kan

    if getattr(efficient_kan.KANLinear.reset_parameters, "_rebuttal_fast", False):
        return

    def _fast_reset(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            self.spline_weight.data.zero_()
            if getattr(self, "enable_standalone_scale_spline", False):
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    _fast_reset._rebuttal_fast = True  # type: ignore[attr-defined]
    efficient_kan.KANLinear.reset_parameters = _fast_reset


def load_transcoder(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> KANCrossLayerTranscoder:
    """Load a Spline-CLT, local linear CLT, or published hub CLT (read-only).

    Auto-detects format the same way as ``macag.factories.replacement_model``:
    ``metadata.safetensors`` → ``load_spline_clt``; otherwise → ``load_clt``
    (covers ``mntss/*`` hub snapshots used as the linear Gemma/Llama anchors).
    """
    _skip_kan_spline_init()
    path = Path(checkpoint_path)
    device_t = torch.device(device)
    if (path / "metadata.safetensors").exists():
        model = load_spline_clt(str(path), device=device_t, dtype=dtype)
    else:
        from circuit_tracer.transcoder.cross_layer_transcoder import load_clt

        model = load_clt(
            str(path),
            device=device_t,
            dtype=dtype,
            lazy_decoder=True,
            lazy_encoder=False,
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model  # type: ignore[return-value]


def load_val_dataset(
    activation_dir: str | Path,
    split: str | None = "val",
) -> ActivationDataset:
    """Open the validation activation mmap in streaming mode (O(batch) RAM)."""
    try:
        return ActivationDataset.load(str(activation_dir), split=split)
    except FileNotFoundError:
        if split is None:
            raise
        # Legacy single-file layout without split suffixes.
        return ActivationDataset.load(str(activation_dir), split=None)


def sample_indices(n_total: int, n_samples: int, seed: int) -> list[int]:
    """Deterministic sample subset: fixed-seed permutation prefix."""
    generator = torch.Generator().manual_seed(seed)
    n = min(n_samples, n_total)
    return torch.randperm(n_total, generator=generator)[:n].tolist()


@dataclass
class Provenance:
    """Accumulates one row per reported number; written as provenance.csv.

    Satisfies paper_rebuttal_todo.md §0.3.2/§2.7: every number that can end
    up in a rebuttal table is traceable to a checkpoint, script, and seed,
    and marked cached vs recomputed. This file is internal — never pasted
    to OpenReview — so paths are NOT scrubbed here.
    """

    script: str
    checkpoint: str = ""
    run_id: str = ""
    seed: int | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        table: str,
        cell: str,
        value: Any,
        mode: str = "recomputed",
        checkpoint: str | None = None,
    ) -> None:
        self.rows.append(
            {
                "table": table,
                "cell": cell,
                "value": value,
                "run_id": self.run_id,
                "checkpoint": checkpoint if checkpoint is not None else self.checkpoint,
                "script": self.script,
                "git_sha": git_sha(),
                "seed": self.seed,
                "mode": mode,
            }
        )

    def write(self, out_dir: str | Path) -> Path:
        out_path = Path(out_dir) / "provenance.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "table", "cell", "value", "run_id", "checkpoint",
            "script", "git_sha", "seed", "mode",
        ]
        with open(out_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return out_path


def emit(
    out_dir: str | Path,
    name: str,
    payload: dict[str, Any],
    markdown: str,
    scrub_markdown: bool = True,
) -> tuple[Path, Path]:
    """Write `<name>.json` (raw) and `<name>.md` (scrubbed) side by side.

    JSON keeps real paths for internal use; markdown is the OpenReview-safe
    rendering.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{name}.json"
    md_path = out / f"{name}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    md_path.write_text(scrub(markdown) if scrub_markdown else markdown)
    return json_path, md_path


def fmt(value: float | None, digits: int = 4) -> str:
    """Uniform numeric formatting for markdown tables ('NOT FOUND' for None)."""
    if value is None:
        return "NOT FOUND"
    return f"{value:.{digits}g}"
