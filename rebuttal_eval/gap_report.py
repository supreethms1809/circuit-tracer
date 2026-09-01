"""§5 gap report: join the REQ manifest against what actually exists.

Encodes the REQ-1..15 manifest as data, inspects the rebuttal results
directory and the inventory, and emits the §5 table with the four
load-bearing absence flags. Refuses to certify tables if any check_*.json
under the results directory reports status FAIL (§4 blocking protocol).

Usage:
  python -m rebuttal_eval.gap_report --results-dir results/rebuttal \
      --out-dir results/rebuttal/gap_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rebuttal_eval.common import emit, git_sha

#: REQ manifest (spec §1). evidence: glob patterns under --results-dir whose
#: presence marks the requirement as (at least partially) satisfied.
REQ_MANIFEST: list[dict[str, Any]] = [
    {"id": "REQ-1", "priority": "P0", "what": "Converged GPT-2 small, 4 variants x 3 seeds, full metrics",
     "evidence": ["check_reconstruction/check_reconstruction_gpt2s_v3_*.json"]},
    {"id": "REQ-4", "priority": "P0 (decisive)", "what": "Scored auto-interp (detection + fuzzing) with L0",
     "evidence": ["autointerp*/autointerp*.json"]},
    {"id": "REQ-3", "priority": "P0", "what": "Cross-model: Spline AND Linear on GPT-2 large and Qwen",
     "evidence": ["check_reconstruction/check_reconstruction_gpt2l_*.json",
                  "check_reconstruction/check_reconstruction_qwen_*.json"]},
    {"id": "REQ-6", "priority": "P0", "what": "Cosine diagnosis: mechanism, per-position distribution, old vs new",
     "evidence": ["check_reconstruction/compare_*/compare_submitted_vs_converged.json"]},
    {"id": "REQ-7", "priority": "P0", "what": "Compute cost incl. scaling with d_t",
     "evidence": ["attr_scaling*/attr_scaling*.json", "inference_bench*/inference_bench*.json",
                  "wandb*/wandb_compute.json"]},
    {"id": "REQ-5", "priority": "P1", "what": "Fidelity on 600 RAVEL prompts and held-out natural text",
     "evidence": ["natural*/aggregate_metrics.json"]},
    {"id": "REQ-2", "priority": "P1", "what": "Submitted-vs-converged delta table",
     "evidence": ["check_reconstruction/compare_*/compare_submitted_vs_converged.json"]},
    {"id": "REQ-8", "priority": "P1", "what": "float32 necessity at inference",
     "evidence": ["dtype_ablation*/dtype_ablation*.json"]},
    {"id": "REQ-11", "priority": "P1", "what": "Parameter-matching pairing stated explicitly",
     "evidence": ["check_params/check_params.json"]},
    {"id": "REQ-9", "priority": "P2", "what": "Published-reference anchor table",
     "evidence": ["anchors/*.json"]},
    {"id": "REQ-10", "priority": "P2", "what": "Top-k fidelity k in {1,5,10}",
     "evidence": ["topk*/*.json"]},
    {"id": "REQ-12", "priority": "P3", "what": "Jacobian vs activation-patching correlation", "evidence": []},
    {"id": "REQ-13", "priority": "P3", "what": "W_in spectrum analysis", "evidence": []},
    {"id": "REQ-14", "priority": "P3", "what": "Spline hyperparameter sensitivity", "evidence": []},
    {"id": "REQ-15", "priority": "P3", "what": "MLP-encoder ablation at matched params", "evidence": []},
]

#: Separately load-bearing absences (spec §5).
LOAD_BEARING = {
    "REQ-3": "Linear baselines on GPT-2 large / Qwen — without them the scale claim is one-sided",
    "REQ-4": "Scored (not merely generated) auto-interp — the only lever that touches R1",
    "REQ-6": "Trivial/mean baselines (§4.3) — unrequested but potentially decisive",
}


def find_failures(results_dir: Path) -> list[str]:
    failures = []
    for check_file in results_dir.rglob("check_*.json"):
        try:
            payload = json.loads(check_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("status") == "FAIL":
            failures.append(str(check_file))
    return failures


def assess(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for req in REQ_MANIFEST:
        hits: list[str] = []
        for pattern in req["evidence"]:
            hits.extend(str(p) for p in results_dir.glob(pattern))
        if req["priority"].startswith("P3"):
            recommendation = "acknowledge and defer (future work)"
        elif hits:
            recommendation = "present — format for the response"
        else:
            recommendation = "run now" if req["priority"].startswith("P0") else "run if window allows"
        rows.append(
            {
                "id": req["id"],
                "priority": req["priority"],
                "what": req["what"],
                "evidence_found": hits,
                "missing": not hits and not req["priority"].startswith("P3"),
                "load_bearing": LOAD_BEARING.get(req["id"], ""),
                "recommendation": recommendation,
            }
        )
    return rows


def _render_markdown(rows: list[dict[str, Any]], failures: list[str]) -> str:
    lines = [
        "# Gap report (spec §5)",
        "",
        f"git: {git_sha()}",
        "",
    ]
    if failures:
        lines += [
            "## BLOCKING: §4 check failures — do not format rebuttal tables",
            "",
            *[f"- `{f}`" for f in failures],
            "",
        ]
    lines += [
        "| REQ | Priority | What | Status | Recommendation |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        status = (
            f"{len(row['evidence_found'])} artifact(s)"
            if row["evidence_found"]
            else "**MISSING**" if row["missing"] else "deferred"
        )
        lines.append(
            f"| {row['id']} | {row['priority']} | {row['what']} "
            f"| {status} | {row['recommendation']} |"
        )
    load_bearing_missing = [r for r in rows if r["missing"] and r["load_bearing"]]
    if load_bearing_missing:
        lines += ["", "## Load-bearing absences", ""]
        lines += [f"- **{r['id']}**: {r['load_bearing']}" for r in load_bearing_missing]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/rebuttal")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    failures = find_failures(results_dir)
    rows = assess(results_dir)
    payload = {
        "git_sha": git_sha(),
        "results_dir": str(results_dir),
        "blocking_failures": failures,
        "requirements": rows,
    }
    emit(args.out_dir, "gap_report", payload, _render_markdown(rows, failures))
    n_missing = sum(r["missing"] for r in rows)
    print(
        f"gap report: {n_missing} missing requirement(s), "
        f"{len(failures)} blocking failure(s) -> {args.out_dir}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
