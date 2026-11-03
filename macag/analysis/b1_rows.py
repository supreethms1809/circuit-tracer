"""B1 long-form rows: one row per (CLT, prompt, leg, method) at Game 1 |E*|."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from macag.utils.final_protocol import (
    acdc_target_for_leg,
    leg_evidence_size,
    leg_is_degenerate,
    matched_k_read,
    primary_leg,
)

LEGS = ("frozen", "unfrozen")
BASELINE_METHODS = ("random", "influence", "eap_syed", "singleton", "acdc")
METHODS = ("game1",) + BASELINE_METHODS

ROW_FIELDS = [
    "clt",
    "task",
    "prompt_id",
    "slug",
    "leg",
    "method",
    "is_primary_leg",
    "primary_leg_rule",
    "leg_degenerate",
    "eval_k",
    "set_size",
    "capped_f",
    "capped_s",
    "capped_n",
    "recoverable_range",
    "n_over_r",
    "keep_distance",
    "remove_below_empty",
    "kl_faithfulness",
    "matched_k_status",
    "kl_matched_k_status",
    "kl_sidecar_matched_k",
    "k_grid_continuous",
    "check_kl_k_eq_game1",
    "check_notes",
    "code_version",
]


def _task_of(slug: str) -> str:
    if "_arc_easy_" in slug:
        return "arc_easy"
    if "_ioi_" in slug:
        return "ioi"
    if "_mcqa_" in slug:
        return "mcqa"
    return "other"


def _num(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, (int, float)):
        return repr(float(x)) if isinstance(x, float) else str(x)
    return str(x)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _score_fields(scores: Mapping[str, Any] | None) -> dict[str, Any]:
    if not scores:
        return {
            "capped_f": None,
            "capped_s": None,
            "capped_n": None,
            "recoverable_range": None,
            "n_over_r": None,
            "keep_distance": None,
            "remove_below_empty": None,
        }
    r = scores.get("recoverable_range")
    n = scores.get("necessity")
    n_over_r: float | None = None
    if isinstance(r, (int, float)) and r > 0 and isinstance(n, (int, float)):
        n_over_r = float(n) / float(r)
    return {
        "capped_f": scores.get("faithfulness"),
        "capped_s": scores.get("sufficiency"),
        "capped_n": n,
        "recoverable_range": r,
        "n_over_r": n_over_r,
        "keep_distance": scores.get("keep_distance"),
        "remove_below_empty": scores.get("remove_below_empty"),
    }


def k_grid_continuous(results: Mapping[Any, Any] | None, budget: int) -> tuple[bool, str]:
    """Prefix curves should include every integer k in [0, budget]."""
    if not results:
        return False, "no_results"
    ks = {int(k) for k in results if str(k).lstrip("-").isdigit()}
    for k in range(0, budget + 1):
        if k not in ks:
            return False, f"missing_k_{k}"
    return True, "ok"


def _kl_game1_leg(kl_payload: Mapping[str, Any] | None, leg: str) -> Any:
    if not kl_payload:
        return None
    block = (kl_payload.get("game1") or {}).get(leg)
    if not isinstance(block, Mapping):
        return None
    kd = block.get("kl_divergence") or block
    if isinstance(kd, Mapping):
        return kd.get("faithfulness")
    return None


def _kl_from_sidecar(
    kl_payload: Mapping[str, Any] | None, leg: str, method: str
) -> tuple[Any, Any, Any]:
    if not kl_payload:
        return None, None, None
    baselines = kl_payload.get("baselines") or {}
    if baselines.get("per_leg"):
        leg_block = baselines.get(leg) or {}
        sidecar_k = leg_block.get("matched_k")
        sidecar_status = leg_block.get("matched_k_status")
        ent = (leg_block.get("methods") or {}).get(method)
        if not isinstance(ent, Mapping):
            return None, sidecar_k, sidecar_status
        kd = ent.get("kl_divergence") or ent.get("kl_faithfulness") or ent
        if isinstance(kd, Mapping):
            return kd.get("faithfulness"), sidecar_k, sidecar_status
        return None, sidecar_k, sidecar_status
    ent = (baselines.get("methods") or {}).get(method)
    if isinstance(ent, Mapping):
        kd = ent.get("kl_divergence") or ent.get("kl_faithfulness") or ent
        if isinstance(kd, Mapping):
            return kd.get("faithfulness"), baselines.get("matched_k"), baselines.get("matched_k_status")
    return None, None, None


def _kl_from_baselines_leg(leg_block: Mapping[str, Any] | None) -> tuple[Any, Any]:
    if not leg_block:
        return None, None
    kl = leg_block.get("kl_faithfulness")
    if isinstance(kl, Mapping):
        return kl.get("faithfulness"), leg_block.get("kl_faithfulness_matched_k_status")
    return None, leg_block.get("kl_faithfulness_matched_k_status")


def _scores_game1_leg(game1: Mapping[str, Any], leg: str) -> dict[str, Any]:
    block = game1.get(leg) if game1.get("freeze_mode") == "both" else game1
    if not isinstance(block, Mapping):
        return {}
    return dict(block.get("scores") or {})


def _scores_baseline_at_k(
    method: str,
    entry: Mapping[str, Any],
    leg: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, int | None, str, Mapping[Any, Any] | None]:
    """Logit-gap scores at Game 1 |E*| for one baseline method and leg."""
    legs = entry.get("legs") or {}
    leg_block = legs.get(leg) if isinstance(legs, Mapping) else None
    if not isinstance(leg_block, Mapping):
        leg_block = {}

    status = str(spec.get("status", "no_game1"))
    target_k = spec.get("target_k")
    if status != "ok" or target_k is None:
        return None, None, status, leg_block.get("results")

    k = int(target_k)

    if method == "acdc":
        mk = leg_block.get("matched_k") or {}
        mk_status = str(mk.get("status") or status)
        if mk_status == "ok" and isinstance(mk.get("scores"), Mapping):
            achieved = mk.get("achieved_k")
            size = achieved if achieved is not None else mk.get("target_k")
            return dict(mk["scores"]), int(size) if size is not None else k, mk_status, None
        bbs = leg_block.get("best_by_size") or {}
        hit = bbs.get(str(k)) or bbs.get(k)
        if isinstance(hit, Mapping) and isinstance(hit.get("scores"), Mapping):
            return dict(hit["scores"]), k, "ok", None
        return None, k, mk_status if mk_status != "ok" else "unavailable_at_k", None

    results = leg_block.get("results") or entry.get("results") or {}
    read = matched_k_read(results, spec)
    st = str(read.get("status", "no_game1"))
    sc = read.get("scores")
    if isinstance(sc, Mapping):
        ev = read.get("evidence") or []
        size = len(ev) if ev is not None else k
        return dict(sc), int(size), st, results
    return None, k, st, results


def extract_run_dir(
    run_dir: Path,
    *,
    clt: str,
    slug: str,
    score_subdir: str = "logit_gap",
) -> list[dict[str, Any]]:
    """Read one prompt cell and emit B1 rows."""
    kind = run_dir / score_subdir if (run_dir / score_subdir).is_dir() else run_dir
    g1 = _load_json(kind / "macag_game1.json")
    bl = _load_json(kind / "macag_baselines.json")
    kl = _load_json(kind / "macag_kl_faithfulness.json")
    if g1 is None and bl is None:
        return []

    budget = int((bl or {}).get("params", {}).get("budget", 128))
    task = _task_of(slug)
    prompt_id = str((g1 or bl or {}).get("input_id") or slug)
    raw_cv = (g1 or bl or {}).get("code_version", "")
    if isinstance(raw_cv, Mapping):
        code_version = str(raw_cv.get("git_commit") or raw_cv)
    else:
        code_version = str(raw_cv or "")

    pl_name, pl_rule = primary_leg(g1)
    rows: list[dict[str, Any]] = []

    for leg in LEGS:
        if g1 and g1.get("freeze_mode") == "both" and leg not in g1:
            continue
        spec = acdc_target_for_leg(g1, leg, budget, None)
        leg_degen = leg_is_degenerate(g1, leg) if g1 else False
        eval_k = leg_evidence_size(g1, leg) if g1 else None
        if spec.get("status") == "ok" and spec.get("target_k") is not None:
            eval_k = int(spec["target_k"])

        for method in METHODS:
            notes: list[str] = []
            kl_sidecar_k: Any = None
            kl_sidecar_st: Any = None
            if method == "game1":
                if g1 is None:
                    continue
                scores = _scores_game1_leg(g1, leg)
                mk_status = "degenerate_leg" if leg_degen else "ok"
                if leg_degen:
                    scores = {}
                set_size = eval_k if eval_k is not None else leg_evidence_size(g1, leg)
                leg_g1 = (g1.get(leg) or {}) if g1.get("freeze_mode") == "both" else g1
                kl_f = None
                if isinstance(leg_g1, Mapping):
                    klb = leg_g1.get("kl_faithfulness")
                    if isinstance(klb, Mapping):
                        kl_f = klb.get("faithfulness")
                if kl_f is None:
                    kl_f = _kl_game1_leg(kl, leg)
                kl_mk_st = "ok" if kl_f is not None and not leg_degen else mk_status
                k_cont, k_note = False, "n/a_game1"
            else:
                if bl is None:
                    continue
                entry = (bl.get("methods") or {}).get(method)
                if not isinstance(entry, Mapping):
                    continue
                scores, set_size, mk_status, results_for_grid = _scores_baseline_at_k(
                    method, entry, leg, spec
                )
                legs = entry.get("legs") or {}
                leg_block = legs.get(leg) if isinstance(legs, Mapping) else None
                kl_f, kl_mk_st = _kl_from_baselines_leg(leg_block if isinstance(leg_block, Mapping) else None)
                kl_sidecar_f, kl_sidecar_k, kl_sidecar_st = _kl_from_sidecar(kl, leg, method)
                if kl_f is None and kl_sidecar_f is not None:
                    kl_f = kl_sidecar_f
                if method in ("random", "influence", "eap_syed", "singleton"):
                    k_cont, k_note = k_grid_continuous(results_for_grid, budget)
                else:
                    k_cont, k_note = True, "n/a_acdc"

            if method != "game1" and scores is None and mk_status not in ("ok",):
                notes.append(f"no_scores:{mk_status}")

            check_kl = ""
            if method != "game1":
                check_kl = False
                if eval_k is not None and kl_sidecar_k is not None:
                    check_kl = int(kl_sidecar_k) == int(eval_k)
                elif eval_k is not None and kl_f is not None and str(kl_mk_st or "") == "ok":
                    check_kl = True
                if (
                    eval_k is not None
                    and kl_f is not None
                    and kl_sidecar_k is not None
                    and not check_kl
                ):
                    notes.append(f"kl_k_mismatch:sidecar={kl_sidecar_k},eval_k={eval_k}")

            sf = _score_fields(scores)
            rows.append(
                {
                    "clt": clt,
                    "task": task,
                    "prompt_id": prompt_id,
                    "slug": slug,
                    "leg": leg,
                    "method": method,
                    "is_primary_leg": leg == pl_name,
                    "primary_leg_rule": pl_rule,
                    "leg_degenerate": leg_degen,
                    "eval_k": eval_k,
                    "set_size": set_size,
                    **sf,
                    "kl_faithfulness": kl_f,
                    "matched_k_status": mk_status if method != "game1" else ("degenerate_leg" if leg_degen else "ok"),
                    "kl_matched_k_status": kl_mk_st or kl_sidecar_st,
                    "kl_sidecar_matched_k": kl_sidecar_k if method != "game1" else None,
                    "k_grid_continuous": k_cont if method != "game1" else "",
                    "check_kl_k_eq_game1": check_kl,
                    "check_notes": ";".join(notes) if notes else (k_note if method != "game1" and not k_cont else ""),
                    "code_version": code_version,
                }
            )
    return rows


def iter_campaign_rows(
    campaign_root: Path,
    *,
    score_subdir: str = "logit_gap",
    clt_tags: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clt_dir in sorted(campaign_root.iterdir()):
        if not clt_dir.is_dir() or not clt_dir.name.startswith(("gemma2", "llama")):
            continue
        if clt_tags and clt_dir.name not in clt_tags:
            continue
        for slug_dir in sorted(clt_dir.iterdir()):
            if not slug_dir.is_dir() or not slug_dir.name.startswith("mib_"):
                continue
            rows.extend(
                extract_run_dir(
                    slug_dir,
                    clt=clt_dir.name,
                    slug=slug_dir.name,
                    score_subdir=score_subdir,
                )
            )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _num(row.get(k)) for k in ROW_FIELDS})
