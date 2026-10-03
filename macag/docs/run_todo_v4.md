# MACAG Result Collection — TMLR-250 Pass A/B/C (**v4**)

Updated 2026-08-13. Aligns with [`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md).  
**Shapley is deferred** (add later with `GOLD_PER_TASK` / shapley pass).

Dallas–Austin was a one-prompt smoke test. **MIB is the paper campaign.**

| Knob | v4 value |
| --- | --- |
| Prompts | MIB IOI 100 + MCQA 50 + ARC-Easy 50 (+ InterpBench separate) |
| Graph export | **`node_threshold=1.0`, `edge_threshold=1.0`**, large `max_feature_nodes` |
| Prefilter / connected | **off** |
| Selection | **`logit_gap`** |
| KL | **post-hoc rescore** on saved sets |
| Game 1 | dual-freeze, **unbudgeted** (natural `|E|`) |
| Game 2 | **abr + fp** |
| Pass A baselines | **`influence` only** (no duplicate `game1`, no `eap_graph`, no Shapley) |
| Ranking F-vs-k | `REPORT_BUDGET=128` (matched-k vs Game1 `|E*|` at analysis) |
| Seeds | **0** first |
| Fleet | 4 nodes / 8 GPUs per CLT (array `0-3`), requeue OK |
| Outdirs | `macag_mib_tmlr250v4_{h200,gemma25m,llama}` (fresh; do not reuse v3) |

## Method map (Dallas suite → MIB passes)

| Method | Pass | Notes |
| --- | --- | --- |
| Attribute graph | A | Unpruned export |
| Game 1 dual-freeze | A | Unbudgeted |
| Game 2 ABR+FP | A | Same oracle |
| influence | A | Top-k curves to 128; match Game1 size in tables |
| KL rescore | A (end) / analysis | Not a selector |
| eap_syed | **B** | Feature AtP; needs corrupt from MIB JSON |
| acdc (ported) | **B** | τ grid; no hard target-k |
| eap_edge | **C** | True AutoCircuit; Track B |
| acdc_edge | **C** | True AutoCircuit; Track B |
| Shapley | — | **Out for now** |

## 1. Pass A — launch

```bash
scripts/slurm/submit_all_tmlr250_v4.sh
# or without cancelling existing v4 jobs:
scripts/slurm/submit_all_tmlr250_v4.sh --no-cancel
```

Per CLT:

```bash
sbatch scripts/slurm/submit_macag_mib_h200_v4.sh
sbatch scripts/slurm/submit_macag_mib_gemma25m_v4.sh
sbatch scripts/slurm/submit_macag_mib_llama_v4.sh
```

Aggregate after arrays finish:

```bash
RESULTS_ROOT=/gscratch/$USER/macag_mib_tmlr250v4_h200 \
  RUN_ANALYSIS=1 SEEDS=0 scripts/run_mib_benchmark.sh
```

## 2. Pass B — eap_syed + ported ACDC

```bash
scripts/slurm/submit_all_tmlr250_v4_pass_b.sh
```

Uses `REPORT_BUDGET=128` for curve length; **do not** interpret ACDC as budget-8 matched unless you set `ACDC_TARGET_K` explicitly.

## 3. Pass C — original edge EAP/ACDC (Track B)

```bash
scripts/slurm/submit_all_tmlr250_v4_pass_c.sh
```

Writes `macag_original_baselines.json` (never merge into feature `macag_baselines.json`).  
Report F/S/N and KL for edge circuits; **no Jaccard vs Game 1**.

Optional follow-up: edge logit-gap S/N/F scorer on saved circuits (same as Dallas `EDGE_SNF`).

## 4. Later — Shapley

When ready:

```bash
GOLD_PER_TASK=50 CLTS=gemma2-426k \
  scripts/run_macag_shapley_pass.sh /gscratch/$USER/macag_mib_tmlr250v4_h200/macag_mib_seed0
```

(or the existing shapley SLURM submitters pointed at `tmlr250v4_*` roots).

## 5. Later — Game 1 / Game 2 ablations (after full Pass A/B/C)

**v4 Pass A ships one Game 1 config + two Game 2 solvers only** (paper protocol).  
Do **not** expand the mass campaign mid-pass. After A/B/C finish, pick a subset of:

| Ablation | Why (later) | Notes |
| --- | --- | --- |
| `α=0` Game 1 | Unlearning / pure necessity ([`APPLICATIONS_METRICS.md`](APPLICATIONS_METRICS.md)) | Same graphs; new Game1 outdir or kind suffix |
| `α=1` Game 2 | Steering / pure sufficiency | Same |
| `score_kind` ∈ {`logit`, `prob`, `kl_divergence`} | Foil-free unlearning; optional auditor KL-selection | Keep logit_gap as primary tables |
| Game 2 frozen **and** unfrozen | Freeze sensitivity (case studies historically) | MIB mass = single freeze by J.B8 |
| Pruned graph (`node/edge` 0.8/0.98) | Compare to unpruned; separate outdir tree | Never mix with `tmlr250v4_*` |
| Ablation `mean` / resample | Off-manifold / intervention sensitivity | Track A identity must stay labeled |
| `λ` / `β` / `ε` sweeps | Robustness appendix | Subsample prompts |
| Budgeted Game 1 product stops | App claims vs matched-k baselines | Baselines already use `REPORT_BUDGET` |

Shapley (§4) and these ablations are both **post–full-pass** work; prioritize Shapley-gold if tables need agreement metrics first.

## Fairness reminders

- Matched-k tables: Game1 natural `|E|` vs influence / eap_syed prefixes.  
- ACDC: F vs `|E|` by τ (density tradeoff).  
- Track B: separate tables; claim “vs native-edge ACDC/EAP on F/S/N/KL,” not “beats Conmy on features.”  
- Full checklist: `TMLR_EVAL_RECIPE.md` §7.

## Outputs

```
/gscratch/$USER/macag_mib_tmlr250v4_<clt>/macag_mib_seed0/<clt_tag>/<slug>/
  graphs/<slug>.json
  logit_gap/
    macag_game1.json
    macag_game2_{abr,fp}.json
    macag_baselines.json          # influence; + eap_syed/acdc after Pass B
    macag_kl_faithfulness.json
    macag_original_baselines.json # after Pass C
    oracle_kwargs.json            # includes corrupted_prompt when present
```
