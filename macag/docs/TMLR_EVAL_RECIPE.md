# TMLR evaluation recipe (MACAG + baselines)

**Status:** canonical protocol for the TMLR MACAG evaluation campaign.  
**Goal:** fair, reviewer-defensible baseline comparisons. A previous paper was withdrawn over incorrect baselines — this document exists so that does not happen again.

Companion docs (do not contradict them; this recipe operationalizes them):

- [`baseline_originals_and_ports.md`](baseline_originals_and_ports.md) — what each original paper does vs what we port
- [`baseline_method_map.md`](baseline_method_map.md) — stable method IDs / aliases
- [`baseline_original_track.md`](baseline_original_track.md) — true AutoCircuit edge track
- [`macag.md`](macag.md) — hub; Game 1 / Game 2 / freeze protocol live in `macag_framework.md` / `macag_game1.md` / `macag_game2.md`
- [`macag_experiments_v3.md`](macag_experiments_v3.md) — completed v3 MIB numbers (pruned, budget 8)
- [`macag_dallas_austin.md`](macag_dallas_austin.md) — unpruned one-prompt suite that motivated v4
- [`APPLICATIONS_METRICS.md`](APPLICATIONS_METRICS.md) — unlearning / steering / auditor: headline metrics and Game 1 tuning (not the paper-table protocol)

If a launch script or suite disagrees with this recipe, **fix the script**.

---

## 0. Fairness contract (non-negotiable)

### Two tracks — never mix claims

| Track | Universe | Ablation | Shared with Game 1? | Allowed claims |
| --- | --- | --- | --- | --- |
| **A. Feature selector** | CLT feature nodes in the circuit-tracer graph | MACAG oracle (default zero-ablation; four modes) | **Yes** — same candidates + same `v` | Matched-k F, F-vs-k AUC, Jaccard / prec@k vs Game1 or Shapley-gold, oracle cost |
| **B. Native edge pipeline** | AutoCircuit factorized residual edges | Corrupt resample patching | **No** | Size, KL(full‖circuit), logit-gap S/N/F, wall time — **pipeline** comparison only |

**Forbidden (withdrawal-class errors):**

1. Jaccard / prec@k / “agreement with Game 1” between feature nodes and native edges.
2. Saying “we beat Conmy ACDC / Syed EAP” when only the **port** was run — say “ported τ-prune / feature AtP under MACAG `v`.”
3. Saying the port *is* the published algorithm (it keeps the selection *rule*, not the full original pipeline).
4. Comparing methods on **different graphs**, different `score_kind`, different freeze flags, or different candidate universes without labeling it.
5. Matching on **equal KL** as the primary budget (forces huge sets; kills the sparsity claim; not what Game 1 optimizes).
6. Using Neuronpedia prune defaults (`node_threshold=0.8`, `edge_threshold=0.98`) while claiming an “unpruned / full CLT” evaluation.
7. Passing `--budget 0` to Game 1 (stops at `|E|=0`).
8. Running a second `game1` inside `--baseline-methods` and treating it as the primary Game 1.
9. Silent graph reuse when requested prune thresholds differ from metadata.

When in doubt: **if node IDs are not the same type of object under the same oracle, it is not a selector-isolation comparison.**

### What “fair” means on Track A

For every prompt cell:

1. **One** attribution graph JSON (shared).
2. **One** candidate universe = feature nodes present in that graph (default `feature_type == "cross layer transcoder"`).
3. **One** oracle config: same `score_kind`, freeze, ablation, α, λ, corrupt prompt (if any).
4. Every method returns a subset (or ranking) of **those** candidates.
5. Head-to-head numbers use the **same** `FaithfulnessMetrics` under that oracle:
   - `F = α·S + (1−α)·N` (default α=0.5)
   - Primary match: `|E| = |E★_Game1|` (per freeze leg when dual-freeze)
   - Also report F-vs-k curves / AUC and oracle-call cost

KL is a **post-hoc rescore** of saved sets (`macag/kl_rescore.py`), not the main selection objective for the MIB campaign.

---

## 1. Graph construction

```text
node_threshold = 1.0
edge_threshold = 1.0
max_feature_nodes = large enough to include essentially all active features
  (e.g. 1_000_000; document if a hard cap still binds)
```

- Build once per `(model, CLT, prompt)`; all methods consume that graph.
- Setup **must fail** if an existing graph’s `metadata.node_threshold` ≠ requested threshold unless `FORCE_REBUILD_GRAPH=1`.
- Never delete/rebuild mid-campaign without bumping an experiment id / outdir.

**Rationale (Dallas–Austin):** 0.8/0.98 prune left 279 features; unpruned export had 2028. Selector comparisons on the pruned graph are a different experiment.

---

## 2. Primary selection objective

| Role | Setting |
| --- | --- |
| **Main MIB / paper tables** | `score_kind=logit_gap` |
| **Post-hoc metric** | KL faithfulness on saved evidence (`rescore_kl`) |
| **Case study only** | Optional KL-**selection** run (Dallas-style); label clearly; do not replace Pass A |

**Rationale:** Game 1’s causal story and dual-freeze mediation are defined on logit gap. Selecting on KL changes the stop geometry and is far more expensive; IOI equal-KL matching needed hundreds of features and is not a fair sparse budget.

---

## 3. Method suite

### 3.1 Track A — feature selectors (primary)

| Method ID | What it is | Fair use |
| --- | --- | --- |
| **Game 1** | Dual-freeze greedy under MACAG `v` | Primary MACAG result; natural `|E|` + mediation |
| **Game 2** | ABR + FP contrastive | Overlap / unique-y / unique-foil; same graph+oracle |
| `influence` | Top-k by circuit-tracer **raw** influence | Matched-k to Game1; F-vs-k |
| `eap_syed` | Syed/Nanda AtP on **feature** activations | Needs length-matched corrupt; matched-k; F-vs-k |
| `acdc` | Ported top-down **node** τ-prune under MACAG `v` | τ grid; report F vs `|E|` (density–F tradeoff) |
| `shapley` | MC Shapley-gold ranking under MACAG `v` | Deferred pass; matched-k / agreement secondary |

**Omit from main tables unless explicitly discussed:**

- `eap` / `eap_graph` (graph path-effect cousin — not Syed)
- `acdc_native` (heads/MLPs — not Jaccardable to Game1)
- Baseline harness `game1` duplicate

**Game 1 knobs (fixed for campaign):**

```text
freeze_mode = both
stop_metric = raw_relative   # forced under freeze_mode=both
budget = None                # omit --budget; never pass 0
alpha = 0.5
lambda = 0.02
faithfulness_eps = 0.1
connected = false
prefilter_top_k = off
```

**Game 2 knobs:**

```text
solvers = abr fp
same graph + oracle as Game 1
MIB mass campaign: single freeze convention (see `macag_appendix_legacy.md` J.B8; v3 ran dual-freeze **Game 1** and single-convention Game 2)
Unpruned ~2k-candidate graphs: ABR is multi-hour — use case-study subset
  and/or reduced abr_iters for mass sweep; full ABR+FP on case studies
```

**Deferred after full Pass A/B/C** (not in the mass campaign): α∈{0,1}, alternate `score_kind`, Game 2 dual-freeze, pruned-graph trees, mean/resample ablation, λ/β/ε sweeps, budgeted Game 1 product stops — see [`run_todo_v4.md`](run_todo_v4.md) §5. Application tuning lives in [`APPLICATIONS_METRICS.md`](APPLICATIONS_METRICS.md); do not conflate with paper-table knobs above.

**Corrupt prompt (eap_syed + edge track):** length-checked against clean (same token length). Example two-hop: Dallas→Houston city swap. Store in `oracle_kwargs.json`.

### 3.2 Track B — native edge originals (secondary)

| Method ID | Implementation | Reporting |
| --- | --- | --- |
| `eap_edge` | UFO-101 AutoCircuit AtP ranking + size sweep | KL vs k; logit-gap S/N/F on chosen k’s |
| `acdc_edge` | UFO-101 AutoCircuit ACDC τ survivors | Native τ circuit size; KL; logit-gap S/N/F |

- Write `macag_original_baselines.json` (or per-method sidecars then merge) — **never** merge into `macag_baselines.json`.
- Native mode: no fake edge budget unless a table is explicitly “budget-matched edges.”
- Caption every edge result: *different universe, different ablation, not selector isolation.*

---

## 4. Metrics and paper tables

### Table 1 — Feature head-to-head (Track A)

Per model × task (aggregate with CIs when n allows):

- Game1 `|E|` (frozen / unfrozen), F, S, N, oracle_calls  
- Each baseline at **matched k** = Game1 `|E|` (same freeze leg for dual-freeze comparisons)  
- F-vs-k AUC (budget grid, e.g. up to 128)  
- Optional: prec@k / Jaccard vs Shapley-gold (secondary)  
- Post-hoc KL faithfulness on the same sets  

### Table 2 — ACDC density tradeoff (Track A)

- For each τ: `|E|`, F, S, N  
- Highlight that high F at small τ is **not** a sparse win (Dallas: τ=0.5 → 14 nodes, F below Game1@6)

### Table 3 — Edge pipeline (Track B)

- Circuit size, KL(full‖circuit), logit-gap S/N/F  
- No Game1 Jaccard column  

### Table 4 — Diagnostics

- Attention-mediation verdict rates by task  
- Game2 overlap_rate (expect ~0 on MIB)  
- Frozen↔unfrozen evidence Jaccard  

---

## 5. Models, data, scale

- **Models / CLTs:** gemma2-426k, gemma2-2.5M, llama32-524k (campaign defaults).  
- **MIB:** planned n per task; do not silently drop prompts after baseline failures.  
- **Case studies (full suite, unpruned graph):** Dallas–Austin; selected IOI prompts with edge track + Shapley.  
- Separate gscratch outdirs per experiment; never mix pruned and unpruned trees.

---

## Operational launch pattern

**Paper campaign (MIB, many prompts):** prompt-parallel Pass A → B → C — see [`run_todo_v4.md`](run_todo_v4.md).

```bash
scripts/slurm/submit_all_tmlr250_v4.sh          # Pass A
scripts/slurm/submit_all_tmlr250_v4_pass_b.sh   # eap_syed + acdc
scripts/slurm/submit_all_tmlr250_v4_pass_c.sh   # eap_edge + acdc_edge
# Shapley later
```

**Case studies (few prompts, deep):** method-exclusive GPU shards after shared setup (Dallas prototype):

- `scripts/run_macag_dallas_llama_setup.sh`
- `scripts/run_macag_dallas_llama_method.sh`
- `scripts/slurm/launch_macag_dallas_llama.sh`

Shapley is deferred / last. Edge S/N/F runs after edge circuits are written.

---

## 7. Pre-submission baseline audit checklist

Run this before claiming any “beats baseline” sentence:

- [ ] Track A methods share one graph SHA and one `oracle_kwargs` SHA (or equivalent identity block).  
- [ ] Graph `node_threshold` / `edge_threshold` are 1.0 (or the paper explicitly studies pruned graphs).  
- [ ] Primary selection `score_kind` is `logit_gap`; KL numbers are labeled post-hoc (unless a KL-selection case study).  
- [ ] Matched-k uses Game1 natural `|E|`, not equal-KL size.  
- [ ] `eap_syed` corrupt prompt token length == clean.  
- [ ] Port names in text: “ported ACDC / feature AtP,” not “ACDC / EAP as published,” unless Track B.  
- [ ] Track B results have no Jaccard-vs-Game1.  
- [ ] No duplicate baseline-`game1` in the primary table.  
- [ ] Dual-freeze Game1 reports mediation; freeze flags match across Track A methods in that cell.  
- [ ] Ablation mode identical across Track A (default zero).  
- [ ] Every method ID in tables resolves via `baseline_method_map.md`.  
- [ ] Negative / near-floor baselines (e.g. eap_syed on Dallas) are reported honestly, not dropped.  
- [ ] Dense ACDC wins are not described as sparse circuit discoveries.  
- [ ] Scripts used for the camera-ready tables are pinned (commit hash) in the paper or supplement.

---

## 8. One-sentence protocol (abstract / methods)

> On **unpruned** CLT attribution graphs, we select circuits with **Game 1 (dual freeze, logit-gap, natural size)** and compare to **influence, feature AtP (eap_syed), ported ACDC, and Shapley-gold** under the **same feature candidates and intervention oracle** at matched k; we **rescore KL post-hoc**; we run **Game 2** for contrastive structure; and we report **true edge EAP/ACDC** only as a **separate** pipeline baseline with KL and logit-gap S/N/F.

---

## 9. Claim language cheatsheet

| Allowed | Not allowed |
| --- | --- |
| “Game 1 exceeds top-k influence and feature AtP at matched evidence size under the shared MACAG oracle.” | “Game 1 beats Syed et al. EAP.” (unless Track B + careful wording) |
| “Ported node τ-prune under `v` needs hundreds of features to match Game 1’s sparse F.” | “ACDC fails” without saying *ported* vs *native edge* |
| “Native edge ACDC kept 3 edges with poor logit-gap S/N/F on this prompt.” | “ACDC found a 3-edge circuit that is comparable to Game 1’s 6 features” |
| “Post-hoc KL faithfulness on Game 1’s set is …” | “Game 1 optimizes KL” (main campaign) |

---

## 10. Change control

Any deviation from §§0–4 for a paper table requires:

1. A written note in the suite config / README for that run.  
2. Explicit labeling in the paper (appendix if exploratory).  
3. An update to this file if the deviation becomes the new default.

**Owner expectation:** if a baseline is wrong, we fix it and re-run before submission — we do not patch the narrative around a broken comparison.
