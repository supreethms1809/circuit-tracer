# MACAG applications: metrics and Game 1 / Game 2 tuning

**Status:** working notes for three practical applications and how to score / stop the games for each.

**Does not replace** [`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md) (paper baseline protocol) or [`INTERPRETATION_GUIDE.md`](INTERPRETATION_GUIDE.md) (how to read JSON). Those remain the source of truth for campaign tables. This document is the source of truth for *application* claims. The Dallas \(\alpha\) split that motivates the table below is in [`macag_dallas_austin.md`](macag_dallas_austin.md).

Demo notebooks (illustrative, not the eval protocol):

- `/gscratch/ssuresh/macag_mib_tmlr250v3_h200/macag_gpu_unlearning_demo.ipynb` (and `macag_live_unlearning_demo.ipynb`, `macag_surgical_unlearning_demo.ipynb`)
- `/gscratch/ssuresh/macag_mib_tmlr250v3_h200/macag_game2_steering_demo.ipynb`
- `/gscratch/ssuresh/macag_mib_tmlr250v3_h200/macag_auditor_demo.ipynb`

IOI numbers cited below are from `mib_gemma2_ioi_0000_comparison.md` in the same directory.

---

## 0. Short answer

F/S/N are the right **causal vocabulary**, but default **F** (\(\alpha=0.5\)) is the wrong **headline** for all three apps.

| Application | Game | Headline metric | Select on | \(\alpha\) | Run type |
| --- | --- | --- | --- | --- | --- |
| Surgical unlearning | Game 1 | Necessity **N** (unlearning drop) | logit_gap if a substitute token is known; else raw `logit` / `prob` of the forgotten token | **0** (pure N) | Goal-oriented on knockout; budgeted k-sweep only for baselines |
| Contrastive steering | Game 2 | Keep-only gap / argmax flip + **overlap_rate** | logit_gap(\(y\), \(y_{\text{foil}}\)) | **1** (pure S) | Goal-oriented until both clamps flip; budgeted for matched-k vs ActAdd |
| Transcoder / model auditor | Game 1 frozen **and** unfrozen | **Recoverable range** \(S_{\text{all}}-S_{\text{empty}}\) | logit_gap for task mediation; optional second Game 1 on KL for dictionary completeness | 0.5 is fine (diagnosis is all vs empty, not \(E^*\)) | Core run needs no budget; optional goal-oriented \(|E^*|\) as circuit complexity |

**KL is evaluation, not the Game 1 / Game 2 objective** for these three apps. It is the collateral / dictionary-completeness check. Successful foil steering *should* raise KL.

The current demos all use **budgeted \(k=8\)** with default \(\alpha=0.5\). That is the right setup for baseline tables, not for the application claims.

---

## 1. Metric definitions (recap)

Oracle modes for a set \(E\) (see `macag/utils/metrics.py`):

| Mode | Meaning |
| --- | --- |
| `all` | unmodified model |
| `empty` | all candidate features ablated (error nodes remain → error floor) |
| `keep_only` | only \(E\) active |
| `remove` | \(E\) ablated, rest active |

Derived:

\[
\begin{aligned}
S &= S_{\text{keep}}(E) - S_{\text{empty}} && \text{(sufficiency)} \\
N &= S_{\text{all}} - S_{\text{remove}}(E) && \text{(necessity)} \\
F &= \alpha S + (1-\alpha) N && \text{(faithfulness; default }\alpha=0.5\text{)} \\
R &= S_{\text{all}} - S_{\text{empty}} && \text{(recoverable range)}
\end{aligned}
\]

- High **S**: \(E\) alone can reconstruct the score (keep-only / clamp experiments).
- High **N**: ablating \(E\) destroys the score (knockout / unlearning).
- **F** mixes those two jobs. Default Game 1/2 optimize F on logit gap — a circuit-*explanation* compromise, not any of the three operational tests.
- **R**: how much of the score features can move at all. Sign of \(R\) under frozen vs unfrozen attention is the auditor diagnosis. If \(R \le 0\), normalized F/S/N are degenerate; read raw scores (`INTERPRETATION_GUIDE.md` §2).

Default scalar score is **logit gap** \(S = \mathrm{logit}(y) - \mathrm{logit}(y_{\text{foil}})\). Other `score_kind` values: `logit`, `prob`, `negative_loss`, `kl_divergence` (full-distribution; score \(= -\mathrm{KL}(P_{\text{ref}}\|P_{\text{int}})\), so \(S_{\text{all}}\equiv 0\)), `answer_span`.

KL faithfulness is a **post-hoc rescore** of saved evidence (`macag/kl_rescore.py`), not the MIB selection objective ([`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md) §2).

---

## 2. The three applications

### 2.1 Surgical circuit unlearning (Game 1)

**Claim.** Identify a tiny feature set and ablate it at runtime so a target concept collapses, without weight edits (ROME/MEMIT-style) and without wrecking the rest of the model.

**Operational success.** After `remove(E)`:

- target logit / logit gap drops (unlearning drop \(= N\));
- preferably argmax leaves the forgotten token (or \(P(\text{target})\) falls below a floor);
- collateral on the rest of the distribution stays small (KL or a retain-set CE — **not** currently measured in the demos; they hardcode collateral \(= 0\)).

**What F/S/N do here.**

| Metric | Role |
| --- | --- |
| **N** | **Primary.** This *is* the unlearning drop. |
| **S** | Diagnostic only: high S means \(E\) is a complete concept circuit, not merely a bottleneck. A high-S / low-N set reconstructs the concept but does not unlearn it. |
| **F (\(\alpha=0.5\))** | **Misaligned.** Can rank a reconstruction circuit above a knockout circuit. |

**IOI caution.** Gemma-2 IOI Game 1 (frozen, \(k=8\)): \(S=+18.0\), \(N=+1.625\), \(F=+9.81\). Keep-only overshoots the clean gap (\(+22.9\) vs clean \(+4.75\)). That set is a strong reconstruction circuit and a weak unlearner. Ranking by F overstates unlearning. The Llama live demo’s “unlearning %” \(= N / S_{\text{all}}\) is the right *ratio*; calling it 100% unlearning is not, unless `remove` actually inverts or zeros the concept.

**Select on.**

- `logit_gap` if the desired substitute token is known (unlearn \(y\) *in favor of* \(y_{\text{foil}}\)).
- Raw `logit` or `prob` of the forgotten token if the goal is “stop saying \(y\)” without a specific replacement.
- **\(\alpha = 0\)** (pure necessity). \(\lambda\) still for sparsity.
- **Do not select on KL.** KL does not know which concept to erase; matching dense-EAP KL on IOI needed ~384–450 features, which kills the surgical claim.

**Evaluate on.** N, `remove` score, \(P(\text{target})\), argmax flip; post-hoc KL / retain-set CE as collateral. Report S as a completeness check, not the headline.

### 2.2 Contrastive target vs foil steering (Game 2)

**Claim.** Find disjoint \(E_y\) and \(E_{\text{foil}}\), then clamp feature activations to flip the answer without changing the prompt — sparse, monosemantic pathways vs ActAdd residual steering.

**Operational success.**

1. `keep_only` / clamp of \(E_{\text{foil}}\) inverts the logit gap (foil wins argmax).
2. Clamp of \(E_y\) restores the target.
3. Overlap rate \(|E_y \cap E_{\text{foil}}| / |E_y \cup E_{\text{foil}}|\) near 0.

The steering demo’s numbers (illustrative): baseline gap \(+16.75\) → clamp foil \(-23.13\) → clamp target \(+16.42\), 0% overlap.

**What F/S/N do here.**

| Metric | Role |
| --- | --- |
| **S** | **Primary.** Clamping is a keep-only intervention. |
| **N** | Secondary: whether the other circuit is a bottleneck. |
| **F (\(\alpha=0.5\))** | Okay for discovery; weak as a product metric. A high-F foil set can still fail to flip argmax. |
| **overlap_rate** | Game 2-specific product metric. Disjointness is the contrastive claim. |

**Select on.** `logit_gap(y, y_foil)` with **\(\alpha = 1\)** (pure sufficiency). \(\beta\) high enough for ~0 overlap. Allow \(|E_y| \neq |E_{\text{foil}}|\).

**Do not select on KL.** KL is target-free and **anti-aligned** with foil steering: a working \(E_{\text{foil}}\) *moves away* from the clean distribution. Negative foil KL-faithfulness (\(-0.62\) on IOI) is evidence the foil circuit is working, not that it failed. Use KL only as a *retain-set* collateral check (did syntax / unrelated mass collapse?).

**Evaluate on.** Sign of keep-only logit gap, argmax flip both ways, overlap_rate; KL as collateral only.

### 2.3 Automated model / transcoder auditor

**Claim.** Diagnose whether a behavior lives in dictionary features or in attention routing, and whether a wider CLT fixes that. Distinct from MSE / \(L_0\) / CE recovered, which score vector approximation and miss routing.

**Operational success (core).** Sign of recoverable range under **frozen vs unfrozen** attention, on the same graph:

| Model (demo figures) | Frozen \(S_{\text{all}}\) | Frozen \(S_{\text{empty}}\) | Range \(R\) | Diagnosis |
| --- | ---: | ---: | ---: | --- |
| Llama-3.2-1B (524k) | +16.75 | −3.16 | **+19.91** | Feature-mediated |
| Gemma-2-2B (426k) IOI | +15.66 | +18.20 | **−2.54** | Attention-mediated |
| Gemma-2-2B (2.5M) IOI | +15.82 | +17.94 | **−2.12** | Still attention-mediated (6× width does not flip the sign) |

This is an **all vs empty** fact. It does **not** require a selected \(E^*\). F/S/N of a \(k=8\) set are supporting evidence, not the verdict.

**What F/S/N do here.**

| Metric | Role |
| --- | --- |
| **Recoverable range \(R\)** | **Primary diagnosis.** Compare frozen vs unfrozen (freeze-flip). |
| **S of a small \(E^*\)** | Secondary: dictionary has a usable *task* circuit. |
| **N of a small \(E^*\)** | Secondary: features are not fully redundant. |
| **F** | Fine to report; not the audit verdict. |

**Select on.**

- Task mediation: Game 1 on **logit_gap**, `--freeze-mode both`, `stop_metric=raw_relative` (required for comparable legs; `normalized` is degenerate when \(R\) collapses).
- Dictionary completeness (optional second run): goal-oriented Game 1 on **KL**. How many features until keep-only KL matches a dense reconstruction? This is a capacity audit, not a surgical circuit.

\(\alpha=0.5\) is acceptable because the diagnosis does not depend on \(E^*\).

---

## 3. KL divergence: when it helps and when it hurts

MACAG KL score is \(-\mathrm{D}_{\mathrm{KL}}(P_{\text{ref}}\|P_{\text{int}})\) at the scored position. Causal KL metrics reuse the same S/N/F formulas on that score (`INTERPRETATION_GUIDE.md` + `kl_rescore.py`).

**Use KL for**

- Collateral after unlearning: did we wreck unrelated mass, or only the target/foil pair?
- Dictionary completeness in the auditor: feature count to match dense-EAP keep KL.
- Cross-method comparison that does **not** reuse Game 1’s logit-gap objective (circularity fix in the TMLR recipe).

**Do not use KL as the Game 1/2 objective for these apps**

- It is target-free: it does not know which concept to erase or which foil to promote.
- Matching `eap_edge@4096` KL on the IOI prompt needed ~384 (frozen) / ~450 (unfrozen) features vs Game 1 \(E^*\) of 7–8. Frozen Game 1 \(E^*\) keep KL ≈ 5.66 vs empty ≈ 6.34, so \(F_{\mathrm{KL}}\approx 0.47\): a strong logit-gap circuit that barely reconstructs the full distribution.
- Unfrozen \(E^*\) can look strong on \(F_{\mathrm{KL}}\) via a huge empty floor (~19.6) while necessity stays near 0 — do not read that as knockout.
- Foil steering *should* increase KL.

Equal-KL matching as a primary budget is already forbidden for paper tables ([`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md) fairness contract item 5). Same rule here.

---

## 4. Game 1 tuning for applications

Game 1 (`macag/games/game1_min_faithful.py`) greedily maximizes

\[
U_1(E) = F(E) - \lambda |E|
\]

and optionally stops early. The knobs that change the *application* meaning of \(E^*\) are \(\alpha\), `score_kind`, `budget` / `fill_budget`, `faithfulness_eps` / `stop_metric`, `lam`, and `freeze_attention`.

### 4.1 Alpha (sufficiency vs necessity mix)

| Setting | What Game 1 prefers | Use |
| --- | --- | --- |
| \(\alpha=0.5\) (default) | Balanced explanation circuit | Paper tables, auditor reporting |
| \(\alpha=0\) | High **N** (knockout) | **Unlearning selection** |
| \(\alpha=1\) | High **S** (keep-only reconstruction) | Steering-style keep-only sets; Game 2 selection |

Default \(\alpha=0.5\) on IOI produced the high-S / low-N \(E^*\) above. For unlearning, that is the wrong preference.

### 4.2 Score kind

| `score_kind` | When to *select* | When to *evaluate* |
| --- | --- | --- |
| `logit_gap` | Unlearning with a known substitute; steering; auditor task mediation | Always report for the task pair |
| `logit` / `prob` | Unlearning “stop saying \(y\)” with no foil | Calibrated unlearning % (`prob`) |
| `negative_loss` | Target log-prob rather than a foil | Same |
| `kl_divergence` | Optional auditor dictionary-capacity Game 1 only | Collateral / completeness for all three apps |
| `answer_span` | Multi-token answers | When the product is a span, not a next token |

### 4.3 Budgeted vs goal-oriented

**Budgeted.** `--budget k` (optionally `--fill-budget`): stop at \(|E|=k\). Needed for matched-size vs EAP / ACDC / Shapley / Influence.

- Failure mode: \(k=8\) is arbitrary for a product claim. A concept may need 3 nodes or 30. `fill_budget` can add nodes with non-positive gain just to hit \(k\).

**Goal-oriented (current Game 1).** `--faithfulness-eps` + `--stop-metric`:

- `normalized` (frozen, \(R>0\)): stop when \(F_{\text{norm}} \ge 1-\varepsilon\). Degenerate when \(R\) collapses.
- `raw_relative` (unfrozen / `freeze-mode both`): stop before adding a node whose **lam-free** marginal F gain is \(< \varepsilon \times\) the first feature’s F gain.

`--freeze-mode both` **forces** `raw_relative` on both legs; `normalized` + `both` is rejected so the legs stay comparable.

- Failure mode for apps: the eps stop is on **mixed F**, so it can halt on a high-S / low-N set (bad unlearner) or before a clamp actually flips argmax.

**Goal-oriented on the operational test (what the apps actually need).** Not a first-class Game 1 stop today — read it off a k-sweep or add a stop:

- Unlearning: stop when `remove` score \(\le 0\), or \(P(\text{target})\) below a floor, or \(N / S_{\text{all}}\) above a target fraction. Minimize \(|E^*|\) subject to that.
- Steering: Game 2 has **no** `faithfulness_eps`. Grow until keep-only argmax flips both ways (or take the first \(k\) on a sweep that flips). Allow different sizes on the two sides.
- Auditor: core diagnosis is all vs empty (no budget). Optional: \(|E^*|\) at eps as circuit complexity; optional KL Game 1 size as dictionary capacity.

### 4.4 Other Game 1 knobs

- \(\lambda\): sparsity. Too high → one node or empty set. Keep modest (paper defaults 0.01–0.02); let eps / operational stop set size rather than cranking \(\lambda\).
- `min_gain`: extra floor on utility improvement; leave at 0 unless noise is a problem.
- `prefilter_top_k`: speeds search; can hide the true circuit. For application claims, prefer no prefilter or a large pool.
- `connected`: only if you need a graph-connected evidence set; IOI/unlearning demos do not require it.
- `freeze_attention`: frozen is the default circuit-tracer convention. For unlearning/steering *on attention-mediated tasks* (Gemma IOI), an unfrozen (or dual) run is required or N/S will describe the wrong pathway. The auditor *must* run both.

### 4.5 Game 2 knobs (steering)

Game 2 has no eps stop. Size is controlled by `budget`, \(\lambda\), \(\beta\), `min_gain`.

- \(\beta\): overlap penalty. Raise until overlap is ~0 without emptying \(E_{\text{foil}}\).
- Do not force the same \(k\) on both agents for the product claim; matched \(k\) is only for tables.
- Solver (ABR vs fictitious play) can change *which* nodes are selected; disjointness itself has been solver-stable on MIB. Report overlap_rate, not a single node list, as the robust claim.

---

## 5. Recommended run recipes

### Unlearning (application claim)

1. Game 1, \(\alpha=0\), no `fill_budget`.
2. `score_kind=logit_gap` (known substitute) or `logit`/`prob` (no substitute).
3. Stop when knockout is met (remove-score / \(P(\text{target})\)), not when F saturates. Until that stop exists, run a k-sweep and pick the smallest \(k\) that meets the test.
4. Report \(|E^*|\) as surgical cost, N as efficacy, post-hoc KL as collateral.
5. Dual-freeze if `recoverable_range` is weak/negative under frozen attention.

### Unlearning (paper comparison)

Budgeted sweep \(k=1..8\) (or 16) vs baselines on the **same N curve** (not F). Post-hoc KL rescore. Same graph, candidates, freeze, ablation as Track A in the TMLR recipe.

### Steering (application claim)

1. Game 2 on `logit_gap`, \(\alpha=1\), \(\beta\) high enough for ~0 overlap.
2. Grow until clamp(\(E_{\text{foil}}\)) and clamp(\(E_y\)) both flip argmax.
3. Allow \(|E_y| \neq |E_{\text{foil}}|\).
4. Evaluate overlap_rate + keep-only gap sign; KL only as retain-set collateral.

### Steering (paper comparison)

Matched-\(k\) vs ActAdd / residual steering. Report overlap and flip rate, not KL-faithfulness of \(E_{\text{foil}}\).

### Auditor

1. **Core:** `all` vs `empty`, frozen and unfrozen. Classify feature-mediated (\(R>0\) frozen) vs attention-mediated (\(R\le 0\) frozen, typically \(R>0\) unfrozen).
2. Optional goal-oriented Game 1 on logit_gap: \(|E^*|\) at eps = circuit complexity.
3. Optional goal-oriented Game 1 on KL: size to a KL target = dictionary completeness.
4. Budgeted \(k=8\) is the least informative of the three for this app.

---

## 6. Practical takeaway

Keep logit-gap F/S/N in the paper as the shared causal language. For the three applications, split them:

- unlearning is knockout (**N**);
- steering is reconstruction of a chosen pair (**S**) with disjointness;
- the auditor is the error-floor / freeze protocol (**recoverable range**).

KL is how you prove you did not quietly smash the rest of the distribution — except in steering, where moving the distribution is the point and KL is only a retain-set check.

Select on logit gap (or target `logit`/`prob` for foil-free unlearning). Always rescore with KL. Run **goal-oriented** for the product claim and **budgeted** only when you need matched-\(k\) baselines.
