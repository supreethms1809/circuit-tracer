> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). Oracle, metrics, freeze protocol, KL rescoring (former `macag.md` §2).

## 2. Framework

### 2.0 Notation

| Symbol | Meaning |
|--------|---------|
| $G=(C,A)$ | attribution graph: candidate feature nodes $C$, attribution edges $A$ |
| $C$ | candidate node set (the universe MACAG selects from); $|C|$ its size |
| $E, E_y, E_{\text{foil}}$ | evidence set (Game 1), target / foil evidence sets (Game 2) |
| $E^*$ | the returned evidence set; $|E^\*|$ its size |
| $y, y_{\text{foil}}$ | target and foil labels/tokens |
| $S_{\bullet}(\cdot)$ | oracle score in mode $\bullet \in \{$all, empty, keep, remove$\}$ |
| $v(S)\equiv f(S)$ | characteristic / faithfulness function of coalition $S$ |
| $\Delta_i(S)$ | marginal value of node $i$ to $S$: $v(S\cup\{i\})-v(S)$ |
| $\alpha$ | sufficiency/necessity mix (default 0.5) |
| $\lambda$ | sparsity penalty (CLI default 0.01; pipeline / v3 / Dallas **0.02**) |
| $\beta$ | Game 2 overlap penalty (CLI default 0.1; pipeline / v3 / Dallas **0.2**) |
| $\varepsilon$ | Game 1 early-stop threshold (`faithfulness_eps`) |
| $K$ | max solver rounds, ABR or fictitious play (Game 2; `abr_iters`) |
| $B$ | evidence-size budget |
| $\Phi$ | Game 2 exact potential function ([§3.4](macag_foundations.md#34-equilibrium-analysis)) |
| $\phi_i^{\text{Shapley}}, \phi_i^{\text{Banzhaf}}$ | per-feature gold credit ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)) |

### 2.1 Oracle Scoring

MACAG operates through a **scoring oracle** that measures model behavior under feature interventions. For a set of feature nodes $E$ and a target class $y$, four canonical intervention modes define the oracle:

| Mode | Description | Notation |
|------|-------------|----------|
| **All** | All feature nodes active (unmodified model) | $S_{\text{all}}(y)$ |
| **Empty** | All feature nodes ablated (null model) | $S_{\text{empty}}(y)$ |
| **Keep-only** | Only nodes in $E$ active, rest ablated | $S_{\text{keep}}(E, y)$ |
| **Remove** | Nodes in $E$ ablated, rest active | $S_{\text{remove}}(E, y)$ |

The score function is typically the **logit gap** between target and foil tokens:

$$S(y) = \text{logit}(y_{\text{target}}) - \text{logit}(y_{\text{foil}})$$

Other scoring modes are supported: raw logit, softmax probability, negative cross-entropy loss, and a full-distribution `kl_divergence` mode (the score is $-\mathrm{KL}(P_{\text{ref}}\,\|\,P_{\text{int}})$ at the scored position, where $P_{\text{ref}}$ is the clean-model distribution — see [§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)). KL is target-free: $S_{\text{all}} \equiv 0$ by construction, and the reference logits are computed once on the clean pass and cached.

**Implementation**: Interventions are implemented via the circuit-tracer `ReplacementModel`, which performs feature-level ablation (setting $a_f = 0$). Each intervention maps to a `(layer, position, feature_idx, ablation_value)` specification.

Attention patterns (and LayerNorm denominators) may be **frozen at their clean values or left free to recompute**, controlled by the scoring-time flag `freeze_attention` (default `True`). This choice is not cosmetic: it changes the **Empty** baseline and therefore every derived metric. See [§2.3](#23-attention-freezing-and-the-error-floor).

#### Oracle Caching

Oracle calls are expensive (each requires a forward pass through the transformer with modified feature activations). MACAG implements a **memoization cache** keyed by `(mode, type(target), str(target), frozenset(nodes), universe_fingerprint)`. The type is included in the key to differentiate integer vs. string targets with the same string representation. The universe fingerprint covers the universe-dependent modes (`empty`, `keep_only`, `remove` — they ablate the candidate universe, its complement, or its intersection with the request), so cached scores are invalidated automatically when `restrict_universe` changes the ablation universe.

Cache statistics (oracle_calls, cache_hits, cache_size) are tracked per-game and reported in the output, enabling analysis of computational efficiency across CLT variants.

### 2.2 Derived Metrics

From the four oracle scores, MACAG derives:

**Sufficiency** — does $E$ alone reproduce the model's behavior?

$$\text{sufficiency}(E) = S_{\text{keep}}(E) - S_{\text{empty}}$$

**Necessity** — is $E$ required for the model's behavior?

$$\text{necessity}(E) = S_{\text{all}} - S_{\text{remove}}(E)$$

**Faithfulness** — weighted combination of sufficiency and necessity:

$$\text{faithfulness}(E) = \alpha \cdot \text{sufficiency}(E) + (1 - \alpha) \cdot \text{necessity}(E)$$

where $\alpha \in [0, 1]$ balances the two (default $\alpha = 0.5$). When $\alpha = 1$, faithfulness equals sufficiency (can the evidence reproduce the behavior alone?). When $\alpha = 0$, faithfulness equals necessity (does removing the evidence destroy the behavior?).

**Utility** — faithfulness penalized by evidence size:

$$U(E) = \text{faithfulness}(E) - \lambda |E|$$

where $\lambda \geq 0$ is the sparsity penalty (default $\lambda = 0.01$). This encodes a preference for smaller evidence sets.

**Sparsity** — fraction of candidates not selected:

$$\text{sparsity}(E) = 1 - \frac{|E|}{|C|}$$

where $C$ is the full candidate set.

**Error-floor-aware (normalized) metrics.** Reconstruction-error nodes are, by default, never ablated, so $S_{\text{empty}}$ is not zero — it is the residual score the model retains from error nodes (and, when attention is unfrozen, from attention) when *all* features are off. To separate "how much the features can explain" from this floor, MACAG also reports a normalized view:

$$\text{error\_floor} = S_{\text{empty}}, \qquad
\text{recoverable\_range} = S_{\text{all}} - S_{\text{empty}}$$

$$\text{sufficiency}_{\text{norm}} = \frac{\text{sufficiency}(E)}{\text{recoverable\_range}}, \quad
\text{necessity}_{\text{norm}} = \frac{\text{necessity}(E)}{\text{recoverable\_range}}, \quad
\text{faithfulness}_{\text{norm}} = \alpha\,\text{sufficiency}_{\text{norm}} + (1-\alpha)\,\text{necessity}_{\text{norm}}$$

When $|\text{recoverable\_range}| < \varepsilon_{\text{range}}$ the normalized metrics are set to $0$ to avoid division by a vanishing denominator. This matters because `recoverable_range` can be **zero or negative**: ablating all features (especially with unfrozen attention) may fail to lower — or may even raise — the $S_{\text{empty}}$ baseline, which means the behavior lives in attention / error nodes rather than in features. In that regime the normalized metrics are degenerate and the **raw** sufficiency/necessity/faithfulness should be read instead. Attention-mediated tasks (e.g. IOI) routinely show negative `recoverable_range` under frozen attention.

### 2.3 Attention Freezing and the Error Floor

The single most consequential scoring choice in MACAG is whether attention is frozen. It interacts with the error floor to produce two failure modes we explicitly correct for; understanding it is required to read any Game 1 result.

**What the flag does.** `freeze_attention=True` holds every attention pattern at the value it took on the clean (unablated) forward pass, so feature ablations only remove the *direct* feature contribution to the residual stream. With `freeze_attention=False`, attention is recomputed from the ablated activations, so removing a feature also removes everything that feature would have caused *through* attention on downstream positions.

**Failure mode 1 — frozen attention hides features from the minimal set.** Under frozen attention, any upstream feature whose downstream effect is mediated by attention is already "paid for" by the frozen pattern. The Empty baseline $S_{\text{empty}}$ keeps that contribution even with *all* features ablated, so the greedy never needs to add the upstream feature to recover the behavior — it looks redundant. The minimal faithful set therefore collapses to a few late-layer / final-token features and silently drops the city/structure features that actually drive the circuit. Re-running with `freeze_attention=False` forces attention to be reconstructed from features, which **recruits those upstream and early-layer features back into the minimal set**. The `analyze_frozen_vs_unfrozen.py` analyzer quantifies exactly this: it counts upstream features (reverse-position $> 0$, i.e. not at the prediction token) and early-layer features recovered when attention is unfrozen. *Empirical caveat:* the **direction** of the evidence-set change is task-dependent — two-hop relational prompts can *recruit* upstream features when unfrozen ([legacy appendix](macag_appendix_legacy.md)), but on v3 IOI/MCQA/ARC (matched budget 8) unfreezing *shrinks* mean $|E^\star|$ and is what flips Gemma frozen $R<0$ to unfrozen $R\approx +11$. The convention-invariant statement is that frozen attention *distorts* what the minimal set must contain (it can hide attention-mediated upstream features, or add spurious necessity that unfreezing removes); which way it errs is itself a per-task diagnostic.

**Failure mode 2 — the normalized denominator goes degenerate.** The normalized metrics divide by `recoverable_range` $= S_{\text{all}} - S_{\text{empty}}$. Two things push this denominator toward zero or negative:
- *Frozen attention on an attention-mediated task* (e.g. IOI): the answer lives in the frozen attention pattern, so ablating all features barely moves the score — $S_{\text{empty}} \approx S_{\text{all}}$ and the range collapses, sometimes going negative (the 426k gemma IOI runs are 10/10 such reconstruction "failures").
- *Unfrozen attention*: ablating all features and letting attention recompute can collapse or even invert the baseline, so $S_{\text{empty}}$ can exceed $S_{\text{all}}$.

Either way the normalized metric divides by a near-zero/negative number and produces wild values (the "weird denominator" results). The fixes are layered:
1. `metrics.py` guards the division (`_RANGE_EPS = 1e-9`) and zeroes the normalized metrics when $|\text{recoverable\_range}| < \varepsilon_{\text{range}}$, so the output is never a spurious huge ratio.
2. Game 1's normalized early-stop (`stop_metric=normalized`) inherits the same weakness, so a `raw_relative` stop was added that never touches the denominator (see [§4](macag_game1.md#4-game-1-minimal-faithful-evidence)).
3. For analysis, the **raw** sufficiency / necessity / faithfulness_delta are denominator-free and are the metrics to report whenever the range is unreliable. `analyze_robust_frozen_vs_unfrozen.py` re-reads the stored raw scores (no new model runs) precisely to sidestep the broken denominator.

**Recommended practice.**

| Situation | Attention | Stop metric | Read |
|-----------|-----------|-------------|------|
| Clean, feature-mediated task | frozen | `normalized` | normalized + raw |
| Attention-mediated task (IOI), or `recoverable_range` ≤ 0 | unfrozen | `raw_relative` | raw |
| Diagnosing where the behavior lives (the matched protocol) | `--freeze-mode both` | `raw_relative` (forced on both legs) | `attention_mediation` block (verdict, range flip, upstream/early recruitment) |

Because `freeze_attention` is a scoring-time argument, the attribution graph is unchanged between frozen and unfrozen runs. The **matched protocol is now built into the CLI**: `run_macag game1 --freeze-mode both` builds the ReplacementModel oracle once, derives a freeze-flipped twin sharing the same model (`macag.scoring.derive_oracle_with_freeze` — fresh cache per leg, since every intervention score depends on the freeze convention), and runs both legs under identical budget / prefilter / eps / α / λ with `stop_metric=raw_relative` forced on both (the `normalized` stop is degenerate on the unfrozen leg, so mixing stop rules would make the legs incomparable; passing `--stop-metric normalized` with `--freeze-mode both` is an error). The output carries a per-prompt `attention_mediation` block (§4.5). **Historical caveat:** the pre-existing two-invocation sweeps (`scripts/run_macag_unfrozen*.sh`, analyzed by `experiments/analyze_frozen_vs_unfrozen.py`) were *not* parameter-matched — the stored unfrozen runs raised the Game 1 budget 8 → 20 and prefilter 20 → 30. `all`/`empty` — and therefore `recoverable_range` — are budget-independent, so the §10.4 sign-flip diagnostic is unaffected; evidence sizes and upstream-feature counts in those stored runs are partially confounded by the lifted cap ([§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt)). The pass-2 re-run should use `--freeze-mode both`, which removes that confound by construction. <!-- pass-2 note: matched-protocol decision resolved 2026-06-11; implemented as --freeze-mode both -->

### 2.4 Reported Metrics: Consolidated Definitions

Every quantity that appears in the results (§10, Appendix C) and what it means. Raw quantities are in logit-gap units; normalized quantities are unitless ratios.

| Metric | Definition | Reads as | Caveat |
|--------|------------|----------|--------|
| `all` $S_{\text{all}}$ | score, no ablation | baseline target–foil gap | — |
| `empty` $S_{\text{empty}}$ | score, all features ablated | the **error floor** (residual from error nodes / frozen attention) | sign matters (see below) |
| `keep_only` $S_{\text{keep}}(E)$ | score, only $E$ active | how much $E$ alone reconstructs | — |
| `remove` $S_{\text{remove}}(E)$ | score, $E$ ablated from full | residual without $E$ | — |
| **sufficiency** | $S_{\text{keep}}(E)-S_{\text{empty}}$ | can $E$ *alone* drive the behavior? | raw; denominator-free |
| **necessity** | $S_{\text{all}}-S_{\text{remove}}(E)$ | does removing $E$ break it? | raw; denominator-free |
| **faithfulness** $f/v$ | $\alpha \cdot $ suff $+(1-\alpha)\cdot$ nec | overall causal explanation of $E$ | raw; the games' objective |
| **utility** $U$ | $f(E) - \lambda E $ (G1); $ -\beta E \cap E_{\text{other}} $ added (G2) | sparsity-penalized objective | — |
| **recoverable_range** | $S_{\text{all}}-S_{\text{empty}}$ | how much score features *can* recover above the floor | **≤0 ⇒ behavior not in features; normalized metrics degenerate** |
| **$E^*_{normalized}$** | raw $\div$ recoverable_range | fraction of recoverable range explained | unreliable when range ≤ 0 |
| **evidence size** $E^*$ | # nodes selected | parsimony | capped by budget $B$ |
| **sparsity** | $1-E^*/C$ | fraction of candidates *not* used | depends on $C$ |
| **upstream-feature count** | # nodes in $E^*$ with reverse-position $>0$ (not at the prediction token) | how much of the circuit is *upstream* structure vs. final-token readout | key for the frozen/unfrozen contrast |
| **early-layer count** | # nodes in $E^*$ in early layers | depth profile of evidence | — |
| **target_preferred** | $S_{\text{all}}>0$ (model predicts target over foil at baseline) | is the oracle measuring the right thing? | **exclude `False` rows from faithfulness aggregates** |
| **overlap_rate** (G2) | $\|E_y\cap E_{\text{foil}}\|/\|E_y\cup E_{\text{foil}}\|$ | target/foil feature sharing (0 = disjoint) | denominator-free; attention-invariant |
| **shared / unique_y / unique_foil** (G2) | $E_y\cap E_{\text{foil}}$, $E_y\setminus E_{\text{foil}}$, $E_{\text{foil}}\setminus E_y$ | contrastive decomposition | — |
| **range-flip rate** | fraction of prompts with recoverable_range $<0$ frozen but $\ge0$ unfrozen | **the attention-mediation diagnostic** (§10.4); emitted per-prompt as `attention_mediation.range_flip` / `verdict` by `--freeze-mode both` | report with CI once multi-seed |
| **oracle_calls / cache_hits** | # model forward passes / memoized hits | compute cost; cache efficiency | denominator for cost-ratio vs Shapley |
| **converged** (G2) | solver dynamics stabilized within $K$ (iterates repeated; FP also: frequencies within `fp_tol`) | solution-quality flag | the returned allocation is the best round (`best_iteration`), not necessarily the final/converged iterate |
| **cross-seed / cross-prompt Jaccard** | set agreement of $E^*$ across repeats | stability of the selected circuit | needs multi-seed (Phase 1) |
| **kl_faith** (`kl_faithfulness`) | faithfulness recomputed on the *already-selected* evidence with `score_kind="kl_divergence"` ([§2.5](#25-kl-rescoring-a-selection-independent-faithfulness-metric)) | selection-independent faithfulness check (evaluation metric ≠ greedy objective) | evaluation-only; in nats, not logit-gap units — compare across methods, not against raw faith |

**Sign conventions to keep straight.** A *negative* `empty` means the model prefers the foil once features are ablated (features carry the whole behavior — healthy). A *positive* `empty` larger than `all` means ablation *helped* the target → negative `recoverable_range` → a reconstruction failure where the normalized view is meaningless and only raw scores are valid.

### 2.5 KL Rescoring: a Selection-Independent Faithfulness Metric

Game 1 greedily maximizes logit-gap faithfulness and is then reported on that same
quantity — the circularity threat in [§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt).
The implemented mitigation is a second, **selection-independent** faithfulness
metric: every stored evidence set is *re-scored* (never re-selected) under
`score_kind="kl_divergence"`, where the oracle score is
$-\mathrm{KL}(P_{\text{ref}}\,\|\,P_{\text{int}})$ over the full next-token
distribution at the scored position ($P_{\text{ref}}$ = clean model,
$P_{\text{int}}$ = ablated model). This matches the MIB/ACDC convention of
comparing the intervened circuit to the full model, and it is **foil-free** — it
also answers the "logit-gap / single-foil is the wrong metric" objection.

Properties worth stating:
- $S_{\text{all}} \equiv 0$ (the reference compared to itself), and every other
  mode is $\le 0$, so under KL: `error_floor` $= -\mathrm{KL}(P_{\text{ref}}\|P_{\text{empty}}) \le 0$
  and `recoverable_range` $= \mathrm{KL}(P_{\text{ref}}\|P_{\text{empty}}) \ge 0$ **always** —
  the degenerate negative-denominator regime of [§2.3](#23-attention-freezing-and-the-error-floor)
  cannot occur under KL (though the range can still be near zero, with the same
  $\varepsilon_{\text{range}}$ guard).
- The same four-mode oracle and `FaithfulnessMetrics` machinery is reused, so
  KL sufficiency/necessity/faithfulness are defined exactly as in
  [§2.2](#22-derived-metrics), just in nats.
- Cost is 4 oracle calls per evidence set (all/empty/keep/remove), per freeze leg.

**Implementation.** `macag/scoring.py` adds the `kl_divergence` score kind
(`compute_kl_score`; reference logits cached from the clean pass);
`macag/kl_rescore.py` + `python -m macag.cli.rescore_kl --run-dir <dir> | --root <sweep>`
walk saved run directories, rebuild the oracle from the stored
`oracle_kwargs.json` (per freeze leg for dual-freeze Game 1 outputs), and re-score
Game 1 evidence, Game 2 target/foil allocations, and every baseline selector's
selected set. Output: a per-run sidecar `macag_kl_faithfulness.json`, plus
`kl_faithfulness` blocks embedded into `macag_game1.json` / `macag_game2.json` /
`macag_baselines.json`. The per-prompt pipeline runs it as its final step and the
sweep drivers run it as the first analysis step, before the aggregators
(`KL_RESCORE=0` to skip; pipeline flag `--skip-kl-rescore`), so the aggregation
analyzers can emit `kl_faith` columns. The machinery is parametrized by a
`RescoreSpec` (2026-07-02), and a second flavor ships with it:
`python -m macag.cli.rescore_altfoil` re-scores the same stored evidence under
the run's own score kind but an **alternate foil token** (`--foil-token`, or
per-slug `metadata.alt_incorrect_token` from a manifest via `--bench`), writing
`macag_altfoil_faithfulness.json` / `altfoil_faithfulness` blocks — the
foil-choice robustness check of §11.3. The ACDC manifest now carries
`alt_incorrect_token` for every prompt (IOI: the corrupted-prompt third name;
greater_than: another below-threshold year; docstring: a generic wrong
parameter). Smoke-verified on a stored MIB IOI run (2026-07-02): both dual-freeze
legs rescored under the absent-name foil, sidecar written and blocks embedded
alongside `kl_faithfulness` (`summary.csv`, `baselines.csv`,
`frozen_vs_unfrozen.csv`). Unit coverage: `tests/test_macag_kl_scoring.py`.

---
