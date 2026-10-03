> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). PROVENANCE / DO NOT CITE AS CURRENT. Pre-v3 two-hop tables, Appendix C, and the 2026-07-18 campaign audit (J). Replaced for paper numbers by `macag_experiments_v3.md` and `macag_dallas_austin.md`.

> **Do not quote these numbers in new text.** They predate the TMLR-250 v3 campaign and/or the 2026-06-09 solver fixes. Use [`macag_experiments_v3.md`](macag_experiments_v3.md) and [`macag_dallas_austin.md`](macag_dallas_austin.md) instead.

## 9. Case Study: Attention Mediation and CLT Capacity (Gemma-2, Llama-3.2)

> The remainder of this document (§9–§11) is **one application** of MACAG, not part
> of the framework. It feeds MACAG the attribution graphs of three publicly
> available cross-layer transcoders and runs the identical game machinery on each.
> Nothing in §2–§8 depends on these results; a different study would swap in
> different graphs and read the same metrics. *(These runs use public linear CLTs
> on Gemma-2 / Llama-3.2; they do not include a Spline-CLT or GPT-2 — see
> [§11.1](macag_discussion.md#111-what-this-case-study-does-not-yet-cover).)*

### 9.1 Setup

**Models and transcoders.** Three CLTs span a clean capacity control and a cross-model check:

| Tag | Model | Transcoder set | Candidate pool $|C|$ (per prompt) |
|-----|-------|----------------|-----------------------------------|
| `gemma2-426k` | google/gemma-2-2b | `mntss/clt-gemma-2-2b-426k` | ~311–338 |
| `gemma2-2.5M` | google/gemma-2-2b | `mntss/clt-gemma-2-2b-2.5M` | ~297–321 |
| `llama32-524k` | meta-llama/Llama-3.2-1B | `mntss/clt-llama-3.2-1b-524k` | ~260–291 |

`gemma2-426k` → `gemma2-2.5M` is the **capacity** axis (model and architecture fixed, ~6× transcoder width); `gemma2-*` → `llama32-524k` is the **cross-model** axis. A fourth CLT (`gpt-oss-20b / mntss/clt-131k`) was configured but **did not run**: `transformer_lens` does not recognize `openai/gpt-oss-20b`, so it is absent from all results.

**Prompts.** Two sets:
- *Two-hop factual* — 8 prompts of the form "Fact: The capital of the state containing {CITY} is", target = capital, foil = state (the city→state→capital circuit), shared across all three CLTs.
- *ACDC benchmark* — `indirect_object_identification` (10) and `docstring_completion` (3); `greater_than` is excluded because its target/foil share a first token, giving a degenerate first-token logit gap.

**Parameters** (verified against the stored run JSONs). Game 1 **frozen**: $\alpha{=}0.5$, $\lambda{=}0.02$, budget 8, prefilter top-20, $\varepsilon{=}0.1$, `normalized` stop. Game 1 **unfrozen** (two-hop and ACDC): same $\alpha$, $\lambda$, $\varepsilon$ but budget **20** and prefilter top-**30** — *not* parameter-matched to the frozen legs ([§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)). Game 2 (frozen *and* unfrozen): $\beta{=}0.2$, ABR iters 4, budget 8, prefilter top-20. Each prompt is run **frozen** and, reusing the same graph, **unfrozen**. Unfrozen stop metrics: the two-hop unfrozen rerun in `macag_unfrozen/` used the `normalized` stop (against §2.3's own recommendation — see the C.3 source note); a second `raw_relative` rerun lives in `macag_unfrozen_raw/`; the ACDC-benchmark unfrozen runs used `raw_relative` because their frozen `recoverable_range` is negative. **All stored `raw_relative` runs predate the 2026-06-09 stop fix** and used the buggy λ-penalized variant (§10 provenance box). Regardless of which stop selected the evidence, unfrozen results must be *read* on raw metrics only — normalized values are meaningless there (§2.3).

### 9.2 What Was Run

| Experiment | CLTs | Prompts | Attention | Status |
|------------|------|---------|-----------|--------|
| Two-hop generalization sweep | gemma2-426k | 8 two-hop | frozen | ✅ 8/8 |
| Capacity / cross-model compare | all 3 | 8 two-hop | frozen | ✅ (gpt-oss failed) |
| Game 1 frozen vs unfrozen | all 3 | 8 two-hop | both | ✅ 24/24 |
| Game 2 frozen vs unfrozen | all 3 | 8 two-hop | both | ✅ 24/24 |
| ACDC-benchmark Game 1 | all 3 | 13 IOI+docstring | both | ✅ 39/39 |

This is a single-seed study; multi-seed repeats are not yet present ([§11.1](macag_discussion.md#111-what-this-case-study-does-not-yet-cover)).

### 9.3 Baselines and Evaluation Protocol

The CLT-variant comparison above varies the *input graph*. The complementary — and, for positioning MACAG, more important — axis varies the *selection method* on a fixed graph: it asks whether Game 1's intervention-based greedy actually beats the cheaper incumbents and how close it gets to the expensive gold. Each baseline ([§1.3](macag_motivation.md#13-relation-to-prior-work)) supports a specific claim:

| Baseline | What it produces | Claim the comparison supports | Status |
|----------|------------------|-------------------------------|--------|
| **Top-k influence** | the $k$ highest-`influence` graph nodes | MACAG's *search* beats trusting attribution magnitude (higher faithfulness at matched $|E|$) | **implemented + run** on MIB (`macag/baselines/influence.py`; graph-rank only, zero oracle calls — [J.A1](#ja-baselines-and-the-shared-characteristic-function)); nonlinear §10.7 numbers provisional |
| **EAP / attribution patching** | top-$k$ nodes by first-order patching score | *real* interventions beat the local-linear scores the graph is already built from | **implemented + run** (`macag/baselines/eap.py`, graph-derived signed-path-effect variant seeded at the target/foil logits); nonlinear benchmark (§10.7): faith\@8 1.15, loses 58/60 — IOI/multi-hop pending |
| **ACDC** | a minimal circuit via top-down edge pruning | bottom-up node selection is competitive at far lower granularity/cost | **implemented + run** (ported node version, `macag/baselines/acdc_prune.py`, τ-sweep); ACDC benchmark prompts wired in (`macag/data/acdc_benchmark_prompts.json`); nonlinear benchmark (§10.7) at uncapped k≈193 (not budget-matched); native-granularity comparison still open |
| **Shapley / Banzhaf (gold)** | exact-ish per-feature credit ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)); gold set = top-$k$ Shapley prefix | greedy evidence ≈ top-Shapley features at a fraction of the oracle cost | **implemented** (`macag/baselines/shapley_select.py`); **not executed in MIB** ([J.A5](#ja-baselines-and-the-shared-characteristic-function)); nonlinear §10.7 gold numbers provisional ([J.C14](#jc-campaign-status-and-coverage)); Banzhaf is diagnostic/reported credit only in the paper's run config ([§11.2](macag_discussion.md#112-framing-options-for-the-paper-pick-one) item 7), though the harness can fall back to it as gold ([J.A4](#ja-baselines-and-the-shared-characteristic-function)) |

**Common protocol.** Every method scores the *same* candidate node set under the *same* oracle (`logit_gap`, `freeze_attention` per [§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)), and is compared on three axes: faithfulness at matched evidence size, evidence size at matched faithfulness, and oracle calls. The intended headline ("one killer result") is a single plot: **MACAG reaches Shapley-gold-level faithfulness at ~45× fewer oracle calls, and dominates top-k-influence / EAP / ACDC on faithfulness-per-feature** (ranking agreement with gold is moderate — prec@k 0.46 / Jaccard 0.33 — and is reported as a secondary, metric-independent signal, not the headline).

> **Status note.** All four baselines and the head-to-head harness are now
> implemented: selectors live in `macag/baselines/` (influence, EAP, MC
> Shapley/Banzhaf over the MACAG oracle, ported ACDC, plus the B3.2 brute-force
> optimality-gap tool) and `python -m macag.cli.run_baselines` runs every
> selector on the same candidate set and oracle, emitting per-k evidence/scores,
> per-method oracle-call counts, precision@k/Jaccard vs Shapley-gold, the
> faithfulness-vs-size AUC, and the Spearman(EAP, greedy-marginal) linearity
> diagnostic. The repo's `attribution/shapley.py` is still a spline-CLT
> attribution tool, **not** the MACAG Shapley-gold
> ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)). The harness has unit
> coverage on toy oracles (`tests/test_macag_baselines.py`) **and has now been
> run on real CLT graphs** — the 60-prompt nonlinear benchmark
> (`results/macag_nonlinear_connected/`, reported in §10.7/C.7 with bootstrap CIs
> and paired Wilcoxon tests). The harness is now also **wired into the sweep
> drivers**: `run_macag_acdc.sh` / `run_macag_mib.sh` run every selector
> per-prompt (writing `macag_baselines.json` per run directory) and
> `experiments/analyze_macag_baselines.py` aggregates a sweep root into
> `baselines.csv` (faith@own-k, faith-per-feature, AUC, precision@k/Jaccard vs
> gold, oracle calls, plus `kl_faith_*` columns from [§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric)) —
> so the IOI/multi-hop head-to-head is now an *execution* task, not a build task.
> The full 3-seed MIB-bench campaign ([§9.5](#95-mib-bench-full-campaign-setup)) is in
> progress. **As-run roots** are the split gscratch trees
> `/gscratch/$USER/macag_mib_{h200,gemma25m,llama}/macag_mib_seed{0,1,2}/`
> ([J.C15](#jc-campaign-status-and-coverage)); the older planned path
> `results/macag_mib_seed{0,1,2}/` is **[STALE]** for the live campaign. The campaign
> supplies the multi-seed Shapley variance that was previously open; the
> ACDC-benchmark re-run through the consolidated driver remains open.

### 9.4 Analysis Protocol (raw outputs → claims)

The reproducible path from a run to a reported number, so results can be regenerated and audited.

1. **Run** the pipeline per prompt (`scripts/run_macag_pipeline.sh`) or batched via the consolidated drivers: `scripts/run_macag_acdc.sh` (ACDC benchmark; the old `run_macag_sweep.sh` / `run_macag_unfrozen*.sh` two-pass scripts are consolidated into it — `FREEZE_MODE=both` is the default) and `scripts/run_macag_mib.sh` (same pipeline over MIB-bench prompts), with per-CLT parallel launchers `run_macag_{acdc,mib}_parallel.sh` (shared logic in `macag_parallel_common.sh`; resume via `--skip-attribute/--skip-game1/--skip-game2`). Each run directory gets `macag_game1.json` (dual-freeze schema when `FREEZE_MODE=both`), `macag_game2_{abr,fp}.json` (+ canonical `macag_game2.json`), `macag_baselines.json` (per-prompt head-to-head, unless `SKIP_BASELINES=1`), `macag_kl_faithfulness.json` ([§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric)), and `oracle_kwargs.json`, each with `params`, `evidence`, `scores`, `stats`.
2. **Validity filter.** Drop prompts with `target_preferred = False` (oracle is measuring the wrong direction) from any faithfulness aggregate; keep them only for coverage/robustness counts. (In C.2/C.3 this removes 3 llama rows.)
3. **Pick the right metric per regime** ([§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)): if `recoverable_range > 0` you may report normalized; if `≤ 0` report **raw** sufficiency/necessity/faithfulness only. Report the `kl_faith` column alongside raw faithfulness as the selection-independent cross-check ([§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric)); its range is non-negative by construction, so it is also usable in the degenerate-denominator regime.
4. **Aggregate** with the `experiments/analyze_*.py` scripts. For the consolidated sweep roots (`results/macag_acdc`, `results/macag_mib`, `results/macag_nonlinear*`) the drivers auto-run four aggregators against `--root`/`--bench`: `analyze_macag_acdc.py` → `summary.csv` (raw + `kl_faith` faithfulness, per-prompt `verdict`/`range_flip`, per-CLT×task flip rate), `analyze_game2_abr_vs_fp.py` → `abr_vs_fp.csv` (ABR↔FP evidence Jaccard, overlap, utilities), `analyze_macag_baselines.py` → `baselines.csv` (per-method faith@own-k, faith-per-feature, AUC, precision@k/Jaccard vs Shapley-gold, oracle calls), and `analyze_acdc_frozen_vs_unfrozen.py --root` → `frozen_vs_unfrozen.csv` (dual-freeze legs from one JSON; legacy `--frozen-root`/`--unfrozen-root` fallback for the old two-pass results). The older CSVs in `macag/macagresults/` remain the source for §10.1–10.6. Per CLT/task report: mean raw faithfulness, mean $|E^\*|$, mean upstream count, fraction with `range < 0`, mean overlap_rate.
5. **Attention-mediation diagnostic.** Run each prompt with `--freeze-mode both` and aggregate the **range-flip rate** = fraction with `attention_mediation.range_flip == true`, read directly from the dual-run JSONs (no pairing step). This is the headline of §10.4. `experiments/analyze_frozen_vs_unfrozen.py` remains as the legacy path for the pre-existing two-invocation (non-parameter-matched) sweep results.
6. **Uncertainty** (Phase 1): bootstrap over prompts (resample with replacement, 10k draws) → 95% CI for every aggregate; for the flip rate, CI on the proportion.
7. **Baselines** (Phase 2): run each selector through the *same* oracle on the *same* $C$; compare at **matched $|E|$** and via faithfulness-vs-size AUC.
8. **Report** each number against [§10.6](#106-claims-to-evidence-to-research-question-mapping) so every claim has a traceable source.

### 9.5 MIB-Bench Full Campaign: Setup

This is the paper's primary quantitative evaluation — the successor to the 8-prompt
two-hop sweep and the 13-prompt ACDC benchmark of [§9.1](#91-setup), scaled to
community-standard prompts and a 3-seed design. It grounds the prompt sets and the
baseline family in the **MIB circuit-localization benchmark**
(`mib-bench/*` on HuggingFace, MIB circuit-track code vendored at
`external/MIB-circuit-track`, one-time setup `external/setup_mib.sh`). What follows is
the full protocol; results are reported in §10 once the campaign completes.

**9.5.1 Prompts.** `experiments/build_mib_benchmark_prompts.py` exports MIB-bench
HuggingFace prompts into `macag/data/mib_benchmark_prompts.json` — the same per-prompt
schema as the hand-written ACDC manifest (clean/corrupted prompt, single-token
target/foil) plus `mib_model` / `mib_task` / `mib_split` routing metadata, with prompts
tokenizer-aligned to their MIB model. Public CLTs exist for gemma-2-2b and
Llama-3.2-1B. The manifest routes **both** families:
`mib_model=gemma2` → `{gemma2-426k, gemma2-2.5M}`, `mib_model=llama3` →
`llama32-524k` (see `benchmarks_info.mib_model_to_clt_tags` in the JSON). The
**as-run campaign includes `llama32-524k`** on IOI (500/500) and MCQA (50/50) —
complete — while arc_easy is incomplete ([J.C11](#jc-campaign-status-and-coverage),
[§11.2](macag_discussion.md#112-framing-options-for-the-paper-pick-one) item 3). Scope the
two-family claim to IOI and MCQA.

> **[STALE] do not cite:** any older prose that said Llama-3.2 "carries no MIB
> prompts," that `llama32-524k` workers are "unused," or that the MIB campaign is
> gemma-only. Those claims referred to an early gemma-only export / single-node
> plan and are false for the live campaign ([J.C15](#jc-campaign-status-and-coverage),
> [§9.5.5](#955-compute)).

Benchmark-size export (both models; IOI capped at 500):

```bash
python experiments/build_mib_benchmark_prompts.py \
  --models gemma2 llama3 --tasks ioi mcqa arc_easy --split validation \
  --task-limit ioi=500
```

Task sizes (validation split): `mcqa` (copycolors) and `arc_easy` are exported **in
full** — 50 and 570 prompts respectively, their entire validation splits. `ioi` is
**capped at 500** of its ~10,000 validation prompts (`--task-limit ioi=500`) — running
the full IOI split at the campaign's per-prompt cost was estimated at roughly 2
GPU-years and judged not worth it relative to the marginal statistical value past 500
prompts; `run_mib_benchmark.sh` refuses to launch against a prompt file with
$\le 10$ IOI prompts (`ALLOW_SMALL_JSON=1` overrides this refusal for a smoke run) so a
pilot-sized manifest cannot be mistaken for the campaign manifest.

> **As actually run ([Appendix J.C12](#jc-campaign-status-and-coverage)):** the manifests are
> **IOI 500**, **MCQA 50** (gemma2-2.5M: 38), **arc_easy 211** (gemma2) / **570** (llama3) —
> not 560 per task. The planned 50-prompt gold subset has **0** completed runs.

**9.5.2 Per-prompt game parameters.** Every cell runs the same consolidated pipeline as
the case study ([§9.4](#94-analysis-protocol-raw-outputs--claims), `scripts/run_macag_pipeline.sh`), with the
pipeline's defaults used unchanged — these are the same values as [§9.1](#91-setup)
except the freeze protocol, which is dual **for Game 1 only**:

> **Scope correction ([J.B8](#jb-protocol-integrity)).** `FREEZE_MODE=both` splits the
> *Game 1* legs. Game 2, the baselines pass, and KL rescoring each run a **single**
> attention convention (the stored `freeze_attention`). Do not claim that every
> headline experiment runs both conventions.


| Parameter | Value |
|---|---|
| $\alpha$ | 0.5 |
| $\lambda$ | 0.02 |
| $\beta$ | 0.2 |
| Game 1/2 budget | 8 |
| prefilter top-$k$ | 20 |
| Game 2 solvers | `abr` **and** `fp` (both run; [§5.2](macag_game2.md#52-algorithm-simultaneous-best-response-the-abr-solver)/[§5.2.1](macag_game2.md#521-fictitious-play-solver-solverfp)), $K{=}4$ rounds, `fp_tol`$=10^{-3}$ |
| Game 1 `freeze_mode` | `both` (matched dual-freeze legs, [§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)) |
| Game 1 stop | `raw_relative`, $\varepsilon{=}0.1$ (forced by `freeze_mode=both`) |
| ablation | zero (`ABLATION_MODE=zero`) |
| score kind | `logit_gap` (no MIB task here has a first-token target/foil collision, so `answer_span` is not needed) |
| candidate universe | full graph feature-node set, `--connected` (the CLI default) |

Each cell also gets a post-hoc KL rescore ([§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric)) of every stored evidence set,
run automatically at the end of `run_macag_pipeline.sh`.

**9.5.3 Baseline harness: fast pass + deferred gold pass.** Every cell also runs the
[§9.3](#93-baselines-and-evaluation-protocol) head-to-head harness on the identical graph and oracle. Because MC
Shapley/Banzhaf is $\approx$90% of a cell's baseline cost ($\approx$34k oracle calls vs.
Game 1's $\approx$760), the campaign splits it into two passes:

- **Fast pass** (`BASELINE_METHODS=influence,eap,game1,acdc`, run inline by
  `run_macag_mib.sh`): top-$k$ influence, graph-derived EAP, a second budget-matched
  Game 1 run (`game1_connected=True` matching the standalone CLI), and **budget-matched
  ACDC** — `acdc_target_size` bisects the prune threshold $\tau$ to the target evidence
  size (`ACDC_TARGET_K=-1` $\Rightarrow$ match `--budget`$=8$; up to 24 bisection
  iterations against the oracle's memoization cache; `exact=false` is reported honestly
  when 1/16-quantized bf16 scores make the exact budget unreachable, with
  `achieved_k` recorded instead of silently substituting a nearby size).
- **Gold pass** (`scripts/run_macag_shapley_pass.sh`, run after the fast pass
  completes): MC-Shapley and MC-Banzhaf ([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)) over the same oracle, 64
  permutations, antithetic pairing, seeded by the campaign seed
  (`SHAPLEY_SEED=$SEED`) — restricted to the first `GOLD_PER_TASK=50` prompts per
  task (matched by slug index, `SLUG_REGEX='_(ioi|mcqa|arc_easy)_00[0-4][0-9]$'`) to
  bound the gold-baseline cost, then merged back into `macag_baselines.json` via
  `python -m macag.cli.merge_baselines` (pure JSON recombination — no extra oracle
  calls) and re-KL-rescored.

**9.5.4 Three-seed design and orchestration.** The whole protocol (fast pass, gold
pass, post-processing) is driven by `scripts/run_mib_benchmark.sh`. On the cluster
the supported entrypoint is the three CLT-split SLURM submitters
(`scripts/slurm/submit_all_mib.sh` → `submit_macag_mib_{h200,gemma25m,llama}.sh`),
each with its own `RESULTS_ROOT` under `/gscratch/$USER/`:

```bash
# Preferred (as-run): one job per CLT group, separate gscratch roots
scripts/slurm/submit_all_mib.sh

# Legacy single-process form (still valid for local smoke; not the live campaign)
nohup scripts/run_mib_benchmark.sh > results/mib_campaign.log 2>&1 &
```

`SEEDS="0 1 2"` (default) controls the only stochastic stage of an otherwise
deterministic pipeline — the MC-Shapley/Banzhaf sampling seed — so the three
replicates give the cross-seed ranking-stability check called for in roadmap **B1.3**
([§3.6](macag_foundations.md#36-relation-to-shapley-and-banzhaf-credit)); every other stage (graph construction, greedy Game 1/2, ACDC
bisection) is seed-independent, hence identical bar bf16 run-to-run jitter across
processes. **As-run layout:** each submitter writes
`/gscratch/$USER/macag_mib_{h200,gemma25m,llama}/macag_mib_seed<SEED>/`. The older
planned single-tree path `results/macag_mib_seed<SEED>/` is **[STALE]** for the live
campaign ([J.C15](#jc-campaign-status-and-coverage)). After seeds finish,
`scripts/macag_combine_seeds.py` concatenates per-seed analyzer CSVs with a `seed`
column (point it at the gscratch seed dirs, not the unused `results/` path). Every
stage is resumable: `run_mib_benchmark.sh` and the per-prompt pipeline both skip
already-completed work, so a killed or crashed run restarts from where it left off
with `SEEDS=<remaining>` (or another `sbatch` of the same submitter).

**9.5.5 Compute.**

> **As-run (2026-07, authoritative for citation).** The MIB campaign runs as
> **three concurrent SLURM jobs** on separate 2×H200 nodes (`ai4wy-2`, 24h walltime),
> one CLT group each:
>
> | Job script | CLT | `RESULTS_ROOT` | Default workers | `MAX_WORKERS_PER_GPU` |
> |---|---|---|---|---|
> | `submit_macag_mib_h200.sh` | `gemma2-426k` | `…/macag_mib_h200` | 12 | 6 |
> | `submit_macag_mib_gemma25m.sh` | `gemma2-2.5M` | `…/macag_mib_gemma25m` | 6 | 8 |
> | `submit_macag_mib_llama.sh` | `llama32-524k` | `…/macag_mib_llama` | 8 | 8 |
>
> Within a node, shards of the **same** CLT run in parallel (`macag_parallel_common.sh`);
> a per-GPU slot semaphore caps concurrent GPU work. `llama3` MIB prompts **do**
> route to `llama32-524k` (500 IOI + 50 MCQA + 570 arc_easy in the manifest). Long
> `arc_easy` prompts need a lower concurrency than ioi/mcqa (peak memory scales with
> prompt length; see comments in `submit_macag_mib_h200.sh`). Resume after walltime
> via `submit_all_mib.sh`. Details / stale-claim list: [J.C15](#jc-campaign-status-and-coverage).

> **[STALE] early single-node GH200 plan — do not cite as the live campaign.** An
> older revision of this subsection described *all* work on one NVIDIA GH200
> (96 GB), with `gemma2-426k` and `gemma2-2.5M` never concurrent, worker defaults
> `WORKERS_GEMMA2_426K=8` / `WORKERS_GEMMA2_2_5M=3`, and the false claim that
> "`llama32-524k`'s workers are unused — no MIB prompts route to it." That was a
> gemma-only / single-node draft. The ~7.5 cells/hour and "1120-cell ≈ one week"
> throughput notes attached to that plan are likewise **not** the as-run H200
> three-job rates. Keep the historical OOM lesson (collateral OOMs under high
> concurrency; mitigated by resume + `MAX_WORKERS_PER_GPU`), but quote hardware and
> routing from the as-run table above.

**9.5.6 Post-processing (per seed, after the gold pass).** `run_mib_benchmark.sh` runs,
in order: `experiments/analyze_macag_baselines.py` (refresh `baselines.csv` with the
merged gold columns), `scripts/macag_bootstrap_wilcoxon.py` (bootstrap CIs over prompts
+ paired Wilcoxon signed-rank tests with Holm correction, Game 1 vs. each baseline,
[§B1.2](macag_appendix_roadmap.md#appendix-b-roadmap-to-a-submission-ready-evaluation)), `experiments/plot_faithfulness_curves.py`
(per-method faithfulness-vs-size curves with CI bands, [§B3.1](macag_appendix_roadmap.md#appendix-b-roadmap-to-a-submission-ready-evaluation)), and
`experiments/analyze_gold_circuits.py --task ioi --include-baselines` (the
(layer, token-role) IOI gold-circuit scorer of [§9.6](#96-interpbench-native-component-gold-circuit-validation-setup)'s sibling diagnostic, **B4.1** —
precision/component-recall of every selector's evidence against the published IOI
circuit's layer bands and token roles). The same four aggregate CSVs as the ACDC sweep
([§9.4](#94-analysis-protocol-raw-outputs--claims) step 4: `summary.csv`, `abr_vs_fp.csv`, `baselines.csv`,
`frozen_vs_unfrozen.csv`) are produced automatically as part of the fast-pass driver
(`run_macag_mib.sh`'s built-in analysis stage) before the gold pass runs.

**9.5.7 External reference frame (historical, not part of the running campaign).**
Separately from the campaign above, `docs/mib-reproduction.md` documents a one-off
calibration exercise that reproduced the MIB circuit-track's own methods (EAP,
EAP-IG variants, clean-corrupted, exact patching, information-flow-routes) against
the MIB paper's published CPR/CMD numbers, to sanity-check that the [§9.3](#93-baselines-and-evaluation-protocol) EAP
baseline (a cheaper, graph-derived variant) is calibrated against the real
implementation. A gpt2/IOI smoke run is preserved at `results/mib_reproduction/`; the
driver script itself has since been pruned from `scripts/` (the repository's script
tree was narrowed to the two active campaign chains — this one and InterpBench,
[§9.6](#96-interpbench-native-component-gold-circuit-validation-setup)), so this is context for the EAP baseline's provenance, not a
reproducible step of the current protocol. *Caveat if resurrected:* MIB leaderboard
methods select **edges over native components** and report CPR/CMD, while MACAG
selects **CLT feature nodes** (or, in [§9.6](#96-interpbench-native-component-gold-circuit-validation-setup), native components) and reports
faith@k / precision-recall — the shared object is the task/prompt distribution, not
the metric, so cross-benchmark numbers are context, not a head-to-head.

### 9.6 InterpBench Native-Component Gold-Circuit Validation: Setup

Every diagnostic in [§9.5](#95-mib-bench-full-campaign-setup) validates MACAG's evidence against a
*heuristic* layer/token-role encoding of the published IOI circuit (roadmap **B4.1**,
[Appendix B](macag_appendix_roadmap.md#appendix-b-roadmap-to-a-submission-ready-evaluation)), because CLT features have no exact
ground-truth membership. InterpBench closes that gap by giving MACAG a model whose
circuit is known **exactly**, at native-component granularity, so both precision *and*
recall are well-defined (roadmap **B4.2**).

**Model and candidate universe.** `mib-bench/interpbench` (hub-hosted) is a 6-layer,
4-head transformer trained to implement IOI, whose ground-truth circuit — `m0`, `a1.h1`,
`a2.h1`, `a4.h1` (MLP-0 and three attention heads, MIB naming) — is published exactly
via `interpbench_graph.json`'s `in_graph` flags. It is loaded as a `HookedTransformer`
with `use_hook_mlp_in=True`, `use_attn_result=True`, `use_split_qkv_input=True` (required
for the per-head ablation hooks; the evaluation config differs from the training config
that produced the checkpoint). Because there are no CLT features to select over, the
oracle backend swaps the `ReplacementModel` feature scorer for
`macag.scoring_components.HookedComponentInterventionScorer`, which ablates **native
components** — attention heads `a{layer}.h{head}` and MLPs `m{layer}` — via forward
hooks, implementing the identical four-mode oracle contract ([§2.1](macag_framework.md#21-oracle-scoring)) the CLT games
use. The candidate universe is every component in the model
(`component_universe`: $6 \times 4 = 24$ heads $+ 6$ MLPs $= 30$ nodes), so Game 1, Game
2, and the MC-Shapley/Banzhaf estimator all run **completely unmodified** against this
backend — the only thing that changed is what an "intervention" ablates.

**Prompts.** MIB's IOI validation split, capped at `--limit 500` (`LIMIT=500`,
matching the [§9.5](#95-mib-bench-full-campaign-setup) IOI cap so both benchmarks probe the same prompt budget).

**Protocol** (`experiments/run_interpbench_macag.py`, one call per seed):
1. Build the component-level graph/candidate set for the prompt.
2. Run `estimate_shapley` (64 permutations, antithetic) over the component oracle,
   producing a per-component credit ranking; score it against the gold `in_graph`
   flags with node-level **AUROC** and **average precision** (`macag.eval.gold_circuits`)
   — the InterpBench-native counterpart of the MIB circuit track's own (edge-level)
   AUROC metric, computed here at node level.
3. Run Game 1 with `--budget 4` (matching $|{\text{gold circuit}}|=4$, so the
   comparison is against a same-size draw rather than an arbitrary cutoff); score the
   returned evidence against the gold node set with set-level **precision / recall /
   F1** — recall is well-defined here (unlike the CLT-feature case) because gold and
   evidence share the same component ontology.

**Three-seed design and orchestration**, mirroring [§9.5.4](#95-mib-bench-full-campaign-setup):

```bash
nohup scripts/run_interpbench_benchmark.sh > results/interpbench_campaign.log 2>&1 &
```

`SEEDS="0 1 2"` again seeds only the MC-Shapley/Banzhaf sampling (and the bootstrap
resampling in the aggregate); each seed writes `results/interpbench_macag_seed<SEED>/interpbench_macag.csv`
plus a bootstrap-CI aggregate, and a seed whose output CSV already exists is skipped on
re-run (crash-safe, same resumability contract as [§9.5](#95-mib-bench-full-campaign-setup)). Knobs: `SHAPLEY_PERMUTATIONS=64`,
`BUDGET=4`, `DEVICE=cuda` — all defaulted to the campaign configuration in the script
header, overridable via environment variables for debugging runs.

**Why this is a distinct diagnostic from §9.5's gold-circuit check.** The MIB campaign's
`analyze_gold_circuits.py` pass asks whether MACAG's CLT-feature evidence reads from the
*known layer bands and token roles* of the IOI circuit on the *real* gemma CLTs — a
necessarily heuristic comparison, because "feature 41823 of a 426k-feature CLT" has no
canonical mapping to "S-inhibition head." InterpBench asks the sharper question — exact
node-for-node recovery — but only on a small synthetic model built to *have* an exact
answer, not on the CLTs the rest of the paper studies. The two results are complementary,
not substitutable: agreement between them is the strongest gold-circuit evidence
available; disagreement localizes whether a shortfall is a CLT-representation problem
(features don't align with the named components) or a Game-1-selection problem (the
right evidence exists but greedy doesn't find it), since InterpBench removes the CLT
from the loop entirely.

---

## 10. Results

> **How to read these.** All numbers are **raw** scores (logit-gap units),
> single seed, computed from the analysis CSVs in `macag/macagresults/`. Per
> [§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor), normalized faithfulness is
> unreliable wherever `recoverable_range` is small or negative — which is most of
> the ACDC benchmark — so the headline metrics here are raw sufficiency,
> raw faithfulness, evidence size, upstream-feature count, and the sign of
> `recoverable_range`. **Never read normalized metrics on unfrozen runs** — the
> denominator is unreliable there and the values are meaningless (§2.3).

> **⚠ Result provenance (pass 1, 2026-06-10).** Every number in §10 and Appendix C
> comes from runs executed **2026-06-03/05** (`macag/macagresults/`, code state
> ≤ `d5000d7`), predating the **2026-06-09** fixes in `71a2ef6`:
> - the `raw_relative` stop then compared **λ-penalized utility gains**, not the
>   λ-free faithfulness gains described in §4.2 — all stored `raw_relative` runs
>   (`macag_unfrozen_raw/`, `macag_acdc_unfrozen/`) used the buggy stop, so their
>   evidence sizes and faithfulness-at-stop will change on re-run;
> - Game 2 **best-iterate tracking / `best_iteration` did not exist** — the 48
>   stored Game 2 runs return the final iterate (low risk: all converged in 2
>   rounds);
> - oracle-call counters were **not reset per solve** — treat the C.6 cost numbers
>   (~808 / ~1761) as provisional.
>
> **Robust to all of the above:** `recoverable_range` signs (the §10.4 flip
> diagnostic) and Game 2 `overlap_rate`, which depend on neither the stop rule nor
> the returned iterate. **Also remember** the frozen/unfrozen parameter mismatch:
> frozen legs ran budget 8 / prefilter 20, unfrozen legs 20 / 30 (§2.3, §9.1,
> §11.3). Regenerate every table below from a post-`71a2ef6` re-run before quoting
> numbers in the paper. **§10.7 / Appendix C.7 are the exception — that
nonlinear-benchmark run is already post-`71a2ef6` and may be quoted as-is.**
<!-- TODO(pass-2): refresh §10.1–10.6 + Appendix C.1–C.6 numbers (§10.7/C.7 already post-fix) -->

### 10.1 The two-hop factual circuit (gemma2-426k, frozen)

All 8 city→capital prompts are target-preferred. MACAG recovers a small, high-faithfulness evidence set with the normalized stop, and the *contrastive* structure is perfectly clean — every prompt has **overlap_rate = 0.0** (target and foil evidence are disjoint). One prompt (philadelphia-harrisburg) has a **negative `recoverable_range`** (−2.6): a reconstruction failure where ablating all features does not collapse the behavior, so its normalized faithfulness is meaningless (−2.5) and must be read raw.

| Quantity | Value (mean over 8 prompts) |
|----------|------------------------------|
| target-preferred | 8 / 8 |
| `recoverable_range` < 0 (reconstruction failures) | 1 / 8 (philadelphia) |
| evidence size $|E^*|$ | 5.25 (range 1–8) |
| Game 2 overlap_rate | 0.0 (all 8) |
| oracle calls — Game 1 / Game 2 | ~808 / ~1761 |

### 10.2 Capacity and cross-model comparison (frozen)

Same 8 prompts, three CLTs (gpt-oss-20b failed to load):

| CLT | target-preferred | recon. failures (range < 0) | mean faith_norm (valid) | mean sparsity | mean overlap |
|-----|:---:|:---:|:---:|:---:|:---:|
| gemma2-426k | 8/8 | 1/8 | 0.99 | 0.984 | 0.0 |
| gemma2-2.5M | 8/8 | 1/8 | 1.10 | 0.983 | 0.0 |
| llama32-524k | 5/8 | 2/8 | 0.97 | 0.978 | 0.0 |

*Counting conventions (from `analyze_clt_comparison.py`; stated here because the
raw C.2 table otherwise looks inconsistent with this one):* "recon. failures"
counts negative `recoverable_range` **only among target-preferred prompts** —
llama has 5/8 negative ranges in C.2, but three of those are on prompts the model
does not even perform, which makes them invalid rows, not transcoder failures.
"mean faith_norm (valid)" averages over prompts that are target-preferred **and**
have positive range.

**Capacity did not fix reconstruction failures.** Increasing the gemma CLT ~6× (426k → 2.5M) left exactly one prompt with negative `recoverable_range` in both — the behavior that lives outside the features is a property of the *task/attention*, not of transcoder width. The cross-model llama CLT is weaker on this template (only 5/8 prompts target-preferred, 2 reconstruction failures). Contrastive separation (overlap 0.0) holds across all three.

### 10.3 Frozen vs unfrozen Game 1: attention hides upstream features

Reusing each frozen graph with `freeze_attention=False` recruits **more upstream features** (reverse-position > 0, i.e. not at the prediction token) into the minimal set — the mechanism predicted in [§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor):

| CLT | upstream (frozen) | upstream (unfrozen) | $|E^*|$ frozen | $|E^*|$ unfrozen |
|-----|:---:|:---:|:---:|:---:|
| gemma2-426k | 1.25 | **2.50** | 5.5 | 5.1 |
| gemma2-2.5M | 1.25 | **3.38** | 5.25 | 9.75 |
| llama32-524k | 3.62 | 3.62 | 6.25 | 6.25 |

The two gemma CLTs roughly double their upstream-feature count when attention is unfrozen; llama's **means** are unchanged on the two-hop set (upstream 3.62 → 3.62, $|E^*|$ 6.25 → 6.25) — an averaging coincidence, not per-prompt stability (dallas upstream 3 → 0, portland 3 → 8; see C.3). The effect is real but **noisy at single seed** — individual prompts can destabilize under unfrozen attention (e.g. gemma2-2.5M portland-salem, whose *normalized* faithfulness blows up to −15.8 because `range_u` collapses to −0.375; the raw faithfulness is a sane 5.94 — exactly the §2.3 rule that normalized values must never be read on unfrozen runs), which is why these need multiple seeds and why the `raw_relative` stop exists.

*Caveats for this table:* the unfrozen legs ran at budget 20 / prefilter 30 vs the frozen legs' 8 / 20, so the evidence-size growth (2.5M's 9.75 exceeds the frozen cap of 8) and part of the upstream recruitment is mechanically enabled by the lifted cap (§2.3); and these unfrozen runs selected evidence with the `normalized` stop, which §2.3 recommends against unfrozen — the `raw_relative` rerun in `macag_unfrozen_raw/` gives different unfrozen numbers (e.g. miami $E_u$ 15 vs 13). <!-- TODO(pass-2): re-run with matched budget + fixed raw_relative stop and regenerate this table -->

> **Task-dependence (matched-protocol correction, [§10.7](#107-nonlinear-benchmark-baseline-head-to-head-and-matched-frozenunfrozen-pass-2)/C.7).** "Unfreezing recruits upstream features" is **specific to the two-hop relational circuit**. On the matched-budget nonlinear benchmark (no lifted-cap confound), unfreezing instead *shrinks* the set — upstream 3.45 → 2.87, $|E|$ 5.37 → 4.38 — the same direction as the ACDC/IOI sweep (§10.4). So the headline mechanism here is **"frozen attention *adds spurious necessity* that unfreezing removes,"** and recruitment vs shrinkage is itself a per-task diagnostic, not a universal law. Treat the gemma "doubling" above as a property of multi-hop tasks until replicated under matched budgets.

### 10.4 ACDC benchmark: the attention-mediation result

> **Primary scale for this claim is now MIB IOI at $n=500$ per model**
> ([J.D21](#jd-numbers-behind-claims-already-in-the-draft)): gemma2-426k **402/500**
> `attention_mediated`, llama32-524k **476/500** `feature_mediated`. The $n=13$
> ACDC-benchmark table below is the historical small study that first showed the
> direction; upgrade paper prose to the $n=500$ counts.

This is the strongest finding. On IOI + docstring (13 prompts/CLT), frozen attention makes the behavior look *unrecoverable from features* — `recoverable_range` is negative almost everywhere — and unfreezing flips it positive:
<!-- TODO(pass-2): the unfrozen faith/E/upstream columns below came from the buggy (λ-penalized) raw_relative stop; the range columns and flip counts are stop-independent and robust. -->

| CLT | range < 0, **frozen** | range < 0, **unfrozen** | upstream frozen → unfrozen | $|E^*|$ frozen → unfrozen |
|-----|:---:|:---:|:---:|:---:|
| gemma2-426k | **12 / 13** | 1 / 13 | 5.5 → 2.3 | 6.6 → 3.3 |
| gemma2-2.5M | **13 / 13** | 1 / 13 | 6.0 → 3.7 | 7.2 → 5.1 |
| llama32-524k | 0 / 13 | 0 / 13 | 6.4 → 3.4 | 7.6 → 5.0 |

For gemma, the ACDC-benchmark tasks (especially IOI) are **attention-mediated**: with attention frozen, ablating all features barely moves the logit gap (range strongly negative, e.g. IOI ranges of −8 to −24), so feature-level faithfulness is ill-defined. Across gemma's 26 ACDC-benchmark cases (IOI + docstring × two CLTs), `recoverable_range` is **negative in 25/26 when frozen and non-negative in 24/26 when unfrozen**, with **23/26 strictly flipping sign** (negative→non-negative); unfreezing attention recruits the features that drive the behavior. Llama's CLT already places the behavior in features (range positive throughout, 0/13 negative), a genuine cross-model difference MACAG surfaces automatically. This is the empirical payoff of the frozen/unfrozen + raw-metric machinery in [§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor).

### 10.5 Contrastive separation is robust (Game 2)

**Primary (MIB campaign — cite this).** Across **1712/1712** completed Game 2 runs (three CLTs × three tasks; abr and fp solvers; [J.D17](#jd-numbers-behind-claims-already-in-the-draft)), **overlap_rate = 0.0**. Disjointness is solver-invariant; the *identity* of $(E_y, E_{\text{foil}})$ is not (abr/fp agree on ~1% of prompts). MIB Game 2 runs a single attention convention ([J.B8](#jb-protocol-integrity)).

**Historical small case study (two-hop, pre-`71a2ef6`).** Across all 24 frozen and 24 unfrozen two-hop Game 2 runs, overlap was also 0.0 (48/48). That figure is superseded for headline citation by the MIB count above.

### 10.6 Claims to Evidence to Research-Question Mapping

How each result supports the contributions and RQs in [§1.4](macag_motivation.md#14-problem-statement-research-questions-and-contributions). "Status" marks whether the current data settles the claim or whether a roadmap phase is still needed.

| Claim | RQ / Contribution | Evidence (this doc) | Status |
|-------|-------------------|---------------------|--------|
| Minimal node sets are causally sufficient + necessary | RQ1 / C3 | §10.1, C.1 (faith_norm ≈ 0.83–1.32 on feature-mediated prompts) | partial on these graphs — search > influence is shown on the nonlinear benchmark (§10.7); the Phase-2 sweep on the two-hop/IOI graphs is pending |
| Real interventions beat attribution magnitude / EAP | RQ1 / C1,C3 | MIB fast baselines on IOI/MCQA ([J.A5](#ja-baselines-and-the-shared-characteristic-function)); §10.7/C.7 nonlinear tables provisional | **partial** — MIB gold absent; nonlinear artifacts missing ([J.C14](#jc-campaign-status-and-coverage)) |
| Target/foil carried by distinct features | RQ2 / C4 | §10.5, [J.D17](#jd-numbers-behind-claims-already-in-the-draft) (overlap 0.0 in **1712/1712** MIB); C.5 historical 48/48 | strong (denominator-free; solver-invariant disjointness) |
| Contrastive separation is not a scoring artifact | RQ2 / C4 | §10.5 (holds frozen *and* unfrozen) | supported; widen task families |
| MACAG diagnoses attention- vs feature-mediation | RQ3 / C5 | [J.D21](#jd-numbers-behind-claims-already-in-the-draft) IOI $n=500$: gemma2-426k **402/500** `attention_mediated`, llama32-524k **476/500** `feature_mediated`; §10.4/C.4 are the small historical ACDC-bench ($n=13$) | **shown** at MIB scale; upgrade §10.3/§10.4 prose accordingly |
| Diagnosis is model/task dependent | RQ3 / C5 | §10.4 (gemma vs llama; IOI vs docstring) | supported, small n |
| Capacity ≠ more faithful circuit | RQ4 / C5 | §10.2, C.2 (failure moves 426k→2.5M) | suggestive; needs more CLTs |
| Game 2 equilibrium exists & solver stabilizes | C4 | §3.4 (exact-potential existence proof; greedy-Jacobi dynamics not guaranteed, mitigated by best-iterate + FP; empirically `converged=True`) | existence theory done; convergence empirical, single-config |
| Greedy ≈ gold-level faithfulness at far lower cost | C2 / RQ1 | §10.7/C.7 tables only; MIB gold pass = 0 outputs ([J.C11](#jc-campaign-status-and-coverage)) | **provisional / not citeable** until gold runs or nonlinear artifacts recover ([J.C14](#jc-campaign-status-and-coverage)) |

The two formerly **missing** rows are now populated by the matched-protocol
nonlinear-benchmark run ([§10.7](#107-nonlinear-benchmark-baseline-head-to-head-and-matched-frozenunfrozen-pass-2), [Appendix C.7](#c7-nonlinear-benchmark-matched-protocol-baselines--frozenunfrozen)):
the Phase-2 baseline head-to-head exists at single seed, so RQ1 and the "search is
worth it" half of C1 are now *shown* (pending multi-seed CIs from Phase 1/5).

### 10.7 Nonlinear-benchmark baseline head-to-head and matched frozen/unfrozen (pass-2)

> **Provisional / appendix-only ([§11.2](macag_discussion.md#112-framing-options-for-the-paper-pick-one) item 4, [J.C14](#jc-campaign-status-and-coverage)).**
> The tables below document a reported post-`71a2ef6` run, but
> `results/macag_nonlinear_connected/` is **not present** on this filesystem and
> cannot currently be regenerated. **Do not cite 44.7× / ≈760 vs ≈34k / 60/60 as
> headline paper numbers.** Prefer MIB Game 2 disjointness (1712/1712) and the
> IOI attention-mediation diagnostic ($n=500$) from [Appendix J](#appendix-j-code-and-campaign-audit-2026-07-18).

A second case study runs the **matched-protocol** `run_macag game1 --freeze-mode
both` (one model load, both legs at budget 8 / prefilter 20 / `raw_relative` /
`connected=True`) plus the full Phase-2 baseline harness over a new **nonlinear
benchmark**: 4 task families (`boolean_logic`, `negation_polarity`,
`context_polysemy`, `hard_semantic_foil`) × 5 prompts × 3 CLTs (gemma2-426k,
gemma2-2.5M, llama32-524k) = **60 prompts**. These are single-step completion prompts
chosen so the target/foil contrast is a nonlinear function of context (e.g. *"We sat
on the grassy bank of the"* → `river` vs `money`), complementing the multi-hop
two-hop/IOI prompts of §10.1–10.5. Full per-prompt rows: [Appendix C.7](#c7-nonlinear-benchmark-matched-protocol-baselines--frozenunfrozen);
source CSVs `results/macag_nonlinear_connected/{baselines,summary,frozen_vs_unfrozen,abr_vs_fp}.csv`.
This run is **post-`71a2ef6`** (fixed `raw_relative` stop, per-solve oracle-stat
reset, Game-2 best-iterate) and is the first to carry real baseline numbers — the
Phase-2 head-to-head was previously "implemented, not run."

**(1) Game 1 owns the efficiency frontier (the C2/RQ1 result).** At matched budget
$k\!=\!8$, means with **95% bootstrap CIs over prompts** (10 000 resamples) and
**paired Wilcoxon** tests of Game 1 vs each budget-matched baseline (Holm-corrected,
matched-pairs rank-biserial effect $r$). The resampling unit is the *prompt*, not a
random seed: Game 1 / influence / EAP / ACDC are deterministic given a fixed graph +
oracle, so prompt variation is the only meaningful source of uncertainty
(reproduce: `scripts/macag_bootstrap_wilcoxon.py` <!-- TODO(pass-2): this script is not currently checked into scripts/ — restore it from the run environment before submission -->).

| selector | faith@8 [95% CI] | faith / feature [95% CI] | oracle calls | vs G1: median Δ | win/60 | $p$ (Holm) | $r$ |
|---|--:|--:|--:|--:|--:|--:|--:|
| top-k influence | 0.08 [−0.02, 0.20] | 0.010 [−0.003, 0.025] | 0 | +4.95 | 60/60 | <1e‑9 | 1.00 |
| EAP (graph-derived) | 1.15 [0.55, 1.83] | 0.144 [0.065, 0.226] | 0 | +3.72 | 58/60 | <1e‑9 | 0.98 |
| MC Shapley (gold) | 4.72 [4.02, 5.45] | 0.590 [0.500, 0.683] | 32 495 | +1.00 | 44/60 | 8.2e‑5 | 0.60 |
| **MACAG Game 1** | **5.50 [4.88, 6.15]** | **0.859 [0.747, 0.977]** | **718** | — | — | — | — |
| ACDC (τ-prune, k≈193) | 10.50 [8.60, 12.46] | 0.067 [0.050, 0.088] | 2 612 | *(not matched)* | — | — | — |

(faith/feature is per-prompt faith ÷ selected-set size — the budget-8 selectors
divide by 8, Game 1 by its own $|E|$ (mean 6.8). All numbers in this table are the
**all-60** aggregates; the target-preferred-only (n=50) variants are quoted where
marked. AUC, prec\@k 0.46 and Jaccard 0.33 vs gold are in C.7 — the gold-agreement
columns are defined only on the 34/60 prompts where Game 1 filled the full k=8
budget, since a shorter evidence set has no k-matched gold prefix.) Reading the table
in the order the evidence is strongest:

- **Cost (provisional — do not headline).** The stored table reports Game 1 at **44.7× fewer** oracle calls than
  Shapley-gold (95% CI [43.5, 45.8]), **cheaper on 60/60 prompts**, one-sided Wilcoxon
  $p<10^{-9}$. These figures are **not re-verifiable** here ([J.C14](#jc-campaign-status-and-coverage)); park them until artifacts recover.
- **Faith-per-feature separates cleanly.** Game 1 **0.859 [0.747, 0.977]** vs Shapley
  0.590 [0.500, 0.683] — the **CIs do not overlap** (target-preferred-only: 0.938
  [0.820, 1.068] vs 0.636). Unlike raw faith, fpf cannot be inflated by
  spending more features, which is exactly how ACDC posts a high raw number; its fpf
  collapses to 0.067 (worse than everything but influence) at k≈193.
- **Raw faith\@8: Game 1 wins on a clear majority, not uniformly.** Median paired
  advantage over Shapley is +1.00 ($p=8.2\times10^{-5}$, $r=0.60$), but the **win-rate
  is 44/60** — Shapley wins ~16 prompts. The honest sentence is "significantly higher
  faithfulness on most prompts," and this comparison is on *Game 1's own greedy
  objective*, so we treat it as supporting, not headline (see threats, §11.3).
- **Influence ≈ noise, EAP weak.** Both lose on 58–60/60 prompts at $r\ge0.98$ — the
  §A.3 "is search needed?" floor answers *yes, decisively*.

ACDC is **excluded from the significance tests**: its uncapped k≈193 is not
budget-matched to the k=8 selectors, so its higher raw faith (10.5, "wins" 46/60) is a
ceiling at ~28× the feature budget, not a competitor. Restricting to the 50
target-preferred prompts (§(2): the logit-gap oracle is ill-posed on the other 10)
*strengthens* every effect — Game 1 faith 5.96, fpf 0.938, vs-Shapley median Δ +1.08
($p=9.6\times10^{-5}$), cost 44.8× — so the benchmark-hygiene filter helps rather than
hides the result.

**(2) Attention-mediation is the exception, not the rule, on this benchmark.** Of 60
prompts the per-prompt verdict is **42 feature_mediated, 12 indeterminate, 6
attention_mediated**, and only **6/60 (10%) range-flip** (frozen `range` < 0 →
unfrozen ≥ 0). The flips are concentrated almost entirely in **gemma `hard_semantic_foil`**
(sem_03/04/05 on both gemma CLTs) plus one llama `boolean_logic` case; `context_polysemy`
is **15/15 feature_mediated**. This sharply contrasts the §10.4 ACDC/IOI result where
gemma flipped 23/26 — i.e. these single-step "nonlinear" completions are
**predominantly feature-carried**, and attention-mediation is a property of
*multi-hop relational* tasks (IOI), not of nonlinearity per se. `boolean_logic` is the
weak family (7/15 indeterminate, and 9/60 of the not-target-preferred prompts live
here — the logit-gap oracle is measuring the wrong thing when the model does not even
prefer the target).

**(3) Unfreezing *shrinks* evidence here — opposite of the two-hop sweep.** Mean
upstream-feature count goes **3.45 → 2.87** and evidence size **5.37 → 4.38** when
attention is unfrozen (consistent across all three CLTs). This matches the §10.4
ACDC/IOI direction (unfreeze → fewer features once attention recomputes) and is the
**opposite** of the §10.3/C.3 two-hop sweep (unfreeze → *recruits* upstream features).
So the "frozen attention hides upstream features" claim is **task-dependent**: it is a
property of the two-hop relational circuit, not a universal — a threat-to-validity
correction to the §10.3 framing ([§11.3](#113-threats-to-validity--reviewer-rebuttals-to-pre-empt)).

**(4) Contrastive disjointness (provisional on this root; superseded at MIB scale).**
The stored nonlinear tables report overlap **0.0** on 60 ABR + 60 FP runs (with the
small two-hop 48/48, **180/180** on that combined historical count) and close ABR/FP
utilities. **Do not treat ABR≈FP set-identity as a robustness result:** on the MIB
campaign, abr/fp agree on only ~1% of prompts while disjointness holds on
**1712/1712** ([J.D17](#jd-numbers-behind-claims-already-in-the-draft)) — multiple
equilibria, not solver failure. Cite MIB 1712/1712 for the paper; keep this
subsection as provisional secondary evidence only.

---

## Appendix C: Full Per-Prompt Results

Raw material for tables/figures, taken verbatim from the analysis CSVs in
`macag/macagresults/`. Single seed; all scores in logit-gap units. Columns:
`E*`/`E_f`/`E_u` = evidence size (frozen/unfrozen); `up` = upstream feature count
(reverse-position > 0, i.e. not at the prediction token); `range` =
`recoverable_range` = all − empty; `faith`/`suff`/`nec` with `_norm` = divided by
`range` (unreliable when |range| small/negative — read raw); `pref` = model
predicts target at baseline.

### C.1 Two-hop factual, gemma2-426k, frozen (`macag_sweep/summary.csv`)

> *Source note (pass 1).* `summary.csv` is dated 2026-06-03 but the
> `macag_game1.json` files now in `macag_sweep/` are from a 2026-06-04 re-run, and
> the two disagree slightly on the same prompts (miami `empty` 6.75 here vs 6.5 in
> the JSON, range 1.5 vs 1.75; detroit $E^*$ 1 vs 2; houston 6 vs 7). C.3's frozen
> columns read the JSONs — which is why C.1 and C.3 differ on the same frozen runs.
> <!-- TODO(pass-2): regenerate summary.csv and this table from a single re-run. -->

| slug | all | empty | range | E* | faith_norm | suff_norm | nec_norm | sparsity | overlap |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| chicago-springfield | 2.75 | -1.81 | 4.56 | 5 | 0.99 | 1.11 | 0.86 | 0.99 | 0 |
| cleveland-columbus | 3.56 | -2.52 | 6.08 | 8 | 0.93 | 0.23 | 1.62 | 0.98 | 0 |
| dallas-austin | 5.0 | 1.31 | 3.69 | 4 | 1.03 | -0.05 | 2.12 | 0.99 | 0 |
| detroit-lansing | 3.5 | 1.41 | 2.09 | 1 | 0.91 | 1.94 | -0.12 | 1.00 | 0 |
| houston-austin | 4.62 | -0.75 | 5.38 | 6 | 0.91 | 0.05 | 1.77 | 0.98 | 0 |
| miami-tallahassee | 8.25 | 6.75 | 1.5 | 2 | 1.32 | 0.06 | 2.58 | 0.99 | 0 |
| philadelphia-harrisburg | 1.62 | 4.25 | **-2.62** | 8 | **-2.52** | -1.48 | -3.57 | 0.98 | 0 |
| portland-salem | 5.19 | -1.16 | 6.34 | 8 | 0.83 | 0.6 | 1.05 | 0.98 | 0 |

*Observations.* (i) `empty` is frequently **negative** (chicago −1.81, cleveland
−2.52, houston −0.75, portland −1.16): with all features ablated and attention
frozen, the model prefers the *foil*, so features carry the entire target
preference — a clean feature-mediated signature, and the regime where the normalized
metric behaves (faith_norm ≈ 0.83–1.32). (ii) philadelphia is the lone
reconstruction failure: `empty` = +4.25 > `all` = +1.62, so ablating features
*raises* the gap → negative range, nonsense normalized faith (−2.52); its raw
numbers are the ones to quote. (iii) miami has a high error floor (`empty` 6.75):
most of the behavior survives full ablation, small recoverable range (1.5),
faith_norm inflated to 1.32. (iv) Evidence size varies 1–8 with no obvious tie to
range; detroit needs a single feature. (v) overlap = 0 everywhere.

### C.2 Capacity / cross-model, two-hop, frozen (`macag_clt_compare/comparison_per_prompt.csv`)

| CLT | slug | pref | range | faith_norm | E* | overlap |
|---|---|:-:|--:|--:|--:|--:|
| gemma2-426k | chicago-springfield | True | 4.56 | 0.99 | 5 | 0 |
| gemma2-426k | cleveland-columbus | True | 6.08 | 0.93 | 8 | 0 |
| gemma2-426k | dallas-austin | True | 3.69 | 1.03 | 4 | 0 |
| gemma2-426k | detroit-lansing | True | 2.09 | 0.91 | 1 | 0 |
| gemma2-426k | houston-austin | True | 5.38 | 0.91 | 6 | 0 |
| gemma2-426k | miami-tallahassee | True | 1.5 | 1.32 | 2 | 0 |
| gemma2-426k | philadelphia-harrisburg | True | -2.62 | -2.52 | 8 | 0 |
| gemma2-426k | portland-salem | True | 6.34 | 0.83 | 8 | 0 |
| gemma2-2.5M | chicago-springfield | True | 3.14 | 1.05 | 3 | 0 |
| gemma2-2.5M | cleveland-columbus | True | 8.62 | 0.85 | 8 | 0 |
| gemma2-2.5M | dallas-austin | True | 8.75 | 0.54 | 8 | 0 |
| gemma2-2.5M | detroit-lansing | True | **-3.44** | -1.95 | 8 | 0 |
| gemma2-2.5M | houston-austin | True | 7.47 | 0.58 | 8 | 0 |
| gemma2-2.5M | miami-tallahassee | True | 1.44 | 1.17 | 2 | 0 |
| gemma2-2.5M | philadelphia-harrisburg | True | 1.19 | 2.45 | 1 | 0 |
| gemma2-2.5M | portland-salem | True | 3.22 | 1.04 | 4 | 0 |
| llama32-524k | chicago-springfield | True | 3.39 | 1.08 | 1 | 0 |
| llama32-524k | cleveland-columbus | True | -1.84 | -3.44 | 8 | 0 |
| llama32-524k | dallas-austin | True | 6.06 | 0.92 | 5 | 0 |
| llama32-524k | detroit-lansing | **False** | -4.75 | -1.74 | 8 | 0 |
| llama32-524k | houston-austin | True | 5.27 | 0.91 | 7 | 0 |
| llama32-524k | miami-tallahassee | True | -6.22 | -0.88 | 8 | 0 |
| llama32-524k | philadelphia-harrisburg | **False** | -8.75 | -0.73 | 8 | 0 |
| llama32-524k | portland-salem | **False** | -3.06 | -2.27 | 5 | 0 |

*Observations.* (i) The reconstruction failure **moves** with capacity: 426k fails
on philadelphia, 2.5M fails on detroit instead — capacity changes *which* prompt is
unrecoverable, not *whether* one is. (ii) 2.5M tends to larger evidence sets at high
range (dallas 8.75/8, houston 7.47/8) → lower faith_norm (0.54, 0.58): more capacity
spreads the behavior over more features, so a budget-8 set recovers a smaller
*fraction* of the larger range. (iii) llama is the weak case: 3/8 not
target-preferred and 4/8 negative range — when a prompt is not even target-preferred
the logit-gap oracle is measuring the wrong thing, so those rows should be excluded
from faithfulness aggregates (a protocol note for Phase 1).

### C.3 Game 1 frozen vs unfrozen, two-hop (`macag_unfrozen/robust_frozen_vs_unfrozen.csv`)

> *Source note (pass 1).* Two unfrozen two-hop reruns exist: `macag_unfrozen/`
> (`normalized` stop — **this table**) and `macag_unfrozen_raw/` (`raw_relative`
> stop; also the source of C.5's Game 2 CSV). The raw-stop rerun gives different
> unfrozen numbers (e.g. miami $E_u$ 15 vs 13, dallas 10 vs 8). Both unfrozen
> reruns used budget 20 / prefilter 30 vs the frozen legs' 8 / 20 (§9.1). All
> columns here are raw scores — fine to read under either attention mode.
> <!-- TODO(pass-2): one matched-protocol re-run; quote the fixed-raw_relative version. -->

| CLT | slug | pref | E_f | E_u | up_f | up_u | suff_f | suff_u | faith_f | faith_u | range_f | range_u |
|---|---|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma2-426k | dallas-austin | T | 4 | 8 | 0 | 2 | 0.0 | 3.88 | 3.91 | 6.59 | 3.81 | 6.94 |
| gemma2-426k | houston-austin | T | 7 | 5 | 0 | 3 | 0.0 | 2.56 | 5.34 | 5.41 | 5.31 | 5.69 |
| gemma2-426k | chicago-springfield | T | 5 | 3 | 2 | 2 | 4.88 | 2.5 | 4.44 | 4.38 | 4.56 | 4.12 |
| gemma2-426k | miami-tallahassee | T | 2 | 13 | 0 | 8 | 0.03 | 9.72 | 1.95 | 10.86 | 1.75 | 15.84 |
| gemma2-426k | detroit-lansing | T | 2 | 2 | 1 | 1 | 3.78 | 6.56 | 3.61 | 4.88 | 2.31 | 4.62 |
| gemma2-426k | portland-salem | T | 8 | 1 | 3 | 1 | 4.34 | 3.66 | 5.14 | 1.64 | 6.34 | 0.94 |
| gemma2-426k | cleveland-columbus | T | 8 | 7 | 1 | 2 | 1.28 | 2.69 | 5.58 | 6.53 | 6.12 | 7.06 |
| gemma2-426k | philadelphia-harrisburg | T | 8 | 2 | 3 | 1 | 4.88 | 3.44 | 6.5 | 2.78 | -2.31 | 1.62 |
| gemma2-2.5M | dallas-austin | T | 8 | 7 | 3 | 2 | 4.31 | 6.38 | 4.72 | 5.22 | 8.75 | 5.5 |
| gemma2-2.5M | houston-austin | T | 8 | 1 | 1 | 0 | 2.97 | 0.09 | 4.33 | 0.89 | 7.47 | 0.12 |
| gemma2-2.5M | chicago-springfield | T | 3 | 17 | 1 | 5 | 3.95 | 0.25 | 3.29 | 6.97 | 3.14 | -2.91 |
| gemma2-2.5M | miami-tallahassee | T | 2 | 14 | 0 | 5 | 0.12 | 5.84 | 1.69 | 7.61 | 1.44 | 9.5 |
| gemma2-2.5M | detroit-lansing | T | 8 | 17 | 1 | 4 | 1.31 | 3.34 | 6.72 | 9.86 | -3.44 | -2.22 |
| gemma2-2.5M | portland-salem | T | 4 | 13 | 1 | 8 | 3.12 | 3.69 | 3.34 | 5.94 | 3.22 | -0.38 |
| gemma2-2.5M | cleveland-columbus | T | 8 | 4 | 3 | 1 | 8.44 | 2.75 | 7.31 | 3.65 | 8.62 | 3.81 |
| gemma2-2.5M | philadelphia-harrisburg | T | 1 | 5 | 0 | 2 | 5.69 | 5.28 | 2.91 | 5.27 | 1.19 | 5.66 |
| llama32-524k | dallas-austin | T | 5 | 1 | 3 | 0 | 4.38 | 0.09 | 5.59 | 2.44 | 6.06 | 2.44 |
| llama32-524k | houston-austin | T | 7 | 2 | 5 | 1 | 6.3 | 1.27 | 4.77 | 2.76 | 5.27 | 2.96 |
| llama32-524k | chicago-springfield | T | 1 | 7 | 0 | 4 | 4.59 | 1.22 | 3.66 | 5.22 | 3.39 | -1.97 |
| llama32-524k | miami-tallahassee | T | 8 | 6 | 4 | 2 | 1.84 | 9.58 | 5.45 | 7.9 | -6.22 | -1.73 |
| llama32-524k | detroit-lansing | F | 8 | 8 | 5 | 4 | 11.62 | 7.66 | 8.28 | 7.52 | -4.75 | -6.34 |
| llama32-524k | portland-salem | F | 5 | 11 | 3 | 8 | 9.19 | 15.09 | 6.95 | 9.73 | -3.06 | -2.91 |
| llama32-524k | cleveland-columbus | T | 8 | 9 | 6 | 7 | 8.41 | 10.94 | 6.35 | 7.48 | -1.84 | -3.81 |
| llama32-524k | philadelphia-harrisburg | F | 8 | 6 | 3 | 3 | 3.5 | 8.02 | 6.41 | 7.05 | -8.75 | -0.05 |

*Observations.* (i) Upstream recruitment is real but **prompt-dependent**: gemma
dallas 0→2, houston 0→3, miami 0→8 gain upstream features when unfrozen, while
others are flat or drop (portland 3→1). Aggregate gemma upstream roughly doubles
(§10.3) but the per-prompt spread is large — argues for more prompts + CIs. (ii)
Unfrozen can **destabilize**: chicago-2.5M range goes 3.14→−2.91 and the set
balloons to 17, portland-2.5M faith 3.34→5.94 but range goes negative — unfreezing
sometimes *creates* a reconstruction failure by collapsing `empty`. (iii) gemma
frozen `suff` is often ~0 (dallas 0.0, houston 0.0, miami 0.03) while faith is
positive: frozen sufficiency (keep-only − empty) is near zero because frozen
attention already reconstructs the behavior from almost nothing — the necessity term
carries the frozen faithfulness. This is the cleanest single illustration of the
"frozen hides features" claim.

### C.4 ACDC Game 1 frozen vs unfrozen (`macag_acdc_unfrozen/acdc_frozen_vs_unfrozen.csv`)

> *Source note (pass 1).* The unfrozen columns were selected with the pre-fix
> (λ-penalized) `raw_relative` stop (§10 provenance box): the `range` columns and
> flip counts are stop-independent and robust; `faith_u` / `E_u` / `up_u` are
> provisional. Unfrozen legs ran budget 20 / prefilter 30 vs frozen 8 / 20.
> <!-- TODO(pass-2): regenerate with the fixed stop and matched budgets. -->

| CLT | task | slug | range_f | range_u | faith_f | faith_u | E_f | E_u | up_f | up_u |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma2-426k | docstring | docstring_01 | 2.5 | 25.44 | 7.06 | 13.62 | 1 | 3 | 0 | 2 |
| gemma2-426k | docstring | docstring_02 | -1.56 | 22.75 | 8.03 | 13.12 | 8 | 2 | 3 | 0 |
| gemma2-426k | docstring | docstring_03 | -8.56 | 23.38 | 3.84 | 14.53 | 8 | 2 | 5 | 1 |
| gemma2-426k | IOI | ioi_01 | -8.12 | 7.25 | 5.69 | 7.31 | 8 | 3 | 7 | 2 |
| gemma2-426k | IOI | ioi_02 | -18.5 | 6.38 | 8.69 | 5.94 | 6 | 4 | 6 | 3 |
| gemma2-426k | IOI | ioi_03 | -12.88 | 17.31 | 6.19 | 12.16 | 8 | 3 | 7 | 2 |
| gemma2-426k | IOI | ioi_04 | -9.88 | 7.83 | 12.31 | 6.88 | 8 | 2 | 8 | 1 |
| gemma2-426k | IOI | ioi_05 | -15.5 | 7.47 | 4.19 | 13.05 | 6 | 7 | 4 | 6 |
| gemma2-426k | IOI | ioi_06 | -6.5 | 0.06 | 8.06 | 4.5 | 8 | 3 | 8 | 2 |
| gemma2-426k | IOI | ioi_07 | -12.25 | 16.12 | 8.38 | 7.84 | 8 | 2 | 8 | 1 |
| gemma2-426k | IOI | ioi_08 | -19.12 | 16.19 | 10.75 | 10.88 | 4 | 3 | 4 | 2 |
| gemma2-426k | IOI | ioi_09 | -24.12 | 3.62 | 4.94 | 7.22 | 8 | 2 | 7 | 1 |
| gemma2-426k | IOI | ioi_10 | -6.75 | -0.25 | 8.75 | 10.44 | 5 | 7 | 5 | 7 |
| gemma2-2.5M | docstring | docstring_01 | -8.0 | 25.19 | 6.34 | 15.03 | 5 | 3 | 2 | 2 |
| gemma2-2.5M | docstring | docstring_02 | -0.25 | 27.0 | 8.44 | 17.06 | 8 | 3 | 4 | 2 |
| gemma2-2.5M | docstring | docstring_03 | -4.19 | 18.41 | 5.16 | 10.61 | 8 | 3 | 2 | 1 |
| gemma2-2.5M | IOI | ioi_01 | -8.25 | 4.19 | 12.5 | 1.81 | 8 | 6 | 8 | 6 |
| gemma2-2.5M | IOI | ioi_02 | -24.0 | -3.5 | 7.12 | 2.94 | 4 | 4 | 3 | 2 |
| gemma2-2.5M | IOI | ioi_03 | -22.12 | 8.62 | 6.44 | 15.31 | 7 | 11 | 7 | 9 |
| gemma2-2.5M | IOI | ioi_04 | -10.75 | 8.25 | 11.5 | 7.03 | 7 | 3 | 7 | 2 |
| gemma2-2.5M | IOI | ioi_05 | -23.25 | 3.88 | 5.69 | 9.38 | 8 | 5 | 7 | 4 |
| gemma2-2.5M | IOI | ioi_06 | -10.44 | 2.38 | 8.78 | 7.69 | 8 | 7 | 7 | 6 |
| gemma2-2.5M | IOI | ioi_07 | -27.5 | 11.19 | 6.75 | 5.97 | 8 | 8 | 8 | 5 |
| gemma2-2.5M | IOI | ioi_08 | -13.75 | 13.5 | 13.81 | 9.06 | 8 | 2 | 8 | 1 |
| gemma2-2.5M | IOI | ioi_09 | -25.62 | 3.75 | 6.12 | 4.72 | 7 | 5 | 7 | 4 |
| gemma2-2.5M | IOI | ioi_10 | -10.5 | 0.75 | 10.94 | 5.72 | 8 | 6 | 8 | 4 |
| llama32-524k | docstring | docstring_01 | 27.28 | 22.44 | 12.81 | 15.16 | 8 | 4 | 5 | 1 |
| llama32-524k | docstring | docstring_02 | 17.66 | 15.5 | 16.2 | 16.22 | 8 | 9 | 4 | 5 |
| llama32-524k | docstring | docstring_03 | 17.05 | 21.19 | 6.88 | 17.41 | 8 | 7 | 7 | 4 |
| llama32-524k | IOI | ioi_01 | 12.5 | 4.31 | 4.8 | 5.34 | 8 | 5 | 6 | 3 |
| llama32-524k | IOI | ioi_02 | 14.88 | 6.64 | 8.91 | 9.23 | 8 | 7 | 8 | 5 |
| llama32-524k | IOI | ioi_03 | 10.58 | 8.12 | 6.52 | 5.03 | 8 | 3 | 7 | 3 |
| llama32-524k | IOI | ioi_04 | 11.5 | 11.38 | 4.66 | 7.87 | 4 | 6 | 4 | 5 |
| llama32-524k | IOI | ioi_05 | 9.78 | 12.12 | 6.62 | 8.33 | 8 | 4 | 7 | 3 |
| llama32-524k | IOI | ioi_06 | 8.58 | 6.78 | 3.9 | 4.14 | 8 | 3 | 8 | 2 |
| llama32-524k | IOI | ioi_07 | 12.81 | 6.17 | 5.59 | 5.99 | 8 | 5 | 8 | 4 |
| llama32-524k | IOI | ioi_08 | 12.44 | 8.52 | 6.54 | 8.01 | 8 | 6 | 4 | 5 |
| llama32-524k | IOI | ioi_09 | 10.25 | 8.47 | 7.59 | 6.92 | 8 | 3 | 8 | 2 |
| llama32-524k | IOI | ioi_10 | 7.22 | 5.48 | 4.64 | 5.59 | 7 | 3 | 7 | 2 |

*Observations.* (i) The gemma flip is dramatic and consistent: every gemma IOI row
has strongly negative `range_f` (−6 to −27.5) → positive `range_u` except
ioi_02-2.5M (−3.5) and the two ioi_10/ioi_06 near-zero cases. The magnitude of the
frozen negativity (e.g. −27.5) is itself a measure of *how* attention-mediated the
task is. (ii) docstring is **less** attention-mediated than IOI even on gemma
(docstring_01-426k already positive frozen at +2.5), consistent with docstring being
a more "local"/feature-carried completion. (iii) llama is positive throughout
(frozen ranges +7 to +27) — its CLT genuinely places IOI behavior in features; this
is the cross-model control that makes the gemma result a *diagnosis* rather than a
universal artifact. (iv) Unfreezing **shrinks** evidence and upstream count on gemma
(e.g. ioi_04 8→2 features, 8→1 upstream) — once attention recomputes, fewer features
are needed, the opposite of the two-hop sweep where unfreezing *recruited* features.
This sign difference between tasks is worth a dedicated paragraph in the paper.

### C.5 Game 2 contrastive, frozen vs unfrozen (`macag_unfrozen_raw/game2_frozen_vs_unfrozen.csv`)

<!-- TODO(pass-2): re-verify after re-run; predates best-iterate tracking (all runs converged in 2 rounds, so low risk). -->
All 24 prompts × {frozen, unfrozen}: **overlap_f = overlap_u = 0.0** and
**shared = 0** in every row. The unique-set sizes (|E_y|, |E_foil|) shift slightly
between frozen/unfrozen (e.g. gemma2-2.5M dallas |E_foil| 8→4, cleveland 8→4) but the
disjointness never breaks. Full row-level numbers are in the CSV. **Do not carry
48/48 into the paper as the headline** — cite MIB **1712/1712**
([J.D17](#jd-numbers-behind-claims-already-in-the-draft)); this 48/48 figure is the
historical two-hop case study only.

### C.6 Oracle cost (from `*/macag_game{1,2}.json` stats)

Two-hop gemma-426k: Game 1 mean **807.5** oracle calls (range reflects budget 8 ×
prefilter 20 + full singleton scan over ~325 candidates), Game 2 mean **1761.25**
(the runs converged after 2 of the configured 4 ABR rounds, two greedy solves per
round). Cache hit counts exceed oracle calls (e.g. one prompt:
762 calls / 838 hits for G1; 1756 / 5596 for G2), i.e. **>50% of probes are served
from cache** — the memoization in [§2.1](macag_framework.md#21-oracle-scoring) is load-bearing. These
are the denominators for the "MACAG vs Shapley cost ratio" claim once Shapley is run
(Phase 2). *Caveat:* these stats predate the per-solve counter reset (`71a2ef6`), so
Game 2 counters on a shared oracle may include residue from the preceding Game 1
solve — re-derive on re-run. <!-- TODO(pass-2): recompute cost stats from the re-run. -->

### C.7 Nonlinear benchmark: matched-protocol baselines + frozen/unfrozen (`results/macag_nonlinear_connected/`)

> **Provisional / appendix-only.** Source directory missing on this filesystem
> ([J.C14](#jc-campaign-status-and-coverage)); do not cite as headline. Tables kept
> for provenance only.
>
> *Source note.* First **post-`71a2ef6`** run (fixed `raw_relative` stop, per-solve
> oracle-stat reset, Game-2 best-iterate) and the first with **real baseline
> numbers**. Single load via `run_macag game1 --freeze-mode both`; both legs matched
> at budget 8 / prefilter 20 / `raw_relative` / `connected=True` (no 8/20-vs-20/30
> confound of C.3–C.5). 60 prompts = 4 task families × 5 prompts × 3 CLTs. `faith_f`
> / `range_f` / `n` are the **frozen** Game-1 leg; `verdict`/`flip` from the
> per-prompt `attention_mediation` block. Baseline columns are faith@budget-8 from
> `baselines.csv`. **Statistics:** means carry 95% bootstrap CIs over prompts
> (10 000 resamples) and Game 1 vs each matched baseline is tested with paired
> Wilcoxon signed-rank (Holm-corrected, rank-biserial $r$); the prompt is the
> resampling unit because the non-Shapley selectors are deterministic per graph.
> Single seed on the CLT/graph; uncertainty is over the 60-prompt sample. Reproduce
> with `scripts/macag_bootstrap_wilcoxon.py`.

**Aggregate baseline head-to-head (n=60, all prompts).** Influence and EAP run at the
graph layer (0 oracle calls); Shapley/Game 1/ACDC use the real ReplacementModel
oracle. ACDC is uncapped (mean $k\approx193$) and is **excluded from the paired tests**
(not budget-matched); all others budget-8.

| selector | faith@8 [95% CI] | $|E|$ (k) | faith/feat [95% CI] | AUC | oracle calls | prec@k | Jaccard | vs G1: Δ̃ | win/60 | $p$ (Holm) | $r$ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| top-k influence | 0.08 [−0.02, 0.20] | 8.0 | 0.010 [−0.003, 0.025] | 0.03 | 0 | 0.004 | 0.002 | +4.95 | 60/60 | <1e‑9 | 1.00 |
| EAP (graph-derived) | 1.15 [0.55, 1.83] | 8.0 | 0.144 [0.065, 0.226] | 0.81 | 0 | 0.206 | 0.128 | +3.72 | 58/60 | <1e‑9 | 0.98 |
| MC Shapley (gold) | 4.72 [4.02, 5.45] | 8.0 | 0.590 [0.500, 0.683] | 3.48 | 32 495 | — | — | +1.00 | 44/60 | 8.2e‑5 | 0.60 |
| **MACAG Game 1** | **5.50 [4.88, 6.15]** | **6.8** | **0.859 [0.747, 0.977]** | **4.28** | **718** | **0.46** | **0.33** | — | — | — | — |
| ACDC (τ-prune) | 10.50 [8.60, 12.46] | 193.4 | 0.067 [0.050, 0.088] | — | 2 612 | — | — | *(not matched)* | — | — | — |

prec@k / Jaccard are vs the gold Shapley top-k; Δ̃ = median paired difference (Game 1
− baseline). **Cost (provisional — do not headline; artifacts missing — [J.C14](#jc-campaign-status-and-coverage)):**
stored table reports Game 1 at **44.7× fewer** oracle
calls than Shapley, 95% CI [43.5, 45.8], cheaper on **60/60** prompts, one-sided
Wilcoxon $p<10^{-9}$. **Faith/feature:** Game 1 0.859 [0.747, 0.977] vs Shapley
0.590 [0.500, 0.683] with
**non-overlapping CIs** — and unlike raw faith it cannot be inflated by spending
features (ACDC's 0.067 at k≈193 is the cautionary case). **Raw faith\@8:** Game 1's
median paired advantage over Shapley is +1.00 ($p=8.2\times10^{-5}$, $r=0.60$) but the
win-rate is **44/60**, so the claim is "higher on most prompts," not "uniformly," and
it is on Game 1's own greedy objective (§11.3). ACDC's raw-faith "wins" (46/60) are at
~28× the feature budget. On the 50 target-preferred prompts every effect strengthens
(Game 1 faith 5.96, fpf 0.938, vs-Shapley Δ̃ +1.08 at $p=9.6\times10^{-5}$, cost 44.8×).

**Verdict mix by task** (per-prompt frozen-vs-unfrozen `attention_mediation.verdict`):

| task | feature_mediated | indeterminate | attention_mediated |
|---|:-:|:-:|:-:|
| boolean_logic | 7 | 7 | 1 |
| negation_polarity | 12 | 3 | 0 |
| context_polysemy | **15** | 0 | 0 |
| hard_semantic_foil | 8 | 2 | 5 |
| **total** | **42** | **12** | **6** |

Only **6/60 range-flip** (frozen `range`<0 → unfrozen ≥0); they are:

| CLT | task | slug | range_f → range_u | faith_f → faith_u | $|E|$_f → $|E|$_u |
|---|---|---|--:|--:|--:|
| gemma2-426k | hard_semantic_foil | sem_03 | −9.00 → 8.00 | 5.88 → 2.56 | 4 → 2 |
| gemma2-426k | hard_semantic_foil | sem_04 | −4.28 → 2.13 | 2.58 → 2.25 | 7 → 5 |
| gemma2-426k | hard_semantic_foil | sem_05 | −5.38 → 1.31 | 4.31 → 6.16 | 2 → 7 |
| gemma2-2.5M | hard_semantic_foil | sem_03 | −11.38 → 2.52 | 7.31 → 1.82 | 5 → 4 |
| gemma2-2.5M | hard_semantic_foil | sem_04 | −4.47 → 3.50 | 4.64 → 3.69 | 6 → 4 |
| llama32-524k | boolean_logic | bool_01 | −1.45 → 0.38 | 3.12 → 2.82 | 8 → 4 |

**Frozen → unfrozen aggregates (per CLT).** Unfreezing **shrinks** evidence and
upstream count on *every* CLT here — opposite of the C.3 two-hop sweep, same as the
C.4 IOI sweep:

| CLT | upstream f→u | $|E|$ f→u | range f→u | faith f→u |
|---|--:|--:|--:|--:|
| gemma2-426k | 3.45 → 2.95 | 5.40 → 4.40 | 2.87 → 5.22 | 4.48 → 3.83 |
| gemma2-2.5M | 3.30 → 2.80 | 5.45 → 4.35 | 3.94 → 5.06 | 5.28 → 3.86 |
| llama32-524k | 3.60 → 2.85 | 5.25 → 4.40 | 6.34 → 4.47 | 5.84 → 4.38 |
| **all** | **3.45 → 2.87** | **5.37 → 4.38** | **4.38 → 4.91** | **5.20 → 4.02** |

**Game 2 (contrastive), ABR vs FP** (`abr_vs_fp.csv`): all 60 converge in 2
iterations under both solvers; **overlap = 0.0 in every row** for both. ABR≈FP:
target-set Jaccard 0.63, foil-set 0.55; target utility 5.50 (ABR) vs 5.45 (FP); mean
$|E_y|$ 6.80/6.92, $|E_{\text{foil}}|$ 7.28/7.02. Combined with the 60 frozen Game-1
verdicts → **180/180 disjoint** on this historical root (extends C.5's 48/48).
**Cite MIB 1712/1712 instead** ([J.D17](#jd-numbers-behind-claims-already-in-the-draft));
this subsection is provisional ([J.C14](#jc-campaign-status-and-coverage)).

**Full per-prompt table** (frozen Game-1 + baseline faith@8; ✱ = range-flip;
faith/feat best in **bold** is Game 1; full 40-column data in `baselines.csv`):

| CLT | task | slug | pref | faith_f | range_f | n | verdict | flip | infl | eap | shap | **G1** | acdc (k) |
|---|---|---|:-:|--:|--:|--:|---|:-:|--:|--:|--:|--:|--:|
| gemma2-426k | bool | bool_01 | F | 2.22 | 2.48 | 5 | feat |  | 0.32 | 1.58 | 4.01 | **2.61** | 10.39 (221) |
| gemma2-426k | bool | bool_02 | F | 4.76 | 3.78 | 6 | feat |  | 0.03 | 1.86 | 4.48 | **5.12** | 4.23 (8) |
| gemma2-426k | bool | bool_03 | T | 5.62 | 0 | 7 | indet |  | 0.03 | 0.72 | 3.66 | **5.75** | 3.61 (213) |
| gemma2-426k | bool | bool_04 | T | 2.49 | −2.52 | 3 | indet |  | 0.30 | −1.17 | 2.92 | **2.61** | 1.44 (206) |
| gemma2-426k | bool | bool_05 | F | 4.33 | 3.16 | 4 | feat |  | 0.17 | 1.88 | 3.11 | **4.23** | 9 (217) |
| gemma2-426k | neg | neg_01 | T | 6.09 | 3.31 | 7 | feat |  | 0.25 | 0.56 | 3.16 | **6.09** | 7.20 (162) |
| gemma2-426k | neg | neg_02 | T | 4.44 | 2.88 | 2 | feat |  | −0.31 | 3.22 | 3.44 | **4.88** | 7.59 (140) |
| gemma2-426k | neg | neg_03 | T | 7.19 | 2.38 | 7 | feat |  | −0.23 | 0.22 | 6.03 | **7.25** | 9.09 (119) |
| gemma2-426k | neg | neg_04 | F | 2.89 | −8.47 | 8 | indet |  | −0.48 | −3.25 | 0.25 | **2.38** | −0.38 (185) |
| gemma2-426k | neg | neg_05 | T | 7.23 | 3.33 | 4 | feat |  | −1.51 | −1.35 | 5.65 | **7.87** | 9.18 (203) |
| gemma2-426k | poly | poly_01 | T | 2.81 | 7.38 | 2 | feat |  | 0.19 | 1.78 | 5.28 | **7.12** | 12.84 (187) |
| gemma2-426k | poly | poly_02 | T | 3.88 | 24.25 | 8 | feat |  | 0.25 | 6.50 | 11.19 | **3.70** | 28.41 (193) |
| gemma2-426k | poly | poly_03 | T | 5.22 | 5.25 | 2 | feat |  | −0.09 | −1.28 | 3.47 | **5.66** | 14.67 (260) |
| gemma2-426k | poly | poly_04 | T | 3.81 | 12.38 | 8 | feat |  | 0.19 | 0.91 | 1.69 | **3.81** | 17.50 (298) |
| gemma2-426k | poly | poly_05 | T | 4.66 | 8.69 | 6 | feat |  | 0 | 2.53 | 6.75 | **4.78** | 22.09 (244) |
| gemma2-426k | sem | sem_01 | T | 4.88 | 2.75 | 8 | feat |  | 0.17 | −1.48 | 5.11 | **5.02** | 9.33 (164) |
| gemma2-426k | sem | sem_02 | T | 4.23 | 5.12 | 8 | feat |  | −0.12 | 0.56 | 3.66 | **4.09** | 9.53 (248) |
| gemma2-426k | sem | sem_03 | T | 5.88 | −9 | 4 | attn | ✱ | 0.84 | 0.44 | 7.25 | **7.56** | 2.12 (187) |
| gemma2-426k | sem | sem_04 | T | 2.58 | −4.28 | 7 | attn | ✱ | −0.17 | −0.56 | 2.81 | **3.09** | 1.86 (263) |
| gemma2-426k | sem | sem_05 | T | 4.31 | −5.38 | 2 | attn | ✱ | 0.19 | −0.50 | 2.81 | **5.62** | −0.59 (280) |
| gemma2-2.5M | bool | bool_01 | F | 2.06 | 2.81 | 8 | feat |  | −0.16 | 1.81 | 3.25 | **2** | 7.97 (186) |
| gemma2-2.5M | bool | bool_02 | F | 3.26 | 5.50 | 8 | feat |  | −0.02 | 2.75 | 4.28 | **1.94** | 9.09 (193) |
| gemma2-2.5M | bool | bool_03 | T | 1.62 | −5.62 | 3 | indet |  | 0.20 | −0.72 | 0.44 | **1.88** | −1.66 (190) |
| gemma2-2.5M | bool | bool_04 | T | 4.09 | −3.62 | 2 | indet |  | 0.12 | −1.84 | 3.78 | **4.31** | 0.47 (182) |
| gemma2-2.5M | bool | bool_05 | F | 4.31 | 7.25 | 5 | feat |  | 0.05 | 2.97 | 4.33 | **4.97** | 9.61 (210) |
| gemma2-2.5M | neg | neg_01 | T | 6.25 | 8.25 | 2 | feat |  | 0.03 | 2.30 | 4.88 | **6.86** | 11.05 (154) |
| gemma2-2.5M | neg | neg_02 | T | 5.78 | 10.69 | 2 | indet |  | 0.03 | 5.22 | 5.66 | **6.47** | 12.88 (151) |
| gemma2-2.5M | neg | neg_03 | T | 8.81 | 0.88 | 8 | feat |  | −0.06 | 0.31 | 2.38 | **7.44** | 6.09 (151) |
| gemma2-2.5M | neg | neg_04 | F | 3.45 | −3.97 | 4 | indet |  | −0.16 | −4.20 | 1.55 | **3.18** | 1.02 (229) |
| gemma2-2.5M | neg | neg_05 | T | 9.70 | 4.22 | 8 | feat |  | 0.50 | −1.83 | 6.16 | **9.03** | 10.16 (217) |
| gemma2-2.5M | poly | poly_01 | T | 4.16 | 7.31 | 7 | feat |  | 0.25 | 1.09 | 4.75 | **3.94** | 11.88 (195) |
| gemma2-2.5M | poly | poly_02 | T | 2.58 | 16.50 | 6 | feat |  | 0.06 | −0.99 | −0.02 | **2.32** | 22.73 (255) |
| gemma2-2.5M | poly | poly_03 | T | 6.16 | 6.31 | 3 | feat |  | −0.34 | 0.19 | 3.09 | **6.62** | 12.19 (294) |
| gemma2-2.5M | poly | poly_04 | T | 3.41 | 12.38 | 8 | feat |  | 0.09 | −0.72 | 1.69 | **3.56** | 16.66 (314) |
| gemma2-2.5M | poly | poly_05 | T | 11.22 | 16.38 | 8 | feat |  | 0.03 | 3.22 | 11.06 | **11.06** | 24.16 (252) |
| gemma2-2.5M | sem | sem_01 | T | 7.96 | 8.98 | 6 | feat |  | −0.13 | −0.74 | 7.09 | **8.57** | 14.99 (253) |
| gemma2-2.5M | sem | sem_02 | T | 4.69 | 3.50 | 7 | feat |  | −0.19 | −1.33 | 4.80 | **4.80** | 7.32 (252) |
| gemma2-2.5M | sem | sem_03 | T | 7.31 | −11.38 | 5 | attn | ✱ | −0.19 | 1.06 | 5.19 | **7.12** | −1.91 (234) |
| gemma2-2.5M | sem | sem_04 | T | 4.64 | −4.47 | 6 | attn | ✱ | 0.08 | 0.36 | 1.55 | **3.48** | −0.95 (282) |
| gemma2-2.5M | sem | sem_05 | T | 4.19 | −3.19 | 3 | indet |  | −0.03 | 1.50 | 2.28 | **4.22** | 0.03 (252) |
| llama32-524k | bool | bool_01 | F | 3.12 | −1.45 | 8 | attn | ✱ | −0.11 | −0.58 | 2 | **3.05** | 3.36 (143) |
| llama32-524k | bool | bool_02 | T | 4.47 | 2.28 | 5 | feat |  | 0.64 | 0.56 | 3.59 | **4.48** | 4.66 (162) |
| llama32-524k | bool | bool_03 | T | 3.52 | 2.27 | 3 | indet |  | −0.23 | 1.86 | 3.84 | **4.39** | 6.47 (145) |
| llama32-524k | bool | bool_04 | T | 3.64 | 1.81 | 2 | indet |  | 0.65 | 0.51 | 2.81 | **3.76** | 6.98 (125) |
| llama32-524k | bool | bool_05 | F | 2.09 | −1.38 | 6 | indet |  | 0.40 | −0.36 | 1.31 | **2.58** | 4.70 (135) |
| llama32-524k | neg | neg_01 | T | 5.71 | 7.41 | 4 | feat |  | −0.21 | 4.61 | 6.49 | **6.80** | 14.41 (94) |
| llama32-524k | neg | neg_02 | T | 6.17 | 4.56 | 5 | feat |  | 2.14 | 4.55 | 6.49 | **6.64** | 9.30 (87) |
| llama32-524k | neg | neg_03 | T | 11.94 | 7.50 | 8 | feat |  | 0.76 | 9.03 | 10.12 | **11.59** | 14.56 (81) |
| llama32-524k | neg | neg_04 | T | 7.38 | 7.06 | 5 | feat |  | 0.23 | 4.66 | 7.28 | **7.48** | 11.81 (132) |
| llama32-524k | neg | neg_05 | T | 5.80 | 5.62 | 4 | feat |  | 0.36 | 3.23 | 6.97 | **6.12** | 10.22 (126) |
| llama32-524k | poly | poly_01 | T | 4.70 | 6.91 | 7 | feat |  | −0.05 | −0.22 | 3.33 | **5.38** | 18.56 (158) |
| llama32-524k | poly | poly_02 | T | 9.06 | 12.94 | 8 | feat |  | 0.02 | −0.18 | 8.87 | **10.20** | 21.73 (180) |
| llama32-524k | poly | poly_03 | T | 10.31 | 15.25 | 3 | feat |  | −0.03 | 8.06 | 12.87 | **12.55** | 28.80 (235) |
| llama32-524k | poly | poly_04 | T | 2.29 | 5.70 | 8 | feat |  | −0.14 | 1.25 | 2.98 | **2.11** | 21.25 (255) |
| llama32-524k | poly | poly_05 | T | 3.69 | 8.54 | 5 | feat |  | 0.22 | 0.12 | 1.27 | **4.37** | 20.59 (204) |
| llama32-524k | sem | sem_01 | T | 7.14 | 10.92 | 4 | feat |  | 0.88 | −0.15 | 7.35 | **9.01** | 19.35 (146) |
| llama32-524k | sem | sem_02 | T | 8.62 | 12.80 | 7 | feat |  | −0.55 | 5.95 | 8.20 | **8.80** | 21.25 (190) |
| llama32-524k | sem | sem_03 | T | 6.32 | 9.81 | 5 | indet |  | 0.22 | 2.05 | 11.23 | **6.94** | 16.17 (167) |
| llama32-524k | sem | sem_04 | T | 2.75 | 2.38 | 5 | feat |  | −0.25 | 0.27 | 2.62 | **2.62** | 8.75 (232) |
| llama32-524k | sem | sem_05 | T | 8.11 | 5.88 | 3 | feat |  | −0.25 | 0.41 | 8.50 | **8.16** | 13.94 (162) |

*Observations.* (i) **Game 1 is the efficient frontier.** Across the 60 rows its
faith\@8 tracks or beats Shapley-gold at a fraction of the cost, and it dominates
faith-per-feature; ACDC's larger raw faith always comes with a 100–300-feature set
(the `acdc (k)` column), so it is not a budget-matched competitor — it is the
"spend everything" ceiling. (ii) **Influence ≈ noise** (column `infl` hovers near 0,
often negative), the clearest possible answer to A.3's "is search needed?" — yes.
(iii) **`context_polysemy` is the clean feature-mediated family** (15/15, no flips, no
indeterminates) — the bank/bat/spring polysemy is carried by features, exactly the
nonlinear-context case the benchmark targets. (iv) **`hard_semantic_foil` on gemma is
where attention mediates** (5 of 6 flips); the magnitude of frozen negativity (sem_03:
−9 to −11) again measures *how* attention-mediated a prompt is, mirroring §10.4. (v)
**`boolean_logic` is the soft spot**: 7/15 indeterminate and the home of 5 of the 10
not-target-preferred (`pref=F`) prompts — when the model does not prefer the target
the logit-gap oracle is ill-posed, so those rows should be excluded from faithfulness
aggregates (the Phase-1 protocol note, now reproduced on a second benchmark).

---

## Appendix J: Code and Campaign Audit (2026-07-18)

Answers to `macag/docs/code_questions.md` List 1, verified against the code and
against the live campaign artifacts under `/gscratch/ssuresh/macag_mib_*`.
Every answer cites the file/line or artifact it was read from. Where an answer
contradicts a claim elsewhere in this document, the stale claim is flagged
**[STALE]** and should be corrected before submission.

**Audit scope.** Code at branch `claude/silly-feistel`. Campaign artifacts as of
2026-07-18 18:33 UTC, with three SLURM jobs (8250/8251/8252) still **RUNNING** —
counts below are a snapshot of an unfinished campaign, not a final tally.

### J.A Baselines and the shared characteristic function

**A1 — EAP and top-k influence ranking basis: graph-derived only.**
Neither baseline touches the oracle. `select_top_influence` reads the node's
`influence` metadata (`macag/baselines/influence.py:44`); `select_top_eap`
propagates the graph's own edge weights (`macag/baselines/eap.py:154`, via
`compute_eap_node_scores`). Confirmed empirically: `selection_stats.oracle_calls
== 0` for both methods in every `macag_baselines.json` in the campaign.

Ranking under a shared $v$ therefore does **not** hold. What holds is that every
method's selected prefixes are *scored* under one $v$ — `_evaluate_prefixes`
(`macag/cli/run_baselines.py:191-209`) applies the same
`compute_faithfulness_metrics` at the same `alpha` to every method's `ranking[:k]`.

> **Resolution of the open NOTE in §3.6 (List 2, item 4): the weak form is the
> only defensible claim.** Use: *"every selected set is evaluated under one
> characteristic function $v$ at one $\alpha$."* Do **not** write "all methods
> rank under one $v$" — that is false for influence and EAP, which is precisely
> the intended contrast (they pay zero interventions; MACAG pays real ones).

**A2 — ACDC's criterion: the same oracle score, same $\alpha$.**
`acdc_prune` prunes on `coalition_value(oracle, target, ..., alpha)`
(`macag/baselines/acdc_prune.py:78,84`) — the identical $\alpha$-mixed
keep_only/remove faithfulness the games optimize, not KL. Caveat: the attention
convention is whatever the oracle was constructed with, and the baselines pass
runs a **single** convention (see B8).

**A3 — Gold estimators: identical oracle and $\alpha$, but no shared cache, one
convention, and not executed.** `estimate_shapley` / `estimate_banzhaf` call the
same `coalition_value` (`macag/baselines/shapley_select.py:106,119,128,166,173`).
Memoization is deliberately **not** shared across methods: `_run_selection` calls
`oracle.clear_cache()` and `oracle.reset_stats()` before every selector
(`run_baselines.py:130`) so per-method oracle cost is honest. Only one attention
convention is ever run. **Neither estimator ran in the MIB campaign** — see A5/C11.

**A4 — Gold selection rule: top-$k$ prefix, and Banzhaf *can* select.**
The gold set at budget $k$ is exactly `ranking[:k]` of the estimated ranking
(`run_baselines.py:203`). Banzhaf is not merely reported: it is a first-class
`KNOWN_METHOD` (`run_baselines.py:58`) that produces a ranking through the same
`select_top_shapley` path, and `_comparison_block` **falls back to Banzhaf as the
gold reference when Shapley is absent** (`run_baselines.py:236`). It is excluded
from `DEFAULT_METHODS` but not from selection by construction.

> **List 2, item 8:** if the paper's official line is "Banzhaf is diagnostic
> credit only, never selection," that is a statement about the *run
> configuration*, not the code. Either state it as such, or drop the Banzhaf
> fallback at `run_baselines.py:236`.

**A5 — Exact baseline set: implemented ≠ executed.**
*Implemented:* `influence`, `eap`, `shapley`, `banzhaf`, `game1`, `acdc`
(`run_baselines.py:58`), **plus** exact best-subset brute force (B3.2,
`macag/baselines/bruteforce.py`, via `--bruteforce-k`).
*Executed in the MIB campaign:* `["influence", "eap", "game1", "acdc"]` only —
read from `params.methods` in every `macag_baselines.json`.
**No Shapley, no Banzhaf, no brute force ran.** The paper must either report the
gold pass from a completed run or explicitly exclude it.

### J.B Protocol integrity

**B6 — Connectivity: on everywhere.** `params.connected: true` in every
`macag_game1.json` and `params.game1_connected: true` in every
`macag_baselines.json` in the campaign. It is the default in both CLIs; the
`--no-connected` escape hatch (`run_baselines.py:327`) is used in no config.

**B7 — Error nodes: never ablated, enforced by construction.**
Candidate extraction admits only `feature_type == "cross layer transcoder"`
(`macag/factories/replacement_model.py:29-31,216`), and the scorer can only
ablate nodes in `_all_nodes ⊆ node_to_intervention.keys()`
(`macag/scoring.py:492-509`). The `include_error_nodes=True` opt-in does not
enable ablation — it **raises** (`replacement_model.py:190-205`), because error
nodes carry a sentinel `feature == -1` and have no ablatable index. KL rescoring
rebuilds the oracle through the same factory (`macag/kl_rescore.py:70`) and so
inherits the same guard. This guardrail is strictly stronger than the doc claims.

**B8 — Both attention conventions: Game 1 only. [STALE]**
`FREEZE_MODE` defaults to `both` (`scripts/run_macag_mib.sh:31`) and every
`macag_game1.json` carries `freeze_mode: "both"` with populated `frozen` and
`unfrozen` legs. **Game 2, the baselines pass, and KL rescoring each run one
convention only** — `rescore_run_dir` reads the stored `freeze_attention` for
Game 2 and baselines (`macag/kl_rescore.py:341,348`) and only splits legs for
Game 1 (`:322-330`). Any sentence asserting that *every* headline experiment runs
both conventions is false; only the Game 1 / diagnostic axis does.

**B9 — What the seed controls: MC sampling only.**
The seed feeds `random.Random(seed)` in `estimate_shapley` / `estimate_banzhaf`
(`shapley_select.py:103,163`) and nothing else. Game 1's greedy is fully
deterministic: ties break on `_sort_key` (`macag/games/game1_min_faithful.py:235`),
the prefilter ranking is always computed in full and sorted deterministically
(`:77`), and prompt subsampling is fixed by the manifest.

> Confirmed: **Game 1's selection path is seed-invariant.** Cross-seed
> selected-set stability is therefore trivially 1.0 and **must not be reported as
> a result.** (Consistent with `macag_mib_seed1/` and `seed2/` containing no
> per-prompt artifacts — re-running Game 1 under a new seed would reproduce
> seed 0 exactly.)

**B10 — Hyperparameters as run (uniform across all cells; nothing swept).**
Configured at `scripts/run_macag_pipeline.sh:58-97`, confirmed against
`params` blocks in the artifacts:

| Parameter | As run | Note |
|---|---|---|
| `alpha` | 0.5 | |
| `lambda` | **0.02** | not the CLI default 0.01 |
| `budget` $B$ | 8 | |
| `min_gain` | 0.0 | |
| `prefilter_top_k` | 20 | out of ~490–750 candidates |
| `faithfulness_eps` | 0.1 | |
| `stop_metric` | `raw_relative` | denominator-free; correct given B/D16 |
| `connected` | `true` | |
| `acdc_taus` | 0.001–0.5 (6 pts) | `acdc_target_k = -1` (budget-matched) |
| `shapley_permutations` / `seed` | 64 / 0 | **configured but never exercised** |

### J.C Campaign status and coverage

**C11 — Which cells completed. The campaign is genuinely two-family, but single-seed
and incomplete.** Snapshot 2026-07-18 18:33 UTC (`logs/mib_monitor/counts.json`),
counted by presence of `macag_game1.json`:

| CLT | Task | Game 1 | Game 2 (abr/fp) | Baselines | KL | Gold |
|---|---|---:|---:|---:|---:|---:|
| gemma2-426k | ioi | 500 / 500 | 500 | 500 | 498 | 0 |
| gemma2-426k | mcqa | 50 / 50 | 50 | 49 | 49 | 0 |
| gemma2-426k | arc_easy | **11 / 211** | 11 | 0 | 0 | 0 |
| gemma2-2.5M | ioi | 500 / 500 | 500 | 500 | **328** | 0 |
| gemma2-2.5M | mcqa | 35 / 38 | 31 | 30 | 30 | 0 |
| llama32-524k | ioi | 500 / 500 | 500 | 500 | 500 | 0 |
| llama32-524k | mcqa | 50 / 50 | 50 | 50 | 50 | 0 |
| llama32-524k | arc_easy | **70 / 570** | 69 | 63 | 63 | 0 |

- **Llama-3.2 is included and is complete on ioi (500/500) and mcqa (50/50).**
  The two-family case study is supportable **on ioi and mcqa**; it is not
  supportable on arc_easy (12% and 5% complete).
- **Only seed 0 has artifacts.** `macag_mib_seed1/` and `seed2/` contain launcher
  logs and no runs. Given B9 this costs nothing for Game 1, but it means no
  multi-seed evidence exists for anything stochastic.
- **The gold pass produced zero outputs.** No `macag_baselines_shapley.json`
  exists anywhere; the only trace is
  `macag_mib_llama/macag_mib_seed0/status.shapley.w0.txt`, containing a single
  line: `FAIL llama32-524k/mib_llama3_arc_easy_0000`.
- `gemma2-426k / arc_easy` is failing, not merely slow: 211 prompt directories
  were created, 11 produced Game 1 output, and the hour-over-hour delta is +0.

**C12 — Prompt counts as run. [STALE]** The planned "560 per task, 50-prompt gold
subset" does not match any cell. Actual manifest sizes: **ioi 500**, **mcqa 50**
(gemma2-2.5M: 38), **arc_easy 211** (gemma2) / **570** (llama3). The gold subset
is 0 (C11).

**C15 — Compute / Llama routing / result roots: several §9.5 claims are [STALE].**
Verified against `macag/data/mib_benchmark_prompts.json`,
`scripts/slurm/submit_macag_mib_*.sh`, and `/gscratch/ssuresh/macag_mib_*`:

| Stale claim (older §9.5 / notes) | As-run truth |
|---|---|
| "`llama32-524k` workers unused — no MIB prompts route to it" | Manifest has **500+50+570** `mib_model=llama3` prompts → `llama32-524k`; live job `macag-mib-llama` writes `…/macag_mib_llama` |
| "MIB campaign is gemma-only" / Llama "absent from this campaign" | Three-CLT campaign; Llama complete on IOI+MCQA (C11) |
| "All work on a single GH200 (96 GB)" with sequential CLT groups | **Three concurrent 2×H200 jobs**, one CLT each (`submit_all_mib.sh`) |
| Worker defaults 8 / 3 / 4 unused | Defaults **12 / 6 / 8** shards with `MAX_WORKERS_PER_GPU` 6 / 8 / 8 |
| Roots at `results/macag_mib_seed{0,1,2}/` | Split gscratch roots `macag_mib_{h200,gemma25m,llama}/macag_mib_seed*` |
| Export snippet `--models gemma2` only | Live JSON includes **gemma2 and llama3**; rebuild with both |

Corrected prose: [§9.5.1](#951-prompts), [§9.5.4](#954-three-seed-design-and-orchestration),
[§9.5.5](#955-compute). When editing further, prefer the as-run table in §9.5.5 over
any GH200-era paragraph that still mentions "no MIB prompts route to" Llama.

**C13 — InterpBench: nothing ran.** `scripts/run_interpbench_benchmark.sh` and
`macag/eval/gold_circuits.py` exist, but no InterpBench artifact is present
anywhere on this filesystem. The component→feature mapping and the
component-level Shapley configuration are therefore **unvalidated by execution**.
Phase 4 remains not started, as §12.3 already states.

**C14 — Nonlinear benchmark: results not present in this checkout.**
`results/macag_nonlinear_connected/` — the source cited for §10.7 and Appendix
C.7 — does not exist on this filesystem, nor do `macag_sweep/`,
`macag_clt_compare/`, `macag_unfrozen/`, or `macag_acdc_unfrozen/`. The only
directory under `results/` is `ioi_seed0_analysis/`. The §10.7/C.7 tables cannot
currently be regenerated or re-verified; treat them as **provisional pending
artifact recovery or re-run** before they carry a headline claim.

### J.D Numbers behind claims already in the draft

**D15 — Provenance of the headline numbers: correctly scoped, currently
unverifiable.** "44.7× fewer oracle calls," "≈760 vs ≈34k calls," and "60/60
disjoint" all derive from the **nonlinear benchmark, $n=60$** (this document
already scopes them correctly at §1 line 133 and §10.7). They are **not** MIB
campaign numbers. Because the underlying artifacts are missing (C14), none can be
re-derived today.

> The intro's "two-family case study" and the "60/60" figure describe **two
> different experiments.** Reading them in one paragraph implies the 60/60 result
> came from the Gemma/Llama campaign; it did not. Separate them.
>
> Note the disjointness claim is now *far* better supported than 60/60: across
> the MIB campaign, `overlap_rate == 0.000` on **1712/1712** completed Game 2
> runs spanning three CLTs and three tasks (D17). This is the single most robust
> result in the project and should be re-quoted at the larger $n$.

**D16 — Recoverable range per cell: collapsed and negative on Gemma-2 IOI
(frozen).** Mean `recoverable_range` over completed Game 1 legs:

| CLT | Task | frozen mean | % ≤ 0 | unfrozen mean | % ≤ 0 |
|---|---|---:|---:|---:|---:|
| gemma2-426k | ioi | **−10.54** | **95%** | +6.63 | 15% |
| gemma2-426k | mcqa | +7.13 | 8% | +15.91 | 0% |
| gemma2-426k | arc_easy | +3.56 | 27% | +14.54 | 9% |
| gemma2-2.5M | ioi | **−13.62** | **96%** | +5.84 | 7% |
| gemma2-2.5M | mcqa | +7.67 | 0% | +14.78 | 0% |
| llama32-524k | ioi | +9.58 | 1% | +5.58 | 4% |
| llama32-524k | mcqa | +26.55 | 0% | +24.97 | 0% |
| llama32-524k | arc_easy | +23.58 | 0% | +25.08 | 0% |

Independently corroborated by `results/ioi_seed0_analysis/ioi_aggregate.json`:
`range_frozen` mean −10.54, 95% CI **[−11.16, −9.94]** (entirely negative), and
`recon_fail_among_pref` = **474/500**.

Two readings are both correct and must be stated together:

1. **This is the attention-mediation signal, not an artifact.** The verdict rule
   is exactly `range_frozen < 0 and range_unfrozen >= 0 → attention_mediated`
   (`macag/utils/attention_mediation.py:157-162`). A collapsed frozen range on
   Gemma IOI *is* the phenomenon the paper is reporting.
2. **It nevertheless voids every normalized metric on those cells.** When
   `recoverable_range ≤ 0`, `faithfulness_normalized`, `sufficiency_normalized`,
   and `necessity_normalized` all degenerate to 0 (`macag/scoring.py`, and see
   the worked example in §2.3). Any *faithfulness comparison* — MACAG vs
   baselines, capacity comparisons, faith-per-feature — computed on Gemma-2 IOI
   **frozen** is a comparison against a vanishing or negative denominator and
   must be dropped, or restricted to raw (un-normalized) faithfulness.

   The choice of `stop_metric = raw_relative` (B10) already anticipates this and
   is the right call; the residual exposure is in *reporting*, not in selection.

Cells safe for normalized reporting under both conventions: all Llama-3.2 cells,
Gemma mcqa, Gemma arc_easy (frozen 27% ≤ 0 — caveat).

**D17 — Game 2 solver agreement: the two solvers essentially never agree.**
Comparing `macag_game2_abr.json` against `macag_game2_fp.json` on identical
prompts (agreement = both $E_y$ and $E_{\text{foil}}$ identical):

| CLT | Task | abr == fp |
|---|---|---|
| gemma2-426k | ioi | 3 / 500 (1%) |
| gemma2-426k | mcqa | 0 / 50 (0%) |
| gemma2-426k | arc_easy | 0 / 11 (0%) |
| gemma2-2.5M | ioi | 3 / 500 (1%) |
| gemma2-2.5M | mcqa | 0 / 31 (0%) |
| llama32-524k | ioi | 3 / 500 (1%) |
| llama32-524k | mcqa | 0 / 50 (0%) |
| llama32-524k | arc_easy | 0 / 69 (0%) |

`best_iteration == 1` on **every** run in all three CLTs. Per
`macag/games/game2_contrastive.py:356,358,423`, `best_iteration` is initialized
to 0 (= the initial empty allocation won) and set to `iteration` when a round
improves combined utility; the loop is 1-indexed. So `best_iteration == 1` means
**round 1 produced the winning allocation and no later round improved on it** —
both solvers do search, improve once, and then converge immediately. Meanwhile
`overlap_rate == 0.000` on **1712/1712** completed Game 2 runs.

> **Reading: multiple equilibria, not solver failure.** §3.4's PSNE theorem
> guarantees *existence*, not *uniqueness*. Two solvers that each converge in one
> round, both to a fully disjoint allocation, but to *different* disjoint
> allocations on ~99% of prompts, is the signature of a game with many pure
> equilibria — not of a solver that fails to search. The one-round convergence is
> consistent with this: with `overlap_rate == 0` achievable immediately, the
> first best-response already lands on an equilibrium.
>
> Consequences for the draft, in decreasing confidence:
> - **Reportable now:** disjointness is solver-invariant and holds on 1712/1712
>   runs across three CLTs and three tasks. This is the robust Game 2 finding.
> - **Must be stated, not hidden:** the *identity* of the selected pair is
>   solver-dependent (~1% agreement). Game 2 selects *an* equilibrium, not *the*
>   equilibrium.
> - **Open, worth one experiment:** confirm the multiplicity reading directly —
>   e.g. verify both solvers' outputs are individually stable under
>   best-response, or count distinct equilibria on a few prompts. Cheap, and it
>   converts the caveat into a result.
>
> Do **not** report abr-vs-fp agreement as a robustness check.

**D18 — KL rescoring coverage: broad, with two holes.**
`rescore_run_dir` (`macag/kl_rescore.py:287-358`) covers Game 1 (both legs), Game
2 (target and foil), and **every** method present in `macag_baselines.json`
including ACDC's `best_by_size` (`:177-215`). Gold is not covered because gold
never ran. Per-cell coverage is in C11 — note gemma2-2.5M ioi is only 328/500.

The **alternate-foil variant is implemented but has no entrypoint and produced no
outputs.** `altfoil_spec` exists and is unit-tested
(`tests/test_macag_kl_scoring.py:187-219`), but the module its own docstring
points to — `macag.cli.rescore_altfoil` (`macag/kl_rescore.py:7`) — **does not
exist**; `macag/cli/` contains only `rescore_kl.py`. The §11.3 "single foil"
threat is therefore **not yet answered by any run.**

**D19 — Statistics: implemented as described.**
`scripts/macag_bootstrap_wilcoxon.py` implements paired Wilcoxon signed-rank with
`zero_method="pratt"` (`:117`) and a step-down `holm_correct` (`:121-144`).
The correction family is **per metric family, across the non-`game1` methods
compared against `game1`** (`:165-192`) — i.e. with the executed method set (A5)
the family size is **3** (influence, eap, acdc), not 5. If the gold pass is later
added, the family becomes 5 and every reported adjusted $p$ must be recomputed.

**D20 — Oracle-call accounting: measured, not derived.**
Counted in `ScoringOracle._oracle_calls` (`macag/scoring.py:120`) and surfaced
per component: per-method `selection_stats.oracle_calls` with a cleared cache
(`run_baselines.py:130,184`), per-Game-1-leg `stats.oracle_calls`
(`game1_min_faithful.py:292`), and the shared evaluation pass separately as
`stats.evaluation_oracle_calls` (`run_baselines.py:604`). Representative measured
values from one gemma2-426k IOI prompt (488 candidates): influence 0, EAP 0,
Game 1 1204, ACDC 3542, evaluation 5708.

> Game 1's measured cost here (~1200 calls with `prefilter_top_k=20`) is larger
> than the "≈760" figure quoted in §1 and §9.5, which came from the nonlinear
> benchmark at a different candidate count. Do not mix the two.

**D21 — Diagnostic: fully implemented in-code, per prompt.**
`compute_attention_mediation_diagnostic`
(`macag/utils/attention_mediation.py:136-197`) applies a **strict zero-threshold**
rule (quoted in B/D16) and writes a complete per-prompt block — both raw ranges,
`range_flip`, `reverse_flip`, `verdict`, both evidence sets, their Jaccard, and
upstream/early-layer counts — into `macag_game1.json` under `attention_mediation`
(`macag/cli/run_macag.py:437`). Classification is **not** deferred to downstream
analysis. Raw ranges are retained alongside the verdict so a confidence band can
be applied later without re-running.

Verdict distribution on IOI (500 prompts each) — the cross-model contrast, now at
$n=500$ per model rather than the 8 prompts of §10.3:

| CLT | attention_mediated | feature_mediated | indeterminate |
|---|---:|---:|---:|
| gemma2-426k | **402 (80%)** | 26 | 72 |
| llama32-524k | 4 | **476 (95%)** | 20 |

> This is the strongest result the campaign has produced and it directly supports
> framing **B** in §11.2. It is currently under-claimed: §10.3/§10.4 report it at
> $n=8$.

### J.E Consequences for the draft

Items requiring edits before submission, in priority order:

1. **§10.5 / Game 2 — scope the claim.** Report disjointness (1712/1712,
   solver-invariant); state that the selected *pair* is solver-dependent (~1%
   abr/fp agreement, D17). Optionally run the cheap equilibrium-multiplicity
   check before drafting.
2. **§3.6 + Experimental Setup — use the weak shared-$v$ sentence** (A1). This
   closes List 2, item 4.
3. **Anywhere claiming both attention conventions throughout — restrict to Game 1**
   (B8).
4. **Gemma-2 IOI frozen normalized metrics — drop or restrict to raw** (D16).
5. **§1 intro — separate the two experiments.** "Two-family case study" (MIB, real,
   ioi+mcqa only) is not the source of "60/60" (nonlinear benchmark, $n=60$)
   (D15). Re-quote disjointness at $n=1712$.
6. **§9.5 prompt counts — 500/50/211/570, not 560** (C12).
7. **Baseline set — report as {influence, EAP, budget-matched ACDC}**, with
   Shapley/Banzhaf gold explicitly marked *not yet run* (A5), and Holm family
   size 3 (D19).
8. **§10.3/§10.4 — upgrade to $n=500$ per model** (D21).
9. **§11.3 single-foil threat — still unanswered**; alt-foil has no runnable CLI
   (D18).
10. **Never report cross-seed selected-set stability for Game 1** (B9).
