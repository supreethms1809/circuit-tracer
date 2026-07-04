# MACAG

MACAG is the graph-first downstream analysis layer over `circuit_tracer` graphs.
It does not generate circuit graphs; it consumes a circuit graph JSON, runs two
evidence-allocation games over the feature nodes, and can write an annotated
graph back for the existing `circuit_tracer` frontend.

- **Game 1** — minimal faithful evidence subset (`game1`).
- **Game 2** — contrastive evidence allocation with overlap penalty (`game2`).

```
graph JSON ──► run_macag (game1 / game2) ──► result JSON ──► annotate_graph ──► UI
```

## Quickstart: end-to-end pipeline

`scripts/run_macag_pipeline.sh` runs the whole chain for a single prompt:
attribute (build a `circuit_tracer` graph) → `game1` + `game2` + baseline head-to-head
→ `annotate_graph` → optionally serve the UI. It assumes the conda env is already
active and uses `python` directly.

```bash
scripts/run_macag_pipeline.sh \
  --prompt "Fact: The capital of the state containing Dallas is" \
  --target " Austin" --foil " Texas" \
  --model google/gemma-2-2b \
  --transcoder-set mntss/clt-gemma-2-2b-426k \
  --slug dallas-austin \
  --outdir results/macag_demo \
  --device cuda
```

Outputs land under `--outdir`:

- `graphs/<slug>.json` — attribution graph
- `oracle_kwargs.json` — generated kwargs for the ReplacementModel oracle
- `macag_game1.json`, `macag_game2.json` — game results
- `macag_baselines.json` — head-to-head baseline comparison (influence, EAP, Shapley, game1, ACDC)
- `graphs/macag-<slug>.json` — annotated graph, registered in
  `graphs/graph-metadata.json` under the slug `macag-<slug>` so it shows up in the
  dropdown with a leading **MACAG** label (via `metadata.title_prefix`).

Useful flags (run `-h` for the full list): `--device cuda|cpu` (MPS is
unsupported — safetensors lazy decoder), `--skip-attribute` to reuse an existing
graph, `--skip-baselines` to skip the head-to-head harness, `--serve --port 8041`
to launch the visualization server at the end, and solver knobs
`--prefilter-top-k --budget --beta --abr-iters --alpha --lam --eps`.

## `run_macag` CLI

`python -m macag.cli.run_macag {game1,game2} ...`. Set `PYTHONPATH=.` when running
from the repo root.

### Common arguments (both games)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--graph-json` | required | Circuit graph JSON. |
| `--target` | required | Target class/label (the pipeline uses `y`). |
| `--output-json` | required | Where to write the result JSON. |
| `--input-id` | `unknown` | Identifier echoed into the output. |
| `--alpha` | `0.5` | Faithfulness mix: `alpha*sufficiency + (1-alpha)*necessity`. |
| `--lam` | `0.01` | Sparsity penalty λ. |
| `--budget` | `None` | Optional cap on `|E|`. |
| `--prefilter-top-k` | `None` | Keep only the top-k singletons before greedy search. |
| `--connected` | off | Require evidence connectedness. Connectivity routes through intermediate feature/error nodes, but **not** through logit or embedding nodes — those are hubs in pruned attribution graphs (nearly every feature touches a logit node), and allowing them would make the constraint vacuous. |
| `--min-gain` | `0.0` | Minimum positive marginal gain to add a node. |
| `--candidates-file` | `None` | Restrict the candidate universe (`.json` list or text). |
| `--include-error-nodes` | off | Admit MLP reconstruction-error nodes as ablation candidates (opt-in C1; not supported by the feature-index intervention path — errors if error nodes are present). |
| `--no-cache` | off | Disable oracle memoization. |
| `--no-progress` | (progress on) | Silence solver progress logs/bars. |
| `--log-every` | `50` | Fallback log frequency when `tqdm` is unavailable. |

Oracle selection (mutually exclusive entrypoints):

- `--toy-oracle-json` — toy additive oracle for quick tests.
- `--oracle-factory module:function` plus `--oracle-kwargs-json` (inline) and/or
  `--oracle-kwargs-file` (JSON file) for real interventions.

### Game 1 specific

- `--faithfulness-eps` — optional stop condition in `[0, 1]`; interpretation
  depends on `--stop-metric`.
- `--stop-metric {normalized,raw_relative}`:
  - **`normalized`** — stop when `faithfulness_normalized >= 1 - eps`. Divides by
    `recoverable_range` (`all - empty`), the correct error-floor-aware target with
    **frozen attention**. Goes degenerate when that range collapses toward
    zero/negative, producing spurious early/late stops.
  - **`raw_relative`** — denominator-free diminishing-returns stop: stop before
    adding a node whose marginal raw faithfulness gain (λ-free `faithfulness_delta`
    increase, not the λ-penalized utility gain) is `< eps *` the first feature's
    faithfulness gain. Stable regardless of the error floor, so use it when
    `recoverable_range` is unreliable (**unfrozen attention**).
  - Unset, it resolves from `--freeze-mode`: `normalized` for `frozen`,
    `raw_relative` for `unfrozen`/`both`. `normalized` + `--freeze-mode both` is
    rejected — the two legs would stop under non-comparable rules.
- `--freeze-mode {frozen,unfrozen,both}` (default `frozen`):
  - **`frozen`** — use the factory-built oracle as-is (`freeze_attention`
    defaults to true in the factory; a kwargs-built unfrozen oracle is honored
    with a warning).
  - **`unfrozen`** — derive a freeze-flipped oracle from the factory-built one
    (same model, fresh cache).
  - **`both`** — the matched frozen/unfrozen protocol: one model load, two
    oracles, both Game 1 legs run under identical budget/prefilter/eps/α/λ with
    `raw_relative` forced, and the output gains `frozen` / `unfrozen` leg
    payloads plus an `attention_mediation` block (verdict, range flip, evidence
    overlap, upstream/early-layer recruitment). See
    [Frozen vs unfrozen attention](#frozen-vs-unfrozen-attention).
  - `unfrozen`/`both` require a ReplacementModel-backed oracle
    (`--oracle-factory`), not `--toy-oracle-json`.

### Game 2 specific

- `--foil` — required foil class/label (the pipeline uses `y_foil`).
- `--beta` — overlap penalty β (default `0.1`).
- `--abr-iters` — max solver iterations for either solver (default `10`).
- `--solver {abr,fp}` (default `abr`) — update rule:
  - **`abr`** — alternating best response: each agent best-responds to the
    opponent's **last** evidence set (Jacobi updates). Can land in 2-cycles;
    best-iterate tracking returns the highest combined-utility allocation seen.
  - **`fp`** — fictitious play: each agent best-responds to the opponent's
    **empirical history** of evidence sets. The opponent only enters
    `game2_utility` through the overlap penalty, so the expected utility against
    the empirical mixture is exact and cheap:
    `E[u(E)] = faithfulness_delta(E) - lam*|E| - beta * sum_{n in E} p_t(n)`,
    where `p_t(n)` is the fraction of past rounds the opponent included node
    `n`. No extra oracle calls are needed versus ABR. Prefer `fp` when ABR
    cycles or fails to converge.
- `--fp-tol` (default `1e-3`) — fictitious play stops early when the max change
  in empirical node frequencies across both agents falls below this tolerance
  (or when both best responses repeat).

With `--solver fp` the output JSON gains an `fp` block with
`node_frequencies_y` / `node_frequencies_foil` — empirical inclusion frequency
per node, a soft evidence-membership score. Reported `scores` always use the
hard overlap of the returned joint allocation, so ABR and FP runs are directly
comparable.

Game 2 selects on raw `game2_utility` (no `faithfulness_eps` stop), so
the Game 1 stop-metric fix does not apply to it; `overlap_rate` is
denominator-free as well.

### Toy-oracle example

```bash
PYTHONPATH=. python -m macag.cli.run_macag game2 \
  --graph-json /path/to/graph.json \
  --target y --foil y_foil \
  --toy-oracle-json /path/to/toy_oracle.json \
  --output-json /tmp/macag_out.json
```

## Real interventions: ReplacementModel oracle

`macag.factories.replacement_model:create_replacement_model_oracle` is the
canonical integration entrypoint. It builds a scorer from either a hub
`transcoder_set` or a local checkpoint (`local_clt_path`), and automatically
restricts candidates to the feature nodes present in the graph JSON
(`feature_type == "cross layer transcoder"` by default).

1. Create oracle kwargs JSON:

```json
{
  "model_name": "google/gemma-2-2b",
  "transcoder_set": "mntss/clt-gemma-2-2b-426k",
  "prompt": "The capital of the state containing Denver is",
  "graph_json": "/absolute/path/to/circuit.json",
  "backend": "transformerlens",
  "score_kind": "logit_gap",
  "target_token_by_label": {
    "y": " Colorado",
    "y_foil": " Wyoming"
  },
  "foil_by_target": {
    "y": "y_foil",
    "y_foil": "y"
  },
  "freeze_attention": true
}
```

2. Run:

```bash
PYTHONPATH=. python -m macag.cli.run_macag game2 \
  --graph-json /absolute/path/to/circuit.json \
  --target y --foil y_foil \
  --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle \
  --oracle-kwargs-file /absolute/path/to/oracle_kwargs.json \
  --output-json /tmp/macag_game2.json
```

Notes:
- `target_token_by_label` entries can be token strings or explicit ids (`"id:12345"`).
- `freeze_attention` is a **scoring-time** argument: the attribution graph is
  unchanged, so you can reuse the same graph and flip `freeze_attention` to
  re-score frozen vs unfrozen. See [Frozen vs unfrozen attention](#frozen-vs-unfrozen-attention).
- `score_kind` defaults to `logit_gap`; `backend` defaults to `transformerlens`.
- `local_clt_path` can point to either a standard `circuit_tracer` CLT checkpoint
  or a `spline_clt` checkpoint directory (auto-detected via `metadata.safetensors`).
  When set, `transcoder_set` is optional.

## Baseline head-to-head: `run_baselines`

`python -m macag.cli.run_baselines` runs the Phase-2 baseline selectors against
Game 1 on the **same** candidate node set and the **same** oracle, so only the
selection rule differs (macag.md §9.3 / Appendix A). Selectors live in
`macag/baselines/`:

| Method | Module | What it does | Selection cost |
|--------|--------|--------------|----------------|
| `influence` | `baselines/influence.py` | top-k by the graph's `influence` metadata | 0 oracle calls |
| `eap` | `baselines/eap.py` | signed path-effect on the target (−foil) logit, propagated through the weighted links | 0 oracle calls |
| `shapley` / `banzhaf` | `baselines/shapley_select.py` | MC Shapley (antithetic permutations) / MC Banzhaf over the MACAG coalitional v(S) — the gold credit reference | O(perms × \|C\|) |
| `game1` | `games/game1_min_faithful.py` | MACAG's greedy itself (prefixes of `selected_order`) | O(\|E*\| × \|C\|) |
| `acdc` | `baselines/acdc_prune.py` | ported ACDC: top-down prune when removal moves v by < τ; τ-sweep | O(\|C\|) per τ |

```bash
PYTHONPATH=. python -m macag.cli.run_baselines \
  --graph-json /absolute/path/to/circuit.json \
  --target y --budget 8 \
  --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle \
  --oracle-kwargs-file /absolute/path/to/oracle_kwargs.json \
  --methods influence,eap,shapley,game1,acdc \
  --shapley-permutations 64 --shapley-seed 0 \
  --bruteforce-k 4 \
  --output-json /tmp/baselines.json
```

The output JSON contains, per method, the full ranking, per-k evidence sets
scored with the games' `FaithfulnessMetrics`, and per-method selection oracle
costs (fresh cache per method, so counts are standalone-honest); plus a
`comparison` block with faithfulness@matched-k, faithfulness-vs-size AUC,
precision@k / Jaccard vs the Shapley-gold ranking, pairwise Jaccard at the
budget, and Spearman diagnostics (including EAP-score vs Game-1 marginal gain,
the §A.5 linearity probe). `--bruteforce-k` additionally reports the exact
best size-k subset and each method's optimality gap (roadmap B3.2; guard:
prefilter the pool, the search refuses > `--bruteforce-max-evals` subsets).

## Visualize in the `circuit_tracer` frontend

```bash
PYTHONPATH=. python -m macag.cli.annotate_graph \
  --graph-json /absolute/path/to/circuit.json \
  --macag-result-json /tmp/macag_out.json \
  --output-json /absolute/path/to/circuit_macag.json

python -m circuit_tracer start-server \
  --graph_file_dir /absolute/path/to
```

The annotator adds `qParams.pinnedIds` and supernodes labeled `MACAG:shared`,
`MACAG:unique_y`, `MACAG:unique_foil` (or `MACAG:E_star` for Game 1), and updates
`graph-metadata.json` so the new graph appears in the server dropdown. Then open
the annotated slug in the browser UI. (The end-to-end pipeline does this for you
and registers the graph under `macag-<slug>` with a `MACAG …` title prefix in the UI dropdown.)

## Auto-suggest supernodes

If you do not have manually annotated supernodes, derive candidates from graph
metrics (`influence`, `activation`) plus connectivity:

```bash
PYTHONPATH=. python -m macag.cli.suggest_supernodes \
  --graph-json /absolute/path/to/circuit.json \
  --output-supernodes-json /absolute/path/to/auto_supernodes.json \
  --output-candidates-json /absolute/path/to/auto_candidates.json \
  --output-graph-json /absolute/path/to/circuit_with_auto_supernodes.json \
  --replace-existing-supernodes
```

Use the produced candidates file directly with MACAG via `--candidates-file`.

## Experiment drivers

Result collection runs on **two benchmarks — MIB and InterpBench — one command
each**; all campaign configuration (3-seed loop, fast/gold baseline split,
analysis) is set inside the scripts. Everything reads/writes under `results/`.

| Script | Role |
| --- | --- |
| `scripts/run_mib_benchmark.sh` | **MIB campaign entry point.** Per seed: GPU-parallel fast-pass sweep, same-seed Shapley-gold pass (first 50/task), KL rescore, analyzer CSVs, bootstrap/Wilcoxon, faithfulness curves, gold-circuit IOI scoring; then cross-seed aggregation. |
| `scripts/run_interpbench_benchmark.sh` | **InterpBench campaign entry point.** Runs `experiments/run_interpbench_macag.py` once per seed (node-level AUROC/precision vs exact ground-truth circuits). |
| `scripts/run_macag_mib_parallel.sh` | GPU-parallel launcher: per-CLT worker pools over the MIB sweep (426k=8, 2.5M=3 workers; sources `macag_parallel_common.sh`). |
| `scripts/run_macag_mib.sh` | Sweep runner: iterates (CLT, prompt) cells, calls the pipeline per prompt, auto-runs the analyzers when unsharded / `ANALYZE_ONLY=1`. |
| `scripts/run_macag_pipeline.sh` | Single-prompt pipeline: attribute → game1 (dual-freeze) → game2 (abr+fp) → baselines → annotate → KL. Also the tool for one-off debugging runs. |
| `scripts/run_macag_shapley_pass.sh` | Deferred MC-Shapley gold baseline over a finished root; merges into `macag_baselines.json`. |
| `scripts/macag_kill_sweep.sh` | Kill orphaned GPU workers after an interrupted sweep. |
| `scripts/macag_bootstrap_wilcoxon.py` | Bootstrap CIs + Wilcoxon tests per sweep root. |
| `scripts/macag_combine_seeds.py` | Cross-seed CSV concatenation + mean/std summaries. |

```bash
# the entire result collection (see macag/docs/run_todo.md for setup + details):
nohup scripts/run_interpbench_benchmark.sh > results/interpbench_campaign.log 2>&1 &
nohup scripts/run_mib_benchmark.sh > results/mib_campaign.log 2>&1 &

# smoke test on the pilot-sized prompt JSON, single seed:
ALLOW_SMALL_JSON=1 SEEDS=0 scripts/run_mib_benchmark.sh
```

Output layout: `results/macag_mib_seed<SEED>/<clt_tag>/<slug>/` with
`macag_game1.json` (dual-freeze: `frozen`/`unfrozen` legs +
`attention_mediation`), `macag_game2_{abr,fp}.json`, `macag_baselines.json`, and
the attribution graph. Aggregate CSVs land in the same root
(`summary.csv`, `abr_vs_fp.csv`, `baselines.csv`, `frozen_vs_unfrozen.csv`).
Per-run KL faithfulness is written to `macag_kl_faithfulness.json` and embedded in
the game/baseline JSON as `kl_faithfulness`; `summary.csv` / `baselines.csv` include
`kl_faith` columns. Set `KL_RESCORE=0` to skip the KL pass.

Manifests:
- `macag/data/mib_benchmark_prompts.json` — MIB tasks (`ioi`, `mcqa`, `arc_easy`),
  built by `experiments/build_mib_benchmark_prompts.py` (collected benchmark).
- `macag/data/acdc_benchmark_prompts.json` / `nonlinear_benchmark_prompts.json` —
  internal diagnostic sets (own ACDC-style tasks; Spline-CLT stress prompts).
  Not part of result collection; usable with `run_macag_pipeline.sh` ad hoc.

## Frozen vs unfrozen attention

`freeze_attention` is the single most consequential scoring switch and it changes
how you should read Game 1. **The recommended protocol is the built-in dual run:**

```bash
PYTHONPATH=. python -m macag.cli.run_macag game1 \
  --graph-json graphs/<slug>.json --target y \
  --oracle-factory "macag.factories.replacement_model:create_replacement_model_oracle" \
  --oracle-kwargs-file oracle_kwargs.json \
  --budget 8 --prefilter-top-k 20 --faithfulness-eps 0.1 \
  --freeze-mode both \
  --output-json results/<slug>/game1_dual.json
```

This loads the model once, runs frozen and unfrozen legs under identical
parameters (`raw_relative` stop forced on both), and emits an
`attention_mediation` block per prompt: the `verdict`
(`attention_mediated` / `feature_mediated` / `indeterminate`), the
`recoverable_range` sign flip, evidence-set overlap, and upstream/early-layer
recruitment counts. Aggregating `attention_mediation.range_flip` across prompts
gives the range-flip rate directly — no pairing script needed.

For single-mode runs:

- **Frozen** (default): attention carries upstream structure "for free," so the
  minimal faithful set tends to be small and `recoverable_range` is positive. Use
  `--stop-metric normalized`.
- **Unfrozen** (`--freeze-mode unfrozen`): ablate-all plus free attention can
  collapse the `empty` baseline, driving `recoverable_range` toward
  zero/negative and making the normalized faithfulness degenerate. The stop
  metric defaults to `raw_relative`; read the **raw**
  sufficiency/necessity/faithfulness (which have no denominator) rather than the
  normalized view. The ACDC/IOI prompts are attention-mediated and show negative
  `recoverable_range` when frozen, which is exactly why the unfrozen ACDC driver
  uses `raw_relative`.

For dual-run outputs, `annotate_graph` takes `--freeze-select
{frozen,unfrozen,both}` (default `both`) and labels the groups
`MACAG:frozen:E_star` / `MACAG:unfrozen:E_star`.

## Operational notes

- `macag/cache` and `macag/output` are runtime artifacts, not source code; avoid
  staging large cached model artifacts from `macag/cache`.
- `macag/docs` is thesis-writing reference material; the active package is the
  top-level `macag/` Python package.
- Factory import paths are restricted to the `macag.` and `circuit_tracer.`
  prefixes (see `_ALLOWED_FACTORY_MODULE_PREFIXES` in `run_macag.py`).
- See `macag/docs/INTERPRETATION_GUIDE.md` for how to read the result JSON and
  diagnose common failure modes.
