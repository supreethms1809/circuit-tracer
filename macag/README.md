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
attribute (build a `circuit_tracer` graph) → `game1` + `game2` → `annotate_graph`
→ optionally serve the UI. It assumes the conda env is already active and uses
`python` directly.

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
- `graphs/<slug>-macag.json` — annotated graph, registered in
  `graphs/graph-metadata.json` under the slug `<slug>-macag` so it shows up in the
  server dropdown.

Useful flags (run `-h` for the full list): `--device cuda|cpu` (MPS is
unsupported — safetensors lazy decoder), `--skip-attribute` to reuse an existing
graph, `--serve --port 8041` to launch the visualization server at the end, and
solver knobs `--prefilter-top-k --budget --beta --abr-iters --alpha --lam --eps`.

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
- `--stop-metric {normalized,raw_relative}` (default `normalized`):
  - **`normalized`** — stop when `faithfulness_normalized >= 1 - eps`. Divides by
    `recoverable_range` (`all - empty`), the correct error-floor-aware target with
    **frozen attention**. Goes degenerate when that range collapses toward
    zero/negative, producing spurious early/late stops.
  - **`raw_relative`** — denominator-free diminishing-returns stop: stop before
    adding a node whose marginal raw faithfulness gain (λ-free `faithfulness_delta`
    increase, not the λ-penalized utility gain) is `< eps *` the first feature's
    faithfulness gain. Stable regardless of the error floor, so use it when
    `recoverable_range` is unreliable (**unfrozen attention**).

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
and registers the graph under `<slug>-macag`.)

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

Batch wrappers around the pipeline, plus their analyzers. All read/write under
`results/`.

| Driver | What it does | Analyzer |
| --- | --- | --- |
| `scripts/run_macag_sweep.sh` | Run the pipeline over a prompt manifest sharing one template (e.g. the two-hop city→state→capital circuit), so the circuit can be compared across facts. | `experiments/analyze_macag_sweep.py` |
| `scripts/run_macag_clt_compare.sh` | Run the sweep once per cross-layer transcoder in `experiments/macag_clt_compare.json` (capacity/cross-model control). | `experiments/analyze_clt_comparison.py` |
| `scripts/run_macag_acdc.sh` | Full pipeline over the ACDC benchmark prompts (`macag/data/acdc_benchmark_prompts.json`) for all runnable CLTs. | `experiments/analyze_macag_acdc.py` |
| `scripts/run_macag_unfrozen.sh` | Re-run Game 1 unfrozen (higher budget) reusing the frozen graphs + kwargs. | `experiments/analyze_frozen_vs_unfrozen.py`, `experiments/analyze_robust_frozen_vs_unfrozen.py` |
| `scripts/run_macag_unfrozen_game2.sh` | Unfrozen contrastive Game 2, matched to the frozen Game 2 params. | `experiments/analyze_game2_frozen_vs_unfrozen.py` |
| `scripts/run_macag_acdc_unfrozen.sh` | Unfrozen Game 1 (`raw_relative` stop) on the ACDC prompts. | `experiments/analyze_acdc_frozen_vs_unfrozen.py` |

Manifests:
- `experiments/macag_generalization_prompts.json` — the shared template + per-fact
  `{slug, city, target, foil}` rows.
- `experiments/macag_clt_compare.json` — the CLTs to compare (`run: false` reuses
  an existing sweep dir in place).
- `macag/data/acdc_benchmark_prompts.json` — ACDC tasks
  (`indirect_object_identification`, `greater_than`, `docstring_completion`) with
  clean/corrupted prompts and correct/incorrect tokens.

## Frozen vs unfrozen attention

`freeze_attention` is the single most consequential scoring switch and it changes
how you should read Game 1:

- **Frozen** (default): attention carries upstream structure "for free," so the
  minimal faithful set tends to be small and `recoverable_range` is positive. Use
  `--stop-metric normalized`.
- **Unfrozen**: ablate-all plus free attention can collapse the `empty` baseline,
  driving `recoverable_range` toward zero/negative and making the normalized
  faithfulness degenerate. Use `--stop-metric raw_relative`, and read the **raw**
  sufficiency/necessity/faithfulness (which have no denominator) rather than the
  normalized view. The ACDC/IOI prompts are attention-mediated and show negative
  `recoverable_range` when frozen, which is exactly why the unfrozen ACDC driver
  uses `raw_relative`.

## Operational notes

- `macag/cache` and `macag/output` are runtime artifacts, not source code; avoid
  staging large cached model artifacts from `macag/cache`.
- `macag/docs` is thesis-writing reference material; the active package is the
  top-level `macag/` Python package.
- Factory import paths are restricted to the `macag.` and `circuit_tracer.`
  prefixes (see `_ALLOWED_FACTORY_MODULE_PREFIXES` in `run_macag.py`).
- See `macag/docs/INTERPRETATION_GUIDE.md` for how to read the result JSON and
  diagnose common failure modes.
