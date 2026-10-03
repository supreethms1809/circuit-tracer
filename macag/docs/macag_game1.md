> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). Game 1 objective, greedy solver, connectivity, outputs (former `macag.md` §4).

## 4. Game 1: Minimal Faithful Evidence

### 4.1 Objective

Find a small subset of feature nodes that explains the model's prediction (smallness is priced into the objective via $\lambda$, not certified — §1.4 guardrails):

$$E^* = \arg\max_{E \subseteq C} \left[\alpha \cdot \left(S_{\text{keep}}(E) - S_{\text{empty}}\right) + (1 - \alpha) \cdot \left(S_{\text{all}} - S_{\text{remove}}(E)\right) - \lambda |E|\right]$$

### 4.2 Algorithm: Greedy Hill-Climbing

**Pseudocode**:
```
Algorithm: MACAG Game 1 — Greedy Hill-Climbing
──────────────────────────────────────────────
Input:  Graph G, Oracle O, target y, candidates C,
        α, λ, ε (optional), stop_metric ∈ {normalized, raw_relative},
        budget B (optional)
Output: Evidence set E*, utility U*

1.  E ← ∅,  first_faith_gain ← nil
2.  if prefilter_top_k:
3.      C ← PREFILTER(C, O, y, α, λ, top_k)    // rank by singleton U({n})
4.  repeat
5.      if B ≠ nil and |E| ≥ B: break
6.      U_curr ← α·(O.keep(E,y) - O.empty(y)) + (1-α)·(O.all(y) - O.remove(E,y)) - λ|E|
7.      n* ← nil,  best_gain ← min_gain
8.      for each n ∈ C \ E:
9.          if connected and ¬CONNECTED_THROUGH(E ∪ {n}): skip   // hub-excluding, §4.4
10.         U_trial ← UTILITY(E ∪ {n})
11.         gain ← U_trial - U_curr
12.         if gain > best_gain:
13.             best_gain ← gain,  n* ← n
14.             faith_gain* ← Δfaith(E ∪ {n}) − Δfaith(E)   // λ-free faithfulness gain
15.         else if gain = best_gain and str(n) < str(n*):
16.             n* ← n,  faith_gain* ← Δfaith(E ∪ {n}) − Δfaith(E)
17.     if n* = nil: break                          // no improving move
18.     // raw_relative stop (BEFORE adding): diminishing returns vs first feature,
19.     // measured on RAW faithfulness gains (no λ penalty) so λ cannot distort it
20.     if ε ≠ nil and stop_metric = raw_relative and first_faith_gain > 0
21.            and faith_gain* < ε · first_faith_gain: break
22.     E ← E ∪ {n*}
23.     if first_faith_gain = nil: first_faith_gain ← faith_gain*
24.     // normalized stop (AFTER adding): error-floor-aware faithfulness target
25.     if ε ≠ nil and stop_metric = normalized
26.            and faithfulness_norm(E) ≥ 1 − ε: break
27. return E, UTILITY(E)
```

**Step-by-step**:

1. Initialize $E = \emptyset$
2. **(Optional) Prefilter**: Rank all candidates by singleton utility $U(\{n\})$, keep top-$k$ (default: no prefilter). This uses the same `game1_utility()` function as the main solver, biasing toward sparsity-conscious candidate selection.
3. At each step, evaluate all remaining candidates and add the one with the highest marginal utility gain: $$n^* = \arg\max_{n \in C \setminus E} \left[U(E \cup \{n\}) - U(E)\right]$$ Ties are broken alphabetically (by `_sort_key()`) for deterministic reproducibility.
4. Stop when:
   - No candidate provides positive marginal gain exceeding `min_gain`, or
   - Budget $|E| \geq B$ is reached, or
   - The faithfulness early-stop condition (set by `faithfulness_eps` $=\varepsilon$ and `stop_metric`) is met — see below.

**Stop metric** (`stop_metric`, default `normalized`) controls how the optional $\varepsilon$ early-stop is interpreted:

- **`normalized`**: after each addition, stop when the error-floor-aware faithfulness reaches its target, $\text{faithfulness}_{\text{norm}}(E) \geq 1 - \varepsilon$. This is the correct target with **frozen attention**, but goes degenerate when `recoverable_range` collapses toward zero/negative (e.g. unfrozen attention), producing spurious early/late stops.
- **`raw_relative`**: a denominator-free diminishing-returns rule checked *before* adding a node — stop when the best available marginal raw faithfulness gain falls below $\varepsilon \cdot (\text{the first added feature's gain})$. The gains in this test are the $\lambda$-free `faithfulness_delta` increases, NOT the $\lambda$-penalized utility gains the greedy maximizes — otherwise the sparsity penalty would shift both sides of the ratio test and distort the stop whenever $\lambda > 0$. Because it never divides by `recoverable_range`, it is stable when that range is unreliable, and is the recommended choice for **unfrozen-attention** and attention-mediated (IOI-style) runs.

**Approximation guarantee**: If the faithfulness function $f(E)$ is submodular and monotone, the greedy algorithm achieves a $(1 - 1/e) \approx 0.632$ approximation ratio to the optimal solution under a cardinality constraint (Nemhauser et al., 1978). The sparsity penalty $-\lambda|E|$ is modular and does not affect the approximation ratio. In practice, neural network logit gaps are not guaranteed submodular, so this serves as a best-case bound rather than a formal guarantee.

### 4.3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\alpha$ | 0.5 | Sufficiency vs. necessity balance |
| $\lambda$ | 0.01 | Sparsity penalty coefficient. CLI default; the pipeline / v3 / Dallas campaigns use **0.02** (`LAM=0.02`) |
| `faithfulness_eps` ($\varepsilon$) | None | Early-stop threshold in $[0,1]$; interpreted via `stop_metric` |
| `stop_metric` | resolved from `freeze_mode` | `normalized` (error-floor-aware) or `raw_relative` (denominator-free). CLI default: `normalized` for `--freeze-mode frozen`, `raw_relative` for `unfrozen`/`both`; `normalized` + `both` is rejected (non-comparable legs) |
| `freeze_mode` (CLI) | `frozen` | `frozen`: factory-built oracle as-is; `unfrozen`: derive a freeze-flipped oracle; `both`: matched dual run + `attention_mediation` diagnostic. `unfrozen`/`both` require a ReplacementModel-backed oracle (not `--toy-oracle-json`) |
| budget | None | Hard maximum on evidence set size |
| prefilter_top_k | None | Pre-filter candidates by singleton gain |
| connected | **CLI `True`** (`--no-connected` to disable); **Python API `False`** | Require evidence to form a connected subgraph. v3 / Dallas / `run_macag_pipeline.sh` pass `--no-connected` (`CONNECTED=0`) |
| min_gain | 0.0 | Minimum marginal gain required to add a node |

### 4.4 Connectivity Constraint

When `connected=true`, the algorithm only considers candidates that would maintain weak connectivity of the evidence set (checked via BFS treating the graph as undirected, routing through intermediate nodes that are not themselves candidates). This produces more interpretable circuits at the cost of potentially lower faithfulness.

**Hub exclusion**: logit and embedding nodes are NOT allowed as connectivity intermediates. In pruned attribution graphs they are extreme hubs — in a representative GPT-2 graph, ~47% of all edges terminate in 10 logit nodes, making the entire graph a single weak component. Routing connectivity through them would render the constraint vacuous (every feature pair "connected" because both influence the output). With the exclusion, connectivity means membership in the same feature/error-node sub-circuit: on the same graph, the constraint distinguishes one 291-node computational component from 46 isolated features instead of accepting all pairs. Error nodes remain valid intermediates (they are part of the per-layer computation path). The exclusion set is configurable via `CircuitGraph.connected_through(..., exclude_intermediate_types=...)`; passing `None` restores the permissive behavior.

### 4.5 Output

`EvidenceSetResult` containing:
- **evidence**: the selected node set $E^*$
- **selected_order**: insertion order of nodes (for reproducibility)
- **induced_subgraph**: the subgraph induced by $E^*$
- **metrics**: `FaithfulnessMetrics` — raw scores (`all`, `empty`, `keep_only`, `remove`, `sufficiency`, `necessity`, `faithfulness`) plus the error-floor-aware view (`error_floor`, `recoverable_range`, `sufficiency_normalized`, `necessity_normalized`, `faithfulness_normalized`)
- **utility**: final utility score
- **sparsity**: fraction of candidates not selected
- **iterations**: number of greedy steps taken
- **candidate_count** / **total_candidates**: candidates after prefilter / before prefilter
- **oracle_calls**, **cache_hits**, **cache_size**: oracle/memoization profile
- **params**: the resolved knobs (`alpha`, `lambda`, `budget`, `faithfulness_eps`, `stop_metric`, `prefilter_top_k`, `connected`, `min_gain`)

The CLI (`run_macag game1`) serializes this as `{params, evidence, scores, stats}`, mirroring the Game 1 evidence into the Game 2 schema keys (`E_star`/`E_y`/ `unique_y`) so a single annotator handles both games.

**Dual-freeze output** (`--freeze-mode both`). Single-mode JSON is the inner payload `{params, evidence, scores, stats}` with no top-level `freeze_mode` (freeze lives on the oracle / `params.freeze_attention`). Dual-mode wraps two of those payloads:

```json
{
  "input_id": "...", "target": "y", "foil": "y_foil", "game": "game1",
  "freeze_mode": "both",
  "params": { "...shared matched solver params...", "stop_metric": "raw_relative", "matched": true },
  "frozen":   { "params": {"...", "freeze_attention": true},  "evidence": {"E_star": ["..."]}, "scores": {"..."}, "stats": {"..."} },
  "unfrozen": { "params": {"...", "freeze_attention": false}, "evidence": {"E_star": ["..."]}, "scores": {"..."}, "stats": {"..."} },
  "attention_mediation": {
    "range_frozen": -3.1, "range_unfrozen": 4.2,
    "range_flip": true, "reverse_flip": false, "verdict": "attention_mediated",
    "evidence_size_frozen": 4, "evidence_size_unfrozen": 7,
    "evidence_jaccard": 0.3,
    "evidence_shared": ["..."], "evidence_only_frozen": ["..."], "evidence_only_unfrozen": ["..."],
    "upstream_count_frozen": 0, "upstream_count_unfrozen": 3,
    "early_count_frozen": 0, "early_count_unfrozen": 2,
    "n_layers": 26, "final_ctx_idx": 8
  }
}
```

Each leg sub-dict is exactly the single-mode payload minus the envelope, so the annotator (`annotate_graph --freeze-select {frozen,unfrozen,both}`) reuses the same group builder per leg (`MACAG:frozen:E_star` / `MACAG:unfrozen:E_star`). The `attention_mediation` verdict rule (strict zero threshold; raw ranges are reported so confidence bands can be applied downstream):

| `range_frozen` | `range_unfrozen` | `verdict` | flags |
|---|---|---|---|
| $<0$ | $\ge0$ | `attention_mediated` | `range_flip` (the §10.4 flip) |
| $\ge0$ | $\ge0$ | `feature_mediated` | — |
| $<0$ | $<0$ | `indeterminate` (behavior not recoverable from features under either convention) | — |
| $\ge0$ | $<0$ | `indeterminate` (unexpected inversion) | `reverse_flip` |

`upstream_count_*` counts evidence nodes at `ctx_idx < final_ctx_idx` (equivalent to the legacy reverse-position $>0$ convention — logit nodes carry the final prompt position, so the max position over the graph is the prediction token); `early_count_*` counts evidence nodes with `layer < n_layers/3`, with `n_layers` inferred from feature/error-node layers only. Layer/position resolve from node metadata first, then the `{layer}_{feature}_{pos}` node-ID convention; if the graph carries neither, the count fields are `null` (keys always present — stable schema for downstream aggregation). Note "matched" means matched *parameters*: each leg's prefilter ranks singletons under its own oracle, so the retained pools may differ — same $k$, mode-specific gains, by design.

---
