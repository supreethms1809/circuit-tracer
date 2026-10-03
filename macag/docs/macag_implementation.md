> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). Graph wrapper, candidate policies, ReplacementModel oracle, pipeline (former `macag.md` §6–8).

## 6. Graph and Candidate Selection

### 6.1 Circuit Graph Wrapper

MACAG operates on a lightweight directed graph (`CircuitGraph`) with per-node metadata:

```python
class CircuitGraph:
    _node_metadata: dict[NodeId, dict[str, Any]]
    _succ, _pred: dict[NodeId, set[NodeId]]  # adjacency lists
```

Key operations:
- `subgraph(nodes)`: extracts induced subgraph for evidence visualization (nodes/edges sorted for cross-process determinism)
- `is_weakly_connected(nodes)`: subset-internal BFS connectivity check (treats graph as undirected)
- `connected_through(nodes)`: the connectivity check used by the solvers — routes through intermediate feature/error nodes but excludes logit/embedding hubs (see §4.4)
- `from_dict()` / `to_dict()`: JSON serialization compatible with circuit-tracer graph format (`node_id` takes precedence over `id`, matching the intervention loader)

### 6.2 Candidate Extraction

Candidates are extracted from circuit-tracer graph JSON:
1. Scan all nodes for matching `feature_type` (default: `"cross layer transcoder"`)
2. Parse node ID format: `"{layer}_{feature}_{position}"` → `(layer, position, feature_idx)` tuple
3. Build intervention map: each candidate maps to a `(layer, ctx_idx, feature_idx, 0.0)` ablation spec

**Filtering**: Candidates are restricted to feature nodes present in the graph JSON, ensuring MACAG and the scorer stay aligned to the traced circuit. The `node_universe` parameter on `ReplacementModelInterventionScorer` enforces this restriction.

### 6.3 Candidate Policies

The paper runner supports configurable candidate selection (`candidate_policy.strategy`):
- **graph_features** (default): feature nodes from the circuit graph ranked by `influence` (tie-broken by activation, then node ID), truncated to the policy's `top_k` (default 40)
- **auto_supernodes**: runs the supernode proposer ([§8.4](#84-supernode-suggestion)) over the graph and takes the union of all proposed supernode members as the candidate pool (the proposed groups are also written to `auto_supernodes.json`)

Both strategies share the `top_k` / `feature_types` knobs; there is no separate "top_k" strategy — truncation is a parameter of both.

---

## 7. Oracle Backend

### 7.1 ReplacementModel Scorer

The primary oracle backend uses circuit-tracer's `ReplacementModel` for real interventions:

```python
class ReplacementModelInterventionScorer:
    def score_keep_only(self, nodes, target) -> float
    def score_remove(self, nodes, target) -> float
    def score_all(self, target) -> float
    def score_empty(self, target) -> float
```

Each intervention:
1. Maps node IDs to `(layer, position, feature_idx, ablation_value)` specs
2. Calls `model.feature_intervention()` with the spec list
3. Extracts last-token logits from the output
4. Computes scalar score via configurable mode: `logit_gap` (default), `logit`, `prob`, `negative_loss`, or `kl_divergence` (full-distribution, target-free; reference logits cached from the clean pass — [§2.5](macag_framework.md#25-kl-rescoring-a-selection-independent-faithfulness-metric))

By default (`freeze_attention=True`), interventions hold attention patterns and LayerNorm denominators at their clean values, isolating the direct effect of feature ablation from second-order attention changes. Setting `freeze_attention=False` lets attention recompute from the ablated activations, capturing attention-mediated effects; the four scoring methods above (`score_empty` in particular) are evaluated under whichever mode is set, which is why the flag changes every derived metric. See [§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor).

### 7.2 Auto-Detection of CLT Variant

The factory `create_replacement_model_scorer()` auto-detects the CLT type:
- If checkpoint directory contains `metadata.safetensors` → loads via `load_spline_clt()` (Spline-CLT)
- Otherwise → loads via `load_clt()` from circuit-tracer (standard linear CLT)

This makes MACAG transparent to the encoder architecture — the same selection and scoring code runs on both.

### 7.3 Toy Oracle

For testing and development, `ToyAdditiveInterventionScorer` provides a simple additive model:
- Each node has a fixed weight
- `keep_only(E)` = base_score + $\sum_{n \in E}$ weight($n$)
- `remove(E)` = all_score - $\sum_{n \in E}$ weight($n$)

This allows unit testing of Game 1/Game 2 solvers without a transformer model.

---

## 8. Integration Pipeline

> **Two consumption paths (read this first).** MACAG is consumed two ways, and
> mixing them is a recurring source of confusion: (1) the **paper-runner path**
> ([§8.2](#82-paper-runner-integration), [§6.3](#63-candidate-policies)) — the
> Spline-CLT/GPT-2 paper suites, where candidate selection goes through
> `candidate_policy` (top-k-by-influence, default 40) and results land in
> `macag_records.jsonl`; and (2) the **CLI path** (`macag/cli/run_macag.py` +
> `scripts/run_macag_*.sh`), used by the MIB campaigns and Dallas–Austin, where
> the candidate set is the graph's full feature-node set (v3 pruned:
> mean $\sim 748$ / $796$ / $505$ on Gemma-426k / 2.5M / Llama; Dallas unpruned:
> **2028**) and any narrowing happens via the solver's `prefilter_top_k` (off on
> v3). Those runs do **not** use §6.3's candidate policies. Pre-v3 two-hop
> graphs were smaller (~260–338); do not quote that range as current.

### 8.1 End-to-End Flow

```
1. Obtain a CLT (train one, or load a public checkpoint as in the case study)
2. Generate circuit graph via causal attribution for each prompt
3. Extract candidate feature nodes from graph (feature_type = "cross layer transcoder")
4. Build ReplacementModel scorer (auto-detects CLT variant)
5. Run Game 1 and/or Game 2 on each prompt (optionally over multiple ε / seeds)
6. Record per-prompt metrics to macag_records.jsonl
7. Aggregate metrics across prompts (and seeds / variants if present)
8. (Optional) Annotate graph with evidence sets for visualization
```

### 8.2 Paper Runner Integration

The paper suite runner (`PaperSuiteRunner`) orchestrates MACAG as a stage:
- Reads `prompt_metrics.jsonl` from the evaluation stage (filtering by `include_macag=true`)
- For each prompt: loads graph JSON, selects candidates, runs configured games
- Supports multiple Game 1 configurations (e.g., $\varepsilon = 0.10$ and $\varepsilon = 0.05$) and Game 2 configurations in a single pipeline invocation
- Records to `macag_records.jsonl` with full provenance (checkpoint path, git commit, seed, suite name)

### 8.3 Graph Annotation

After MACAG completes, evidence sets can be merged back into the circuit-tracer graph JSON for visualization:

- **Game 1**: Creates a single supernode `"[prefix]:E_star"` containing all evidence nodes
- **Game 2**: Creates supernodes for `shared`, `unique_y`, and `unique_foil` with descriptive labels
- Merges into `qParams.pinnedIds` (individual nodes) and `qParams.supernodes` (grouped nodes)
- Updates `graph-metadata.json` index (with graceful failure if circuit-tracer metadata indexing is unavailable)

### 8.4 Supernode Suggestion

For large circuit graphs, `suggest_supernodes.py` automatically discovers candidate supernodes:
1. **Salience ranking**: scores nodes by a weighted combination of influence, activation magnitude, and token probability
2. **Grouping**: finds connected components via BFS, optionally splits by context position
3. **Chunking**: breaks large groups into `max_group_size` (default 12) chunks
4. **Labeling**: auto-generates labels from the most common `clerp` (human-readable feature description) metadata

---
