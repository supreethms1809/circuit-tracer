# Baselines: original papers vs MACAG ports

The ports are intentional. MACAG Game 1 selects a small set of **CLT/SAE feature nodes** under an intervention game `v(S)` (α-mixed sufficiency and necessity on a scalar score, default **target−foil logit gap**). A fair test of that selector keeps the **same candidates and the same evaluation oracle**. Running the published ACDC or Syed EAP pipelines unchanged would compare different objects (native edges vs feature nodes), different ablations (corrupted patch vs zero-ablation), and different metrics (KL vs logit-gap faithfulness). That is a valid *pipeline* comparison, but it does not isolate the selection rule. Jaccard / precision@k against Game 1 or Shapley-gold is only meaningful on the ported, shared-universe track.

**Operational protocol for TMLR tables:** [`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md).  
**Cluster launch runbook (v4):** [`run_todo_v4.md`](run_todo_v4.md) — Pass A/B/C, no Shapley until a later pass.  
**Dallas–Austin numbers** (one-prompt Llama case study: Game 1, prefilter, \(\alpha\), Track A ports, Track B native edges): [`macag_dallas_austin.md`](macag_dallas_austin.md) and the long tables below.

**Shared MACAG evaluation (all ports, after selection).** Every selected set is scored with `FaithfulnessMetrics` under the same oracle:

```
v(S) = α · suff(S) + (1 − α) · nec(S)
```

where sufficiency is keep-only(S) vs empty and necessity is all vs remove(S). Default scalar score is `logit_gap`. KL faithfulness is a **post-hoc rescore**, not the selection objective. Matched budget k, faithfulness-vs-size AUC, and oracle-call counts are the comparison axes.

---

## 1. ACDC (Conmy et al., 2023)

**Original paper.** Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. *Towards Automated Circuit Discovery.* NeurIPS 2023 (arXiv:2304.14997). Method ID here: `acdc` / `acdc_ported`. Related but not original: `acdc_native`.

### Problem the paper solves

Find a **minimal subgraph of the transformer’s native computational DAG** that still implements a specified behavior (a circuit), without hand-specified candidates. The unit of search is an **edge** between components (attention heads, MLPs, embeddings, logits).

### Original method (brief)

Algorithm 1 is **top-down edge pruning**:

1. Start from the full computational graph.
2. Traverse edges in reverse topological order (output side first).
3. For each edge, **activation-patch** it with a **corrupted** (counterfactual) run.
4. If the change in the task metric is below a threshold τ, drop the edge.
5. Repeat until the remaining graph is the circuit.

Ablation is **corrupted patching**, not zeroing. The search is path-dependent: the order of pruning matters.

### Original evaluation metrics

- **KL divergence** between the full model H and the circuit G: typically `KL(G ‖ H)` or `KL(H ‖ G)` on the next-token distribution (task-dependent in the paper; KL is the canonical faithfulness metric).
- Circuit **size** (edges / nodes remaining) as the minimality axis.
- Task-specific checks (e.g. IOI logit difference) in case studies, always on the **native** graph.

### Why we port instead of running ArthurConmy ACDC

MACAG does not search native edges. Its players are **CLT feature nodes** already present in a circuit-tracer graph. Dropping in the ArthurConmy repo would:

- operate on a different universe (heads/MLPs vs features),
- use a different intervention (corrupted edge patch vs feature zero-ablation),
- optimize a different metric (KL vs logit-gap v),
- make node-set agreement with Game 1 undefined.

The port (`macag/baselines/acdc_prune.py`) keeps Conmy’s **selection rule** — one reverse-topological pass; drop if `Δmetric < τ` — and applies it to **feature nodes** under MACAG’s coalitional v (zero-ablation keep/remove mix). That answers: *given this CLT candidate pool and this v, does top-down τ-prune beat Game 1’s bottom-up greedy?* It does **not** answer: *does MACAG beat published ACDC circuits?*

`acdc_native` applies the **same node τ-rule** to heads/MLPs. That is closer to Conmy’s universe but still **not** Algorithm 1 (nodes not edges; zero not corrupted patch; v not KL). Do not Jaccard it against Game 1.

### Metric: same or different?

**Different for selection, aligned for reporting.**

|  | Original ACDC | MACAG port (`acdc`) |
| --- | --- | --- |
| Selection metric | KL (or task metric) under corrupted edge patch | MACAG `v(S)` under zero-ablation of feature nodes |
| Why change | Shared-v selector isolation: prune decisions must be comparable to Game 1’s objective |  |
| Reporting | KL vs circuit size | Same v / logit-gap faithfulness at matched k; optional KL **rescore** of the stored set |

KL rescoring (`macag/kl_rescore.py`) is the closest original-paper metric we report, but it is applied **after** selection, not used to prune.

---

## 2. EAP / attribution patching (Syed et al., 2023; Nanda, 2023)

**Original papers.**

- Nanda, *Attribution Patching: Activation Patching At Industrial Scale* (2023 blog): the first-order formula.
- Syed, Quirke, Nanda. *Attribution Patching Outperforms Automated Circuit Discovery* (NeurIPS 2023 ATTRIB workshop / arXiv:2310.10348): that formula used for **edge** scoring and **top-k circuit discovery**.

Method ID here: `eap_syed` (aliases `eap_ap`, `attribution_patching`). **Not** `eap` / `eap_graph`, which is Jacobi path-effect on exported link weights (an influence cousin, not Syed).

### Problem the papers solve

Activation patching estimates the causal effect of replacing a clean activation with a corrupted one, but costs one forward per edge. Attribution patching (AtP) approximates that effect for **all edges at once** with two forwards and one backward, so you can rank edges and take a top-k circuit cheaply. Syed et al. argue this discovers circuits that match or beat ACDC at far lower cost.

### Original method (brief)

For an edge (or activation) e, with clean/corrupted activations `e_clean`, `e_corr` and loss/metric L on the clean run:

```
Δ_e L  ≈  (e_corr − e_clean)ᵀ ∇_{e_clean} L
```

Syed EAP: score **native edges** of the computational graph this way, keep the top-k by `|Δ|`, and treat that set as the circuit. Nanda’s post is the estimator; Syed is circuit discovery with that estimator.

### Original evaluation metrics

- **Faithfulness of the discovered edge set**, typically **KL** between full model and circuit, and/or **logit difference** on IOI-style tasks.
- Comparison to ACDC on the **same native-edge circuits** (size vs faithfulness, wall-clock / number of forwards).
- The AtP score itself is not the reported task metric; it is the **ranking** signal.

### Why we port instead of running native-edge Syed EAP

Game 1’s candidates are **CLT feature nodes**, not attention/MLP edges. Native-edge EAP would again change the universe, so set overlap with Game 1 is meaningless.

The port (`macag/baselines/eap_syed.py`) applies **Syed’s formula to each MACAG candidate feature** `(layer, pos, feat)`:

```
Δ_f L  ≈  (a_f^corr − a_f^clean) · ∂L/∂a_f^clean
```

then ranks by `|Δ|` and takes the top-k prefix. L is the same target−foil logit gap the oracle uses. Gradients cannot go through `ReplacementModel.feature_intervention` (`@torch.no_grad`); `∂L/∂a` is estimated as the **decoder write direction(s) contracted with residual-stream gradients** at `feature_output_hook` (the linearized residual path circuit-tracer already uses). Clean/corrupted activations still come from a paired prompt (required).

That answers: *does first-order AtP ranking of the same features beat interventional greedy?* It does **not** claim we ran Syed’s native-edge algorithm.

### Metric: same or different?

**Same scoring formula; different object; evaluation aligned with MACAG.**

|  | Original Syed EAP | MACAG port (`eap_syed`) |
| --- | --- | --- |
| Ranking signal | AtP on **native edges** | AtP on **CLT feature activations** (same algebra) |
| Task / circuit metric | KL and/or logit difference of the edge circuit | After selection: MACAG v / logit-gap faithfulness of the **feature** prefix |
| Why | Isolate selector vs Game 1 on one candidate pool |  |

The AtP inner product is the original estimator. The **head-to-head number** in our tables is not Syed’s KL(circuit ‖ model); it is v at matched k, plus optional KL rescore. If the foil logit is missing from the graph/oracle map, graph-EAP is marked unavailable rather than silently becoming target-only; Syed EAP requires a corrupted prompt of equal token length.

---

## 3. Attribution-graph influence (Lindsey et al., 2025 / circuit-tracer)

**Original source.** Lindsey et al., *Attribution Graphs* (Transformer Circuits, 2025) and the **circuit-tracer / Neuronpedia** prune used to publish those graphs. Method ID here: `influence`. There is no separate “attention influence” paper: the quantity is **node influence on the attribution graph**, including attention-head error nodes and CLT features. Ranking uses **raw** node influence (`influence_raw`), not the cumulative coverage slider.

### Problem it solves

An attribution graph is dense. Pruning keeps a readable subgraph: nodes (and edges) that account for most of the **indirect influence** onto the prompt’s output logits, computed from local-linear direct effects (the same linearized residual model as attribution patching).

### Original method (brief)

1. Build the prompt-specific attribution graph (direct effects between features, errors, tokens, logits).
2. Convert to **indirect / path influence** onto the logit nodes (normalized adjacency, path sum).
3. **Neuronpedia /** `prune_graph`**:** keep nodes until they cover a fraction of total influence (default node threshold **0.8**), then prune edges similarly, then drop disconnected leftovers.

That is **cumulative coverage thresholding**, not a fixed top-k. The JSON field `influence` is this cumulative score (smallest = strongest retained node). `influence_raw` is the magnitude used for ranking.

### Original evaluation metrics

The attribution-graph work evaluates **interpretability of the pruned graph** (case studies, whether the remaining nodes explain the logit), not a coalitional faithfulness v. Operationally the prune itself is the evaluation: coverage of influence mass, graph size, qualitative circuits. There is no ACDC-style `KL(G ‖ H)` in the prune API.

### Why we port (top-k instead of 80% coverage)

MACAG already consumes a **pruned** circuit-tracer graph. The question is whether Game 1’s search beats simply reading the scores that built that graph, at a **matched evidence size k**. Cumulative 80% coverage yields a different `|E|` on every prompt, so it is not a fair matched-k baseline.

The port (`macag/baselines/influence.py`) is therefore: **rank remaining feature candidates by** `influence_raw` **(larger = stronger) and take the top-k prefix.** Legacy graphs that only store cumulative `influence` are ranked **ascending**. No extra model calls for selection; the set is then scored under MACAG v like every other method.

This is the **primary fair floor**: same graph, same nodes, cheapest selector. It is **not** a port of a different paper’s algorithm; it is the graph’s own importance score used as a selector.

### Metric: same or different?

**Different role, same underlying influence scores.**

|  | Attribution-graph prune | MACAG `influence` |
| --- | --- | --- |
| Cut | Cumulative fraction of influence (e.g. 80%) | Fixed top-k by raw magnitude |
| Reported metric | Graph coverage / qualitative circuits | MACAG v / logit-gap faithfulness of that top-k |
| Why change | Matched-k comparison to Game 1; the 80% rule is a different budget |  |

The **score** is the original circuit-tracer influence. The **number in the table** is MACAG faithfulness, not “fraction of influence retained.”

---

## 4. Shapley values (Shapley, 1953)

**Original paper.** L. S. Shapley, *A Value for n-Person Games*, in *Contributions to the Theory of Games II*, 1953. Method ID here: `shapley`. This is **not** a circuit-discovery paper. In MACAG it is the unique efficient, symmetric, linear, null-player **credit** on the same game (N, v) Game 1 optimizes.

### Problem the original theory solves

In a cooperative game, how should the coalition’s surplus be divided among players so that the division is uniquely determined by natural fairness axioms (efficiency, symmetry, dummy/null player, additivity/linearity)?

### Original method (brief)

Player i’s Shapley value is the average marginal contribution over all orderings:

```
φ_i = (1 / |N|!) · Σ_π [ v(S_π^{<i} ∪ {i}) − v(S_π^{<i}) ]
```

Exact evaluation is `O(|N| · |N|!)`. Practice uses **Monte Carlo permutation sampling** (here: antithetic pairs, default 64 permutations).

### Original evaluation metrics

Shapley (1953) has no ML metrics. Later ML/XAI uses include:

- **efficiency gap** `Σ_i φ_i − (v(N) − v(∅))` (should be ~0 for Shapley),
- fidelity of explanations, cost vs exact enumeration.

In mechanistic interpretability, Shapley-style credit has been used on **neurons/components** under various v; those papers are not the source of our estimator. Our v is MACAG faithfulness, not reconstruction or accuracy drop on a different object.

### Why this is a “port” of the value, not a circuit paper

There is no official “Shapley circuit discovery” implementation to wrap. What we need is gold **per-feature credit under MACAG’s v**, so we can ask how close Game 1’s set is to the top-Shapley features and at what oracle cost.

The implementation (`macag/baselines/shapley_select.py`) estimates φ by permutation sampling on the **same** `coalition_value` Game 1 uses, then **selects the top-k prefix of φ**. That last step is **ours**, not Shapley’s: the axioms define credit, not a unique size-k circuit. Top-k-by-φ is not the v-optimal size-k set when v is non-submodular (brute force, `bruteforce`, measures that gap).

Do **not** confuse this with `attribution/shapley.py`, which estimates Shapley over a **spline-CLT reconstruction** game — a different v on a different object.

### Metric: same or different?

**Same mathematical φ; v and the use as a selector are MACAG-specific.**

|  | Shapley (1953) | MACAG `shapley` |
| --- | --- | --- |
| Object | Abstract coalitional game | Feature nodes, v = MACAG faithfulness |
| Output | Per-player credits summing to `v(N) − v(∅)` | Credits **plus** a top-k ranking used as a baseline set |
| Evaluation | Axioms / efficiency | Those credits, plus the same v-faithfulness of the top-k set vs Game 1; precision@k / Jaccard vs this ranking as gold |

The **credit formula** is original. The **table metric** (faithfulness@k, agreement with Game 1) is MACAG’s. Efficiency of the Monte Carlo estimator is a wiring check (each sampled permutation telescopes); it is not a task metric from 1953.

---

## Summary: what you may claim

| Baseline | Original claim | Fair MACAG claim | Unfair claim |
| --- | --- | --- | --- |
| ACDC | Minimal native **edge** circuit via τ-prune + corrupted patch + KL | Game 1 vs **ported node τ-prune** on CLT features under v | “Beats Conmy et al. / ArthurConmy ACDC” |
| EAP-Syed | Top-k **native edges** by AtP | Game 1 vs **AtP ranking of the same features** | “Beats Syed et al. edge EAP” |
| Influence | Prune graph to ~80% influence mass | Game 1 vs **top-k raw influence** on the same graph | Treating cumulative `influence` descending as strength |
| Shapley | Unique fair credit on (N, v) | Game 1 vs **top-k MC Shapley on the same v** (gold ranking, not optimal set) | “Game 1 approximates Shapley” or wrapping `attribution/shapley.py` |

To compare against **published ACDC or Syed pipelines** as systems, run those implementations and compare **end-to-end** size–faithfulness / KL of the finished circuits — never Jaccard on node IDs. That experiment is separate from this harness.

**Implemented original-pipeline track:** see [`baseline_original_track.md`](baseline_original_track.md).
Method IDs `eap_edge` / `acdc_edge` wrap UFO-101 **auto-circuit** (true Syed EAP +
Conmy ACDC on the factorized edge graph) and write `macag_original_baselines.json`.

---

## If we used the original methods as published

This section does not replace the ports. It states what we would **lose**, what we would **gain**, and **whether the published method can even be the Game 1 baseline**.

The test is always: can the original algorithm take MACAG’s candidate feature nodes, return a set, and be scored under the same \(v\) at matched \(k\)? If not, it is a **pipeline** comparison (two different circuit-discovery systems), not a **selector** comparison.

### ACDC (ArthurConmy / Conmy et al. Algorithm 1)

**Can we run it as-is?** Yes as a separate experiment (wrap the published edge-pruner on Gemma/Llama with corrupted prompts). **No** as a drop-in Game 1 baseline: it does not accept CLT feature nodes, does not speak MACAG’s four-mode oracle, and does not return a subset of the same IDs.

|  |  |
|---|---|
| **Miss** | Selector isolation. No shared candidate pool, so no Jaccard / precision@k vs Game 1 or Shapley-gold. Cannot attribute a win to “greedy vs τ-prune” — universe (edges vs features), ablation (corrupt vs zero), search unit (edge vs node), and metric (KL vs \(v\)) all change at once. Path-dependent edge pruning also need not produce a size-\(k\) set, so matched-budget tables break unless you add extra τ search. |
| **Gain** | External validity vs the **published ACDC pipeline**. You can say whether MACAG+CLT circuits recover behavior more cheaply than Conmy’s native-edge circuits, using KL and size—the metrics reviewers associate with that paper. |
| **Why not as the primary baseline** | Game 1’s contribution is allocation over an **already-built feature graph**. Original ACDC never sees that graph. Using it as the main head-to-head would answer a different RQ and would make “beats ACDC” easy to overclaim. |
| **When to use the original** | Optional second table: end-to-end KL / logit-gap vs circuit size, **no node-ID overlap**. Do not treat `acdc_native` as a substitute for Algorithm 1. |

### EAP-Syed (native-edge attribution patching)

**Can we run it as-is?** Yes on native edges (two forwards + one backward, top-\(k\) edges). **No** as Game 1’s feature selector: Syed’s graph is heads/MLPs/residual edges, not CLT `(layer, pos, feat)` nodes.

|  |  |
|---|---|
| **Miss** | The question “does interventional greedy beat first-order scores **on these features**?” Native-edge AtP ranks different objects, so agreement with Shapley-gold and Game 1 is undefined. You also lose a cheap, paired ranking signal inside the same harness (AtP on features is one extra clean/corrupt + backward). |
| **Gain** | Fidelity to Syed/Nanda: the published algorithm, the published graph, and their KL / logit-difference circuit metrics. Stronger “vs the EAP paper” sentence if you only compare **finished-circuit** faithfulness vs size/cost. |
| **Why not as the primary baseline** | Attribution graphs are already AtP-like on residual directions. The scientifically tight control is: same features, replace Game 1’s search with Syed’s **formula**. Replacing the whole graph construction confounds encoder, edges, and selector. |
| **When to use the original** | Same as original ACDC: a pipeline table. Keep `eap_syed` for selector isolation. Keep `eap` / `eap_graph` labeled as path-effect, never as Syed. |

### Attribution-graph influence (Lindsey et al. / circuit-tracer prune)

**Can we run it as-is?** Yes — it is already how graphs are built (`prune_graph`, ~80% influence coverage). **Not** as a matched-\(k\) selector without changing the cut.

|  |  |
|---|---|
| **Miss** | A fair size-matched comparison. Coverage-0.8 yields a different \(\lvert E\rvert\) every prompt (often much larger than Game 1’s budget). Game 1 would look better on faithfulness-per-feature and worse or better on raw faithfulness for accidental budget reasons, not because search is better. Cumulative `influence` ranked descending is the wrong order. |
| **Gain** | Exact Neuronpedia/circuit-tracer prune: the circuit the frontend actually shows. Useful as a “default published graph” reference (completeness of the un-MACAG’d prune). |
| **Why we still port (top-\(k\))** | The original method **is** the right importance score; only the **cut** must change so \(k\) matches Game 1. That is a smaller adaptation than ACDC/EAP. Using raw influence top-\(k\) **is** using the paper quantity; dropping the 80% rule is required for the table, not a different theory. |
| **When to use the original cut** | Report prune size and coverage as graph-construction stats, not as a baseline row next to Game 1@\(k=8\). |

### Shapley (1953)

**Can we run “the original” as-is?** The 1953 paper is a definition, not a circuit repo. Exact Shapley on \(\lvert C\rvert \sim\) hundreds of features is intractable. Any practical method is already an estimator. Using some other paper’s Shapley (neurons, tokens, reconstruction) would be a **different \(v\)**, not more original.

|  |  |
|---|---|
| **Miss** | If we imported an off-the-shelf XAI Shapley (e.g. input-token or neuron Shapley, or `attribution/shapley.py` reconstruction): gold ranking on the **wrong game**. Agreement with Game 1 would mix “greedy vs credit” with “faithfulness vs reconstruction.” If we skipped Shapley entirely: no axiomatically unique credit, no cost-vs-gold, no interaction diagnostic (greedy marginal vs \(\phi\)). |
| **Gain** | Exact enumeration only on tiny pools (the brute-force baseline already covers optimal **sets**, which is a different question than optimal **credits**). A textbook estimator on MACAG \(v\) **is** the original method applied to our game. |
| **Why the port is the original method** | Shapley is defined relative to \((N,v)\). Choosing MACAG’s \(v\) and \(N=\) feature candidates **is** using Shapley as-is. What is extra (and ours) is treating top-\(k\) of \(\phi\) as a circuit; say so. |
| **When not to use another Shapley** | Never wrap `attribution/shapley.py` or a generic explainer as this baseline. |

### Cross-cutting: ports vs originals as two tables

| Design | What you can conclude | What you cannot |
|---|---|---|
| **Ports (primary)** | Game 1 is a better/worse **selector** than AtP ranking, τ-prune, influence, or top-Shapley **on CLT features under \(v\)** | That MACAG beats the ACDC or EAP **papers** as published systems |
| **Originals (optional)** | MACAG+CLT vs published **pipelines** on KL / size / cost of the finished circuit | That greedy search, rather than CLTs or logit-gap, caused the difference; any node-set overlap |

Use originals **in addition to** ports only if you want the pipeline claim, and keep the tables separate. Replacing the ports with originals would **miss** the only comparison that tests MACAG’s actual contribution (allocation on a feature graph). Keeping only the ports **misses** a reviewer-facing “same algorithm as the 2023 repos” check, which is optional and more expensive, not required for a valid Game 1 paper.

---

## Why MACAG vs faster ACDC / EAP-style methods

Game 1 is **O(|candidates| × |E|)** interventional forwards. Influence, feature AtP, ported ACDC, and native-edge ACDC/EAP finish much faster because they **rank or prune once**; they do not search under the keep/remove game. On an unpruned ~2k-node CLT graph, dual-freeze Game 1 is hours per prompt. Matched-k F being comparable is expected if you **force** Game 1’s size onto a cheap ranking: you are scoring prefixes of that order, not asking those methods to choose a sparse stop.

MACAG is not “faster circuit discovery.” It is a different **product**: a **small, intervention-tested set** with an explicit stop, dual-freeze diagnosis, and (Game 2) contrastive structure.

### What the other methods buy you

| Method | Cheap because | What you get |
| --- | --- | --- |
| Influence / graph EAP | scores already in the graph | a ranking, not a tested circuit |
| Feature AtP (`eap_syed`) | 1–2 extra forwards + backward | ranking of the **same** CLT nodes |
| Ported ACDC | one top-down τ pass | a **density** knob; high F often needs many nodes |
| Native edge ACDC/EAP | AutoCircuit on heads/MLPs | a **different object** (edges, corrupt patch) — Track B; see [`baseline_original_track.md`](baseline_original_track.md) |

### What MACAG is for (if F@k is similar)

1. **Natural size, not a budget.** Game 1 stops when extra nodes stop paying (`raw_relative`). Rankings need you to pick \(k\). ACDC needs \(\tau\). That is the difference for unlearning / steering / audit: *how many features do I actually clamp or delete?* Dallas full KL Game 1: **5–6** nodes; ported ACDC at high F was **dense** (167 nodes at \(\tau=0.1\)). Details: [Dallas–Austin case study](#dallasaustin-case-study-llama-32-1b).
2. **S and N separately, not only F.** Influence/AtP do not optimize knockout vs reconstruction. High-S / low-N sets look good on F and **fail unlearning**. Game 1 (and \(\alpha\)) chooses that tradeoff; baselines are scored after the fact.
3. **Dual freeze is a diagnosis, not a hyperparameter.** Frozen vs unfrozen recoverable range is the auditor claim (features vs attention). Cheap rankers do not produce that pair unless you run them twice and still lack a mediation verdict.
4. **Game 2 is not a faster ACDC.** Disjoint \(E_y\) / \(E_{\text{foil}}\) is the steering claim. No baseline in the suite solves that game.
5. **Same oracle as the claim.** Ports are fair **selector** controls (Track A). Native ACDC/EAP are a **pipeline** comparison (Track B). MACAG’s advantage is not “beats Conmy on edges”; it is allocation **on the CLT graph you already built**.

### Practical cost recipe

- **Discovery / screening:** influence, AtP, or a **large** prefilter + loose \(\varepsilon\).
- **The circuit you ship:** unbudgeted Game 1 on that pool (or full pool for case studies).
- **Do not** use Game 1 as a drop-in replacement for “rank every edge in 30s.”

If matched-k F is close, MACAG’s remaining advantages are **smaller actionable \(E^*\)**, **S/N/freeze diagnostics**, and **Game 2** — not wall-clock. If those three do not matter for the application, MACAG is the wrong tool and the faster rankers are the right ones.

Prefilter note (Dallas KL Game 1): `top_k=20` was ~6× fewer oracle calls but changed the set (frozen Jaccard vs full \(=0.20\)); `top_k=50` + \(\varepsilon=0.01\) recovered F (even overshot frozen) without recovering the same \(E^*\) (Jaccard \(=0.13\)). Treat prefilter as a speed/identity tradeoff, not a free approximation. Full tables: [Dallas–Austin case study](#dallasaustin-case-study-llama-32-1b).

Canonical paper protocol: [`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md). Full application tuning: [`APPLICATIONS_METRICS.md`](APPLICATIONS_METRICS.md).

---

## Dallas–Austin case study (Llama-3.2-1B)

One prompt, one unpruned CLT graph, every selector in this document. This is **not** the MIB campaign table ([`TMLR_EVAL_RECIPE.md`](TMLR_EVAL_RECIPE.md) / [`run_todo_v4.md`](run_todo_v4.md)). It is the case study that motivated the v4 protocol (unpruned export, unbudgeted Game 1, logit-gap selection, KL as rescore, influence-only in Pass A) and the application split (\(\alpha \in \{0, 0.5, 1\}\)).

Artifacts live under

```
/gscratch/$USER/macag_dallas_austin_llama/llama32-524k/dallas-austin/
  graphs/dallas-austin.json
  kl_divergence/          # job 13913 Track A + Track B + KL Game 1 ablations
  logit_gap/              # jobs 14031–14035 Game 1 only (does not overwrite KL)
```

Job **13913** (4-node method array) **did run Game 2 and Shapley** on this prompt, in parallel with Game 1 / influence / EAP / ACDC / edge methods. Both shards hit the gp-2 **24h cap** before writing JSON (`run_macag` / `run_baselines` originally flushed only at process end). Progress is in `logs/method.game2.wid1.out` and `logs/method.shapley.wid5.out`.

Completion path (in flight):

- **14040** — from-scratch rerun of Game 2 + Shapley (old code, no checkpoint). **Leave running for Shapley.** Its Game 2 shard restarts ABR from empty on 2028 nodes and will not finish in 24h.
- ABR checkpoint harvested from the 13913 log: `kl_divergence/macag_game2_abr.ckpt.json` (iter 1 foil, \|E_y\|=39, \|E_foil\|=29). Full-universe Game 2 resume (**14052**) was cancelled: another 24h on 2028 nodes still cannot complete ABR 2–4 + FP.
- **Game 2 PF50**: singleton prefilter `top_k=50`, tagged JSON (`macag_game2_abr_prefilter50.json`, plus `_alpha1` when \(\alpha\neq 0.5\)). Does **not** overwrite the harvested full-universe ckpt. ABR only (`GAME2_SOLVERS=abr`). **14054/14055** failed immediately (`run_macag` missing `--no-cache`). **14060** KL \(\alpha=0.5\); **14061** logit-gap \(\alpha=1\).
- **14056** Shapley-only remaining after 14040. **14057** finalize after 14056.

Log-reconstructed full-universe Game 2 / Shapley numbers below stand until JSON files land. PF50 Game 2 is a **different** candidate pool (same identity caveat as Game 1 PF50).

### Setup (shared by every Dallas run)

| Knob | Value |
| --- | --- |
| Prompt | `Fact: The capital of the state containing Dallas is` |
| Target / foil | ` Austin` / ` Texas` (token ids 19816 / 8421 in the Syed AtP extras) |
| Corrupted prompt (Track A Syed + Track B) | `Fact: The capital of the state containing Houston is` (same token length) |
| Model / CLT | `meta-llama/Llama-3.2-1B` / `mntss/clt-llama-3.2-1b-524k` |
| Graph export | `node_threshold=1.0` (unpruned), BOS-prepended, 11 prompt tokens |
| Graph size | **2200** nodes, **1,140,436** links |
| Node mix | 2028 `cross layer transcoder` (MACAG candidates), 151 MLP reconstruction error, 11 embedding, 10 logit |
| Game 1 | dual-freeze (`--freeze-mode both`), `budget=None`, \(\lambda=0.02\), `stop_metric=raw_relative`, `connected=false` |
| Stop rule | add while \(\Delta F_{\mathrm{next}} \ge \varepsilon \cdot \Delta F_{\mathrm{first}}\) (first feature, **not** previous step) |
| Default \(\varepsilon\) | \(0.1\); tighter ablations use \(0.01\) |
| Default \(\alpha\) | \(0.5\) unless named otherwise |
| Feature ID | `{layer}_{feat}_{ctx_idx}`; last content token is `ctx_idx=10` |

Oracle modes: `all` (clean), `empty` (all 2028 features ablated; error/attention floor remains), `keep_only`(\(E\)), `remove`(\(E\)). Then \(S = s_{\mathrm{keep}}-s_{\mathrm{empty}}\), \(N = s_{\mathrm{all}}-s_{\mathrm{remove}}\), \(F = \alpha S + (1-\alpha) N\), recoverable range \(R = s_{\mathrm{all}}-s_{\mathrm{empty}}\).

**KL score** is \(s = -\mathrm{KL}(P_{\mathrm{full}}\|P_{\mathrm{int}})\), so \(s_{\mathrm{all}}\equiv 0\). Keep-only of a sparse \(E^*\) is **not** “the circuit copies the model” unless \(\mathrm{KL}(\mathrm{full}\|\mathrm{keep})\) is near 0.

**Logit-gap score** is \(\mathrm{logit}(\texttt{Austin})-\mathrm{logit}(\texttt{Texas})\). Clean model: \(s_{\mathrm{all}} = 1.625\) (CLT Game 1) / \(1.662\) (native-edge AutoCircuit scorer; small tokenizer/path difference).

### Track A — KL Game 1, full universe (job 13913)

Selection on KL, \(\alpha=0.5\), \(\varepsilon=0.1\), no prefilter, 2028 candidates. This is the “does a tiny feature set match the full next-token distribution?” run. It does not.

| Leg | \|E\| | F | S | N | keep_only \(= -\mathrm{KL}(\mathrm{full}\|\mathrm{keep})\) | remove | empty | \(R\) | oracle calls | iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Frozen** | 5 | 6.75 | 2.44 | 11.06 | **−6.88** (KL ≈ 6.88 nats) | −11.06 | −9.31 | 9.31 | 24,310 | 5 |
| **Unfrozen** | 6 | 10.08 | 9.72 | 10.44 | **−5.41** (KL ≈ 5.41 nats) | −10.44 | −15.13 | 15.13 | 28,354 | 6 |

Frozen \(E^*\): `0_12390_10`, `0_25454_10`, `15_16228_10`, `3_18384_10`, `9_25557_10`.  
Unfrozen \(E^*\): `0_12232_1`, `0_18332_8`, `0_25454_10`, `0_3193_7`, `15_16228_10`, `4_7513_9`.  
Shared: `{0_25454_10, 15_16228_10}` (Jaccard \(2/9 = 0.22\)).

Read this as: Game 1 found a **5–6 node** KL *gain* vs empty, not a reconstruction. Keep-only is still 5–7 nats from the full model. Necessity is large because `remove` is *worse* than empty (ablating \(E^*\) while leaving the other 2020 features on hurts KL more than ablating everything). Sufficiency is modest on frozen (\(S=2.44\)): those five nodes only close a fraction of the empty floor.

Hub feature **`0_25454_10`** (layer 0, last token) is in both legs. Singleton F ≈ 4.63 frozen / 4.77 unfrozen — most of the first-step \(\Delta F\) that the \(\varepsilon=0.1\) stop is relative to.

### Track A — KL prefilter × \(\varepsilon\) (same graph, new output stems)

Prefilter ranks candidates by **singleton** Game 1 utility, keeps top-\(k\), then runs greedy only on that pool. Full-universe JSON is never overwritten (`macag_game1_prefilter{k}_eps{ε}.json`).

| Config | wall | frozen \|E\| / F / calls | unfrozen \|E\| / F / calls | speedup vs full (frozen calls) |
| --- | --- | --- | --- | --- |
| Full (no PF, \(\varepsilon=0.1\)) | ~hours (24h job, Game 1 finished) | 5 / 6.75 / 24,310 | 6 / 10.08 / 28,354 | 1× |
| PF20, \(\varepsilon=0.1\) | 2,095 s (~35 min) | **1** / 4.69 / 4,098 | 5 / 8.44 / 4,230 | **5.9×** |
| PF50, \(\varepsilon=0.01\) | 2,797 s | 13 / 7.66 / 5,178 | 8 / 9.52 / 4,788 | 4.7× |
| PF500, \(\varepsilon=0.01\) | 14,932 s (~4.1 h) | 24 / 8.61 / 27,460 | 19 / 12.75 / 22,680 | **0.89× (slower)** |

Jaccard of \(E^*\) vs full Game 1:

|  | frozen J | unfrozen J | frozen shared |
| --- | --- | --- | --- |
| PF20 vs full | **0.20** | **0.10** | only `0_25454_10` |
| PF50 vs full | 0.13 | 0.08 | `0_25454_10`, `15_16228_10` |
| PF500 vs full | 0.16 | 0.04 | four of five frozen nodes; still misses `9_25557_10` |
| PF50 vs PF500 | 0.42 | 0.13 | 11 frozen nodes overlap |

PF20 frozen **collapsed to one node** (`0_25454_10`): \(\varepsilon=0.1\) vs that singleton’s huge first-step \(\Delta F\) stops immediately. F drop vs full: −2.06 frozen / −1.64 unfrozen.

PF50 + tight \(\varepsilon\) **overshoots frozen F** (+0.91) with a *different* 13-node set. Keep-only KL is actually **better** than full Game 1 (5.25 vs 6.88 nats) — Game 1’s F-stop is not minimizing keep-only KL.

PF500 is not an approximation of full Game 1: more nodes, similar or higher call count, Jaccard still \(\le 0.16\).

**Why PF20 cannot recover full \(E^*\).** Singleton ranks of the frozen full-run nodes among 2028 candidates (pool dump `macag_game1_prefilter20_pool.json`):

| Node | singleton rank | singleton utility | in top-20? |
| --- | --- | --- | --- |
| `0_25454_10` | **1** | +4.605 | yes |
| `0_12390_10` | 20 | +0.159 | on the cutoff |
| `15_16228_10` | 19 | +0.160 | yes |
| `3_18384_10` | **174** | −0.014 | **no** |
| `9_25557_10` | **531** | −0.026 | **no** |

Unfrozen full \(E^*\): ranks 1, 6, 8, **37**, **158**, **683**. Three of six nodes are outside top-20; `15_16228_10` is a *negative* singleton and only pays inside a coalition. Prefilter by singleton utility **throws away complementary nodes**. That is the identity/speed tradeoff, not a bug in the dump.

Post-hoc keep-only KL ranking among KL circuits (lower = closer to full model), from `macag_game1_kl_circuit_compare.json`:

- Frozen closest keep: PF50 (KL 5.25, \|E\|=13) > PF20 (6.72, \|E\|=1) > full (6.88, \|E\|=5).
- Unfrozen closest keep: full (5.41, \|E\|=6) > PF20 (7.06) > PF50 (7.09).

Sparse KL of 4–7 nats is still not a full-distribution match. Do not advertise any of these as “the model distilled into five features.”

### Track A — ports on the same KL frozen oracle

Same graph, same 2028 candidates, same KL kwargs (`freeze_attention=true`, so comparable to the **frozen** Game 1 leg). Budget 128 for prefix curves. Shapley was run on this oracle (job 13913 wid5); see below — the sidecar JSON was not flushed before the 24h kill.

**Influence** (graph `influence_raw`, 0 selection forwards):

- Ranking starts `0_25454_10`, `11_30856_10`, `15_16228_10`, `11_14930_10`, `14_31510_10`.
- F@k: k=1 → 4.61; **k=5 → 5.52**; k=6 → 5.83; k=8 → 5.89; k=20 → 6.77; k=32 → 7.44; **k=128 → 9.12**.
- AUC (trapezoid, k=0..128) = **8.26**.
- Matched to Game 1’s frozen \|E\|=5: influence F **5.52 vs Game 1 F 6.75**. Game 1 wins at natural size. Influence only catches Game 1’s F once the prefix is tens of nodes.

**Feature AtP / `eap_syed`** (3 forwards + 1 backward; Houston corrupt; decoder·residual-grad):

- Ranking starts `13_31952_10`, `12_7970_10`, `11_14930_10`, `12_21369_10`, `9_25557_10` — **not** the Game 1 hub first.
- F@k: k=1 → 0.07; **k=5 → 0.11**; k=8 → 0.39; k=12 → 0.48; then **negative** F around k=19–27; **k=128 → 0.30**.
- AUC = **0.47**.
- First-order AtP on these CLT features is a **weak selector** under KL v. That is the fair “does the Syed formula beat greedy on the same nodes?” answer on this prompt: **no**.

**Ported ACDC** (top-down \(\tau\) on feature nodes, same v; 24,232 oracle calls — same order as a full Game 1):

| \(\tau\) | \|kept\| | F (KL v) |
| --- | --- | --- |
| 0.001 | 1919 | 12.32 |
| 0.01 | 1850 | 11.87 |
| 0.05 | 766 | 12.35 |
| 0.1 | 167 | **12.45** |
| 0.2 | 50 | 7.77 |
| 0.5 | **14** | **5.56** |

\(\tau=0.5\) (closest to Game 1’s size): 14 nodes, F=5.56, S=4.09, N=7.03, keep_only=−5.22 — **below** Game 1’s 5-node F=6.75. High-F ACDC is the **dense** end (167–1919 nodes). That is the “ACDC at high F was dense” claim with numbers. Overlap with Game 1 frozen \(E^*\) at \(\tau=0.5\): `0_25454_10` is kept; most of Game 1’s other four are not the ACDC 14.

### Track A — KL Game 2 (job 13913 wid1; log-reconstructed)

Ran. Config: `python -m macag.cli.run_macag game2`, `score_kind=kl_divergence`, solver **ABR then FP**, `budget=None`, \(\alpha=0.5\), \(\lambda=0.02\), \(\beta=0.2\), `abr-iters=4`, 2028 candidates. Log: `logs/method.game2.wid1.out` (~12 MB of per-candidate tqdm).

ABR **iteration 1 of 4** started 2026-08-13 18:41:15. There is **no Game 1-style \(\varepsilon\) stop**: each ABR half (Austin \(E_y\), then Texas \(E_{\mathrm{foil}}\)) is a full greedy grow until marginal gain dies. At ~1.1 s/candidate × 2028, each added node costs ~15–22 min.

**Austin side (\(E_y\))** finished 39 nodes (18:55 → 07:38 next day). First five adds are **exactly frozen Game 1 \(E^*\)**, same order:

| step | node | gain | wall |
| --- | --- | --- | --- |
| 1 | `0_25454_10` | 4.732 | 18:55 |
| 2 | `15_16228_10` | 0.464 | 19:09 |
| 3 | `9_25557_10` | 0.449 | 19:23 |
| 4 | `3_18384_10` | 0.511 | 19:38 |
| 5 | `0_12390_10` | 0.589 | 19:53 |

Then it kept going (`0_28426_5`, `11_14930_9`, …) down to gain **0.003** at \|E_y\|=39. Game 1’s \(\varepsilon=0.1\) vs first-step \(\Delta F\) would have stopped at 5; Game 2 has no such brake.

**Texas side (\(E_{\mathrm{foil}}\))** started ~08:20 on 08-14 and reached **29 nodes** (last add `0_2284_4` at 18:32). The 24h kill hit during `ABR[1] foil sweep=30 |E|=29` (~700/2028). FP never started. **No `macag_game2_abr.json`.**

Overlap at kill: **18 shared nodes**, Jaccard \(18/(39+29-18)=0.36\). The first eight \(E_y\) nodes are all in \(E_{\mathrm{foil}}\). \(\beta=0.2\) did **not** produce disjoint circuits.

That is the steering-doc claim in numbers: **KL is target-free**, so both sides climb the same KL-gain ridge (`0_25454_10` first on both, gain ≈ 4.62–4.73). A working foil circuit should move *away* from the clean distribution; KL Game 2 instead copies the Austin KL set onto Texas. Do not score Game 2 on KL. Logit-gap Game 2 on this prompt was not in 13913 (`SCORE_KIND=kl_divergence`).

14040 restarted ABR from empty (old binary, no resume). Full-universe resume was cancelled. The finishing path is **PF50 Game 2** (`macag_game2_abr_prefilter50.json`), not another 2028-node ABR.

### Track A — KL Shapley (job 13913 wid5; log)

Ran. `run_baselines --methods shapley`, 64 antithetic permutations, seed 0, budget 128 for the top-k prefix curve, same KL frozen oracle. Log: `logs/method.shapley.wid5.out`.

| checkpoint | time (log) |
| --- | --- |
| start (model load) | 2026-08-13 18:41 |
| 8/64 | 21:21 |
| 16/64 | 00:24 (08-14) |
| 24/64 | 03:27 |
| 32/64 | 06:29 |
| 40/64 | 09:34 |
| 48/64 | 12:42 |
| **56/64** | **15:47** |

~3.0–3.1 h per 8 permutations (~23 min/perm) on 2028 features. 64/64 would have been ~3 h after 56/64; the 24h job ended first. **No `macag_baselines_shapley.json`** — φ, efficiency gap, and F@k vs Game 1 were never flushed. 14040 restarted from permutation 0.

So: Shapley **was** the gold-ranking baseline on Dallas KL; the table row is waiting on 14040 (may finish 64/64) or the Shapley-only remaining job after it. Influence/AtP/ACDC comparisons above do not substitute for it.

### Track A — logit-gap Game 1 (jobs 14031–14035)

Same graph, **new** `logit_gap/` kwargs (KL artifacts untouched). Clean gap \(s_{\mathrm{all}}=1.625\). No full-universe logit-gap Game 1 (24h cap); all runs used a prefilter.

Frozen empty ≈ 0 to +0.14: with attention frozen, ablating all CLT features **kills** the Dallas→Austin fact (\(R \approx +1.5\) to \(+1.7 > 0\)). Feature-mediated on this prompt.

Unfrozen empty ≈ **+4.0 to +4.3 > all**: ablating every CLT feature *increases* Austin−Texas because attention/error paths still carry the fact. \(R < 0\). Unfrozen \(F\) can look fine while relative faithfulness is meaningless; auditor verdict is the **frozen** sign of \(R\).

#### \(\alpha\) split at PF20, \(\varepsilon=0.1\) (the application result)

Same 20-node pool, only \(\alpha\) changes. Frozen:

| \(\alpha\) | Role | \|E\| | F | S | N | keep_only | remove | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | pure N (unlearning) | 4 | 7.53 | **−0.39** | **7.53** | **−0.25** | **−5.91** | 2,228 s |
| **0.5** | paper F | 6 | 5.40 | 4.02 | 6.78 | **4.06** | **−5.16** | 2,405 s |
| **1** | pure S (copy / steer) | 4 | 4.66 | **4.66** | **−0.25** | **4.69** | **+1.88** | 2,570 s |

- \(\alpha=0\): `remove` flips the gap to −5.91 (**Texas wins**). `keep_only` ≈ 0 — the knockout set does **not** reconstruct Austin. \(E^*=\) `0_25454_10`, `0_28426_5`, `1_26758_5`, `9_25557_10`.
- \(\alpha=1\): `keep_only=4.69` **overshoots** clean 1.625. `remove=+1.88` — ablating \(E^*\) does **not** kill the fact. \(E^*=\) `0_13365_8`, `0_16582_4`, `0_2776_8`, `0_3622_9`.
- \(\alpha=0.5\): both jobs at once (keep overshoots, remove flips). \(E^*=\) `0_13365_8`, `0_13931_2`, `0_2118_9`, `0_25454_10`, `1_15735_4`, `9_25557_10`.

Frozen Jaccard: \(\alpha=0\) vs \(\alpha=1\) = **0**. Knockout and copy circuits on this prompt are **disjoint**. \(\alpha=0.5\) shares 2/4 with \(\alpha=0\) and 1/4 with \(\alpha=1\). Default F is a mixture set, not “the” circuit.

Unfrozen \(\alpha=0\): N=6.94, keep≈0.06, remove=−5.31. Unfrozen \(\alpha=1\): S=5.63, keep=**9.94**, remove=+1.75. Same qualitative split; unfrozen empty floor makes S of a copy set look even larger.

#### Logit-gap prefilter / \(\varepsilon\) at \(\alpha=0.5\)

| Config | frozen \|E\| / F / keep / remove / calls | unfrozen \|E\| / F / keep / remove / calls | wall |
| --- | --- | --- | --- |
| PF20, \(\varepsilon=0.1\) | 6 / 5.40 / 4.06 / −5.16 / 4,258 | 10 / 4.11 / 6.59 / −4.06 / 4,350 | 2,405 s |
| PF50, \(\varepsilon=0.01\) | 18 / 6.28 / 5.38 / −5.50 / 5,518 | 12 / 4.45 / 6.63 / −4.69 / 5,104 | 2,753 s |
| PF500, \(\varepsilon=0.01\) | **32** / **9.74** / **10.56** / **−7.28** / 35,004 | 18 / 6.94 / 12.38 / −4.13 / 21,718 | 16,232 s (~4.5 h) |

Bigger pool + tighter stop → larger \(E^*\) and larger F. Frozen PF500 keep_only=10.56 is a strong **copy** (well above 1.625) **and** remove=−7.28 is a strong **knockout**. That is \(\alpha=0.5\) doing both because the pool is large enough to pick mixed nodes — not because PF500 recovered a unique true circuit (Jaccard vs PF20 frozen = 0.06).

PF20 vs PF50 frozen Jaccard = 0.20; PF20 vs PF500 = 0.06. Same identity warning as KL.

### Track B — native-edge ACDC / EAP (UFO-101 auto-circuit)

Factorized Q/K/V residual graph, **195,865** edges. Corrupt = Houston prompt. **Do not Jaccard against Game 1.** Report F/S/N and KL of the *edge* circuit only.

**`eap_edge`** (true Syed AtP): 14 forwards + 1 backward, 31.5 s. Native ranking, no single circuit size. Prefix sweep (corrupt-patch the complement):

| k | circuit logit-gap | KL(full ‖ circuit) |
| --- | --- | --- |
| 1–32 | ≈ **2.97** | ≈ **0.48** |
| 64 | 2.95 | 0.47 |
| 256 | 2.20 | 0.32 |
| 1024 | 2.07 | 0.23 |
| 4096 | 1.48 | 0.24 |
| 195,865 (all) | 1.66 | 0 |

Small AtP edge sets sit at gap **2.97**, which is exactly the **empty/corrupt** gap on this native scorer (`all=1.66`, `empty=2.97`). Keep-only of k=8: S≈0, N=−0.19, F=−0.09 (`macag_original_edge_logit_gap_faithfulness_eap_edge.json`). A tiny native EAP circuit here behaves like “leave the model in the Houston-patched world,” not like restoring Dallas→Austin. Gap only returns to clean ~1.66 when almost every edge is kept. That is a **pipeline** finding, not a selector-vs-Game-1 finding.

**`acdc_edge`** (true Conmy \(\tau=0.1\), KL prune, resample-corrupt): 5,898 s (~1.6 h). Final circuit **3 edges**: `A15.7->Resid End`, `MLP 12->MLP 13`, `MLP 13->Resid End`. Native scores: logit-gap **2.97**, KL **0.48** — same empty-like point as small EAP. Logit-gap S/N/F rescore: S≈0, N=−0.54, F=−0.27. A 3-edge ACDC circuit is not a Dallas→Austin mediator under keep/remove on this graph.

Track B takeaway on this prompt: published native-edge methods return **tiny or ranked-dense** objects whose keep-only gap matches the corrupt/empty floor, while CLT Game 1’s logit-gap \(\alpha=0.5\) PF20 frozen set (6 nodes) actually **overshoots** clean keep-only (4.06 vs 1.63) **and** flips remove (−5.16). Different objects, different interventions — which is why the two tables must stay separate.

### Incomplete on this prompt

| Artifact | Status |
| --- | --- |
| Game 2 ABR (KL, 2028 nodes) | **ran** 13913 wid1; ABR iter 1/4; \|E_y\|=39, \|E_foil\|=29, 18 overlap; ckpt harvested; **14040** rerunning from empty (will not finish); full-universe resume **14052 cancelled** |
| Game 2 ABR PF50 (KL, \(\alpha=0.5\)) | **14060** (14054 died on missing `--no-cache`) — `kl_divergence/macag_game2_abr_prefilter50.json` |
| Game 2 ABR PF50 (logit_gap, \(\alpha=1\)) | **14061** (14055 same crash) — `logit_gap/macag_game2_abr_prefilter50_alpha1.json` |
| Game 2 FP | not started; run after PF50 ABR JSON if needed |
| Shapley 64-perm (KL) | **ran** 13913 wid5 to **56/64**; **14040** in flight; **14056** Shapley-only after 14040 |
| `kl_divergence/macag_baselines.json` merge | finalize **14057** after 14056 |
| Full-universe logit-gap Game 1 (no prefilter) | not run (cancelled 14029/14030; replaced by PF20 \(\alpha\) grid) |
| Shapley on **logit_gap** | not run (13913 used `SCORE_KIND=kl_divergence`) |

### What this case study supports in the rest of this doc

1. **Ports vs originals as two tables.** Track A influence/AtP/ACDC and Track B native edges are not interchangeable rows. Native 3-edge ACDC F is not a Game 1 competitor.
2. **Matched-k F can look “close” while the product is different.** Influence F@5=5.52 vs Game 1 F=6.75 is the fair selector gap. Influence F@128=9.12 is a different budget.
3. **Feature AtP can fail on the same nodes.** `eap_syed` AUC 0.47 vs influence 8.26 vs Game 1 natural-size 6.75.
4. **Ported ACDC high F ⇒ dense.** \(\tau=0.1\) keeps 167 features; Game 1 stopped at 5.
5. **\(\alpha\) is the application, not a regularizer.** Frozen PF20: \(\alpha=0\) unlearns, \(\alpha=1\) copies, Jaccard 0 between those \(E^*\).
6. **Prefilter is not a free approximation.** Singleton top-20 drops Game 1 nodes at ranks 174 and 531; Jaccard vs full ≤ 0.20.
7. **Auditor is frozen \(R\), not unfrozen \(E^*\).** Frozen \(R>0\); unfrozen empty > all.
8. **KL Game 1 ≠ dictionary copy.** Keep-only 5–7 nats. Use KL as rescore / collateral, as the TMLR recipe already requires.
9. **KL Game 2 is not contrastive.** Dallas ABR on KL grew \(E_y\) to 39 and \(E_{\mathrm{foil}}\) to 29 with **18 shared nodes** (first eight Austin nodes all on the Texas side). \(\beta=0.2\) did not disjoint them. Select Game 2 on logit-gap, \(\alpha=1\).
10. **Unbudgeted Game 2 / Shapley on ~2k nodes do not finish in 24h.** Game 2 has no \(\varepsilon\) stop; Shapley was 56/64 at the wall. Dallas finishing path is **Game 2 PF50** (tagged JSON) plus Shapley resume, not another full-universe ABR.

Paper MIB v4 should **not** copy Dallas KL selection. Dallas logit-gap + \(\alpha=0.5\) + no prefilter is the campaign protocol; the KL / PF / \(\alpha\) grid is how we learned that.

---

## Applications: unlearning, steering, auditor

F/S/N are the right **causal vocabulary**, but default **F** (\(\alpha=0.5\)) is the wrong **headline** for all three apps. Paper tables may still use F; product claims should not.

| Application | Game | Headline metric | Select on | \(\alpha\) | Run type |
| --- | --- | --- | --- | --- | --- |
| Surgical unlearning | Game 1 | Necessity **N** (unlearning drop) | logit_gap if a substitute token is known; else raw `logit` / `prob` of the forgotten token | **0** (pure N) | Goal-oriented on knockout; budgeted k-sweep only for baselines |
| Contrastive steering | Game 2 | Keep-only gap / argmax flip + **overlap_rate** | logit_gap(\(y\), \(y_{\text{foil}}\)) | **1** (pure S) | Goal-oriented until both clamps flip; budgeted for matched-k vs ActAdd |
| Transcoder / model auditor | Game 1 frozen **and** unfrozen | **Recoverable range** \(R = S_{\text{all}}-S_{\text{empty}}\) | logit_gap for task mediation; optional second Game 1 on KL for dictionary completeness | 0.5 is fine (diagnosis is all vs empty, not \(E^*\)) | Core run needs no budget; optional goal-oriented \(|E^*|\) as circuit complexity |

**KL is evaluation, not the Game 1 / Game 2 objective** for these three apps. It is the collateral / dictionary-completeness check. Successful foil steering *should* raise KL.

Oracle modes: `all` (clean), `empty` (all candidates ablated; error floor remains), `keep_only` (only \(E\)), `remove` (\(E\) ablated). Then \(S = s_{\mathrm{keep}} - s_{\mathrm{empty}}\), \(N = s_{\mathrm{all}} - s_{\mathrm{remove}}\), \(F = \alpha S + (1-\alpha) N\).

### Surgical unlearning (Game 1)

**Claim.** Tiny feature set; ablate at runtime so a target concept collapses, without weight edits.

**Success after `remove(E)`:** target logit / gap drops (\(N\)); preferably argmax leaves the forgotten token; collateral (KL / retain-set CE) stays small.

- **N** is the product metric. **S** is completeness (high S / low N = reconstructs but does not unlearn). **F (\(\alpha=0.5\))** is misaligned and can rank a copy circuit above a knockout circuit.
- Select with \(\alpha=0\). Do **not** select on KL (target-free; equal-KL sets are huge).
- IOI caution: a high-S / low-N Game 1 set can look faithful and still be a weak unlearner.
- Dallas frozen PF20: \(\alpha=0\) \(E^*\) (4 nodes) has keep_only \(=-0.25\) and remove \(=-5.91\) (Texas wins); \(\alpha=1\) \(E^*\) is **disjoint** and does not knockout (remove \(=+1.88\)). See the case study \(\alpha\) table.

### Contrastive steering (Game 2)

**Claim.** Disjoint \(E_y\) and \(E_{\mathrm{foil}}\); clamp to flip the answer without changing the prompt.

**Success:** keep-only of \(E_{\mathrm{foil}}\) inverts the gap (foil wins argmax); clamp of \(E_y\) restores the target; overlap_rate near 0.

- **S** is primary (clamping is keep-only). **overlap_rate** is the contrastive claim. **F** can look good while argmax still fails.
- Select on logit_gap with \(\alpha=1\). Do **not** select on KL: a working foil circuit *moves away* from the clean distribution.
- Dallas KL Game 2 (ABR, unbudgeted): \(E_y\) and \(E_{\mathrm{foil}}\) shared 18 nodes (Jaccard 0.36), both starting at `0_25454_10`. That is the anti-alignment, not a steering success.

### Transcoder / model auditor

**Claim.** Does the behavior live in dictionary features or in attention routing? Distinct from MSE / \(L_0\) / CE recovered.

**Core verdict** is sign of \(R\) under frozen vs unfrozen attention on the **same** graph (all vs empty — no \(E^*\) required). Feature-mediated: \(R>0\) frozen. Attention-mediated: \(R\le 0\) frozen, typically \(R>0\) unfrozen. Wider CLT need not flip the sign.

Run Game 1 with `--freeze-mode both` and `stop_metric=raw_relative` (forced under `both`; `normalized` is degenerate when \(R\) collapses). Optional second Game 1 on KL = dictionary-capacity audit, not a surgical circuit.

### KL: when it helps and when it hurts

Use KL for collateral after unlearning, dictionary completeness, and cross-method comparison that does **not** reuse Game 1’s logit-gap objective.

Do not use KL as the Game 1/2 objective for these apps: it is target-free; equal-KL matching forces huge sets; foil steering *should* increase KL. Same ban as TMLR fairness contract item 5.

### Tuning (application claims)

| Knob | Unlearning | Steering | Auditor |
| --- | --- | --- | --- |
| \(\alpha\) | 0 | 1 (Game 2) | 0.5 is fine |
| Stop | knockout (`remove` / \(P(y)\)), not F sat. | grow until both argmax flips | all vs empty; optional \(\lvert E^*\rvert\) at \(\varepsilon\) |
| Budget | only for paper matched-k vs baselines | only vs ActAdd | \(k=8\) is least informative |
| Freeze | dual if frozen \(R\) weak/negative | dual on attention-mediated tasks | **must** run both |
| Prefilter | prefer off or a large pool | same | same |

`--freeze-mode both` forces `raw_relative`: stop when \(\Delta F_{\mathrm{next}} < \varepsilon \cdot \Delta F_{\mathrm{first}}\) (first feature, not previous step).

### Practical takeaway

Keep logit-gap F/S/N in the paper as the shared causal language. For applications, split them: unlearning is knockout (**N**); steering is reconstruction of a chosen pair (**S**) with disjointness; the auditor is the error-floor / freeze protocol (**recoverable range**). Select on logit gap (or `logit`/`prob` for foil-free unlearning). Always rescore with KL. Run **goal-oriented** for the product claim and **budgeted** only when you need matched-\(k\) baselines.
