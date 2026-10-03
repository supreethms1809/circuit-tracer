> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). Game 2 objective, ABR / fictitious play, foil resolution (former `macag.md` §5).

## 5. Game 2: Contrastive Evidence

### 5.1 Objective

Find two evidence sets — one for the target class, one for the foil — that are maximally faithful to their respective classes while being minimally overlapping:

$$E_y^* = \arg\max_{E_y} \left[\text{faithfulness}(E_y) - \lambda |E_y| - \beta |E_y \cap E_{\text{foil}}^*|\right]$$
$$E_{\text{foil}}^* = \arg\max_{E_{\text{foil}}} \left[\text{faithfulness}(E_{\text{foil}}) - \lambda |E_{\text{foil}}| - \beta |E_{\text{foil}} \cap E_y^*|\right]$$

where $\beta \geq 0$ is the **overlap penalty** (default $\beta = 0.1$), encouraging the two evidence sets to identify *different* features for target vs. foil.

### 5.2 Algorithm: Simultaneous Best Response (the "ABR" solver)

> *Naming note:* the solver is called `abr` in code and CLI for historical
> reasons, but the implemented update is **simultaneous (Jacobi)**, not
> alternating (Gauss-Seidel): in every round, *both* agents best-respond to the
> opponent's evidence set **from the previous round**. The two best-responses
> are independent given that frozen opponent, so the default implementation
> runs them **concurrently** (`--parallel-agents`, default on) and only
> communicates at the round **barrier**: join \(E_y', E_{\text{foil}}'\),
> score the joint allocation, update fictitious-play frequencies, test
> convergence, then persist. A 24h-kill resume with checkpoint `phase=foil`
> stays sequential so a completed y-side is not recomputed. On one GPU the
> shared `ReplacementModel` is serialized inside `ScoringOracle`; the barrier
> is still required for correctness (two independent Game 1 jobs with no
> exchange would be \(\beta=0\)).

**Pseudocode**:
```
Algorithm: MACAG Game 2 — Simultaneous Best Response ("abr")
─────────────────────────────────────────────────────────────
Input:  Graph G, Oracle O, targets (y, y_foil), candidates C,
        α, λ, β, K (max rounds), budget B (optional)
Output: Best joint allocation (E_y*, E_foil*), convergence flag, best_iteration

1.  E_y ← ∅,  E_foil ← ∅
2.  best ← evaluate(∅, ∅),  best_iteration ← 0       // combined hard-overlap utility
3.  for k = 1, ..., K:
4.      // BOTH players respond to the SAME frozen opponent from round k-1
5.      E_y'    ← GREEDY_BEST_RESPONSE(G, O, y,      E_foil, C, α, λ, β, B)
6.      E_foil' ← GREEDY_BEST_RESPONSE(G, O, y_foil, E_y,    C, α, λ, β, B)
7.      if evaluate(E_y', E_foil') > best:
8.          best ← evaluate(E_y', E_foil'),  best_iteration ← k
9.      if E_y' = E_y and E_foil' = E_foil:
10.         converged ← true; break
11.     E_y ← E_y',  E_foil ← E_foil'
12. return best allocation, converged, best_iteration   // best round, NOT last iterate

──────────────────────────────────────────────────────────────────────────────
Subroutine: GREEDY_BEST_RESPONSE(G, O, target, w_other, C, α, λ, β, B,
                                  connected, min_gain, prefilter_top_k)
──────────────────────────────────────────────────────────────────────────────
Input:  Graph G, Oracle O, target,
        w_other : NodeId → [0,1]   // opponent inclusion weight. ABR passes hard
                                    // 0/1 weights (1.0 for every node in the
                                    // opponent's LAST evidence set); FP passes the
                                    // opponent's empirical per-node inclusion
                                    // frequency p_t(n) (§5.2.1). Both are handled
                                    // by the same linear overlap term below.
        candidates C, α, λ, β, budget B (optional),
        connected, min_gain, prefilter_top_k (optional)
Output: Best-response evidence set E

// U₂(E) = faithfulness(E) − λ|E| − β·Σ_{n∈E} w_other(n)
// This is exact for both ABR (w_other ∈ {0,1}, so the sum is the hard overlap
// |E ∩ E_other|) and FP (w_other ∈ [0,1], so the sum is the EXPECTED overlap
// against the opponent's empirical mixture — no extra oracle calls either way).

1.  E ← ∅
2.  if prefilter_top_k:
3.      C ← rank C by U₂({n}) for each singleton n ∈ C, keep top-k    // PREFILTER_WITH_OVERLAP
4.  repeat
5.      if B ≠ nil and |E| ≥ B: break
6.      U_curr ← U₂(E)
7.      n* ← nil,  best_gain ← min_gain
8.      for each n ∈ C \ E:
9.          if connected and |E ∪ {n}| > 1 and ¬CONNECTED_THROUGH(E ∪ {n}): skip
10.         gain ← U₂(E ∪ {n}) − U_curr
11.         if gain > best_gain:
12.             best_gain ← gain,  n* ← n
13.         else if gain = best_gain and str(n) < str(n*):
14.             n* ← n
15.     if n* = nil: break                          // no improving response
16.     E ← E ∪ {n*}
17. return E
```

This is the single greedy routine both solvers call — it differs from the Game 1 greedy of [§4.2](macag_game1.md#42-algorithm-greedy-hill-climbing) only in the utility ($U_2$ instead of $U_1$) and in taking `w_other` as an extra input; the ABR/FP top-level loops differ only in what they pass as `w_other` (hard opponent membership vs. empirical frequency).

**Step-by-step**:

1. Initialize $E_y = \emptyset$, $E_{\text{foil}} = \emptyset$
2. Repeat for up to $K$ rounds (default $K = 10$):
   - **Player $y$**: Greedy hill-climb to optimize $U_2(E_y \mid E_{\text{foil}}^{(k-1)})$
   - **Player foil**: Greedy hill-climb to optimize $U_2(E_{\text{foil}} \mid E_y^{(k-1)})$
   - Both see the same round-$(k{-}1)$ opponent (Jacobi symmetry), so neither player has a first-mover information advantage and the players are exactly exchangeable — symmetric oracles provably yield symmetric per-agent utilities (pinned by a regression test).
3. Track the best joint allocation seen (by combined hard-overlap utility); stop early when both responses repeat, else stop at the round cap.
4. **Return the best round's allocation** and `best_iteration`; `converged` describes the dynamics, not the returned round ([§3.4](macag_foundations.md#34-equilibrium-analysis)).

Each player's greedy step uses a modified utility that includes the overlap penalty:

$$U_2(E, E_{\text{other}}) = \text{faithfulness}(E) - \lambda |E| - \beta \cdot \text{overlap}(E, E_{\text{other}})$$

**Candidate prefiltering** in Game 2 accounts for the existing evidence from the other player: `_prefilter_with_overlap_penalty()` ranks candidates using the full $U_2$ function, not just singleton gain.

**Convergence**: the solver reports `converged=true` when $(E_y^{(k)}, E_{\text{foil}}^{(k)}) = (E_y^{(k-1)}, E_{\text{foil}}^{(k-1)})$ — a fixed point at which neither agent's greedy response changes, i.e. a greedy approximation of a pure-strategy Nash equilibrium. A PSNE exists because Game 2 is an exact potential game, but the simultaneous update means even exact best responses can 2-cycle, and the greedy inner solver is approximate — see [§3.4](macag_foundations.md#34-equilibrium-analysis) for the precise statement. In practice the solver stabilizes in 2–4 rounds because:
- The greedy inner solver is deterministic given fixed opponents
- The overlap penalty (v3 / Dallas $\beta = 0.2$; CLI default $0.1$) is mild enough that agents settle quickly once they establish non-overlapping "cores"
- The finite candidate pool (v3 pruned graphs: mean $\sim 500$–$800$ features; Dallas unpruned: $2028$) limits the space of possible oscillations

When the dynamics do oscillate, best-iterate tracking returns the highest-combined-utility round, so a cycling pair $(\{A\},\{A\}) \leftrightarrow (\{B\},\{B\})$ does not degrade the reported result.

### 5.2.1 Fictitious-Play Solver (`solver="fp"`)

The second solver damps oscillation by replacing the opponent's last iterate with its **empirical history**: at round $t$, each agent best-responds to the mixture that includes node $n$ with probability $p_t(n)$ = the fraction of past rounds the opponent's evidence contained $n$. Because the opponent enters $U_2$ only through the overlap penalty, which is linear in node membership, the expected utility against this mixture is computed *exactly* as $\beta\sum_{n\in E} p_t(n)$ — no additional oracle calls over ABR.

**Pseudocode**:
```
Algorithm: MACAG Game 2 — Fictitious Play ("fp")
─────────────────────────────────────────────────────────────
Input:  Graph G, Oracle O, targets (y, y_foil), candidates C,
        α, λ, β, K (max rounds), budget B (optional), fp_tol
Output: Best joint allocation (E_y*, E_foil*), convergence flag, best_iteration,
        node_frequencies_y, node_frequencies_foil

1.  E_y ← ∅,  E_foil ← ∅
2.  counts_y ← {},  counts_foil ← {}         // per-node inclusion counts across rounds
3.  freq_y ← {},  freq_foil ← {}             // empirical inclusion frequencies p_t(n)
4.  best ← evaluate(∅, ∅),  best_iteration ← 0    // combined HARD-overlap utility, §3.4
5.  for k = 1, ..., K:
6.      // round 1: freq_y = freq_foil = {} ⇒ identical to ABR round 1
7.      E_y'    ← GREEDY_BEST_RESPONSE(G, O, y,      w_other=freq_foil, C, α, λ, β, B)
8.      E_foil' ← GREEDY_BEST_RESPONSE(G, O, y_foil, w_other=freq_y,    C, α, λ, β, B)
9.      if evaluate(E_y', E_foil') > best:              // evaluate() uses HARD overlap
10.         best ← evaluate(E_y', E_foil'),  best_iteration ← k
11.     for n ∈ E_y':    counts_y[n]    ← counts_y.get(n, 0) + 1
12.     for n ∈ E_foil': counts_foil[n] ← counts_foil.get(n, 0) + 1
13.     freq_y'    ← { n : counts_y[n] / k    for n in counts_y }
14.     freq_foil' ← { n : counts_foil[n] / k for n in counts_foil }
15.     Δfreq ← max( max_n |freq_y'(n) − freq_y(n)|, max_n |freq_foil'(n) − freq_foil(n)| )
16.     if E_y' = E_y and E_foil' = E_foil:
17.         converged ← true
18.     else if k > 1 and Δfreq < fp_tol:
19.         converged ← true
20.     freq_y ← freq_y',  freq_foil ← freq_foil'
21.     E_y ← E_y',  E_foil ← E_foil'
22.     if converged: break
23. return best allocation, converged, best_iteration, freq_y, freq_foil    // best round, NOT last iterate
```

Round 1 matches ABR round 1 exactly because `freq_y = freq_foil = {}` (an empty mapping means `w_other(n) = 0` for every candidate, the same as ABR's round-1 empty opponent set). From round 2 onward the two solvers diverge: ABR's `w_other` is the opponent's single last set (hard 0/1), FP's is the running empirical frequency (soft, in $[0,1]$) — both are consumed by the identical `GREEDY_BEST_RESPONSE` subroutine of [§5.2](#52-algorithm-simultaneous-best-response-the-abr-solver), so the two top-level loops share every line of the inner greedy and differ only in what they feed it as the opponent weights and in the extra frequency-convergence check (line 18).

Properties (all pinned by regression tests):
- **Round 1 is identical to ABR round 1** (both respond to an empty history).
- **Early stop** when the best responses repeat, or when the empirical frequencies of both agents change by less than `fp_tol` (default $10^{-3}$).
- **Reported metrics always use the hard overlap** of the returned joint allocation, so ABR and FP results are directly comparable.
- FP additionally returns `node_frequencies_y` / `node_frequencies_foil` — the per-node empirical inclusion frequencies, a *soft evidence membership* useful when near-interchangeable features make any single hard set arbitrary.

### 5.3 Evidence Decomposition

After convergence:

$$\text{shared} = E_y \cap E_{\text{foil}}, \quad \text{unique}_y = E_y \setminus E_{\text{foil}}, \quad \text{unique}_{\text{foil}} = E_{\text{foil}} \setminus E_y$$

**Overlap rate** (Jaccard similarity):

$$\text{overlap\_rate} = \frac{|E_y \cap E_{\text{foil}}|}{|E_y \cup E_{\text{foil}}|}$$

Lower overlap indicates that the model uses distinct features for target vs. foil predictions — a sign of well-separated, interpretable circuits.

### 5.4 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\beta$ | 0.1 | Overlap penalty between target and foil evidence. CLI default; pipeline / v3 / Dallas use **0.2** |
| $K$ (`abr_iters`) | 10 | Maximum solver rounds (ABR or FP). CLI default; pipeline / v3 use **4** |
| `solver` | `"abr"` | `"abr"` (respond to opponent's last set) or `"fp"` (fictitious play: respond to opponent's empirical history, [§5.2.1](#521-fictitious-play-solver-solverfp)) |
| `fp_tol` | $10^{-3}$ | FP only: stop when both agents' empirical node frequencies change by less than this |
| All Game 1 parameters | CLI-same | $\alpha$, $\lambda$, budget, prefilter, connected. Game 2 has no `faithfulness_eps`/`stop_metric`. **Connected:** CLI default on (`--connected`); campaigns pass `--no-connected`. v3 Game 2 is a **single** freeze (`oracle_kwargs.freeze_attention=true`), not `--freeze-mode both`. |

### 5.5 Output

`ContrastiveEvidenceResult` containing:
- **evidence_y**, **evidence_foil**: the returned joint allocation — the **best-combined-utility round**, not necessarily the final iterate ([§3.4](macag_foundations.md#34-equilibrium-analysis))
- **shared**, **unique_y**, **unique_foil**: decomposition
- **metrics_y**, **metrics_foil**: separate `FaithfulnessMetrics` (same raw + normalized fields as Game 1), evaluated on the returned allocation
- **utility_y**, **utility_foil**: each side's $U_2$ under the hard overlap of the returned allocation
- **overlap_rate**: Jaccard similarity
- **converged**: whether the *dynamics* stabilized (iterates repeated, or FP frequencies within `fp_tol`) — a property of the run, not of the returned round
- **iterations**: number of solver rounds taken
- **best_iteration**: the round whose allocation is returned (0 = the initial empty allocation beat every solver round; the solver logs a warning when this happens despite non-empty rounds, since it means $\lambda/\beta$ outweighed realized faithfulness)
- **node_frequencies_y**, **node_frequencies_foil**: FP only — empirical per-node inclusion frequencies across rounds (empty dicts under ABR)
- **total_candidates**, **sparsity_y**, **sparsity_foil**: candidate pool and per-side sparsity
- **oracle_calls**, **cache_hits**, **cache_size**: oracle/memoization profile (per-solve)
- **params**: resolved knobs (adds `beta`, `abr_iters`, `solver`, `fp_tol` to the Game 1 set)

**Note — Game 2 is unaffected by the `stop_metric` issue.** Game 2 selects on the raw $U_2$ utility via ABR and has no `faithfulness_eps` early-stop, and `overlap_rate` is denominator-free, so neither depends on `recoverable_range`. The frozen→unfrozen attention switch therefore changes only the underlying faithfulness scores, not how Game 2 selects.

### 5.6 Foil Resolution

Game 2 requires a target-foil pair. The foil mapping is configured as:
- If exactly 2 targets: auto-creates bidirectional map (e.g., " Paris" ↔ " Lyon")
- Otherwise: requires explicit `foil_by_target` mapping

Target tokens are resolved to vocabulary indices via the model's tokenizer, supporting:
- Direct integer indices
- String tokens (tokenized, enforcing single-token constraint)
- Explicit `"id:<int>"` format

---
