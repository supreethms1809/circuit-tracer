> **Part of the MACAG docs pack.** Hub: [`macag.md`](macag.md). Coalitional game, submodularity, Game 2 potential, Shapley/Banzhaf (former `macag.md` §3).

## 3. Game-Theoretic Foundations

### 3.0 The Underlying Coalitional Game

MACAG's two games are both built on a single **cooperative (coalitional) game** whose **players are the feature nodes** $N = C$ and whose **characteristic function is the faithfulness contribution** of a coalition $S \subseteq N$:

$$v(S) = \alpha\bigl(S_{\text{keep}}(S) - S_{\text{empty}}\bigr) + (1-\alpha)\bigl(S_{\text{all}} - S_{\text{remove}}(S)\bigr), \qquad v(\emptyset)=0.$$

This is the same $v$ throughout the document (the faithfulness function $f$). Once the game is posed this way, the standard solution concepts apply, and the different MACAG objects are different questions *about the same $v$*:

- **Per-feature credit** — how much is each player worth? The classical answers are the **Shapley value** and the **Banzhaf value** ([§3.6](#36-relation-to-shapley-and-banzhaf-credit)).
- **Best small coalition** — what is the most valuable cheap coalition? This is **Game 1** (a sparsity-penalized coalition-selection problem solved greedily).
- **Two-player extension** — split the players between a target and a foil agent with an overlap cost: **Game 2**.

The "single-agent optimizer" description of Game 1 below is the *solver's* view of this coalition-selection problem; the *game-theoretic* object is the coalitional game $(N, v)$.

### 3.1 Game Classification

As optimization problems, both MACAG games belong to the class of **cooperative combinatorial optimization games** — specifically, they are instances of **weighted maximum coverage / set function optimization** over the ground set of feature nodes. The key properties:

| Property | Game 1 | Game 2 |
|----------|--------|--------|
| Players | 1 (evidence selector) | 2 (target agent, foil agent) |
| Strategy space | $2^C$ (subsets of candidates) | $2^C \times 2^C$ (one subset per agent) |
| Payoff | $U_1(E) = f(E) - \lambda \| E \| $ | $U_2(E_i, E_j) = f(E_i) - \lambda\|E_i\| - \beta \|E_i \cap E_j \|$ |
| Game type | Single-agent optimization | Two-player symmetric game with externalities |
| Solution concept | Greedy maximum ($(1{-}1/e)$ only if submodular — [§3.2](#32-the-value-function-and-submodularity)) | Pure-strategy Nash equilibrium (exact potential game — [§3.4](#34-equilibrium-analysis)) |
| Interaction | None | Negative externality via overlap penalty |

### 3.2 The Value Function and Submodularity

The core value function underlying both games is the **faithfulness function**:

$$f(E) = \alpha \cdot \underbrace{\left(S_{\text{keep}}(E) - S_{\text{empty}}\right)}_{\text{sufficiency}} + (1-\alpha) \cdot \underbrace{\left(S_{\text{all}} - S_{\text{remove}}(E)\right)}_{\text{necessity}}$$

**Properties of the characteristic function $v \equiv f$.** Useful facts a paper can
state up front:
- *Grounded:* $v(\emptyset) = \alpha(S_{\text{empty}} - S_{\text{empty}}) + (1-\alpha)(S_{\text{all}} - S_{\text{all}}) = 0$. The error floor is subtracted on the sufficiency side and the full circuit on the necessity side, so the empty coalition is worth zero by construction.
- *Bounded above by the recoverable range:* the grand coalition gives $v(C) = \alpha(S_{\text{all}} - S_{\text{empty}}) + (1-\alpha)(S_{\text{all}} - S_{\text{empty}}) = S_{\text{all}} - S_{\text{empty}} = \text{recoverable\_range}$. Hence the normalized objective $v(E)/v(C)$ targets "fraction of the recoverable range explained," and is exactly why a non-positive range makes the normalized view degenerate ([§2.3](macag_framework.md#23-attention-freezing-and-the-error-floor)).
- *Not guaranteed monotone:* adding a feature can lower $v$ (a feature whose ablation *helps* the target, i.e. a suppressor/foil-aligned feature, has negative marginal value). The $-\lambda|E|$ term further makes the *utility* non-monotone by design. Greedy therefore needs the `min_gain`/stop machinery rather than running to $E = C$.
- *Not guaranteed superadditive:* redundancy ($v(A\cup B) < v(A)+v(B)$) and synergy ($>$) both occur — see the violation example below.

**Submodularity analysis.** $v$ is *submodular* iff marginal gains diminish: $\Delta_i(S) \ge \Delta_i(T)$ for all $S \subseteq T$ and $i \notin T$. When $v$ is additionally monotone, the greedy algorithm achieves the classic $(1-1/e)\approx 0.632$ guarantee under a cardinality constraint (Nemhauser, Wolsey & Fisher, 1978); the modular $-\lambda|E|$ penalty shifts every value by a constant per element and so preserves the relative ordering of marginal gains (greedy with a modular penalty is equivalent to greedy on $v$ with an adjusted `min_gain`).

**Where it breaks — a concrete construction.** Submodularity fails exactly when features interact. Two canonical cases over the logit-gap oracle:
- *Synergy / joint necessity (super-modular spike).* Suppose a behavior needs both a "subject" feature $a$ and a "relation" feature $b$ (e.g. the two hops of city→state and state→capital). Individually each is nearly useless: $v(\{a\}) \approx v(\{b\}) \approx 0$, but together $v(\{a,b\}) \gg 0$. Then $\Delta_b(\emptyset) \approx 0 < \Delta_b(\{a\}) \gg 0$ — an *increasing* marginal return, the opposite of submodular. A pure greedy that adds the single best feature first can stall (no feature has positive singleton gain) and miss the pair; this is the failure the `prefilter`+budget and the two-hop prompts are most likely to expose.
- *Redundancy (the benign direction).* Two interchangeable copies $a,a'$ of the same computation: $v(\{a\}) = v(\{a'\}) = v(\{a,a'\})$. Here $\Delta_{a'}(\{a\}) = 0 < \Delta_{a'}(\emptyset)$ — submodular, and harmless: greedy takes one and the sparsity penalty rejects the other. Large/overcomplete dictionaries produce many such pairs (this is the mechanism behind the near-zero cross-seed Jaccard observed for wide CLTs in older notes).

**Consequences for MACAG (honest characterization).** The logit gap under feature ablation is **not** guaranteed submodular, so the $(1-1/e)$ bound is a *best case*, not a theorem about MACAG. Greedy is nonetheless a reasonable solver because:
1. empirically most features contribute near-independently to the gap (the synergy spikes above are the exception, concentrated on genuinely multi-hop prompts);
2. the sparsity penalty $\lambda$ keeps the search out of the deep diminishing-returns tail where violations accumulate;
3. the singleton prefilter biases toward high-marginal candidates, where diminishing-returns violations are rarer.
The right way to *report* this (not just argue it) is the empirical optimality gap in roadmap **B3.2**: brute-force the best size-$k$ subset on small pools and compare to greedy, and separately compare greedy's selection to the Shapley ranking ([§3.6](#36-relation-to-shapley-and-banzhaf-credit)) — large greedy↔Shapley disagreement localizes the non-submodular prompts.

### 3.3 Game 2 as a Congestion Game

Game 2 can be viewed as a **two-player congestion game** where the shared resource is the set of feature nodes. Each agent (target and foil) wants to claim features for its own evidence set, but overlapping features incur a cost $\beta$ per shared node:

$$U_2(E_i \mid E_j) = f(E_i) - \lambda|E_i| - \beta|E_i \cap E_j|$$

This creates a **negative externality**: if Player $y$ includes a node already in $E_{\text{foil}}$, it pays the overlap penalty $\beta$. The penalty pushes the agents toward **complementary** evidence sets — different features for the target vs. foil prediction.

**Key insight**: The overlap penalty $\beta$ controls the degree of separation:
- $\beta = 0$: Both agents optimize independently (may converge to identical sets)
- $\beta \to \infty$: Agents are forced to select disjoint evidence sets
- $\beta = 0.1$ (default): Mild pressure toward separation, allowing shared features when they are highly faithful

### 3.4 Equilibrium Analysis

**Game 2 equilibrium concept**: MACAG seeks a **pure-strategy Nash equilibrium (PSNE)** — a pair $(E_y^*, E_{\text{foil}}^*)$ where neither agent can improve its utility by unilaterally changing its evidence set:

$$U_2(E_y^* \mid E_{\text{foil}}^*) \geq U_2(E' \mid E_{\text{foil}}^*) \quad \forall E' \subseteq C$$
$$U_2(E_{\text{foil}}^* \mid E_y^*) \geq U_2(E' \mid E_y^*) \quad \forall E' \subseteq C$$

**Existence via an exact potential function.** Game 2 is an *exact potential game*, which is the clean way to argue both existence and convergence. Define

$$\Phi(E_y, E_{\text{foil}}) = \big[f(E_y) - \lambda|E_y|\big] + \big[f(E_{\text{foil}}) - \lambda|E_{\text{foil}}|\big] - \beta\,|E_y \cap E_{\text{foil}}|.$$

The overlap term is the *only* coupling between the two players and it enters each player's utility identically: when player $y$ changes $E_y$ with $E_{\text{foil}}$ fixed,

$$U_2(E_y' \mid E_{\text{foil}}) - U_2(E_y \mid E_{\text{foil}}) = \Phi(E_y', E_{\text{foil}}) - \Phi(E_y, E_{\text{foil}}),$$

because the $f(E_{\text{foil}})-\lambda|E_{\text{foil}}|$ term is constant under $y$'s move and the shared $-\beta|E_y\cap E_{\text{foil}}|$ term is common to both $U_2$ and $\Phi$ (symmetrically for the foil player). So every unilateral *improving* move strictly increases the single scalar $\Phi$. Two consequences:
- **A pure-strategy Nash equilibrium exists.** The joint strategy space $2^C \times 2^C$ is finite, so $\Phi$ attains a maximum; any maximizer is a PSNE (no player can improve, since improving would raise $\Phi$ past its max). This is the finite-improvement property (FIP) of finite potential games — it does not need $C$ to be small, only finite.
- **Exact *sequential* best response cannot cycle.** If players move one at a time, each move is a unilateral improvement, hence (weakly) increases $\Phi$; a strict increase cannot repeat a state, so exact sequential best-response dynamics reach a fixed point in finitely many steps.

**Two caveats that matter in practice.**
1. *The implemented update is simultaneous (Jacobi), not sequential.* Both agents best-respond to the **same frozen opponent from the previous round** (this keeps the two players symmetric — neither sees a fresher opponent than the other). The FIP argument above is for unilateral moves; under simultaneous updates even *exact* best responses can 2-cycle (the regression tests construct exactly such an oscillation: $(\{A\},\{A\}) \leftrightarrow (\{B\},\{B\})$ under symmetric weights and high $\beta$). Existence of a PSNE is unaffected — only the convergence-of-dynamics argument changes.
2. *The inner solver is greedy*, an approximate best response, so even sequential dynamics would inherit the guarantee only up to greedy's sub-optimality in the non-submodular regime ([§3.2](#32-the-value-function-and-submodularity)).

The honest version of the convergence claim is therefore: **a PSNE exists (exact potential game); the implemented greedy-Jacobi dynamics are not guaranteed to reach it, are capped at $K$ rounds, and are protected by best-iterate tracking and an optional fictitious-play solver** (below).

**Best-iterate tracking (always on).** The solver evaluates every round's joint allocation $(E_y, E_{\text{foil}})$ by its **combined hard-overlap utility** $U_2^y + U_2^{\text{foil}} = \Phi - \beta\,|E_y\cap E_{\text{foil}}|$ (the overlap penalty is paid by *both* players, so the sum counts it twice where $\Phi$ counts it once; the two objectives coincide exactly on disjoint allocations, which is the empirically common case — overlap 0.0 in all 48 case-study runs). It **returns the best round seen, not the final iterate**, and reports `best_iteration` (0 means the initial empty allocation beat every round — a sign the $\lambda/\beta$ penalties outweigh realized faithfulness). `converged` refers to the *dynamics* (iterates repeated / frequencies stabilized), independently of which round is returned.

**Fictitious play (`solver="fp"`).** As a damping alternative to ABR, each agent can best-respond to the opponent's **empirical mixture** of past evidence sets instead of its last iterate. Because the opponent enters $U_2$ only through the overlap penalty, which is linear in membership, the expected utility against the mixture is exact: the penalty term becomes $\beta\sum_{n\in E} p_t(n)$, where $p_t(n)$ is the fraction of past rounds the opponent included $n$ — no extra oracle calls. FP stops early when best responses repeat or when both agents' empirical frequencies change by less than `fp_tol`; round 1 is identical to ABR round 1. Reported metrics always use the hard overlap of the returned joint allocation, so ABR and FP results are directly comparable; FP additionally reports the per-node inclusion frequencies (soft evidence membership).

**Empirics.** Convergence is observed within 2–4 rounds (the case-study runs used `abr_iters=4` and report `converged=True`); the first round fixes each player's high-value core and later rounds only adjust the few features near the $\beta$ margin. Near-interchangeable features for both players are the usual cause of non-convergence.

### 3.5 Complexity

| Component | Per-prompt complexity |
|-----------|----------------------|
| Game 1 greedy sweep | $O(\|E^*\| \cdot \|C\|)$ oracle calls |
| Game 1 with prefilter | $O(k)$ prefilter + $O(\|E^*\| \cdot k)$ greedy |
| Game 2 single ABR iteration | $O(\|E_y\| \cdot \|C\| + \|E_{\text{foil}}\| \cdot \|C\|)$ oracle calls |
| Game 2 full ABR | $O(K \cdot (\|E_y\| + \|E_{\text{foil}}\|) \cdot \|C\|)$ oracle calls |
| Oracle call | 1 forward pass through transformer with modified activations |

where $|E^*|$ is the final evidence set size, $|C|$ is the candidate pool size, $k$ is the prefilter budget, and $K$ is the ABR iteration count.

The **oracle memoization cache** reduces actual forward passes significantly. The cache key is `(mode, type(target), str(target), frozenset(nodes), universe_fingerprint)` ([§2.1](macag_framework.md#21-oracle-scoring)), so identical intervention sets across different game iterations or agents are computed only once. Empirically, cache hit rates range from 30–60% for Game 1 and 50–80% for Game 2 (where the two agents probe overlapping subsets). Oracle-call/cache-hit counters are reset at solver entry, so reported stats are per-solve even when one oracle is reused across games.

### 3.6 Relation to Shapley and Banzhaf Credit

> **Non-claim.** MACAG does not claim to approximate the Shapley value, and Game 1's
> greedy is not an estimator of it. Shapley and Banzhaf are used here as *reference
> credit measures over the same $v$* — a yardstick for how much a cheap greedy
> selection gives up against gold-standard credit, and at what oracle cost. Nothing
> in the framework requires the greedy solution to converge to either.
>
> **Shared-$v$ scope (measured — [Appendix J.A1](macag_appendix_legacy.md#ja-baselines-and-the-shared-characteristic-function)).**
> Every selected set is *evaluated* under one characteristic function $v$ at one
> $\alpha$. Methods are **not** all *ranked* under $v$: top-k influence and EAP are
> read off the graph and issue zero oracle calls, which is exactly the contrast the
> comparison is designed to expose. Estimator details and the top-$k$-prefix gold
> selection rule live in the Experimental Setup baselines section, not here.

Because the games are built on the coalitional game $(N, v)$ ([§3.0](#30-the-underlying-coalitional-game)), the classical per-player credit measures are directly available, and they clarify exactly what Game 1's greedy *is* and *is not*.

For a player (feature) $i$, the marginal contribution to a coalition $S$ is $\Delta_i(S) = v(S \cup \{i\}) - v(S)$. The two gold-standard attributions average this differently:

$$\phi_i^{\text{Shapley}} = \frac{1}{|N|!}\sum_{\pi}\Delta_i(S_\pi^{<i}) \quad\text{(average over all orderings)}, \qquad \phi_i^{\text{Banzhaf}} = \frac{1}{2^{|N|-1}}\sum_{S \subseteq N\setminus i}\Delta_i(S) \quad\text{(average over all coalitions)},$$

where $S_\pi^{<i}$ is the set of players preceding $i$ in permutation $\pi$. Equivalently, Shapley weights coalitions by $\tfrac{|S|!\,(|N|-|S|-1)!}{|N|!}$ (uniform over *ranks*), while Banzhaf weights all coalitions equally (uniform over *subsets*) — the two differ only in that weighting, which is why Banzhaf is less sensitive to where in an ordering a strongly-interacting feature happens to fall.

**Why Shapley is the principled per-feature credit (axioms).** $\phi^{\text{Shapley}}$ is the unique attribution satisfying: **efficiency** ($\sum_i \phi_i = v(N) - v(\emptyset) = \text{recoverable\_range}$ — the credits exactly partition the recoverable range); **symmetry** (features with identical marginals everywhere get equal credit — the two redundant copies $a,a'$ of [§3.2](#32-the-value-function-and-submodularity) split their shared value); **null player** (a feature with $\Delta_i(S)=0$ for all $S$ gets zero — a feature that never moves the gap; note a *suppressor* whose ablation helps the target has negative, not zero, marginal value and so receives negative credit); and **linearity** (credit is additive across score functions, e.g. target-logit and foil-logit pieces of the logit gap). These axioms are exactly the properties one wants from a circuit attribution, which is what makes Shapley the right *gold* reference even though it is not a *selector*.

**What Game 1's greedy computes instead.** At each step the greedy adds $\arg\max_i \Delta_i(S)$ for the *one* coalition $S$ it has built so far — a single marginal contribution along a single, greedily chosen permutation, not an average over all of them. So:

- Game 1 answers **"which small coalition is most valuable?"** (set selection), while Shapley/Banzhaf answer **"how much is each player worth?"** (credit assignment). They are different questions over the same $v$.
- The greedy is exponentially cheaper — $O(|E^*|\cdot|C|)$ oracle calls vs. the $2^{|C|}$ coalitions Shapley/Banzhaf average over (Monte-Carlo–estimated in practice, but still far more samples than the greedy).
- When $v$ is submodular, the greedy coalition carries the $(1-1/e)$ guarantee ([§3.2](#32-the-value-function-and-submodularity)); the Shapley/Banzhaf values carry no such selection guarantee because they are not a selection rule.

**How they are used here.** The Shapley value of the coalitional game $(N, v)$ plays two roles for MACAG: (1) as the **gold-standard baseline** (§9.3) — does Game 1's greedy evidence set coincide with the top-Shapley features, and at what fraction of the oracle cost?; and (2) as a **diagnostic** — large disagreement between a feature's Shapley value and its greedy marginal flags strong feature interactions (non-submodularity), i.e. the regime where the greedy guarantee is only heuristic.

**The implementation (`macag/baselines/shapley_select.py`).** `estimate_shapley` is a Monte-Carlo *permutation-sampling* estimator over the MACAG intervention oracle. Each sampled permutation walks the candidate pool front-to-back and charges every feature its marginal $\Delta_i(S)$ at the coalition built so far. Every coalition is priced by the shared `coalition_value` helper (`macag/baselines/common.py`), which returns `compute_faithfulness_metrics(...).faithfulness_delta` — *exactly* the $v(S)$ of [§3.0](#30-the-underlying-coalitional-game), evaluated through the oracle's `keep_only`/`remove` ablations (`ReplacementModel` forward passes in the real backend) at the same $\alpha$ Game 1 optimizes; every selected set in the harness is *evaluated* under this one $v$ at the same $\alpha$ ([J.A1](macag_appendix_legacy.md#ja-baselines-and-the-shared-characteristic-function)); influence and EAP *rank* from graph scores alone (zero oracle calls), so the comparison does **not** isolate the selection rule under a shared ranking $v$. With `antithetic=True` (the default) each sampled permutation is paired with its exact reversal, which cancels order noise for near-additive $v$. Because each permutation's marginals telescope to $v(N)-v(\emptyset)$, the estimate is *exactly* efficient for any number of permutations (and since both CLI drivers `restrict_universe` the oracle to the candidate pool, $v(N)$ *is* the recoverable range, so the credits partition exactly the quantity the tables report); the returned `ShapleyEstimate` carries per-node standard errors plus an `efficiency_gap` field, which for the permutation estimator should be zero up to float error (a wiring diagnostic, not a statistical one). Cost: $|C|$ coalition evaluations per permutation, each at most two fresh oracle calls (`keep_only` + `remove`; the empty/full coalitions are memoized after the first hit) — with the default 64 permutations this is what dominates the gold baseline's cost (the ~32.5k mean oracle calls of §10.7). `estimate_banzhaf` is implemented alongside it: per sample it draws a coalition $S$ by independent fair coin flips and charges every node $v(S\cup\{i\}) - v(S\setminus\{i\})$; its all-coalitions average is less order-sensitive than Shapley for highly interacting features, making it the natural second gold reference (for Banzhaf, which does not satisfy efficiency even exactly, `efficiency_gap` is a genuinely informative quantity rather than a float-error check). Both estimators are exposed through `select_top_shapley(estimator="shapley"|"banzhaf")`, which returns the same best-first `SelectionResult` as every other baseline — so "Shapley evidence at budget $k$" means the top-$k$ prefix of the estimated-credit ranking.

**Harness wiring (`macag/cli/run_baselines.py`).** `shapley` is in the default method list (`influence,eap,shapley,game1,acdc`); `banzhaf` is opt-in via `--methods`. Knobs: `--shapley-permutations` (default 64), `--banzhaf-samples` (default 64), `--shapley-seed` (default 0), `--no-antithetic`; $\alpha$ comes from the same `--alpha` all methods share. The harness's per-$k$ agreement block (precision@$k$ / Jaccard, emitted as `agreement_vs_shapley`) uses the Shapley ranking as gold when it ran, falling back to Banzhaf otherwise, and the §A.5 Spearman linearity diagnostic correlates the Shapley/EAP/influence scores against Game 1's per-step marginal gains.

**Not the spline-CLT Shapley (`attribution/shapley.py`).** The repository contains a second, unrelated Shapley module, and the two must not be conflated (nor one wrapped around the other — `shapley_select.py` deliberately does not reuse it). They share only the generic estimator technique — Monte-Carlo permutation sampling with antithetic pairing — which is precisely what invites the confusion; everything that matters is different:

| | `macag/baselines/shapley_select.py` (MACAG gold) | `attribution/shapley.py` (spline-CLT tool) |
|---|---|---|
| Players | feature *nodes of a traced circuit graph* | active features of a `KANCrossLayerTranscoder` |
| Value function $v$ | §3.0 target–foil faithfulness ($\alpha$-mixed `keep_only`/`remove`) | reconstruction-MSE reduction, or logit-direction projection |
| Evaluation backend | MACAG `ScoringOracle` → full-model interventions (`ReplacementModel` forward passes) | the transcoder alone (`encode`/`decode_dense`); no transformer in the loop |
| Purpose | gold per-feature credit to benchmark Game 1's selection | edge weights for the spline-CLT attribution graph |
| Entry points | `estimate_shapley`, `estimate_banzhaf`, `select_top_shapley` | `shapley_attribution`, `shapley_logit_attribution` |

`attribution/shapley.py` must never be cited as MACAG's Shapley-gold: it is a different $v$ on a different object and is not wired to the MACAG oracle.

**Status.** The estimator is implemented and unit-tested (`tests/test_macag_baselines.py`). Nonlinear-benchmark gold numbers in §10.7/C.7 (including the 44.7× cost gap) are **provisional** — `results/macag_nonlinear_connected/` is missing ([J.C14](macag_appendix_legacy.md#jc-campaign-status-and-coverage)); do not cite that cost ratio. **v3 MIB Shapley is partial, not zero:** Llama `results@8` on **196/200** prompts; Gemma-426k **7/200** (ARC-Easy); Gemma-2.5M **0/200**. On Llama, top-8 Shapley F is $11.12\pm 0.53$ vs Game 1 $13.03$ — expected when $v$ has synergies (Game 1 optimizes the set; Shapley ranks average marginals). 64 antithetic permutations, not exact. See [`macag_experiments_v3.md`](macag_experiments_v3.md).

---
