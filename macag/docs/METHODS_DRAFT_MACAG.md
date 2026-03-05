# Methods Draft (Copy-Ready): MACAG

You can adapt this text directly into your thesis chapter.

## 1. Setup and Notation

We represent a mechanistic circuit as a directed graph \(G=(V,E)\), where each node corresponds to an interpretable unit (for example, transcoder feature nodes). For a fixed input \(x\), we evaluate a scalar target behavior score under interventions.

The scoring interface exposes four core quantities for each target label \(c\):

- \(S_{\text{all}}(c)\): score with the full circuit active.
- \(S_{\text{empty}}(c)\): score with full ablation over the intervention universe.
- \(S_{\text{keep}}(A,c)\): score when only node subset \(A\subseteq V\) is active.
- \(S_{\text{remove}}(A,c)\): score when \(A\) is ablated from the full circuit.

In implementation, these are provided by a memoized `ScoringOracle` over an intervention backend (`macag/scoring.py`).

## 2. Faithfulness Objective

For a subset \(A\), we define:

\[
\text{Sufficiency}(A,c)=S_{\text{keep}}(A,c)-S_{\text{empty}}(c)
\]

\[
\text{Necessity}(A,c)=S_{\text{all}}(c)-S_{\text{remove}}(A,c)
\]

\[
\Delta_c(A)=\alpha\,\text{Sufficiency}(A,c)+(1-\alpha)\,\text{Necessity}(A,c)
\]

where \(\alpha\in[0,1]\) controls the sufficiency/necessity tradeoff.

## 3. Game 1 (Single-Agent Sparse Faithful Evidence)

Game 1 solves:

\[
\max_{A\subseteq V}\;U(A)=\Delta_y(A)-\lambda|A|
\]

with sparsity coefficient \(\lambda\ge 0\).

Solver:

- greedy add-only hill climb from \(A=\emptyset\),
- add node with largest positive marginal gain,
- stop on no positive gain, budget reached, or optional faithfulness threshold.

Implementation:

- `solve_game1` in `macag/games/game1_min_faithful.py`.

## 4. Game 2 (Two-Agent Contrastive Evidence Allocation)

For target \(y\) and foil \(y'\), two agents choose \(A_y\) and \(A_{y'}\):

\[
U_y(A_y,A_{y'})=\Delta_y(A_y)-\lambda|A_y|-\beta|A_y\cap A_{y'}|
\]
\[
U_{y'}(A_{y'},A_y)=\Delta_{y'}(A_{y'})-\lambda|A_{y'}|-\beta|A_y\cap A_{y'}|
\]

where \(\beta\ge0\) penalizes overlap.

Outputs:

- shared evidence: \(A_y\cap A_{y'}\),
- unique target evidence: \(A_y\setminus A_{y'}\),
- unique foil evidence: \(A_{y'}\setminus A_y\).

Solver:

- alternating best response (ABR),
- each best response is solved by greedy add-only search,
- terminate on stability or max ABR iterations.

Implementation:

- `solve_game2` in `macag/games/game2_contrastive.py`.

## 5. Scoring Backends and Caching

The architecture decouples solver logic from model-specific intervention semantics:

- protocol: `InterventionScorer`,
- memoized wrapper: `ScoringOracle`,
- concrete backends:
  - `ToyAdditiveInterventionScorer` for fast tests,
  - `ReplacementModelInterventionScorer` for real circuit interventions.

Memoization key:

\[
(\text{mode},\text{target},\text{frozenset(nodes)})
\]

This reduces repeated expensive intervention calls across greedy and ABR loops.

## 6. Practical Search Controls

To manage runtime:

- optional candidate prefilter (`--prefilter-top-k`),
- optional size budget (`--budget`),
- optional connectedness constraint (`--connected`),
- minimum gain threshold (`--min-gain`),
- default progress logging for long runs.

## 7. Scalar Score Choice

The implementation supports:

- `logit`,
- `prob`,
- `negative_loss`,
- `logit_gap` (default for contrastive setups).

For `logit_gap`, foil mapping is required and configured in oracle kwargs.

## 8. Complexity Discussion (Proposal-Level)

Let \(n=|V|\), \(b\)=budget, \(T\)=ABR iterations.

- Game 1 worst-case candidate evaluations are approximately \(O(nb)\), each requiring multiple score calls.
- Game 2 adds ABR alternation: approximately \(O(Tnb)\) per agent.
- Effective runtime is dominated by model interventions, so memoization and candidate filtering are essential.

## 9. Implementation-to-Theory Traceability

- Formal metrics: `macag/utils/metrics.py`
- Oracle + intervention semantics: `macag/scoring.py`
- Game 1 optimization: `macag/games/game1_min_faithful.py`
- Game 2 optimization: `macag/games/game2_contrastive.py`
- CLI orchestration and JSON schema: `macag/cli/run_macag.py`

This one-to-one mapping supports a clear "theory to implementation" argument in the thesis.

## 10. Planned Framework Evolution (Not Yet Implemented)

For proposal clarity, include this implementation status statement:

"At this stage, MACAG is implemented as a standalone proof-of-concept codebase. A post-prelims engineering phase will align MACAG's internal modular structure with CDEA-style abstractions while keeping the two frameworks in separate repositories."

Planned changes (future work):

- separate objective definitions from solver mechanics in dedicated core modules,
- define explicit game registry and configuration objects for extensibility,
- standardize reporting contracts across MACAG and CDEA outputs,
- maintain backend adapters as thin, domain-specific layers.

Important claim discipline:

- implemented now:
  - intervention-based MACAG games with greedy and ABR solvers,
  - oracle caching, candidate filtering controls, CLI reproducibility.
- planned later:
  - deeper architectural alignment with CDEA patterns,
  - cross-repository schema harmonization and comparative harnesses.
