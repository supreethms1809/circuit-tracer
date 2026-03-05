# Chapter Blueprint: MACAG Proof-of-Concept

Use this as the structure for your prelims proposal chapter.

## Chapter Goal

State that this chapter introduces a game-theoretic framework for mechanistic circuit evidence allocation, then validates feasibility with a working implementation and preliminary experiments.

## Architecture Positioning (Add Early in Chapter)

State this explicitly in 1-2 paragraphs:

- MACAG and CDEA remain separate repositories in the current stage.
- the design direction is aligned: protocol-first modules, explicit objectives, solver modularity, and stable reporting contracts.
- repository separation is intentional risk control, not conceptual divergence.

Suggested sentence:
"The current implementation stage prioritizes methodological clarity and reproducibility by keeping MACAG and CDEA in separate repositories while aligning their internal architecture patterns for future interoperability."

## Recommended Chapter Structure

## 1. Problem Motivation

- Why current mechanistic explanations are hard to compare across targets/foils.
- Why "small but faithful" evidence subsets matter.
- Why contrastive decomposition (unique vs shared evidence) is useful.

Suggested claim:
"We need explanations that are intervention-faithful, sparse, and contrastive."

## 2. Formal Problem Setup

- Define graph \(G=(V,E)\), input \(x\), target \(y\), foil \(y'\).
- Define intervention-based scores:
  - \(S_{\text{all}}\), \(S_{\text{empty}}\), \(S_{\text{keep}}(E)\), \(S_{\text{remove}}(E)\).
- Define faithfulness delta:
  - \(\Delta(E)=\alpha(S_{\text{keep}}(E)-S_{\text{empty}})+(1-\alpha)(S_{\text{all}}-S_{\text{remove}}(E))\).

## 3. Game Formulations

- Game 1 objective:
  - \(U(E)=\Delta(E)-\lambda|E|\).
- Game 2 objectives:
  - \(U_y(E_y,E_{y'})=\Delta_y(E_y)-\lambda|E_y|-\beta|E_y\cap E_{y'}|\).
  - \(U_{y'}(E_{y'},E_y)=\Delta_{y'}(E_{y'})-\lambda|E_{y'}|-\beta|E_y\cap E_{y'}|\).

## 4. Solvers and Computational Strategy

- Game 1: greedy add-only hill climb.
- Game 2: alternating best response (ABR), greedy inner loops.
- Efficiency controls:
  - oracle memoization,
  - candidate prefilter,
  - budget constraints.

## 5. Implementation

Map each concept to code modules:

- Graph wrapper: `macag/graph.py`
- Scoring interface + caching: `macag/scoring.py`
- Faithfulness metrics: `macag/utils/metrics.py`
- Game 1 solver: `macag/games/game1_min_faithful.py`
- Game 2 solver: `macag/games/game2_contrastive.py`
- CLI execution + JSON outputs: `macag/cli/run_macag.py`
- Visualization annotation bridge: `macag/cli/annotate_graph.py`

## 6. Preliminary Results

Report:

- number of selected nodes,
- faithfulness metrics,
- overlap/shared/unique decomposition,
- oracle calls and cache hit statistics.

State clearly that these are proof-of-concept results and not final scientific claims.

## 7. Limitations and Threats to Validity

- quality depends on foil choice,
- quality depends on candidate set,
- greedy optimization may miss better global subsets,
- intervention semantics depend on backend/model tooling.

## 8. Proposed Next Steps

- stronger baselines and larger evaluation set,
- sensitivity analysis over \(\alpha,\lambda,\beta\),
- scalability improvements and richer game variants,
- better feature labeling and interpretation support.

## 9. Framework Evolution Plan (Proposal Section)

Add a short section that distinguishes:

- current state: implemented MACAG proof-of-concept in its own codebase,
- planned state: CDEA-aligned modular refactor in MACAG plus shared output contract across repos,
- deferred decision: deeper codebase unification after interface stabilization.

Use this section to show engineering maturity and realistic staging.

## Claim Discipline (Important for Prelims)

Make only claims supported by the current implementation:

- "Implemented and runnable"
- "Supports intervention-faithfulness scoring"
- "Supports sparse and contrastive evidence extraction"
- "Demonstrates feasibility on concrete circuits"

Avoid overclaiming:

- "discovers ground-truth mechanisms"
- "proves causal sufficiency in general"
