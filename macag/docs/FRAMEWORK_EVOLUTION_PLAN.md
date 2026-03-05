# Framework Evolution Plan (Proposal-Facing)

This document captures the architecture decision for the next phase:

- keep MACAG and CDEA in separate repositories for now,
- align MACAG's internal design patterns to CDEA-style modularity,
- define explicit interoperability contracts so future convergence remains possible.

## 1. Decision Summary

Current decision:

- repository strategy: separate repositories,
- design strategy: aligned architecture patterns.

Rationale:

- MACAG and CDEA currently optimize different mathematical objects:
  - MACAG: discrete node-set search with intervention oracles,
  - CDEA: continuous mask optimization with gradient-based allocators.
- forcing early code unification would introduce brittle abstractions before requirements stabilize.
- separate repos preserve momentum while still enabling shared concepts at the interface level.

## 2. What "CDEA-Aligned" Means for MACAG

The alignment target is structural, not code-copying.

Planned principles:

- protocol-first boundaries (oracle, objective, solver, reporting),
- explicit game-mode configuration objects,
- clear separation between domain adapters and optimization logic,
- reusable reporting schema and experiment metadata.

## 3. Planned MACAG Architecture Changes

Target module layering (in future MACAG work):

- `core/types.py`
  - typed result objects, config dataclasses, run metadata.
- `core/objectives.py`
  - faithfulness and utility terms separated from search mechanics.
- `core/solvers/`
  - greedy, ABR, and future local-search variants.
- `core/oracle.py`
  - intervention scoring protocol and caching wrapper.
- `adapters/`
  - backend-specific intervention semantics (ReplacementModel, future backends).
- `games/`
  - game definitions that compose objective + solver + constraints.
- `reporting/`
  - JSON/CSV writers with stable schema.

This mirrors CDEA's separation of `objective`, `allocator`, `runner`, and modality adapters, while preserving MACAG's set-based algorithmic needs.

## 4. Interoperability Contract Across Repositories

Even with separate repos, keep these shared contracts:

- common terminology:
  - `target`, `foil`, `shared`, `unique`, `faithfulness`, `sparsity`.
- common reporting envelope:
  - `input_id`, `params`, `scores`, `stats`, `evidence`.
- common experiment metadata:
  - model id, prompt/input spec, hardware, dtype, seed, commit hash.

This enables cross-framework comparison in thesis tables without codebase coupling.

## 5. Proposal Language Guidance

Claim the following:

- "MACAG and CDEA are separate implementations under a coordinated architectural program."
- "The next phase standardizes interfaces and reporting to enable comparative studies."
- "Repository separation is an intentional staging decision to reduce integration risk."

Avoid claiming:

- "A unified codebase is already implemented."
- "Both frameworks share identical optimization pipelines."

## 6. Post-Prelims Convergence Criteria

Only consider deeper convergence after these are true:

- stable game definitions and objective terms across at least two evaluation suites,
- mature backend adapter boundaries,
- stable reporting schema with low churn,
- no loss in runtime or reproducibility from added abstraction.

## 7. How to Cite This in the Proposal

Use one short paragraph:

"This work follows a staged architecture strategy. In the current stage, MACAG (mechanistic circuit evidence allocation) and CDEA (vision-focused contrastive decomposition) remain in separate repositories to preserve implementation velocity and methodological clarity. In the next stage, we align both systems through shared interface contracts for objectives, scoring, solvers, and reporting, enabling controlled cross-framework comparison without premature codebase unification."
