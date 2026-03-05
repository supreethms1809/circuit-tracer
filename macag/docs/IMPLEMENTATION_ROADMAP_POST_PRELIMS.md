# Implementation Roadmap (Post-Prelims)

This roadmap is for planned framework changes, not current implemented state.

## Phase 0: Freeze Current Baseline

Goals:

- keep current MACAG runs reproducible,
- avoid breaking thesis experiment scripts while architecture work is pending.

Deliverables:

- pinned experiment configs and command templates,
- baseline results snapshot and interpretation notes.

## Phase 1: Internal Modularization in MACAG

Goals:

- align MACAG layout to CDEA-style separation of concerns.

Tasks:

- extract formal objective terms into dedicated modules,
- isolate solver interfaces from concrete game definitions,
- standardize typed run/result objects,
- centralize reporting utilities.

Exit criteria:

- no behavior regression on existing test suite,
- identical JSON outputs for baseline runs (except added metadata keys).

## Phase 2: Shared Contract Definition Across Repos

Goals:

- enable consistent evaluation and reporting across MACAG and CDEA.

Tasks:

- define shared schema for run metadata and evidence decomposition,
- standardize names for overlap/faithfulness/sparsity metrics,
- add lightweight adapters for converting each framework output into the shared schema.

Exit criteria:

- one comparative table can be generated from both repos without manual editing.

## Phase 3: Extensible Game Registry in MACAG

Goals:

- make adding new game types routine and low risk.

Tasks:

- introduce game registration by config + factory pattern,
- separate game definition from solver implementation,
- add validator for required inputs and target/foil semantics.

Exit criteria:

- at least one new game prototype added without modifying core solver internals.

## Phase 4: Backend and Performance Hardening

Goals:

- improve practical usability on real intervention workloads.

Tasks:

- stronger cache policy controls and cache diagnostics,
- better candidate prefilter interfaces,
- device/backend reliability hardening (CPU/MPS/CUDA where supported),
- additional run progress diagnostics and failure recovery messaging.

Exit criteria:

- successful long-run experiments with transparent progress and robust restart behavior.

## Phase 5: Cross-Framework Study Pack

Goals:

- support dissertation-level comparative analysis.

Tasks:

- build shared experiment templates,
- define common ablation suite,
- document threat models and validity constraints per framework.

Exit criteria:

- proposal claims can be backed by side-by-side, schema-consistent evidence.

## Risks and Mitigations

Risk:

- abstraction overhead reduces runtime clarity.
Mitigation:

- keep adapter layers thin and benchmark each refactor step.

Risk:

- schema churn invalidates earlier experiment tables.
Mitigation:

- version output schema and provide migration scripts.

Risk:

- repository drift between CDEA and MACAG.
Mitigation:

- maintain a short shared contract spec and quarterly sync checkpoints.

## Proposal Timeline Mapping

- near-term (prelims): Phase 0 evidence + written architecture plan.
- short-term (first implementation cycle): Phases 1-2.
- mid-term (research expansion): Phases 3-4.
- dissertation integration: Phase 5.
