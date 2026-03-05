# Prelims Defense Q&A (MACAG)

Use these as concise responses for likely committee questions.

## Q1. What is your core contribution?

I propose and implement a game-theoretic framework for intervention-faithful evidence allocation on mechanistic circuit graphs, with:

- Game 1 for sparse faithful evidence extraction,
- Game 2 for contrastive decomposition into shared and unique evidence.

## Q2. Why game theory here?

Game 2 naturally models competing explanatory objectives for target vs foil. The overlap penalty formalizes contrastiveness rather than treating it as post hoc subtraction.

## Q3. How is faithfulness defined?

By interventions, not attributions alone:

- sufficiency via keep-only,
- necessity via remove-from-full,
- combined through a tunable \(\alpha\) mix.

## Q4. Why greedy/ABR instead of global optimization?

Exact global search is intractable with expensive intervention scoring. Greedy and ABR are deterministic, simple, and provide a strong proof-of-concept baseline.

## Q5. What makes this implementation credible?

- Explicit scoring protocol and backend abstraction,
- memoized oracle with call/caching statistics,
- real-model intervention backend and toy backend,
- CLI reproducibility and JSON outputs,
- unit tests covering scoring, games, factory parsing, and annotation.

## Q6. What are the main limitations?

- Heuristic optimization (no global optimality guarantees),
- dependence on candidate pool quality,
- sensitivity to foil choice,
- runtime dominated by intervention calls,
- current MPS compatibility issue for CLT safetensors.

## Q7. How do you avoid overclaiming?

I frame the system as a feasibility demonstration, report sensitivity analyses, and distinguish intervention-faithful subset quality from claims of complete or unique mechanistic truth.

## Q8. What are immediate next steps after prelims?

- richer baselines and ablations,
- broader prompt suite and statistical reporting,
- solver improvements (e.g., swap/add/drop local search),
- modular game registry for adding new games,
- improved interpretability metadata and feature labeling.

## Q9. What would count as success for the next phase?

- stable contrastive decompositions across prompts and hyperparameters,
- clear overlap-vs-beta behavior,
- reproducible runtime/quality tradeoff curves,
- evidence that extracted sets are both sparse and intervention-faithful.

## Q10. Why is this thesis-worthy?

It connects formal objective design, mechanistic interventions, algorithmic search, and empirical evaluation in a single coherent system that can be extended into a broader research program.

## Q11. Why keep MACAG and CDEA in separate repositories?

They currently optimize different objects (discrete node sets vs continuous masks) with different computational backends. Separate repositories reduce integration risk and let each framework mature without premature abstraction coupling.

## Q12. If they are separate, how are they scientifically connected?

Through shared conceptual contracts: objective decomposition, contrastive/shared-unique reporting, and standardized experiment metadata. I treat them as staged implementations under one research program, not unrelated tools.

## Q13. Are you planning a unified codebase later?

Possibly, but only after interfaces stabilize. The immediate plan is architecture alignment and schema interoperability first; full codebase unification is explicitly deferred until it improves reliability and reproducibility rather than adding overhead.
