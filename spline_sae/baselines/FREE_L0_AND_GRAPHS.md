# Experiment protocol (post SAE probes) — 2026-08-02

## Shelved
- Nonlinear decoders (`mlp` / `kan` decoder). Circuit tracing needs a **linear** decoder.
- Matched `target_l0≈51` as the primary linear-vs-spline compare (dictionaries barely overlap).
- **RAVEL** graphs — not used for the graph claim; prefer Neuronpedia published graphs.

## Active
1. **Free-L0 layer transcoder** (Gemma L24, `resid_pre → Δ resid_post`):
   - same `λ_sparsity=4e-4`, `target_l0=0`
   - KAN BaseJump + **linear** decoder vs linear JumpReLU + linear decoder
   - Jobs: see `baselines/freel0_layer_tc_l24_jobids.txt`
   - Report natural L0 + holdout NMSE (secondary)

2. **Graph / attribution primary eval** (Neuronpedia published, not RAVEL):
   - Local copies: `results/neuronpedia_graphs/`
   - Source set: `gemmascope-transcoder-16k` on `gemma-2-2b`
   - Featured: `gemma-fact-dallas-austin`, `gemma-addition`, `gemma-small-big-fr`
   - Also on Neuronpedia graph/info (not featured): `ndag`, `rhyme-bent`, `rhyme-bright`
   - Fair compare = **same prompts**, regenerate graphs with hub Gemma Scope (linear baseline)
     vs a Gemma-compatible Spline-CLT (linear decoder). Do **not** match node IDs into the
     published JSON (those are Gemma Scope feature indices).
   - Metrics: `keep_only_gap_ratio`, `gap_drop_ratio`, `graph_completeness_score`,
     retained feature/error node counts — same as paper runner.
