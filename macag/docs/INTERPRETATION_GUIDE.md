# Interpretation Guide for MACAG Outputs

Use this when results "do not make sense" at first glance.

## 1. Read JSON in This Order

1. `params`: confirms what optimization problem you actually solved.
2. `evidence`: selected sets and decomposition.
3. `scores`: faithfulness and utility quantities.
4. `stats`: computational profile and convergence.

Files are produced by `macag/cli/run_macag.py`.

## 2. Meaning of Core Scores

For a set \(E\):

- `all`: full-circuit score.
- `empty`: full-ablation score on the intervention universe.
- `keep_only`: score when only \(E\) remains active.
- `remove`: score when \(E\) is ablated from the full circuit.

Derived:

- `sufficiency = keep_only - empty`
- `necessity = all - remove`
- `faithfulness = alpha * sufficiency + (1-alpha) * necessity`

Interpretation:

- High sufficiency means selected evidence alone can recover behavior.
- High necessity means removing selected evidence harms behavior.

## 3. Game 2 Decomposition Semantics

- `shared = E_y ∩ E_foil`
- `unique_y = E_y - E_foil`
- `unique_foil = E_foil - E_y`

If `E_foil` is empty, your foil objective was weak under current settings (foil token, penalties, or candidate set).

## 4. Common Failure Modes

## A. "Everything is from one layer"

Usually candidate filtering caused it. Check:

- `--candidates-file` contents,
- `--prefilter-top-k`,
- layer distribution in candidates.

## B. "Only one node selected"

Possible causes:

- \(\lambda\) too high,
- budget too low,
- candidate set too narrow,
- weak foil definition in Game 2.

## C. "No meaningful foil decomposition"

Possible causes:

- foil token is not competitive for that prompt,
- \(\beta\) too high too early,
- \(\lambda\) too high for foil side,
- insufficient ABR iterations.

## D. "Run is active but silent"

Progress is enabled by default now. If no logs appear:

- verify you are on updated code,
- ensure stdout is not redirected,
- check if process is alive.

## 5. Recommended Sanity Checks Before Claiming Results

- Verify target/foil token IDs are single-token and semantically valid.
- Run at least one full-candidate configuration (no prefilter).
- Compare full vs prefiltered candidate runs.
- Sweep \(\beta\) and show overlap trend.
- Report oracle calls and caching stats for transparency.

## 6. Device Caveat

Current MPS path is not fully supported for CLT safetensors loading (`safe_open(..., device="mps:0")` fails). Use CPU or CUDA for reliable runs.
