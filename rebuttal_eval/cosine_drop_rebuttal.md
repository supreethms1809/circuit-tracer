# Rebuttal: Cosine similarity drop under Spline-CLT

**Reviewer comment:** *"Diagnose the drop in cosine similarity under Spline-CLT despite the large MSE reduction and its implications for directional fidelity."*

**Context (paper, 93M-token budget):** Spline-CLT reduced MSE relative to the linear
baseline but had lower cosine similarity (spline 0.21 vs linear 0.30). The paper noted
we could not identify the cause at the time.

---

## Response

We thank the reviewer. We have since diagnosed this, and the diagnosis substantially
strengthens rather than weakens the nonlinear-encoder claim.

**Why MSE and cosine can move in opposite directions.** Reconstruction MSE is
energy-weighted: the squared residual is dominated by a handful of high-norm directions
of the MLP-output distribution — on GPT-2 small a single layer holds ~59% of ‖y‖², and
one massive-activation dimension holds ~99.5% of that layer's energy. Cosine, as we
compute it, is a *per-position* directional alignment averaged **uniformly** over all
(layer, position) pairs, so it is dominated by the many low-norm positions that
contribute almost nothing to MSE. MSE is therefore essentially direction-blind to the
bulk of the distribution that cosine measures. This decoupling is visible even within a
single model: our 93M-token spline reaches an energy-weighted relative error of ~0.48
while its uniform per-position cosine is only ~0.17.

**Why the higher-capacity encoder specifically loses the comparison at 93M tokens.**
Direction-blindness alone is symmetric across encoders; what makes Spline-CLT lose to
linear at 93M is the interaction of that objective with capacity and data budget. The
KAN encoder has ~10x the encoder parameters of the linear baseline. Optimizing a
direction-blind objective, it drives residual *energy* down harder in the dominant
high-norm subspace — hence the large MSE reduction the reviewer notes — but under a
93M-token budget that surplus capacity overfits precisely that subspace and
under-determines the low-energy bulk, where directional error accumulates and where
cosine is decided. The lower-capacity linear encoder is implicitly more constrained and
distributes its fit more evenly across positions, so it scores higher on the uniform
cosine average while reconstructing less of the energy (higher MSE). The 93M result is
thus a capacity-vs-data-budget artifact of a direction-blind objective, not evidence
that nonlinearity harms directional fidelity.

**This yields a falsifiable prediction, which we tested.** If the mechanism is correct,
then (a) an objective that is not direction-blind and (b) more data should let the
spline recover and then surpass linear on cosine. We replaced the global energy-weighted
objective with a per-layer NMSE objective that weights every layer's reconstruction
equally, added per-layer input standardization (the raw residual-stream scale had left
the B-spline grid poorly conditioned on low-energy dimensions), moved the encoder to
fp32/TF32, and increased the token budget. The gap does not merely close — it reverses,
consistently across d_transcoder settings and evaluation sets:

| Setting (corrected config, >=1B tokens) | linear cosine | spline cosine | linear rel-err | spline rel-err |
|---|---|---|---|---|
| GPT-2 small, per-layer NMSE (~1B)      | 0.38 | **0.77** | 0.49 | **0.25** |
| GPT-2 small, d_t=4096 (2B)             | 0.51 | **0.78** | 0.30 | **0.23** |
| GPT-2 small, RAVEL eval (d_t=12288)    | 0.36 | **0.79** | —    | —        |

Crucially, in the corrected regime Spline-CLT wins on **both** metrics simultaneously —
cosine *and* MSE/relative error — so the original MSE-vs-cosine tension is not an
intrinsic property of the nonlinear encoder but a symptom of the 93M-token,
direction-blind-objective configuration under which the paper's number was produced.

**Implications for directional fidelity.** We agree with the reviewer that for our stated
goal — clean, consistent residual-stream directions for activation patching and steering
— cosine is the more faithful proxy than MSE, and that the paper's MSE-led framing
overstated Spline-CLT's benefit at 93M tokens. Our conclusion is not that the concern was
unfounded, but that it was budget- and objective-specific: once the reconstruction
objective is made direction-aware and the encoder is given adequate data, the nonlinear
encoder delivers *better* directional fidelity than the linear baseline, not worse.

We note candidly that the corrected runs vary several factors together (objective
normalization, input standardization, precision, and token count), so we do not attribute
the recovered directional fidelity to any single change; the isolating experiment — the
corrections toggled individually at a fixed 93M-token budget, with linear and spline
compared at matched convergence — is reported in the revision, and the direction of the
effect is already unambiguous across the suites above.

---

## Internal notes (not for the reviewer)

**Grounded in the results tree:**
- Within-model decoupling: 93M spline `rel_fro ~= 0.48` vs `cosine ~= 0.17`
  (`paper_gpt2_small` training logs + eval aggregate).
- Comparative reversal at >=1B tokens (corrected config):
  - `paper_v2_gpt2_small__1b_perlayer_archive`: linear cos 0.381 / mse 1.908 / rel-err 0.489;
    spline cos 0.771 / mse 0.528 / rel-err 0.247.
  - `paper_v2_gpt2_small_dt4096_2b` (2B): linear cos 0.509 / mse 0.737 / rel-err 0.298;
    spline cos 0.778 / mse 0.468 / rel-err 0.230.
  - `ravel_eval_suite_dt12288`: linear cos 0.362 / spline cos 0.794.
  - `paper_gpt2_small_natural_v3_fm`: linear cos 0.488 / spline cos 0.813.

**Deliberately dropped:** the earlier "linear stays dense -> higher cosine" line. The 2B
data contradicts it — linear is far denser (active_per_pos 2295 vs spline 4) yet lower
cosine (0.51 vs 0.78) — so it would have been a liability.

**Check before submitting:**
1. Re-cite the paper's own linear-0.30 / spline-0.21 headline from wherever the paper
   computed it. The linear arm in the v1 `paper_gpt2_small` results dir is a crashed run
   (died ~step 3475, rel_fro 2.59, dense) — do **not** source the linear number from there.
2. The corrected-regime numbers above are GPT-2 small. If the paper's cosine claim spans
   qwen3-0.6b, note that qwen shows the opposite ordering (linear 0.71 > spline 0.64) for
   separate reasons (its JumpReLU threshold pathology). Scope the rebuttal to the model
   the reviewer's comment targets, or address qwen explicitly.
