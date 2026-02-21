# KAN-CLT Implementation Tasks

Work through these phases sequentially. Each phase has a clear go/no-go checkpoint before proceeding.

---

## Phase 0: Environment & Codebase Setup
**Goal**: Get circuit-tracer running, understand its internals, set up project structure.

- [ ] **0.1** Fork `github.com/safety-research/circuit-tracer` into project
- [ ] **0.2** Install circuit-tracer and run the tutorial notebook (`demos/circuit_tracing_tutorial.ipynb`) on GPT-2 or Gemma-2-2B to verify everything works
- [ ] **0.3** Clone `github.com/Blealtan/efficient-kan` and run its basic test to verify KAN layer works
- [ ] **0.4** Read and annotate these specific files in circuit-tracer:
  - `circuit_tracer/replacement_model.py` — understand how CLT replaces MLPs
  - `circuit_tracer/attribution.py` (or equivalent) — find where backward Jacobians are computed, find where `W_enc` matrix multiply happens
  - `circuit_tracer/transcoder.py` (or equivalent) — understand the transcoder data structure (encoder weights, decoder weights, JumpReLU threshold)
- [ ] **0.5** Create the project directory structure as specified in CLAUDE.md
- [ ] **0.6** Write a short document: `docs/circuit_tracer_internals.md` listing:
  - Which functions assume linear encoder (file, line number, what they do)
  - How transcoder weights are loaded and stored
  - How the ReplacementModel hooks into the base model's forward pass
  - What format the attribution graph uses (node types, edge format)

**Checkpoint**: Can you run circuit-tracer end-to-end and can you point to every line that assumes a linear encoder? If yes, proceed.

---

## Phase 1: KAN Encoder Module
**Goal**: Build a standalone KAN encoder that has the same interface as the linear encoder in circuit-tracer's transcoder.

- [ ] **1.1** Implement `kan_clt/kan_encoder.py`:
  - Class `KANEncoder` that wraps efficient-kan's `KANLinear`
  - Input: residual stream activations `x` of shape `(batch, d_model)` where d_model=768 for GPT-2
  - Output: feature pre-activations of shape `(batch, n_features)`
  - Hyperparameters: `grid_size` (default 5), `spline_order` (default 3), `n_features`
  - Must support: `forward(x)`, and a property `basis_expansion_dim` so we know the effective dimensionality

- [ ] **1.2** Write `tests/test_kan_encoder.py`:
  - Test shapes: input (batch, 768) → output (batch, n_features)
  - Test gradient flow: can we backprop through it?
  - Test spline inspection: can we extract the learned spline coefficients for a specific feature?
  - Compare parameter count vs equivalent linear encoder

- [ ] **1.3** Verify compute feasibility: time a forward + backward pass with realistic dimensions
  - d_model=768, n_features=4096, batch_size=256, grid_size=5
  - Compare wall time against linear encoder W_enc @ x
  - If >10x slower, consider reducing grid_size to 3 or using fast-kan (RBF approximation)

**Checkpoint**: KAN encoder produces correct shapes, gradients flow, and forward pass is < 10x slower than linear. If yes, proceed.

---

## Phase 2: KAN Cross-Layer Transcoder
**Goal**: Build the full KAN-CLT module that can replace MLPs in a transformer.

- [ ] **2.1** Implement `kan_clt/kan_transcoder.py`:
  - Class `KANCrossLayerTranscoder`
  - For each layer l: one `KANEncoder` (reads residual stream at layer l)
  - For each layer l: linear decoder matrices `W_dec^(l→l'), W_dec^(l→l+1), ...` (writes to all subsequent layers)
  - JumpReLU activation with learnable threshold per feature
  - `forward(x_l)` → feature activations `a_l`
  - `decode(a_l, target_layer)` → contribution to MLP output at target_layer
  - `reconstruct_mlp(layer_l)` → sum of all decoder contributions from this and previous layers

- [ ] **2.2** Implement `kan_clt/training/loss.py`:
  - `reconstruction_loss(y_hat, y_true)` — MSE between KAN-CLT output and true MLP output, summed across layers
  - `sparsity_loss(activations, decoder_norms, lambda, c)` — same formula as Anthropic: λ Σ tanh(c · ||W_dec_i|| · a_i)
  - `total_loss(model, batch)` — combines both

- [ ] **2.3** Implement `kan_clt/training/data.py`:
  - `ActivationDataset`: run GPT-2 on OpenWebText subset, cache residual stream activations and MLP outputs at all layers
  - Store as memory-mapped files for efficient training
  - Target: ~10M tokens worth of activations

- [ ] **2.4** Implement `kan_clt/training/train.py`:
  - Standard PyTorch training loop
  - Adam optimizer (NOT LBFGS — doesn't scale to this size)
  - Learning rate schedule: warmup + cosine decay
  - Logging: reconstruction loss per layer, sparsity, feature activation density
  - Checkpointing: save best model by validation reconstruction loss
  - Config via dataclass or YAML

- [ ] **2.5** Write `tests/test_kan_transcoder.py`:
  - Test that reconstruction shapes match MLP output shapes
  - Test that cross-layer decoding works (feature at layer 3 can write to layer 5)
  - Test loss computation

**Checkpoint**: KAN-CLT can be instantiated, forward pass produces correct shapes, loss is computed. If yes, proceed to training.

---

## Phase 3: Train and Validate KAN-CLT
**Goal**: Train KAN-CLT on GPT-2 small, verify it reconstructs MLPs adequately.

- [ ] **3.1** Collect activation dataset from GPT-2 small on OpenWebText (~10M tokens)
- [ ] **3.2** Train KAN-CLT with initial hyperparameters:
  - n_features_per_layer: 4096 (match typical SAE/CLT size)
  - grid_size: 5, spline_order: 3
  - sparsity λ: sweep [0.01, 0.05, 0.1]
  - Learning rate: 1e-4 with warmup
  - Train for ~50K steps, evaluate every 5K
- [ ] **3.3** Also train a baseline linear CLT with the same architecture (just swap KAN encoder for linear encoder) at matched parameter count — this is your direct comparison
- [ ] **3.4** Evaluate replacement model accuracy:
  - Top-1 match rate between KAN-CLT replacement model and original GPT-2
  - Compare against baseline linear CLT
  - Compare per-layer reconstruction MSE
- [ ] **3.5** Evaluate feature sparsity:
  - Average number of active features per token position
  - Feature activation frequency distribution (should be heavy-tailed)
  - Compare against baseline

**Checkpoint**: KAN-CLT replacement model achieves comparable or better top-1 accuracy to matched linear CLT. If significantly worse (>10% gap), debug before proceeding — try adjusting grid_size, sparsity, learning rate. If reconstruction fundamentally fails, this is a negative result worth documenting.

---

## Phase 4: Causal Attribution Engine
**Goal**: Build attribution graphs without assuming encoder linearity.

- [ ] **4.1** Implement `attribution/causal.py`:
  - `ablation_attribution(replacement_model, prompt, target_token)`:
    - Run forward pass, collect all active features and their activations
    - For each active feature s: set a_s = 0, run forward pass, measure change in every downstream feature and output logit
    - Edge weight A_{s→t} = a_t_original - a_t_ablated
    - Return graph in same format as circuit-tracer expects
  - Optimization: batch ablations where possible (ablate features at different positions simultaneously if they don't interact)

- [ ] **4.2** Implement `attribution/shapley.py`:
  - `shapley_attribution(replacement_model, prompt, target_token, n_samples=1000)`:
    - Identify set of active features S at each position
    - For output logit: compute Shapley values using permutation sampling
    - For each target feature t: compute Shapley values of source features that feed into it
    - Return graph with Shapley values as edge weights
  - Use antithetic sampling for variance reduction

- [ ] **4.3** Implement `attribution/graph.py`:
  - Ensure output format is compatible with circuit-tracer's pruning and visualization
  - Node types: feature, embedding, error, logit (same as circuit-tracer)
  - Edge format: source_id, target_id, weight (same as circuit-tracer)

- [ ] **4.4** Write `tests/test_attribution.py`:
  - Test on tiny model (2-layer transformer) where you can verify attribution by hand
  - Test that causal attribution edges sum to approximately the total effect
  - Test that Shapley attribution satisfies efficiency axiom (values sum to total payoff)

- [ ] **4.5** Test the full pipeline: KAN-CLT → causal attribution → pruning → visualization
  - Use circuit-tracer's pruning code on your graphs
  - Verify the visualization frontend renders your graphs correctly

**Checkpoint**: Can generate attribution graphs for arbitrary prompts, graphs render in the frontend, and attribution values are numerically sensible. If yes, proceed.

---

## Phase 5: Evaluation and Comparison
**Goal**: Rigorously compare KAN-CLT against standard CLT.

- [ ] **5.1** Feature monosemanticity comparison:
  - For both KAN-CLT and baseline CLT: compute max-activating examples for top 200 features
  - Automated scoring: use a language model to rate concept coherence of max-activating examples
  - Manual inspection: pick 50 features from each, rate interpretability
  - Compare distributions

- [ ] **5.2** Circuit extraction on IOI task:
  - Prompt format: "John gave Mary the book. She gave it to ___"
  - Extract circuit using both KAN-CLT and baseline CLT
  - Compare: number of features needed for 90% causal effect
  - Compare: interpretability of identified features

- [ ] **5.3** Circuit extraction on addition task:
  - Prompt format: "25 + 37 = " 
  - Same comparison as IOI

- [ ] **5.4** Circuit extraction on factual recall:
  - Prompt format: "The capital of France is"
  - Same comparison

- [ ] **5.5** Spline interpretability analysis (unique to KAN-CLT):
  - For top features: extract learned spline shapes
  - Run PySR symbolic regression on spline input-output pairs
  - Document features where symbolic form is interpretable
  - This analysis is impossible with linear CLT — it's your unique contribution

- [ ] **5.6** Mechanistic faithfulness:
  - For each method: perturb features in the attribution graph direction, measure whether downstream effects match predictions
  - Compare KAN-CLT causal attribution vs CLT Jacobian attribution

- [ ] **5.7** Ablation studies:
  - Replace KAN encoder with MLP encoder of same parameter count — does the advantage hold?
  - Vary grid_size: 3, 5, 7, 10 — is there a sweet spot?
  - Vary sparsity: does monosemanticity scale with sparsity for KAN features as it does for linear features?

**Checkpoint**: You have quantitative results comparing KAN-CLT vs CLT across multiple metrics and tasks. Write up results regardless of whether KAN-CLT wins — negative results are publishable too.

---

## Phase 6: Paper and Integration
**Goal**: Write up results, connect to thesis framework.

- [ ] **6.1** Game-theoretic integration:
  - Frame the Shapley attribution as your multi-agent evidence allocation framework
  - Each feature is an agent, each computes a local function, Shapley allocates credit
  - Show how this connects to your Dynamic Anchors / MADA work

- [ ] **6.2** Write paper:
  - Intro: linearity assumption in current circuit tracing is a bottleneck
  - Method: KAN-CLT architecture + causal/Shapley attribution
  - Results: comparison tables, circuit visualizations, spline analysis
  - Discussion: when does nonlinearity matter? connection to thesis

- [ ] **6.3** Package and release:
  - Clean up code, add documentation
  - Release as extension to circuit-tracer

---

## Quick Reference: What To Tell Claude Code

When starting a Claude Code session, say something like:

**Session 1 (setup):**
> "Read CLAUDE.md and TASKS.md. We're starting Phase 0. Fork circuit-tracer, set up the project, and help me understand the circuit-tracer internals — specifically which functions assume a linear encoder."

**Session 2 (KAN encoder):**
> "Read CLAUDE.md. We're on Phase 1. Implement the KAN encoder module using efficient-kan. Start with kan_encoder.py and tests."

**Session 3 (training):**
> "Read CLAUDE.md. We're on Phase 2. Build the full KAN-CLT transcoder and training pipeline."

And so on. The CLAUDE.md gives Claude Code the architectural context it needs without re-explaining every session.
