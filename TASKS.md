# Spline-CLT Implementation Tasks

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

- [ ] **1.1** Implement `spline_clt/kan_encoder.py`:
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

## Phase 2: Spline Cross-Layer Transcoder
**Goal**: Build the full Spline-CLT module that can replace MLPs in a transformer.

- [ ] **2.1** Implement `spline_clt/kan_transcoder.py`:
  - Class `KANCrossLayerTranscoder`
  - For each layer l: one `KANEncoder` (reads residual stream at layer l)
  - For each layer l: linear decoder matrices `W_dec^(l→l'), W_dec^(l→l+1), ...` (writes to all subsequent layers)
  - JumpReLU activation with learnable threshold per feature
  - `forward(x_l)` → feature activations `a_l`
  - `decode(a_l, target_layer)` → contribution to MLP output at target_layer
  - `reconstruct_mlp(layer_l)` → sum of all decoder contributions from this and previous layers

- [ ] **2.2** Implement `spline_clt/training/loss.py`:
  - `reconstruction_loss(y_hat, y_true)` — MSE between Spline-CLT output and true MLP output, summed across layers
  - `sparsity_loss(activations, decoder_norms, lambda, c)` — same formula as Anthropic: λ Σ tanh(c · ||W_dec_i|| · a_i)
  - `total_loss(model, batch)` — combines both

- [ ] **2.3** Implement `spline_clt/training/data.py`:
  - `ActivationDataset`: run GPT-2 on OpenWebText subset, cache residual stream activations and MLP outputs at all layers
  - Store as memory-mapped files for efficient training
  - Target: ~10M tokens worth of activations

- [ ] **2.4** Implement `spline_clt/training/train.py`:
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

**Checkpoint**: Spline-CLT can be instantiated, forward pass produces correct shapes, loss is computed. If yes, proceed to training.

---

## Phase 3: Train and Validate Spline-CLT
**Goal**: Train Spline-CLT on GPT-2 small, verify it reconstructs MLPs adequately.

- [x] **3.1** Collect activation dataset from GPT-2 small on OpenWebText (~8.5K sequences, ~40GB)
- [x] **3.2** Train Spline-CLT (running on GH200, config: `experiments/configs/gpt2_small.yaml`)
  - d_transcoder=4096, grid_size=5, spline_order=3, λ=0.05, lr=1e-4, 50K steps
- [x] **3.3** Linear CLT baseline: `experiments/configs/gpt2_small_linear_baseline.yaml`
  - Same d_transcoder=4096, identical training setup, encoder_type=linear
- [ ] **3.4** Evaluate replacement model accuracy (`experiments/compare_models.py` ready):
  - `python experiments/compare_models.py --kan-checkpoint ... --linear-checkpoint ...`
  - Per-layer MSE, cosine similarity, relative error, active features/pos
- [ ] **3.5** Evaluate feature sparsity (included in compare_models.py output)

**Checkpoint**: Spline-CLT replacement model achieves comparable or better top-1 accuracy to matched linear CLT. If significantly worse (>10% gap), debug before proceeding — try adjusting grid_size, sparsity, learning rate. If reconstruction fundamentally fails, this is a negative result worth documenting.

---

## Phase 4: Causal Attribution Engine
**Goal**: Build attribution graphs without assuming encoder linearity.

- [x] **4.1** `attribution/causal.py` — single-feature ablation attribution
  - `ablation_attribution()`: zero out a_s, measure change in all downstream features and output
  - `build_attribution_graph()`: full adjacency matrix compatible with circuit-tracer Graph format

- [x] **4.2** `attribution/shapley.py` — Monte Carlo Shapley attribution
  - `shapley_attribution()`: permutation sampling + antithetic pairs, target="reconstruction"|"feature"
  - `shapley_logit_attribution()`: project onto logit direction for token-level attribution
  - `_feature_to_feature_shapley()`: (n_active × n_active) feature-to-feature matrix

- [x] **4.3** `attribution/graph.py` API verified — perfect match with `circuit_tracer.graph.Graph.__init__` signature

- [x] **4.4** Tests: `tests/test_attribution.py` (6 tests) + `tests/test_shapley.py` (13 tests)

- [ ] **4.5** End-to-end pipeline test with trained checkpoint:
  - `python experiments/run_circuit.py --checkpoint <path> --prompt "..." --shapley`
  - Verify graph renders in circuit-tracer visualization frontend

**Checkpoint**: Can generate attribution graphs for arbitrary prompts, graphs render in the frontend, and attribution values are numerically sensible. If yes, proceed.

---

## Phase 5: Evaluation and Comparison
**Goal**: Rigorously compare Spline-CLT against standard CLT.

- [ ] **5.1** Feature monosemanticity comparison (`eval/monosemanticity.py` ready):
  - `collect_max_activating_examples(model, dataset, top_n_features=200)`
  - Returns Gini coefficients, max-activating token positions, JSON export
  - Automated scoring: use a language model to rate concept coherence
  - Manual inspection: pick 50 features from each, rate interpretability

- [ ] **5.2** Circuit extraction on IOI task:
  - `python experiments/run_circuit.py --prompt "John gave Mary the book. She gave it to" --shapley`
  - Compare: number of features needed for 90% causal effect vs linear CLT
  - Compare: interpretability of identified features

- [ ] **5.3** Circuit extraction on addition task:
  - `python experiments/run_circuit.py --prompt "25 + 37 ="`

- [ ] **5.4** Circuit extraction on factual recall:
  - `python experiments/run_circuit.py --prompt "The capital of France is"`

- [x] **5.5** Spline interpretability analysis (`experiments/analyze_splines.py` ready):
  - Extracts spline transfer functions for top features by activation frequency
  - Saves CSV + PNG plots; `--no-plot` for headless environments
  - Optional: pipe CSV to PySR for symbolic regression
  - `python experiments/analyze_splines.py --checkpoint <path> --n-features 20`

- [ ] **5.6** Mechanistic faithfulness:
  - Perturb features in attribution direction, measure downstream effects match predictions
  - Compare Spline-CLT causal attribution vs linear CLT Jacobian attribution

- [ ] **5.7** Ablation studies:
  - Vary grid_size: 3, 5, 7, 10 — is there a sweet spot?
  - Vary sparsity λ: does monosemanticity scale with sparsity for KAN features?
  - MLP encoder at matched parameter count (to isolate nonlinearity vs parameter count)

**Checkpoint**: You have quantitative results comparing Spline-CLT vs CLT across multiple metrics and tasks. Write up results regardless of whether Spline-CLT wins — negative results are publishable too.

---

## Phase 6: Paper and Integration
**Goal**: Write up results, connect to thesis framework.

- [ ] **6.1** Game-theoretic integration:
  - Frame the Shapley attribution as your multi-agent evidence allocation framework
  - Each feature is an agent, each computes a local function, Shapley allocates credit
  - Show how this connects to your Dynamic Anchors / MADA work

- [ ] **6.2** Write paper:
  - Intro: linearity assumption in current circuit tracing is a bottleneck
  - Method: Spline-CLT architecture + causal/Shapley attribution
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
> "Read CLAUDE.md. We're on Phase 2. Build the full Spline-CLT transcoder and training pipeline."

**Session N (evaluation, current):**
> "Read CLAUDE.md and TASKS.md. Training is complete on the GH200. Checkpoints are in checkpoints/. Run Phase 5 evaluation: compare_models.py, run_circuit.py on IOI/addition/factual recall tasks, analyze_splines.py."

The CLAUDE.md gives Claude Code the architectural context it needs without re-explaining every session.
