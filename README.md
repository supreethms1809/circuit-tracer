# Spline Circuit Tracer

Spline-CLT replaces the linear encoder in Anthropic's [cross-layer transcoder (CLT)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) pipeline with a B-spline / KAN encoder, enabling nonlinear feature detection while preserving downstream circuit tracing, intervention, and visualization workflows. The decoder stays linear so features retain clean directions in residual stream space.

This repository is a fork of [`safety-research/circuit-tracer`](https://github.com/safety-research/circuit-tracer) extended with:

- a Spline-CLT encoder (`spline_clt/`) that can be swapped between `kan` and `linear` modes,
- a config-driven paper evaluation runner (`paper-eval`) for NeurIPS-style suites.

## Core Hypothesis

Anthropic's CLT assumes linear feature boundaries (linear encoder → JumpReLU). Features defined by nonlinear activation patterns are missed. A KAN encoder captures these while keeping the decoder linear, so downstream circuit tracing and intervention workflows continue to apply.

```
Standard CLT:   a^l = JumpReLU(W_enc^l · x^l)
Spline-CLT:     a^l = JumpReLU(KAN_enc^l(x^l))
Decoder (both): y_hat^l = Σ W_dec^(l'→l) · a^(l')
```

## Installation

### 1. Create the conda environment

```bash
conda create -n ct python -y
conda activate ct
```

### 2. Install PyTorch

Pick the build that matches your CUDA / platform from the [PyTorch install matrix](https://pytorch.org/get-started/locally/). 

### 3. Install this repository

From the repo root:

```bash
pip install -e .
```

This installs `circuit-tracer`, `spline_clt`, `macag`, and the `paper-eval` entrypoint, along with `transformer-lens`, `efficient-kan`, and the rest of the dependencies pinned in `pyproject.toml`.

### 4. Verify the install

```bash
python -c "import circuit_tracer, spline_clt, macag; print('ok')"
pytest tests/test_paper_config.py tests/test_kan_encoder.py -q
```

## Quick Start: Paper Suites

The conference-facing entrypoint is `paper-eval`. It expands a JSON suite under `experiments/paper_configs/suites/` through dataset collection, training, evaluation, aggregation, and report generation.

### Validate a suite (no compute)

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json --validate-only
```

### Print the expanded job matrix

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json --dry-run
```

### End-to-end CPU smoke trial

Exercises every stage of the pipeline on a tiny budget; finishes in a few minutes on CPU.

```bash
paper-eval --suite experiments/paper_configs/suites/smoke_trial_gpt2.json
```

### Full NeurIPS GPT-2 suite

Recommended on a GH200 / H100-class node:

```bash
paper-eval --suite experiments/paper_configs/suites/neurips_core_gpt2.json
```

### Available suites

| Suite | Purpose |
|---|---|
| `suites/smoke_trial_gpt2.json` | End-to-end smoke validation on CPU |
| `suites/neurips_core_gpt2.json` | Core GPT-2 small comparison (spline vs linear) |
| `suites/neurips_high_gpt2.json` | High-budget GPT-2 small |
| `suites/neurips_high_gpt2_pythia160m.json` | High-budget GPT-2 + Pythia-160M |

### Operational flags

- `--re-evaluate` — drop eval artifacts but keep checkpoints
- `--worker-id` / `--num-workers` — shard a suite across nodes
- `--stages` — intersect with the suite's enabled stages (e.g. `--stages train,evaluate`)

For final campaigns, do not override suite-chosen seeds or hyperparameters on the CLI; use `--stages` and worker sharding only.

### Suite outputs

Each suite writes:

```
resolved_config.json     # fully resolved OmegaConf snapshot
manifest.json            # job matrix + artifact paths
per_example_metrics.jsonl
aggregate_metrics.json
tables.csv
report.md
figures/
```

Full output schema and metric definitions live in [`docs/paper-evaluation.md`](docs/paper-evaluation.md), [`docs/metric_definitions.md`](docs/metric_definitions.md), and [`docs/methodology_comparison.md`](docs/methodology_comparison.md).

## Manual Pipeline (without the runner)

The paper runner orchestrates these steps, but you can also run them individually.

### Collect activations

```bash
python experiments/train_spline_clt.py --collect-data --model gpt2 --device cuda
```

Writes ~40 GB to `data/activations/`. The training path streams from the on-disk mmap, so RAM stays O(batch).

### Train

```bash
# Spline-CLT
python experiments/train_spline_clt.py --config experiments/configs/gpt2_small.yaml

# Matched linear baseline
python experiments/train_spline_clt.py --config experiments/configs/gpt2_small_linear_baseline.yaml
```

### Evaluate and compare

```bash
python experiments/run_pipeline.py \
    --kan-checkpoint    checkpoints/gpt2_small/spline_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --output-dir results/eval_run1 \
    --shapley
```

### Single-prompt circuit

```bash
python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --model gpt2 \
    --max-features 64 \
    --output results/circuits/eiffel.pt
```

### Spline shape analysis

```bash
python experiments/analyze_splines.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --n-features 20 \
    --output-dir results/splines
```

## Attribution Methods

| Method | File | Cost | Notes |
|---|---|---|---|
| Causal ablation | `attribution/causal.py` | O(n_active) forward passes | Exact per-feature causal effect |
| Jacobian encoder vectors | `spline_clt/kan_encoder.py` | local | Local linear approximation of the KAN encoder, compatible with the upstream circuit-tracer attribution pipeline |

## Testing

```bash
# Paper / integration regression set
pytest \
    tests/test_graph.py \
    tests/test_attribution.py \
    tests/test_paper_config.py \
    tests/test_paper_reporting.py \
    tests/test_paper_runner.py \
    tests/test_replacement_model_alignment.py -q

# Core Spline-CLT tests
pytest tests/test_kan_encoder.py tests/test_kan_transcoder.py \
       tests/test_attribution.py -v

# Everything, including upstream circuit-tracer tests
pytest tests/ -v
```

## Repository Layout

```
spline_clt/        # Spline-CLT package: encoder, transcoder, training, paper runner
attribution/       # Causal attribution, graph adapters
eval/              # Replacement accuracy, monosemanticity metrics
experiments/       # Training entrypoints, evaluation pipeline, paper configs
circuit_tracer/    # Upstream circuit-tracer (graph build, prune, frontend)
docs/              # Paper evaluation, metric definitions, methodology comparison
tests/
```

A more detailed map, including the role of each module, lives in [`CLAUDE.md`](CLAUDE.md).

## Upstream Circuit-Tracer

The upstream `circuit_tracer` CLI (`circuit-tracer attribute ...`), Neuronpedia integration, tutorial notebooks, and pretrained transcoder list (Gemma-2 2B, Llama-3.2 1B, Qwen-3 family) remain available unchanged. See the [upstream README](https://github.com/safety-research/circuit-tracer) and [`demos/circuit_tracing_tutorial.ipynb`](demos/circuit_tracing_tutorial.ipynb) for the original linear-CLT workflow.

## Cite

Upstream circuit-tracer:

```
@misc{circuit-tracer,
  author = {Hanna, Michael and Piotrowski, Mateusz and Lindsey, Jack and Ameisen, Emmanuel},
  title = {circuit-tracer},
  howpublished = {\url{https://github.com/safety-research/circuit-tracer}},
  year = {2025}
}
```
