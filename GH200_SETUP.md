# GH200 Setup Guide

## Environment Setup

```bash
git clone git@github.com:supreethms1809/circuit-tracer.git
cd circuit-tracer
git checkout claude/silly-feistel
conda activate ct
pip install -e ".[dev]"
pip install git+https://github.com/Blealtan/efficient-kan.git
pip install datasets pyyaml  # for training pipeline
```

## Training Config Changes

Edit `experiments/configs/gpt2_small.yaml`:

```yaml
# GH200 has 96GB HBM3 — you can be much more aggressive
device: cuda
dtype: bfloat16          # GH200 has native bf16 support, 2x memory savings
batch_size: 32           # up from 4, GH200 has plenty of memory
d_transcoder: 4096       # can go to 8192 or 16384 if you want
data_dir: data/activations  # default collection path
```

## Full End-to-End Pipeline (recommended)

Runs data collection → Spline-CLT training → linear CLT baseline → evaluation → comparison.
All stages auto-skip if outputs already exist.

```bash
conda activate ct

# Quick smoke-test (2–3 min, CPU-only, verifies the full pipeline works)
python experiments/run_e2e.py \
    --device cpu --dtype float32 \
    --n-tokens 5000 --total-steps 20 --d-transcoder 64 --batch-size 2

# Full GH200 run (collect 10M tokens, train 50k steps, d_transcoder=4096)
python experiments/run_e2e.py \
    --device cuda --dtype bfloat16 \
    --n-tokens 10000000 --total-steps 50000 \
    --d-transcoder 4096 --grid-size 5 \
    --batch-size 32 \
    --output-dir results/gh200_run1

# Full run + circuit tracing, spline analysis, monosemanticity
python experiments/run_e2e.py \
    --device cuda --dtype bfloat16 \
    --n-tokens 10000000 --total-steps 50000 \
    --d-transcoder 4096 --batch-size 32 \
    --run-circuits --run-splines --run-monosemanticity \
    --output-dir results/gh200_run1_full
```

Final output is a side-by-side comparison table showing Spline-CLT vs linear CLT:
- MSE, cosine similarity, relative error per layer
- Sparsity (active features/position, activation density)
- Verdict: "Spline-CLT better/worse by X% MSE"

Results and a Markdown report are written to `--output-dir`.

## Step-by-Step (if you need to run stages individually)

### Data Collection

Uses `Salesforce/wikitext` (wikitext-2-raw-v1) by default — Parquet-based, no deprecated
HuggingFace dataset scripts required.

```bash
conda activate ct
python experiments/train_spline_clt.py --collect-data --model gpt2 --device cuda
```

Activations are saved to `data/activations/` (configured via `data_dir` in the YAML).

### Training (Spline-CLT only)

```bash
conda activate ct
python experiments/train_spline_clt.py \
    --config experiments/configs/gpt2_small.yaml \
    --device cuda
```

### Evaluation only (with existing checkpoints)

```bash
# Reconstruction metrics + comparison table
python experiments/compare_models.py \
    --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations --device cuda

# Full evaluation pipeline (circuits, splines, monosemanticity)
python experiments/run_pipeline.py \
    --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --output-dir results/eval_run1 \
    --device cuda --dtype bfloat16

# Circuit tracing for a single prompt
python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --device cuda
```

## Memory Profile (measured on a 48 GB GPU, batch_size=4 × 128 tokens)

| Phase              | GPU memory |
|--------------------|-----------|
| Model (4096 feat)  | ~1.3 GB   |
| Forward + backward | ~2.5 GB   |
| Peak (training)    | ~15 GB    |

The training loop uses a dense decode path (`decode_dense`) which avoids the OOM that
would occur with the sparse decoder when features are not yet sparse at initialization.
At batch_size=32 on a GH200 (96 GB), peak usage will be ~30–35 GB — well within limits.

## Key Notes for GH200

- The Grace CPU and Hopper GPU share a unified memory bus (NVLink-C2C at 900 GB/s), so data loading is very fast — no PCIe bottleneck
- Use `bfloat16` over `float16` — GH200's Hopper architecture has better bf16 throughput and no loss scaling needed
- The KAN encoder's B-spline basis computation is compute-bound (not memory-bound), so the GH200's FP32/BF16 throughput will be the bottleneck — `bfloat16` helps here
- `efficient-kan` uses standard PyTorch ops, so it works on ARM (Grace) + Hopper without any custom CUDA kernels needed
