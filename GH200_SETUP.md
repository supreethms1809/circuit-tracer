# GH200 Setup Guide

## Environment Setup

```bash
git clone git@github.com:supreethms1809/circuit-tracer.git
cd circuit-tracer
git checkout claude/silly-feistel
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
```

## Data Collection (run on the GH200)

```bash
python experiments/train_kan_clt.py --collect-data --model gpt2 --device cuda
```

## Training

```bash
python experiments/train_kan_clt.py \
    --config experiments/configs/gpt2_small.yaml \
    --device cuda
```

## Key Notes for GH200

- The Grace CPU and Hopper GPU share a unified memory bus (NVLink-C2C at 900 GB/s), so data loading is very fast — no PCIe bottleneck
- Use `bfloat16` over `float16` — GH200's Hopper architecture has better bf16 throughput and no loss scaling needed
- The KAN encoder's B-spline basis computation is compute-bound (not memory-bound), so the GH200's FP32/BF16 throughput will be the bottleneck — `bfloat16` helps here
- `efficient-kan` uses standard PyTorch ops, so it works on ARM (Grace) + Hopper without any custom CUDA kernels needed
