"""Diagnostic: reproduce the val_loss discrepancy on the real 2-GPU FSDP path.

Context: `paper_gpt2_small_spline_1b_v2` logged val_loss ~0.2375, but reloading
the `_best` checkpoint and running the identical encode->decode_dense->recon path
gives abs MSE ~0.736 / nmse ~0.092 (which matches the train-step recon log). The
gap only lives in the runtime eval path, which runs under
`FSDP.summon_full_params(..., rank0_only=True)` with `MixedPrecision(bf16)`. This
script reproduces that exact path and compares it, on the SAME val data, against:

  A) runtime path : real `evaluate()` on an FSDP-wrapped model inside summon (bf16)
  B) truth        : plain fp32 unwrapped model.encode/decode_dense (rank 0)
  C) train-forward: model_for_train(x) tuple path inside summon (what training logs)

Whatever the mechanism (dtype, summon, path divergence), the three numbers localize
it. Also dumps threshold + activation-density stats as each path actually sees them.

Run (2 GPUs):
    torchrun --nproc_per_node=2 scripts/diag_val_eval.py
Env overrides:
    DIAG_CKPT  : checkpoint dir (default: v2 seed101 _best)
    DIAG_VAL   : val cache dir  (default: gpt2_2b_v2 val)
    DIAG_MAX   : cap val sequences (default: 2048; set 0 for the full streaming set)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from spline_clt.kan_transcoder import load_spline_clt
from spline_clt.training.data import ActivationDataset
from spline_clt.training.loss import reconstruction_loss
from spline_clt.training.train import TrainConfig, evaluate

CKPT = os.environ.get(
    "DIAG_CKPT",
    "/cluster/ai4wy/gscratch/ssuresh/results/paper/paper_gpt2_small_spline_1b_v2/"
    "runs/spline_feature_match_gpt2_small_1b_v2/seed_101/checkpoints/"
    "spline_fm_gpt2_small_1b_v2_seed101_best",
)
VAL = os.environ.get(
    "DIAG_VAL", "/gscratch/ssuresh/shared/activations/gpt2_2b_v2/val/gpt2"
)
MAX = int(os.environ.get("DIAG_MAX", "2048"))  # 0 => full streaming val


def _p(rank: int, *a) -> None:
    if rank == 0:
        print(*a, flush=True)


def recon_over(model, ds, device, dtype, rank) -> tuple[float, float, float]:
    """abs MSE, nmse, L0 over ds via the encode/decode_dense eval path."""
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    tot_mse = tot_y2 = 0.0
    l0s: list[float] = []
    nb = 0
    with torch.no_grad():
        for batch in dl:
            x = batch["mlp_inputs"].to(device=device, dtype=dtype)
            y = batch["mlp_outputs"].to(device=device, dtype=dtype)
            b, nl, s, d = x.shape
            x = x.permute(1, 0, 2, 3).reshape(nl, b * s, d)
            y = y.permute(1, 0, 2, 3).reshape(nl, b * s, d)
            a = model.encode(x)
            yh = model.decode_dense(a, input_acts=x)
            tot_mse += ((yh.float() - y.float()) ** 2).mean().item()
            tot_y2 += (y.float() ** 2).mean().item()
            l0s.append((a > 0).float().sum(dim=(0, 2)).mean().item())
            nb += 1
    mse = tot_mse / max(1, nb)
    y2 = tot_y2 / max(1, nb)
    return mse, mse / max(1e-8, y2), sum(l0s) / max(1, len(l0s))


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    dtype = torch.bfloat16
    _p(rank, f"=== diag_val_eval | world={world} | ckpt={CKPT}")
    _p(rank, f"    val={VAL} | max_samples={MAX or 'FULL'} | eval dtype=bf16")

    # --- Model to FSDP-wrap: build in fp32 (mirrors build_session), load _best ---
    model = load_spline_clt(CKPT, device=device, dtype=torch.float32)
    thr = model.effective_threshold.float()
    _p(rank, f"    loaded threshold mean={thr.mean():.4f} min={thr.min():.4f} "
             f"max={thr.max():.4f} | enc_std L5 max={model.enc_input_std[5].float().max():.1f}")

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    model_for_train = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        ignored_modules=list(model.encoders),
        device_id=device,
    )

    # --- val dataset: streaming (runtime) or bounded slice ---
    if MAX > 0:
        val_ds = ActivationDataset.load(VAL, split="val", max_samples=MAX)
    else:
        val_ds = ActivationDataset.load(VAL, split="val")
    _p(rank, f"    val sequences: {len(val_ds)}")

    cfg = TrainConfig(
        n_layers=model.n_layers, d_model=model.d_model,
        d_transcoder=model.d_transcoder, encoder_type="kan",
        batch_size=64, num_workers=0, pin_memory=False,
        persistent_workers=False, prefetch_factor=None, dtype="bfloat16",
    )

    # === Path A: runtime eval — real evaluate() inside summon (bf16, rank0_only) ===
    val_loss_A = float("nan")
    dist.barrier()
    with FSDP.summon_full_params(model_for_train, writeback=False, rank0_only=True):
        if rank == 0:
            val_loss_A = evaluate(model, val_ds, cfg, device, dtype)
            # Path C: train-forward tuple path on one batch, same summoned params
            b0 = next(iter(DataLoader(val_ds, batch_size=64, num_workers=0)))
            x = b0["mlp_inputs"].to(device, dtype); y = b0["mlp_outputs"].to(device, dtype)
            bb, nl, s, d = x.shape
            x = x.permute(1, 0, 2, 3).reshape(nl, bb * s, d)
            y = y.permute(1, 0, 2, 3).reshape(nl, bb * s, d)
            acts, yh, _ = model.forward(x)
            recon_C = ((yh.float() - y.float()) ** 2).mean().item()
            l0_C = (acts > 0).float().sum(dim=(0, 2)).mean().item()
            thr_in = model.effective_threshold.float()
            _p(rank, f"    [inside summon] threshold mean={thr_in.mean():.4f} "
                     f"max={thr_in.max():.4f} | yh|abs|mean={yh.float().abs().mean():.4f} "
                     f"y|abs|mean={y.float().abs().mean():.4f}")
            _p(rank, f"    Path C (train-forward, 1 batch): recon={recon_C:.4f} L0={l0_C:.1f}")
    dist.barrier()

    # === Path B: truth — unwrapped fp32 model.encode/decode on rank 0 ===
    # (summon writeback=False left the sharded params intact; rebuild clean fp32.)
    if rank == 0:
        truth = load_spline_clt(CKPT, device=device, dtype=torch.float32)
        mse_bf, nmse_bf, l0_bf = recon_over(truth, val_ds, device, torch.bfloat16, rank)
        mse_fp, nmse_fp, l0_fp = recon_over(truth, val_ds, device, torch.float32, rank)
        print("\n================= RESULTS (rank 0) =================", flush=True)
        print(f"Path A  runtime evaluate() [FSDP summon, bf16] : val_loss = {val_loss_A:.4f}")
        print(f"Path B  truth eval  [fp32 unwrapped, bf16 data]: abs MSE  = {mse_bf:.4f}  nmse={nmse_bf:.4f}  L0={l0_bf:.1f}")
        print(f"Path B  truth eval  [fp32 unwrapped, fp32 data]: abs MSE  = {mse_fp:.4f}  nmse={nmse_fp:.4f}  L0={l0_fp:.1f}")
        print("Expected from logs: runtime val_loss~0.2375 ; true recon~0.73 (nmse~0.092)")
        print("If A≈0.24 and B≈0.73  -> bug is IN the FSDP-summon eval path.")
        print("If A≈B≈0.73           -> runtime 0.2375 came from other live state, look elsewhere.")
        print("===================================================", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
