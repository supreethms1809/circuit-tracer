"""Re-score checkpoints on the raw vs. de-duplicated val holdout.

Loads each checkpoint (full model, no FSDP needed for eval) and reports absolute
reconstruction MSE, nmse (FVU), and L0 over each val cache. Use it to see how much
the duplicated/leaked windows inflate the logged val_loss, and to re-pick `_best`
vs `_final` on a clean holdout.

    python scripts/rescore_clean_val.py \
        --ckpt best=/…/_best final=/…/_final \
        --val  raw=/…/val/gpt2 clean=/…/val_clean/gpt2
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from spline_clt.kan_transcoder import load_spline_clt
from spline_clt.training.data import ActivationDataset

_DEF_CKPT = (
    "/cluster/ai4wy/gscratch/ssuresh/results/paper/paper_gpt2_small_spline_1b_v2/"
    "runs/spline_feature_match_gpt2_small_1b_v2/seed_101/checkpoints/"
    "spline_fm_gpt2_small_1b_v2_seed101_"
)
_DEF_VAL = "/gscratch/ssuresh/shared/activations/gpt2_2b_v2/"


@torch.no_grad()
def score(model, val_dir: str, device, dtype, max_samples: int) -> tuple[float, float, float, int]:
    # RAM-load a bounded slice (contiguous read) — far faster than the streaming
    # per-sample mmap path when we just need a fixed eval set on GPU.
    ds = ActivationDataset.load(val_dir, split="val", max_samples=max_samples)
    # batch 64: the dense decode materializes (n_layers, b*seq, d_transcoder); at
    # d_transcoder=12288 a larger batch balloons to tens of GB and thrashes.
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    tot_mse = tot_y2 = 0.0
    l0s: list[float] = []
    nb = 0
    import time as _t
    _t0 = _t.time()
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
        if nb % 8 == 0:
            print(f"      [{val_dir.split('/')[-2]}] batch {nb}, "
                  f"{nb * 64} seqs ({_t.time() - _t0:.0f}s)", flush=True)
    mse = tot_mse / max(1, nb)
    y2 = tot_y2 / max(1, nb)
    return mse, mse / max(1e-8, y2), sum(l0s) / max(1, len(l0s)), len(ds)


def _kv(pairs: list[str], default_suffixes: dict[str, str]) -> dict[str, str]:
    if pairs:
        return {p.split("=", 1)[0]: p.split("=", 1)[1] for p in pairs}
    return default_suffixes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", nargs="*", default=[], help="name=path pairs")
    ap.add_argument("--val", nargs="*", default=[], help="name=dir pairs")
    ap.add_argument("--max-samples", type=int, default=100000,
                    help="cap sequences RAM-loaded per val set (clean has ~2577)")
    args = ap.parse_args()

    ckpts = _kv(args.ckpt, {"best": _DEF_CKPT + "best", "final": _DEF_CKPT + "final"})
    # Default to the clean holdout only; the raw full-set number is known (nmse~0.027)
    # and the streaming full raw eval is slow. Add raw=... via --val to include it.
    vals = _kv(args.val, {"clean": _DEF_VAL + "val_clean/gpt2"})

    # --- skip the wasteful efficient-kan spline init: curve2coeff (an lstsq) runs
    # at KANLinear construction, and every one of those weights is overwritten from
    # the checkpoint on load. Zero-init the spline instead. ---
    import efficient_kan  # noqa: E402
    import math as _math

    def _fast_reset(self):  # noqa: ANN001
        torch.nn.init.kaiming_uniform_(self.base_weight, a=_math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            self.spline_weight.data.zero_()
            if getattr(self, "enable_standalone_scale_spline", False):
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=_math.sqrt(5) * self.scale_spline
                )

    efficient_kan.KANLinear.reset_parameters = _fast_reset

    import time as _t

    # --- Phase 1: load ALL checkpoints on CPU BEFORE any CUDA call. Initializing
    # the CUDA context and THEN running heavy/threaded CPU BLAS deadlocks on the
    # GH200 Grace CPU; keeping all host-side loading strictly pre-CUDA avoids it. ---
    cpu_models = {}
    for cname, cpath in ckpts.items():
        _l0 = _t.time()
        print(f"  [cpu-load] {cname} …", flush=True)
        m = load_spline_clt(cpath, device=torch.device("cpu"), dtype=torch.float32)
        m.eval()
        cpu_models[cname] = m
        print(f"  [cpu-load] {cname} done ({_t.time() - _l0:.0f}s)", flush=True)

    # --- Phase 2: now touch CUDA, move each model to GPU, and eval. ---
    if not torch.cuda.is_available():
        raise SystemExit("[rescore] CUDA not available — need a GPU for d_transcoder=12288 eval.")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    print(f"device={device} n_gpu={torch.cuda.device_count()} dtype=bf16", flush=True)
    print(f"{'ckpt':>7} {'val':>7} {'nseq':>7} {'absMSE':>9} {'nmse':>8} {'L0':>7}", flush=True)
    print("-" * 52, flush=True)
    rows = {}
    for cname, m in cpu_models.items():
        model = m.to(device)
        for vname, vdir in vals.items():
            mse, nmse, l0, n = score(model, vdir, device, dtype, args.max_samples)
            rows[(cname, vname)] = nmse
            print(f"{cname:>7} {vname:>7} {n:>7} {mse:>9.4f} {nmse:>8.4f} {l0:>7.1f}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n== best-vs-final on CLEAN val (lower nmse = better generalization) ==")
    for vname in vals:
        pair = {c: rows.get((c, vname)) for c in ckpts}
        if all(v is not None for v in pair.values()):
            winner = min(pair, key=pair.get)
            print(f"  {vname}: " + ", ".join(f"{c}={pair[c]:.4f}" for c in ckpts) + f"  -> {winner}")


if __name__ == "__main__":
    main()
