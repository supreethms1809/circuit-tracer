#!/usr/bin/env python3
"""Minimal multi-node torch.distributed connectivity smoke test.

Verifies c10d rendezvous + process-group init + CPU and (if available) NCCL
all-reduce. Intended for cluster hostname / DNS / IB cutovers — not training.

Launch the same way as paper-eval multinode::

    srun ... torchrun --nnodes=$SLURM_NNODES --nproc_per_node=2 \\
        --rdzv_backend=c10d --rdzv_id=$SLURM_JOB_ID \\
        --rdzv_endpoint=$HEAD_NODE:$MASTER_PORT \\
        scripts/diag_multinode_rdzv.py
"""

from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist


def _env(name: str, default: str = "?") -> str:
    return os.environ.get(name, default)


def main() -> None:
    local_rank = int(_env("LOCAL_RANK", "0"))
    rank = int(_env("RANK", "0"))
    world_size = int(_env("WORLD_SIZE", "1"))
    host = socket.gethostname()
    try:
        fqdn = socket.getfqdn()
    except OSError:
        fqdn = host

    print(
        f"[rdzv-smoke] start host={host} fqdn={fqdn} "
        f"rank={rank}/{world_size} local_rank={local_rank} "
        f"MASTER_ADDR={_env('MASTER_ADDR')} MASTER_PORT={_env('MASTER_PORT')} "
        f"cuda={'yes' if torch.cuda.is_available() else 'no'}",
        flush=True,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    t0 = time.perf_counter()
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        device_id=device if device.type == "cuda" else None,
    )
    init_s = time.perf_counter() - t0
    print(
        f"[rdzv-smoke] init_process_group ok backend={backend} "
        f"rank={rank} took={init_s:.2f}s",
        flush=True,
    )

    # CPU gloo path always available via a separate group if we used NCCL,
    # but a single all_reduce on the default group is enough.
    t1 = time.perf_counter()
    x = torch.ones(1, device=device, dtype=torch.float32) * (rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = float(world_size * (world_size + 1) / 2)
    got = float(x.item())
    if abs(got - expected) > 1e-3:
        raise RuntimeError(
            f"rank {rank}: all_reduce sum mismatch got={got} expected={expected}"
        )
    reduce_s = time.perf_counter() - t1

    # Gather hostnames so rank 0 can print the full membership map.
    hosts = [None] * world_size if rank == 0 else None
    dist.gather_object(f"{host}|{fqdn}", hosts, dst=0)

    if rank == 0:
        print(
            f"[rdzv-smoke] all_reduce ok sum={got} (expected={expected}) "
            f"took={reduce_s:.3f}s",
            flush=True,
        )
        print("[rdzv-smoke] membership:", flush=True)
        assert hosts is not None
        for r, h in enumerate(hosts):
            print(f"  rank {r}: {h}", flush=True)
        print(
            f"[rdzv-smoke] PASS world_size={world_size} "
            f"init={init_s:.2f}s reduce={reduce_s:.3f}s",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rdzv-smoke] rank {rank} exit clean", flush=True)


if __name__ == "__main__":
    main()
