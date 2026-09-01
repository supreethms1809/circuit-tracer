"""Verify JumpReLU optimizer grouping under the production FSDP wrapper."""

from __future__ import annotations

import os
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from spline_clt.training.train import (
    TrainConfig,
    _build_optimizers,
    _gather_optimizer_state,
    _load_optimizer_from_training_state,
    _threshold_metrics,
    build_session,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    with tempfile.TemporaryDirectory(prefix=f"fsdp-threshold-r{rank}-") as tmpdir:
        config = TrainConfig(
            n_layers=2,
            d_model=8,
            d_transcoder=16,
            encoder_type="linear",
            threshold_init=0.01,
            jumprelu_bandwidth=0.001,
            normalize_inputs=False,
            batch_size=1,
            total_steps=1,
            warmup_steps=0,
            use_fsdp=True,
            dtype="bfloat16",
            device="cuda",
            checkpoint_dir=tmpdir,
            log_dir=tmpdir,
            wandb_mode="disabled",
        )
        session = build_session(config)
        groups = session.optimizer.param_groups
        assert len(groups) == 2, f"expected 2 optimizer groups, got {len(groups)}"
        assert groups[0]["weight_decay"] == config.weight_decay
        assert groups[0]["eps"] == 1e-8
        assert groups[1]["weight_decay"] == config.threshold_weight_decay
        assert groups[1]["eps"] == config.threshold_adam_eps

        threshold = session.model.activation_function.threshold
        assert any(p is threshold for p in groups[1]["params"])
        before = threshold.detach().float().clone()

        # Put every preactivation inside the narrow STE window so this diagnostic
        # tests optimizer plumbing rather than randomly missing a 1e-3 interval.
        with FSDP.summon_full_params(session.model_for_train, writeback=True), torch.no_grad():
            for encoder in session.model.encoders:
                encoder.W_enc.zero_()
            session.model.b_enc.fill_(config.threshold_init + 0.25 * config.jumprelu_bandwidth)

        x = torch.zeros(
            config.n_layers,
            32,
            config.d_model,
            device=session.device,
            dtype=session.dtype,
        )
        activations, _, _ = session.model_for_train(x)
        threshold_metrics = _threshold_metrics(
            session.model,
            distributed=True,
            device=session.device,
        )
        assert threshold_metrics["stats/threshold_mean"] > 0
        loss = activations.float().sum()
        loss.backward()
        local_grad_max = torch.tensor(
            0.0 if threshold.grad is None else float(threshold.grad.detach().abs().max()),
            device=session.device,
        )
        dist.all_reduce(local_grad_max, op=dist.ReduceOp.MAX)
        assert torch.isfinite(local_grad_max)
        assert local_grad_max > 0
        session.optimizer.step()
        after = threshold.detach().float()
        local_update_max = torch.tensor(
            0.0 if after.numel() == 0 else float((after - before).abs().max()),
            device=session.device,
        )
        dist.all_reduce(local_update_max, op=dist.ReduceOp.MAX)
        assert local_update_max > 0
        saved_main, saved_local = _gather_optimizer_state(
            session.model_for_train,
            session.optimizer,
            session.optimizer_local,
            use_fsdp=True,
            rank=rank,
        )
        if rank == 0:
            assert len(saved_main["param_groups"]) == 2
        restored_optimizer, restored_local = _build_optimizers(
            session.model,
            session.model_for_train,
            config,
            lr=config.learning_rate,
        )
        _load_optimizer_from_training_state(
            session.model_for_train,
            restored_optimizer,
            restored_local,
            saved_main,
            saved_local,
            use_fsdp=True,
            distributed=True,
        )
        assert len(restored_optimizer.param_groups) == 2

        if rank == 0:
            print(
                {
                    "optimizer_groups": len(groups),
                    "main_weight_decay": groups[0]["weight_decay"],
                    "main_eps": groups[0]["eps"],
                    "threshold_weight_decay": groups[1]["weight_decay"],
                    "threshold_eps": groups[1]["eps"],
                    "threshold_grad_abs_max": float(local_grad_max),
                    "threshold_update_abs_max": float(local_update_max),
                    "saved_optimizer_groups": len(saved_main["param_groups"]),
                    "restored_optimizer_groups": len(
                        restored_optimizer.param_groups
                    ),
                },
                flush=True,
            )

        if session.log_file is not None:
            session.log_file.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
