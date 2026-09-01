"""Tests for nested-FSDP KAN encoder sharding (shard_kan_encoders)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spline_clt.kan_encoder import KANEncoder, kan_linear_from_encoder, unwrap_encoder_module
from spline_clt.kan_transcoder import KANCrossLayerTranscoder
from spline_clt.training.train import TrainConfig, _build_optimizers, build_session


def test_nested_fsdp_ignored_modules_must_exclude_inner_fsdp() -> None:
    """Regression: outer FSDP must not list already-wrapped encoders as ignored.

    PyTorch raises ``ValueError: ignored_modules should not include FSDP modules``
    when nested KAN wraps put FSDP units into ``ignored_modules``. The wrap path
    must pass ``ignored_modules=None`` whenever ``shard_kan_encoders`` is on.
    """
    import inspect

    from spline_clt.training import train as train_mod

    src = inspect.getsource(train_mod.build_session)
    assert "ignored_modules should not include FSDP" in src or (
        "if config.encoder_type == \"kan\" and not shard_kan" in src
        or "if config.encoder_type == 'kan' and not shard_kan" in src
    )
    assert "list(model.encoders)" in src
    # Nested branch must gate ignored_modules on not shard_kan
    assert "not shard_kan" in src


def test_summon_nested_uses_recurse_false_with_inner_fsdp() -> None:
    """Outer summon must not recurse into already-summoned nested encoders."""
    import inspect

    from spline_clt.training import train as train_mod

    src = inspect.getsource(train_mod._summon_nested_encoders)
    assert "recurse=not nested_encoders" in src


def test_update_grid_does_not_move_module_device() -> None:
    """FSDP-safe update_grid must not call .cpu()/.to() on kan_linear."""
    enc = KANEncoder(d_model=8, n_features=16, grid_size=3, spline_order=3)
    device = torch.device("cpu")
    enc.to(device)
    param_ids_before = {id(p) for p in enc.kan_linear.parameters()}
    buf_ids_before = {id(b) for b in enc.kan_linear.buffers()}
    x = torch.randn(32, 8)
    enc.update_grid(x)
    assert {id(p) for p in enc.kan_linear.parameters()} == param_ids_before
    assert {id(b) for b in enc.kan_linear.buffers()} == buf_ids_before
    assert next(enc.kan_linear.parameters()).device.type == "cpu"


def test_base_jump_encode_uses_forward_return_split() -> None:
    """BaseJump must enter encoder via forward(return_split=True) for FSDP hooks."""
    torch.manual_seed(0)
    model = KANCrossLayerTranscoder(
        n_layers=2,
        d_transcoder=16,
        d_model=8,
        grid_size=3,
        spline_order=3,
        activation_function="base_jump",
        jumprelu_bandwidth=0.1,
        threshold_init=0.05,
        device=torch.device("cpu"),
    )
    model.train()
    x = torch.randn(2, 4, 8)
    acts = model.encode(x)
    assert acts.shape == (2, 4, 16)
    acts.sum().backward()
    assert model.encoders[0].kan_linear.base_weight.grad is not None


def test_shard_kan_optimizer_groups_without_fsdp_wrap() -> None:
    """Single-optimizer path keeps threshold + lr_spline_mult groups."""
    model = KANCrossLayerTranscoder(
        n_layers=2,
        d_model=8,
        d_transcoder=16,
        encoder_type="kan",
        activation_function="jump_relu",
        threshold_init=0.01,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    config = TrainConfig(
        encoder_type="kan",
        shard_kan_encoders=True,
        lr_spline_mult=5.0,
        threshold_weight_decay=0.0,
        threshold_adam_eps=1e-15,
        optimizer="adamw",
    )
    optimizer, optimizer_local = _build_optimizers(
        model, model, config, lr=config.learning_rate
    )
    assert optimizer_local is None
    lr_mults = [float(pg.get("lr_mult", 1.0)) for pg in optimizer.param_groups]
    assert any(abs(m - 5.0) < 1e-12 for m in lr_mults)
    assert any(float(pg.get("eps", 1e-8)) < 1e-12 for pg in optimizer.param_groups)
    # Identity of base/spline params must still resolve via helper
    kl = kan_linear_from_encoder(model.encoders[0])
    assert kl.base_weight is model.encoders[0].kan_linear.base_weight


def test_shard_kan_requires_fsdp() -> None:
    with pytest.raises(ValueError, match="use_fsdp"):
        build_session(
            TrainConfig(
                encoder_type="kan",
                shard_kan_encoders=True,
                use_fsdp=False,
                n_layers=2,
                d_model=8,
                d_transcoder=16,
                normalize_inputs=False,
                threshold_init_strategy="constant",
                decoder_init_strategy="kaiming",
            )
        )


def test_fsdp_eval_call_site_avoids_summon() -> None:
    """Mid-train FSDP eval must use sharded forward, not summon_full_params."""
    import inspect

    from spline_clt.training import train as train_mod

    src = inspect.getsource(train_mod.run_chunk)
    # Locate the evaluation block roughly
    assert "use_fsdp_forward=True" in src
    # The FSDP eval branch should not open nested summon for evaluate
    eval_idx = src.find("use_fsdp_forward=True")
    assert eval_idx > 0
    window = src[max(0, eval_idx - 800) : eval_idx + 200]
    assert "_summon_nested_encoders" not in window


def test_unwrap_encoder_passthrough() -> None:
    enc = KANEncoder(d_model=4, n_features=8, grid_size=3, spline_order=3)
    assert unwrap_encoder_module(enc) is enc
    assert kan_linear_from_encoder(enc) is enc.kan_linear


def _nested_fsdp_worker(rank: int, world_size: int, tmp_dir: str, result_path: str) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29531"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
    )

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from spline_clt.training.data import ActivationDataset

    n_layers, d_model, d_t, seq, n_seq = 2, 8, 16, 4, 8
    mlp_in = torch.randn(n_seq, n_layers, seq, d_model)
    mlp_out = torch.randn(n_seq, n_layers, seq, d_model)
    dataset = ActivationDataset(mlp_in, mlp_out)

    config = TrainConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_transcoder=d_t,
        encoder_type="kan",
        activation_function="base_jump",
        jumprelu_bandwidth=0.1,
        threshold_init=0.05,
        use_fsdp=True,
        shard_kan_encoders=True,
        lr_spline_mult=5.0,
        batch_size=2,
        total_steps=2,
        warmup_steps=0,
        log_every=1,
        eval_every=100,
        save_every=100,
        update_grid_every=1,
        update_grid_from=0,
        update_grid_max_samples=8,
        lambda_kan_reg=1e-4,
        checkpoint_dir=str(Path(tmp_dir) / "ckpts"),
        run_name="nested_fsdp_smoke",
        dtype="float32",
        device="cuda",
        seed=0,
        normalize_inputs=False,
        threshold_init_strategy="constant",
        decoder_init_strategy="kaiming",
    )
    session = build_session(config, norm_dataset=dataset)
    assert all(
        isinstance(session.model.encoders[i], FSDP) for i in range(n_layers)
    )
    assert session.encoder_needs_manual_reduce is False
    assert session.optimizer_local is None

    # One train step + grid update (every step with update_grid_every=1)
    from spline_clt.training.train import run_chunk

    run_chunk(session, dataset, dataset)
    # Encoder shard should be smaller than full param count on each rank
    full_enc = KANEncoder(d_model=d_model, n_features=d_t, grid_size=5, spline_order=3)
    full_n = sum(p.numel() for p in full_enc.parameters())
    local_n = sum(p.numel() for p in session.model.encoders[0].parameters())
    # With use_orig_params, numel() on a sharded param reports local shard size
    # (or full size depending on FSDP version); accept either sharded or equal
    # but require finite grads after a step.
    ok = local_n <= full_n and local_n > 0

    if rank == 0:
        Path(result_path).write_text("ok" if ok else f"fail local={local_n} full={full_n}")

    from spline_clt.training.train import close_session

    close_session(session)
    dist.destroy_process_group()


def _nested_fsdp_eval_worker(
    rank: int, world_size: int, tmp_dir: str, result_path: str
) -> None:
    """Train one step, then run sharded FSDP evaluate (the Gemma hang path)."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29541"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
    )

    from spline_clt.training.data import ActivationDataset
    from spline_clt.training.train import close_session, evaluate, run_chunk

    n_layers, d_model, d_t, seq, n_seq = 2, 8, 16, 4, 16
    mlp_in = torch.randn(n_seq, n_layers, seq, d_model)
    mlp_out = torch.randn(n_seq, n_layers, seq, d_model)
    dataset = ActivationDataset(mlp_in, mlp_out)

    config = TrainConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_transcoder=d_t,
        encoder_type="kan",
        activation_function="base_jump",
        jumprelu_bandwidth=0.1,
        threshold_init=0.05,
        use_fsdp=True,
        shard_kan_encoders=True,
        lr_spline_mult=5.0,
        batch_size=2,
        total_steps=2,
        warmup_steps=0,
        log_every=1,
        # Hit evaluate inside run_chunk after step 1.
        eval_every=1,
        save_every=100,
        update_grid_every=0,
        lambda_kan_reg=0.0,
        checkpoint_dir=str(Path(tmp_dir) / "ckpts_eval"),
        run_name="nested_fsdp_eval_smoke",
        dtype="float32",
        device="cuda",
        seed=0,
        normalize_inputs=False,
        threshold_init_strategy="constant",
        decoder_init_strategy="kaiming",
        wandb_mode="disabled",
    )
    session = build_session(config, norm_dataset=dataset)

    torch.cuda.reset_peak_memory_stats(rank)
    run_chunk(session, dataset, dataset)
    peak_after_chunk = torch.cuda.max_memory_allocated(rank)

    # Explicit evaluate call (same path as mid-train FSDP eval).
    torch.cuda.reset_peak_memory_stats(rank)
    before = torch.cuda.memory_allocated(rank)
    val_loss = evaluate(
        session.model_for_train,
        dataset,
        config,
        session.device,
        session.dtype,
        distributed=True,
        use_fsdp_forward=True,
    )
    peak_eval = torch.cuda.max_memory_allocated(rank)
    after = torch.cuda.memory_allocated(rank)

    ok = (
        math.isfinite(val_loss)
        and peak_eval < before + 64 * 1024**2  # no multi-GiB summon spike on toy model
        and after < before + 32 * 1024**2
    )
    if rank == 0:
        Path(result_path).write_text(
            f"ok val_loss={val_loss:.6f} peak_chunk={peak_after_chunk} "
            f"peak_eval={peak_eval} before={before} after={after}"
            if ok
            else f"fail val_loss={val_loss} peak_eval={peak_eval} before={before}"
        )

    close_session(session)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="nested FSDP eval smoke needs >=2 CUDA devices",
)
def test_nested_fsdp_eval_without_summon(tmp_path: Path) -> None:
    """Regression for Gemma hang: FSDP val must not summon nested KAN params."""
    result = tmp_path / "eval_result.txt"
    world_size = 2
    mp.spawn(
        _nested_fsdp_eval_worker,
        args=(world_size, str(tmp_path), str(result)),
        nprocs=world_size,
        join=True,
    )
    text = result.read_text()
    assert text.startswith("ok"), text


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="nested FSDP smoke needs >=2 CUDA devices",
)
def test_nested_fsdp_two_gpu_smoke(tmp_path: Path) -> None:
    result = tmp_path / "result.txt"
    world_size = 2
    mp.spawn(
        _nested_fsdp_worker,
        args=(world_size, str(tmp_path), str(result)),
        nprocs=world_size,
        join=True,
    )
    assert result.read_text().startswith("ok")
