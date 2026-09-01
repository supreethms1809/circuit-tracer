"""Tests for rebuttal_eval Wave-0 scripts (check_reconstruction, check_params)."""

import numpy as np
import pytest
import torch

from rebuttal_eval import check_params, check_reconstruction
from rebuttal_eval.common import sample_indices, scrub
from spline_clt.kan_transcoder import KANCrossLayerTranscoder


def test_scrub_removes_identifying_paths():
    text = (
        "checkpoint at /gscratch/ssuresh/results/paper/run1 on ai4wy-203 "
        "repo /cluster/ai4wy/home/ssuresh/circuit-tracer user ssuresh"
    )
    cleaned = scrub(text)
    assert "ssuresh" not in cleaned
    assert "gscratch" not in cleaned
    assert "ai4wy" not in cleaned


def test_sample_indices_deterministic():
    a = sample_indices(1000, 32, seed=101)
    b = sample_indices(1000, 32, seed=101)
    c = sample_indices(1000, 32, seed=202)
    assert a == b
    assert a != c
    assert len(a) == 32


def test_row_stats_matches_hand_computation():
    y_true = torch.tensor([[[3.0, 4.0], [1.0, 0.0]]])  # (1 layer, 2 pos, 2 dim)
    y_hat = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    stats = check_reconstruction._row_stats(y_hat, y_true)
    assert stats["cos"][0, 0].item() == pytest.approx(1.0)
    assert stats["cos"][0, 1].item() == pytest.approx(0.0)
    assert stats["rel"][0, 0].item() == pytest.approx(0.0)
    assert stats["rel"][0, 1].item() == pytest.approx(np.sqrt(5.0), rel=1e-5)
    assert stats["y_norm"][0, 0].item() == pytest.approx(5.0)


def test_inequality_logic_synthetic():
    """The universal per-position bound holds for any prediction; the spec
    bound (1 - E[cos]) holds whenever all cosines are non-negative."""
    torch.manual_seed(0)
    y_true = torch.randn(3, 16, 8)
    for scale in (0.0, 0.5, 1.0, -1.0):
        y_hat = y_true * scale + 0.1 * torch.randn_like(y_true)
        stats = check_reconstruction._row_stats(y_hat, y_true)
        rel = stats["rel"].numpy()
        cos = stats["cos"].numpy()
        universal = np.sqrt(np.clip(1 - cos**2, 0, None))
        universal = np.where(cos < 0, np.maximum(universal, 1.0), universal)
        assert rel.mean() >= universal.mean() - 1e-4
        if (cos >= 0).all():
            assert rel.mean() >= (1 - cos.mean()) - 1e-4


@pytest.mark.parametrize("encoder_type", ["kan", "linear"])
def test_check_params_reconciles_tiny_model(tmp_path, encoder_type):
    model = KANCrossLayerTranscoder(
        n_layers=2,
        d_transcoder=16,
        d_model=8,
        encoder_type=encoder_type,
        grid_size=5,
        spline_order=3,
        activation_function="jump_relu",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    ckpt = tmp_path / "ckpt"
    model.to_safetensors(str(ckpt))
    result = check_params.inspect_checkpoint(ckpt)

    assert result["meta"]["encoder_type"] == encoder_type
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert result["actual_total_params"] == trainable
    if encoder_type == "kan":
        assert result["spline_scaler_enabled"]
        assert result["reconciles_with"] == "3-term"
    else:
        assert result["reconciles_with"] in ("3-term", "2-term")
    assert not result["unrecognized_keys"]


def test_check_reconstruction_end_to_end(tmp_path):
    """Full script run on a tiny model + synthetic activation dataset."""
    torch.manual_seed(0)
    n_samples, n_layers, n_pos, d_model = 6, 2, 4, 8
    model = KANCrossLayerTranscoder(
        n_layers=n_layers,
        d_transcoder=16,
        d_model=d_model,
        encoder_type="linear",
        activation_function="jump_relu",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    ckpt = tmp_path / "ckpt"
    model.to_safetensors(str(ckpt))

    data_dir = tmp_path / "acts"
    data_dir.mkdir()
    shape = (n_samples, n_layers, n_pos, d_model)
    for name in ("mlp_inputs_val", "mlp_outputs_val"):
        arr = torch.randn(shape, dtype=torch.bfloat16).view(torch.int16).numpy()
        np.save(data_dir / f"{name}.npy", arr)

    out_dir = tmp_path / "out"
    exit_code = check_reconstruction.main(
        [
            "--checkpoint", str(ckpt),
            "--activation-dir", str(data_dir),
            "--out-dir", str(out_dir),
            "--n-samples", "4",
            "--device", "cpu",
            "--label", "tiny",
        ]
    )
    assert exit_code == 0
    result_json = out_dir / "check_reconstruction_tiny.json"
    assert result_json.exists()
    assert (out_dir / "check_reconstruction_tiny.md").exists()
    assert (out_dir / "provenance.csv").exists()

    import json

    payload = json.loads(result_json.read_text())
    assert payload["status"] == "PASS"
    predictors = payload["section_4_3"]["predictors"]
    # Zero predictor: rel err exactly 1, cosine 0, by definition.
    assert predictors["zero"]["rel_err_global_frobenius"] == pytest.approx(1.0, abs=1e-5)
    assert predictors["zero"]["cos_mean"] == pytest.approx(0.0, abs=1e-5)
    # Mean predictor explains no variance relative to itself.
    assert predictors["dataset_mean"]["variance_explained_aggregate"] == pytest.approx(
        0.0, abs=1e-5
    )
    # The §4.1 inequality holds on every predictor's own stats.
    for row in predictors.values():
        assert row["rel_err_per_position_mean"] >= (1 - row["cos_mean"]) - 1e-3
