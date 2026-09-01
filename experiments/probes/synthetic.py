"""Synthetic residual dataset with known nonlinear features.

Each sample: x ~ N(0,I)^{n_layers × seq × d}. Ground-truth MLP outputs are
produced by a sparse set of nonlinear feature functions of x (products and
radial bumps in a subspace). Linear encoders cannot represent these exactly;
a working KAN path should recover them better under matched capacity.
"""

from __future__ import annotations

import torch

from spline_clt.training.data import ActivationDataset


def make_synthetic_nonlinear_dataset(
    *,
    n_samples: int = 1024,
    n_layers: int = 4,
    seq_len: int = 16,
    d_model: int = 64,
    n_features: int = 8,
    noise: float = 0.05,
    seed: int = 0,
) -> ActivationDataset:
    """Build in-memory ActivationDataset with nonlinear ground truth.

    Feature definitions (applied per layer independently on each position):
      - product gates: x[..., 2i] * x[..., 2i+1]
      - radial bumps: exp(-||x_sub||^2) on a 2D subspace
    Decoder directions are random orthonormal columns; outputs are sum_f a_f w_f.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, n_layers, seq_len, d_model, generator=g)

    # Random unit decoder directions (n_features, d_model)
    w = torch.randn(n_features, d_model, generator=g)
    w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    acts = []
    for f in range(n_features):
        if f % 2 == 0:
            i = (2 * f) % (d_model - 1)
            a = x[..., i] * x[..., i + 1]
        else:
            i = (2 * f) % (d_model - 2)
            sub = x[..., i : i + 2]
            a = torch.exp(-sub.pow(2).sum(dim=-1))
        acts.append(a)
    # (n_features, n_samples, n_layers, seq)
    a_stack = torch.stack(acts, dim=0)
    # y[..., d] = sum_f a[f] * w[f, d]
    y = torch.einsum("fnls,fd->nlsd", a_stack, w)
    y = y + noise * torch.randn(y.shape, generator=g)

    return ActivationDataset(x.clone(), y.clone())


def feature_recovery_score(
    model,
    dataset: ActivationDataset,
    *,
    device: torch.device,
    max_samples: int = 128,
) -> dict[str, float]:
    """Crude recovery diagnostic: how well ŷ matches y vs a linear baseline proxy.

    Reports rel_fro and whether nonlinear structure is present in residuals of a
    per-layer linear least-squares fit of y on x (should be large on this toy).
    """
    from spline_clt.training.loss import paper_style_reconstruction_sparsity_metrics

    n = min(max_samples, len(dataset))
    xs, ys, yhats = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            x = sample["mlp_inputs"].to(device=device, dtype=torch.float32)
            y = sample["mlp_outputs"].to(device=device, dtype=torch.float32)
            acts = model.encode(x)
            y_hat = model.decode_dense(acts, input_acts=x)
            xs.append(x.cpu())
            ys.append(y.cpu())
            yhats.append(y_hat.cpu())
    x_all = torch.stack(xs, dim=0)
    y_all = torch.stack(ys, dim=0)
    yh_all = torch.stack(yhats, dim=0)

    # Flatten for metrics: fake activations of ones for sparsity stats unused
    dummy_a = torch.ones(y_all.shape[1], y_all.shape[0] * y_all.shape[2], 1)
    # Use mean over samples of per-sample recon metrics
    rel = []
    for i in range(n):
        y = y_all[i]
        yh = yh_all[i]
        metrics = paper_style_reconstruction_sparsity_metrics(
            yh, y, torch.zeros(y.shape[0], y.shape[1], 8)
        )
        rel.append(metrics["reconstruction/rel_fro_error"])

    # Nonlinear residual of OLS y ~ x per layer (fraction of variance unexplained by linear map)
    unexplained = []
    for layer in range(y_all.shape[1]):
        X = x_all[:, layer].reshape(-1, x_all.shape[-1]).double()
        Y = y_all[:, layer].reshape(-1, y_all.shape[-1]).double()
        # least squares
        sol = torch.linalg.lstsq(X, Y).solution
        pred = X @ sol
        ss_res = (Y - pred).pow(2).sum()
        ss_tot = (Y - Y.mean(dim=0)).pow(2).sum().clamp_min(1e-8)
        unexplained.append(float((ss_res / ss_tot).item()))

    return {
        "rel_fro_mean": float(sum(rel) / max(1, len(rel))),
        "linear_unexplained_frac_mean": float(sum(unexplained) / max(1, len(unexplained))),
        "linear_unexplained_frac_per_layer": unexplained,
    }
