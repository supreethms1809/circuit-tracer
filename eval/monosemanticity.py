"""Feature monosemanticity evaluation for KAN-CLT (Phase 5.1).

For each of the top N features (by activation frequency), collects the
max-activating examples from the dataset and computes basic monosemanticity
metrics:
  - Activation distribution: how heavy-tailed is each feature?
  - Token-level max-activating examples (for manual inspection)
  - Concept coherence score (if a language model scorer is available)

Usage:
    from eval.monosemanticity import collect_max_activating_examples
    examples = collect_max_activating_examples(model, dataset, top_n_features=200)
"""

import os
import sys
import json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from kan_clt.kan_transcoder import KANCrossLayerTranscoder
from kan_clt.training.data import ActivationDataset


@dataclass
class FeatureExample:
    """A single max-activating example for a feature."""
    layer: int
    feature_id: int
    activation_value: float
    sample_idx: int
    position: int
    # Context window (token indices if available, else position markers)
    context_positions: list[int] = field(default_factory=list)


@dataclass
class FeatureReport:
    """Summary report for one feature's monosemanticity."""
    layer: int
    feature_id: int
    activation_frequency: float
    mean_activation: float
    max_activation: float
    # Gini coefficient of activation distribution (0=uniform, 1=maximally sparse)
    gini_coefficient: float
    # Top-K max-activating examples
    top_examples: list[FeatureExample] = field(default_factory=list)


def gini_coefficient(values: torch.Tensor) -> float:
    """Compute Gini coefficient of a non-negative distribution.

    Gini → 1 means a single example accounts for all activation (maximally
    monosemantic), Gini → 0 means uniformly distributed activation.

    Args:
        values: 1D tensor of non-negative activation values.

    Returns:
        Gini coefficient in [0, 1].
    """
    values = values.float().abs()
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = values.sort().values
    indices = torch.arange(1, n + 1, dtype=torch.float32)
    return float((2 * (indices * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)


@torch.no_grad()
def collect_max_activating_examples(
    model: KANCrossLayerTranscoder,
    dataset: ActivationDataset,
    top_n_features: int = 200,
    top_k_examples: int = 10,
    n_samples: int = 500,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> list[FeatureReport]:
    """Collect max-activating examples for the most active features.

    Args:
        model: Trained KAN-CLT or linear CLT.
        dataset: Activation dataset.
        top_n_features: Number of top features (by activation frequency) to analyze.
        top_k_examples: Number of max-activating examples to keep per feature.
        n_samples: Number of dataset samples to scan.
        device: Compute device.
        dtype: Data dtype.

    Returns:
        List of FeatureReport, one per top feature, sorted by activation frequency.
    """
    if device is None:
        device = next(model.parameters()).device

    n_samples = min(n_samples, len(dataset))
    n_layers = model.n_layers
    d_transcoder = model.d_transcoder

    # Pass 1: compute activation frequency per feature
    print(f"Pass 1: computing activation frequencies over {n_samples} samples...")
    sum_freq = torch.zeros(n_layers, d_transcoder)
    sum_act = torch.zeros(n_layers, d_transcoder)
    max_act = torch.zeros(n_layers, d_transcoder)
    n_pos_total = 0

    indices = torch.randperm(len(dataset))[:n_samples].tolist()
    for idx in tqdm(indices, desc="Frequency scan"):
        sample = dataset[idx]
        x_in = sample["mlp_inputs"].to(device=device, dtype=dtype)
        acts = model.encode(x_in).float().cpu()   # (n_layers, seq, d_transcoder)
        n_pos = acts.shape[1]
        sum_freq += (acts > 0).float().sum(dim=1)
        sum_act += acts.sum(dim=1)
        max_act = torch.maximum(max_act, acts.max(dim=1).values)
        n_pos_total += n_pos

    freq = sum_freq / n_pos_total
    mean = sum_act / n_pos_total

    # Find top features by activation frequency
    flat_freq = freq.reshape(-1)
    top_flat = flat_freq.topk(top_n_features).indices
    top_layer_feat = [(int(i // d_transcoder), int(i % d_transcoder)) for i in top_flat]

    print(f"Pass 2: collecting top-{top_k_examples} max-activating examples for each feature...")

    # Per feature: keep a running heap of (activation, sample_idx, pos)
    # Stored as list of (neg_act, sample_idx, position) sorted ascending (smallest last)
    feature_heaps: dict[tuple[int, int], list[tuple[float, int, int]]] = {
        lf: [] for lf in top_layer_feat
    }
    # All activations across all samples for Gini computation (list of tensors)
    feature_all_acts: dict[tuple[int, int], list[float]] = {
        lf: [] for lf in top_layer_feat
    }

    for idx in tqdm(indices, desc="Example collection"):
        sample = dataset[idx]
        x_in = sample["mlp_inputs"].to(device=device, dtype=dtype)
        acts = model.encode(x_in).float().cpu()   # (n_layers, seq, d_transcoder)

        for (l, f) in top_layer_feat:
            feat_acts = acts[l, :, f]  # (seq_len,)
            # Collect all nonzero activations for Gini
            nonzero = feat_acts[feat_acts > 0]
            feature_all_acts[(l, f)].extend(nonzero.tolist())

            # Track top-k examples
            heap = feature_heaps[(l, f)]
            for pos in range(feat_acts.shape[0]):
                val = feat_acts[pos].item()
                if val <= 0:
                    continue
                if len(heap) < top_k_examples:
                    heap.append((val, idx, pos))
                    heap.sort(key=lambda x: x[0])
                elif val > heap[0][0]:
                    heap[0] = (val, idx, pos)
                    heap.sort(key=lambda x: x[0])

    # Build reports
    reports = []
    for (l, f) in top_layer_feat:
        heap = sorted(feature_heaps[(l, f)], key=lambda x: -x[0])
        all_vals = torch.tensor(feature_all_acts[(l, f)]) if feature_all_acts[(l, f)] else torch.zeros(1)

        examples = [
            FeatureExample(
                layer=l,
                feature_id=f,
                activation_value=val,
                sample_idx=sidx,
                position=pos,
            )
            for (val, sidx, pos) in heap
        ]

        reports.append(FeatureReport(
            layer=l,
            feature_id=f,
            activation_frequency=freq[l, f].item(),
            mean_activation=mean[l, f].item(),
            max_activation=max_act[l, f].item(),
            gini_coefficient=gini_coefficient(all_vals),
            top_examples=examples,
        ))

    reports.sort(key=lambda r: -r.activation_frequency)
    return reports


def save_reports(reports: list[FeatureReport], path: str) -> None:
    """Save feature reports to JSON for downstream analysis or visualization."""
    data = []
    for r in reports:
        data.append({
            "layer": r.layer,
            "feature_id": r.feature_id,
            "activation_frequency": r.activation_frequency,
            "mean_activation": r.mean_activation,
            "max_activation": r.max_activation,
            "gini_coefficient": r.gini_coefficient,
            "top_examples": [
                {
                    "activation_value": e.activation_value,
                    "sample_idx": e.sample_idx,
                    "position": e.position,
                }
                for e in r.top_examples
            ],
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(reports)} feature reports to {path}")


def print_summary(reports: list[FeatureReport], top_n: int = 20) -> None:
    """Print a human-readable summary of the top features."""
    print(f"\n{'='*70}")
    print(f"Feature monosemanticity summary — top {min(top_n, len(reports))} features")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Layer':<6} {'Feat':<6} {'Freq':>8} {'Mean':>8} {'Max':>8} {'Gini':>8}")
    print("-" * 60)
    for i, r in enumerate(reports[:top_n]):
        print(f"{i+1:<5} {r.layer:<6} {r.feature_id:<6} "
              f"{r.activation_frequency:>8.4f} {r.mean_activation:>8.4f} "
              f"{r.max_activation:>8.4f} {r.gini_coefficient:>8.4f}")

    # Summary statistics
    ginis = [r.gini_coefficient for r in reports]
    freqs = [r.activation_frequency for r in reports]
    print(f"\n{'Average Gini (top {len(reports)} features):':<40} {sum(ginis)/len(ginis):.4f}")
    print(f"{'Average activation frequency:':<40} {sum(freqs)/len(freqs):.4f}")
    high_gini = sum(1 for g in ginis if g > 0.8)
    print(f"{'Features with Gini > 0.8 (highly specific):':<40} {high_gini}/{len(reports)}")
