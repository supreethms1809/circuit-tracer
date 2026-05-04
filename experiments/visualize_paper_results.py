#!/usr/bin/env python3
"""Create Plotly visualizations for config-driven paper run outputs.

The script reads artifacts produced under ``results/paper/<suite_name>``:

- ``runs/*/seed_*/evaluation/evaluation_summary.json``
- ``runs/*/seed_*/evaluation/prompt_metrics.jsonl``
- ``runs/*/seed_*/evaluation/monosemanticity_records.jsonl``
- ``runs/*/seed_*/evaluation/splines/*.csv``
- ``runs/*/seed_*/training_records.jsonl``

It writes standalone HTML figures plus an ``index.html`` and manifest. Missing
artifacts are skipped so it can be run while a paper suite is still in progress.
Use ``--seed-count`` to compare the first N available seeds per variant; variants
with fewer seeds are plotted with the available seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit(
        "Plotly is required for this script. Install the project dependencies or run: "
        "python -m pip install plotly"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]


METRIC_GROUPS: list[tuple[str, list[tuple[str, str, bool, bool]]]] = [
    (
        "Reconstruction",
        [
            ("MSE Total", "reconstruction.mse_total", False, True),
            ("Cosine Similarity", "reconstruction.cosine_similarity", True, False),
            ("Relative Error", "reconstruction.relative_error", False, False),
        ],
    ),
    (
        "Sparsity",
        [
            ("Active Features / Position", "reconstruction.active_per_pos", False, True),
            ("Activation Density", "reconstruction.activation_density", False, True),
        ],
    ),
    (
        "Replacement Fidelity",
        [
            ("Top-1 Match Rate", "prompt_metrics.top1_match_rate", True, False),
            ("KL Divergence", "prompt_metrics.kl_divergence", False, False),
        ],
    ),
    (
        "Monosemanticity",
        [
            ("Mean Gini", "monosemanticity.mean_gini", True, False),
            ("Median Gini", "monosemanticity.median_gini", True, False),
            ("Fraction Gini > 0.7", "monosemanticity.fraction_gini_gt_0_7", True, False),
        ],
    ),
    (
        "Spline Nonlinearity",
        [
            ("Mean Nonlinearity Score", "splines.mean_nonlinearity_score", True, False),
            ("Median Nonlinearity Score", "splines.median_nonlinearity_score", True, False),
            ("Fraction Nonlinear > 0.05", "splines.fraction_nonlinear_gt_0_05", True, False),
        ],
    ),
    (
        "Circuit Faithfulness",
        [
            ("Keep-Only Gap Ratio", "circuit_metrics.keep_only_gap_ratio", True, False),
            ("Gap Drop Ratio", "circuit_metrics.gap_drop_ratio", True, False),
            ("Graph Completeness", "circuit_metrics.graph_completeness_score", True, False),
            ("Retained Feature Nodes", "circuit_metrics.retained_feature_node_count", False, False),
            ("Retained Error Nodes", "circuit_metrics.retained_error_node_count", False, False),
        ],
    ),
]

PROMPT_METRICS = [
    ("Top-1 Match Rate", "top1_match_rate"),
    ("KL Divergence", "kl_divergence"),
    ("Keep-Only Gap Ratio", "keep_only_gap_ratio"),
    ("Graph Completeness", "graph_completeness_score"),
]

TRAINING_METRICS = [
    ("Total Loss", "loss/total", True),
    ("Reconstruction Loss", "loss/reconstruction", True),
    ("Active Features / Position", "stats/active_features_per_pos", True),
    ("Learning Rate", "lr", False),
]

COLORWAY = [
    "#1f77b4",
    "#d62728",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#7f7f7f",
]


# Camera-ready typography. Sizes target 1700px-wide figures rendered to PNG
# at scale=3, so on-page they read as ~10–14pt at NeurIPS column width.
FONT_FAMILY = "Helvetica, Arial, sans-serif"
TITLE_FONT = {"family": FONT_FAMILY, "size": 26, "color": "#111"}
AXIS_TITLE_FONT = {"family": FONT_FAMILY, "size": 18, "color": "#222"}
TICK_FONT = {"family": FONT_FAMILY, "size": 14, "color": "#222"}
LEGEND_FONT = {"family": FONT_FAMILY, "size": 14, "color": "#111"}
SUBPLOT_TITLE_FONT = {"family": FONT_FAMILY, "size": 18, "color": "#111"}
ANNOTATION_FONT = {"family": FONT_FAMILY, "size": 13, "color": "#444"}
VALUE_LABEL_FONT = {"family": FONT_FAMILY, "size": 13, "color": "#111"}
EXPORT_SCALE = 3  # PNG export resolution multiplier


def overlay_legend(position: str = "top-right") -> dict[str, Any]:
    """Legend overlaid inside the plotting area, anchored to a corner."""
    anchors = {
        "top-right": {"x": 0.99, "xanchor": "right", "y": 0.99, "yanchor": "top"},
        "top-left": {"x": 0.01, "xanchor": "left", "y": 0.99, "yanchor": "top"},
        "bottom-right": {"x": 0.99, "xanchor": "right", "y": 0.01, "yanchor": "bottom"},
        "bottom-left": {"x": 0.01, "xanchor": "left", "y": 0.01, "yanchor": "bottom"},
    }
    anchor = anchors.get(position, anchors["top-right"])
    return {
        "legend": {
            "orientation": "v",
            **anchor,
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(0,0,0,0.18)",
            "borderwidth": 1,
            "font": LEGEND_FONT,
            "tracegroupgap": 4,
        },
        "margin": {"l": 90, "r": 30, "t": 90, "b": 90},
    }


def apply_camera_ready(
    fig: "go.Figure",
    *,
    title: str | None = None,
    legend_position: str = "top-right",
    show_legend: bool = True,
    extra: dict[str, Any] | None = None,
) -> "go.Figure":
    """Apply consistent camera-ready styling: fonts, centered title, overlay legend."""
    layout: dict[str, Any] = {
        "template": "plotly_white",
        "font": {"family": FONT_FAMILY, "size": 14, "color": "#222"},
        "title": {
            "text": title or "",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
            "yanchor": "top",
            "font": TITLE_FONT,
        },
        "showlegend": show_legend,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
    }
    layout.update(overlay_legend(legend_position))
    if extra:
        layout.update(extra)
    fig.update_layout(**layout)
    # Subplot titles (annotations) get a bigger consistent font.
    if hasattr(fig.layout, "annotations"):
        for ann in fig.layout.annotations:
            if ann.text and not ann.bgcolor:  # skip the corner direction labels we add elsewhere
                ann.font = SUBPLOT_TITLE_FONT
    fig.update_xaxes(
        title_font=AXIS_TITLE_FONT,
        tickfont=TICK_FONT,
        showline=True,
        linewidth=1,
        linecolor="#444",
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        automargin=True,
    )
    fig.update_yaxes(
        title_font=AXIS_TITLE_FONT,
        tickfont=TICK_FONT,
        showline=True,
        linewidth=1,
        linecolor="#444",
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        automargin=True,
    )
    return fig


def fmt_value(v: float) -> str:
    """Compact human-readable value label."""
    if v != v:  # NaN
        return ""
    av = abs(v)
    if av >= 1000:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.2f}"
    if av >= 0.1:
        return f"{v:.3f}"
    if av >= 0.001:
        return f"{v:.4f}"
    if av == 0:
        return "0"
    return f"{v:.2e}"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb color to a Plotly rgba() string."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(31, 119, 180, {alpha})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "figure"


@dataclass(frozen=True)
class SummaryRecord:
    variant_name: str
    variant_label: str
    seed: int
    path: Path
    summary: dict[str, Any]


RunKey = tuple[str, int]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1)))


def metric_value(summary: dict[str, Any], metric_path: str) -> float | None:
    value: Any = summary
    for part in metric_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return safe_float(value)


def compact_label(label: str) -> str:
    replacements = {
        "Spline-CLT": "Spline",
        "Linear CLT": "Linear",
        "feature-match": "feat-match",
        "param-match": "param-match",
    }
    result = label
    for old, new in replacements.items():
        result = result.replace(old, new)
    result = re.sub(r",\s*d_t=", ", d=", result)
    result = result.replace(" to ", " vs ")
    return result


def variant_short_label(name: str, label: str, *, multiline: bool = False) -> str:
    """Readable short label for axes/legends; full labels stay in hover text."""
    name_text = name.lower()
    label_text = label.lower()
    text = f"{label_text} {name_text}"
    # Prefer the variant key for encoder family because labels like
    # "Linear ... param-match to spline@..." legitimately mention the other
    # family in the matching target.
    if "linear" in name_text:
        encoder = "Linear"
    elif "spline" in name_text or "kan" in name_text:
        encoder = "Spline"
    elif "linear" in label_text:
        encoder = "Linear"
    elif "spline" in label_text or "kan" in label_text:
        encoder = "Spline"
    else:
        encoder = name

    if "feature" in text or "feat" in text:
        match = "FM"
    elif "param" in text:
        match = "PM"
    else:
        match = ""

    d_transcoder = extract_d_transcoder(name, label)
    pieces = [encoder]
    if match:
        pieces.append(match)
    if d_transcoder is not None:
        pieces.append(f"d={d_transcoder}")
    separator = "<br>" if multiline else " "
    return separator.join(pieces)


def variant_axis_labels(records: list[SummaryRecord], variants: list[str]) -> list[str]:
    return [
        variant_short_label(variant, label_for_variant(records, variant), multiline=True)
        for variant in variants
    ]


def variant_legend_labels(records: list[SummaryRecord], variants: list[str]) -> dict[str, str]:
    return {
        variant: variant_short_label(variant, label_for_variant(records, variant), multiline=False)
        for variant in variants
    }


def suite_root_from_args(args: argparse.Namespace) -> Path:
    explicit_results_dir = args.results_dir or args.results_dir_arg
    if explicit_results_dir is not None:
        return Path(explicit_results_dir).expanduser().resolve()
    if args.suite_root is not None:
        return Path(args.suite_root).expanduser().resolve()
    return (Path(args.results_root).expanduser().resolve() / args.suite)


def load_variant_labels(suite_root: Path) -> dict[str, str]:
    config_path = suite_root / "resolved_config.json"
    if not config_path.exists():
        return {}
    config = read_json(config_path)
    variants = config.get("model_variants", {})
    if not isinstance(variants, dict):
        return {}
    labels: dict[str, str] = {}
    for variant_name, payload in variants.items():
        if isinstance(payload, dict) and isinstance(payload.get("label"), str):
            labels[str(variant_name)] = payload["label"]
    return labels


def discover_summaries(suite_root: Path) -> list[SummaryRecord]:
    label_map = load_variant_labels(suite_root)
    summaries: list[SummaryRecord] = []
    for path in sorted(suite_root.glob("runs/*/seed_*/evaluation/evaluation_summary.json")):
        try:
            summary = read_json(path)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping invalid JSON: {path} ({exc})", file=sys.stderr)
            continue

        variant_name = str(summary.get("variant_name") or path.parents[2].name)
        seed_raw = summary.get("seed")
        if seed_raw is None:
            seed_match = re.match(r"seed_(\d+)", path.parents[1].name)
            seed = int(seed_match.group(1)) if seed_match else -1
        else:
            seed = int(seed_raw)

        variant_label = label_map.get(variant_name, variant_name)
        summaries.append(
            SummaryRecord(
                variant_name=variant_name,
                variant_label=variant_label,
                seed=seed,
                path=path,
                summary=summary,
            )
        )
    return summaries


def select_seed_count(records: list[SummaryRecord], seed_count: int | None) -> list[SummaryRecord]:
    """Keep up to seed_count seeds per variant, preserving sorted seed order."""
    if seed_count is None:
        return sorted(records, key=lambda record: (record.variant_name, record.seed))
    if seed_count < 1:
        raise SystemExit("--seed-count must be >= 1")

    selected: list[SummaryRecord] = []
    for variant, variant_records in records_by_variant(records).items():
        kept = variant_records[:seed_count]
        selected.extend(kept)
        if len(variant_records) < seed_count:
            print(
                f"[WARN] {variant}: requested {seed_count} seeds but only "
                f"{len(variant_records)} found; using available seeds."
            )
    return sorted(selected, key=lambda record: (record.variant_name, record.seed))


def allowed_runs(records: list[SummaryRecord]) -> set[RunKey]:
    return {(record.variant_name, record.seed) for record in records}


def record_run_key(record: dict[str, Any]) -> RunKey | None:
    variant = record.get("variant_name")
    seed = record.get("seed")
    if variant is None or seed is None:
        return None
    try:
        return str(variant), int(seed)
    except (TypeError, ValueError):
        return None


def seed_from_dir(path: Path) -> int | None:
    match = re.match(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else None


def file_run_key(path: Path) -> RunKey | None:
    """Extract (variant, seed) from files under runs/<variant>/seed_<n>/..."""
    try:
        seed = seed_from_dir(next(parent for parent in path.parents if parent.name.startswith("seed_")))
    except StopIteration:
        return None
    if seed is None:
        return None

    for parent in path.parents:
        if parent.parent.name == "runs":
            return parent.name, seed
    return None


def suites_with_summaries(results_root: Path) -> list[str]:
    suites: list[str] = []
    if not results_root.exists():
        return suites
    for child in sorted(path for path in results_root.iterdir() if path.is_dir()):
        if any(child.glob("runs/*/seed_*/evaluation/evaluation_summary.json")):
            suites.append(child.name)
    return suites


def extract_d_transcoder(name: str, label: str) -> int | None:
    """Extract d_t/d_transcoder-style widths for trend-friendly sorting."""
    candidates = [label, name]
    patterns = [
        r"d_t\s*=\s*(\d+)",
        r"d_transcoder\s*=\s*(\d+)",
        r"(?:^|_)dt(\d+)(?:_|$)",
    ]
    for candidate in candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate)
            if match:
                return int(match.group(1))
    return None


def encoder_sort_group(text: str) -> int:
    lowered = text.lower()
    if "spline" in lowered or "kan" in lowered:
        return 0
    if "linear" in lowered:
        return 1
    return 2


def match_sort_group(text: str) -> int:
    lowered = text.lower()
    if "feature" in lowered or "feat" in lowered:
        return 0
    if "param" in lowered:
        return 1
    return 2


def variant_order(records: list[SummaryRecord]) -> list[str]:
    names = sorted({record.variant_name for record in records})

    def sort_key(name: str) -> tuple[int, int, int, int, str]:
        label = next(record.variant_label for record in records if record.variant_name == name)
        sort_text = f"{label} {name}"
        d_transcoder = extract_d_transcoder(name, label)
        if d_transcoder is not None:
            return (
                0,
                d_transcoder,
                encoder_sort_group(sort_text),
                match_sort_group(sort_text),
                label.lower(),
            )
        return (
            1,
            encoder_sort_group(sort_text),
            match_sort_group(sort_text),
            0,
            label.lower(),
        )

    return sorted(names, key=sort_key)


def label_for_variant(records: list[SummaryRecord], variant_name: str) -> str:
    return next(record.variant_label for record in records if record.variant_name == variant_name)


def colors_for_variants(variants: list[str]) -> dict[str, str]:
    return {variant: COLORWAY[index % len(COLORWAY)] for index, variant in enumerate(variants)}


def records_by_variant(records: list[SummaryRecord]) -> dict[str, list[SummaryRecord]]:
    grouped: dict[str, list[SummaryRecord]] = {}
    for record in records:
        grouped.setdefault(record.variant_name, []).append(record)
    for group in grouped.values():
        group.sort(key=lambda record: record.seed)
    return grouped


def write_figure(
    fig: go.Figure,
    output_dir: Path,
    filename: str,
    *,
    write_png: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": path.stem,
            "scale": EXPORT_SCALE,
        },
    }
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, config=config)
    if write_png:
        png_path = path.with_suffix(".png")
        try:
            fig.write_image(
                png_path,
                scale=EXPORT_SCALE,
                width=fig.layout.width,
                height=fig.layout.height,
            )
        except Exception as exc:  # kaleido may be missing
            print(f"  [warn] PNG export failed for {png_path.name}: {exc}")
    return path


def set_readable_xaxis(fig: go.Figure, *, row: int | None = None, col: int | None = None) -> None:
    fig.update_xaxes(
        tickangle=0,
        automargin=True,
        title_standoff=20,
        tickfont={"size": 10},
        row=row,
        col=col,
    )


def set_readable_yaxis(fig: go.Figure, *, row: int | None = None, col: int | None = None) -> None:
    fig.update_yaxes(
        automargin=True,
        title_standoff=18,
        tickfont={"size": 10},
        row=row,
        col=col,
    )


def all_metric_specs() -> list[tuple[str, str, str, bool, bool]]:
    return [
        (group_name, title, key, higher_is_better, log_y)
        for group_name, group in METRIC_GROUPS
        for title, key, higher_is_better, log_y in group
    ]


def build_metric_overview(records: list[SummaryRecord]) -> go.Figure | None:
    if not records:
        return None

    metrics = [(title, key, higher_is_better, log_y) for _, title, key, higher_is_better, log_y in all_metric_specs()]
    variants = variant_order(records)
    grouped = records_by_variant(records)
    labels = variant_axis_labels(records, variants)
    colors = colors_for_variants(variants)
    cols = 3
    rows = math.ceil(len(metrics) / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[title for title, _, _, _ in metrics],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    for metric_index, (title, key, higher_is_better, log_y) in enumerate(metrics):
        row = metric_index // cols + 1
        col = metric_index % cols + 1
        means: list[float] = []
        stds: list[float] = []
        seed_x: list[str] = []
        seed_y: list[float] = []
        hover: list[str] = []

        for variant, label in zip(variants, labels):
            values: list[float] = []
            for record in grouped.get(variant, []):
                value = metric_value(record.summary, key)
                if value is None:
                    continue
                values.append(value)
                seed_x.append(label)
                seed_y.append(value)
                hover.append(f"{record.variant_label}<br>seed={record.seed}<br>{title}={value:.6g}")
            means.append(mean(values) if values else float("nan"))
            stds.append(sample_std(values))

        fig.add_trace(
            go.Bar(
                x=labels,
                y=means,
                error_y={"type": "data", "array": stds, "visible": True, "thickness": 1.5},
                marker_color=[colors[variant] for variant in variants],
                marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=0.8,
                name=title,
                text=[fmt_value(v) for v in means],
                textposition="outside",
                textfont=VALUE_LABEL_FONT,
                cliponaxis=False,
                showlegend=False,
                hovertemplate="%{x}<br>mean=%{y:.6g}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=seed_x,
                y=seed_y,
                mode="markers",
                marker={"color": "rgba(0,0,0,0.7)", "size": 7, "line": {"color": "white", "width": 1}},
                text=hover,
                name="Seeds",
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        if log_y:
            fig.update_yaxes(type="log", row=row, col=col)
        direction = "↑" if higher_is_better else "↓"
        fig.add_annotation(
            text=direction,
            xref=f"x{metric_index + 1 if metric_index > 0 else ''} domain",
            yref=f"y{metric_index + 1 if metric_index > 0 else ''} domain",
            x=0.97,
            y=0.97,
            showarrow=False,
            font={"family": FONT_FAMILY, "size": 22, "color": "rgba(40,80,40,0.55)" if higher_is_better else "rgba(120,40,40,0.55)"},
        )

    apply_camera_ready(
        fig,
        title="Paper Run Metric Overview  (mean ± seed std; dots = seeds)",
        show_legend=False,
        extra={"height": max(950, rows * 360), "width": 1750, "bargap": 0.28},
    )
    return fig


def build_single_metric_figure(
    records: list[SummaryRecord],
    *,
    group_name: str,
    title: str,
    key: str,
    higher_is_better: bool,
    log_y: bool,
) -> go.Figure | None:
    variants = variant_order(records)
    grouped = records_by_variant(records)
    labels = variant_axis_labels(records, variants)
    legend_labels = variant_legend_labels(records, variants)
    colors = colors_for_variants(variants)

    means: list[float] = []
    stds: list[float] = []
    seed_x: list[str] = []
    seed_y: list[float] = []
    hover: list[str] = []

    for variant, label in zip(variants, labels):
        values: list[float] = []
        for record in grouped.get(variant, []):
            value = metric_value(record.summary, key)
            if value is None:
                continue
            values.append(value)
            seed_x.append(label)
            seed_y.append(value)
            hover.append(
                f"{record.variant_label}<br>"
                f"seed={record.seed}<br>"
                f"{title}={value:.6g}"
            )
        means.append(mean(values) if values else float("nan"))
        stds.append(sample_std(values))

    if not any(math.isfinite(value) for value in means):
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=means,
            error_y={"type": "data", "array": stds, "visible": True, "thickness": 1.5},
            marker_color=[colors[variant] for variant in variants],
            marker_line_color="rgba(0,0,0,0.3)",
            marker_line_width=0.8,
            text=[fmt_value(v) for v in means],
            textposition="outside",
            textfont=VALUE_LABEL_FONT,
            cliponaxis=False,
            hovertemplate="%{x}<br>mean=%{y:.6g}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=seed_x,
            y=seed_y,
            mode="markers",
            marker={"color": "rgba(0,0,0,0.75)", "size": 10, "line": {"color": "white", "width": 1}},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    if log_y:
        fig.update_yaxes(type="log")
    direction = "↑ higher is better" if higher_is_better else "↓ lower is better"
    fig.add_annotation(
        text=direction,
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.99,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 14, "color": "#555"},
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1,
        borderpad=4,
    )
    fig.update_xaxes(title_text="Variant")
    fig.update_yaxes(title_text=title + (" (log scale)" if log_y else ""))
    apply_camera_ready(
        fig,
        title=f"{group_name}: {title}",
        show_legend=False,
        extra={"height": 720, "width": 1200, "bargap": 0.35},
    )
    return fig


def build_per_layer_mse(records: list[SummaryRecord]) -> go.Figure | None:
    variants = variant_order(records)
    grouped = records_by_variant(records)
    colors = colors_for_variants(variants)
    fig = go.Figure()
    added = False

    for variant in variants:
        layer_values = [
            record.summary.get("reconstruction", {}).get("mse_per_layer")
            for record in grouped.get(variant, [])
        ]
        layer_values = [
            values
            for values in layer_values
            if isinstance(values, list) and all(safe_float(value) is not None for value in values)
        ]
        if not layer_values:
            continue

        n_layers = min(len(values) for values in layer_values)
        per_layer = [[float(values[layer]) for values in layer_values] for layer in range(n_layers)]
        xs = list(range(n_layers))
        means = [mean(values) for values in per_layer]
        stds = [sample_std(values) for values in per_layer]
        upper = [mu + std for mu, std in zip(means, stds)]
        lower = [max(mu - std, 1e-12) for mu, std in zip(means, stds)]
        label = variant_short_label(variant, label_for_variant(records, variant))
        color = colors[variant]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=upper,
                mode="lines",
                line={"width": 0, "color": color},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=lower,
                mode="lines",
                line={"width": 0, "color": color},
                fill="tonexty",
                fillcolor=hex_to_rgba(color, 0.16),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"size": 7},
                name=label,
                hovertemplate="layer=%{x}<br>MSE=%{y:.6g}<extra></extra>",
            )
        )
        added = True

    if not added:
        return None

    fig.update_xaxes(title_text="Transformer Layer")
    fig.update_yaxes(title_text="MSE (log scale)", type="log")
    apply_camera_ready(
        fig,
        title="Per-Layer Reconstruction MSE",
        legend_position="top-right",
        extra={"height": 900, "width": 1600},  # 16:9
    )
    return fig


def build_reconstruction_scatter(records: list[SummaryRecord]) -> go.Figure | None:
    variants = variant_order(records)
    grouped = records_by_variant(records)
    colors = colors_for_variants(variants)
    legend_labels = variant_legend_labels(records, variants)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["MSE vs Mean Gini", "Cosine Similarity vs Mean Gini"],
        horizontal_spacing=0.1,
    )
    added = False

    for variant in variants:
        label = legend_labels[variant]
        color = colors[variant]
        xs_mse: list[float] = []
        xs_cos: list[float] = []
        ys: list[float] = []
        text: list[str] = []
        for record in grouped.get(variant, []):
            mse = metric_value(record.summary, "reconstruction.mse_total")
            cos = metric_value(record.summary, "reconstruction.cosine_similarity")
            gini = metric_value(record.summary, "monosemanticity.mean_gini")
            if mse is None or cos is None or gini is None:
                continue
            xs_mse.append(mse)
            xs_cos.append(cos)
            ys.append(gini)
            text.append(f"{record.variant_label}<br>seed={record.seed}")

        if not ys:
            continue
        for col, xs in [(1, xs_mse), (2, xs_cos)]:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    text=text,
                    mode="markers",
                    marker={"size": 13, "color": color, "line": {"color": "white", "width": 1}},
                    name=label,
                    showlegend=col == 1,
                    hovertemplate="%{text}<br>x=%{x:.6g}<br>mean_gini=%{y:.6g}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=[mean(xs_mse)],
                y=[mean(ys)],
                mode="markers",
                marker={"symbol": "star", "size": 20, "color": color, "line": {"color": "black", "width": 1}},
                name=f"{label} mean",
                showlegend=False,
                hovertemplate=f"{label}<br>mean MSE=%{{x:.6g}}<br>mean Gini=%{{y:.6g}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[mean(xs_cos)],
                y=[mean(ys)],
                mode="markers",
                marker={"symbol": "star", "size": 20, "color": color, "line": {"color": "black", "width": 1}},
                showlegend=False,
                hovertemplate=f"{label}<br>mean cosine=%{{x:.6g}}<br>mean Gini=%{{y:.6g}}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        added = True

    if not added:
        return None

    fig.update_xaxes(title_text="MSE Total (↓ lower is better)", row=1, col=1, type="log")
    fig.update_xaxes(title_text="Cosine Similarity (↑ higher is better)", row=1, col=2)
    fig.update_yaxes(title_text="Mean Gini (↑ higher is better)", row=1, col=1)
    fig.update_yaxes(title_text="Mean Gini (↑ higher is better)", row=1, col=2)
    apply_camera_ready(
        fig,
        title="Reconstruction Quality vs Monosemanticity",
        legend_position="bottom-right",
        extra={"height": 680, "width": 1700},
    )
    return fig


def iter_record_files(suite_root: Path, pattern: str) -> list[Path]:
    return sorted(suite_root.glob(pattern))


def load_prompt_records(suite_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_record_files(suite_root, "runs/*/seed_*/evaluation/prompt_metrics.jsonl"):
        records.extend(read_jsonl(path))
    return [record for record in records if record.get("status", "ok") == "ok"]


def build_prompt_family_boxes(suite_root: Path, records: list[SummaryRecord]) -> go.Figure | None:
    allowed = allowed_runs(records)
    prompt_records = [
        record for record in load_prompt_records(suite_root) if record_run_key(record) in allowed
    ]
    if not prompt_records:
        return None

    variants = variant_order(records)
    label_map = {
        variant: variant_short_label(variant, label_for_variant(records, variant), multiline=True)
        for variant in variants
    }
    families = sorted({str(record.get("family")) for record in prompt_records if record.get("family")})
    family_colors = colors_for_variants(families)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[title for title, _ in PROMPT_METRICS],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )
    added = False

    for metric_index, (title, key) in enumerate(PROMPT_METRICS):
        row = metric_index // 2 + 1
        col = metric_index % 2 + 1
        for family in families:
            xs: list[str] = []
            ys: list[float] = []
            text: list[str] = []
            for record in prompt_records:
                if record.get("family") != family:
                    continue
                variant = str(record.get("variant_name"))
                if variant not in label_map:
                    continue
                value = safe_float(record.get(key))
                if value is None:
                    continue
                xs.append(label_map[variant])
                ys.append(value)
                text.append(
                    f"{record.get('variant_label', variant)}<br>"
                    f"seed={record.get('seed')}<br>"
                    f"prompt={record.get('prompt_id')}<br>"
                    f"family={family}"
                )
            if not ys:
                continue
            fig.add_trace(
                go.Box(
                    x=xs,
                    y=ys,
                    text=text,
                    name=family,
                    legendgroup=family,
                    marker_color=family_colors[family],
                    boxpoints="all",
                    jitter=0.45,
                    pointpos=0,
                    showlegend=metric_index == 0,
                    hovertemplate="%{text}<br>" + title + "=%{y:.6g}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            added = True
        set_readable_xaxis(fig, row=row, col=col)
        set_readable_yaxis(fig, row=row, col=col)

    if not added:
        return None

    apply_camera_ready(
        fig,
        title="Prompt-Level Metrics by Benchmark Family",
        legend_position="top-right",
        extra={"height": 900, "width": 1700, "boxmode": "group"},
    )
    return fig


def load_monosemanticity_records(suite_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_record_files(suite_root, "runs/*/seed_*/evaluation/monosemanticity_records.jsonl"):
        records.extend(read_jsonl(path))
    return records


def build_monosemanticity_distribution(suite_root: Path, records: list[SummaryRecord]) -> go.Figure | None:
    allowed = allowed_runs(records)
    mono_records = [
        record
        for record in load_monosemanticity_records(suite_root)
        if record_run_key(record) in allowed
    ]
    if not mono_records:
        return None

    variants = variant_order(records)
    label_map = variant_legend_labels(records, variants)
    colors = colors_for_variants(variants)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Gini Coefficient Distribution", "Activation Frequency vs Gini"],
        horizontal_spacing=0.12,
    )
    added = False

    for variant in variants:
        label = label_map[variant]
        values: list[float] = []
        freq: list[float] = []
        text: list[str] = []
        for record in mono_records:
            if record.get("variant_name") != variant:
                continue
            gini = safe_float(record.get("gini_coefficient"))
            activation_frequency = safe_float(record.get("activation_frequency"))
            if gini is None:
                continue
            values.append(gini)
            if activation_frequency is not None:
                freq.append(activation_frequency)
                text.append(
                    f"{record.get('variant_label', variant)}<br>"
                    f"seed={record.get('seed')}<br>"
                    f"L{record.get('layer')} F{record.get('feature_id')}"
                )
        if not values:
            continue
        fig.add_trace(
            go.Violin(
                y=values,
                name=label,
                legendgroup=variant,
                line_color=colors[variant],
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        if freq:
            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=values[: len(freq)],
                    text=text,
                    mode="markers",
                    marker={"size": 6, "opacity": 0.45, "color": colors[variant]},
                    name=label,
                    legendgroup=variant,
                    showlegend=False,
                    hovertemplate="%{text}<br>frequency=%{x:.6g}<br>gini=%{y:.6g}<extra></extra>",
                ),
                row=1,
                col=2,
            )
        added = True

    if not added:
        return None

    fig.update_yaxes(title_text="Gini Coefficient", row=1, col=1)
    fig.update_xaxes(title_text="Activation Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Gini Coefficient", row=1, col=2)
    apply_camera_ready(
        fig,
        title="Monosemanticity Feature Distributions",
        legend_position="bottom-right",
        extra={"height": 720, "width": 1700},
    )
    return fig


def downsample(records: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(records) <= max_points:
        return records
    stride = math.ceil(len(records) / max_points)
    return records[::stride]


def _interp(xs: list[float], ys: list[float], q: float) -> float | None:
    if not xs or q < xs[0] or q > xs[-1]:
        return None
    lo, hi = 0, len(xs) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if xs[mid] <= q:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[lo]
    t = (q - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


def build_training_curves(suite_root: Path, records: list[SummaryRecord], max_points: int) -> go.Figure | None:
    training_files = iter_record_files(suite_root, "runs/*/seed_*/training_records.jsonl")
    if not training_files:
        return None
    allowed = allowed_runs(records)
    variants = variant_order(records)
    label_map = {
        record.variant_name: variant_short_label(record.variant_name, record.variant_label)
        for record in records
    }
    colors = colors_for_variants(variants)

    # variant -> metric_key -> list of (seed, xs, ys)
    series: dict[str, dict[str, list[tuple[int, list[float], list[float]]]]] = {}
    for path in training_files:
        run_key = file_run_key(path)
        if run_key is None or run_key not in allowed:
            continue
        variant_name, seed = run_key
        run_records = [
            record
            for record in read_jsonl(path)
            if record.get("record_type") == "train_step" and safe_float(record.get("step")) is not None
        ]
        run_records.sort(key=lambda record: float(record["step"]))
        run_records = downsample(run_records, max_points=max_points)
        if not run_records:
            continue
        for title, key, _log_y in TRAINING_METRICS:
            xs: list[float] = []
            ys: list[float] = []
            for record in run_records:
                step = safe_float(record.get("step"))
                value = safe_float(record.get(key))
                if step is None or value is None:
                    continue
                xs.append(step)
                ys.append(value)
            if ys:
                series.setdefault(variant_name, {}).setdefault(key, []).append((seed, xs, ys))

    if not series:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[title for title, _, _ in TRAINING_METRICS],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )
    added = False

    for variant_name in variants:
        if variant_name not in series:
            continue
        label = label_map.get(variant_name, variant_name)
        color = colors.get(variant_name, "#1f77b4")
        for metric_index, (title, key, log_y) in enumerate(TRAINING_METRICS):
            row = metric_index // 2 + 1
            col = metric_index % 2 + 1
            seed_runs = series[variant_name].get(key, [])
            if not seed_runs:
                continue
            # Common grid = sorted union of all steps for this variant/metric.
            grid = sorted({x for _seed, xs, _ys in seed_runs for x in xs})
            means: list[float] = []
            stds: list[float] = []
            grid_used: list[float] = []
            for q in grid:
                vals: list[float] = []
                for _seed, xs, ys in seed_runs:
                    v = _interp(xs, ys, q)
                    if v is not None:
                        vals.append(v)
                if not vals:
                    continue
                means.append(mean(vals))
                stds.append(sample_std(vals))
                grid_used.append(q)
            if not means:
                continue
            n_seeds = len(seed_runs)
            # Per-seed thin background traces (visible only in legend toggle).
            for seed, xs, ys in seed_runs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line={"color": color, "width": 1, "dash": "dot"},
                        opacity=0.35,
                        name=f"{label} seed {seed}",
                        legendgroup=f"{variant_name}-seeds",
                        showlegend=metric_index == 0,
                        visible="legendonly",
                        hovertemplate=f"seed={seed}<br>step=%{{x}}<br>{title}=%{{y:.6g}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            if n_seeds >= 2:
                upper = [m + s for m, s in zip(means, stds)]
                lower = [
                    max(m - s, 1e-12) if log_y else m - s
                    for m, s in zip(means, stds)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=grid_used, y=upper, mode="lines",
                        line={"width": 0, "color": color},
                        showlegend=False, hoverinfo="skip",
                    ),
                    row=row, col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=grid_used, y=lower, mode="lines",
                        line={"width": 0, "color": color},
                        fill="tonexty", fillcolor=hex_to_rgba(color, 0.18),
                        showlegend=False, hoverinfo="skip",
                    ),
                    row=row, col=col,
                )
            fig.add_trace(
                go.Scatter(
                    x=grid_used,
                    y=means,
                    mode="lines",
                    name=f"{label} (n={n_seeds})",
                    legendgroup=variant_name,
                    line={"color": color, "width": 2.5},
                    showlegend=metric_index == 0,
                    hovertemplate=(
                        "step=%{x}<br>" + title + "=%{y:.6g}"
                        + ("<br>±%{customdata:.6g}" if n_seeds >= 2 else "")
                        + "<extra></extra>"
                    ),
                    customdata=stds if n_seeds >= 2 else None,
                ),
                row=row,
                col=col,
            )
            if log_y:
                fig.update_yaxes(type="log", row=row, col=col)
            added = True

    if not added:
        return None

    fig.update_xaxes(title_text="Step")
    apply_camera_ready(
        fig,
        title="Training Curves",
        legend_position="top-right",
        extra={"height": 900, "width": 1750},
    )
    return fig


def parse_spline_feature(path: Path) -> str:
    match = re.match(r"layer(\d+)_feat(\d+)\.csv", path.name)
    if not match:
        return path.stem
    return f"L{match.group(1)} F{match.group(2)}"


def build_spline_curves(
    suite_root: Path,
    records: list[SummaryRecord],
    max_files: int,
) -> go.Figure | None:
    allowed = allowed_runs(records)
    variant_rank = {variant: index for index, variant in enumerate(variant_order(records))}
    spline_files = [
        path
        for path in iter_record_files(suite_root, "runs/*/seed_*/evaluation/splines/*.csv")
        if file_run_key(path) in allowed
    ]
    spline_files = sorted(
        spline_files,
        key=lambda path: (
            variant_rank.get((file_run_key(path) or ("", -1))[0], len(variant_rank)),
            (file_run_key(path) or ("", -1))[1],
            str(path),
        ),
    )
    if not spline_files:
        return None
    spline_files = spline_files[:max_files]

    cols = 2
    rows = math.ceil(len(spline_files) / cols)
    subplot_titles = []
    for path in spline_files:
        run_key = file_run_key(path)
        variant = run_key[0] if run_key is not None else path.parents[3].name
        seed = run_key[1] if run_key is not None else path.parents[2].name.replace("seed_", "")
        matching_records = [record for record in records if record.variant_name == variant]
        if matching_records:
            variant_label = variant_short_label(variant, matching_records[0].variant_label)
        else:
            variant_label = variant
        subplot_titles.append(f"{variant_label} seed {seed}: {parse_spline_feature(path)}")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    added = False

    for file_index, path in enumerate(spline_files):
        row = file_index // cols + 1
        col = file_index % cols + 1
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "t" not in fieldnames:
                continue
            columns: dict[str, list[float]] = {field: [] for field in fieldnames}
            for csv_row in reader:
                for field in fieldnames:
                    value = safe_float(csv_row.get(field))
                    if value is not None:
                        columns[field].append(value)

        xs = columns.get("t", [])
        if not xs:
            continue
        for dim_name in [field for field in fieldnames if field != "t"]:
            ys = columns.get(dim_name, [])
            if len(ys) != len(xs):
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=dim_name,
                    legendgroup=dim_name,
                    showlegend=file_index == 0,
                    hovertemplate="t=%{x:.4g}<br>" + dim_name + "=%{y:.6g}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            added = True
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=row, col=col)

    if not added:
        return None

    fig.update_xaxes(title_text="Input value")
    fig.update_yaxes(title_text="KAN output")
    apply_camera_ready(
        fig,
        title=f"Representative Saved Spline Curves (first {len(spline_files)} CSV files)",
        legend_position="top-right",
        extra={"height": max(750, rows * 360), "width": 1700},
    )
    return fig


def load_spline_records(suite_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_record_files(suite_root, "runs/*/seed_*/evaluation/spline_records.jsonl"):
        records.extend(read_jsonl(path))
    return records


def build_nonlinearity_distribution(
    suite_root: Path, records: list[SummaryRecord]
) -> go.Figure | None:
    """Per-feature nonlinearity score: violin per variant + score vs activation freq."""
    allowed = allowed_runs(records)
    spline_records = [
        record
        for record in load_spline_records(suite_root)
        if record_run_key(record) in allowed
    ]
    if not spline_records:
        return None

    variants = variant_order(records)
    label_map = variant_legend_labels(records, variants)
    colors = colors_for_variants(variants)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Nonlinearity Score Distribution",
            "Activation Frequency vs Nonlinearity",
        ],
        horizontal_spacing=0.12,
    )
    added = False

    for variant in variants:
        label = label_map[variant]
        scores: list[float] = []
        freq: list[float] = []
        text: list[str] = []
        for record in spline_records:
            if record.get("variant_name") != variant:
                continue
            score = safe_float(record.get("nonlinearity_score"))
            if score is None:
                continue
            scores.append(score)
            activation_frequency = safe_float(record.get("activation_frequency"))
            if activation_frequency is not None:
                freq.append(activation_frequency)
                text.append(
                    f"{record.get('variant_label', variant)}<br>"
                    f"seed={record.get('seed')}<br>"
                    f"L{record.get('layer_id')} F{record.get('feature_id')}"
                    f"<br>top_dim={record.get('top_input_dim')}"
                )
        if not scores:
            continue
        fig.add_trace(
            go.Violin(
                y=scores,
                name=label,
                legendgroup=variant,
                line_color=colors[variant],
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
                points="all",
                jitter=0.3,
                marker={"size": 3, "opacity": 0.5},
            ),
            row=1,
            col=1,
        )
        if freq:
            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=scores[: len(freq)],
                    text=text,
                    mode="markers",
                    marker={"size": 6, "opacity": 0.5, "color": colors[variant]},
                    name=label,
                    legendgroup=variant,
                    showlegend=False,
                    hovertemplate=(
                        "%{text}<br>frequency=%{x:.6g}"
                        "<br>nonlinearity=%{y:.6g}<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )
        added = True

    if not added:
        return None

    fig.add_hline(
        y=0.05, line_dash="dash", line_color="gray", line_width=1,
        annotation_text="0.05 threshold", annotation_position="top right",
        row=1, col=1,
    )
    fig.add_hline(
        y=0.05, line_dash="dash", line_color="gray", line_width=1,
        row=1, col=2,
    )
    fig.update_yaxes(title_text="Nonlinearity Score (residual / variance)", row=1, col=1)
    fig.update_xaxes(title_text="Activation Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Nonlinearity Score", row=1, col=2)
    apply_camera_ready(
        fig,
        title="Spline Nonlinearity (per top-active KAN feature)",
        legend_position="top-right",
        extra={"height": 700, "width": 1700},
    )
    return fig


def load_reeval_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                fv = safe_float(value)
                row[key] = fv if fv is not None else value
            rows.append(row)
    return rows


REEVAL_GRAPH_METRICS: list[tuple[str, str, bool, bool]] = [
    ("Graph Replacement Score", "graph_replacement_score", True, False),
    ("Graph Completeness Score", "graph_completeness_score", True, False),
    ("Active Features (n)", "n_active_features", False, True),
    ("Cross-Position F→F Edges", "cross_position_ff_edges", True, False),
    ("Feature→Feature Edge L1", "feat_edge_l1", True, True),
    ("Error→Source Edge L1", "err_edge_l1", False, False),
]


def _reeval_variant_key(row: dict[str, Any]) -> str:
    return f"{row.get('suite_name','')}::{row.get('variant_name','')}"


def _reeval_short_label(suite: str, variant: str, encoder: str | None, dt: int | None) -> str:
    base = variant.replace("_gpt2_small", "")
    if encoder:
        enc = "Spline" if encoder.lower() == "kan" else "Linear"
    else:
        enc = ""
    pieces = [enc] if enc else []
    if dt is not None:
        pieces.append(f"d={dt}")
    short_variant = base
    if "feature_match" in base:
        short_variant = "FM"
    elif "param_match" in base:
        short_variant = "PM"
    elif base.startswith("dt"):
        short_variant = ""
    if short_variant:
        pieces.append(short_variant)
    suite_tag = "sweep" if "sweep" in suite else "main"
    pieces.append(f"[{suite_tag}]")
    return " ".join(pieces) if pieces else variant


def build_reeval_metric_overview(rows: list[dict[str, Any]]) -> go.Figure | None:
    if not rows:
        return None
    # Group rows by (suite, variant) → seeds → prompts
    grouped: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _reeval_variant_key(row)
        grouped.setdefault(key, []).append(row)
        meta.setdefault(key, {
            "suite": row.get("suite_name", ""),
            "variant": row.get("variant_name", ""),
            "encoder": row.get("encoder_type"),
            "d_transcoder": row.get("d_transcoder"),
        })

    keys = sorted(
        grouped.keys(),
        key=lambda k: (
            meta[k]["suite"],
            (meta[k]["d_transcoder"] if isinstance(meta[k]["d_transcoder"], (int, float)) else 1e12),
            meta[k]["variant"],
        ),
    )
    labels = [
        _reeval_short_label(
            meta[k]["suite"], meta[k]["variant"],
            meta[k]["encoder"],
            int(meta[k]["d_transcoder"]) if isinstance(meta[k]["d_transcoder"], (int, float)) else None,
        )
        for k in keys
    ]
    palette = [COLORWAY[i % len(COLORWAY)] for i in range(len(keys))]

    cols = 3
    n = len(REEVAL_GRAPH_METRICS)
    rows_count = math.ceil(n / cols)
    fig = make_subplots(
        rows=rows_count,
        cols=cols,
        subplot_titles=[t for t, _, _, _ in REEVAL_GRAPH_METRICS],
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )

    for i, (title, field, higher_is_better, log_y) in enumerate(REEVAL_GRAPH_METRICS):
        r = i // cols + 1
        c = i % cols + 1
        means: list[float] = []
        stds: list[float] = []
        seed_x: list[str] = []
        seed_y: list[float] = []
        hover: list[str] = []
        for key, label in zip(keys, labels):
            vals = [row[field] for row in grouped[key] if isinstance(row.get(field), (int, float))]
            for row in grouped[key]:
                v = row.get(field)
                if isinstance(v, (int, float)):
                    seed_x.append(label)
                    seed_y.append(float(v))
                    hover.append(
                        f"{meta[key]['suite']}<br>{meta[key]['variant']}<br>"
                        f"seed={row.get('seed')}<br>prompt={row.get('prompt_id')}<br>{title}={v:.6g}"
                    )
            means.append(mean(vals) if vals else float("nan"))
            stds.append(sample_std(vals))
        fig.add_trace(
            go.Bar(
                x=labels,
                y=means,
                error_y={"type": "data", "array": stds, "visible": True, "thickness": 1.5},
                marker_color=palette,
                marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=0.8,
                text=[fmt_value(v) for v in means],
                textposition="outside",
                textfont=VALUE_LABEL_FONT,
                cliponaxis=False,
                showlegend=False,
                hovertemplate="%{x}<br>mean=%{y:.6g}<extra></extra>",
            ),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=seed_x, y=seed_y,
                mode="markers",
                marker={"color": "rgba(0,0,0,0.7)", "size": 6, "line": {"color": "white", "width": 0.7}},
                text=hover,
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=r, col=c,
        )
        if log_y:
            fig.update_yaxes(type="log", row=r, col=c)
        direction = "↑" if higher_is_better else "↓"
        fig.add_annotation(
            text=direction,
            xref=f"x{i+1 if i > 0 else ''} domain",
            yref=f"y{i+1 if i > 0 else ''} domain",
            x=0.97, y=0.97, showarrow=False,
            font={"family": FONT_FAMILY, "size": 22, "color": "rgba(40,80,40,0.55)" if higher_is_better else "rgba(120,40,40,0.55)"},
        )

    apply_camera_ready(
        fig,
        title="Re-evaluated Attribution Graph Metrics  (CPU rebuild, fixed unified path)",
        show_legend=False,
        extra={"height": max(950, rows_count * 380), "width": 1750, "bargap": 0.28},
    )
    return fig


def build_reeval_dt_scaling(rows: list[dict[str, Any]]) -> go.Figure | None:
    """Scaling curves for the dt_sweep variants: metric vs d_transcoder."""
    sweep_rows = [
        row for row in rows
        if isinstance(row.get("d_transcoder"), (int, float)) and "sweep" in str(row.get("suite_name", ""))
    ]
    if not sweep_rows:
        return None

    # Aggregate: mean over (seed, prompt) per d_transcoder × encoder_type
    by_dt: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in sweep_rows:
        enc = str(row.get("encoder_type", "?"))
        dt = int(row["d_transcoder"])
        key = (enc, dt)
        bucket = by_dt.setdefault(key, {f: [] for f, *_ in [(m[1], m[0]) for m in REEVAL_GRAPH_METRICS]})
        for _, field, _, _ in REEVAL_GRAPH_METRICS:
            v = row.get(field)
            if isinstance(v, (int, float)):
                bucket[field].append(float(v))

    encoders = sorted({k[0] for k in by_dt})
    cols = 3
    n = len(REEVAL_GRAPH_METRICS)
    rows_count = math.ceil(n / cols)
    fig = make_subplots(
        rows=rows_count,
        cols=cols,
        subplot_titles=[t for t, _, _, _ in REEVAL_GRAPH_METRICS],
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )
    palette = {enc: COLORWAY[i % len(COLORWAY)] for i, enc in enumerate(encoders)}

    for i, (title, field, _hib, log_y) in enumerate(REEVAL_GRAPH_METRICS):
        r = i // cols + 1
        c = i % cols + 1
        for enc in encoders:
            xs = sorted({k[1] for k in by_dt if k[0] == enc})
            if not xs:
                continue
            ys = [mean(by_dt[(enc, dt)][field]) if by_dt[(enc, dt)][field] else float("nan") for dt in xs]
            errs = [sample_std(by_dt[(enc, dt)][field]) for dt in xs]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    error_y={"type": "data", "array": errs, "visible": True},
                    mode="lines+markers",
                    name=f"{enc}",
                    legendgroup=enc,
                    line={"color": palette[enc], "width": 2},
                    marker={"size": 9},
                    showlegend=i == 0,
                    hovertemplate=f"{enc}<br>d_transcoder=%{{x}}<br>{title}=%{{y:.6g}}<extra></extra>",
                ),
                row=r, col=c,
            )
        if log_y:
            fig.update_yaxes(type="log", row=r, col=c)
        fig.update_xaxes(type="log", row=r, col=c, title_text="d_transcoder")
        set_readable_xaxis(fig, row=r, col=c)
        set_readable_yaxis(fig, row=r, col=c)

    apply_camera_ready(
        fig,
        title="dt-sweep Scaling Curves (rebuilt graph metrics, mean ± seed/prompt std)",
        legend_position="top-right",
        extra={"height": max(900, rows_count * 350), "width": 1750},
    )
    return fig


def _fmt_mean_std(mu: float, sigma: float, n: int) -> str:
    if not math.isfinite(mu):
        return "—"
    if n <= 1:
        return f"{mu:.4g}"
    return f"{mu:.4g} ± {sigma:.2g}"


def write_per_layer_mse_csv(
    output_dir: Path,
    records: list[SummaryRecord],
) -> tuple[Path, Path] | None:
    """Emit per-layer MSE as long CSV (variant, seed, layer, mse) and wide mean±std pivot."""
    if not records:
        return None
    variants = variant_order(records)
    grouped = records_by_variant(records)
    short_labels = {
        v: variant_short_label(v, label_for_variant(records, v)) for v in variants
    }

    long_rows: list[tuple[str, str, int, int, float]] = []
    per_variant_layer: dict[str, dict[int, list[float]]] = {}
    max_layer = -1
    for variant in variants:
        per_variant_layer.setdefault(variant, {})
        for record in grouped.get(variant, []):
            values = record.summary.get("reconstruction", {}).get("mse_per_layer")
            if not isinstance(values, list):
                continue
            for layer, val in enumerate(values):
                v = safe_float(val)
                if v is None:
                    continue
                long_rows.append((variant, short_labels[variant], record.seed, layer, v))
                per_variant_layer[variant].setdefault(layer, []).append(v)
                max_layer = max(max_layer, layer)

    if not long_rows:
        return None

    long_path = output_dir / "per_layer_mse_long.csv"
    with long_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "variant_label", "seed", "layer", "mse"])
        for row in long_rows:
            w.writerow(row)

    wide_path = output_dir / "per_layer_mse_wide.csv"
    with wide_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["layer"]
        for variant in variants:
            label = short_labels[variant]
            header.extend([f"{label}_mean", f"{label}_std", f"{label}_n"])
        w.writerow(header)
        for layer in range(max_layer + 1):
            row: list[object] = [layer]
            for variant in variants:
                vals = per_variant_layer.get(variant, {}).get(layer, [])
                if vals:
                    row.extend([mean(vals), sample_std(vals), len(vals)])
                else:
                    row.extend(["", "", 0])
            w.writerow(row)

    return long_path, wide_path


def write_metrics_table(
    output_dir: Path,
    records: list[SummaryRecord],
) -> tuple[Path, Path, Path] | None:
    """Emit per-variant mean ± std tables (wide CSV, long CSV, markdown)."""
    if not records:
        return None
    variants = variant_order(records)
    grouped = records_by_variant(records)
    short_labels = {
        v: variant_short_label(v, label_for_variant(records, v)) for v in variants
    }
    metrics = list(all_metric_specs())  # (group, title, key, higher, log)

    # Long CSV: one row per (group, metric, variant) — easy for downstream tools.
    long_path = output_dir / "metrics_table_long.csv"
    with long_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "metric", "metric_key", "higher_is_better",
                    "variant", "variant_label", "n_seeds", "mean", "std", "values"])
        for group, title, key, higher, _log in metrics:
            for v in variants:
                vals = [
                    metric_value(r.summary, key)
                    for r in grouped.get(v, [])
                ]
                vals = [x for x in vals if x is not None]
                w.writerow([
                    group, title, key, higher,
                    v, short_labels[v], len(vals),
                    f"{mean(vals):.6g}" if vals else "",
                    f"{sample_std(vals):.6g}" if vals else "",
                    ";".join(f"{x:.6g}" for x in vals),
                ])

    # Wide CSV + Markdown: rows = metric, cols = variant ("mean ± std").
    wide_path = output_dir / "metrics_table.csv"
    md_path = output_dir / "metrics_table.md"
    with wide_path.open("w", newline="") as fcsv, md_path.open("w") as fmd:
        w = csv.writer(fcsv)
        header = ["group", "metric", "↑/↓"] + [short_labels[v] for v in variants]
        w.writerow(header)
        fmd.write("# Metrics Table (mean ± std across seeds)\n\n")
        fmd.write("| Group | Metric | ↑/↓ | " + " | ".join(short_labels[v] for v in variants) + " |\n")
        fmd.write("|" + "|".join(["---"] * len(header)) + "|\n")
        last_group = None
        for group, title, key, higher, _log in metrics:
            arrow = "↑" if higher else "↓"
            cells = []
            for v in variants:
                vals = [
                    metric_value(r.summary, key)
                    for r in grouped.get(v, [])
                ]
                vals = [x for x in vals if x is not None]
                if not vals:
                    cells.append("")
                else:
                    cells.append(_fmt_mean_std(mean(vals), sample_std(vals), len(vals)))
            w.writerow([group, title, arrow] + cells)
            group_cell = group if group != last_group else ""
            last_group = group
            fmd.write(f"| {group_cell} | {title} | {arrow} | " + " | ".join(cells) + " |\n")
    return wide_path, long_path, md_path


def write_index(
    output_dir: Path,
    suite_root: Path,
    figures: dict[str, Path],
    records: list[SummaryRecord],
    requested_seed_count: int | None,
) -> None:
    variants = variant_order(records)
    seeds = sorted({record.seed for record in records})
    rows = []
    for variant in variants:
        variant_records = [record for record in records if record.variant_name == variant]
        rows.append(
            "<tr>"
            f"<td>{variant}</td>"
            f"<td>{label_for_variant(records, variant)}</td>"
            f"<td>{', '.join(str(record.seed) for record in variant_records)}</td>"
            "</tr>"
        )

    links = "\n".join(
        f'<li><a href="{path.name}">{title}</a></li>'
        for title, path in figures.items()
    )
    variant_table = "\n".join(rows)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Paper Result Plotly Figures</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; max-width: 1100px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 0.45rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>Paper Result Plotly Figures</h1>
  <p>Suite root: <code>{suite_root}</code></p>
  <p>Plotted {len(records)} evaluation summaries across {len(variants)} variants and {len(seeds)} selected seeds.</p>
  <p>Requested seed count per variant: <code>{requested_seed_count if requested_seed_count is not None else "all available"}</code></p>
  <h2>Figures</h2>
  <ul>
    {links}
  </ul>
  <h2>Discovered Runs</h2>
  <table>
    <thead><tr><th>Variant</th><th>Label</th><th>Seeds</th></tr></thead>
    <tbody>
      {variant_table}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html)


def write_manifest(
    output_dir: Path,
    suite_root: Path,
    figures: dict[str, Path],
    records: list[SummaryRecord],
    requested_seed_count: int | None,
) -> None:
    manifest = {
        "suite_root": str(suite_root),
        "requested_seed_count": requested_seed_count,
        "summary_count": len(records),
        "variants": {
            variant: {
                "label": label_for_variant(records, variant),
                "seeds": [record.seed for record in records if record.variant_name == variant],
            }
            for variant in variant_order(records)
        },
        "figures": {title: str(path) for title, path in figures.items()},
    }
    (output_dir / "plotly_figure_manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir_arg",
        nargs="?",
        default=None,
        help="Optional direct results directory, e.g. results/paper/paper_gpt2_small_backup.",
    )
    parser.add_argument(
        "--suite",
        default="paper_gpt2_small",
        help="Suite name under --results-root. Ignored when --suite-root is provided.",
    )
    parser.add_argument(
        "--results-root",
        default=str(REPO_ROOT / "results" / "paper"),
        help="Directory containing paper suite outputs.",
    )
    parser.add_argument(
        "--suite-root",
        default=None,
        help="Explicit suite output directory, e.g. results/paper/paper_gpt2_small_backup.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Alias for a direct suite results directory. Overrides --suite and --suite-root.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help=(
            "Use up to this many sorted seeds per variant. If omitted, all discovered seeds "
            "are plotted. Variants with fewer seeds use all available seeds."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write HTML figures. Defaults to <suite-root>/figures/plotly.",
    )
    parser.add_argument(
        "--max-training-points",
        type=int,
        default=1200,
        help="Maximum training points per run trace after downsampling.",
    )
    parser.add_argument(
        "--max-spline-files",
        type=int,
        default=12,
        help="Maximum spline CSV files to include in the representative spline figure.",
    )
    parser.add_argument(
        "--reeval-csv",
        default=str(REPO_ROOT / "results" / "paper" / "reeval_cpu" / "per_prompt_graph_metrics.csv"),
        help=(
            "Per-prompt CSV from reeval_paper_results.py. "
            "If the file exists, build extra rebuilt-graph figures (08_reeval_*.html). "
            "Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export each figure as a high-DPI PNG (requires kaleido).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = suite_root_from_args(args)
    if not suite_root.exists():
        raise SystemExit(f"Suite root does not exist: {suite_root}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else suite_root / "figures" / "plotly"
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered_summaries = discover_summaries(suite_root)
    summaries = select_seed_count(discovered_summaries, args.seed_count)
    if not summaries:
        available = suites_with_summaries(Path(args.results_root).expanduser().resolve())
        hint = ""
        if available:
            hint = "\nSuites with evaluation summaries: " + ", ".join(available)
        raise SystemExit(
            f"No evaluation summaries found under {suite_root}/_eval_backup_20260503T210104Z/runs/*/seed_*/evaluation.{hint}"
        )

    figures: dict[str, Path] = {}
    figure_builders = [
        ("Metric Overview", lambda: build_metric_overview(summaries), "01_metric_overview.html"),
        ("Per-Layer MSE", lambda: build_per_layer_mse(summaries), "02_per_layer_mse.html"),
        (
            "Reconstruction vs Monosemanticity",
            lambda: build_reconstruction_scatter(summaries),
            "03_reconstruction_vs_monosemanticity.html",
        ),
        (
            "Prompt Family Metrics",
            lambda: build_prompt_family_boxes(suite_root, summaries),
            "04_prompt_family_metrics.html",
        ),
        (
            "Monosemanticity Distributions",
            lambda: build_monosemanticity_distribution(suite_root, summaries),
            "05_monosemanticity_distributions.html",
        ),
        (
            "Training Curves",
            lambda: build_training_curves(suite_root, summaries, args.max_training_points),
            "06_training_curves.html",
        ),
        (
            "Representative Spline Curves",
            lambda: build_spline_curves(suite_root, summaries, args.max_spline_files),
            "07_spline_curves.html",
        ),
        (
            "Spline Nonlinearity Distribution",
            lambda: build_nonlinearity_distribution(suite_root, summaries),
            "08_spline_nonlinearity.html",
        ),
    ]

    for title, builder, filename in figure_builders:
        fig = builder()
        if fig is None:
            print(f"[SKIP] {title}: no compatible data found")
            continue
        path = write_figure(fig, output_dir, filename, write_png=args.png)
        figures[title] = path
        print(f"[OK] {title}: {path}")

    if args.reeval_csv:
        reeval_path = Path(args.reeval_csv).expanduser().resolve()
        reeval_rows = load_reeval_csv(reeval_path)
        if reeval_rows:
            print(f"[reeval] loaded {len(reeval_rows)} rows from {reeval_path}")
            for title, builder, filename in [
                ("Re-evaluated Graph Metrics", lambda: build_reeval_metric_overview(reeval_rows), "08_reeval_metric_overview.html"),
                ("Re-evaluated dt-sweep Scaling", lambda: build_reeval_dt_scaling(reeval_rows), "09_reeval_dt_scaling.html"),
            ]:
                fig = builder()
                if fig is None:
                    print(f"[SKIP] {title}: no compatible reeval data")
                    continue
                path = write_figure(fig, output_dir, filename, write_png=args.png)
                figures[title] = path
                print(f"[OK] {title}: {path}")
        else:
            print(f"[reeval] {reeval_path} not found or empty — skipping rebuilt-graph plots")

    for group_name, metric_title, key, higher_is_better, log_y in all_metric_specs():
        fig = build_single_metric_figure(
            summaries,
            group_name=group_name,
            title=metric_title,
            key=key,
            higher_is_better=higher_is_better,
            log_y=log_y,
        )
        if fig is None:
            print(f"[SKIP] Metric {metric_title}: no compatible data found")
            continue
        filename = f"01_metric_{slugify(metric_title)}.html"
        path = write_figure(fig, output_dir, filename, write_png=args.png)
        figures[f"Metric: {metric_title}"] = path
        print(f"[OK] Metric {metric_title}: {path}")

    layer_mse_paths = write_per_layer_mse_csv(output_dir, summaries)
    if layer_mse_paths:
        long_p, wide_p = layer_mse_paths
        print(f"[OK] Per-layer MSE (long):     {long_p}")
        print(f"[OK] Per-layer MSE (wide):     {wide_p}")

    table_paths = write_metrics_table(output_dir, summaries)
    if table_paths:
        wide, long, md = table_paths
        print(f"[OK] Metrics table (wide):     {wide}")
        print(f"[OK] Metrics table (long):     {long}")
        print(f"[OK] Metrics table (markdown): {md}")

    write_index(output_dir, suite_root, figures, summaries, args.seed_count)
    write_manifest(output_dir, suite_root, figures, summaries, args.seed_count)
    print(f"\nWrote {len(figures)} figures to {output_dir}")
    print(f"Open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
