"""Factory functions for ReplacementModel-backed MACAG scoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from macag.graph import NodeId
from macag.scoring import ReplacementModelInterventionScorer, ScoreKind, ScoringOracle

LOGGER = logging.getLogger(__name__)


def _load_json(path: str | Path) -> Any:
    resolved = Path(path)
    try:
        text = resolved.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved}: {exc}") from exc


def _normalize_feature_types(feature_types: Sequence[str] | None) -> set[str]:
    if feature_types is None:
        feature_types = ["cross layer transcoder"]
    return {feature_type.strip().lower() for feature_type in feature_types}


def _extract_token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        token_ids = encoded["input_ids"]
    else:
        token_ids = encoded.input_ids

    # Some tokenizers return [[...]] for batched outputs.
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def resolve_target_to_logit_idx(
    tokenizer: Any,
    target_token_by_label: Mapping[Any, str | int],
    strict_single_token: bool = True,
) -> dict[Any, int]:
    """Resolve class labels to vocab indices via tokenizer encoding."""
    target_to_logit_idx: dict[Any, int] = {}
    for label, token_spec in target_token_by_label.items():
        if isinstance(token_spec, int):
            target_to_logit_idx[label] = token_spec
            continue

        token_str = str(token_spec)
        if token_str.startswith("id:"):
            target_to_logit_idx[label] = int(token_str[3:])
            continue

        token_ids = _extract_token_ids(tokenizer, token_str)
        if strict_single_token and len(token_ids) != 1:
            raise ValueError(
                f"Target '{label}' token '{token_str}' tokenized to {len(token_ids)} IDs; "
                "expected exactly 1. Provide an explicit vocab id via `id:<int>`."
            )
        if not token_ids:
            raise ValueError(f"Target '{label}' token '{token_str}' tokenized to an empty sequence.")
        # BPE footgun: for GPT-2-style tokenizers, "Paris" and " Paris" are different
        # single tokens, and next-token continuations mid-sentence almost always use
        # the space-prefixed variant. Warn when both variants are single tokens but
        # differ, so a silently-wrong logit index doesn't corrupt every score.
        if not token_str[:1].isspace():
            spaced_ids = _extract_token_ids(tokenizer, " " + token_str)
            if len(spaced_ids) == 1 and len(token_ids) >= 1 and spaced_ids[0] != token_ids[0]:
                LOGGER.warning(
                    "Target '%s' token '%s' has no leading space; ' %s' is a different "
                    "single token (id %d vs %d). Mid-sentence continuations usually "
                    "need the space-prefixed variant.",
                    label,
                    token_str,
                    token_str,
                    token_ids[0],
                    spaced_ids[0],
                )
        # Next-token prediction scores the FIRST token of the continuation; when
        # strict_single_token is disabled and the label is multi-token, the first
        # sub-token is the relevant logit, not the last (I2).
        target_to_logit_idx[label] = token_ids[0]
    return target_to_logit_idx


_ERROR_NODE_FEATURE_TYPE = "mlp reconstruction error"


def load_feature_node_interventions_from_graph_json(
    graph_json: str | Path,
    feature_types: Sequence[str] | None = None,
    include_error_nodes: bool = False,
) -> dict[NodeId, tuple[int, int, int]]:
    """Extract (layer, pos, feature_idx) interventions for feature nodes in graph JSON.

    `include_error_nodes` is the opt-in C1 escape hatch for making MLP
    reconstruction-error nodes ablatable so that the "empty" baseline is a true
    empty. It is **not** supported by the feature-index intervention path
    (`ReplacementModel.feature_intervention` ablates transcoder feature indices,
    and error nodes have no feature index — their `node_id` encodes a sentinel
    `feature == -1`). Rather than silently mapping an error node to a bogus
    feature index, we raise so callers fall back to the report-only normalized
    metrics (`sufficiency_normalized` / `recoverable_range`), which are the
    supported mitigation.
    """
    payload = _load_json(graph_json)
    nodes = payload.get("nodes", [])

    if include_error_nodes:
        has_error_node = any(
            isinstance(node, Mapping)
            and str(node.get("feature_type", "")).strip().lower() == _ERROR_NODE_FEATURE_TYPE
            for node in nodes
        )
        if has_error_node:
            raise NotImplementedError(
                "include_error_nodes=True requested, but error-vector ablation is not "
                "supported by ReplacementModel.feature_intervention (it ablates transcoder "
                "feature indices only; MLP reconstruction-error nodes have no feature index). "
                "Use the report-only normalized metrics instead: 'sufficiency_normalized', "
                "'necessity_normalized', 'faithfulness_normalized', 'error_floor', and "
                "'recoverable_range' de-bias faithfulness against the retained error floor "
                "without ablating error nodes."
            )

    allowed_types = _normalize_feature_types(feature_types)
    node_to_intervention: dict[NodeId, tuple[int, int, int]] = {}

    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        node_id = node.get("node_id", node.get("id"))
        feature_type = str(node.get("feature_type", "")).strip().lower()

        if node_id is None or feature_type not in allowed_types:
            continue

        layer_raw = node.get("layer")
        pos_raw = node.get("ctx_idx")
        if layer_raw is None or pos_raw is None:
            continue

        parts = str(node_id).split("_")
        if len(parts) != 3:
            LOGGER.debug("Skipping node %s: node_id does not split into 3 parts (got %d)", node_id, len(parts))
            continue

        try:
            layer = int(layer_raw)
            pos = int(pos_raw)
            feature_idx = int(parts[1])
        except (TypeError, ValueError) as exc:
            LOGGER.debug("Skipping node %s: failed to parse numeric fields: %s", node_id, exc)
            continue

        node_to_intervention[node_id] = (layer, pos, feature_idx)

    if not node_to_intervention:
        raise ValueError(
            "No feature-node interventions found in graph JSON. Check feature_type values "
            "or pass a custom `feature_types` list."
        )
    return node_to_intervention


def _coerce_constrained_layers(value: Any) -> range | None:
    if value is None:
        return None
    if isinstance(value, range):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, stop = int(value[0]), int(value[1])
        return range(start, stop)
    raise ValueError(
        "`constrained_layers` must be None, a range, or a 2-item [start, stop] list/tuple."
    )


def _default_foil_map(
    target_to_logit_idx: Mapping[Any, int],
    foil_by_target: Mapping[Any, Any] | None,
) -> Mapping[Any, Any] | None:
    if foil_by_target:
        return foil_by_target
    if len(target_to_logit_idx) == 2:
        labels = list(target_to_logit_idx.keys())
        return {labels[0]: labels[1], labels[1]: labels[0]}
    return None


def _normalize_model_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(kwargs or {})
    if not normalized:
        return normalized

    if "dtype" in normalized and isinstance(normalized["dtype"], str):
        import torch

        dtype_aliases = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }
        dtype_name = dtype_aliases.get(normalized["dtype"].lower(), normalized["dtype"])
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unknown dtype string for model kwargs: {normalized['dtype']}")
        normalized["dtype"] = getattr(torch, dtype_name)

    if "device" in normalized and isinstance(normalized["device"], str):
        import torch

        normalized["device"] = torch.device(normalized["device"])

    return normalized


def _load_local_transcoders(
    *,
    clt_path: Path,
    model_kwargs: Mapping[str, Any],
    clt_scan: str | list[str] | None,
):
    """Load either a standard CLT or a Spline-CLT checkpoint from a local directory."""
    import yaml

    feature_input_hook = "hook_resid_mid"
    feature_output_hook = "hook_mlp_out"
    config_path = clt_path / "config.yaml"
    metadata_path = clt_path / "metadata.safetensors"

    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
        feature_input_hook = config.get("feature_input_hook", feature_input_hook)
        feature_output_hook = config.get("feature_output_hook", feature_output_hook)
        if clt_scan is None:
            clt_scan = config.get("scan")

    load_kwargs: dict[str, Any] = {
        "feature_input_hook": feature_input_hook,
        "feature_output_hook": feature_output_hook,
        "scan": clt_scan,
    }
    if model_kwargs.get("dtype") is not None:
        load_kwargs["dtype"] = model_kwargs["dtype"]
    if model_kwargs.get("device") is not None:
        load_kwargs["device"] = model_kwargs["device"]

    if metadata_path.exists():
        from spline_clt.kan_transcoder import load_spline_clt

        LOGGER.info("Loading Spline-CLT checkpoint from %s", clt_path)
        return load_spline_clt(str(clt_path), **load_kwargs)

    from circuit_tracer.transcoder.cross_layer_transcoder import load_clt

    LOGGER.info("Loading standard CLT checkpoint from %s", clt_path)
    return load_clt(
        str(clt_path),
        lazy_decoder=True,
        lazy_encoder=False,
        **load_kwargs,
    )


def create_replacement_model_scorer(
    *,
    model_name: str,
    transcoder_set: str | None = None,
    prompt: str,
    graph_json: str | Path,
    target_to_logit_idx: Mapping[Any, int] | None = None,
    target_token_by_label: Mapping[Any, str | int] | None = None,
    backend: str = "transformerlens",
    score_kind: ScoreKind = "logit_gap",
    foil_by_target: Mapping[Any, Any] | None = None,
    default_foil: Any | None = None,
    ablation_value: float = 0.0,
    constrained_layers: Any = None,
    freeze_attention: bool = True,
    feature_types: Sequence[str] | None = None,
    include_error_nodes: bool = False,
    strict_single_token: bool = True,
    model_kwargs: Mapping[str, Any] | None = None,
    local_clt_path: str | Path | None = None,
    clt_scan: str | list[str] | None = None,
) -> ReplacementModelInterventionScorer:
    """Build a ReplacementModelInterventionScorer from circuit-tracer config.

    `local_clt_path` may point to either:
    - a standard `circuit_tracer` CLT checkpoint directory (`load_clt` format), or
    - a `spline_clt` checkpoint directory (`load_spline_clt` format, includes `metadata.safetensors`).
    """
    from circuit_tracer import ReplacementModel

    kwargs = _normalize_model_kwargs(model_kwargs)
    if local_clt_path is not None:
        clt_path = Path(local_clt_path)
        transcoders = _load_local_transcoders(
            clt_path=clt_path,
            model_kwargs=kwargs,
            clt_scan=clt_scan,
        )
        model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=model_name,
            transcoders=transcoders,
            backend=backend,  # type: ignore[arg-type]
            **kwargs,
        )
    else:
        if not transcoder_set:
            raise ValueError("Provide `transcoder_set` when `local_clt_path` is not provided.")
        model = ReplacementModel.from_pretrained(
            model_name=model_name,
            transcoder_set=transcoder_set,
            backend=backend,  # type: ignore[arg-type]
            **kwargs,
        )

    node_to_intervention = load_feature_node_interventions_from_graph_json(
        graph_json=graph_json,
        feature_types=feature_types,
        include_error_nodes=include_error_nodes,
    )

    if target_to_logit_idx is None:
        if target_token_by_label is None:
            raise ValueError(
                "Provide either `target_to_logit_idx` or `target_token_by_label`."
            )
        target_to_logit_idx = resolve_target_to_logit_idx(
            tokenizer=model.tokenizer,
            target_token_by_label=target_token_by_label,
            strict_single_token=strict_single_token,
        )

    scorer = ReplacementModelInterventionScorer(
        model=model,
        prompt=prompt,
        node_to_intervention=node_to_intervention,
        target_to_logit_idx=target_to_logit_idx,
        score_kind=score_kind,
        foil_by_target=_default_foil_map(target_to_logit_idx, foil_by_target),
        default_foil=default_foil,
        ablation_value=ablation_value,
        constrained_layers=_coerce_constrained_layers(constrained_layers),
        freeze_attention=freeze_attention,
    )
    return scorer


@dataclass
class OracleFactoryOutput:
    oracle: ScoringOracle
    candidates: list[NodeId] | None = None


def create_replacement_model_oracle(
    *,
    cache_enabled: bool = True,
    **kwargs: Any,
) -> OracleFactoryOutput:
    """Create ScoringOracle + default candidates for MACAG CLI usage."""
    scorer = create_replacement_model_scorer(**kwargs)
    oracle = ScoringOracle(backend=scorer, cache_enabled=cache_enabled)
    candidates = sorted([str(node_id) for node_id in scorer.node_to_intervention.keys()])
    return OracleFactoryOutput(oracle=oracle, candidates=candidates)
