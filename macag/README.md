# MACAG

This package adds two evidence-allocation games over mechanistic circuit nodes:

- Game 1: minimal faithful evidence subset
- Game 2: contrastive evidence allocation with overlap penalty

## CLI

Run with toy oracle:

```bash
PYTHONPATH=. python -m macag.cli.run_macag game2 \
  --graph-json /path/to/graph.json \
  --target y \
  --foil y_foil \
  --toy-oracle-json /path/to/toy_oracle.json \
  --output-json /tmp/macag_out.json
```

Visualize MACAG evidence inside the existing `circuit-tracer` frontend:

```bash
PYTHONPATH=. python -m macag.cli.annotate_graph \
  --graph-json /absolute/path/to/circuit.json \
  --macag-result-json /tmp/macag_out.json \
  --output-json /absolute/path/to/circuit_macag.json

python -m circuit_tracer start-server \
  --graph_file_dir /absolute/path/to
```

Then open the annotated graph slug in the browser UI. Added `qParams.pinnedIds` and
supernodes show `MACAG:shared`, `MACAG:unique_y`, `MACAG:unique_foil` (or `MACAG:E_star` for Game 1).
The annotator also updates `/absolute/path/to/graph-metadata.json` automatically so the new
graph appears in the server dropdown.

Run with real `ReplacementModel` interventions using factory wiring:

1. Create oracle kwargs JSON:

```json
{
  "model_name": "google/gemma-2-2b",
  "transcoder_set": "gemma",
  "prompt": "The capital of the state containing Denver is",
  "graph_json": "/absolute/path/to/circuit.json",
  "backend": "transformerlens",
  "score_kind": "logit_gap",
  "target_token_by_label": {
    "y": " Colorado",
    "y_foil": " Wyoming"
  },
  "foil_by_target": {
    "y": "y_foil",
    "y_foil": "y"
  },
  "freeze_attention": true
}
```

2. Run:

```bash
PYTHONPATH=. python -m macag.cli.run_macag game2 \
  --graph-json /absolute/path/to/circuit.json \
  --target y \
  --foil y_foil \
  --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle \
  --oracle-kwargs-file /absolute/path/to/oracle_kwargs.json \
  --output-json /tmp/macag_game2.json
```

Notes:
- `target_token_by_label` entries can be token strings or explicit ids (`"id:12345"`).
- The replacement-model factory automatically restricts candidates to feature nodes found in the graph JSON (`feature_type == "cross layer transcoder"` by default).
- `local_clt_path` can point to either a standard `circuit_tracer` CLT checkpoint or a
  `spline_clt` checkpoint directory (auto-detected via `metadata.safetensors`).

## Auto-Suggest Supernodes

If you do not have manually annotated supernodes, you can auto-suggest them from graph metrics
(`influence`, `activation`) plus graph connectivity:

```bash
PYTHONPATH=. python -m macag.cli.suggest_supernodes \
  --graph-json /absolute/path/to/circuit.json \
  --output-supernodes-json /absolute/path/to/auto_supernodes.json \
  --output-candidates-json /absolute/path/to/auto_candidates.json \
  --output-graph-json /absolute/path/to/circuit_with_auto_supernodes.json \
  --replace-existing-supernodes
```

Use the produced candidates file directly with MACAG runs:

```bash
PYTHONPATH=. python -m macag.cli.run_macag game2 \
  --graph-json /absolute/path/to/circuit.json \
  --candidates-file /absolute/path/to/auto_candidates.json \
  --target y \
  --foil y_foil \
  --oracle-factory macag.factories.replacement_model:create_replacement_model_oracle \
  --oracle-kwargs-file /absolute/path/to/oracle_kwargs.json \
  --output-json /tmp/macag_game2_auto_candidates.json
```
