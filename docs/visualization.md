# Visualization

This document explains how to visualize Spline-CLT circuit tracing results using the upstream circuit-tracer tools and KAN-specific spline analysis.

## Overview

Spline-CLT produces attribution results (causal effects, Shapley values) that are converted into the standard `circuit_tracer.graph.Graph` format via the adapter in `attribution/graph.py`. This means all upstream visualization tools — the interactive web frontend, graph pruning, scoring — work with Spline-CLT out of the box.

There are three visualization approaches:

| Approach | Best for | Output |
|----------|----------|--------|
| [Interactive web frontend](#interactive-web-frontend) | Exploring circuits interactively | D3.js in browser |
| [CLI + web server](#cli-workflow) | Quick circuit inspection | Browser at localhost |
| [Notebook SVG](#notebook-visualization) | Papers and presentations | Inline SVG |
| [Spline analysis](#spline-visualization-kan-only) | Understanding KAN-learned functions | PNG plots + CSV |

## Interactive Web Frontend

The circuit-tracer includes a full D3.js web UI with force-directed graph rendering, node/edge inspection, and feature detail panels.

### Step-by-Step (Python)

```python
from spline_clt.kan_transcoder import load_spline_clt
from attribution.causal import build_attribution_graph
from attribution.graph import create_graph_from_attribution
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend.local_server import serve
import transformer_lens
import torch

# 1. Load trained Spline-CLT
model = load_spline_clt("checkpoints/gpt2_small/spline_clt_gpt2_best")

# 2. Load language model and collect activations for a prompt
lm = transformer_lens.HookedTransformer.from_pretrained("gpt2")
prompt = "The Eiffel Tower is located in"
tokens = lm.to_tokens(prompt)
_, cache = lm.run_with_cache(tokens)

# Stack MLP inputs/outputs across layers
mlp_inputs = torch.stack([
    cache[f"blocks.{i}.hook_resid_mid"] for i in range(12)
])  # (n_layers, 1, seq_len, d_model)
mlp_inputs = mlp_inputs.squeeze(1)  # (n_layers, seq_len, d_model)

# 3. Run causal attribution
attribution = build_attribution_graph(model, mlp_inputs, max_features=64)

# 4. Get top predicted tokens for the graph
logits = lm(tokens)
top_k = logits[0, -1].topk(5)
logit_tokens = top_k.indices
logit_probs = top_k.values.softmax(dim=-1)

# 5. Convert to circuit-tracer Graph
graph = create_graph_from_attribution(
    attribution_result=attribution,
    input_string=prompt,
    input_tokens=tokens[0],
    logit_tokens=logit_tokens,
    logit_probabilities=logit_probs,
    cfg=lm.cfg,
)

# 6. Export to JSON for the web UI
create_graph_files(
    graph,
    slug="eiffel_tower",
    output_path="results/graphs/",
    node_threshold=0.8,    # keep top 80% of nodes by influence
    edge_threshold=0.98,   # keep top 98% of edges by weight
)

# 7. Start local server
server = serve("results/graphs/", port=8032)
# Open http://localhost:8032 in your browser
```

### What You See in the Web UI

- **Force-directed graph**: Nodes are features (colored by layer), edges are causal connections weighted by attribution
- **Node inspection**: Click a node to see its layer, position, feature ID, activation value, and influence score
- **Edge inspection**: Hover over edges to see connection weights
- **Feature detail panel**: Activation histograms and max-activating examples for selected features
- **Pruning controls**: Adjust node/edge thresholds interactively
- **URL persistence**: Graph state is encoded in the URL for sharing

### Graph JSON Format

The web UI reads JSON files with this structure:
```json
{
  "metadata": {
    "slug": "eiffel_tower",
    "prompt": "The Eiffel Tower is located in",
    "prompt_tokens": ["The", " Eiffel", " Tower", " is", " located", " in"],
    "node_threshold": 0.8,
    "schema_version": 1
  },
  "nodes": [
    {
      "node_id": "3_42_5",
      "feature": 152,
      "layer": "3",
      "ctx_idx": 5,
      "feature_type": "cross layer transcoder",
      "activation": 2.34,
      "influence": 0.045
    }
  ],
  "links": [
    {"source": "3_42_5", "target": "4_18_5", "weight": 0.23}
  ]
}
```

## CLI Workflow

For quick circuit inspection without writing Python:

```bash
# 1. Generate attribution and save as .pt file
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --max-features 64 \
    --output results/circuits/eiffel.pt

# 2. (Optional) Also run Shapley attribution
conda run -n ct python experiments/run_circuit.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --prompt "The Eiffel Tower is located in" \
    --shapley --shapley-samples 128 \
    --output results/circuits/eiffel_shapley.pt
```

This prints a console summary with:
- Top 20 features ranked by activation magnitude
- Top 10 feature-to-feature edges by causal effect
- (If `--shapley`) Top 10 features by Shapley value

Then convert to the web UI:

```python
from circuit_tracer.graph import Graph
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend.local_server import serve

graph = Graph.from_pt("results/circuits/eiffel.pt")
create_graph_files(graph, slug="eiffel", output_path="results/graphs/")
serve("results/graphs/", port=8032)
```

## Full Pipeline Visualization

The evaluation pipeline (`run_pipeline.py`) generates circuit traces for 7 benchmark prompts automatically:

```bash
conda run -n ct python experiments/run_pipeline.py \
    --kan-checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --linear-checkpoint checkpoints/gpt2_small_linear/linear_clt_gpt2_best \
    --data-dir data/activations \
    --output-dir results/eval_run1 \
    --shapley
```

This produces `.pt` files in `results/eval_run1/circuits/` for each (model, prompt) pair. Convert them all to the web UI:

```python
import glob
from circuit_tracer.graph import Graph
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend.local_server import serve

for pt_file in glob.glob("results/eval_run1/circuits/*.pt"):
    slug = pt_file.split("/")[-1].replace(".pt", "")
    graph = Graph.from_pt(pt_file)
    create_graph_files(graph, slug=slug, output_path="results/all_graphs/")

serve("results/all_graphs/", port=8032)
```

The `graph-metadata.json` file indexes all graphs, and the web UI lets you switch between them.

## Notebook Visualization

For Jupyter notebooks (papers, presentations):

```python
from demos.graph_visualization import (
    create_graph_visualization,
    InterventionGraph,
    Supernode,
)

# Build a simplified node hierarchy from attribution results
# (manually select key features to highlight)
capital_node = Supernode(
    label="'capital' feature",
    layer=3,
    activation_pct=0.95,
    intervention_label="+2x",
)
location_node = Supernode(
    label="'location' feature",
    layer=7,
    activation_pct=0.82,
)
output_node = Supernode(
    label="' Paris'",
    layer=11,
    activation_pct=0.91,
)

graph = InterventionGraph(
    nodes=[[capital_node, location_node], [output_node]],
    prompt="The Eiffel Tower is located in",
)

viz = create_graph_visualization(
    intervention_graph=graph,
    top_outputs=[("Paris", 0.78), ("France", 0.12), ("the", 0.05)],
)

# Display inline in Jupyter
from IPython.display import display
display(viz)
```

This renders an SVG with:
- Node boxes showing activation percentages
- Intervention badges (e.g., "+2x", "-2x")
- Connection arrows colored by type
- Prompt text and top predictions below the graph

## Spline Visualization (KAN Only)

Unique to Spline-CLT — visualizes the learned nonlinear transfer functions in the B-spline encoder.

```bash
conda run -n ct python experiments/analyze_splines.py \
    --checkpoint checkpoints/gpt2_small/spline_clt_gpt2_best \
    --n-features 20 \
    --output-dir results/splines
```

### What Gets Produced

```
results/splines/
├── feature_summary.csv          # All features: layer, feat_id, frequency, mean, max
├── layer0_feat42.csv            # Spline curves: t, dim_3, dim_8, dim_15
├── layer0_feat42.png            # Matplotlib plot
├── layer2_feat128.csv
├── layer2_feat128.png
└── ...
```

Each PNG shows three curves (one per top input dimension) plotting the KAN encoder's transfer function for that feature:
- **x-axis**: input value along that dimension (range -5 to +5)
- **y-axis**: encoder output (feature pre-activation)

### Interpreting Spline Plots

| Curve Shape | What It Means |
|-------------|---------------|
| **Straight line** | KAN behaves linearly for this feature/dimension — no benefit over linear encoder |
| **Step function** | Feature detects a threshold — fires when input exceeds a specific value |
| **Peak/bump** | Feature responds to a specific input range — band-pass behavior |
| **Polynomial curve** | Smooth nonlinear relationship the linear encoder cannot capture |
| **Flat line** | This input dimension doesn't contribute to the feature |

Spline plots are the primary tool for answering the research question: "Did the KAN encoder learn meaningful nonlinear features that a linear encoder would miss?"

## How the Graph Adapter Works

The bridge between Spline-CLT and the circuit-tracer visualization is `attribution/graph.py`:

```
Spline-CLT Attribution          →  create_graph_from_attribution()  →  circuit_tracer.graph.Graph
─────────────────────────────────────────────────────────────────────────────────────────────
active_features (n_active, 3)   →  Feature nodes (layer, pos, feat_id)
activation_values (n_active,)   →  Node activation magnitudes
feature_effects (n_active²)     →  Edge weights in adjacency matrix
output_effects                  →  Feature-to-error/logit edges
```

The nonlinear KAN encoder is transparent to the visualization layer. Encoder directions used for attribution are Jacobian rows (local linear approximation), so the graph structure is the same as it would be for a linear CLT. The only difference is that the encoder directions are input-dependent rather than fixed.

## Key Files

| File | Role |
|------|------|
| `attribution/graph.py` | Converts Spline-CLT attribution → circuit_tracer Graph |
| `circuit_tracer/graph.py` | Graph class with `to_pt()` / `from_pt()` |
| `circuit_tracer/utils/create_graph_files.py` | Graph → JSON for web UI |
| `circuit_tracer/frontend/local_server.py` | Local HTTP server for web UI |
| `circuit_tracer/frontend/assets/` | D3.js web frontend (HTML/JS/CSS) |
| `demos/graph_visualization.py` | SVG rendering for Jupyter notebooks |
| `experiments/run_circuit.py` | CLI for single-prompt circuit tracing |
| `experiments/analyze_splines.py` | KAN spline transfer function visualization |
