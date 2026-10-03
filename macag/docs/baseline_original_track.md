# Original-pipeline baseline track (true AutoCircuit)

Uses **UFO-101 [`auto-circuit`](https://github.com/UFO-101/auto-circuit)** — the
maintained implementation that ArthurConmy's ACDC repo redirects to, and that
documents its EAP routine as an exact Syed et al. 2023 replication.

Writes `macag_original_baselines.json`. **Never** Jaccard against Game 1
(different edge ID universe). Compare **size / logit_gap / KL / wall cost**.

## Methods

| ID | Implementation | Graph |
|----|----------------|-------|
| `eap_edge` | `auto_circuit.prune_algos.edge_attribution_patching` | Factorized residual edges (Q/K/V destinations) |
| `acdc_edge` | `auto_circuit.prune_algos.ACDC.acdc_prune_scores` | Same factorized edge graph; corrupt resample patching |

## Requirements

```bash
pip install auto-circuit
```

TransformerLens 3.x is supported via `macag.baselines.original.tl_compat`
(aliases the renamed KV-cache module). Models are configured with
`use_attn_result`, `use_hook_mlp_in`, and `use_split_qkv_input`.

**GQA note:** upstream auto-circuit builds K/V destinations for all `n_heads`,
but TL `hook_{k,v}_input` has `n_key_value_heads` under GQA (Gemma-2 / Llama-3).
`macag.baselines.original.autocircuit_bridge` patches dest-node construction so
K/V edges use `n_key_value_heads` (Q still uses `n_heads`).

**Dtype note:** original baselines force `float32` — auto-circuit patch einsum
rejects bf16/float mixes common with Gemma/Llama defaults.

**ACDC score ties:** ACDC only writes `+inf` (kept) or `τ` (dropped). Budget
matching must evaluate the **exact** trimmed edge list; auto-circuit's
threshold keep expands the `τ` band to (nearly) the full graph and yields
spurious `kl≈0`.

## Run

Native (paper-style; default for the one-prompt pilot):

```bash
python -m macag.cli.run_original_baselines \
  --oracle-kwargs-file <run>/logit_gap/oracle_kwargs.json \
  --input-id <slug> \
  --budget 0 \
  --acdc-target-k 0 \
  --acdc-taus 0.1 \
  --methods eap_edge,acdc_edge \
  --output-json <run>/logit_gap/macag_original_baselines.json
```

- **EAP**: |AtP| ranking + size→faithfulness `sweep` (no fixed circuit).
- **ACDC**: τ-survivor edge set (no hard k-cap). Compare size / logit_gap / KL
  to MACAG Game 1 via `comparison.game1_ref` in the same JSON.

Budget-matched (optional head-to-head at fixed k):

```bash
python -m macag.cli.run_original_baselines ... --budget 8 --acdc-target-k -1
```

## Claim language

- Fair: “vs Syed EAP / Conmy ACDC as implemented in auto-circuit on the factorized edge graph.”
- Unfair: Jaccard / precision@k vs CLT Game 1 node sets.

## Approximate (legacy) writer-only code

Earlier in-repo approximations live under
`output_edge_approx_*.py` for reference only — **not** used by the CLI.
