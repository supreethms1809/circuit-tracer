# MACAG baseline method map

Stable method IDs used by `python -m macag.cli.run_baselines` and
`macag_baselines.json`. **Do not rename** legacy IDs (`eap`, `acdc`) — historical
results stay comparable. Prefer the explicit aliases in new runs.

| Method ID | Alias(es) | What it actually is | Paper claim |
|-----------|-----------|---------------------|-------------|
| `influence` | — | Top-k by circuit-tracer **raw** node influence (`influence_raw` when present; legacy cumulative `influence` ranked ascending). | Anthropic attribution-graph prune score (Lindsey et al. 2025), used as a selector |
| `eap` | `eap_graph` | **Graph path-effect**: Jacobi path sum on exported link weights, ±1 logit seeds. Zero model calls. | **Not** Syed/Nanda. Cheap graph-derived cousin of influence |
| `eap_syed` | `eap_ap`, `attribution_patching` | **Syed/Nanda attribution patching** on CLT **feature nodes**: `(a_corr − a_clean) · ∂L/∂a_clean`, then top-k by \|score\|. Needs clean + corrupted prompts + ReplacementModel. Production path estimates `∂L/∂a` as decoder write direction(s) contracted with residual grads at `feature_output_hook` (not via `feature_intervention`, which is `@torch.no_grad`) | Syed et al. 2023 EAP scoring formula, applied to MACAG’s feature-node universe (not native edges) |
| `acdc` | `acdc_ported` | Top-down **node** τ-prune on CLT features under MACAG coalitional `v` (zero-ablation keep/remove mix) | Conmy *selection rule* ported to CLT nodes — **not** Algorithm 1 on native edges |
| `acdc_native` | — | Same τ-prune **rule** on native heads/MLPs (`a{l}.h{h}`, `m{l}`) via `HookedComponentInterventionScorer` | Closer to Conmy’s *universe*; still node-level (not edge-level), not the ArthurConmy repo |
| `shapley` | — | MC permutation Shapley over MACAG `v` → top-k ranking | Shapley (1953) credit; gold reference, not a circuit paper |
| `banzhaf` | — | MC Banzhaf over MACAG `v` → top-k ranking | Banzhaf index; diagnostic gold |
| `game1` | — | MACAG Game 1 greedy | This paper |

## Dual-track comparison (fair use)

1. **Selector isolation (same CLT candidates + same feature oracle `v`)**  
   Compare `game1` vs `influence` vs `eap`/`eap_graph` vs `eap_syed` vs `acdc`/`acdc_ported`.  
   This is the apples-to-apples table.

2. **Behavioral / native-granularity**  
   Compare `game1` (feature circuits) vs `acdc_native` (head/MLP circuits) on faithfulness-vs-size / cost only — **not** Jaccard on node IDs.

## Example method lists

```bash
# Legacy-compatible (unchanged IDs; old JSON still valid)
--methods influence,eap,game1,acdc

# Explicit names for new runs (aliases resolve to the same code paths)
--methods influence,eap_graph,eap_syed,game1,acdc_ported,acdc_native

# Syed EAP needs corrupted_prompt in oracle kwargs or --eap-corrupted-prompt
```

## Requirements

| Method | Needs ReplacementModel | Needs `corrupted_prompt` | Scored under CLT feature oracle |
|--------|------------------------|---------------------------|----------------------------------|
| `influence` | no (graph JSON only) | no | yes (evaluation only) |
| `eap` / `eap_graph` | no | no | yes |
| `eap_syed` | **yes** | **yes** | yes |
| `acdc` / `acdc_ported` | yes (for real runs) | no | yes |
| `acdc_native` | loads base HookedTransformer from `model_name` | optional | **no** — uses component oracle |
| `shapley` / `banzhaf` / `game1` | yes | no | yes |
