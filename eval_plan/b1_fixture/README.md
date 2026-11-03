# B1 fixture cells + long-form export

Frozen raw JSONs for schema development (`eval_plan/macag_final_experiment_plan.md` §8.2 B1).

| Folder | Cell | Role |
| --- | --- | --- |
| `degenerate_gemma_ioi/` | `gemma2-426k` / `mib_gemma2_ioi_0071` | Frozen leg degenerate; **primary leg = unfrozen** (D2) |
| `llama_ioi/` | `llama32-524k` / `mib_llama3_ioi_0020` | Typical Llama IOI; **primary leg = frozen** |

Each folder contains: `macag_game1.json`, `macag_baselines.json`, `macag_kl_faithfulness.json`, `macag_game2_abr_b0.json`, `macag_game2_abr_b0.2.json`.

## Extractor

```bash
python experiments/extract_macag_b1_rows.py \
  --campaign-root /gscratch/$USER/paper_final/macag \
  --out eval_plan/b1_fixture/macag_b1_per_leg_rows.csv
```

Implementation: `macag/analysis/b1_rows.py` — one row per `(CLT, prompt, leg, method)` at Game 1 `|E*|`, with k-grid continuity, KL k vs `|E*|`, and `matched_k_status` checks.
