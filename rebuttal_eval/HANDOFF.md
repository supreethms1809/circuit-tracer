# Rebuttal Evaluation — Session Handoff

Context + instructions for continuing the NeurIPS rebuttal results extraction.
The authoritative spec is `paper_rebuttal_todo.md` (repo root): REQ-1..15,
blocking checks §4.1–4.6, output table schemas, 10k-char / no-links /
no-identifying-info constraints. Read its §0 before producing any
OpenReview-bound text. State below is **as of 2026-07-24 afternoon**
(published L0 anchors banked; λ-sweep job 8376 running).

## 1. Environment

```bash
# Non-interactive shells have no conda; always:
source /apps/u/opt/linux/miniconda3/25.11.1/etc/profile.d/conda.sh
conda activate ct                       # main env (torch, transformer_lens, ...)
# vLLM lives in a separate env (aarch64 source build at ~/software/vllm):
conda activate vllm_env                 # vllm 0.21.0
```

- Login node has **no GPU** (`CUDA_VISIBLE_DEVICES="" ` for CPU checks); GPU work
  goes through sbatch on partition `ai4wy-2`, account `uwyo-0002` (GH200-class,
  2×~146GB GPUs per node).
- `HF_HOME=/gscratch/ssuresh` (models under `/gscratch/ssuresh/hub`). Batch
  scripts must export it explicitly.

## 2. CRITICAL: live-run safety
> **2026-07-24 amendment.** No training is running, so the live tree is currently
> safe to edit — and the user's explicit preference is to **edit directly in the
> working tree on the current branch (`claude/silly-feistel`), NOT in worktrees or
> side branches** (fixes must be picked up by the next relaunch, which runs from
> the live tree). The hazard below is still real whenever campaigns ARE running;
> the resolution is **cancel the affected jobs first, then edit in place**. When
> cancelling, classify jobs first: `pv2-paper_*` are the paper campaigns;
> `macag-mib-*` run from a *different* tree (`~/macag/circuit-tracer`) and are
> unrelated. An unused worktree/branch `jumprelu-threshold-fix` may still exist at
> `/gscratch/ssuresh/worktrees/jumprelu-threshold-fix` — safe to delete.

Production paper campaigns run **from this working tree** with `Requeue=1` and
multi-day autoresume (`logs/slurm/wrappers/*.sbatch`). On every requeue they
re-import `spline_clt/` code and re-read their suite JSON. Therefore:

- NEVER edit `spline_clt/paper/{evaluate,reporting,runner,config}.py` or
  `experiments/paper_configs_v3/suites/*.json` in this tree while those jobs
  run (check `squeue -u ssuresh` for `pv2-paper_*` jobs).
- Such edits live on branch **`rebuttal-eval-path-edits`** in worktree
  `/gscratch/ssuresh/worktrees/rebuttal-eval-path` (commit 87fda1e: top-5/10
  fidelity, attribution timing/peak-mem, per-seed aggregation). **Merge into
  main/live tree only after the v3 campaigns finish.**
- New files under new names (this package, new suite JSONs, new sbatch
  scripts) are always safe — the suite launcher selects suites by exact name.
- Seed 303 was deliberately added to all three
  `experiments/paper_configs_v3/suites/*.json` (user-approved): the running v3
  jobs auto-start seed 303 at their next requeue. Do not revert.

## 3. Key paths

| What | Where |
|---|---|
| Results root | `/gscratch/ssuresh/results/paper/` |
| Rebuttal outputs | `results/rebuttal/` (repo-relative) |
| v3 gpt2-small checkpoints | `.../paper_v3_gpt2_small/runs/{spline,linear}_{feature,param}_match_gpt2_small_pv3/seed_*/checkpoints/*_best` |
| Submitted-era (93M) ckpts | `.../paper_gpt2_small/runs/*_feature_match_gpt2_small/seed_101/checkpoints/*_best` |
| gpt2-large v2 / qwen v2 | `.../paper_v2_gpt2_large/`, `.../paper_v2_qwen3_06b/` |
| dt-sweep (spline 768..12288) | `.../paper_v2_gpt2_small_dt_sweep/runs/` |
| Val activations gpt2-small v3 | `/gscratch/ssuresh/shared/activations/paper_v3/gpt2_small_1b/val/gpt2` |
| Val activations gpt2-large | `/gscratch/ssuresh/shared/activations/paper_v2/gpt2_large/val/gpt2-large` |
| Val activations qwen | `/gscratch/ssuresh/shared/activations/paper_v2/qwen3_06b/val/Qwen_Qwen3-0.6B` |
| New suites | `experiments/paper_configs/suites/paper_gpt2_small_{ravel,natural}_v3_fm.json` |

## 4. The rebuttal_eval package — how to run each piece

All scripts emit `<name>.json` (raw, internal) + `<name>.md` (scrubbed for
OpenReview) + `provenance.csv`. Never paste `.json` paths/values without
scrubbing; the `.md` files are the paste-safe rendering.

```bash
# §0.4 inventory (run before extracting anything new)
python -m rebuttal_eval.inventory --out-dir results/rebuttal/inventory

# §4.1-4.3 blocking checks (GPU; sbatch runs ALL checkpoints + compares)
sbatch scripts/slurm/rebuttal_check_reconstruction.sbatch     # env: N_SAMPLES, SMOKE_ONLY=1

# §4.4 parameter reconciliation (CPU, minutes)
python -m rebuttal_eval.check_params --checkpoints <ckpt...> --out-dir results/rebuttal/check_params

# REQ-7/REQ-8 compute benches (GPU)
sbatch scripts/slurm/rebuttal_compute_bench.sbatch

# Suite evals — ALWAYS shard one arm per job (see §6 gotchas):
sbatch scripts/slurm/rebuttal_suite_eval.sbatch <suite_name> 0 2
sbatch scripts/slurm/rebuttal_suite_eval.sbatch <suite_name> 1 2

# Auto-interp (REQ-4): server first, then the run job
sbatch scripts/slurm/launch_vllm.sbatch        # default Qwen2.5-72B-Instruct, TP=2; writes results/rebuttal/vllm_endpoint.json
sbatch scripts/slurm/rebuttal_autointerp.sbatch  # env: N_FEATURES (200)

# Published hub CLT RAVEL anchors (compare top-1/KL to our gpt2 suites)
sbatch scripts/slurm/ravel_hub_eval_run.sbatch \
    experiments/paper_configs/suites/paper_v3_ravel_hub_llama32.json
sbatch scripts/slurm/ravel_hub_eval_run.sbatch \
    experiments/paper_configs/suites/paper_v3_ravel_hub_gemma2.json
# Outputs: /gscratch/ssuresh/results/paper/ravel_eval_suite_v3_hub_{llama32,gemma2}/

# Name features that appear in RAVEL graphs (targeted IDs + clerp writeback).
# Corpus stays wikitext-2 val+test; only the feature sample is graph-derived.
python -m rebuttal_eval.autointerp.name_graphs \
    --graphs-dir <ravel_suite>/runs/.../evaluation/graphs \
    --checkpoint <ckpt> --model gpt2 \
    --out-dir results/rebuttal/autointerp_ravel_<label> \
    --backend openai_compat \
    --endpoint-file results/rebuttal/vllm_endpoint.json \
    --label <label>
# Outputs: feature_list.json, collection.json, explanations.jsonl,
# feature_scores.jsonl, autointerp_*.json/.md, annotated_graphs/*.json
# (clerp + node.autointerp scores). Use --skip-score for explain-only;
# --include-scores-in-clerp to append det/fuzz to the visible label.

# Scorer-matched re-score (REQ-4 vs literature). Published numbers all use
# Llama-3.1-70B, so absolute values are only comparable under that scorer:
VLLM_MODEL=meta-llama/Llama-3.1-70B-Instruct sbatch scripts/slurm/launch_vllm.sbatch
sbatch --dependency=after:<vllm_jid> scripts/slurm/rebuttal_autointerp_llama70b.sbatch
# Reuses each arm's collection.json (identical feature sample => paired vs Qwen).
# Override OUT= to read/write a different tree's results/rebuttal.
# Re-scoring without re-collection: add --skip-collect (collection.json is cached);
# explanations.jsonl / feature_scores.jsonl are resume caches — delete to redo.

# W&B GPU-hours (REQ-7): needs entity/project + run ids or *_training_state.pt files
python -m rebuttal_eval.wandb_pull --entity <e> --project <p> \
    --state-files /gscratch/ssuresh/results/paper/paper_v3_gpt2_small/runs/*/seed_*/checkpoints/*_training_state.pt \
    --out-dir results/rebuttal/wandb

# Final assembly
python -m rebuttal_eval.gap_report --results-dir results/rebuttal --out-dir results/rebuttal/gap_report
python -m rebuttal_eval.check_consistency <final_rebuttal_drafts>.md   # §4.6

# New benchmark generators (safe to re-run; new suite names only)
python scripts/generate_ravel_suite.py --suite-name X --spline-checkpoint P --linear-checkpoint P \
    --d-transcoder N --val-cache-dir D --output experiments/paper_configs/suites/X.json [--unpruned]
python -m rebuttal_eval.gen_natural_suite --template <suite.json> --suite-name Y --output <path>

# Tests
python -m pytest tests/test_rebuttal_eval.py tests/test_rebuttal_autointerp.py -q
```

## 5. Results already banked (results/rebuttal/)

- `check_reconstruction/`: 11 checkpoint/seed combos, **all PASS**; compare
  tables in `compare_{spline,linear}_fm/`. Headlines:
  - Submitted spline (93M): cos 0.17, **25.7% negative-cosine positions**;
    converged (1B): cos 0.814, frac<0 ≈ 1e-4. Diagnosis = genuine
    undertraining/anti-alignment, NOT a metric bug (per-position bound
    `rel ≥ √(1−c²)` holds on every checkpoint).
  - The paper-form inequality (Frobenius rel_err ≥ 1−mean cos) is norm-weighted
    and fails even on healthy converged linear/qwen models → it is NOT a bug
    detector; say so in the rebuttal.
  - §4.3: b_dec-only explains ~nothing everywhere (varExpl ≈ −0.12); converged
    spline FM varExpl 0.936 vs mean predictor. Confound ruled out.
  - Cross-model: gpt2-large spline varExpl 0.70 vs linear 0.57; **qwen linear
    0.954 beats spline 0.932** (report honestly).
- `check_params/`: every KAN checkpoint reconciles EXACTLY with the 3-term
  formula (spline_scaler enabled) → paper Eq. 13 must be corrected.
  Pairing confirmed: spline PM 846.8M ≈ linear FM 849.7M; spline FM 1868.9M ≈
  linear PM 1867.5M.
- `dtype_ablation/`: bf16 inference ≤ 4e-5 NMSE delta → fp32 is a
  TRAINING-time requirement only (grid-update lstsq).
- `inference_bench/`: spline dt12288 15.4ms vs linear 7.3ms per 128-tok window
  (2.1×), peak 7.5 vs 3.5 GiB; gpt2-large and qwen tables present.
- `attr_scaling/`: spline 0.6→6.4 s/prompt over dt 768→12288; linear FM
  1.9s/25.2GiB (7171 active), linear PM 6.6s/73.5GiB (23819 active); spline
  dt12288 10.9GiB (1542 active). **Pooled cost-vs-active-features fit has
  R²=0.34 — do NOT claim "cost tracks active features, not d_t" in pooled
  form.** Time scales with d_t; MEMORY tracks active features (spline wins).
- `autointerp_sota_comparison.md` (**REQ-4 discussion + REQ-9 anchors**, new
  2026-07-24): literature anchors and why our numbers are not yet comparable to
  them. Headlines: **no published auto-interp scores exist for GPT-2 small, and
  none exist for cross-layer transcoders at all** (CLT-Forge arXiv:2603.21014
  reports reconstruction only) — that absence is itself the answer to R1 per
  `paper_rebuttal_todo.md` §2.6. Scale-matched anchor = per-layer transcoder on
  Pythia 160M, detection 0.787 / fuzzing 0.854 (arXiv:2501.18823, Llama-3.1-70B
  scorer). Method floors on the same balanced protocol (arXiv:2410.13928):
  random 0.51, top-k neurons 0.53-0.62, human 0.74-0.75 — our spline arms clear
  the neuron band, the linear arms sit inside it. **Do not report the latter as
  a property of linear CLTs** — see `DIAGNOSIS_linear_wont_sparsify.md` (θ²
  trap); every internal spline-vs-linear autointerp delta is confounded by it,
  and re-scoring does NOT fix that (it fixes only the literature comparison).
  Open sub-item flagged in that file: the L0 column disagrees between
  `DIAGNOSIS_linear_wont_sparsify.md` and
  `autointerp_capacity_matched_comparison.md` by more than the known 12×
  convention factor, and even reverses FM/PM ordering — recompute from one
  source before publishing.
- `autointerp_{spline,linear}_fm_v3/` + `autointerp_matched_comparison.md`
  (REQ-4, **complete**, Qwen2.5-72B-Instruct scorer; superseded for
  literature comparison by the Llama-70B re-score, jobs 8366/8367):
  spline detection 0.663±0.15 / fuzzing 0.625±0.15 (172 feats); linear
  0.579±0.12 / 0.539±0.10 (196 feats). Frequency-matched paired control
  (n=43): fuzzing Δ +0.074 [CI +0.016,+0.133] significant; detection
  Δ +0.048 [CI −0.010,+0.104] directional. Quote the matched numbers
  alongside the headline ones — never the headline alone.
- `inventory/`: full checkpoint/suite-output inventory.

## 6. Gotchas discovered (do not rediscover these)

1. **KAN model construction is slow** (curve2coeff lstsq per layer at init,
   ~30+ min at dt=12288). `rebuttal_eval.common.load_transcoder` patches
   `KANLinear.reset_parameters` to zero-init (weights are overwritten by the
   checkpoint load anyway). Any new loader must do the same.
2. **Suite evals must be sharded one arm per job**: the runner writes
   `prompt_metrics.jsonl` only at END of each (variant, seed) arm — a 24h
   timeout mid-arm loses the whole arm. `--worker-id w --num-workers 2`
   partitions by (variant, seed).
3. **vLLM on GH200**: this source build selects the DeepGEMM FP8 backend on
   Hopper and aborts; `VLLM_USE_DEEP_GEMM=0` (set in launch_vllm.sbatch) fixes
   it. The endpoint file is removed by the server job's exit trap; the
   autointerp client (`resolve_endpoint`) raises if the advertised slurm job
   is dead, waits/retries while it's alive.
4. **Two L0 conventions in outputs**: `check_reconstruction` reports
   mean-per-layer-per-token; `autointerp/collect` reports summed-over-layers
   (per-token total, = 12× the other on gpt2-small). Label whichever you
   paste. The spline↔linear L0 gap is ~200× → the L0-matched autointerp
   subsample control (§2.3) is mandatory.
5. **Submitted MSE convention mismatch**: cached submitted linear-FM MSE 44.7
   vs 298.5 recomputed on the clean v3 holdout (different val set /
   convention). The REQ-2 delta table must state this; do not mix them.
6. **RAVEL wording**: no disentanglement (cause/isolate) eval exists — the 600
   prompts are a corpus for replacement fidelity. Say "600 prompts drawn from
   RAVEL" (§2.4), and keep this absence in the gap report.
7. Natural-text suite entries use `family:"factual"` + `split:"natural_heldout"`
   (avoids widening the config Literal in the live tree; the split field
   carries the designation).
8. `scrub()` output shows `$RESULTS/results/paper/...` (slightly redundant but
   non-identifying) — acceptable.

---

# SESSION UPDATE — 2026-07-24 (afternoon): published L0 anchors + λ-sweep launched

## G. Published linear CLT L0 measured (HANDOFF §D.1 DONE)

Script: `rebuttal_eval/measure_ref_clt_l0.py` + `scripts/slurm/rebuttal_ref_clt_l0.sbatch`.
Corpus: 64 long wikitext-2 test lines, max_len 128, seed 101. Convention =
`check_reconstruction` L0/layer/token. Outputs in `results/rebuttal/ref_clt_l0/`.

| Source | Act | d_t | L0/layer/tok | L0/tok total | Density/layer |
|---|---|---:|---:|---:|---:|
| `mntss/clt-llama-3.2-1b-524k` | JumpReLU (heterogeneous θ) | 32768 | **16.6** | 265 | 0.051% |
| `mntss/clt-gemma-2-2b-426k` | ReLU (no threshold keys) | 16384 | **12.6** | 327 | 0.077% |
| OUR linear FM v3 s101 (val) | JumpReLU frozen θ=0.00105 | 12288 | **2368** | 28420 | 19.3% |
| OUR spline FM v3 s101 (val) | JumpReLU | 12288 | **11.1** | 133 | 0.090% |

**Takeaway.** A healthy published linear CLT sits at **~13–17 active features per
layer per token**. Our linear is **~143× denser** than llama; our spline is already
inside the published band. Gemma hits the same sparse regime with *plain ReLU*
(no learned thresholds at all) — so a properly trained linear encoder + sparsity
penalty is sufficient; JumpReLU is not the only path.

Llama thresholds remain the smoking gun for "trained vs frozen": 133–501 unique
values/layer, median θ 0.003–0.112. Ours: exactly 1 unique value.

Gotchas fixed while building the measurer (do not rediscover):
1. `sbatch` from the agent shell cannot reach the slurm controller — wrap as
   `bash -lc 'sbatch ...'`.
2. `conda activate ct` can silently miss on compute nodes → pin
   `${HOME}/.conda/envs/ct/bin/python`.
3. `HF_HUB_OFFLINE=1` breaks transformers' mistral-regex `model_info` ping even
   with a warm cache. Load HF weights via local snapshot + `hf_model=` +
   `local_files_only=True`; keep the hub *name* for TransformerLens config mapping.
4. Wikitext via `datasets.load_dataset` hits a non-writable lock under
   `/gscratch/.../datasets/` → read the arrow file directly
   (`Dataset.from_file(...)`).

## H. λ-sweep probe LAUNCHED (job 8376 → FAILED → relaunched)

Suite: `experiments/paper_configs_v3/suites/probe_lambda_sweep_gpt2_small.json`
Wrapper: `logs/slurm/wrappers/probe_lambda_sweep.sbatch`
Output: `/gscratch/ssuresh/results/paper/probe_lambda_sweep_gpt2_small/`

**8376 FAILED** after 14 min with **SIGBUS (exit −7)** during chunk-1 collect.
Root cause: `collection_chunk_n_tokens=50M` × gpt2-small bf16 in+out
(36 KiB/tok) ≈ **1.7 TiB**, past the ~894 GiB node `/lscratch`.

**Relaunch 8378** (8377 cancelled mid-flight): **10k steps**, **20M-token chunks**,
**520M-token budget** (~26 chunks; sized for ~10k steps at ~32% dedup keep),
warmup 1000, wall 24h. Same four λ arms. Monitor `squeue -j 8378`.

| Variant | λ |
|---|---:|
| `linear_fm_lambda_0p002` | 0.002 (v3 control) |
| `linear_fm_lambda_0p01` | 0.01 |
| `linear_fm_lambda_0p05` | 0.05 |
| `linear_fm_lambda_0p2` | 0.2 |

**Success criterion:** ≥1 arm with L0/layer/token in **~15–50** and cosine ≳ 0.5.
That arm's λ becomes the linear baseline for the v3/v2 relaunch. If *no* λ reaches
the band (tanh saturation), next lever is θ_init from the target-L0 preactivation
quantile (HANDOFF §C), not higher λ.

Monitor: `squeue -j 8378`; `tail -f logs/slurm/probe_lambda_sweep_8378.err` (tqdm);
per-arm `stats/l0_*` and `stats/threshold_*` in W&B project `spline-clt-paper-v3`
or the run `training_records.jsonl`.

## I. Next session priority

1. Read λ-sweep L0/cosine frontier once **8378** finishes; pick λ (or escalate to θ_init).
2. Update v3/v2 linear model JSONs with the chosen λ (+ confirm threshold defaults).
3. Relaunch cancelled campaigns.
4. Optionally plumb `threshold_init` / `jumprelu_bandwidth` into `TrainingSettings`
   so suite JSON can override without relying on TrainConfig defaults.

---

# SESSION UPDATE — 2026-07-24 (morning; read before afternoon §G–I above)

**Nothing is training. All three v3/v2 production campaigns were CANCELLED
(user-directed) and are to be relaunched after the linear baseline is fixed.**

## A. BLOCKING FINDING: the linear baseline never sparsified

Full writeup: `results/rebuttal/DIAGNOSIS_linear_wont_sparsify.md`.

v3 gpt2-small seed 101, clean holdout:

| | spline FM | spline PM | linear FM | linear PM |
|---|---|---|---|---|
| cosine | 0.814 | 0.798 | **0.487** | **0.445** |
| rel_fro | 0.253 | 0.262 | 0.335 | 0.370 |
| NMSE | 0.064 | 0.069 | 0.112 | 0.137 |
| **L0/layer/token** | 11.1 | 9.9 | **1,501** | **3,446** |

The linear arms are nearly dense yet reconstruct *worse*. Root cause, fully pinned:

1. `jumprelu.backward` gives `threshold_grad ∝ θ/bandwidth`; `kan_transcoder.py`
   stores **log θ** and passes `exp(logθ)`, so the chain rule adds a second θ →
   **dL/d(log θ) ∝ θ²/bandwidth**. `JumpReLU(...)` was constructed with **no
   bandwidth arg → default 2**. Measured `exp_avg` = **1.9e-11**.
2. **Adam did not normalize it back**: measured **√v = 1.9e-11 ≪ eps = 1e-8**, so
   the update was eps-dominated and damped ~530×. (This is the step that makes
   the θ² suppression actually matter — Adam is otherwise scale-invariant.)
3. NOT an equilibrium: `|m|/√v = 0.87`, 100% of entries > 0.5 (healthy params:
   W_enc 0.21, b_enc 0.39). `exp_avg` uniformly **negative** → the learned signal
   pushes θ **down**; the tanh penalty saturates and never pushes it up.
4. **AdamW `weight_decay=0.01` on log θ** (one group, no exclusion) was the ONLY
   mover: predicted Δlogθ 0.0501 at lr=5e-5 vs **observed 0.0497** (0.8% match).
   Explains the 7-significant-figure uniformity across all 147,456 features.

**External confirmation (`mntss/clt-llama-3.2-1b-524k`, a properly trained linear
CLT):** thresholds are heterogeneous — **165–501 unique values per layer**
(median θ 0.003–0.11, all > 0). Ours has **exactly 1 unique value**. Scale-free
proof the threshold never trained.

## B. Fixes applied (live tree, branch `claude/silly-feistel`, NOT committed)

- `spline_clt/training/train.py`: `threshold_init` 0.001→**0.01**; new
  `jumprelu_bandwidth=0.001`, `threshold_weight_decay=0.0`,
  `threshold_adam_eps=1e-15`; log θ split into its **own AdamW group**
  (`create_optimizer(..., threshold_params=...)`, `_build_optimizers`).
- `spline_clt/kan_transcoder.py`: `jumprelu_bandwidth` plumbed into `JumpReLU(...)`.
- `spline_clt/training/train.py`: **added `stats/threshold_{mean,median,max}`
  logging** — the threshold was never logged, which is how a whole campaign ran
  with a dead gate unnoticed. Keep this.
- Verified: threshold lands in group 1 (`wd=0, eps=1e-15`), θ-gradient **441×
  larger**. Tests pass: `test_kan_transcoder/test_kan_encoder` (40),
  `test_paper_config/test_paper_runner/test_replacement_model_alignment/test_graph` (9).

## C. CRITICAL: the fixes above are necessary but NOT sufficient

Measured on the trained linear encoder against the clean holdout (bf16 bits →
must `.view(torch.bfloat16)`, the int16 npy is **not** quantized):

- current θ=0.00105 → L0 ≈ 885–1,190/layer (logged 1,501; train-vs-val sample gap)
- θ needed for **L0 ≈ 11/layer** (spline-matched) = **9.3–12.6** (~3.0–3.7σ of
  pre-activations), i.e. **~10⁴× larger**, ≈ **9 nats** of travel in log θ.
- Adam moves log θ by at most ~lr/step → 29,000 × 0.87 × 5e-5 ≈ **1.3 nats**.
  **~5–7× short.** Threshold learning can NEVER be the sparsity mechanism at
  lr=5e-5. **θ_init is the lever, not threshold-lr** (and raising threshold-lr is
  a gamble: the measured θ-gradient points toward *denser*).

## D. User's framing + agreed next step

The user's position (important, follow it): *the concern is that the linear
baseline is not sparsifying enough to be a good baseline — NOT that linear must
match spline.* Linear failing to reach spline's sparsity/fidelity is a legitimate
result; a linear baseline sitting at an accidental frozen-θ operating point is not.

**Agreed plan: calibrate against pretrained reference linear CLTs in the HF cache**
(`HF_HOME=/gscratch/ssuresh`), all fully downloaded:

| repo | base model | layers × feats | size |
|---|---|---|---|
| `mntss/clt-llama-3.2-1b-524k` | meta-llama/Llama-3.2-1B | 16 × 32768 | 20G |
| `mntss/clt-gemma-2-2b-426k` | google/gemma-2-2b | 26 × ~16k | 27G |
| `mntss/clt-gemma-2-2b-2.5M` | google/gemma-2-2b | 26 × ~96k | 160G |

Snapshot layout: `W_enc_{L}.safetensors` (contains `W_enc_L`, `b_enc_L`, `b_dec_L`,
`threshold_L`), `W_dec_{L}.safetensors`, `config.yaml`. Load with
`circuit_tracer.transcoder.cross_layer_transcoder.load_clt(<snapshot_dir>)`.
NOTE: the gemma repos do **not** expose `threshold_0` under that name — inspect
their key naming before reusing the llama script.

**TODO (next session, in priority order):**
1. ~~Measure L0/layer/token of the reference CLTs~~ → **DONE**, see §G
   (llama 16.6, gemma 12.6). Our linear is 143× too dense; spline already matches.
2. ~~Compare our linear to that~~ → **DONE** (§G table).
3. Tune the linear arm toward that L0. **λ-sweep LAUNCHED** as job 8376
   (§H); read the frontier next. If λ alone cannot hit ~15–50, set θ_init from
   the target-L0 preactivation quantile (§C).
4. Then relaunch the v3/v2 campaigns.

**Probe infrastructure already built and validated (reusable):**
`experiments/paper_configs_v3/suites/probe_threshold_fix_gpt2_small.json` +
`logs/slurm/wrappers/probe_threshold_fix.sbatch` — cold-start (resume disabled),
train-only, 3000 steps, isolated output dir. Two probes (8368, 8369) were launched
and cancelled: 8368 ran the spline arm first (wasted time), 8369 tested
θ_init=0.01 which section C proves is ~1000× too small. **Do not re-run the probe
as-is** — set θ_init from data (target-L0 quantile) or drive sparsity via λ first.
Also note: dedup drops ~68% of collected windows (`collected=156288 → train=50657`),
so 60M tokens ≈ 1,200 steps, not 3,000 — size `n_tokens` accordingly.

## E. Consistency caveat that must propagate

The capacity-matched autointerp result finalized this session
(`autointerp_capacity_matched_comparison.md`) scores these same never-sparsified
linear checkpoints. Its header now carries the caveat. **Any spline-vs-linear
claim from v3 checkpoints — reconstruction, autointerp, faithfulness — is
confounded until the linear baseline is retrained.** Keep the REQ-9 published
linear-CLT anchor ready as a fallback claim in case the retrain doesn't converge
inside the rebuttal window.

## F. Also completed this session (durable, not blocked)

- **Capacity-matched pairing corrected.** The right spline-vs-linear comparison
  holds **parameters** fixed, which cross-pairs the arms:
  spline FM (dt12288, 1.87B) ↔ linear PM (dt27008, 1.87B); spline PM (dt5568,
  0.85B) ↔ linear FM (dt12288, 0.85B). Same-d_t pairing gives spline 2.2× the
  params and is what a reviewer objected to. Use this pairing EVERYWHERE
  (compute, autointerp, REQ-5).
  - `results/rebuttal/compute_capacity_matched.md` — at matched params spline
    attribution is ≈equal time and **4–7× less peak memory**; raw forward 1.3–1.6×
    time at ≈equal memory. (Reviewer's compute question is about **inference**,
    not training GPU-hours.)
  - `results/rebuttal/autointerp_capacity_matched_comparison.md` (+ `.py`,
    `_results.json`) — fuzzing favors spline significantly in BOTH pairs
    (+0.150 [+.083,+.217] at 1.87B; +0.072 [+.027,+.118] at 0.85B); detection
    +0.065 (sig) at 1.87B, −0.009 (null) at 0.85B. Reimplementation reproduces
    the published same-d_t FM–FM row exactly (n=43, det +0.048, fuz +0.074).
- **W&B coords found** (do not re-derive): entity `uwyo`; project
  `spline-clt-paper-v3` (v3 campaigns) and `spline-clt-neurips` (v2). API authed
  via `~/.netrc`. Per-run GPU count = `SLURM_NNODES × 2` (the multinode launcher
  runs ONE torchrun spanning all nodes). Training GPU-hours is **not** what the
  reviewer asked for — deprioritized.
- gpt2-small autointerp is complete for **all four** variants (200 feats each).
  gpt2-large **spline** done (detection 0.684±0.18 n=166, fuzzing 0.649±0.16
  n=192); gpt2-large linear was mid-run and qwen not started when phase 2 was
  interrupted — re-check `results/rebuttal/autointerp_*`.

---

## 7. In flight as of handoff — STALE, see SESSION UPDATE §A–F above
(The job table below is from 2026-07-23. As of the 2026-07-24 update the three
production campaigns 8346/8347(→8364)/8348 were CANCELLED and no training is
running. Autointerp/vLLM job ids also changed. Re-check `squeue -u ssuresh`.)

| Job | What | Note |
|---|---|---|
| 8358/8359 | RAVEL v3 suite, spline/linear arms | ~10-13h each; on completion aggregate lands under `/gscratch/ssuresh/results/paper/paper_gpt2_small_ravel_v3_fm/` |
| 8360/8361 | natural-text suite, spline/linear arms | ditto, `paper_gpt2_small_natural_v3_fm/` |
| 8362 | vLLM 72B server (relaunched for phase 2) | DONE — cancelled 2026-07-24 09:19 once phase 2 finished; replaced by 8366 |
| 8363 | **autointerp phase 2** (`scripts/slurm/rebuttal_autointerp_phase2.sbatch`) | COMPLETED — six remaining checkpoints: gpt2-small v3 PM pair, gpt2-large v2 pair (base `gpt2-large`), qwen v2 pair (base `Qwen/Qwen3-0.6B`); outputs `results/rebuttal/autointerp_{spline,linear}_pm_v3`, `..._fm_gpt2l`, `..._fm_qwen` |
| 8366 | **vLLM Llama-3.1-70B-Instruct server** (TP=2) | scorer-matched re-score; weights cached at `/gscratch/ssuresh/hub` (132G) |
| 8367 | **autointerp Llama-70B re-score** (`scripts/slurm/rebuttal_autointerp_llama70b.sbatch`) | all EIGHT arms re-explained+re-scored under the literature's scorer; reuses each arm's `collection.json` so the feature sample is identical → Qwen-vs-Llama is a PAIRED comparison; outputs `results/rebuttal/autointerp_llama70b_<label>/` |
| 8346/8347/8348 | production v2-large / v3-qwen / v3-small campaigns | do not disturb; seed 303 auto-starts on requeue |

Monitors do NOT carry across sessions: a new session must check
`squeue -u ssuresh` and `tail logs/slurm/rebuttal_*_<jobid>.out` directly.
Phase-1 autointerp (gpt2-small FM pair) is DONE and banked. If the vLLM server
dies mid-phase-2, relaunch `launch_vllm.sbatch` and resubmit
`rebuttal_autointerp_phase2.sbatch` — every phase resumes from its caches
(`collection.json`, `explanations.jsonl`, `feature_scores.jsonl`; delete these
only to force a redo).

### After phase 2 lands (per model pair)
Repeat the frequency-matched paired comparison exactly as in
`results/rebuttal/autointerp_matched_comparison.md` (nearest-neighbor on log10
activation frequency, |Δlog10| ≤ 0.5, without replacement; paired bootstrap
2000 resamples seed 101, computed from the two `feature_scores.jsonl` files)
and append per-model rows. Never quote headline autointerp numbers without the
matched control beside them. Expectations: gpt2-large spline is extremely
sparse (L0 ~1.6/layer) so many features may fail the minimum-context
threshold — report the reduced N, don't hide it; qwen is the pair where linear
wins reconstruction, so treat its autointerp outcome as an open question and
report whatever comes out.

## 8. Remaining work, in priority order
### (SUPERSEDED — the linear-baseline retrain in SESSION UPDATE §D now gates
### items 1–3 and 6 below, since every spline-vs-linear number depends on it.)

1. Autointerp phase 2 (job 8363) → complete the §2.3 table (8 rows: 4
   gpt2-small variants + large pair + qwen pair) + per-model matched controls.
2. RAVEL + natural suite aggregates → REQ-5 table ("600 prompts drawn from
   RAVEL" + natural rows + original-20 rows from existing
   `neurips_core` outputs for continuity).
3. `gap_report.py` over everything; `check_consistency.py` over final drafts.
4. `wandb_pull.py` needs the W&B entity/project from the user (never guessed).
5. REQ-9 published-anchor table (literature numbers; needs web search or user
   input — nothing in-repo).
6. When v3 campaigns + seed 303 finish: rerun the headline table with 3 seeds
   via the per-seed aggregation (on branch `rebuttal-eval-path-edits`; merge
   it first), and re-run suite evals against seed-202/303 checkpoints.
7. Deferred as future work in the rebuttal: REQ-12/13/14/15 (§4.5 W_in
   spectrum etc.).
