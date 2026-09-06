# Stage 10 — Phase 1 Implementation Plan (LSTM residual layer)

**Status:** in progress (Phase 1 = LSTM only; GNN is Phase 2)
**Owner doc for:** the LightGBM → LSTM residual-refinement layer that sits
between the model predictions and the optimizer.

> Division of labour with the rest of `docs/`: this is a build plan (like
> `optimizer_redesign.md`). The conceptual note will be
> `docs/components/stage10-refinement.md`; the evidence layer will be
> `docs/stage10_phase1_report.md`.

---

## 0. Design decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Relationship to LightGBM | **Residual / stacked.** GBM stays the base predictor; the LSTM learns `E[actual − mu_gbm]` and `Var[·]`. | Lowest risk to the 2468 / 2252 baselines; GBM is the fallback; clean ablation. |
| Model family, Phase 1 | **LSTM only.** GNN deferred to Phase 2. | Clean ablation story (GBM → GBM+LSTM → GBM+LSTM+GNN); isolate failure causes; LSTM alone clearing the gates is a valid contribution. |
| Optimizer integration | **Both paths, behind `STAGE10=off|on`.** Mirrors `OPTIMIZER=legacy|mp`. | Legacy is the headline-number path; mp is the forward-looking / live path. |
| FDR & intel multipliers | **Left exactly where they are.** Stage 10 corrects raw GBM `mu` only; FDR + `AVAIL_MULT` + rotation tiers + `DGW_PRED_MULT` + `OWN_BOOST_GW1` + loyalty all apply *after* Stage 10, unchanged. | `STAGE10=off` must be byte-identical to baselines (hard gate). Stage 10 sees clean `mu` and never learns around the FDR hack. |
| Validation strictness | **Same rules as everything else, no exceptions.** Walk-forward by season, GW1 blind, sequences truncated at kickoff, per-GW update in-season. | Thesis-defensible; rules #1–#7 in `CLAUDE.md`. |
| Uncertainty output | LSTM emits `(residual, sigma)`; `sigma` replaces the empirical-headroom `q90` in `prediction_matrix.py`. | Directly targets the q90 coverage gap (0.836 → 0.90). |
| Training environment | **Train locally with torch; run inference in pure numpy (option D kept).** Windows Smart App Control briefly blocked the torch DLLs; once they executed during a SAC-off window SAC whitelisted them permanently (trust is sticky), so `torch==2.14.0` now runs on Windows with SAC re-enabled. `stage10_train.py` runs on the dev machine. If a torch upgrade or SAC cache clear ever re-blocks it, training moves to WSL2 — the `.npz` checkpoints keep the simulator running meanwhile. | `stage10_infer.py` / `season_simulator` / the Optuna loop never import torch (reproducibility + SAC insurance). |

---

## 1. File structure

```
pipeline/
  stage10_oof.py            # step 1: walk-forward OOF GBM predictions → residual target   [DONE]
  stage10_sequence.py       # step 2: leakage-safe per-player sequence builder             [DONE]
  stage10_model.py          # step 3: torch LSTM (training) + numpy forward (runtime)      [DONE]
  stage10_train.py          # step 3: walk-forward pretrain + gate 2/3 report (local torch) [DONE]
  stage10_infer.py          # step 3: torch-free runtime refiner (loads .npz)              [DONE]
  stage10_refine.py         # step 4/5: runtime hook  refine(pool, gw, ...) -> pool

models/stage10/
  oof_preds.csv             # name, season, GW, position, mu_gbm_oof, actual, residual, regime  [DONE]
  pretrain_<valseason>.pt   # torch checkpoint per walk-forward fold
  pretrain_<valseason>.npz  # numpy weights + norm stats — the RUNTIME path
  ft_<season>_gw<t>.pt      # in-season fine-tuned heads (cache) — step 6
  stage10_config.json       # arch, hyperparams, feature spec
  stage10_calibration.json  # NLL curves + gate 2 (MAE) + gate 3 (q90 coverage)

tests/
  test_stage10_sequence.py  # leakage + cross-season-bleed asserts (plain python)         [DONE]
  test_stage10_identity.py  # STAGE10=off reproduces baselines byte-for-byte              (step 4)
  test_stage10_shapes.py    # no NaN, tensor shapes, empty-sequence handling              (step 4)

docs/
  stage10_phase1_plan.md            # this file
  components/stage10-refinement.md   # component note (+ links from index/system-overview/data-flow)
  decisions/stage10-residual-lstm.md # ADR
  stage10_phase1_report.md           # evidence: gates, numbers, ablation

requirements.txt            # + torch==<pin>  (CPU build)
CLAUDE.md                   # Stage 10 section
```

`stage10_refine.py` is the analogue of `prediction_matrix.py` — the single seam
the rest of the system touches. Torch is imported lazily so `STAGE10=off` never
loads it (protects the reproducibility gate and keeps the Optuna loop torch-free).

---

## 2. Sequence builder (`stage10_sequence.py`) — leakage-safe

One sample = **(player, target GW `t`, target season `S`)**. The sequence is the
player's timeline of GWs `1 … t-1` **within season `S` only** (season reset —
rule #3). Cross-season continuity is carried by the static `career_*` features,
not the recurrence (same as the GBM).

- Max length 37; left-padded + masked; `pack_padded_sequence` on true lengths.
- **DGW** = one timestep, realized-outcome features = summed stats of both
  fixtures (same accumulation as `load_player_history()`), `is_dgw=1`.
- **Blank** for that club = timestep with `is_blank=1`, zeroed outcomes.
- **Absent** player at GW `g` = timestep with `did_not_feature=1`,
  `gap_since_played` incremented.
- Timestep index is the **GW number**, never the match number.

### Per-timestep feature vector (GW `g`, `g < t`) — ~36 floats

| Group | Fields | Leakage note |
|---|---|---|
| Pre-kickoff snapshot as of `g` | the 27 `FEAT_COLS` from GWs `1…g-1` | Stage-6 leakage rules |
| GBM baseline at `g` | `mu_gbm_oof[g]` | a prediction, not an outcome |
| Realized outcome of `g` | `points, minutes, goals, assists, cs, saves, bonus, bps` | **read only at timestep g** |
| Residual at `g` | `actual[g] − mu_gbm_oof[g]` | teacher signal, same rule |
| Fixture context of `g` | `was_home, fdr, is_dgw, is_blank` | pre-kickoff |
| Availability / rotation at `g` | intel_03 tier (ordinal), intel_04 rotation score | pre-deadline |
| Continuity | `gap_since_played, did_not_feature, is_first_appearance, streak_len` | pre-kickoff |
| Volatility | rolling std of points & minutes, last 5 played | pre-kickoff |
| Position | one-hot GK/DEF/MID/FWD (constant) | static |

### Query row (target GW `t`)
Not a timestep. The pre-kickoff snapshot for GW `t` (`build_rolling_pool`
output the simulator already builds) + `mu_gbm` for `t` (the **real production
online-retrained prediction** at runtime, not OOF). Output = `f(h_{t-1}, query_row)`.

### Leakage contract (asserted in `test_stage10_sequence.py`)
```
build_sequence(pid, season, t):
  - every timestep g has g < t
  - no field references season != `season`
  - the OOF prediction for season-S rows was trained only on seasons < S (+ S, GW<t)
  - mutating actual[t] does not change the returned sequence tensor
```

---

## 3. LSTM architecture (`stage10_model.py`)

Small on purpose — FWD 6.5k rows, GK 5.2k.

```
Input:  [B,L,F] timestep feats (F≈36), [B,Q] query feats (Q≈30),
        [B] lengths, [B] position id

InputNorm    LayerNorm(F), affine = stored pretrain mean/std
PosEmb       Embedding(4, 8), concat to every timestep and the query
LSTM         input F+8, hidden 64, layers 2, dropout 0.2, batch_first, packed → h_last[B,64]
QueryMLP     Linear(Q+8→64)·GELU·Dropout(0.2)
Fuse         concat(h_last, query_repr)[B,128] → Linear(128→64)·GELU·Dropout(0.2)
Head_resid   Linear(64→1)                       → r_hat (unbounded)
Head_logvar  Linear(64→1) → σ = softplus(·)+0.5
Confidence   g = min(1, n_played/6);  r_applied = g · r_hat
             σ shrinks toward position prior when n_played < 3
```

≈ 90–110k params. Shared trunk across all four positions (position via the
8-dim embedding only) — the key choice for data sparsity; ablated vs 4 separate
LSTMs in the report.

**Loss:** Gaussian NLL on the residual +
`1e-4·||θ||²`. Optional `0.1·pinball(actual, mu_gbm+r_hat, 0.9)` term if NLL
alone leaves q90 coverage short. Sample weights = season recency `[1,1.5,2,2.5,3]`
(same as Stage 7).

---

## 4. Training regime

### 4.1 Step 0 — OOF target (`stage10_oof.py`)

For every historical row we need `mu_gbm_oof` from a GBM that never saw that
row's season at/after that gameweek.

**Protocol** (matches `season_simulator`'s online-retrain weighting):
```
predict (season S, GW t) with a model trained on
   {rows: season < S}                      weight 1.0
 ∪ {rows: season == S and GW < t}           weight t   (= 1 + completed_gws)
t == 1 → historical only, no in-season rows (blind; == train_gw1_models)
```

**Phase-1 simplification — row construction.** Production uses two row-builders
(`build_hist_rows` for history from the Stage-6 CSVs, `build_retrain_rows` for
in-season from the live FPL-API `player_history.csv`). Historical seasons have
no `player_history.csv`. Phase 1 therefore uses **one consistent construction —
the Stage-6 `train_*.csv` rows — for both training and prediction.** This is a
cleaner walk-forward than the production dual-method mix, and the *runtime*
residual in `stage10_refine` is always computed against the real production
`mu` regardless of how the training target was built. A future refinement can
replay the exact harness (`build_season_inputs.py`) for 2023-25.

Config imported from `season_simulator` (never reimplemented): `FEAT_COLS`,
`LGBM_PARAMS`, `_COL_ALIAS`, `TRAIN_FILES`. Missing position-specific columns
(`prev_int_per_90` etc. for MID/FWD) → `reindex(columns=FEAT_COLS).fillna(0.0)`,
same as `build_hist_rows`.

Output `models/stage10/oof_preds.csv`. Cost: 5 val seasons × ~38 GW × 4 pos
LightGBM fits (~760 fits) ≈ 20–40 min one-time. `--quick` = per-season only
(24 fits) for dev iteration.

### 4.2 Step 1 — walk-forward pretrain (`stage10_train.py --pretrain`)

Folds — shifted one season vs Stage 7 because **2019-20 has no OOF residuals**
(no prior season to walk-forward from), so it is GBM-history-only and never a
residual-target season:

| Fold | Train | Val | Checkpoint |
|---|---|---|---|
| 1 | 2020-21 | 2021-22 | `pretrain_2021-22.{pt,npz}` |
| 2 | …2021-22 | 2022-23 | `pretrain_2022-23.{pt,npz}` |
| 3 | …2022-23 | 2023-24 | `pretrain_2023-24.{pt,npz}` |
| 4 | …2023-24 | 2024-25 | `pretrain_2024-25.{pt,npz}` |
| final | …2024-25 | 2025-26 | `pretrain_2025-26.{pt,npz}` |

(2019-20 rows still feed the GBM that produced every OOF prediction — only the
*LSTM residual target* can't be defined for that season.)

≤ 80 epochs, AdamW lr 1e-3, batch 256, cosine decay, early-stop on val NLL
(patience 8), grad-clip 1.0. Seeds fixed; `use_deterministic_algorithms(True)`;
`OMP_NUM_THREADS=1`; CPU. Norm stats computed on the pretrain split only.

### 4.3 Step 2 — in-season per-GW fine-tune

At GW `t` of season `S`: start from `pretrain_<S>.pt`, **freeze LSTM trunk +
PosEmb + InputNorm**, fine-tune **only QueryMLP + Fuse + both heads** on
season-`S` rows GW `1…t-1` (residuals vs the online GBM). ≤ 15 epochs, lr 3e-4,
early-stop on the last-20%-by-GW tail. ~3–8 s/GW. Cache to `ft_<S>_gw<t>.pt`.

Cold start: `t ≤ 3` → skip fine-tune, use pretrained head; the confidence gate
already shrinks `r_hat`. `STAGE10` is a **no-op for GW1** (blind test, empty
sequence).

### 4.4 Determinism
`STAGE10=off`: no torch import, no code-path change → byte-identical.
`STAGE10=on`: deterministic within a machine; report documents a ±1–2 season-pt
tolerance band across machines. Headline 2468/2252 numbers are `STAGE10=off`.

---

## 5. Optimizer integration

New flag in `season_simulator.py`: `STAGE10 = os.environ.get("STAGE10", "off")`.

`stage10_refine.refine(pool, gw, season, hist_lookup, avail_gws, rot_risk, base_key,
out_resid_key="mu_resid", out_sigma_key="sigma_hat")` — batched, idempotent,
no-op when off.

### 5.1 Legacy — `predict_pool()`
Split the per-player loop:
```
loop 1:  p["mu_raw"] = model.predict(feats)          # unchanged math
------   if STAGE10 == "on":
             refine(pool, gw, SIM_SEASON, ..., base_key="mu_raw")
             p["mu_raw"] += p.get("mu_resid", 0.0)
loop 2:  pred = p["mu_raw"]; then FDR / zero-min / GW1 own / blend / PRED_CAP /
         loyalty — ALL UNCHANGED, operating on the corrected value
```
`STAGE10=="off"` → the split loop must emit identical floats (gate 1).

### 5.2 MILP — `prediction_matrix.build_matrix()`
Add a parallel clean accumulator `mu_clean[pid]` (raw sum, no per-fixture FDR)
and an optional `resid_fn=None` kwarg (default → identical behavior). When set:
```
r_hat, sig = resid_fn(pid, g, mu_clean[pid])
mu  = mu + r_hat                       # added after per-fixture FDR sum
q90 = mu + Z90 * sig * sqrt(n_fix)     # learned sigma replaces empirical headroom
```
For non-DGW this is exactly "FDR then +resid"; for DGW the resid is not itself
FDR-scaled (accepted approximation — the LSTM already saw fixture difficulty).
q90 swap independently toggleable via `STAGE10_Q90=on|off` for the ablation.

### 5.3 Untouched
FDR post-multipliers, `AVAIL_MULT` / rotation tiers, `DGW_PRED_MULT`,
`OWN_BOOST_GW1`, loyalty, `PRED_CAP` / `XI_PRED_CAP`, chip logic, captain
heuristics.

---

## 6. Validation gates

| # | Gate | Pass criterion | Test |
|---|---|---|---|
| 1 | Reproducibility | `STAGE10=off` → `season_simulation*.json` byte-identical to baseline (minus `generated_at`), legacy **and** mp; torch never imported when off | `test_stage10_identity.py` |
| 2 | Residual MAE | OOF `MAE(mu_gbm + r_hat) ≤ MAE(mu_gbm)` for every (position × fold); mean improvement > 0, 95% bootstrap CI excludes 0 | `stage10_train.py --report` |
| 3 | q90 calibration | OOF coverage `P(actual ≤ q90) ∈ [0.88, 0.92]` (from 0.836); captain regret not worse than legacy 6.5–7.3/GW | `--report` + `backtest_metrics.py` |
| 4 | Full-season A/B | `STAGE10=on ≥ off` on 2025-26; no regression on 2023-24 / 2024-25; legacy and mp | 6 off + 6 on sims, `backtest_metrics.py` |
| 5 | No leakage | every timestep `g < t`; season-S OOF trained only on `< S` (+ S, GW<t); mutating `actual[t]` doesn't change the tensor | `test_stage10_sequence.py` |
| 6 | No cross-season bleed | no sequence spans a season boundary; `career_*` only via `_lookup_career` | `test_stage10_sequence.py` |
| 7 | Shape / NaN | empty seq → `r_hat=0`; all-absent → `r_hat=0`; no NaN/Inf; DGW timestep = summed stats | `test_stage10_shapes.py` |
| 8 | Determinism | two `STAGE10=on` runs, same machine → identical season total | CI script |

Ablation table for the thesis: GBM-only → GBM+LSTM(shared) → GBM+LSTM(4 nets)
→ GBM+LSTM+q90-head, across 3 seasons × 2 optimizer paths.

Tests are plain python, no pytest (matches `tests/test_milp_core.py`).

---

## 7. New data engineered (all from existing files — no new external data)

| Artifact | From | By |
|---|---|---|
| `models/stage10/oof_preds.csv` | `train_*.csv` + `season_simulator` train/predict config | `stage10_oof.py` |
| Per-timestep features | `player_history.csv` (outcomes), `build_rolling_pool` logic (snapshot), `fixtures_raw.csv`, `availability.json`, `rotation_risk.json` | `stage10_sequence.py` |
| Derived timestep fields | gap/streak/first-appearance flags, rolling std(points/minutes), `is_dgw`, `is_blank` | `stage10_sequence.py` |
| `days_rest` (optional) | `fixtures_raw.csv` `kickoff_time` deltas — verify populated first; drop if sparse | `stage10_sequence.py` |
| Norm stats | pretrain split only | `stage10_train.py` → `stage10_config.json` |

No change to `feature_engineering_stage6.py` or the `train_*.csv` schema in Phase 1.

---

## 8. Compute (CPU-only, no GPU)

| Task | Cost | Frequency |
|---|---|---|
| OOF generation | 20–40 min | one-time (rerun if `LGBM_PARAMS`/`FEAT_COLS` change) |
| LSTM walk-forward pretrain, 6 checkpoints | 15–30 min | per architecture change |
| In-season head fine-tune | ~4 min/season; ~12 min for the 3-season A/B | per A/B (cached) |
| Full A/B sim matrix (6 off + 6 on) | ~3–4 h (dominated by `season_simulator` itself) | per validation cycle |
| Runtime overhead, one live GW | < 10 s | per weekly recommendation |

First full Phase 1 result: ~1 day wall-clock, mostly the existing season sims.

---

## 9. Phase-1 risks (before the GNN)

1. **Low residual SNR** — GBM MAE 2.0–2.8 on a high-variance target; expected
   gain 0.05–0.15, maybe zero for GK/FWD. Framing: the q90 win and the
   temporal-momentum ablation are the contribution even if MAE barely moves.
2. **Data sparsity** — FWD 6.5k / GK 5.2k rows. Mitigation: shared trunk,
   ~100k params, dropout, weight decay, early stop, position embedding.
3. **OOF ↔ production GBM drift** — import `season_simulator` config, don't
   reimplement; assert internal walk-forward consistency + magnitude sanity
   (MAE in the 2.0–2.9 band from `stage7_results.json`).
4. **Early-season fine-tune noise (GW2–6)** — freeze trunk, confidence gate,
   `STAGE10` no-op for GW1 (consider GW ≤ 3).
5. **Determinism vs "bit-identical"** — only binds `STAGE10=off`; document the
   on-run tolerance; pin torch, single-thread BLAS, deterministic algorithms.
6. **DGW/blank timestep indexing** — index by GW number; explicit test case.
7. **Two injection points** — `refine()` returns pool unchanged when off; gate 1
   covers both paths; MILP change is a single optional kwarg.
8. **MILP q90 swap interaction** — independently toggleable (`STAGE10_Q90`) to
   isolate its effect from the Phase 5 captain-channel work.
9. **Cross-season harness compat** — build sequences from `hist_lookup` (season-
   agnostic), not `player_history.csv` directly; smoke-test on 2023-24.
10. **torch on Python 3.13** — verify install in a throwaway venv as step 0;
    fall back to an older pin if needed.

---

## 10. Build order (confirmation gate after each step)

1. ✅ `stage10_oof.py` — OOF residual target, distributions verified (bias ≈ 0,
   MAE matches `stage7_results.json`, 2:1 skew toward missed hauls)
2. ✅ `stage10_sequence.py` + `test_stage10_sequence.py` — 11/11 leakage tests pass
3. ✅ `stage10_model.py` (torch + numpy, parity Δ 1.9e-7) + `stage10_train.py` +
   `stage10_infer.py`. Gate 2 (MAE) mean +0.026 all folds positive; gate 3
   (q90) mean coverage 0.890, deployment fold 0.913 (from 0.836). Fixes:
   `LAMBDA_R` L2 on r_hat, MAE-based checkpoint, per-position train-calibrated z.
4. ✅ `stage10_refine.py` + `predict_pool` restructure (`_finalize` helper,
   `stage10_resid_fn`) + `test_stage10_identity.py` + `test_stage10_shapes.py`.
   **Gate 1 PASS all 4 configs:** legacy/off & mp/off byte-identical to
   pre-wiring at GW1-38 (2026-27 live AND 2024-25 full-season backtest);
   legacy/on & mp/on run clean over a full season; torch never imported.
   `STAGE10=on` writes a separate `*_s10.json`. Tests 6/6 + 6/6.
5. ✅ `build_matrix` `resid_fn` callback (mp path) — folded into step 4; gate 1
   covers mp/off byte-identical + mp/on full-season.
6. ✅ in-season head fine-tune (`stage10_finetune.py`, offline, freezes trunk) +
   gate 8. **Result: fine-tune HURTS** — holdout MAE 2024-25 −0.001..−0.008,
   2023-24 −0.003..−0.054 (head overfits the thin partial-season sample). Kept
   for experimentation, **OFF by default** (`STAGE10_FT=on` to enable, `ft_*.npz`
   gitignored). Gate 8: fine-tune re-run byte-identical; `STAGE10=on` season
   sim deterministic (numpy-only runtime) — two mp/2024-25 runs identical.
7. full A/B matrix → gate 4; write `stage10_phase1_report.md`
8. docs: component note + ADR + index/system-overview/data-flow links + CLAUDE.md
