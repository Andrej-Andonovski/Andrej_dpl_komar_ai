# Stage 10 Phase 1 — LSTM Residual Layer: Report

Status: **COMPLETE — SHIPPED for `OPTIMIZER=mp`** (2026-09-07). Legacy path
works and helps on 2 of 3 seasons but is inconsistent (one regression) —
kept functional, `mp` is the recommended config.

Plan: `docs/stage10_phase1_plan.md`. Artifacts: `pipeline/stage10_*.py`,
`models/stage10/` (`oof_preds.csv`, `pretrain_<S>.{pt,npz}`,
`stage10_config.json`, `stage10_calibration.json`), `tests/test_stage10_*.py`,
runs in `data/intel/season_simulation_*_s10.json`.

> **Post-ship LSTM refinements (2026-09-07+).** After Phase 1 shipped, a series
> of targeted LSTM fixes was worked through (`docs/stage10_phase1_plan.md` §
> "LSTM refinements"). §12 below tracks them; the gate-2/3/4 tables in §§2-3-9
> are the *shipped Phase 1* numbers and are annotated where a fix changed them.

## 1. What was built

A stacked residual layer between the LightGBM predictions and the optimizer:

```
LightGBM μ  →  LSTM(μ, sequence)  →  (r̂, σ)  →  optimizer
                                     r_applied = gate(r̂) added to μ
                                     q90 = μ + r_applied + z·σ_eff  (captain ceiling)
```

- **`stage10_oof.py`** — walk-forward out-of-fold GBM predictions under the exact
  production online-retrain protocol (`season_simulator.LGBM_PARAMS` / `FEAT_COLS`,
  weight `1+t`). `models/stage10/oof_preds.csv`, 51,120 rows. The residual
  `actual − μ_gbm` is the training target. Bias −0.0002, MAE 2.301 (matches
  `stage7_results.json`). **16 % of rows are missed hauls (`residual ≥ 3`)
  carrying 40 % of the total |residual| mass** — the asymmetric upside the LSTM
  targets.
- **`stage10_sequence.py`** — leakage-safe per-player GW-sequence builder
  (within-season, truncated at kickoff). Timestep `F=56`, query `Q=39`;
  `SAMPLE_SPEC` is the runtime contract. 11/11 leakage tests.
- **`stage10_model.py`** — 2-layer LSTM (hidden 64) + 8-dim position embedding +
  QueryMLP + Fuse + `(r̂, log σ²)` heads. **Shared trunk across all four
  positions** (position enters only via the embedding) — the data-sparsity
  choice (FWD 6.5 k rows). Two implementations that must agree < 1e-4: torch
  (`ResidualLSTM`, training) and a dependency-free `NumpyResidualLSTM` (runtime).
- **`stage10_train.py`** — 5 walk-forward folds, val 2021-22 … 2025-26. Gaussian
  NLL + `LAMBDA_R·mean(r̂²)` (ridge on the output). Checkpoint + early-stop on
  **val gated-MAE**, not NLL. Per-position `z` calibrated on the fold's training
  set (0.915 quantile of standardized residuals, floored at 1.2816).
- **`stage10_refine.py`** — the runtime hook. Rebuilds sequences from the live
  `hist_lookup`; `μ_gbm` timesteps come from `raw_mu_history` the sim accumulates
  each GW (walk-forward correct). numpy-only.
- **`stage10_infer.py`** — torch-free checkpoint loader + gate + q90.
- **`stage10_finetune.py`** — in-season head fine-tune. **Disabled** (§5).

## 2. Gate 2 — residual MAE (OOF, per fold × position)

`MAE(μ_gbm + gate(r̂))` vs `MAE(μ_gbm)`. Positive delta = improvement.

| fold | ALL | GK | DEF | MID | FWD | q90 cov (ALL) |
|---|---|---|---|---|---|---|
| 2021-22 | **+0.037** | +0.009 | +0.002 | +0.073 | +0.035 | 0.882 |
| 2022-23 | **+0.043** | +0.025 | +0.017 | +0.066 | +0.047 | 0.874 |
| 2023-24 | **+0.042** | +0.006 | +0.013 | +0.068 | +0.064 | 0.866 |
| 2024-25 | **+0.004** | +0.005 | +0.006 | +0.003 | +0.004 | 0.916 |
| 2025-26 | **+0.004** | +0.000 | +0.006 | −0.000 | +0.015 | 0.913 |
| **mean** | **+0.026** | | | | | **0.890** |

Every fold's ALL row is positive with a bootstrap CI excluding 0. **MID/FWD gain
+0.05–0.07 on folds 1–3**; the data-rich folds 4–5 the model stays conservative
near the GBM (the L2 penalty correctly not forcing signal that isn't there — the
recent-season residual has little exploitable structure). One flagged cell
(2025-26 MID −0.0003) is floating-point noise.

> **Known train/serve gap in these numbers (disclosed for the thesis).** These
> OOF MAE figures were computed on offline sequences that carry the real `o_bps`
> timestep feature (bonus-points-system score per past GW). At *inference* time
> `o_bps` was hardcoded to 0 because `load_player_history` never read `bps` — so
> the shipped Phase 1 A/B (§9) ran with an `o_bps=0` train/serve gap. **Fix 2
> (§12) closes it.** Measured cost of the gap, `o_bps=0 → real`, by checkpoint:
> 2023-24 MID **+0.037** / ALL +0.015 (the weak 3-season checkpoint leaned on
> it); 2024-25 and 2025-26 ≈ 0 (the L2-penalised model on more data doesn't use
> a single weak feature). So the §2 numbers slightly *overstate* the shipped
> inference quality on the 2023-24 fold and are accurate for 2024-25/2025-26.

## 3. Gate 3 — q90 coverage

`P(actual ≤ q90)`, target [0.88, 0.92], baseline (`prediction_matrix` empirical
headroom) 0.836.

```
2021-22 0.882 | 2022-23 0.874 | 2023-24 0.866 | 2024-25 0.916 | 2025-26 0.913
mean 0.890  — PASS. Deployment fold (2025-26) 0.913.
```

Folds 1–3 slightly undercover at the ALL level (thin training data → z
calibration runs optimistic); folds 4–5 and the mean are in band. This is the
mechanism by which Stage 10 feeds the MILP captain channel
(`kappa = π·[(1−θ)μ + θ·q90]`, θ=0.5).

## 4. Fix history (first pretrain run FAILED gate 2)

The first run: mean ALL-delta **−0.042**, 14/25 cells regressed, fold 5 diverged
(`r̂ → −68`, `σ → 177`). The residual-mean head had weak signal
(corr(r̂, residual) = GK +0.28 / DEF +0.17 / MID +0.08 / FWD +0.23) but a weak
predictor added to a noisy target *raises* MAE, and NLL was flat while MAE
diverged (σ absorbing the error). Three fixes, all principled, all kept:

1. **`LAMBDA_R = 0.4`** — L2 penalty on `r̂` (ridge on the output). Self-scaling
   (a model outputting −68 pays `0.4·68²`), shrinks the weak signal toward 0,
   structurally kills the runaway. No hard clamp.
2. **Checkpoint / early-stop on val gated-MAE**, not NLL.
3. **Per-position train-calibrated `z`** — `{GK 1.62, DEF 1.45, MID 1.57, FWD 1.56}`
   for the final fold. Decouples the MAE-optimal checkpoint from coverage level.

`LAMBDA_LV` (L2 on log-variance) was tried at 0.02, over-shrank σ, cost ~0.04
coverage — dropped.

**Ablation** (shared trunk vs 4 position-specific nets) was run once, before the
regularization fix, and showed shared ≈ position-net (a wash, tiny margins). Not
re-run post-fix; the shared trunk is the shipped choice for the small-data
strength-borrowing argument, not a measured win.

**torch ↔ numpy parity:** max |Δr| 1.9e-7, max |Δσ| 1.4e-6 on the shipped
checkpoint (`stage10_train.py --check-numpy`).

## 5. Step 6 — in-season head fine-tune: measured negative, DISABLED

From the frozen `pretrain_<S>` checkpoint, fine-tune only the head stack on that
season's GW<t residuals. Holdout MAE (pretrain → ft):

```
2024-25:  −0.001 … −0.008
2023-24:  −0.003 … −0.054   (degrades through the season)
```

The head overfits the thin partial-season sample. `checkpoint_for()` only picks
`ft_*.npz` when `STAGE10_FT=on`; default is the pretrain checkpoint. `ft_*.npz`
gitignored. `stage10_finetune.py` kept for experimentation.

## 6. Gate 1 — `STAGE10=off` is a true no-op

Byte-identical to the pre-wiring code (verified via `git stash` of the wiring,
JSON diff minus `generated_at`):

| config | off | on |
|---|---|---|
| legacy / 2026-27 live | **IDENTICAL** | runs clean |
| mp / 2026-27 live | **IDENTICAL** | runs clean |
| legacy / 2024-25 (full 38 GW) | **IDENTICAL** | runs clean |
| mp / 2024-25 (full 38 GW) | **IDENTICAL** | runs clean |

- `season_simulator` imports `stage10_refine` only inside `if STAGE10=="on"`, so
  **torch never loads** (the whole runtime chain is numpy-only).
- `predict_pool`: FDR/zero-min/GW1/blend/cap/loyalty extracted verbatim into a
  `_finalize()` helper; `stage10_resid_fn=None` → identical code path and float
  math, no `_mu_raw`/`stage10_*` keys on pool rows.
- `build_matrix`: `resid_fn=None` → byte-identical (guarded `mu_clean`
  accumulator, `resid_g = {}`).
- `STAGE10=on` writes a separate `*_s10.json`.

Tests: `test_stage10_identity.py` 7/7, `test_stage10_shapes.py` 6/6.

## 7. Gate 8 — determinism

- fine-tune re-run → byte-identical `.npz`.
- `STAGE10=on` season sim → numpy-only runtime → deterministic. Two mp/2024-25
  full-season runs identical (2419, same hash).

## 8. Captain q90 fix (legacy)

The legacy path stashed `p["stage10_q90"]` but nothing read it. Added a
`kappa` analogue in `select_captain` only (ILP and `_finalize` untouched):
`raw_pred := (1−CAP_Q90_W)·raw_pred + CAP_Q90_W·q90_rescaled`, `CAP_Q90_W=0.35`
(env-tunable). No-op when `stage10_q90` is absent (off, or GW1). Recovered the
legacy 2024-25 A/B from −10 to +0.

## 9. Gate 4 — full-season A/B (`STAGE10` off → on)

| optimizer | season | off | on | Δ | |
|---|---|---|---|---|---|
| **mp** | 2023-24 | 2190 | 2204 | **+14** | blind¹ |
| **mp** | 2024-25 | 2379 | 2419 | **+40** | blind¹ |
| **mp** | 2025-26 | 2224 | 2305 | **+81** | deployment proxy² |
| **legacy** | 2023-24 | 2266 | 2254 | **−12** | blind¹ |
| **legacy** | 2024-25 | 2398 | 2398 | **+0** | blind¹ |
| **legacy** | 2025-26 | 2124 | 2197 | **+73** | deployment proxy² |

```
mp     mean +45   (+14, +40, +81)   every season positive, grows with checkpoint quality
legacy mean +20   (−12,  +0, +73)   inconsistent; deployment season strong
```

¹ *Blind* — the checkpoint for season S was walk-forward-validated on S and the
hyperparameters (`LAMBDA_R`, z-quantile) were frozen after. 2023-24/2024-25 are
the honest generalization numbers.
² *Deployment proxy* — the 2025-26 A/B uses `pretrain_2025-26.npz`, whose
walk-forward val season *was* 2025-26 (fold 5). Leakage-clean (no 2025-26 in
training) but the hyperparameters were partly chosen looking at fold-5 gate
numbers — mild optimism. This is the closest available stand-in for a live
2026-27 run (2026-27 has no actuals yet). 2025-26 inputs built from the vaastav
repo via `build_season_inputs.py` (slightly more correct GW1 prices than the
original 2468 snapshot — not directly comparable to that headline number).

**Per-GW character (2025-26):**
- mp +81 — broad: +13…+25 stream GW25-38, 4 captain-marked wins, residual/q90
  shifted chip timing net-positively (tc1 8→5, tc2 33→26, fh2 26→31).
- legacy +73 — chips *identical* off vs on (rigid legacy chip policy); pure
  squad+captain, 3 captain-marked wins mid-season. `CAP_Q90_W` doing the work.

**Verdict.** `mp` **passes gate 4 cleanly** — all three seasons positive, no
regression, deployment season strongest. `legacy` clears the bar on 2 of 3
seasons (2024-25 +0, 2025-26 +73) but **regresses −12 on 2023-24** — the
high-variance captain swings on that calendar (GW33 +46 / GW34 −41). Legacy is
kept functional and documented; `mp` is the recommended config.

## 10. Shipping config

```
STAGE10=on OPTIMIZER=mp RULES_MODE=corrected
```

For the live 2026-27 run the refiner loads `pretrain_2025-26.npz` (trained on
2020-24, no 2026-27 data). Value delivered: **+45 pts/season mean** on backtests
(+81 on the deployment proxy) and q90 coverage 0.836 → 0.89.

## 11. Known limitations / Phase 1.5+ candidates

- **Legacy inconsistency** — the residual survives mp's clean
  `FDR(μ_clean) + r → MILP` path but on legacy is pushed through ~4 downstream
  multiplicative transforms (`_finalize` FDR, DGW ×2, availability ×, top-11
  scale) and the q90-in-heuristic is high-variance. Fixes (apply `r` after
  `_finalize`; wire q90 deeper) are Phase 1.5 if legacy is kept past the
  optimizer redesign.
- **Runtime sequence approximations** — `o_gc` timestep → 0 (hist_lookup lacks
  `goals_conceded`), slow features (`prev_*` / `career_*`) held at the current
  GW. `o_bps` was also 0 until fix 2 (§12). ~3 of 56 features; see
  `stage10_refine.py`.
- **Horizon** — only the decision GW is corrected in the mp matrix; future
  horizon GWs (g > t) keep the raw μ. A t-anchored sequence could feed all g.
- **In-sample tuning** — `LAMBDA_R` / z-quantile were selected partly on fold-5
  gate numbers. A held-out hyperparameter fold (train 2020-23, tune on 2024-25,
  report 2025-26) would remove the mild optimism.
- **GNN (Phase 2)** — player-interaction graph on top of the LSTM embeddings;
  design in `docs/stage10_phase1_plan.md` and the user's Phase 2 spec.

## 12. Post-ship LSTM refinements

Worked through in order after Phase 1 shipped; each committed and measured.

### Fix 1 — FWD σ tail (`03a32ef`)
The log-variance head blew up on a few out-of-distribution FWD/MID sequences
(`pretrain_2025-26`: 2 players — Adli, Gyökeres — raw σ to 116-130, reaching the
captain channel via kappa). Soft penalty `LAMBDA_LV_TAIL·mean(relu(log_var−3)²)`
(weight 0.03) tames folds 1-4 to raw σ ≤ 8; it **cannot** fix 2025-26 (that fold
early-stops before the penalty converges, and NLL genuinely wants big σ for two
unpredictable elite haulers). Hard backstop: `sm.apply_gate` clamps `sigma_eff`
at `SIGMA_CAP=15` — a guardrail on kappa arithmetic, applied in both the numpy
runtime and the training-time eval/z-calibration. Effective σ reaching the
optimizer is ≤ 15 every season. Gate 2 unchanged (+0.026), gate 3 0.890 → 0.892,
parity 2.3e-7.

### Fix 2 — `o_bps` train/serve gap (`9ba5082`)
`o_bps` (timestep 36/56) was 0 at inference (`load_player_history` never read
`bps`). Closed: `load_player_history` reads it (DGW-summed), `stage10_refine`
uses it, `build_season_inputs`/`data_fetcher_stage1` emit the column.
Cost of the gap (`o_bps=0 → real`), OOF MAE: **2023-24 MID +0.037** / ALL +0.015;
2024-25 & 2025-26 ≈ 0. Season A/B: **mp 2023-24 +14 → +65** (2255 vs 2190) — the
MID gain amplifies through the optimizer onto XI/captain picks. 2024-25/2025-26
expected ≈ flat (rerun pending). Leakage 11/11, boundary verified, off
byte-identical.

### Fix 3 — penalty-taker feature: RULED OUT with data
Added `penalty_taker` (1.0 = first-choice PK taker, from vaastav `players_raw`
`penalties_order`) to `FEAT_COLS`. Sourcing script kept at
`pipeline/archive/add_penalty_feature.py`. **Reverted** — negative on both axes:

- **Gate 2 MAE:** mean **+0.026 → +0.016**. GBM OOF MAE by position was
  unchanged (`goals_per_game` already encodes "high scorer"), but the 28th
  input dimension degraded the thin early folds (2022-23 +0.043 → +0.004). The
  deployment checkpoint (2025-26) barely used the feature (r̂ moved 0.016 on
  PK-taker rows vs 0.16 for the 3-season 2023-24 checkpoint) — same shrink-off
  pattern as `o_bps`.
- **Captain channel — worse.** `penalty_taker` pushed PK takers up the q90
  ranking (~6-8 places on 2025-26), but PK takers are **high-ceiling and
  blank-prone** — penalty-dependent scorers who return nothing when the PK
  doesn't come. The q90 head *already* over-promotes high-variance players
  (§ Q2 analysis: q90-top1 beat μ-top1 only 6/30 pre-fix); flagging PK takers
  amplified it. Offline captain-regret proxy (q90-pick vs μ-pick, MID/FWD):
  2023-24 −0.22, 2024-25 −0.14, **2025-26 +1.24 (worse)**.

**Structural conclusion:** penalty-taker status is a *ceiling* signal, not a
*mean* signal. Feeding it to a residual-mean model that then derives q90 from a
symmetric-ish σ over-weights the ceiling for captaincy. The right home for it is
**fix 6 — an explicit EV captain objective** `π·E[2·points | plays]` that can
value a PK taker's asymmetric upside without over-captaining the blank weeks.

### Fixes 4-7 (in progress)
4. cheap ceiling features (`max_points_last5`, `hauls_last10`, `p75_points_last8`);
5. pinball q90 head; 6. EV captain objective (see fix-3 conclusion); 7. conformal
calibration on a recent held-out slice. Each measured and either shipped or
ruled out with data, then a full A/B + gate-2/3 + captain-regret refresh.
