---
type: decision
status: active
tags:
  - decision
  - models
  - stage10
  - optimizer
---

# Decision: Stage 10 — LSTM Residual Layer (Phase 1)

## Problem
The four LightGBM models ([[prediction-models]]) are unbiased in the mean but
structurally **cannot spike**: on the walk-forward OOF, 16 % of rows are missed
hauls (`actual − μ ≥ 3`) carrying **40 %** of the total residual error mass,
2:1 skewed toward under-prediction. Their rolling player features are
within-current-season only, so they have no representation of a player's
*trajectory*. Separately, the [[milp-optimizer]]'s captaincy ceiling `q90`
(empirical headroom × √n_fix) under-covered at 0.836 vs a 0.90 target
([[phase1_report]] of the optimizer redesign).

## Alternatives considered
- **Full replacement of the GBM with a sequence model** — rejected: 6.5 k FWD /
  5.2 k GK training rows are far too few for a deep model to beat a tuned GBM
  from scratch; loses the GBM fallback.
- **LightGBM leaf embeddings as one input channel to the LSTM** — more coupling,
  harder to keep leakage-clean, no clear win over the residual framing.
- **Residual (stacked) correction** — chosen. The LSTM predicts
  `actual − μ_gbm`, a bounded mostly-zero-mean target it can only help or no-op
  on. GBM stays the base; easy A/B; low risk to the baselines.
- **Legacy vs mp injection point** — both wired behind one flag. FDR and intel
  multipliers stay where they are (`_finalize` / `build_matrix`) — the residual
  is not moved into that chain, so the LSTM sees a clean raw μ.
- **In-season head fine-tune** — built (`stage10_finetune.py`) and **rejected on
  measurement**: holdout MAE regressed on every checkpoint (2023-24 up to
  −0.054). The head overfits the thin partial-season sample.
- **Torch at runtime** — rejected. Windows Smart App Control blocked the torch
  DLLs mid-build; the resolution (a dependency-free numpy forward pass for
  inference) is kept regardless — it keeps `season_simulator` and the Optuna
  loop torch-free and is a permanent SAC hedge. Training uses torch offline.

## Decision
**Ship Phase 1 (LSTM only) for `OPTIMIZER=mp`.** A 2-layer LSTM (hidden 64,
shared trunk + position embedding) predicts the residual with a Gaussian-NLL +
`LAMBDA_R·mean(r̂²)` objective; a variance head produces a per-position
train-calibrated `q90`. Selected by `STAGE10=off|on` (default off, a true
no-op). Full evidence: [[stage10_phase1_report]].

Results (A/B, `STAGE10` off → on):

| optimizer | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| **mp** | +14 | +40 | +81 |
| legacy | −12 | +0 | +73 |

`mp` is positive on every season (mean +45) and passes the gate cleanly.
`legacy` is inconsistent — one regression (−12 on 2023-24, high-variance captain
swings) — kept **functional and documented**, but `mp` is the recommended
config. Gate 2 (residual MAE) mean +0.026, gate 3 (q90 coverage) 0.836 → 0.89.

## Tradeoffs accepted
- **The mean correction is small** on recent seasons (+0.004 OOF MAE on folds
  4–5) — the recent-season residual has little exploitable temporal structure.
  Most of Phase 1's season-score value on the deployment proxy comes from the
  q90 → captain channel, not the μ correction.
- **Mild in-sample tuning** — `LAMBDA_R` and the z-quantile were chosen partly
  on fold-5 gate numbers; the fully-blind A/B seasons are 2023-24 / 2024-25.
- **Legacy distortion** — on the legacy path the additive residual is pushed
  through ~4 downstream multiplicative transforms; a clean fix (apply `r` after
  `_finalize`) is deferred to Phase 1.5 (legacy is the retiring optimizer).
- Runtime sequence approximations (4 of 56 timestep features) — see
  `stage10_refine.py`.

## Components affected
[[stage10-residual-layer]] (new), [[season-simulator]] (`predict_pool` split +
`STAGE10` flag + `raw_mu_history`), [[milp-optimizer]] (`build_matrix` gains an
optional `resid_fn`), [[prediction-models]] (consumed via `stage10_oof.py`).
[[legacy-ilp-optimizer]] itself is untouched — only `select_captain` gained the
`CAP_Q90_W` blend.

## Future work
Phase 2 — a GNN over player/team/fixture nodes on top of the LSTM embeddings
(design in `docs/stage10_phase1_plan.md`). A held-out hyperparameter fold to
remove the in-sample tuning optimism. Legacy-path residual cleanup if legacy
outlives the [[optimizer-redesign]].

---
See also: [[system-overview]] · [[walkforward-no-leakage]] · [[optimizer-redesign]] · [[player-identity-features]]
