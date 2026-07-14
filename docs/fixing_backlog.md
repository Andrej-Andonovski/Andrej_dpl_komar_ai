# Fixing Backlog — What We Still Need to Add

Status: OPEN (2026-07-15). The build phases (0-4), the cross-season harness,
and the chip scarcity fix are done (see docs/phase*_report.md,
generalization_report.md, chip_scarcity_fix.md). This file is the ordered
list of what remains. Every item ships only with cross-season validation
(2023-24 + 2024-25 + 2025-26); nothing gets tuned on a single season.

Current best configs (corrected rules, Docker `fpl-sim`):
- real/future season:   `OPTIMIZER=mp MP_HORIZON=5 MP_CHIPS=model`
  (neutral-season mean 2292 — beats tuned legacy's 2261)
- 2025-26 thesis ablation point: `MP_CHIPS=legacy` (2156)

## 1. Captain channel (~50 pts vs legacy on 2025-26 — biggest single gap)

Legacy captain avg 6.21/GW vs mp 4.82-5.29. Experiments to run (evaluated
by captain-avg + captain-regret across all three seasons, not just points):
- θ sweep (0, 0.25, 0.5, 0.75) — is the q90 ceiling blend helping at all?
- κ form variants: pure μ; μ + empirical headroom; μ with a premium-anchor
  continuity bonus (the horizon can now express "own the perennial captain")
- π calibration check for premium players (are nailed starters getting
  ~0.95+?)
- Post-solve sanity: compare MILP captain vs best-μ-in-XI captain per GW.

## 2. Eventless-set unanchored chips (residual from the chip fix)

On 2025-26 (Set 1 has no doubles/blanks) wc1@5/tc1@6/bb1@9 still fire
semi-early; legacy chips beat model chips there (2156 vs 2029).
Designed candidate: **season-percentile bar** — fire an unanchored chip only
when its aux value is in the top q of weekly values observed so far this
season. Needs cross-solve memory (a small state file like the purchase
ledger). Tune q on two seasons, hold out the third.

## 3. Cross-solve churn / plan-thrash

T4 (≤1 in, ≤1 out per player) binds within one solve; weekly re-solves
re-free everything → 10-17 buybacks per season. Candidates:
- carry last solve's plan and add a small plan-consistency penalty for
  deviating from your own week-t+1 plan (blueprint R5 mitigation), or
- extend T4 memory: players sold in the last K real weeks cost an extra
  term to re-buy (dampened churn without a hard ban).
Also revisit 2023-24's −136 hit pens (34 hits navigating six blanks) —
check per-hit ROI there before deciding hit_cap changes.

## 4. Bench weight w̄ outside chip weeks

Bench waste 11/GW at H=5 vs 3.3 legacy. Partly deliberate (BB build-up,
auto-sub inventory) — but w̄=0.15 likely overprices bench in non-chip
stretches. Option: chip-aware w̄ (higher within L weeks of a planned BB).
Goes into the sweep.

## 5. Phase 6 — the final cross-season sweep

Optuna over the honest constants (H, δ, δ_c, θ, γ, w̄, MP_WC_BELOW,
hit_cap, percentile-bar q) with the train-2-seasons / hold-out-1 objective,
3 folds. Report in-fold vs out-of-fold gap as the overfit measure.
Harness exists (SIM_SEASON env); needs a small sweep driver script
(mirror optuna_search.py but multi-season objective).

## 6. Calibration re-measures (cheap, fold into any Phase 6 run)

- q90 coverage under the empirical-headroom definition (target ~0.90;
  Z90-era measurement was 0.836)
- π calibration: predicted play-prob vs realized play rate by bucket
- φ re-test with μ-matched buckets (was inverted/confounded in Phase 1)

## 7. Housekeeping / thesis

- Metrics: add captain-avg and chip-value columns to backtest_metrics.py
  summary (currently hand-computed in reports)
- Unpruned sanity solve (blueprint §7.2 valve) — run once, assert pruned
  objective within 0.1%
- Thesis ablation table: legacy vs mp × chips × seasons — all numbers
  already exist in docs/*, needs assembling
- Live-play notes: purchase-ledger state file + weekly price re-sync are
  required before using mp for a real 2026-27 season
