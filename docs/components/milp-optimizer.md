---
type: component
status: in-progress
tags:
  - component
  - optimizer
  - redesign
---

# Multi-Period MILP Optimizer

The **redesign** of the optimization layer: a multi-period Mixed-Integer Linear
Program that plans several gameweeks ahead and folds chips, captaincy, vice, and
bench value into a single program. It is an alternative to the
[[legacy-ilp-optimizer]], selected via the `OPTIMIZER=mp` environment flag, and
is still being validated (Phases 0–6).

## Responsibility
Over a rolling horizon (default `H=5`), decide transfers, starting XI, captain,
vice, bench, and chip usage to maximize discounted expected points under
**corrected FPL rules** (sell-value pricing, free-transfer banking 1..5). It is
built around three modules:
- `prediction_matrix.py` — builds the per-player, per-future-GW matrix
  (`mu`, play-probability `pi`, confidence `phi`, captain ceiling `q90`, price,
  sell value), replacing the legacy post-multipliers with feature-swap
  re-prediction and per-fixture DGW sums.
- `milp_core.py` — `solve_gw` (single week) and `solve_horizon` (multi-week with
  chips as variables), solved with **HiGHS** (CBC fallback).
- `fpl_rules.py` — pure rule accounting (50% sell-on rounding, FT banking).

## Why it exists
The [[legacy-ilp-optimizer]] optimizes one week at a time with ~25 tuned
constants, which memorizes the calendar and cannot reason about banking transfers
or reserving chips for future events. The redesign replaces those constants with
a small set of "honest" ones and lets the solver plan ahead. The full rationale
is the [[optimizer-redesign]] decision; its thesis-critical claim is
**generalization** — it travels to unseen seasons better than the tuned legacy
system ([[generalization_report]]).

## How it interacts
Reads [[prediction-models]] outputs through the prediction matrix; plugs into the
[[season-simulator]] behind `OPTIMIZER=mp` (requires `RULES_MODE=corrected` — the
[[corrected-vs-legacy-rules]] decision) and writes to a separate
`season_simulation_corrected_mp.json` so it never clobbers the production output. Chip placement is shared with [[chip-scheduler]]
(in-model chip variables plus percentile gates). Validated across seasons by the
[[cross-season-harness]].

## Depends on
- [[prediction-models]] (via `prediction_matrix.py`).
- [[feature-engineering]] (must supply features in the matrix's canonical order).

## Depended on by
- [[season-simulator]] (alternative optimizer path).
- [[cross-season-harness]] (the config it stress-tests).

## Assumptions & limitations
- **Not yet production.** Best config to date is `OPTIMIZER=mp MP_HORIZON=5`; in
  Docker it trails the tuned legacy baseline on the home season but wins on
  neutral seasons. Full status, scoreboard, and the fixing backlog are in
  [[HANDOFF]], [[phase0_baseline]], [[phase1_report]], [[phase2_report]],
  [[phase3_report]], [[phase4_report]], and [[chip_scarcity_fix]].
- Objective constants (`THETA`, `GAMMA`, `W_BENCH`, `HIT_COST`, discount `DELTA`,
  …) are documented as measured/calibrated but several are still being tuned in
  the Phase 6 sweep — treat exact values as provisional.

## Related Source Files
- `pipeline/milp_core.py`
- `pipeline/prediction_matrix.py`
- `pipeline/fpl_rules.py`
- `pipeline/phase1_calibration.py`, `pipeline/backtest_metrics.py`

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
