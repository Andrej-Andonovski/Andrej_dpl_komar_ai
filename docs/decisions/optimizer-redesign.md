---
type: decision
status: in-progress
tags:
  - decision
  - optimizer
  - redesign
---

# Decision: Optimizer Redesign (ILP → Multi-Period MILP)

## Problem
The production [[legacy-ilp-optimizer]] optimizes one gameweek at a time and
relies on ~25 hand/Optuna-tuned constants. It cannot reason about banking
transfers, reserving chips for future events, or captain/vice inside the program,
and its constants largely **memorize the 2025-26 calendar** — so it does not
generalize to other seasons ([[generalization_report]]).

## Alternatives considered
- **Keep tuning the legacy ILP** — diminishing returns, and the generalization
  problem is structural, not a tuning issue.
- **Multi-period MILP** with chips/captain/vice/bench in one program and a small
  set of "honest" constants (chosen).

## Decision
Build a rolling-horizon (H=5) MILP: a prediction matrix
(`prediction_matrix.py`), the solver (`milp_core.py`, HiGHS), and pure rule
accounting (`fpl_rules.py`, see [[corrected-vs-legacy-rules]]). Selected via
`OPTIMIZER=mp`. Delivered in documented phases 0–6 — full plan and status in the
blueprint [[optimizer_redesign]] and [[HANDOFF]].

## Tradeoffs accepted
- **Not yet beating the tuned legacy on the home season** (2157 mp-discipline vs
  2252 legacy in Docker), but it wins on neutral seasons — the thesis-relevant
  result. See [[evaluation-metrics-and-results]].
- Higher modeling complexity; a solver dependency (HiGHS/CBC).
- Unanchored chip scarcity is still hard to price (see [[chip-strategy-v2]],
  [[chip_scarcity_fix]]).

## Components affected
[[milp-optimizer]], [[chip-scheduler]], [[season-simulator]]; validated by
[[cross-season-harness]].

## Future work
Phase 5/6 backlog ([[fixing_backlog]]): chip-scarcity pricing, captain channel,
cross-solve churn, bench-weight calibration, and the final Optuna sweep
([[hyperparameter-tuning]]).

---
See also: [[system-overview]] · [[season-simulation]] · [[cross-season-generalization]]
