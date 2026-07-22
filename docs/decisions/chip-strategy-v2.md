---
type: decision
status: in-progress
tags:
  - decision
  - chips
  - strategy
---

# Decision: Chip Strategy v2

## Problem
The original chip policy hardcoded specific gameweeks (around GW17/18/19). Like
the legacy optimizer's constants, this **memorized one calendar** and could not
transfer to other seasons or react to where doubles/blanks actually fall.

## Alternatives considered
- **Keep the hardcoded-GW policy** — kept as `legacy` for thesis ablation.
- **Rolling-horizon, calendar-agnostic scheduler** (chosen, `v2` default).

## Decision
Replace fixed gameweeks with per-chip value functions evaluated over a lookahead
plus all known future events, respecting only FPL set boundaries (Set 1 GW1–19,
Set 2 GW20–38). A `SPACING_GAP` constraint keeps WC and FH ≥ 4 GWs apart — a
structural fix for a manual FH→WC mistake. Selected via `CHIP_STRATEGY=v2`. Design
in [[chip_strategy_redesign]]; implemented in [[chip-scheduler]].

## Tradeoffs accepted
- **Not yet validated** against the 2468 baseline — a GW1–38 backtest plus
  generalization runs are pending (blocked on raw data; see
  [[environment-and-docker]]).
- In the [[milp-optimizer]] path, unanchored chips tend to burn early because
  scarcity beyond the horizon is unpriced — active work in [[chip_scarcity_fix]].

## Components affected
[[chip-scheduler]], [[season-simulator]]; interacts with [[milp-optimizer]]
in-model chip variables.

## Future work
Backtest v2 vs 2468, re-tune the Optuna bars ([[hyperparameter-tuning]]), and
resolve unanchored-chip scarcity pricing.

---
See also: [[system-overview]] · [[optimizer-redesign]] · [[season-simulation]]
