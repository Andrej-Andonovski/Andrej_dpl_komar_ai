---
type: component
status: in-progress
tags:
  - component
  - chips
  - strategy
---

# Chip Scheduler

Decides **when** to play FPL chips — Bench Boost (BB), Triple Captain (TC), Free
Hit (FH), and Wildcard (WC) — across the season. It is a decision policy layered
on top of the optimizer inside the [[season-simulator]], not a standalone stage.

## Responsibility
Choose the gameweek for each chip (two of each per season, split across the two
FPL chip sets: Set 1 GW1–19, Set 2 GW20–38) to maximize the extra points a chip
buys. Each chip has a value function: BB ≈ predicted bench sum, TC ≈ captain's
marginal points, FH ≈ temporary-XI gain on event weeks, WC ≈ rolling squad-rebuild
gain over a horizon.

## Why it exists
Chips are scarce, high-leverage, and timing-sensitive; a naive "play when
available" policy wastes them. The original policy hardcoded specific gameweeks,
which memorized one calendar. **Chip Strategy v2** (the [[chip-strategy-v2]]
decision; design in [[chip_strategy_redesign]]) replaces that with a
calendar-agnostic rolling-horizon scheduler so the policy generalizes to other
seasons.

## How it interacts
Runs inside the [[season-simulator]] each gameweek, reading the optimizer's
squad/predictions to evaluate chip value. Selected via the `CHIP_STRATEGY` flag
(`v2` default | `legacy` kept for thesis ablation). In the [[milp-optimizer]]
path, chips become **in-program variables** (event-anchored bonuses discounted
separately), and unanchored chips on plain weeks are gated by a **percentile
bar** (`chip_percentile.py`) so a chip only fires when its value clears a
historical quantile.

## Depends on
- [[season-simulator]] (runtime context and predictions).
- [[milp-optimizer]] (in-model chip variables; FH/WC value via temp solves).
- `chip_percentile.py` (persistent percentile gates for unanchored chips).

## Depended on by
- [[season-simulator]] (consumes the chip decisions).

## Assumptions & limitations
- **v2 is implemented but not yet fully validated** against the 2468 baseline;
  bars are Optuna-tunable and a `SPACING_GAP` constraint (WC↔FH ≥ 4 GWs) was
  added to prevent a known manual mistake ([`CLAUDE.md`](../../CLAUDE.md)).
- In the MILP redesign, **unanchored chips tend to burn at first eligibility**
  because scarcity beyond the horizon is hard to price — the active fix is
  tracked in [[chip_scarcity_fix]] and [[fixing_backlog]].
- FH currently triggers on doubles/blanks (event weeks); pure blank scenarios are
  a documented gap.

## Related Source Files
- `pipeline/season_simulator.py` (v2 scheduler + `CHIP_STRATEGY` flag)
- `pipeline/milp_core.py` (in-model chip variables)
- `pipeline/chip_percentile.py`
- `tests/test_chip_percentile.py`, `tests/test_milp_chips.py`

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
