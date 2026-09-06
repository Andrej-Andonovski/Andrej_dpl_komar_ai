---
type: decision
status: in-progress
tags:
  - decision
  - feature-engineering
  - prediction-models
---

# Decision: Player-Identity & Context Features

## Problem
The live prediction pool's rolling features (`form_last3`, `avg_points_per_game`,
`minutes_reliability`, etc.) are computed **only from the current live season's
played gameweeks** ([[feature-engineering]], `season_simulator.py:
load_player_history`). Early in a season this means an established multi-season
player and a rookie carry the same amount of "known form" — the model has no
feature anywhere that says "this specific player has a proven track record."
Diagnosed concretely on 2026-27 GW3 (team 7426041 live run): the optimizer
recommended selling a player with 15 pts across GW1-2 and a favourable next
fixture, because his prediction was built almost entirely from 2 gameweeks of
data and a small (n=92), high-variance (std 5.07) historical training bucket
of "expensive forwards" that dilutes his own specific track record. Full
trace in [[player_identity_features]].

## Alternatives considered
- **Raw `player_id` as a categorical feature** — rejected: doesn't generalize
  to new/transferred/promoted-club players, high cardinality risk, and the
  model would need to have seen that exact id in training with enough rows
  to matter.
- **Multi-season career-quality summary features** (prior-season PPG,
  reliability, established-seasons count) — chosen for the first slice.
  Generalizes the same way `prev_league_*` already does for non-PL
  debutants: falls back cleanly to a replacement-level default when there's
  no prior data.
- **New-manager flag** and **transfer-sentiment/ownership momentum** —
  raised alongside player-identity in the same session; scoped but not
  prioritized (see below).

## Decision
**Priority 1 (career-level player quality) implemented 2026-09-04** — see
[[player_identity_features]] for the full spec and implementation notes.
Priorities 2-3 remain design-only. Three feature groups, sequenced by
leverage and data-readiness:
1. **Career-level player quality** (priority 1, SHIPPED) — reuses the 7-season vaastav
   data the project already has; directly closes the diagnosed gap; no new
   data source needed.
2. **New-manager flag** (priority 2) — blocked on sourcing a
   managerial-change dataset (none exists in the pipeline today); needs a
   short spike before it can be scoped further.
3. **Transfer-sentiment/ownership momentum** (priority 3) — data already
   collected by `intel_01` but unused by the predictor; flagged as the
   weakest signal (reactive, not predictive) and, if built at all,
   recommended as an intel-style multiplicative penalty rather than a
   trained feature, to avoid train/serve skew (no historical series exists
   to train it on).

## Tradeoffs accepted (once §1 is built)
- Full 7-season Stage 6 retrain + walk-forward re-validation required —
  not a drop-in patch.
- `FEAT_COLS` (season_simulator.py) and `DEFAULT_FEAT_COLS`
  (prediction_matrix.py) are hand-duplicated today; extending both without
  unifying them keeps that duplication-drift risk. Worth fixing as part of
  the same change.

## Components affected
[[feature-engineering]], [[prediction-models]], [[season-simulator]] (pool
builders only — [[milp-optimizer]] and [[chip-scheduler]] are unaffected,
this is a prediction-input change, not an optimizer change).

## Future work
Data-sourcing spike for the new-manager signal; a historical
`net_transfer_momentum` backfill from vaastav's `transfers_in`/`transfers_out`
columns if the transfer-sentiment signal is ever pursued as a trained
feature instead of an intel-style penalty.

---
See also: [[system-overview]] · [[walkforward-no-leakage]] · [[optimizer-redesign]]
