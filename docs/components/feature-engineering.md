---
type: component
status: active
tags:
  - component
  - features
---

# Feature Engineering

Turns the ingested tables into the four position-split training files the models
learn from. This is **Stage 6** in [[data-flow]].

## Responsibility
Build per-player, per-gameweek feature rows and split them by position into
`train_gk.csv`, `train_def.csv`, `train_mid.csv`, and `train_fwd.csv`
(~51k rows total, target column `total_points`). Feature groups documented in
[`CLAUDE.md`](../../CLAUDE.md) include rolling player form, previous-league stats
for newcomers, team/opponent form (xG-based), fixture difficulty, and market
signals, with position-specific extras (e.g. saves for GK).

## Why it exists
The predictive signal lives in engineered features (rolling windows, xG-derived
strength, previous-league priors), not in raw API columns. Splitting by position
is a hard project rule — the four models never share a table.

## How it interacts
Consumes [[data-pipeline]] outputs; produces the training files consumed by
[[prediction-models]]. The **canonical feature order** is shared with the MILP
redesign: `prediction_matrix.py` asserts its feature list matches the
simulator's, so this component effectively defines the model input contract for
both optimizers.

## Depends on
- [[data-pipeline]] (raw/processed tables).

## Depended on by
- [[prediction-models]] (training files).
- [[milp-optimizer]] indirectly — it must produce features in the same order the
  prediction matrix expects.

## Assumptions & limitations
- Feature construction obeys the temporal-integrity rules — no leakage, no
  cross-season bleed, and excluded identifier columns (`name`, `season`, `GW`,
  `team`, `opponent_team`, `position`, `was_home`, proxy flags) — defined in
  [[walkforward-no-leakage]].
- Training files are validated as 0 NaN / 0 leakage / 0 cross-season bleed per
  [`CLAUDE.md`](../../CLAUDE.md); this note does not re-verify that claim.

## Related Source Files
- `pipeline/feature_engineering_stage6.py`
- `data/processed/train_gk.csv`, `train_def.csv`, `train_mid.csv`, `train_fwd.csv`

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
