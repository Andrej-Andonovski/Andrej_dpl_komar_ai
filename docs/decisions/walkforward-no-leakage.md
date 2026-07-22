---
type: decision
status: active
tags:
  - decision
  - validation
  - leakage
---

# Decision: Walk-Forward Validation & No Leakage

Canonical home for the project's temporal-integrity rules. Other notes reference
this instead of restating them.

## Problem
Fantasy football prediction is a time series. Standard shuffled cross-validation
leaks future information into training and produces offline scores that collapse
in live use. For a thesis whose headline is a *live* season, that would be fatal.

## Alternatives considered
- **Random k-fold / shuffled CV** — higher apparent accuracy, but leaks.
- **Walk-forward CV by season** with strict feature-time rules (chosen).

## Decision
Respect temporal order everywhere. The non-negotiable rules from
[`CLAUDE.md`](../../CLAUDE.md):

1. **GW1 blind test** — zero 2025-26 data ever enters training.
2. **No leakage** — every feature must be knowable *before* kickoff.
3. **No cross-season bleed** — rolling windows partition by season.
4. **Walk-forward validation** — fold *i* trains on seasons 1..*i*, validates
   *i*+1; never shuffle. Fold weights `[1, 1.5, 2, 2.5, 3]` favor recent seasons.
5. **Online retraining** — during a season, full retrain each GW with actuals
   appended (not incremental).

Non-feature identifiers (`name`, `season`, `GW`, `team`, `opponent_team`,
`position`, `was_home`, proxy flags) are excluded from the model matrix.

## Tradeoffs accepted
- Early folds train on little data; recent-weighted folds mitigate this.
- Online retraining makes a season run expensive (a full retrain per GW).
- Lower *apparent* validation accuracy than a leaky setup — accepted as the price
  of honesty.

## Components affected
[[feature-engineering]], [[prediction-models]], [[model-training]]; validated by
[[cross-season-harness]].

## Future work
Stable by design. New features (e.g. a fatigue/minutes proxy) must satisfy rule 2
before inclusion.

---
See also: [[system-overview]] · [[data-flow]] · [[evaluation-metrics-and-results]]
