---
type: decision
status: active
tags:
  - decision
  - models
---

# Decision: Four Separate Position Models

## Problem
Player scoring dynamics differ sharply by position — clean sheets and saves drive
goalkeepers and defenders, goals and assists drive midfielders and forwards. A
single model must average over these regimes.

## Alternatives considered
- **One shared model** with position as a feature.
- **Four separate models**, one per position (chosen).

## Decision
Train four independent models (GK / DEF / MID / FWD). This is a hard project rule
in [`CLAUDE.md`](../../CLAUDE.md): *"4 SEPARATE MODELS — one per position, never
mix."*

## Tradeoffs accepted
- Less training data per model (e.g. ~4.4k GK rows vs ~22k MID) — accepted because
  the signal separation outweighs the smaller samples.
- More models to train, tune, and serialize.

## Components affected
- [[prediction-models]] (four regressors), [[feature-engineering]] (position-split
  training files, with GK/DEF-only feature columns).

## Future work
None planned — this is treated as settled. Any change would ripple through
[[feature-engineering]], [[model-training]], and [[hyperparameter-search]].

---
See also: [[system-overview]] · [[walkforward-no-leakage]]
