---
type: component
status: active
tags:
  - component
  - optimizer
---

# Legacy ILP Optimizer

The **production** squad optimizer: an Integer Linear Program built on **PuLP**
that turns predicted points into a legal FPL squad and starting XI. This is
**Stage 8** and layer 2 of [[system-overview]].

## Responsibility
Given per-player predictions for a gameweek, choose the 15-man squad, starting
XI, and captain that maximize expected points subject to FPL constraints
(budget, 2 GK / 5 DEF / 5 MID / 3 FWD, max 3 per club, valid formation). It is
invoked once per gameweek by the [[season-simulator]].

## Why it exists
Predictions alone don't pick a team — selection is a constrained combinatorial
problem. An ILP guarantees a feasible, optimal-for-the-objective squad under the
rules. This is the optimizer that produced the project's headline results, so it
remains production while the [[milp-optimizer]] redesign is validated.

## How it interacts
Consumes predictions from [[prediction-models]]; the [[season-simulator]] wraps
it with tuned constants, applies [[intelligence-suite]] multipliers to the
predictions beforehand, and layers chip logic (see [[chip-scheduler]]) around it.
`intel_06` reuses this exact optimizer to inject intel penalties between the
prediction and ILP steps with zero code duplication.

## Depends on
- [[prediction-models]] (predicted points).

## Depended on by
- [[season-simulator]] (default optimizer).
- [[intelligence-suite]] (`intel_06_optimizer.py` imports its functions directly).

## Assumptions & limitations
- **Single-gameweek horizon** — it optimizes one GW at a time with no memory of
  prior transfers, allowing sell-then-rebuy churn (an accepted
  [[known-limitations|limitation]]).
- Multi-period reasoning, in-program chips, sell-value pricing, and transfer
  banking are what the [[milp-optimizer]] adds ([[optimizer-redesign]]); the rule
  differences are the [[corrected-vs-legacy-rules]] decision.

## Related Source Files
- `pipeline/ilp_optimizer_stage8.py`
- `pipeline/intel_06_optimizer.py` (intel-penalty wrapper that reuses it)

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
