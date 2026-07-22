---
type: decision
status: active
tags:
  - decision
  - rules
  - optimizer
---

# Decision: Corrected vs Legacy FPL Rules

## Problem
The original ([[legacy-ilp-optimizer]]) simulator took shortcuts with two FPL
mechanics: it did not price owned players at their **sell value** (50% sell-on
tax on price rises) and it approximated **free-transfer banking**. These
shortcuts inflated the achievable score relative to real FPL and made the
[[optimizer-redesign]] impossible to compare fairly.

## Alternatives considered
- **Keep legacy rules** — preserves the historical 2468 number but is not a fair
  yardstick.
- **Corrected rules** — exact sell-value ledger, real FT banking 1..5, no budget
  relaxation (chosen, behind a flag).

## Decision
Add a `RULES_MODE` flag (`legacy` default | `corrected`). Corrected mode uses a
purchase-price ledger, values owned players at sell value in the ILP, and banks
free transfers honestly. Rule accounting is isolated in `fpl_rules.py` (integer
tenths so rounding is exact) with golden tests. Corrected output is written to a
separate file so it never clobbers the production run. See [[phase0_baseline]].

## Tradeoffs accepted
- Corrected mode scores lower in absolute terms, but is the **fair baseline**
  (2252 in Docker) all redesign work is measured against — see
  [[evaluation-metrics-and-results]] and [[environment-and-docker]].
- Two code paths to maintain; the legacy path is kept for thesis ablation.

## Components affected
[[season-simulator]] (`RULES_MODE`), [[milp-optimizer]] (requires corrected mode),
`fpl_rules.py`.

## Future work
Corrected rules are the intended long-term default once the [[milp-optimizer]]
supersedes the legacy path.

---
See also: [[system-overview]] · [[season-simulation]] · [[optimizer-redesign]]
