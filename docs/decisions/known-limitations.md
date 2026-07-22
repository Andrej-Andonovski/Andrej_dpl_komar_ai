---
type: decision
status: active
tags:
  - decision
  - limitations
---

# Decision: Accepted Known Limitations

Canonical catalog of limitations the project **knowingly accepted** — each was
investigated, and in several cases a fix was tried and reverted because it hurt
results. Recorded here so other notes link instead of restating.

## Press-scraper popularity bias
`intel_02` reliably detects only clubs that appear as article section headers;
players mentioned inline (the documented Newcastle case — Guimarães, Schär,
Livramento, Krafth) are missed. A cross-club name-matching fallback was tried in
`intel_03` but caused cascading squad changes (wildcard at GW12 instead of GW17)
costing ~113 pts, so it was reverted. Affects [[intelligence-suite]] /
[[intelligence-gathering]]; design context in [[press_scraper_redesign]].

## Availability not used for transfer decisions
The [[season-simulator]] applies availability/rotation as prediction multipliers
but does not use `intel_03` directly to force transfers, so an injured player
missed by the scraper can persist in the squad.

## Free Hit fires only on doubles
FH triggers on event weeks (doubles/blanks) via the optimizer; a pure blank-only
scenario (e.g. AFCON) does not auto-trigger a Free Hit.

## Sell-then-rebuy churn
The single-GW [[legacy-ilp-optimizer]] has no memory of last week's transfers, so
it can sell and re-buy the same player across consecutive GWs. A sellback penalty
was tested and hurt the overall score. The [[milp-optimizer]] horizon addresses
this structurally (transfer friction / rebuy-gap).

## Environment-bound headline score
The 2468 figure is tied to the original machine's library stack; comparisons must
stay in Docker. This is documented in [[environment-and-docker]] rather than
treated as a defect.

## Components affected
[[intelligence-suite]], [[season-simulator]], [[legacy-ilp-optimizer]].

## Future work
The MILP redesign ([[optimizer-redesign]]) targets churn and chip timing; the
press-scraper redesign targets the club-header bias. The rest are documented and
accepted for the thesis.

---
See also: [[system-overview]] · [[data-flow]]
