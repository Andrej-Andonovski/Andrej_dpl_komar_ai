---
type: reference
status: active
tags:
  - reference
  - glossary
---

# Reference: FPL Glossary

Domain terms used throughout the vault. Defined once here so other notes can use
them without re-explaining.

## Gameweeks & fixtures
- **GW** — Gameweek; one of the 38 scoring rounds in a season.
- **DGW (double)** — a gameweek where a team plays twice; players score across
  both fixtures. The pipeline models these as two per-fixture rows.
- **Blank (BGW)** — a gameweek where a team has no fixture; its players score 0.
- **FDR** — Fixture Difficulty Rating; how hard an opponent is. Used as a model
  feature and (in the legacy path) a post-multiplier.

## Squad & scoring
- **XI** — the 11 players who score; the other 4 are the bench.
- **Auto-subs** — bench players automatically substituted in when a starter
  doesn't play.
- **Captain / Vice** — captain's points double; vice takes over if the captain
  doesn't play.
- **Hit** — a −4 penalty for each transfer beyond the free allowance.
- **FT banking** — unused free transfers roll over, capped at 5 (2025-26 rules).
- **Sell value** — resale price of an owned player: full price if it fell,
  purchase + 50% of any rise (rounded down). Implemented in `fpl_rules.py`; see
  [[corrected-vs-legacy-rules]].

## Chips (two of each per season; see [[chip-scheduler]])
- **BB — Bench Boost** — bench points count this GW.
- **TC — Triple Captain** — captain scores ×3 instead of ×2.
- **FH — Free Hit** — a one-week temporary squad, reverted next GW.
- **WC — Wildcard** — unlimited free transfers for one GW (permanent).

## Metrics & signals
- **xG / xGA** — expected goals for / against; a team-strength signal (Understat).
- **EO — Effective Ownership** — how heavily a player is owned+captained among a
  reference group; here the top-10k (`intel_08`).
- **MAE** — mean absolute error, the primary model metric; see
  [[evaluation-metrics-and-results]].
- **Availability / Rotation risk** — 0–100 scores from [[intelligence-suite]] that
  become prediction multipliers (see [[tuned-parameters]]).

---
Hubs: [[system-overview]]
