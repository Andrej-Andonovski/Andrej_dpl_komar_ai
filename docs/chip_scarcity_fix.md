# Chip Scarcity Fix — A/B Report (fixing stage, step 2)

Status: **SHIPPED — validated cross-season** (2026-07-14).
Fixes the Phase 4 regression (docs/phase4_report.md).

## The fix (three mechanisms, none season-tuned)

1. **Cold-start lockout** — no chip variables at GW ≤ `CHIP_LOCKOUT` (4).
   Kills the tc1@GW1 / wc1@GW2 burns.
2. **Reservation guard extended to TC** (was BB-only): hold BB/TC on
   non-double weeks while a known DGW lies beyond the horizon in the same
   set; disarms automatically at the set deadline.
3. **WC squad-state gate**: wildcard eligible only when ≥ `MP_WC_BELOW` (4)
   owned players sit below their position's replacement level (25th pct of
   the top-40 matrix μ — measured each solve, not tuned), or the set
   deadline is inside the horizon.

Implementation: `chip_state` gains `lockout_until` / `wc_ok`
(`milp_core.solve_horizon`); caller computes the gate from the matrix
(`season_simulator`, `MP_WC_BELOW` env). Tests: 15/15
(`tests/test_milp_chips.py`, incl. lockout, TC-hold, TC-on-double
exception, WC gate).

## A/B result (all runs: corrected rules, H=5, Docker)

| Season | legacy optimizer | mp + legacy chips | mp + model chips (fixed) |
|---|---|---|---|
| 2025-26 (legacy home) | **2252** | 2156 | 2029 (was 1955 broken) |
| 2023-24 (neutral) | 2164 | 2162 | **2174** — all 8 chips used |
| 2024-25 (neutral) | 2359 | 2341 | **2410** — project best |

**On neutral calendars, in-model chips now beat everything** — including
the fully tuned legacy system (+10 on 2023-24, +51 on 2024-25; neutral-season
mean 2292 vs legacy's 2261). Placements are event-anchored: 2023-24 stacked
FH1 on the GW7 double with BB1/TC1 adjacent (one-chip-per-GW forced the
spread — coherent); 2024-25 hit tc2@DGW24, bb2@DGW25, fh2@blank29; 2025-26
hit tc2@DGW26 (a 101-pt week), bb2@DGW33, fh2@blank34.

**On 2025-26 legacy chips still win** (2156 vs 2029) — consistent with the
memorized-calendar thesis all the way down: the legacy chip scheduler's
thresholds are part of the 25 constants tuned on that season, and 2025-26's
eventless Set 1 gives the guards nothing to anchor unanchored chips on
(wc1@5 / tc1@6 / bb1@9 still fire semi-early there).

## Remaining known issues

- Unanchored chips in an eventless set still fire on first-decent-week
  logic. Candidate (Phase 6): a season-percentile bar on the chip aux value
  (fire only if the week's bonus is in the top-q of weeks seen so far) —
  needs cross-solve memory; tune q cross-season.
- 2023-24 run took −136 in hit penalties navigating six blank GWs and still
  won — but hit ROI on that calendar deserves a look in Phase 6.
- FH1 on 2024-25 went unused despite blank GW15 (value below the emergent
  shadow-vs-main gap). Acceptable (better wasted than misused) but worth a
  revisit alongside the percentile bar.

## Standing config recommendation

Neutral-calendar (i.e., real future season) best: `OPTIMIZER=mp
MP_HORIZON=5 MP_CHIPS=model`. For 2025-26-only comparisons vs the 2468-era
system, `MP_CHIPS=legacy` remains the stronger ablation point.
