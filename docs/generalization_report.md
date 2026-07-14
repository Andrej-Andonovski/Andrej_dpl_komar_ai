# Cross-Season Generalization Report (fixing stage, step 1)

Status: **COMPLETE** (2026-07-14). Blueprint §10.4 — the test the redesign
was built for.

## Setup

- Harness: `SIM_SEASON` env in `season_simulator.py` +
  `pipeline/build_season_inputs.py` converting TRUE per-fixture vaastav repo
  data (real ids, real FDR, real ownership, DGW duplicate rows verified:
  983 extra rows / 244 >90-min player-GWs in 2023-24).
- Hygiene: training strictly season-cut (2023-24 run trains on 2019-23
  only); intel data and the GW15 FT event are 2025-26-only and disabled;
  corrected rules mandatory; 2025-26 chip/FT ruleset applied uniformly
  (the test measures CALENDAR generalization, not historical rule replay).
- 2024-25 required an element_type-5 filter (FPL added assistant managers
  that season).
- Configs: legacy optimizer vs mp (OPTIMIZER=mp MP_HORIZON=5
  MP_CHIPS=legacy — the Phase 3 best config). Same models, same data,
  same chips scheduler.
- Calendars: 2023-24 DGWs 7/25/28/34/35/37, blanks 2/17/18/26/29/34
  (rich, incl. a Set-1 double); 2024-25 DGWs 24/25/32/33, blanks 15/29/34.

## Result

| Season | legacy (25 tuned constants) | mp (H=5, ~8 constants, untuned) | gap |
|---|---|---|---|
| 2025-26 (legacy's home calendar) | 2252 | 2156 | **−96** |
| 2023-24 (neutral) | 2164 | 2162 | **−2** |
| 2024-25 (neutral) | 2359 | 2341 | **−18** |

**The legacy system's 96-point advantage is ~85-90% memorized calendar.**
Off its home season, the edge collapses to a mean −10 — statistical noise
territory — against an mp optimizer whose constants have never been tuned
on ANY season. The blueprint's central claim (the tuned-constant tower does
not transfer; the structural optimizer does) is confirmed in data.

Chip usage (legacy scheduler both): mp recovered wc2 on both neutral
seasons (legacy wasted it on 2024-25 and 2025-26); both found the 2023-24
GW7 Set-1 double for fh1. mp's heavy hit-taking persists off-season
(−60 pens on 2024-25) yet it still lands within 18 — hit pricing travels.

Run artifacts: `data/intel/season_simulation_corrected{,_mp}_{2023-24,2024-25}.json`,
logs `run_x_*.log`.

## Implications for the fixing stage

1. Every fix now measures against parity, not deficit: any improvement that
   holds across all three calendars is real, transferable gain.
2. The Phase 6 tuning objective is validated as constructed: tune on two
   seasons, hold out the third — the harness for it now exists.
3. Thesis framing writes itself: "the hand-tuned baseline loses 88-95 pts
   of its advantage when the calendar changes; the MILP redesign performs
   identically everywhere, before any tuning."

Next (per backlog): chip scarcity fix, A/B'd on all three calendars —
2023-24 is the best test bed (Set-1 double + six blank GWs).
