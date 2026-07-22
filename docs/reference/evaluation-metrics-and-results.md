---
type: reference
status: active
tags:
  - reference
  - evaluation
  - results
---

# Reference: Evaluation Metrics & Results

Canonical home for how the system is scored and the numbers it achieved. Other
notes cite figures from here rather than restating them. Environment caveats live
in [[environment-and-docker]].

## Metrics
- **Primary:** MAE (mean absolute error on `total_points`), measured per position.
- **Secondary:** top-N accuracy (did the highest-predicted players score well).
- **Tertiary:** per-position feature-importance plots.
- Model validation is walk-forward — see [[walkforward-no-leakage]].

## Headline result (full GW1–38 season)
- **2468 pts** (~64.9/GW) on the original machine — reported in
  [`CLAUDE.md`](../../CLAUDE.md) as roughly +400 over an average manager.
- Chips in that run: tc1 GW6, bb1 GW8, wc1 GW17, bb2 GW21, tc2 GW23, fh2 GW26;
  net penalties −12.
- **Environment-bound** — Docker reproduces 2236; fair Docker baseline 2252 (see
  [[environment-and-docker]]).

## Improvement history (GW1–28 unless noted)
| Total | Change |
|------:|--------|
| ~429 | Stage 8 baseline (no intel, no snapshots) |
| ~557 | + GW-snapshot features |
| ~616 | + intel penalties + FDR + loyalty |
| ~629 | + auto-subs + bench weight + BB fix |
| 652 | + TC/BB triggers |
| 1760 | switch to LightGBM (random-search trial 220) |
| 1799 | Optuna trial 429 |
| **2468** | full-season run + GW38 Optuna (GW1–38) ← headline |

## Cross-environment scoreboard (Docker, corrected rules)
From [[HANDOFF]] — used to judge the [[optimizer-redesign]]:

| Config | 2025-26 | 2023-24 | 2024-25 |
|--------|--------:|--------:|--------:|
| legacy optimizer (tuned) | **2252** | 2164 | 2359 |
| mp + discipline (shipped mp default) | 2157 | 2186 | 2338 |
| mp + model chips + scarcity fix | 2029 | 2174 | **2410** |

Takeaway ([[generalization_report]]): legacy's home-season edge is largely
memorized calendar; the MILP travels better untuned.

## Related Source Files
- `models/stage7_results.json` (MAE curves per position)
- `data/intel/season_simulation.json` (headline run)
- `pipeline/backtest_metrics.py` (metric computation)

---
Hubs: [[system-overview]] · [[data-flow]]
