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

## Thesis ablation table — legacy vs mp × chips × seasons
Cross-environment scoreboard (Docker, corrected rules unless noted), from
[[HANDOFF]] §1 — the full record used to judge the [[optimizer-redesign]].
All three calendars: 2025-26 (memorized/tuning season), 2023-24 and 2024-25
(neutral, never tuned on).

| Config | 2025-26 | 2023-24 | 2024-25 | Σ 3 seasons |
|---|---:|---:|---:|---:|
| legacy optimizer (25 tuned constants) | **2252** (fair baseline) | 2164 | 2359 | 6775 |
| mp, H=1 (Phase 2) | 2070 | — | — | — |
| mp, H=5 + legacy chip scheduler (Phase 3) | 2156 | 2162 | 2341 | 6659 |
| mp + model chips + scarcity fix | 2029 | 2174 | **2410** | 6613 |
| mp + percentile chip bar (q75) | 2152 | 2206 | 2262 | 6620 |
| **mp + discipline (shipped mp default)** | **2157** | 2186 | 2338 | **6681** |
| mp + percentile bar + discipline | 2084 | **2224** | 2327 | 6635 |

Discipline = `MP_HIT_COST=8, MP_HIT_BUDGET=4, MP_REBUY_GAP=4` (2×2 A/B,
2026-07-15: cut penalties from ~−90/season avg to ≤−12 avg, +68 on the
3-season sum). Percentile bar defaults OFF — nets −46 once discipline is on.

**Chip strategy v2 vs legacy hardcoded schedule** (both on `OPTIMIZER=legacy`,
2025-26 only): v2 = **2180**, hardcoded GW17/18/19 policy = 2252. Closes
[[chip-strategy-v2]] — the rolling-horizon scheduler is calendar-agnostic but
does not yet beat the hand-tuned hardcoded policy on the season it was tuned
on; not re-tested on neutral calendars.

**Phase 6 cross-season Optuna sweep** (in progress, `pipeline/optuna_mp_search.py`,
train on 2023-24+2024-25, hold out 2025-26): best trial so far beats the
θ=0.3 discipline champion (train-sum 4696) by **+141** (trial 16, sum 4837) —
not yet evaluated on the 2025-26 holdout, not yet promoted to defaults.
Live leaderboard: `data/intel/optuna_mp/summary.json`.

Takeaways ([[generalization_report]]):
1. Legacy's ~96-pt home-season edge collapses to −2/−18 on neutral calendars
   — its advantage is ~85-90% memorized calendar, not optimizer skill.
2. mp+model-chips beats the fully tuned legacy system on both neutral seasons;
   legacy only wins on the season it was tuned on.
3. The MILP travels untuned; the legacy ILP does not.

## Related Source Files
- `models/stage7_results.json` (MAE curves per position)
- `data/intel/season_simulation.json` (headline run)
- `pipeline/backtest_metrics.py` (metric computation)

---
Hubs: [[system-overview]] · [[data-flow]]
