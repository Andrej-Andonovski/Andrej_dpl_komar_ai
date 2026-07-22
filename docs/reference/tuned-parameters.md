---
type: reference
status: active
tags:
  - reference
  - parameters
  - configuration
---

# Reference: Tuned Parameters

Canonical home for the numeric constants and environment flags that drive the
[[season-simulator]]. Other notes reference values from here rather than
restating them. How these are found is [[hyperparameter-tuning]]; the redesign's
"honest constants" are in [[milp-optimizer]].

> [!note] Provisional
> These are the production **legacy-path** values recorded in
> [`CLAUDE.md`](../../CLAUDE.md) for the GW1–38 → 2468 run. The
> [[optimizer-redesign]] is replacing them; treat MILP constants as still under
> the Phase 6 sweep.

## Environment flags (`season_simulator.py`)
| Flag | Values | Effect |
|------|--------|--------|
| `MODEL_TYPE` | `lgbm` (prod) | prediction model family |
| `OPTIMIZER` | `legacy` \| `mp` | [[legacy-ilp-optimizer]] vs [[milp-optimizer]] |
| `RULES_MODE` | `legacy` \| `corrected` | see [[corrected-vs-legacy-rules]] |
| `CHIP_STRATEGY` | `v2` \| `legacy` | see [[chip-strategy-v2]] |
| `SIM_SEASON`, `SIM_END_GW` | season / GW | [[cross-season-harness]] |

## Intelligence multiplier (canonical formula)
Applied to predictions before optimization (used by [[intelligence-suite]] /
[[season-simulation]]):
```
avail_mult = availability_pct / 100
rot_mult   = 0.40 if rotation_risk >= 80
           = 0.60 if rotation_risk >= 60
           = 0.80 if rotation_risk >= 40
           = 1.00 otherwise
adjusted_pred = original_pred × avail_mult × rot_mult
```

## Key simulator constants (legacy path, 2468 run)
| Group | Values |
|-------|--------|
| FDR | `FDR_MULT=0.0285` (MID/FWD), `FDR_MULT_DEF=0.084` (GK/DEF) |
| GW1 ownership | `OWN_BOOST_GW1=0.213` |
| Triple Captain | `TC_THRESH=6.17`, `TC_FORM_MIN=6.0`, `TC2_MIN_GW=20` |
| Bench Boost | `BB_THRESH=9.0`, `BB_MIN_GW=8` |
| Captain gates | `CAP_FORM_GATE=6.57`, `CAP_FORM_PENALTY=0.574`, `CAP_STREAK_LIMIT=2`, `CAP_STREAK_MULT=0.899` |
| Wildcard | `WC_THRESH=5` |
| Bench bonus | `BENCH_BONUS_NORMAL=2.71`, `BENCH_BONUS_BB_GW=2.25` |
| DGW | `DGW_PRED_MULT=2.0` |
| Loyalty | GW1–5: 10.0, GW6–10: 2.0, GW11+: 1.0 |
| LightGBM | `n_estimators=200, max_depth=3, lr=0.0439, subsample=0.932, colsample_bytree=0.824, num_leaves=31, min_child_samples=27` |

Full list and the alternate GW38 Optuna trial (7011) are in
[`CLAUDE.md`](../../CLAUDE.md).

## Related Source Files
- `pipeline/season_simulator.py`
- `pipeline/milp_core.py` (redesign constants: `THETA`, `GAMMA`, `W_BENCH`, `HIT_COST`, `DELTA`)
- `data/intel/optuna_search_gw38/summary.json`

---
Hubs: [[system-overview]] · [[repository-map]]
