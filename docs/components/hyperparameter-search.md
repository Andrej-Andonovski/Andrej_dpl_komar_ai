---
type: component
status: active
tags:
  - component
  - tuning
---

# Hyperparameter Search

The offline tuning subsystem: it searches over model hyperparameters **and**
optimizer/strategy constants to find the configuration that scores best over a
full simulated season.

## Responsibility
Run many simulated seasons with different parameter sets and record the best.
Two search strategies:
- **Random search** (`random_search_full.py`) — 250 trials of pure random
  sampling over `MODEL_TYPE`, model hyperparameters, and every optimizer
  constant. Headline finding: LightGBM dominates (14 of the top 15 trials) — the
  [[lightgbm-over-xgboost]] decision.
- **Optuna** (`optuna_search.py`, `optuna_search_gw38.py`, `optuna_mp_search.py`)
  — TPE Bayesian search, warm-started from the best random trial, run at several
  scopes (GW1–28, full GW1–38, and a search for the [[milp-optimizer]] config).

## Why it exists
The [[season-simulator]] exposes dozens of constants (FDR multipliers, captain
gates, chip thresholds, model hyperparameters). Hand-tuning them all is
intractable; a search finds jointly good values and produces the numbers recorded
as "current params" in [`CLAUDE.md`](../../CLAUDE.md).

## How it interacts
Each trial **patches and runs** the [[season-simulator]] (the same mechanism
`run_variants.py` uses to reproduce specific configs), scores the season, and
logs results. The winning parameters are promoted back into the simulator and the
[[prediction-models]] configuration. It is an offline consumer — nothing in the
live pipeline depends on it at runtime.

## Depends on
- [[season-simulator]] (the objective being optimized).
- [[prediction-models]] (model hyperparameters are part of the search space).

## Depended on by
- Nothing at runtime; it **feeds parameter values** into
  [[season-simulator]]/[[prediction-models]] out of band.

## Assumptions & limitations
- Results are **environment-sensitive** for the same reason as the headline score
  (library stack), so cross-environment comparisons are unreliable — see
  [[environment-and-docker]].
- Search scopes differ (GW1–28 vs GW1–38); numbers are only comparable within a
  scope. Best-known results and study artifacts live under
  `data/intel/optuna_search*/` and `random_search_full/`.

## Related Source Files
- `pipeline/random_search_full.py`
- `pipeline/optuna_search.py`, `optuna_search_gw38.py`, `optuna_mp_search.py`
- `pipeline/run_variants.py`
- `data/intel/random_search_full/`, `data/intel/optuna_search/`, `optuna_search_gw38/`

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
