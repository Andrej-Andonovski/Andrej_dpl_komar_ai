---
type: decision
status: active
tags:
  - decision
  - models
---

# Decision: LightGBM over XGBoost

## Problem
The prediction layer needed a gradient-boosting library. The project started on
XGBoost (the stage/file names still say `xgb`), but it was unclear whether that
was the best choice.

## Alternatives considered
- **XGBoost** — the original implementation.
- **LightGBM** — evaluated jointly with all optimizer parameters.

## Decision
Use **LightGBM** (`MODEL_TYPE="lgbm"`). The 250-trial random search found LightGBM
dominant — *14 of the top 15 trials were LGBM* ([`CLAUDE.md`](../../CLAUDE.md)),
lifting the GW1–28 total from 1716 (best XGB) to 1760 and beyond. See
[[evaluation-metrics-and-results]] and [[hyperparameter-tuning]].

## Tradeoffs accepted
- The model files remain named `xgb_*.pkl` for historical continuity — they
  contain LightGBM models. This naming mismatch is a documented gotcha, not a bug.
- XGBoost support is retained in the search space for ablation/comparison.

## Components affected
[[prediction-models]] (the trained models), [[hyperparameter-search]] (search
space and the finding).

## Future work
Settled for this project. A different library would require re-running the joint
search, since optimizer constants were tuned against LightGBM predictions.

---
See also: [[system-overview]] · [[four-position-models]]
