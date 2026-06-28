"""
pipeline/run_variants.py
Run specific simulator variants for GW1-38 comparison (Option C).
Uses the same patching mechanism as optuna_search.py.

Variants:
  - trial220   : Random search trial 220 params (1760 pts on GW1-28)
  - default    : Default LightGBM (no hyperparameter tuning, neutral strategy)
"""
import sys, os, json, time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

ARCHIVE_DIR = os.path.join(_ROOT, "data", "intel", "archive")
os.makedirs(ARCHIVE_DIR, exist_ok=True)


def patch_and_run(params, out_label):
    for key in list(sys.modules.keys()):
        if "season_simulator" in key:
            del sys.modules[key]

    import pipeline.season_simulator as sim

    sim.OUTPUT_JSON = os.path.join(ARCHIVE_DIR, f"season_simulation_{out_label}_gw38.json")

    sim.MODEL_TYPE = params["MODEL_TYPE"]

    sim.XGB_PARAMS = dict(
        n_estimators     = params["model_n_estimators"],
        max_depth        = params["model_max_depth"],
        learning_rate    = params["model_learning_rate"],
        subsample        = params["model_subsample"],
        colsample_bytree = params["model_colsample"],
        min_child_weight = params.get("xgb_min_child_weight", 1),
        random_state     = 42,
        verbosity        = 0,
    )
    sim.LGBM_PARAMS = dict(
        n_estimators      = params["model_n_estimators"],
        max_depth         = params["model_max_depth"],
        num_leaves        = params["lgbm_num_leaves"],
        learning_rate     = params["model_learning_rate"],
        subsample         = params["model_subsample"],
        colsample_bytree  = params["model_colsample"],
        min_child_samples = params["lgbm_min_child_samples"],
        random_state      = 42,
        verbosity         = -1,
    )

    optimizer_keys = {
        "FDR_MULT", "FDR_MULT_DEF", "BENCH_BONUS_NORMAL", "BENCH_BONUS_BB_GW",
        "CAP_FORM_GATE", "CAP_FORM_PENALTY", "OWN_BOOST_GW1",
        "CAP_STREAK_LIMIT", "CAP_STREAK_MULT", "TC_THRESH", "BB_MIN_GW",
        "CAP_FDR_MULT", "CAP_BLANK_PENALTY", "CAP_BLANK_THRESH",
    }
    for key in optimizer_keys:
        if key in params and hasattr(sim, key):
            setattr(sim, key, params[key])

    t0 = time.time()
    log = sim.run_simulation()
    elapsed = round(time.time() - t0)

    total = log["total_actual_pts"]
    penalties = log["total_penalties"]
    chips = [c["chip"] for c in log["chips_used"]]
    print(f"\n  [{out_label}] Total: {total} pts | Penalties: {penalties} | Chips: {chips}")
    print(f"  [{out_label}] Avg/GW: {round(total / 38, 1)} | Elapsed: {elapsed}s")
    print(f"  [{out_label}] Saved -> {sim.OUTPUT_JSON}")
    return total, penalties


# ── Variant 1: Trial 220 (random search best for GW1-28) ─────────────────────
TRIAL_220 = {
    "MODEL_TYPE":             "lgbm",
    "model_n_estimators":     300,
    "model_max_depth":        3,
    "model_learning_rate":    0.03,
    "model_subsample":        0.9,
    "model_colsample":        0.6,
    "xgb_min_child_weight":   1,
    "lgbm_num_leaves":        31,
    "lgbm_min_child_samples": 30,
    "FDR_MULT":               0.02,
    "FDR_MULT_DEF":           0.08,
    "BENCH_BONUS_NORMAL":     0.0,
    "BENCH_BONUS_BB_GW":      2.0,
    "CAP_FORM_GATE":          4.0,
    "CAP_FORM_PENALTY":       0.3,
    "OWN_BOOST_GW1":          0.15,
    "CAP_STREAK_LIMIT":       2,
    "CAP_STREAK_MULT":        0.9,
    "TC_THRESH":              8.0,
    "BB_MIN_GW":              9,
    "CAP_FDR_MULT":           0.1,
    "CAP_BLANK_PENALTY":      0.9,
    "CAP_BLANK_THRESH":       4,
}

# ── Variant 2: Default LightGBM — standard hyperparams, minimal strategy tuning
DEFAULT_LGBM = {
    "MODEL_TYPE":             "lgbm",
    "model_n_estimators":     100,
    "model_max_depth":        -1,    # unlimited (LightGBM default)
    "model_learning_rate":    0.1,
    "model_subsample":        1.0,
    "model_colsample":        1.0,
    "xgb_min_child_weight":   1,
    "lgbm_num_leaves":        31,    # LightGBM default
    "lgbm_min_child_samples": 20,   # LightGBM default
    "FDR_MULT":               0.0,
    "FDR_MULT_DEF":           0.0,
    "BENCH_BONUS_NORMAL":     0.0,
    "BENCH_BONUS_BB_GW":      2.0,
    "CAP_FORM_GATE":          4.0,
    "CAP_FORM_PENALTY":       0.5,
    "OWN_BOOST_GW1":          0.0,
    "CAP_STREAK_LIMIT":       3,
    "CAP_STREAK_MULT":        0.9,
    "TC_THRESH":              7.0,
    "BB_MIN_GW":              8,
    "CAP_FDR_MULT":           0.0,
    "CAP_BLANK_PENALTY":      0.8,
    "CAP_BLANK_THRESH":       4,
}


if __name__ == "__main__":
    results = {}

    print("\n" + "=" * 60)
    print("  VARIANT: Trial 220 (random search best, GW1-38)")
    print("=" * 60)
    total_220, pen_220 = patch_and_run(TRIAL_220, "trial220")
    results["trial220"] = {"total": total_220, "penalties": pen_220}

    print("\n" + "=" * 60)
    print("  VARIANT: Default LightGBM (no tuning, GW1-38)")
    print("=" * 60)
    total_def, pen_def = patch_and_run(DEFAULT_LGBM, "default_lgbm")
    results["default_lgbm"] = {"total": total_def, "penalties": pen_def}

    print("\n" + "=" * 60)
    print("  OPTION C COMPARISON SUMMARY (GW1-38)")
    print("=" * 60)
    trial429 = 2207
    print(f"  Default LGBM (no tuning) : {results['default_lgbm']['total']:>4} pts  ({round(results['default_lgbm']['total']/38,1)}/GW)")
    print(f"  Trial 220 (random search): {results['trial220']['total']:>4} pts  ({round(results['trial220']['total']/38,1)}/GW)")
    print(f"  Trial 429 (Optuna best)  : {trial429:>4} pts  ({round(trial429/38,1)}/GW)")
    print(f"\n  Optuna gain vs random  : +{trial429 - results['trial220']['total']} pts")
    print(f"  Tuning gain vs default : +{trial429 - results['default_lgbm']['total']} pts")
