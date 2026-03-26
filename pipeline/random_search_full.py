"""
pipeline/random_search_full.py
Joint random search over:
  - MODEL_TYPE (xgb vs lgbm)
  - ML model hyperparameters (depth, learning rate, trees, etc.)
  - Optimizer/strategy parameters (captain logic, chip timing, FDR, etc.)

Saves after every trial — safe to Ctrl+C and resume anytime.
Trial 1: current XGBoost best (1716 baseline verification).
"""
import sys
import json
import os
import time
import random

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Search space ───────────────────────────────────────────────────────────────
# MODEL_TYPE switches the boosting library.
# XGB_* and LGBM_* tune the respective model hyperparameters.
# Shared model params (n_estimators, learning_rate, etc.) apply to whichever
# MODEL_TYPE is selected — the trial runner patches the right dict.
# Optimizer params are the same regardless of model.

SEARCH_SPACE = {
    # ── Model selection ────────────────────────────────────────────────────
    "MODEL_TYPE":          ["xgb", "lgbm"],

    # ── Shared model hyperparams ───────────────────────────────────────────
    "model_n_estimators":  [100, 200, 300, 400, 500],
    "model_max_depth":     [3, 4, 5, 6],
    "model_learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10],
    "model_subsample":     [0.6, 0.7, 0.8, 0.9, 1.0],
    "model_colsample":     [0.6, 0.7, 0.8, 0.9, 1.0],

    # ── XGBoost-specific ───────────────────────────────────────────────────
    "xgb_min_child_weight": [1, 3, 5, 7, 10],

    # ── LightGBM-specific ─────────────────────────────────────────────────
    "lgbm_num_leaves":      [15, 31, 63, 127],
    "lgbm_min_child_samples": [5, 10, 20, 30, 50],

    # ── Optimizer / strategy params ────────────────────────────────────────
    "FDR_MULT":            [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
    "FDR_MULT_DEF":        [0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10],
    "BENCH_BONUS_NORMAL":  [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "BENCH_BONUS_BB_GW":   [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "CAP_FORM_GATE":       [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "CAP_FORM_PENALTY":    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "OWN_BOOST_GW1":       [0.05, 0.10, 0.15, 0.20, 0.25],
    "CAP_STREAK_LIMIT":    [2, 3, 4, 5],
    "CAP_STREAK_MULT":     [0.5, 0.6, 0.7, 0.8, 0.9],
    "TC_THRESH":           [8.0, 8.5, 9.0, 9.5, 10.0, 10.5],
    "BB_MIN_GW":           [6, 7, 8, 9, 10],
    "CAP_FDR_MULT":        [0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12],
    "CAP_BLANK_PENALTY":   [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    "CAP_BLANK_THRESH":    [2, 3, 4, 5, 6],
}

N_TRIALS     = 250
RESULTS_DIR  = os.path.join(_ROOT, "data", "intel", "random_search_full")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")

# Trial 1: current best XGBoost config (baseline)
BEST_KNOWN = {
    "MODEL_TYPE":             "xgb",
    "model_n_estimators":     300,
    "model_max_depth":        4,
    "model_learning_rate":    0.05,
    "model_subsample":        0.8,
    "model_colsample":        0.8,
    "xgb_min_child_weight":   3,
    "lgbm_num_leaves":        31,
    "lgbm_min_child_samples": 10,
    "FDR_MULT":               0.025,
    "FDR_MULT_DEF":           0.025,
    "BENCH_BONUS_NORMAL":     0.0,
    "BENCH_BONUS_BB_GW":      2.0,
    "CAP_FORM_GATE":          7.0,
    "CAP_FORM_PENALTY":       0.5,
    "OWN_BOOST_GW1":          0.20,
    "CAP_STREAK_LIMIT":       2,
    "CAP_STREAK_MULT":        0.8,
    "TC_THRESH":              8.5,
    "BB_MIN_GW":              9,
    "CAP_FDR_MULT":           0.10,
    "CAP_BLANK_PENALTY":      0.65,
    "CAP_BLANK_THRESH":       4,
}


def run_trial(trial_id, params):
    for key in list(sys.modules.keys()):
        if "season_simulator" in key:
            del sys.modules[key]

    import pipeline.season_simulator as sim

    sim.OUTPUT_JSON = os.path.join(RESULTS_DIR, f"trial_{trial_id:03d}_sim.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Set model type ─────────────────────────────────────────────────────
    model_type = params.get("MODEL_TYPE", "xgb")
    sim.MODEL_TYPE = model_type

    # ── Patch XGB_PARAMS ───────────────────────────────────────────────────
    sim.XGB_PARAMS = dict(
        n_estimators     = params["model_n_estimators"],
        max_depth        = params["model_max_depth"],
        learning_rate    = params["model_learning_rate"],
        subsample        = params["model_subsample"],
        colsample_bytree = params["model_colsample"],
        min_child_weight = params["xgb_min_child_weight"],
        random_state     = 42,
        verbosity        = 0,
    )

    # ── Patch LGBM_PARAMS ──────────────────────────────────────────────────
    sim.LGBM_PARAMS = dict(
        n_estimators     = params["model_n_estimators"],
        max_depth        = params["model_max_depth"],
        num_leaves       = params["lgbm_num_leaves"],
        learning_rate    = params["model_learning_rate"],
        subsample        = params["model_subsample"],
        colsample_bytree = params["model_colsample"],
        min_child_samples = params["lgbm_min_child_samples"],
        random_state     = 42,
        verbosity        = -1,
    )

    # ── Patch optimizer params ─────────────────────────────────────────────
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
    try:
        log     = sim.run_simulation()
        elapsed = round(time.time() - t0)
        return {
            "trial_id":  trial_id,
            "params":    params,
            "total":     log["total_actual_pts"],
            "penalties": log["total_penalties"],
            "chips":     [c["chip"] for c in log["chips_used"]],
            "elapsed":   elapsed,
            "status":    "ok",
            "gw_scores": [
                {"gw": g["gw"], "actual": g["actual_total"], "chip": g.get("chip")}
                for g in log["gameweeks"]
            ],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "trial_id": trial_id,
            "params":   params,
            "total":    0,
            "status":   f"error: {e}",
            "elapsed":  round(time.time() - t0),
        }


def make_param_list(n_trials, seed):
    rng = random.Random(seed)
    params = [BEST_KNOWN]
    for _ in range(n_trials - 1):
        params.append({k: rng.choice(v) for k, v in SEARCH_SPACE.items()})
    return params


def save_summary(results, completed):
    valid = [r for r in results if r.get("status") == "ok"]
    valid.sort(key=lambda x: x["total"], reverse=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump({
            "completed_trials": completed,
            "best_total":       valid[0]["total"] if valid else 0,
            "best_trial":       valid[0] if valid else None,
            "results":          results,
        }, f, indent=2)


def print_report(results, n_trials):
    valid = [r for r in results if r.get("status") == "ok"]
    valid.sort(key=lambda x: x["total"], reverse=True)
    baseline = next((r for r in results if r["trial_id"] == 1), None)

    xgb_results  = [r for r in valid if r["params"].get("MODEL_TYPE") == "xgb"]
    lgbm_results = [r for r in valid if r["params"].get("MODEL_TYPE") == "lgbm"]

    print("\n" + "=" * 70)
    print("  FULL RANDOM SEARCH COMPLETE — TOP 10 RESULTS")
    print("=" * 70)
    print(f"  {'Rank':<5} {'Trial':<7} {'Model':<6} {'Total':>6}  Key params")
    print("  " + "-" * 64)
    for rank, r in enumerate(valid[:10], 1):
        m    = r["params"].get("MODEL_TYPE", "?")
        d    = r["params"].get("model_max_depth", "?")
        lr   = r["params"].get("model_learning_rate", "?")
        gate = r["params"].get("CAP_FORM_GATE", "?")
        fdr  = r["params"].get("FDR_MULT_DEF", "?")
        print(f"  {rank:<5} {r['trial_id']:<7} {m:<6} {r['total']:>6}  "
              f"depth={d} lr={lr} gate={gate} fdr_def={fdr}")

    print(f"\n  BEST: {valid[0]['total']} pts (trial {valid[0]['trial_id']}, {valid[0]['params']['MODEL_TYPE'].upper()})")
    print(f"  Full params: {valid[0]['params']}")
    if baseline:
        print(f"\n  Baseline (trial 1, XGB): {baseline['total']} pts")
        delta = valid[0]['total'] - baseline['total']
        sign  = "+" if delta >= 0 else ""
        print(f"  Improvement: {sign}{delta} pts")

    if xgb_results:
        print(f"\n  XGBoost best:   {xgb_results[0]['total']} pts (trial {xgb_results[0]['trial_id']})")
    if lgbm_results:
        print(f"  LightGBM best:  {lgbm_results[0]['total']} pts (trial {lgbm_results[0]['trial_id']})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_trials   = args.trials
    total_vals = sum(len(v) for v in SEARCH_SPACE.values())
    all_params = make_param_list(n_trials, args.seed)

    print("=" * 70)
    print(f"  FULL JOINT RANDOM SEARCH — {n_trials} trials")
    print(f"  Tuning: model type + model hyperparams + optimizer params")
    print(f"  Search space: {total_vals} values across {len(SEARCH_SPACE)} params")
    print(f"  Estimated runtime: ~{n_trials * 85 // 60} minutes")
    print("=" * 70)

    results  = []
    start_id = 1
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            existing = json.load(f)
        results  = existing.get("results", [])
        start_id = len(results) + 1
        if results:
            valid = [r for r in results if r.get("status") == "ok"]
            valid.sort(key=lambda x: x["total"], reverse=True)
            print(f"  Resuming from {len(results)} existing trials "
                  f"(best so far: {valid[0]['total']} pts, trial {valid[0]['trial_id']})")

    for trial_id in range(start_id, n_trials + 1):
        params = all_params[trial_id - 1]
        if trial_id == 1:
            print(f"\n  Trial 1: XGBoost best-known config (1716 baseline)")
        result = run_trial(trial_id, params)
        results.append(result)
        if trial_id == 1:
            print(f"  Trial 1 result: {result['total']} pts | {result['elapsed']}s")

        trial_path = os.path.join(RESULTS_DIR, f"trial_{trial_id:03d}.json")
        with open(trial_path, "w") as f:
            json.dump(result, f, indent=2)

        save_summary(results, trial_id)

        valid = [r for r in results if r.get("status") == "ok"]
        valid.sort(key=lambda x: x["total"], reverse=True)
        remaining_s = (n_trials - trial_id) * 85
        model_tag   = params.get("MODEL_TYPE", "?").upper()
        status_str  = f"{result['total']:4d} pts" if result["status"] == "ok" else result["status"][:20]
        print(f"  Trial {trial_id:3d}/{n_trials} | {model_tag:<4} | "
              f"{status_str} | "
              f"{result['elapsed']}s | "
              f"Best: {valid[0]['total']} pts (t{valid[0]['trial_id']}) | "
              f"ETA: {remaining_s // 60}m")

    print_report(results, n_trials)
    print(f"\n  Full results -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
