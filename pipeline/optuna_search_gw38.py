"""
pipeline/optuna_search_gw38.py
Optuna TPE search over the FULL season (GW1-38).

Separate from optuna_search.py (GW1-28) — writes to its own directory.
Does NOT touch data/intel/optuna_search/ or season_simulation.json.

Seed: Trial 220 params — current GW1-38 leader (2296 pts, 60.4/GW).

Results saved to: data/intel/optuna_search_gw38/
  summary.json      — top results + full trial list
  study.db          — Optuna SQLite storage (resumable)
  trial_NNN.json    — per-trial result
  trial_NNN_sim.json — per-trial full simulation output

Run:
  python -m pipeline.optuna_search_gw38 [--trials N] [--seed S] [--lgbm-only]
Resume:
  python -m pipeline.optuna_search_gw38 --trials 400
"""

import sys
import json
import os
import time

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Output paths (separate from GW1-28 search) ────────────────────────────────
RESULTS_DIR  = os.path.join(_ROOT, "data", "intel", "optuna_search_gw38")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")
DB_PATH      = os.path.join(RESULTS_DIR, "study.db")

N_TRIALS_DEFAULT = 200

# ── Seed: Trial 220 params — GW1-38 leader (2296 pts) ─────────────────────────
BEST_KNOWN = {
    "MODEL_TYPE":              "lgbm",
    "model_n_estimators":      300,
    "model_max_depth":         3,
    "model_learning_rate":     0.03,
    "model_subsample":         0.9,
    "model_colsample":         0.6,
    "xgb_min_child_weight":    1,
    "lgbm_num_leaves":         31,
    "lgbm_min_child_samples":  30,
    "FDR_MULT":                0.02,
    "FDR_MULT_DEF":            0.08,
    "BENCH_BONUS_NORMAL":      0.0,
    "BENCH_BONUS_BB_GW":       2.0,
    "CAP_FORM_GATE":           4.0,
    "CAP_FORM_PENALTY":        0.3,
    "OWN_BOOST_GW1":           0.15,
    "CAP_STREAK_LIMIT":        2,
    "CAP_STREAK_MULT":         0.9,
    "TC_THRESH":               8.0,
    "BB_MIN_GW":               9,
    "CAP_FDR_MULT":            0.1,
    "CAP_BLANK_PENALTY":       0.9,
    "CAP_BLANK_THRESH":        4,
}


def suggest_params(trial, lgbm_only=False):
    model_type = trial.suggest_categorical(
        "MODEL_TYPE", ["lgbm"] if lgbm_only else ["xgb", "lgbm"]
    )

    n_estimators  = trial.suggest_int("model_n_estimators", 100, 500, step=100)
    max_depth     = trial.suggest_int("model_max_depth", 3, 6)
    learning_rate = trial.suggest_float("model_learning_rate", 0.01, 0.10, log=True)
    subsample     = trial.suggest_float("model_subsample", 0.6, 1.0)
    colsample     = trial.suggest_float("model_colsample", 0.6, 1.0)

    xgb_min_child_weight  = trial.suggest_int("xgb_min_child_weight", 1, 10)
    lgbm_num_leaves       = trial.suggest_categorical("lgbm_num_leaves", [15, 31, 63, 127])
    lgbm_min_child_samples = trial.suggest_int("lgbm_min_child_samples", 5, 50)

    fdr_mult          = trial.suggest_float("FDR_MULT",           0.0,  0.03)
    fdr_mult_def      = trial.suggest_float("FDR_MULT_DEF",       0.025, 0.10)
    bench_bonus_norm  = trial.suggest_float("BENCH_BONUS_NORMAL",  0.0,  3.0)
    bench_bonus_bb    = trial.suggest_float("BENCH_BONUS_BB_GW",   2.0,  8.0)
    cap_form_gate     = trial.suggest_float("CAP_FORM_GATE",       2.0,  8.0)
    cap_form_penalty  = trial.suggest_float("CAP_FORM_PENALTY",    0.2,  0.7)
    own_boost_gw1     = trial.suggest_float("OWN_BOOST_GW1",       0.05, 0.25)
    cap_streak_limit  = trial.suggest_int("CAP_STREAK_LIMIT",      2,    5)
    cap_streak_mult   = trial.suggest_float("CAP_STREAK_MULT",     0.5,  0.9)
    tc_thresh         = trial.suggest_float("TC_THRESH",            5.0, 11.0)
    bb_min_gw         = trial.suggest_int("BB_MIN_GW",             6,   10)
    cap_fdr_mult      = trial.suggest_float("CAP_FDR_MULT",        0.0,  0.12)
    cap_blank_penalty = trial.suggest_float("CAP_BLANK_PENALTY",   0.5,  0.9)
    cap_blank_thresh  = trial.suggest_int("CAP_BLANK_THRESH",      2,    6)

    return {
        "MODEL_TYPE":              model_type,
        "model_n_estimators":      n_estimators,
        "model_max_depth":         max_depth,
        "model_learning_rate":     learning_rate,
        "model_subsample":         subsample,
        "model_colsample":         colsample,
        "xgb_min_child_weight":    xgb_min_child_weight,
        "lgbm_num_leaves":         lgbm_num_leaves,
        "lgbm_min_child_samples":  lgbm_min_child_samples,
        "FDR_MULT":                fdr_mult,
        "FDR_MULT_DEF":            fdr_mult_def,
        "BENCH_BONUS_NORMAL":      bench_bonus_norm,
        "BENCH_BONUS_BB_GW":       bench_bonus_bb,
        "CAP_FORM_GATE":           cap_form_gate,
        "CAP_FORM_PENALTY":        cap_form_penalty,
        "OWN_BOOST_GW1":           own_boost_gw1,
        "CAP_STREAK_LIMIT":        cap_streak_limit,
        "CAP_STREAK_MULT":         cap_streak_mult,
        "TC_THRESH":               tc_thresh,
        "BB_MIN_GW":               bb_min_gw,
        "CAP_FDR_MULT":            cap_fdr_mult,
        "CAP_BLANK_PENALTY":       cap_blank_penalty,
        "CAP_BLANK_THRESH":        cap_blank_thresh,
    }


def run_simulation(trial_id, params):
    """Patch season_simulator with params and run GW1-38. Returns result dict."""
    for key in list(sys.modules.keys()):
        if "season_simulator" in key:
            del sys.modules[key]

    import pipeline.season_simulator as sim

    sim.OUTPUT_JSON = os.path.join(RESULTS_DIR, f"trial_{trial_id:03d}_sim.json")
    sim.SIM_END_GW  = 38    # explicitly enforce full season
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sim.MODEL_TYPE = params["MODEL_TYPE"]

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
    try:
        log = sim.run_simulation()
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


def save_summary(all_results):
    valid = [r for r in all_results if r.get("status") == "ok"]
    valid.sort(key=lambda x: x["total"], reverse=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump({
            "scope":            "GW1-38 full season",
            "completed_trials": len(all_results),
            "best_total":       valid[0]["total"] if valid else 0,
            "best_trial":       valid[0] if valid else None,
            "results":          all_results,
        }, f, indent=2)


def print_report(all_results):
    valid = [r for r in all_results if r.get("status") == "ok"]
    valid.sort(key=lambda x: x["total"], reverse=True)

    xgb_r  = [r for r in valid if r["params"].get("MODEL_TYPE") == "xgb"]
    lgbm_r = [r for r in valid if r["params"].get("MODEL_TYPE") == "lgbm"]

    print("\n" + "=" * 70)
    print("  OPTUNA GW1-38 SEARCH COMPLETE — TOP 10 RESULTS")
    print("=" * 70)
    print(f"  {'Rank':<5} {'Trial':<7} {'Model':<6} {'Total':>6}  {'Avg/GW':>7}  Key params")
    print("  " + "-" * 68)
    for rank, r in enumerate(valid[:10], 1):
        m    = r["params"].get("MODEL_TYPE", "?")
        d    = r["params"].get("model_max_depth", "?")
        lr   = round(r["params"].get("model_learning_rate", 0), 4)
        gate = round(r["params"].get("CAP_FORM_GATE", 0), 2)
        fdr  = round(r["params"].get("FDR_MULT_DEF", 0), 3)
        avg  = round(r["total"] / 38, 1)
        print(f"  {rank:<5} {r['trial_id']:<7} {m:<6} {r['total']:>6}  {avg:>7}  "
              f"depth={d} lr={lr} gate={gate} fdr_def={fdr}")

    print(f"\n  BEST: {valid[0]['total']} pts  ({round(valid[0]['total']/38,1)}/GW)"
          f"  trial {valid[0]['trial_id']}  {valid[0]['params']['MODEL_TYPE'].upper()}")
    print(f"  Full params: {valid[0]['params']}")

    baseline = next((r for r in all_results if r["trial_id"] == 1), None)
    if baseline:
        print(f"\n  Seeded baseline (trial 1, Trial-220 params): {baseline['total']} pts")
        delta = valid[0]["total"] - baseline["total"]
        sign  = "+" if delta >= 0 else ""
        print(f"  Improvement over seed:  {sign}{delta} pts")

    if xgb_r:
        print(f"\n  XGBoost best:   {xgb_r[0]['total']} pts (trial {xgb_r[0]['trial_id']})")
    if lgbm_r:
        print(f"  LightGBM best:  {lgbm_r[0]['total']} pts (trial {lgbm_r[0]['trial_id']})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",    type=int,  default=N_TRIALS_DEFAULT,
                        help="Number of Optuna trials to run (default 200)")
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--lgbm-only", action="store_true",
                        help="Search LightGBM only (recommended — LGBM dominates)")
    parser.add_argument("--worker",    type=int,  default=0,
                        help="Worker ID for parallel runs (0-9). Each worker uses its own "
                             "trial ID range (worker N starts at N*1000+1) to avoid collisions.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    existing_trial_files = sorted(
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("trial_") and f.endswith(".json") and "_sim" not in f
    )
    for fname in existing_trial_files:
        with open(os.path.join(RESULTS_DIR, fname)) as fh:
            all_results.append(json.load(fh))

    # Each worker owns its own ID range: worker N → N*1000+1 ... N*1000+999
    worker_offset = args.worker * 1000
    worker_ids = {r["trial_id"] for r in all_results
                  if worker_offset < r["trial_id"] <= worker_offset + 999}
    next_trial_id = worker_offset + max(worker_ids - {worker_offset}, default=0) + 1

    storage_url = f"sqlite:///{DB_PATH}"
    sampler     = optuna.samplers.TPESampler(seed=args.seed + args.worker)
    study       = optuna.create_study(
        study_name     = "fpl_ai_optuna_gw38",
        storage        = storage_url,
        direction      = "maximize",
        sampler        = sampler,
        load_if_exists = True,
    )

    if len(study.trials) == 0:
        study.enqueue_trial(BEST_KNOWN)
        print(f"  Seeded trial 1 with Trial-220 params (GW1-38 baseline: 2296 pts)")

    completed_in_study = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])
    # Use worker's own JSON file count for remaining, not shared study total
    worker_completed = len(worker_ids)
    remaining = args.trials - worker_completed

    print("=" * 70)
    print(f"  OPTUNA GW1-38 SEARCH — {args.trials} trials "
          f"({'lgbm only' if args.lgbm_only else 'xgb + lgbm'}) | worker {args.worker}")
    print(f"  Sampler: TPE (Bayesian) | seed={args.seed + args.worker}")
    print(f"  Output:  {RESULTS_DIR}  | ID range: {worker_offset+1}-{worker_offset+999}")
    if completed_in_study > 0:
        best = study.best_value
        print(f"  Resuming — {completed_in_study} total in study, {worker_completed} this worker, best so far: {best} pts")
    print(f"  Running {remaining} new trial(s)...")
    print("=" * 70)

    lgbm_only = args.lgbm_only

    def objective(trial):
        nonlocal next_trial_id
        trial_id = next_trial_id
        next_trial_id += 1

        params = suggest_params(trial, lgbm_only=lgbm_only)
        result = run_simulation(trial_id, params)

        trial_path = os.path.join(RESULTS_DIR, f"trial_{trial_id:03d}.json")
        with open(trial_path, "w") as f:
            json.dump(result, f, indent=2)

        all_results.append(result)
        save_summary(all_results)

        valid = [r for r in all_results if r.get("status") == "ok"]
        valid.sort(key=lambda x: x["total"], reverse=True)
        best_so_far = valid[0]["total"] if valid else 0

        model_tag  = params.get("MODEL_TYPE", "?").upper()
        status_str = f"{result['total']:4d} pts" if result["status"] == "ok" else result["status"][:20]
        print(f"  Trial {trial_id:3d} | {model_tag:<4} | "
              f"{status_str} | "
              f"{result['elapsed']}s | "
              f"Best: {best_so_far} pts (t{valid[0]['trial_id']})")

        if result["status"] != "ok":
            raise optuna.exceptions.TrialPruned()

        return float(result["total"])

    study.optimize(objective, n_trials=remaining)

    print_report(all_results)
    print(f"\n  Full results -> {SUMMARY_PATH}")
    print(f"  Optuna study -> {DB_PATH}")
    print(f"\n  To resume: python -m pipeline.optuna_search_gw38 --trials {args.trials + 100}")


if __name__ == "__main__":
    main()
