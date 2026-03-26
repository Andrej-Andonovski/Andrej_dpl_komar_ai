"""
pipeline/random_search.py
Random hyperparameter search over season simulator constants.
Saves after every trial — safe to Ctrl+C and resume anytime.
"""
import sys
import json
import os
import time
import random

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SEARCH_SPACE = {
    "FDR_MULT":            [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
    "BENCH_BONUS_NORMAL":  [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "BENCH_BONUS_BB_GW":   [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "CAP_FORM_GATE":       [2.0, 3.0, 4.0, 5.0, 6.0],
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

N_TRIALS     = 1000
RESULTS_DIR  = os.path.join(_ROOT, "data", "intel", "random_search")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")

# Current best config — always run as trial 1 for baseline verification
BEST_KNOWN = {
    "FDR_MULT":           0.025,
    "BENCH_BONUS_NORMAL": 0.0,
    "BENCH_BONUS_BB_GW":  2.0,
    "CAP_FORM_GATE":      7.0,   # updated from 5.0 — fine grid sweep confirmed peak 6.7-7.0 (+11 pts)
    "CAP_FORM_PENALTY":   0.5,
    "OWN_BOOST_GW1":      0.20,
    "CAP_STREAK_LIMIT":   2,
    "CAP_STREAK_MULT":    0.8,
    "TC_THRESH":          10.5,
    "BB_MIN_GW":          9,
    "CAP_FDR_MULT":       0.05,
    "CAP_BLANK_PENALTY":  0.65,
    "CAP_BLANK_THRESH":   4,
}


def run_trial(trial_id, params):
    """Run one trial with the given parameter set. Returns result dict."""
    # Fresh module import each time so constants reset cleanly
    for key in list(sys.modules.keys()):
        if "season_simulator" in key:
            del sys.modules[key]

    import pipeline.season_simulator as sim

    # Redirect output JSON so trials don't clobber the main simulation file
    sim.OUTPUT_JSON = os.path.join(RESULTS_DIR, f"trial_{trial_id:03d}_sim.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for key, val in params.items():
        if hasattr(sim, key):
            setattr(sim, key, val)

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
    """
    Pre-generate ALL param combinations before any simulator imports.
    Uses a dedicated Random instance so season_simulator's module-level
    random.seed(42) cannot reset our sampling sequence between trials.
    Trial 1 is always BEST_KNOWN; trials 2..n are random.
    """
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help=f"Number of trials to run (default {N_TRIALS})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_trials   = args.trials
    total_vals = sum(len(v) for v in SEARCH_SPACE.values())

    # Pre-generate ALL param sets before any simulator imports.
    # This is critical: season_simulator calls random.seed(42) at module level,
    # which would reset the global random state between trials if we sampled lazily.
    all_params = make_param_list(n_trials, args.seed)

    print("=" * 60)
    print(f"  RANDOM HYPERPARAMETER SEARCH — {n_trials} trials")
    print(f"  Search space: {total_vals} values across {len(SEARCH_SPACE)} params")
    print(f"  Estimated runtime: ~{n_trials * 78 // 60} minutes")
    print("=" * 60)

    # Resume from existing results if available
    results   = []
    start_id  = 1
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
        params = all_params[trial_id - 1]   # index into pre-generated list
        if trial_id == 1:
            print(f"\n  Trial 1: current best config (baseline verification)")
        result = run_trial(trial_id, params)
        results.append(result)
        if trial_id == 1:
            print(f"  Trial 1 result: {result['total']} pts | {result['elapsed']}s")

        # Save individual trial
        trial_path = os.path.join(RESULTS_DIR, f"trial_{trial_id:03d}.json")
        with open(trial_path, "w") as f:
            json.dump(result, f, indent=2)

        # Save running summary
        save_summary(results, trial_id)

        # Live leaderboard line
        valid = [r for r in results if r.get("status") == "ok"]
        valid.sort(key=lambda x: x["total"], reverse=True)
        remaining_s = (n_trials - trial_id) * 78
        status_str  = f"{result['total']:4d} pts" if result["status"] == "ok" else result["status"][:20]
        print(f"  Trial {trial_id:3d}/{n_trials} | "
              f"{status_str} | "
              f"{result['elapsed']}s | "
              f"Best: {valid[0]['total']} pts (t{valid[0]['trial_id']}) | "
              f"ETA: {remaining_s // 60}m")

    # ── Final report ──────────────────────────────────────────────────────────
    valid = [r for r in results if r.get("status") == "ok"]
    valid.sort(key=lambda x: x["total"], reverse=True)
    baseline = next((r for r in results if r["trial_id"] == 1), None)

    print("\n" + "=" * 60)
    print("  RANDOM SEARCH COMPLETE — TOP 10 RESULTS")
    print("=" * 60)
    print(f"  {'Rank':<5} {'Trial':<7} {'Total':>6}  Params")
    print("  " + "-" * 56)
    for rank, r in enumerate(valid[:10], 1):
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  {rank:<5} {r['trial_id']:<7} {r['total']:>6}  {param_str}")

    print(f"\n  BEST:     {valid[0]['total']} pts  (trial {valid[0]['trial_id']})")
    print(f"  Params:   {valid[0]['params']}")
    if baseline:
        print(f"\n  Baseline (trial 1): {baseline['total']} pts")
        print(f"  Improvement: +{valid[0]['total'] - baseline['total']} pts")

    print(f"\n  Full results -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
