"""
pipeline/random_search_bench25.py
Focused random search with BENCH_BONUS_NORMAL locked at 2.5.

Rationale: across 250 random trials, BENCH_BONUS_NORMAL=2.5 had the highest
average score (1533) and highest minimum (1411). The overall best (1668) used
0.0, but that value has the lowest average — it was a lucky outlier.
This search finds the best combo of all OTHER params given bench bonus = 2.5.

Saves to data/intel/random_search_bench25/ — does NOT touch the original
random_search/ folder or Trial 1's 1668 result.
"""
import sys
import json
import os
import time
import random

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Search space — BENCH_BONUS_NORMAL is FIXED at 2.5 ────────────────────────
SEARCH_SPACE = {
    "FDR_MULT":            [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
    "BENCH_BONUS_NORMAL":  [2.5],          # LOCKED
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

N_TRIALS     = 200
RESULTS_DIR  = os.path.join(_ROOT, "data", "intel", "random_search_bench25")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")

# Trial 1 = Trial 1's params but with BENCH_BONUS_NORMAL forced to 2.5
# (so we can see directly how much 2.5 costs vs the lucky 0.0 run)
BEST_KNOWN_BENCH25 = {
    "FDR_MULT":           0.025,
    "BENCH_BONUS_NORMAL": 2.5,            # changed from 0.0
    "BENCH_BONUS_BB_GW":  2.0,
    "CAP_FORM_GATE":      5.0,
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
    for key in list(sys.modules.keys()):
        if "season_simulator" in key:
            del sys.modules[key]

    import pipeline.season_simulator as sim

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
    rng = random.Random(seed)
    params = [BEST_KNOWN_BENCH25]
    for _ in range(n_trials - 1):
        params.append({k: rng.choice(v) for k, v in SEARCH_SPACE.items()})
    return params


def save_summary(results, completed):
    valid = [r for r in results if r.get("status") == "ok"]
    valid.sort(key=lambda x: x["total"], reverse=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump({
            "completed_trials": completed,
            "bench_bonus_normal_locked": 2.5,
            "best_total":       valid[0]["total"] if valid else 0,
            "best_trial":       valid[0] if valid else None,
            "results":          results,
        }, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed",   type=int, default=99)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_trials   = args.trials
    all_params = make_param_list(n_trials, args.seed)

    print("=" * 60)
    print(f"  FOCUSED SEARCH — BENCH_BONUS_NORMAL=2.5 LOCKED")
    print(f"  {n_trials} trials | best known overall: 1668 (BENCH_BONUS=0.0)")
    print(f"  Trial 1 = original Trial 1 params with bench=2.5 swap")
    print("=" * 60)

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
            print(f"  Resuming from {len(results)} trials "
                  f"(best so far: {valid[0]['total']} pts, trial {valid[0]['trial_id']})")

    for trial_id in range(start_id, n_trials + 1):
        params = all_params[trial_id - 1]
        if trial_id == 1:
            print(f"\n  Trial 1: Trial-1 params with bench=2.5 (direct comparison)")

        result = run_trial(trial_id, params)
        results.append(result)

        if trial_id == 1:
            diff = result["total"] - 1668
            sign = "+" if diff >= 0 else ""
            print(f"  Trial 1 result: {result['total']} pts  ({sign}{diff} vs 1668 baseline)")

        trial_path = os.path.join(RESULTS_DIR, f"trial_{trial_id:03d}.json")
        with open(trial_path, "w") as f:
            json.dump(result, f, indent=2)

        save_summary(results, trial_id)

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
    print("  BENCH=2.5 FOCUSED SEARCH COMPLETE — TOP 10")
    print("=" * 60)
    print(f"  {'Rank':<5} {'Trial':<7} {'Total':>6}  Params")
    print("  " + "-" * 56)
    for rank, r in enumerate(valid[:10], 1):
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  {rank:<5} {r['trial_id']:<7} {r['total']:>6}  {param_str}")

    print(f"\n  BEST (bench=2.5): {valid[0]['total']} pts  (trial {valid[0]['trial_id']})")
    print(f"  vs overall best:  1668 pts  (trial 1, bench=0.0)")
    diff = valid[0]["total"] - 1668
    sign = "+" if diff >= 0 else ""
    print(f"  Difference:       {sign}{diff} pts")
    if baseline:
        print(f"\n  Trial 1 (direct swap): {baseline['total']} pts")
    print(f"\n  Full results -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
