"""
pipeline/trial_runner.py
Runs multiple season simulator configurations and compares results.
Each trial modifies specific constants in season_simulator, runs the
full GW1-28 simulation, and records the season total.

Expected runtime: ~15-25 min per trial × 10 trials = 2.5-4 hours.
Run overnight: python pipeline/trial_runner.py
"""
import sys
import json
import os
import time

# Ensure the project root (parent of pipeline/) is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Trial definitions ─────────────────────────────────────────────────────────
# Each trial changes ONE thing from the current baseline (11-feature, Intel 07).
# "changes" keys must match season_simulator module-level names,
# OR be "FEAT_COLS_EXTRA" (list appended to FEAT_COLS).

TRIALS = [
    {
        "id": 1,
        "name": "Baseline (no team/opp features)",
        "description": "11-feature baseline — Intel 07 bench, availability, captain form gate",
        "changes": {}
    },
    {
        "id": 2,
        "name": "Team own form only",
        "description": "Add team_goals_last3 + team_cs_rate_last3 (no opp features)",
        "changes": {
            "FEAT_COLS_EXTRA": ["team_goals_last3", "team_cs_rate_last3"],
            "OPP_FEATURES": False,
        }
    },
    {
        "id": 3,
        "name": "Opp form only",
        "description": "Add opp_goals_last3 + opp_cs_rate_last3 (no team features)",
        "changes": {
            "FEAT_COLS_EXTRA": ["opp_goals_last3", "opp_cs_rate_last3"],
            "TEAM_FEATURES": False,
        }
    },
    {
        "id": 4,
        "name": "All 4 team/opp features (Intel 08)",
        "description": "Full Intel 08 — team_goals, team_cs, opp_goals, opp_cs",
        "changes": {
            "FEAT_COLS_EXTRA": [
                "team_goals_last3", "team_cs_rate_last3",
                "opp_goals_last3",  "opp_cs_rate_last3",
            ],
        }
    },
    {
        "id": 5,
        "name": "Higher bench bonus (4.0 normal, 8.0 BB)",
        "description": "Intel 07 bench bonus raised — forces stronger bench picks",
        "changes": {
            "BENCH_BONUS_NORMAL": 4.0,
            "BENCH_BONUS_BB_GW":  8.0,
        }
    },
    {
        "id": 6,
        "name": "Lower bench bonus (1.5 normal, 4.0 BB)",
        "description": "Intel 07 bench bonus lowered — less interference with XI",
        "changes": {
            "BENCH_BONUS_NORMAL": 1.5,
            "BENCH_BONUS_BB_GW":  4.0,
        }
    },
    {
        "id": 7,
        "name": "Higher FDR multiplier (0.05)",
        "description": "FDR adjustment more aggressive — hard fixtures penalized more",
        "changes": {
            "FDR_MULT": 0.05,
        }
    },
    {
        "id": 8,
        "name": "Lower FDR multiplier (0.01)",
        "description": "FDR adjustment lighter — fixture difficulty matters less",
        "changes": {
            "FDR_MULT": 0.01,
        }
    },
    {
        "id": 9,
        "name": "Tighter captain form gate (5.0)",
        "description": "Captain form gate raised 4.0->5.0 — harder to captain out-of-form",
        "changes": {
            "CAP_FORM_GATE": 5.0,
        }
    },
    {
        "id": 10,
        "name": "Looser captain form gate (3.0)",
        "description": "Captain form gate lowered 4.0->3.0 — allows more captain diversity",
        "changes": {
            "CAP_FORM_GATE": 3.0,
        }
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

_DATA_INTEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "intel"
)


def run_trial(trial):
    """Run one trial by patching simulator constants and calling run_simulation."""
    print(f"\n{'='*60}")
    print(f"TRIAL {trial['id']}: {trial['name']}")
    print(f"  {trial['description']}")
    if trial["changes"]:
        print(f"  Changes: {trial['changes']}")
    print("=" * 60)

    # Fresh import — delete cached module so constants reset
    for mod_key in list(sys.modules.keys()):
        if "season_simulator" in mod_key:
            del sys.modules[mod_key]

    import pipeline.season_simulator as sim

    # Apply constant overrides
    for key, val in trial.get("changes", {}).items():
        if key == "FEAT_COLS_EXTRA":
            sim.FEAT_COLS = list(sim.FEAT_COLS) + list(val)
            print(f"  [TRIAL] FEAT_COLS += {val}  (now {len(sim.FEAT_COLS)} features)")
        elif hasattr(sim, key):
            setattr(sim, key, val)
            print(f"  [TRIAL] {key} = {val}")
        else:
            print(f"  [WARN]  Unknown key '{key}' — skipped")

    # Redirect output JSON so trials don't overwrite each other's main output
    os.makedirs(_DATA_INTEL_DIR, exist_ok=True)
    trial_json = os.path.join(_DATA_INTEL_DIR,
                               f"trial_{trial['id']:02d}_simulation.json")
    sim.OUTPUT_JSON = trial_json

    t0 = time.time()
    try:
        log = sim.run_simulation()
        elapsed = time.time() - t0

        total     = log["total_actual_pts"]
        penalties = log["total_penalties"]
        chips     = [c["chip"] for c in log["chips_used"]]

        # Save compact trial result
        result_path = os.path.join(_DATA_INTEL_DIR,
                                    f"trial_{trial['id']:02d}_result.json")
        with open(result_path, "w") as f:
            json.dump({
                "trial_id":        trial["id"],
                "trial_name":      trial["name"],
                "description":     trial["description"],
                "changes":         trial["changes"],
                "total_actual_pts": total,
                "total_penalties": penalties,
                "chips_used":      chips,
                "elapsed_seconds": round(elapsed),
                "feat_cols_used":  sim.FEAT_COLS,
                "gw_scores": [
                    {
                        "gw":     g["gw"],
                        "actual": g["actual_total"],
                        "pred":   g["predicted_total"],
                        "chip":   g.get("chip"),
                    }
                    for g in log["gameweeks"]
                ],
            }, f, indent=2)

        return {
            "trial_id":   trial["id"],
            "name":       trial["name"],
            "total":      total,
            "penalties":  penalties,
            "chips":      chips,
            "elapsed":    round(elapsed),
            "status":     "ok",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        elapsed = time.time() - t0
        print(f"  [ERROR] Trial {trial['id']} failed: {e}")
        return {
            "trial_id": trial["id"],
            "name":     trial["name"],
            "total":    0,
            "status":   f"error: {e}",
            "elapsed":  round(elapsed),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=str, default=None,
                        help="Comma-separated trial IDs to run (e.g. 1,3,5). Default: all.")
    args = parser.parse_args()

    if args.trials:
        selected_ids = {int(x) for x in args.trials.split(",")}
        trials_to_run = [t for t in TRIALS if t["id"] in selected_ids]
    else:
        trials_to_run = TRIALS

    print("=" * 60)
    print("  FPL SEASON SIMULATOR — TRIAL RUNNER")
    print(f"  Running {len(trials_to_run)} trial(s)")
    print("=" * 60)

    results = []
    for trial in trials_to_run:
        result = run_trial(trial)
        results.append(result)
        status_str = f"{result['total']} pts" if result["status"] == "ok" else result["status"]
        print(f"\n  [DONE] Trial {result['trial_id']}: {status_str} "
              f"({result['elapsed']}s)")

    # Sort by total pts descending
    results.sort(key=lambda x: x["total"], reverse=True)

    print("\n" + "=" * 60)
    print("  TRIAL RESULTS — RANKED BY SEASON TOTAL")
    print("=" * 60)
    print(f"  {'Rank':<5} {'T#':<4} {'Total':>7} {'Pen':>5}  Name")
    print("  " + "-" * 56)
    for rank, r in enumerate(results, 1):
        pen_str = str(r.get("penalties", "?"))
        marker  = " <-- BEST" if rank == 1 else ""
        print(f"  {rank:<5} {r['trial_id']:<4} {r['total']:>7} {pen_str:>5}  "
              f"{r['name']}{marker}")

    if results:
        best  = results[0]
        worst = results[-1]
        print()
        print(f"  Best:  Trial {best['trial_id']} — {best['total']} pts — {best['name']}")
        print(f"  Worst: Trial {worst['trial_id']} — {worst['total']} pts — {worst['name']}")

    # Save summary
    summary_path = os.path.join(_DATA_INTEL_DIR, "trial_summary.json")
    os.makedirs(_DATA_INTEL_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "trials_run": [t["id"] for t in trials_to_run],
            "results":    results,
            "best_trial": results[0]  if results else None,
            "worst_trial": results[-1] if results else None,
        }, f, indent=2)
    print(f"\n  Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
