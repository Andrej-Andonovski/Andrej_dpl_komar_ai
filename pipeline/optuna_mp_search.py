"""
pipeline/optuna_mp_search.py — Phase 6: cross-season Optuna sweep for the
mp (multi-period MILP) optimizer.

Protocol (docs/HANDOFF.md §5.4): TRAIN on the two neutral calendars
(2023-24 + 2024-25, no intel, no memorized-calendar advantage), objective =
summed season total. HOLD OUT 2025-26 — evaluate only the top-K configs
there at the end and report the fold gap. This is the mp analogue of the
search that took the legacy system 1716 -> 2468.

Search space = the honest constants + discipline/chip knobs, all via env
(no source edits). H is fixed at 5 (Phase 1 calibration; H=6 doubles solve
time for marginal MAE change). Model hyperparams are NOT searched — the
prediction layer is frozen so the sweep attributes gains to the optimizer.

Every trial runs both seasons as subprocesses of season_simulator.py and
reads total_actual_pts from the output JSONs (sequential — the simulator's
output paths are per-season, not per-config).

Results: data/intel/optuna_mp/
  study.db        — SQLite storage, safe to Ctrl+C and resume
  summary.json    — leaderboard, refreshed after every trial
  trial_NNN.json  — params + per-season totals/chips/penalties

Run (inside Docker, repo mounted at /app):
  docker run --rm -v "<repo>:/app" -w /app fpl-sim \
    python -u pipeline/optuna_mp_search.py --trials 40
Smoke the plumbing first:
  ... python -u pipeline/optuna_mp_search.py --trials 1 --end-gw 3
"""

import argparse
import json
import os
import subprocess
import sys
import time

import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR   = os.path.join(_ROOT, "data", "intel", "optuna_mp")
SUMMARY_PATH  = os.path.join(RESULTS_DIR, "summary.json")
DB_PATH       = os.path.join(RESULTS_DIR, "study.db")
TRAIN_SEASONS = ["2023-24", "2024-25"]
HOLDOUT       = "2025-26"
SIM_TIMEOUT_S = 3600          # per season run; full seasons take 10-20 min

# The 2026-07-15 champion (theta=0.3 discipline config, train-sum 4696 =
# 2270 + 2426). Enqueued as trial 0 so TPE starts from the known best.
CHAMPION = {
    "MP_THETA": 0.3, "MP_DELTA": 0.94, "MP_DELTA_CHIP": 0.97,
    "MP_GAMMA": 0.07, "MP_W_BENCH": 0.15,
    "MP_HIT_COST": 8.0, "MP_HIT_BUDGET": 4, "MP_REBUY_GAP": 4,
    "MP_FT_VALUE": 0.0, "MP_FORM_HOLD": 0.0,
    "MP_WC_BELOW": 4, "MP_CHIP_BAR": 0, "MP_CHIP_PERCENTILE_Q": 0.75,
}


def suggest_params(trial):
    return {
        # captain + horizon economics
        "MP_THETA":      trial.suggest_float("MP_THETA", 0.10, 0.60),
        "MP_DELTA":      trial.suggest_float("MP_DELTA", 0.88, 0.99),
        "MP_DELTA_CHIP": trial.suggest_float("MP_DELTA_CHIP", 0.94, 1.00),
        "MP_GAMMA":      trial.suggest_float("MP_GAMMA", 0.03, 0.12),
        "MP_W_BENCH":    trial.suggest_float("MP_W_BENCH", 0.05, 0.30),
        # hit + churn discipline
        "MP_HIT_COST":   trial.suggest_float("MP_HIT_COST", 4.0, 12.0),
        "MP_HIT_BUDGET": trial.suggest_int("MP_HIT_BUDGET", 2, 8),
        "MP_REBUY_GAP":  trial.suggest_int("MP_REBUY_GAP", 0, 6),
        "MP_FT_VALUE":   trial.suggest_float("MP_FT_VALUE", 0.0, 1.5),
        "MP_FORM_HOLD":  trial.suggest_float("MP_FORM_HOLD", 0.0, 6.0),
        # chips
        "MP_WC_BELOW":   trial.suggest_int("MP_WC_BELOW", 3, 6),
        "MP_CHIP_BAR":   trial.suggest_categorical("MP_CHIP_BAR", [0, 1]),
        "MP_CHIP_PERCENTILE_Q":
            trial.suggest_float("MP_CHIP_PERCENTILE_Q", 0.50, 0.90),
    }


def run_season(season, params, end_gw=None):
    """One full-season sim as a subprocess; returns the output JSON."""
    env = dict(os.environ)
    env.update({
        "RULES_MODE": "corrected", "OPTIMIZER": "mp",
        "MP_HORIZON": "5", "MP_CHIPS": "model",
        "SIM_SEASON": season,
    })
    env.update({k: str(v) for k, v in params.items()})
    if end_gw:
        env["SIM_END_GW"] = str(end_gw)

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", os.path.join("pipeline", "season_simulator.py")],
        cwd=_ROOT, env=env, timeout=SIM_TIMEOUT_S,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-800:]
        raise RuntimeError(f"{season} sim failed rc={proc.returncode}: {tail}")

    out = os.path.join(_ROOT, "data", "intel",
                       f"season_simulation_corrected_mp_{season}.json")
    with open(out, encoding="utf-8") as f:
        j = json.load(f)
    return {
        "total": j["total_actual_pts"],
        "penalties": j["total_penalties"],
        "chips": [f"{c['chip']}@{c['gw']}" for c in j["chips_used"]],
        "minutes": round((time.time() - t0) / 60, 1),
    }


def write_summary(study):
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE]
    board = sorted(done, key=lambda t: -t.value)[:10]
    payload = {
        "protocol": {"train": TRAIN_SEASONS, "holdout": HOLDOUT,
                     "objective": "sum of train-season totals"},
        "champion_baseline": 4696,
        "n_complete": len(done),
        "best": ([{"trial": t.number, "value": t.value, "params": t.params}
                  for t in board]),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--end-gw", type=int, default=None,
                    help="SIM_END_GW override — plumbing smoke only")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    study = optuna.create_study(
        study_name="mp_phase6", direction="maximize",
        storage=f"sqlite:///{DB_PATH}", load_if_exists=True)

    if not study.trials:
        study.enqueue_trial(CHAMPION)

    def objective(trial):
        params = suggest_params(trial)
        results = {}
        for season in TRAIN_SEASONS:
            # a single pathological param combo (e.g. a chip-percentile
            # retry loop that never settles) must not take the whole sweep
            # down — prune this trial and keep going.
            try:
                results[season] = run_season(season, params, args.end_gw)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"[trial {trial.number:>3}] {season} failed: {e} "
                      "— pruning", flush=True)
                raise optuna.TrialPruned() from e
        total = sum(r["total"] for r in results.values())
        with open(os.path.join(RESULTS_DIR, f"trial_{trial.number:03d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"params": params, "seasons": results,
                       "objective": total}, f, indent=2)
        per_season = ", ".join(f"{s}:{r['total']}" for s, r in results.items())
        print(f"[trial {trial.number:>3}] {total} ({per_season})", flush=True)
        write_summary(study)
        return total

    # n_trials is the TARGET TOTAL, not "run this many more" — Optuna's
    # own n_trials counts fresh calls, so on resume we must subtract what
    # the study already has (including pruned/failed) to land on the same
    # total instead of drifting past it on every restart.
    remaining = max(0, args.trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"study already has {len(study.trials)} trials "
              f">= target {args.trials} — nothing to do")
    write_summary(study)
    print(f"\nbest: trial {study.best_trial.number} = {study.best_value}")
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
