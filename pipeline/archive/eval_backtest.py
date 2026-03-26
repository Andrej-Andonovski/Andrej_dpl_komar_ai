"""
Stage 8 Back-Test: Run ILP optimizer for GW1-10 of 2025-26 season,
compare predicted squad points against real player_history.csv actuals.

Forces sim_start_gw=1 so sim-GW1 = real GW1, sim-GW10 = real GW10.
Uses real actuals for online retraining (instead of Normal simulation).
Bench substitutions simplified: only starting XI + captain doubling counted.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
from collections import defaultdict

from ilp_optimizer_stage8 import (
    load_models, load_best_params, load_training_data,
    build_player_pool, predict_gw0, predict_horizon,
    run_ilp, split_bench, retrain_models, get_feature_cols,
    detect_dgw, detect_bgw,
    BUDGET, SEED, TARGET_COL,
    DATA_RAW, DATA_PROC, MODELS_DIR, POSITIONS, POS_FILES,
)

np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
PLAYERS_CSV  = os.path.join(DATA_RAW, "players_raw.csv")
HISTORY_CSV  = os.path.join(DATA_RAW, "player_history.csv")
FIXTURES_CSV = os.path.join(DATA_RAW, "fixtures_raw.csv")
FDR_CSV      = os.path.join(DATA_RAW, "fixture_difficulty.csv")
UPCOMING_CSV = os.path.join(DATA_RAW, "player_upcoming_fixtures.csv")
TEAMS_CSV    = os.path.join(DATA_RAW, "teams_raw.csv")

SIM_GWS      = 10   # simulate GW1-10
SIM_START    = 1    # force start at real GW1

SEP = "=" * 65


# ── Actual points lookup ──────────────────────────────────────────────────────
def build_actuals_index(ph):
    """Return dict: (player_id, gameweek) -> total_points"""
    idx = {}
    for _, row in ph.iterrows():
        idx[(int(row["player_id"]), int(row["gameweek"]))] = int(row["total_points"])
    return idx


def actual_gw_score(xi, captain, vice, chip, actuals_idx, gw, bench_gk, bench_out):
    """
    Compute actual FPL points for one GW.
    Includes:
      - Starting XI points
      - Captain doubling (vice if captain blanked)
      - Bench substitutions (first valid bench player for any 0-min XI player)
      - Bench boost: all bench players score too
    Returns (total, detail_dict)
    """
    def pts(pid):
        return actuals_idx.get((pid, gw), 0)

    xi_ids  = [p["player_id"] for p in xi]
    cap_id  = captain["player_id"] if captain else -1
    vic_id  = vice["player_id"]    if vice    else -1
    is_bb   = (chip == "bench_boost")

    # Starting XI base points
    xi_pts = {pid: pts(pid) for pid in xi_ids}
    total  = sum(xi_pts.values())

    # Simple bench sub: if any XI player got 0 pts (likely DNP), sub in bench
    # We approximate: 1 pt = played but blanked; 0 pts = DNP
    bench_outfield = [p["player_id"] for p in bench_out]
    bench_gk_id    = bench_gk[0]["player_id"] if bench_gk else None

    # Build subs queue (ordered)
    sub_queue = bench_outfield[:]

    for xi_p in xi:
        pid = xi_p["player_id"]
        if xi_pts[pid] == 0:
            # Try to sub from bench
            if xi_p["pos_id"] == 1 and bench_gk_id:
                sub_pts = pts(bench_gk_id)
                total += sub_pts
                xi_pts[pid] = sub_pts  # update for captain logic
            else:
                for sub_pid in sub_queue:
                    if pts(sub_pid) > 0:
                        sub_pts = pts(sub_pid)
                        total += sub_pts
                        xi_pts[pid] = sub_pts
                        sub_queue.remove(sub_pid)
                        break

    # Bench boost: add all bench players
    if is_bb:
        if bench_gk_id:
            total += pts(bench_gk_id)
        for pid in bench_outfield:
            total += pts(pid)

    # Captain doubling
    cap_pts = xi_pts.get(cap_id, 0)
    if cap_pts > 1:
        total += cap_pts  # captain played -> double
    else:
        # Captain blanked/DNP -> vice doubles
        vc_pts = xi_pts.get(vic_id, 0)
        total += vc_pts

    return total, xi_pts


# ── Main back-test ────────────────────────────────────────────────────────────
def main():
    print(SEP)
    print("  FPL AI Stage 8 -- BACK-TEST vs Real 2025-26 GW1-10 Actuals")
    print(SEP)

    # Load models + data
    print("\nLoading models...")
    models, model_fcols = load_models()
    best_params = load_best_params()

    print("Loading training data...")
    train_dfs = load_training_data()

    print("Loading FPL API data...")
    players  = pd.read_csv(PLAYERS_CSV)
    ph       = pd.read_csv(HISTORY_CSV)
    fixtures = pd.read_csv(FIXTURES_CSV)
    upcoming = pd.read_csv(UPCOMING_CSV)
    teams    = pd.read_csv(TEAMS_CSV)

    # Index of actual points
    actuals_idx = build_actuals_index(ph)
    print(f"Actual points index built: {len(actuals_idx)} (player,GW) entries")

    # DGW/BGW detection
    dgw_gws      = detect_dgw(fixtures)
    all_team_ids = list(teams["id"].unique())
    all_gws      = list(range(1, 39))
    bgw_by_gw    = detect_bgw(fixtures, all_gws, all_team_ids)

    # Build player pool (using training data feature averages)
    print(f"\nBuilding player pool (sim_start_gw={SIM_START})...")
    player_pool = build_player_pool(
        players, train_dfs, upcoming, teams, sim_start_gw=SIM_START
    )

    # Normalise upcoming
    upcoming = upcoming.dropna(subset=["gameweek"]).copy()
    upcoming["gameweek"] = upcoming["gameweek"].astype(int)

    # ── Simulation state ─────────────────────────────────────────────────────
    current_squad   = None
    free_transfers  = 1
    chips_used      = set()
    new_rows_by_pos = defaultdict(list)

    gw_results = []

    for gw in range(1, SIM_GWS + 1):
        real_gw = SIM_START + (gw - 1)
        print(f"\n{'-'*65}")
        print(f"  Processing sim-GW{gw} (real GW{real_gw})")
        print(f"{'-'*65}")

        pred_gw0   = predict_gw0(player_pool, models, model_fcols)
        horizon    = predict_horizon(
            player_pool, models, real_gw=real_gw,
            upcoming_df=upcoming, dgw_gws=dgw_gws, bgw_teams_by_gw=bgw_by_gw,
            model_fcols=model_fcols,
        )

        # Chip logic (simplified: wildcard GW2 if >= 4 underperformers)
        chip        = None
        is_wildcard = False
        is_freehit  = False

        if current_squad is not None:
            # freehit on DGW
            if "freehit" not in chips_used and real_gw in dgw_gws:
                chip = "freehit"; is_freehit = True
                chips_used.add("freehit")
                free_transfers = 15
            else:
                half   = 1 if real_gw <= 19 else 2
                wc_key = f"wildcard_{half}"
                if wc_key not in chips_used:
                    pos_avgs = {}
                    for pid in [1,2,3,4]:
                        vals = [pred_gw0.get(p["player_id"],0)
                                for p in current_squad if p["pos_id"]==pid]
                        pos_avgs[pid] = np.mean(vals) if vals else 0
                    under = sum(1 for p in current_squad
                                if pred_gw0.get(p["player_id"],0) < pos_avgs.get(p["pos_id"],0))
                    if under >= 4:
                        chip = wc_key; is_wildcard = True
                        chips_used.add(wc_key)
                        free_transfers = 15

        result = run_ilp(
            player_pool     = player_pool,
            horizon_scores  = horizon,
            pred_gw0_scores = pred_gw0,
            budget          = BUDGET,
            prev_squad      = current_squad if gw > 1 else None,
            free_transfers  = free_transfers if gw > 1 else 1,
            is_wildcard     = is_wildcard,
            is_freehit      = is_freehit,
        )

        if result is None:
            print(f"  GW{gw}: ILP failed, skipping.")
            continue

        if not is_freehit:
            current_squad = result["squad"]

        xi      = result["xi"]
        captain = result["captain"]
        vice    = result["vice"]
        penalty = result["penalty"]

        bench_gk, bench_out = split_bench(result["squad"], xi, pred_gw0)

        # Predicted score
        xi_preds  = [pred_gw0.get(p["player_id"], 0) for p in xi]
        cap_mult  = 3 if chip == "triple_captain" else 2
        bb_pts    = sum(pred_gw0.get(p["player_id"],0)
                        for p in bench_gk+bench_out) if chip=="bench_boost" else 0
        cap_pred  = pred_gw0.get(captain["player_id"],0) if captain else 0
        vc_pred   = pred_gw0.get(vice["player_id"],0)*0.5 if vice else 0
        pred_score = sum(xi_preds) + cap_pred + vc_pred + bb_pts - penalty*4

        # Actual score from player_history.csv
        actual_score, xi_actual_pts = actual_gw_score(
            xi, captain, vice, chip, actuals_idx, real_gw, bench_gk, bench_out
        )
        actual_score -= penalty * 4  # apply same penalty

        # Print GW block
        print(f"\n  CHIP: {chip or 'None'}")
        print(f"  {'Player':<22} {'Team':<5} {'Pred':>5}  {'Actual':>6}  Flags")
        print(f"  {'-'*55}")

        cap_id = captain["player_id"] if captain else -1
        vic_id = vice["player_id"]    if vice    else -1

        xi_gk  = [p for p in xi if p["pos_id"]==1]
        xi_def = [p for p in xi if p["pos_id"]==2]
        xi_mid = [p for p in xi if p["pos_id"]==3]
        xi_fwd = [p for p in xi if p["pos_id"]==4]

        def safe(s):
            return s.encode("ascii", "replace").decode("ascii")

        for group_label, group in [("GK",xi_gk),("DEF",xi_def),("MID",xi_mid),("FWD",xi_fwd)]:
            for p in group:
                pid   = p["player_id"]
                pr    = pred_gw0.get(pid,0)
                act   = actuals_idx.get((pid, real_gw), "N/A")
                flags = ""
                if pid == cap_id: flags += " [C]"
                if pid == vic_id: flags += " [V]"
                print(f"  {group_label}: {safe(p['web_name']):<20} {p['team_short']:<5} {pr:5.1f}  {str(act):>6}{flags}")

        def safe(s):
            return s.encode("ascii", "replace").decode("ascii")

        print(f"\n  BENCH:")
        for p in bench_gk:
            pid = p["player_id"]
            pr  = pred_gw0.get(pid, 0)
            act = actuals_idx.get((pid, real_gw), "N/A")
            print(f"  GK:  {safe(p['web_name']):<20} {p['team_short']:<5} {pr:5.1f}  {str(act):>6}")
        for rank, p in enumerate(bench_out[:3], 1):
            pid = p["player_id"]
            pr  = pred_gw0.get(pid, 0)
            act = actuals_idx.get((pid, real_gw), "N/A")
            print(f"  B{rank}:  {safe(p['web_name']):<20} {p['team_short']:<5} {pr:5.1f}  {str(act):>6}")

        print(f"\n  Transfers: {len(result['transfers_in'])} in / {len(result['transfers_out'])} out, penalty={penalty}")
        if result["transfers_in"]:
            for p in result["transfers_in"]:
                print(f"    IN : {safe(p['web_name'])} ({p['team_short']})")
            for p in result["transfers_out"]:
                print(f"    OUT: {safe(p['web_name'])} ({p['team_short']})")
        print(f"\n  Predicted score : {pred_score:.1f}")
        print(f"  Actual score    : {actual_score}")

        cap_name = safe(captain["web_name"]) if captain else "-"
        cap_pos  = captain["pos_name"] if captain else "-"

        gw_results.append({
            "gw":        gw,
            "real_gw":   real_gw,
            "chip":      chip,
            "predicted": round(pred_score, 1),
            "actual":    actual_score,
            "penalty":   penalty,
            "captain":   cap_name,
            "cap_pos":   cap_pos,
        })

        # Online retraining with REAL actuals
        if gw < SIM_GWS:
            print(f"\n  Retraining with real GW{real_gw} actuals...")
            for p in player_pool:
                pid    = p["player_id"]
                pos_id = p["pos_id"]
                row    = p["feat_vec"].copy()
                row[TARGET_COL] = actuals_idx.get((pid, real_gw), 0)
                row["season"]   = "2025-26"
                row["GW"]       = real_gw
                new_rows_by_pos[pos_id].append(row)
            models, model_fcols = retrain_models(
                train_dfs, new_rows_by_pos, best_params, model_fcols
            )

        # Update free transfers
        if is_wildcard or is_freehit:
            free_transfers = 1
        else:
            n_tr = len(result["transfers_in"])
            if n_tr <= free_transfers:
                free_transfers = min(2, free_transfers - n_tr + 1)
            else:
                free_transfers = 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total_pred   = sum(r["predicted"] for r in gw_results)
    total_actual = sum(r["actual"]    for r in gw_results)

    print()
    print(SEP)
    print("  BACK-TEST SUMMARY  (sim-GW1-10  =  real GW1-10)")
    print(SEP)
    print(f"  {'GW':<5} {'Real GW':<9} {'Chip':<15} {'Pred':>7} {'Actual':>7} {'Diff':>7}  {'Captain':<20} {'Pos':<4}")
    print(f"  {'-'*75}")
    for r in gw_results:
        diff = r["actual"] - r["predicted"]
        print(f"  {r['gw']:<5} {r['real_gw']:<9} {str(r['chip'] or '-'):<15} "
              f"{r['predicted']:>7.1f} {r['actual']:>7}  {diff:>+7.1f}  {r['captain']:<20} {r['cap_pos']:<4}")
    print(f"  {'-'*75}")
    print(f"  {'TOTAL':<14}              {total_pred:>7.1f} {total_actual:>7}")
    print()
    print(f"  Predicted total  : {total_pred:.1f} pts")
    print(f"  Actual total     : {total_actual} pts")
    print(f"  Difference       : {total_actual - total_pred:+.1f} pts")
    print(f"  Accuracy         : {total_actual/total_pred*100:.1f}% of predicted")

    # Captain position breakdown
    from collections import Counter
    cap_pos_counts = Counter(r["cap_pos"] for r in gw_results if r["cap_pos"] != "-")
    breakdown = "  ".join(f"{pos} x{cnt}" for pos, cnt in sorted(cap_pos_counts.items()))
    print(f"  Captain breakdown: {breakdown}")
    print(SEP)

    # Save JSON
    out = {
        "sim_start_gw": SIM_START,
        "gameweeks": gw_results,
        "total_predicted": round(total_pred, 1),
        "total_actual": total_actual,
    }
    out_path = os.path.join(MODELS_DIR, "stage8_backtest_patched.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBacktest saved -> {out_path}")


if __name__ == "__main__":
    main()
