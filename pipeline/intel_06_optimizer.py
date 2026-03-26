"""
Intel 06: Stage 8 Enhanced with Pre-Deadline Intelligence
FPL AI Thesis

Runs the exact same Stage 8 pipeline (XGBoost predictions, ILP optimizer,
captain selection, online retraining) but injects intel penalties between
the prediction step and the ILP step.

The ILP naturally avoids injured/rotating players without extra LLM calls.

Output: data/intel/final_squad.json
"""

import os
import sys
import json
import warnings
import copy
import io
import contextlib
import pickle
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Import Stage 8 functions (zero code duplication)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ilp_optimizer_stage8 import (
    load_models, load_best_params, load_training_data, load_fpl_data,
    detect_dgw, detect_bgw, get_sim_start_gw,
    build_player_pool, predict_gw0, predict_horizon,
    run_ilp, split_bench, select_captain, check_chips,
    retrain_models, print_gw_block,
    BUDGET, TARGET_COL, SEP,
)
from pipeline.ilp_optimizer_stage8 import CAP_POS_MULTIPLIERS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_INTEL = os.path.join(ROOT, "data", "intel")
MODELS_DIR = os.path.join(ROOT, "models")

INTEL_AVAIL  = os.path.join(DATA_INTEL, "availability.json")
INTEL_ROT    = os.path.join(DATA_INTEL, "rotation_risk.json")
INTEL_RECS   = os.path.join(DATA_INTEL, "recommendations.json")
OUTPUT_PATH  = os.path.join(DATA_INTEL, "final_squad.json")


# ===========================================================================
# GW-Snapshot: lookup builders
# ===========================================================================
DATA_RAW = os.path.join(ROOT, "data", "raw", "fpl_api")


def build_history_lookup():
    """
    Load player_history.csv into {player_id: {gw: {stat_dict}}}.
    Each stat_dict has: total_points, minutes, goals_scored, assists,
    clean_sheets, saves, bonus, value, was_home, opponent_team_id.
    """
    path = os.path.join(DATA_RAW, "player_history.csv")
    df = pd.read_csv(path)
    lookup = defaultdict(dict)
    for _, row in df.iterrows():
        pid = int(row["player_id"])
        gw = int(row["gameweek"])
        lookup[pid][gw] = {
            "total_points":     int(row["total_points"]),
            "minutes":          int(row["minutes"]),
            "goals_scored":     int(row["goals_scored"]),
            "assists":          int(row["assists"]),
            "clean_sheets":     int(row["clean_sheets"]),
            "saves":            int(row["saves"]),
            "bonus":            int(row["bonus"]),
            "value":            float(row["value"]),
            "was_home":         bool(row["was_home"]) if isinstance(row["was_home"], (bool, np.bool_)) else str(row["was_home"]).strip().lower() == "true",
            "opponent_team_id": int(row["opponent_team_id"]),
        }
    print(f"  player_history lookup: {len(lookup)} players")
    return dict(lookup)


def build_fdr_lookup():
    """
    Load fixtures_raw.csv into {(team_id, gw): fdr}.
    Home team uses team_h_difficulty, away team uses team_a_difficulty.
    """
    path = os.path.join(DATA_RAW, "fixtures_raw.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=["gameweek"])
    lookup = {}
    for _, row in df.iterrows():
        gw = int(row["gameweek"])
        lookup[(int(row["team_h"]), gw)] = int(row["team_h_difficulty"])
        lookup[(int(row["team_a"]), gw)] = int(row["team_a_difficulty"])
    print(f"  FDR lookup: {len(lookup)} team-GW entries")
    return lookup


# ===========================================================================
# GW-Snapshot: refresh player features after each completed GW
# ===========================================================================
def refresh_player_features(player_pool, hist_lookup, fdr_lookup, completed_gw):
    """
    After GW `completed_gw` has been played, update every player's feat_vec
    using actual stats from player_history.csv (GW1 through completed_gw).
    Also updates fixture features for the NEXT GW (completed_gw + 1).
    Flags players with zero total minutes as non-playing.
    """
    next_gw = completed_gw + 1
    updated = 0
    n_zero_mins = 0

    for p in player_pool:
        pid = p["player_id"]
        fv = p["feat_vec"]
        hist = hist_lookup.get(pid, {})

        # Collect actual stats for GW1 through completed_gw
        gw_stats = []
        for g in range(1, completed_gw + 1):
            if g in hist:
                gw_stats.append(hist[g])

        if not gw_stats:
            p["zero_minutes"] = True
            n_zero_mins += 1
            if "minutes_reliability_season" in fv:
                fv["minutes_reliability_season"] = 0.0
            if "form_last3" in fv:
                fv["form_last3"] = 0.0
            if "form_last5" in fv:
                fv["form_last5"] = 0.0
            continue

        # --- Recompute rolling player features ---
        all_pts     = [s["total_points"] for s in gw_stats]
        all_mins    = [s["minutes"] for s in gw_stats]
        all_goals   = [s["goals_scored"] for s in gw_stats]
        all_assists = [s["assists"] for s in gw_stats]
        all_cs      = [s["clean_sheets"] for s in gw_stats]
        all_saves   = [s["saves"] for s in gw_stats]

        played_stats = [s for s in gw_stats if s["minutes"] > 0]
        n_played = len(played_stats)

        # form_last3 / form_last5: mean of last N GW points (all GWs, not just played)
        if "form_last3" in fv:
            fv["form_last3"] = float(np.mean(all_pts[-3:])) if all_pts else 0.0
        if "form_last5" in fv:
            fv["form_last5"] = float(np.mean(all_pts[-5:])) if all_pts else 0.0

        # cumulative_points_season
        cum_pts = sum(all_pts)
        if "cumulative_points_season" in fv:
            fv["cumulative_points_season"] = float(cum_pts)

        # Expanding means (over GWs actually played, to avoid dilution from bench GWs)
        if n_played > 0:
            played_pts     = [s["total_points"] for s in played_stats]
            played_goals   = [s["goals_scored"] for s in played_stats]
            played_assists = [s["assists"] for s in played_stats]
            played_cs      = [s["clean_sheets"] for s in played_stats]
            played_saves   = [s["saves"] for s in played_stats]

            if "avg_points_per_game_season" in fv:
                fv["avg_points_per_game_season"] = float(np.mean(played_pts))
            if "goals_per_game_season" in fv:
                fv["goals_per_game_season"] = float(np.mean(played_goals))
            if "assists_per_game_season" in fv:
                fv["assists_per_game_season"] = float(np.mean(played_assists))
            if "clean_sheet_rate_season" in fv:
                fv["clean_sheet_rate_season"] = float(np.mean(played_cs))
            if "saves_per_game_season" in fv:
                fv["saves_per_game_season"] = float(np.mean(played_saves))
        else:
            for key in ["avg_points_per_game_season", "goals_per_game_season",
                        "assists_per_game_season", "clean_sheet_rate_season",
                        "saves_per_game_season"]:
                if key in fv:
                    fv[key] = 0.0

        # minutes_reliability_season: total_minutes / (completed_gws * 90)
        total_mins = sum(all_mins)
        if total_mins == 0:
            p["zero_minutes"] = True
            n_zero_mins += 1
        else:
            p["zero_minutes"] = False

        if "minutes_reliability_season" in fv:
            fv["minutes_reliability_season"] = min(
                1.0, total_mins / (completed_gw * 90)
            )

        # value & points_per_million (use most recent GW value)
        latest_value = gw_stats[-1]["value"]
        if "value" in fv:
            fv["value"] = latest_value
            p["value"] = latest_value
        if "points_per_million" in fv:
            fv["points_per_million"] = (
                cum_pts / (latest_value / 10.0) if latest_value > 0 else 0.0
            )

        # --- Update fixture features for the NEXT GW ---
        next_hist = hist.get(next_gw)
        if next_hist is not None:
            if "home_advantage" in fv:
                fv["home_advantage"] = 1 if next_hist["was_home"] else 0

            team_id = p["team_id"]
            fdr = fdr_lookup.get((team_id, next_gw))
            if fdr is not None and "current_gw_fdr" in fv:
                fv["current_gw_fdr"] = float(fdr)

        updated += 1

    print(f"  [SNAPSHOT] Refreshed features for {updated}/{len(player_pool)} "
          f"players using GW1-{completed_gw} actuals "
          f"({n_zero_mins} with zero minutes -> filtered)")


def apply_playing_filter(pred_gw0, horizon, player_pool):
    """
    Zero out predictions for players flagged as zero_minutes.
    Called after predict_gw0/predict_horizon, before ILP, from GW2 onwards.
    Returns (filtered_pred, filtered_horizon, n_filtered).
    """
    filtered_pred = dict(pred_gw0)
    filtered_horizon = dict(horizon)
    n_filtered = 0

    for p in player_pool:
        if p.get("zero_minutes", False):
            pid = p["player_id"]
            filtered_pred[pid] = 0.0
            filtered_horizon[pid] = 0.0
            n_filtered += 1

    return filtered_pred, filtered_horizon, n_filtered


# ===========================================================================
# Real actuals from player_history.csv (replaces simulate_actuals)
# ===========================================================================
def get_real_actuals(player_pool, hist_lookup, gw):
    """Return {player_id: actual_total_points} for the given GW from real data."""
    actuals = {}
    for p in player_pool:
        pid = p["player_id"]
        gw_data = hist_lookup.get(pid, {}).get(gw)
        actuals[pid] = gw_data["total_points"] if gw_data else 0
    return actuals


# ===========================================================================
# GW1 ownership boost: use community wisdom when no current-season data
# ===========================================================================
OWNERSHIP_WEIGHT = 0.04
ONGOING_OWNERSHIP_WEIGHT = 0.01  # GW2+ quality floor — adj_pred only


def apply_gw1_ownership_boost(pred_gw0, horizon, player_pool):
    """
    For GW1 only: boost predictions by selected_by_percent.
    Millions of FPL managers research pre-season — their ownership %
    is the strongest quality signal before a ball is kicked.
    Haaland (61.5%) gets +2.5 pts, Lecomte (0.7%) gets +0.03.
    """
    adj_pred = dict(pred_gw0)
    adj_horizon = dict(horizon)
    for p in player_pool:
        pid = p["player_id"]
        own_pct = p["feat_vec"].get("selected", 0.0)
        boost = own_pct * OWNERSHIP_WEIGHT
        adj_pred[pid] = pred_gw0.get(pid, 0) + boost
        adj_horizon[pid] = horizon.get(pid, 0) + boost
    return adj_pred, adj_horizon


def fix_zero_minutes_in_xi(result, adj_pred, zero_minutes_ids):
    """
    Post-ILP fix: if any zero-minutes outfield player ended up in the starting XI
    (because the bench quality constraint forced them to start), swap them out
    with the highest adj_pred eligible bench player.
    Does NOT change squad composition, transfers, or any ILP decision — only the
    final XI assignment.
    """
    if not zero_minutes_ids:
        return result
    xi = list(result["xi"])
    squad_ids = {p["player_id"] for p in result["squad"]}
    xi_ids = {p["player_id"] for p in xi}
    bench_out = [p for p in result["squad"]
                 if p["player_id"] not in xi_ids and p["pos_id"] != 1]

    swapped = []
    for out_p in [p for p in xi if p["player_id"] in zero_minutes_ids and p["pos_id"] != 1]:
        # Find best eligible bench replacement: same position preferred, then adj_pred desc
        candidates = sorted(
            [p for p in bench_out if p["player_id"] not in zero_minutes_ids],
            key=lambda p: (
                0 if p["pos_id"] == out_p["pos_id"] else 1,  # same position first
                -adj_pred.get(p["player_id"], 0),
            ),
        )
        for sub in candidates:
            # Validate formation after swap
            test_xi = [p for p in xi if p["player_id"] != out_p["player_id"]] + [sub]
            n_def = sum(1 for p in test_xi if p["pos_id"] == 2)
            n_fwd = sum(1 for p in test_xi if p["pos_id"] == 4)
            if n_def >= 3 and n_fwd >= 1:
                xi = test_xi
                xi_ids = {p["player_id"] for p in xi}
                bench_out = [p for p in result["squad"]
                             if p["player_id"] not in xi_ids and p["pos_id"] != 1]
                swapped.append((out_p["web_name"], sub["web_name"]))
                break
    if swapped:
        for out_name, in_name in swapped:
            print(f"  [ZERO-MIN] Benched {out_name} (0 min) -> started {in_name}")
        result = dict(result)
        result["xi"] = xi
    return result


def apply_ongoing_ownership_boost(pred_gw0, player_pool):
    """
    GW2+ quality floor: small ownership-weighted boost applied to adj_pred only.
    Prevents elite high-ownership players from being predicted as mediocre
    during brief form dips (e.g. Salah at 2.6 pts after three 2-pt weeks).
    adj_horizon is NOT modified — transfer decisions are unaffected.
    """
    adj_pred = dict(pred_gw0)
    for p in player_pool:
        pid = p["player_id"]
        own_pct = p["feat_vec"].get("selected", 0.0)
        boost = own_pct * ONGOING_OWNERSHIP_WEIGHT
        adj_pred[pid] = pred_gw0.get(pid, 0) + boost
    return adj_pred


# ===========================================================================
# Stage 9 logic: FDR adjustment
# ===========================================================================
FDR_MULT = 0.03


def apply_fdr_adjustment(pred_gw0, player_pool, hp=None):
    """
    Apply Stage 9's FDR multiplier to GW0 predictions.
    adj = raw * (1.0 - fdr_mult * (fdr - 3.0))
    Easy fixtures (FDR 1-2) get a boost, hard fixtures (FDR 4-5) get a cut.
    """
    hp = hp or {}
    fdr_mult = hp.get('fdr_mult', FDR_MULT)
    adjusted = dict(pred_gw0)
    for p in player_pool:
        pid = p["player_id"]
        fdr = p["feat_vec"].get("current_gw_fdr", 3.0)
        fdr_adj = 1.0 - fdr_mult * (fdr - 3.0)
        adjusted[pid] = pred_gw0.get(pid, 0) * fdr_adj
    return adjusted


# ===========================================================================
# Squad loyalty bonus (reduces unnecessary transfers)
# ===========================================================================
MAX_HITS = 1  # max transfer hits before auto-triggering Free Hit


def get_loyalty_bonus(gw, hp=None):
    """
    High loyalty during chip lockout prevents hits and enables FT banking.
    gw=0:  20.0 (emergency re-run — force minimal transfers)
    GW2-4: 10.0 (virtually no hits — bank free transfers instead)
    GW5-6: 2.0  (moderate — allow controlled transfers)
    GW7+:  1.0  (baseline)
    """
    hp = hp or {}
    if gw == 0:
        return 20.0
    if gw <= CHIP_LOCKOUT_GW:
        return hp.get('loyalty_lockout', 10.0)
    if gw <= 6:
        return hp.get('loyalty_mid', 2.0)
    return hp.get('loyalty_late', 1.0)


def apply_squad_loyalty(horizon, current_squad, gw, hp=None):
    """
    Add a loyalty bonus to horizon scores for existing squad members.
    Bonus is higher in early GWs to prevent panic transfers when the model
    has limited current-season data.
    """
    if not current_squad:
        return horizon
    bonus = get_loyalty_bonus(gw, hp=hp)
    boosted = dict(horizon)
    squad_pids = {p["player_id"] for p in current_squad}
    for pid in squad_pids:
        if pid in boosted:
            boosted[pid] += bonus
    return boosted


# ===========================================================================
# Improved chip timing
# ===========================================================================
CHIP_LOCKOUT_GW = 4


def check_chips_improved(gw, real_gw, dgw_gws, chips_used,
                         current_squad, pred_gw0, bench_out):
    """
    Wrapper around Stage 8's check_chips with smarter timing:
    - Block ALL chips in GW1-4 (build a stable squad first)
    - Save bench_boost and triple_captain for DGWs
    """
    if gw <= CHIP_LOCKOUT_GW:
        return None
    raw_chip = check_chips(real_gw, dgw_gws, chips_used,
                           current_squad, pred_gw0, bench_out)
    if raw_chip == "bench_boost" and real_gw not in dgw_gws:
        return None
    if raw_chip == "triple_captain" and real_gw not in dgw_gws:
        return None
    return raw_chip


# ===========================================================================
# Bench ordering optimization
# ===========================================================================
def optimize_bench_order(result, pred_gw0):
    """Sort bench outfield players by predicted points (highest = 1st sub)."""
    bench_outfield = [p for p in result["squad"]
                      if p not in result["xi"] and p["pos_id"] != 1]
    bench_outfield.sort(
        key=lambda p: pred_gw0.get(p["player_id"], 0), reverse=True
    )
    result["bench_outfield"] = bench_outfield


# ===========================================================================
# Auto-sub simulation (mirrors real FPL rules)
# ===========================================================================
def simulate_auto_subs(xi, squad, hist_lookup, gw):
    """
    If a starter got 0 minutes, sub in the first eligible bench player.
    FPL formation rules: min 3 DEF, min 1 FWD after subs.
    GK sub: bench GK replaces starting GK if starting GK got 0 mins.
    Returns (final_xi, subs_made) where subs_made is list of (out, in) tuples.
    """
    xi_set = set(id(p) for p in xi)
    bench_gk = [p for p in squad if id(p) not in xi_set and p["pos_id"] == 1]
    bench_out = [p for p in squad if id(p) not in xi_set and p["pos_id"] != 1]

    final_xi = list(xi)
    subs_made = []

    def pos_count(eleven, pos_id):
        return sum(1 for p in eleven if p["pos_id"] == pos_id)

    def player_minutes(p):
        return hist_lookup.get(p["player_id"], {}).get(gw, {}).get("minutes", 0)

    xi_gk = [p for p in final_xi if p["pos_id"] == 1]
    if xi_gk and player_minutes(xi_gk[0]) == 0 and bench_gk:
        old_gk = xi_gk[0]
        new_gk = bench_gk[0]
        if player_minutes(new_gk) > 0:
            final_xi = [new_gk if id(p) == id(old_gk) else p for p in final_xi]
            subs_made.append((old_gk, new_gk))
            bench_gk = bench_gk[1:]

    used_bench = set()
    for starter in list(final_xi):
        if starter["pos_id"] == 1:
            continue
        if player_minutes(starter) > 0:
            continue
        for j, sub in enumerate(bench_out):
            if j in used_bench:
                continue
            if player_minutes(sub) == 0:
                continue
            test_xi = [sub if id(p) == id(starter) else p for p in final_xi]
            if pos_count(test_xi, 2) < 3:
                continue
            if pos_count(test_xi, 4) < 1:
                continue
            final_xi = test_xi
            subs_made.append((starter, sub))
            used_bench.add(j)
            break

    return final_xi, subs_made


# ===========================================================================
# Load intel data
# ===========================================================================
def load_intel_data():
    """Load availability, rotation risk, and recommendations JSONs."""
    print("\nLoading intel data...")

    with open(INTEL_AVAIL, encoding="utf-8") as f:
        avail = json.load(f)
    n_avail_gws = len(avail.get("gameweeks", {}))
    print(f"  availability.json: {n_avail_gws} GWs")

    with open(INTEL_ROT, encoding="utf-8") as f:
        rot = json.load(f)
    n_rot_gws = len(rot.get("gameweeks", {}))
    print(f"  rotation_risk.json: {n_rot_gws} GWs")

    recs = None
    if os.path.exists(INTEL_RECS):
        with open(INTEL_RECS, encoding="utf-8") as f:
            recs = json.load(f)
        n_rec_gws = len(recs.get("gameweeks", {}))
        print(f"  recommendations.json: {n_rec_gws} GWs")
    else:
        print("  recommendations.json: not found (captain override disabled)")

    return avail, rot, recs


# ===========================================================================
# Intel penalty application
# ===========================================================================
def apply_intel_penalties(pred_gw0, horizon, player_pool, avail_data, rot_data, gw):
    """
    Adjust XGBoost predictions using availability and rotation risk data.
    Returns (adjusted_pred, adjusted_horizon) dicts and a stats dict.
    """
    gw_str = str(gw)
    avail_players = avail_data.get("gameweeks", {}).get(gw_str, {}).get("players", {})
    rot_players   = rot_data.get("gameweeks", {}).get(gw_str, {}).get("players", {})

    adjusted_pred    = dict(pred_gw0)
    adjusted_horizon = dict(horizon)

    n_penalized = 0
    biggest = {"name": None, "avail": 95, "rot_risk": None, "combined_mult": 1.0}

    for p in player_pool:
        pid     = p["player_id"]
        pid_str = str(pid)

        avail_entry = avail_players.get(pid_str)
        avail_pct   = avail_entry["availability_pct"] if avail_entry else 95
        avail_mult  = avail_pct / 100.0

        rot_entry = rot_players.get(pid_str)
        rot_risk  = rot_entry["rotation_risk"] if rot_entry else None
        rot_mult  = 1.0
        if rot_risk is not None:
            if rot_risk >= 80:
                rot_mult = 0.40
            elif rot_risk >= 60:
                rot_mult = 0.60
            elif rot_risk >= 40:
                rot_mult = 0.80

        combined = avail_mult * rot_mult
        if combined < 1.0:
            adjusted_pred[pid]    = pred_gw0.get(pid, 0) * combined
            adjusted_horizon[pid] = horizon.get(pid, 0) * combined
            n_penalized += 1

            if combined < biggest["combined_mult"]:
                biggest = {
                    "name":          p.get("web_name", "?"),
                    "avail":         avail_pct,
                    "rot_risk":      rot_risk,
                    "combined_mult": round(combined, 3),
                }

    stats = {
        "players_penalized": n_penalized,
        "biggest_penalty":   biggest if biggest["name"] else None,
    }
    return adjusted_pred, adjusted_horizon, stats


# ===========================================================================
# Captain override from intel_05 recommendations
# ===========================================================================
CAPTAIN_OVERRIDE_THRESHOLD = 0.5


def apply_captain_override(result, recommendations, gw, pred_gw0):
    """
    If intel_05 recommended a different captain AND that player is in the XI
    AND their adjusted score is >= 0.5 pts higher than the ILP captain,
    override the ILP captain. Returns the source string.
    """
    if recommendations is None:
        return "ilp"

    gw_recs = recommendations.get("gameweeks", {}).get(str(gw), {})
    intel_cap = gw_recs.get("decisions", {}).get("captain", {}).get("name")
    if not intel_cap:
        return "ilp"

    current_cap = result["captain"]["web_name"] if result["captain"] else None
    if intel_cap == current_cap:
        return "ilp+intel_05"

    xi_by_name = {}
    for p in result["xi"]:
        xi_by_name[p["web_name"]] = p
        xi_by_name[p.get("name", "")] = p

    new_cap = xi_by_name.get(intel_cap)
    if new_cap:
        ilp_cap_score = pred_gw0.get(result["captain"]["player_id"], 0)
        new_cap_score = pred_gw0.get(new_cap["player_id"], 0)
        if new_cap_score >= ilp_cap_score + CAPTAIN_OVERRIDE_THRESHOLD:
            result["captain"] = new_cap
            return "intel_05"

    return "ilp"


# ===========================================================================
# Data loading (extracted from main for HPO reuse)
# ===========================================================================
def load_all_data(verbose=True, sim_start_override=None):
    """
    Load all static data needed for simulation.
    Returns a dict with all resources so run_simulation can be called
    multiple times without re-loading.

    Parameters
    ----------
    verbose : bool
        If True, print loading messages (default True).
    sim_start_override : int or None
        If not None, use this value as sim_start_gw instead of auto-detection.
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint("=" * 60)
    vprint("  Intel 06: Stage 8 + Pre-Deadline Intelligence")
    vprint("=" * 60)

    # --- Load Stage 8 resources ---
    vprint("\nLoading models...")
    models, model_fcols = load_models()
    best_params = load_best_params()

    vprint("\nLoading training data...")
    train_dfs = load_training_data()

    vprint("\nLoading FPL API data...")
    players, ph, fixtures, fdr_df, upcoming, teams = load_fpl_data()

    vprint("\nDetecting DGWs and BGWs...")
    dgw_gws      = detect_dgw(fixtures)
    all_team_ids  = list(teams["id"].unique())
    all_gws       = list(range(1, 39))
    bgw_by_gw     = detect_bgw(fixtures, all_gws, all_team_ids)
    vprint(f"  DGW gameweeks: {sorted(dgw_gws.keys()) or 'None found'}")

    if sim_start_override is not None:
        sim_start_gw = sim_start_override
        vprint(f"  sim_start_gw overridden to {sim_start_gw}")
    else:
        sim_start_gw = get_sim_start_gw(fixtures, upcoming)
    vprint(f"  Simulation starts at real GW{sim_start_gw} (maps to sim GW1)")

    vprint("\nBuilding player pool...")
    player_pool = build_player_pool(players, train_dfs, upcoming, teams,
                                    sim_start_gw=sim_start_gw)

    upcoming = upcoming.dropna(subset=["gameweek"]).copy()
    upcoming["gameweek"] = upcoming["gameweek"].astype(int)

    # --- Build GW-snapshot lookups ---
    vprint("\nBuilding GW-snapshot lookups...")
    hist_lookup = build_history_lookup()
    fdr_lookup  = build_fdr_lookup()

    # --- Load intel data ---
    avail_data, rot_data, recommendations = load_intel_data()

    # =======================================================================
    # GW1 value fix: use actual starting prices instead of GW29 now_cost
    # =======================================================================
    gw1_value_fixes = 0
    for p in player_pool:
        pid = p["player_id"]
        gw1_hist = hist_lookup.get(pid, {}).get(1)
        if gw1_hist and "value" in p["feat_vec"]:
            p["feat_vec"]["value"] = gw1_hist["value"]
            p["value"] = gw1_hist["value"]
            gw1_value_fixes += 1
    vprint(f"\n  GW1 value fix: updated {gw1_value_fixes} player prices to actual GW1 values")

    return {
        "models":          models,
        "model_fcols":     model_fcols,
        "best_params":     best_params,
        "train_dfs":       train_dfs,
        "player_pool":     player_pool,
        "upcoming":        upcoming,
        "dgw_gws":         dgw_gws,
        "bgw_by_gw":       bgw_by_gw,
        "hist_lookup":     hist_lookup,
        "fdr_lookup":      fdr_lookup,
        "avail_data":      avail_data,
        "rot_data":        rot_data,
        "recommendations": recommendations,
        "sim_start_gw":    sim_start_gw,
    }


# ===========================================================================
# Simulation (extracted from main for HPO reuse)
# ===========================================================================
def run_simulation(data, hp=None, n_gws=10, verbose=True, save_json=True,
                   fresh_squad_gw=None):
    """
    Run the intel-enhanced GW simulation.

    Parameters
    ----------
    data : dict
        Output of load_all_data().
    hp : dict or None
        Hyperparameter overrides. Keys: bb_threshold, tc_threshold, fdr_mult,
        loyalty_lockout, loyalty_mid, loyalty_late.
    n_gws : int
        Number of GWs to simulate (default 10).
    verbose : bool
        If True, print progress (default True).
    save_json : bool
        If True, save final_squad.json (default True).
    fresh_squad_gw : int or None
        If set, this GW ignores the accumulated squad and runs the ILP with
        no prev_squad (wildcard-style free pick). Used for "best possible team"
        predictions where squad continuity is not required.

    Returns
    -------
    int
        Total actual points scored across all simulated GWs.
    """
    hp = hp or {}

    # Deep-copy mutable state so multiple calls don't contaminate each other
    player_pool  = copy.deepcopy(data['player_pool'])
    models       = {k: pickle.loads(pickle.dumps(v)) for k, v in data['models'].items()}
    model_fcols  = copy.deepcopy(data['model_fcols'])
    best_params  = copy.deepcopy(data['best_params'])
    train_dfs    = {k: v.copy() for k, v in data['train_dfs'].items()}

    # Read-only — safe to share directly
    sim_start_gw    = data['sim_start_gw']
    upcoming        = data['upcoming']
    dgw_gws         = data['dgw_gws']
    bgw_by_gw       = data['bgw_by_gw']
    hist_lookup     = data['hist_lookup']
    fdr_lookup      = data['fdr_lookup']
    avail_data      = data['avail_data']
    rot_data        = data['rot_data']
    recommendations = data['recommendations']

    def _run():
        nonlocal models, model_fcols

        # --- Simulation state ---
        current_squad    = None
        free_transfers   = 1
        chips_used       = set()
        chips_log        = []
        gw_scores        = []
        actual_scores    = []
        squad_values     = []
        penalties_total  = 0
        captain_counts   = defaultdict(int)
        best_gw_info     = {"gw": None, "score": -999, "squad": []}
        new_rows_by_pos  = defaultdict(list)
        captain_sources  = []
        sim_log = {
            "generated_at":           datetime.now(timezone.utc).isoformat(),
            "mode":                   "backtest",
            "model":                  "xgboost + intel_pipeline",
            "gameweeks":              [],
            "total_predicted_points": 0,
            "total_penalties":        0,
            "chips_used":             [],
            "best_gw":                None,
            "most_captained":         None,
            "squad_value_trajectory": [],
        }

        # ===================================================================
        # GW loop
        # ===================================================================
        for gw in range(1, n_gws + 1):
            print(f"\n{'-'*60}")
            print(f"  Processing GW{gw} (Intel-Enhanced)...")
            print(f"{'-'*60}")

            real_gw = sim_start_gw + (gw - 1)

            # --- Predict GW0 ---
            pred_gw0_scores = predict_gw0(player_pool, models, model_fcols)

            # --- 3-GW horizon ---
            horizon_scores = predict_horizon(
                player_pool, models, real_gw=real_gw,
                upcoming_df=upcoming, dgw_gws=dgw_gws, bgw_teams_by_gw=bgw_by_gw,
                model_fcols=model_fcols,
            )

            # === PLAYING FILTER: zero out non-playing players (GW2+) ===
            if gw > 1:
                pred_gw0_scores, horizon_scores, n_filt = apply_playing_filter(
                    pred_gw0_scores, horizon_scores, player_pool,
                )
                if n_filt > 0:
                    print(f"  [FILTER] {n_filt} zero-minute players zeroed out")

            # === INTEL ENHANCEMENT ===
            adj_pred, adj_horizon, intel_stats = apply_intel_penalties(
                pred_gw0_scores, horizon_scores, player_pool,
                avail_data, rot_data, gw,
            )
            if intel_stats["players_penalized"] > 0:
                bp = intel_stats["biggest_penalty"]
                print(f"  [INTEL] {intel_stats['players_penalized']} players penalized"
                      f" | biggest: {bp['name']} (avail={bp['avail']}%,"
                      f" rot={bp['rot_risk']}, mult={bp['combined_mult']})")
            else:
                print(f"  [INTEL] No penalties applied this GW")

            # === OWNERSHIP BOOST (quality floor signal) ===
            if gw == 1:
                adj_pred, adj_horizon = apply_gw1_ownership_boost(
                    adj_pred, adj_horizon, player_pool,
                )
                print(f"  [OWNERSHIP] GW1 boost applied (weight={OWNERSHIP_WEIGHT})")

            # === FDR ADJUSTMENT (Stage 9 logic) ===
            adj_pred = apply_fdr_adjustment(adj_pred, player_pool, hp=hp)

            # === SQUAD LOYALTY (reduce transfer churn) ===
            adj_horizon = apply_squad_loyalty(adj_horizon, current_squad, gw, hp=hp)

            # adj_pred_ilp: passed to ILP (no ongoing ownership boost — avoids cascading
            # through bench quality constraints and ILP objective terms)
            adj_pred_ilp = adj_pred

            # adj_pred: used for post-ILP decisions (captain, bench order, TC/BB triggers)
            # Ongoing ownership boost applied here so it never touches ILP inputs.
            if gw > 1:
                adj_pred = apply_ongoing_ownership_boost(adj_pred_ilp, player_pool)

            # --- Chip check (improved timing) ---
            chip        = None
            is_wildcard = False
            is_freehit  = False

            if current_squad is not None:
                xi_proxy = sorted(
                    [p for p in current_squad if p["pos_id"] != 1],
                    key=lambda p: adj_pred.get(p["player_id"], 0),
                    reverse=True,
                )[:10]
                xi_proxy_ids = {p["player_id"] for p in xi_proxy}
                bench_out_proxy = [
                    p for p in current_squad
                    if p["player_id"] not in xi_proxy_ids and p["pos_id"] != 1
                ]

                chip_raw = check_chips_improved(
                    gw, real_gw, dgw_gws, chips_used,
                    current_squad, adj_pred, bench_out_proxy,
                )

                if chip_raw == "freehit":
                    chip         = "freehit"
                    is_freehit   = True
                    chips_used.add("freehit")
                    chips_log.append({"chip": "freehit", "gw": gw})
                    free_transfers = 15

                elif chip_raw and chip_raw.startswith("wildcard"):
                    chip         = chip_raw
                    is_wildcard  = True
                    chips_used.add(chip_raw)
                    chips_log.append({"chip": chip_raw, "gw": gw})
                    free_transfers = 15

                elif chip_raw == "bench_boost":
                    chip = "bench_boost"
                    chips_used.add("bench_boost")
                    chips_log.append({"chip": "bench_boost", "gw": gw})

                elif chip_raw == "triple_captain":
                    chip = "triple_captain"
                    chips_used.add("triple_captain")
                    chips_log.append({"chip": "triple_captain", "gw": gw})

            # --- ILP (uses adj_pred_ilp — no ongoing ownership boost) ---
            zero_minutes_ids = {
                p["player_id"] for p in player_pool if p.get("zero_minutes", False)
            }
            # fresh_squad_gw: ignore accumulated squad → pick best 15 freely
            _fresh = (fresh_squad_gw is not None and gw == fresh_squad_gw)
            if _fresh:
                print(f"  [FRESH PICK] GW{gw}: no squad constraint — free wildcard pick")
            result = run_ilp(
                player_pool       = player_pool,
                horizon_scores    = adj_horizon,
                pred_gw0_scores   = adj_pred_ilp,
                budget            = BUDGET,
                prev_squad        = None if (_fresh or gw == 1) else current_squad,
                free_transfers    = 15 if _fresh else (free_transfers if gw > 1 else 1),
                is_wildcard       = True if _fresh else is_wildcard,
                is_freehit        = is_freehit,
            )

            if result is None:
                print(f"  GW{gw}: ILP failed, skipping.")
                continue

            # --- Enforce zero hits during chip lockout (GW1-4) ---
            if not is_wildcard and not is_freehit and gw > 1 and gw <= CHIP_LOCKOUT_GW:
                n_tr = len(result["transfers_in"])
                hits = max(0, n_tr - free_transfers)
                if hits > 0:
                    print(f"  [LOCKOUT] ILP wanted {n_tr} transfers ({hits} hits) "
                          f"— hard-capping to {free_transfers} transfers")
                    result = run_ilp(
                        player_pool       = player_pool,
                        horizon_scores    = adj_horizon,
                        pred_gw0_scores   = adj_pred_ilp,
                        budget            = BUDGET,
                        prev_squad        = current_squad,
                        free_transfers    = free_transfers,
                        is_wildcard       = False,
                        is_freehit        = False,
                        max_transfers     = free_transfers,
                    )
                    if result is None:
                        print(f"  GW{gw}: ILP failed on lockout re-run, skipping.")
                        continue

            # --- Auto-trigger Free Hit if too many hits (post-lockout) ---
            if not is_wildcard and not is_freehit and gw > CHIP_LOCKOUT_GW:
                n_tr = len(result["transfers_in"])
                hits = max(0, n_tr - free_transfers)
                if hits > MAX_HITS and "freehit" not in chips_used:
                    print(f"  [CHIP] Auto-triggering Free Hit "
                          f"({n_tr} transfers would cost {hits * 4} pts)")
                    chip = "freehit"
                    is_freehit = True
                    chips_used.add("freehit")
                    chips_log.append({"chip": "freehit", "gw": gw})
                    result = run_ilp(
                        player_pool       = player_pool,
                        horizon_scores    = adj_horizon,
                        pred_gw0_scores   = adj_pred_ilp,
                        budget            = BUDGET,
                        prev_squad        = current_squad,
                        free_transfers    = 15,
                        is_wildcard       = False,
                        is_freehit        = True,
                    )
                    if result is None:
                        print(f"  GW{gw}: ILP failed on Free Hit re-run, skipping.")
                        continue

            # --- Post-ILP: bench zero-minutes players that ended up in XI ---
            result = fix_zero_minutes_in_xi(result, adj_pred, zero_minutes_ids)

            # --- Captain override from intel_05 (with 0.5pt threshold) ---
            cap_source = apply_captain_override(result, recommendations, gw, adj_pred)
            captain_sources.append(cap_source)
            if "intel_05" in cap_source:
                intel_cap = result["captain"]["web_name"] if result["captain"] else "?"
                print(f"  [INTEL] Captain override: {intel_cap} (source: {cap_source})")

            # --- Bench order optimization ---
            optimize_bench_order(result, adj_pred)

            # --- Triple Captain trigger (post-lockout, high captain prediction) ---
            TC_THRESHOLD = hp.get('tc_threshold', 9.5)
            TC_MIN_GW = hp.get('tc_min_gw', 6)
            if (chip is None
                    and gw >= TC_MIN_GW
                    and "triple_captain" not in chips_used
                    and result["captain"]):
                raw_pred = adj_pred.get(result["captain"]["player_id"], 0)
                pos_mult = CAP_POS_MULTIPLIERS.get(result["captain"]["pos_id"], 1.0)
                cap_pred = raw_pred * pos_mult
                if cap_pred >= TC_THRESHOLD:
                    chip = "triple_captain"
                    chips_used.add("triple_captain")
                    chips_log.append({"chip": "triple_captain", "gw": gw})
                    print(f"  [CHIP] Triple Captain triggered "
                          f"({result['captain']['web_name']} pred {cap_pred:.1f})")

            # --- Bench Boost trigger (post-lockout, strong bench) ---
            BB_THRESHOLD = hp.get('bb_threshold', 9.0)
            BB_MIN_GW    = hp.get('bb_min_gw', 6)

            if (chip is None
                    and gw >= BB_MIN_GW
                    and "bench_boost" not in chips_used):
                _xi_set_bb = {p["player_id"] for p in result["xi"]}
                _bench_players_bb = [p for p in result["squad"]
                                     if p["player_id"] not in _xi_set_bb]
                _bench_pred_total = sum(
                    adj_pred.get(p["player_id"], 0) for p in _bench_players_bb
                )
                if _bench_pred_total >= BB_THRESHOLD:
                    chip = "bench_boost"
                    chips_used.add("bench_boost")
                    chips_log.append({"chip": "bench_boost", "gw": gw})
                    print(f"  [CHIP] Bench Boost triggered "
                          f"(bench predicted {_bench_pred_total:.1f} pts)")

            # --- Auto-sub simulation (uses real minutes) ---
            final_xi, auto_subs = simulate_auto_subs(
                result["xi"], result["squad"], hist_lookup, gw
            )
            if auto_subs:
                print(f"  [AUTO-SUB] {len(auto_subs)} sub(s):", end="")
                for out_p, in_p in auto_subs:
                    print(f"  {out_p['web_name']} -> {in_p['web_name']}", end="")
                print()

            # --- Compute actual GW points for HPO scoring ---
            # Uses sim GW (1-based) to match player_history.csv rows,
            # same logic as full_team_points_report.py
            cap_id = result["captain"]["player_id"] if result["captain"] else None
            cap_mult = 3 if chip == "triple_captain" else 2
            _xi_actual = 0
            for _p in final_xi:
                _base = hist_lookup.get(_p["player_id"], {}).get(gw, {}).get("total_points", 0)
                _xi_actual += _base * (cap_mult if _p["player_id"] == cap_id else 1)
            _bench_actual = 0
            if chip == "bench_boost":
                _final_bench_actual = [p for p in result["squad"] if p not in final_xi]
                for _p in _final_bench_actual:
                    _bench_actual += hist_lookup.get(_p["player_id"], {}).get(gw, {}).get("total_points", 0)
            _penalty_pts = result["penalty"] * 4
            gw_actual = _xi_actual + _bench_actual - _penalty_pts
            actual_scores.append(gw_actual)

            # Freehit: squad reverts
            if not is_freehit:
                current_squad = result["squad"]

            if result["captain"]:
                captain_counts[result["captain"]["web_name"]] += 1

            # --- Print GW block (uses adjusted predictions for display) ---
            gw_score = print_gw_block(gw, real_gw, chip, result, adj_pred, dgw_gws)

            # --- Track ---
            squad_val = sum(p["value"] for p in result["squad"])
            gw_scores.append(gw_score)
            squad_values.append(squad_val)
            penalties_total += result["penalty"] * 4

            if gw_score > best_gw_info["score"]:
                best_gw_info = {
                    "gw":    gw,
                    "score": gw_score,
                    "squad": [p["web_name"] for p in result["squad"]],
                }

            bench_players = [p for p in result["squad"] if p not in result["xi"]]
            final_bench = [p for p in result["squad"] if p not in final_xi]
            sim_log["gameweeks"].append({
                "gw":              gw,
                "chip":            chip,
                "predicted_score": gw_score,
                "squad_value":     squad_val,
                "free_transfers":  free_transfers,
                "transfers_in":    [p["web_name"] for p in result["transfers_in"]],
                "transfers_out":   [p["web_name"] for p in result["transfers_out"]],
                "penalty_pts":     result["penalty"] * 4,
                "captain":         result["captain"]["web_name"] if result["captain"] else None,
                "captain_id":      result["captain"]["player_id"] if result["captain"] else None,
                "captain_source":  cap_source,
                "xi":              [p["web_name"] for p in result["xi"]],
                "xi_ids":          [p["player_id"] for p in result["xi"]],
                "xi_final":        [p["web_name"] for p in final_xi],
                "xi_final_ids":    [p["player_id"] for p in final_xi],
                "bench":           [p["web_name"] for p in bench_players],
                "bench_ids":       [p["player_id"] for p in bench_players],
                "bench_final":     [p["web_name"] for p in final_bench],
                "bench_final_ids": [p["player_id"] for p in final_bench],
                "auto_subs":       [[o["web_name"], i["web_name"]] for o, i in auto_subs],
                "intel_adjustments": intel_stats,
            })

            # --- Online retraining with REAL actuals (for next GW) ---
            if gw < n_gws:
                print(f"\n  Real actuals + retraining for GW{gw+1}...")
                actuals = get_real_actuals(player_pool, hist_lookup, gw)

                for p in player_pool:
                    pid    = p["player_id"]
                    pos_id = p["pos_id"]
                    row    = p["feat_vec"].copy()
                    row[TARGET_COL] = actuals[pid]
                    row["season"]   = "2025-26"
                    row["GW"]       = gw
                    new_rows_by_pos[pos_id].append(row)

                models, model_fcols = retrain_models(
                    train_dfs, new_rows_by_pos, best_params, model_fcols
                )

                # --- GW-Snapshot: refresh features with real actuals ---
                refresh_player_features(player_pool, hist_lookup, fdr_lookup, gw)

            # --- Update free transfers (FPL 2025-26: max 5 banked) ---
            if is_wildcard or is_freehit:
                free_transfers = 1
            else:
                n_tr = len(result["transfers_in"])
                if n_tr <= free_transfers:
                    free_transfers = min(5, free_transfers - n_tr + 1)
                else:
                    free_transfers = 1

        # ===================================================================
        # Season Summary
        # ===================================================================
        total_pred     = sum(gw_scores)
        most_captained = (
            max(captain_counts, key=captain_counts.get) if captain_counts else "N/A"
        )
        n_intel_overrides = sum(1 for s in captain_sources if s == "intel_05")

        print()
        print(SEP)
        print(f"  INTEL-ENHANCED SEASON SUMMARY  (GW1-GW{n_gws})")
        print(SEP)
        print(f"  Total predicted points   : {total_pred:.1f}")
        print(f"  Total transfer penalties : -{penalties_total} pts")
        print(f"  Captain overrides (intel): {n_intel_overrides} / {len(captain_sources)}")
        print(f"  Chips used               :", end="")
        if chips_log:
            print()
            for c in chips_log:
                print(f"    {c['chip']} in GW{c['gw']}")
        else:
            print(" None")
        print(f"  Best GW                  : GW{best_gw_info['gw']}"
              f" with {best_gw_info['score']:.1f} pts")
        print(f"  Most captained player    : {most_captained}"
              f" ({captain_counts.get(most_captained, 0)}x)")
        print("  Squad value trajectory   :")
        for i, val in enumerate(squad_values, 1):
            print(f"    GW{i:2d}: GBP{val:.1f}m")

        # Captain picks per GW
        print("\n  Captain picks:")
        for entry in sim_log["gameweeks"]:
            src_tag = f" [{entry['captain_source']}]" if entry.get("captain_source") else ""
            print(f"    GW{entry['gw']:2d}: {entry['captain'] or 'N/A'}{src_tag}")

        print(SEP)

        # --- Save JSON log ---
        if save_json:
            sim_log["total_predicted_points"] = total_pred
            sim_log["total_penalties"]        = penalties_total
            sim_log["chips_used"]             = chips_log
            sim_log["best_gw"]                = best_gw_info
            sim_log["most_captained"]         = most_captained
            sim_log["captain_overrides"]      = n_intel_overrides
            sim_log["squad_value_trajectory"] = [
                {"gw": i + 1, "value": v} for i, v in enumerate(squad_values)
            ]

            os.makedirs(DATA_INTEL, exist_ok=True)
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(sim_log, f, indent=2, ensure_ascii=False)
            print(f"\nIntel-enhanced simulation saved -> {OUTPUT_PATH}")

        return sum(actual_scores)

    if verbose:
        return _run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            return _run()


# ===========================================================================
# Main
# ===========================================================================
def main():
    data = load_all_data(verbose=True)
    run_simulation(data, hp=None, n_gws=10, verbose=True, save_json=True)


if __name__ == "__main__":
    main()
