"""
Stage 8: ILP Squad Optimizer
FPL AI Thesis -- PuLP-based ILP squad optimization

GW1 : fresh 15-man squad from XGBoost predictions
GW2-10: transfer optimization with online-retrained models + 3-GW horizon
Chip logic, bench ordering, captain/vice selection, season summary.
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW  = os.path.join(ROOT, "data", "raw", "fpl_api")
DATA_PROC = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

POS_FILES = {
    1: ("xgb_gk.pkl",  "train_gk.csv"),
    2: ("xgb_def.pkl", "train_def.csv"),
    3: ("xgb_mid.pkl", "train_mid.csv"),
    4: ("xgb_fwd.pkl", "train_fwd.csv"),
}

BUDGET    = 100.0
MAX_CLUB  = 3
SEED      = 42

HORIZON_WEIGHTS = [1.0, 0.7, 0.5]
HORIZON_GWS     = 3

EXCLUDE_COLS = {
    "name", "season", "GW", "team", "opponent_team", "position",
    "was_home", "fdr_is_proxy", "trajectory_is_full",
}
TARGET_COL = "total_points"

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Helper: feature columns
# ---------------------------------------------------------------------------
def get_feature_cols(df):
    drop = EXCLUDE_COLS | {TARGET_COL}
    return [c for c in df.columns if c not in drop]


# ---------------------------------------------------------------------------
# Load models and best params
# ---------------------------------------------------------------------------
def load_models():
    models      = {}
    model_fcols = {}
    for pos_id, (pkl, _) in POS_FILES.items():
        path = os.path.join(MODELS_DIR, pkl)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "model" in obj:
            models[pos_id]      = obj["model"]
            model_fcols[pos_id] = obj.get("feature_cols", None)
        else:
            models[pos_id]      = obj  # plain XGBRegressor
            model_fcols[pos_id] = None
    print("Loaded 4 XGBoost models.")
    return models, model_fcols


def load_best_params():
    path = os.path.join(MODELS_DIR, "stage7_results.json")
    with open(path) as f:
        results = json.load(f)
    pos_name_to_id = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    params = {}
    for pos_name, pos_id in pos_name_to_id.items():
        if pos_name in results:
            params[pos_id] = results[pos_name].get("best_params", {})
        else:
            params[pos_id] = {}
    return params


# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------
def load_training_data():
    train_dfs = {}
    for pos_id, (_, csv_file) in POS_FILES.items():
        path = os.path.join(DATA_PROC, csv_file)
        df = pd.read_csv(path)
        train_dfs[pos_id] = df
        print(f"  Loaded {csv_file}: {len(df)} rows, {len(df.columns)} cols")
    return train_dfs


# ---------------------------------------------------------------------------
# Load FPL API data (print schema for verification)
# ---------------------------------------------------------------------------
def load_fpl_data():
    def load_and_report(fname):
        path = os.path.join(DATA_RAW, fname)
        df = pd.read_csv(path)
        print(f"\n{fname} columns ({len(df.columns)}): {list(df.columns)}")
        print(df.head(2).to_string())
        return df

    players  = load_and_report("players_raw.csv")
    ph       = load_and_report("player_history.csv")
    fixtures = load_and_report("fixtures_raw.csv")
    fdr_df   = load_and_report("fixture_difficulty.csv")
    upcoming = load_and_report("player_upcoming_fixtures.csv")
    teams    = load_and_report("teams_raw.csv")

    return players, ph, fixtures, fdr_df, upcoming, teams


# ---------------------------------------------------------------------------
# DGW / BGW detection
# ---------------------------------------------------------------------------
def detect_dgw(fixtures):
    """Return {gw_int: [team_ids_with_2+_fixtures]}."""
    dgw = {}
    gw_col = "gameweek"
    valid = fixtures.dropna(subset=[gw_col])
    for gw, grp in valid.groupby(gw_col):
        counts = defaultdict(int)
        for _, row in grp.iterrows():
            counts[int(row["team_h"])] += 1
            counts[int(row["team_a"])] += 1
        double_teams = [t for t, c in counts.items() if c >= 2]
        if double_teams:
            dgw[int(gw)] = double_teams
    return dgw


def detect_bgw(fixtures, all_gws, all_team_ids):
    """Return {gw_int: [team_ids_with_no_fixture]}."""
    teams_in_gw = defaultdict(set)
    valid = fixtures.dropna(subset=["gameweek"])
    for _, row in valid.iterrows():
        gw = int(row["gameweek"])
        teams_in_gw[gw].add(int(row["team_h"]))
        teams_in_gw[gw].add(int(row["team_a"]))
    bgw = {}
    for gw in all_gws:
        missing = [t for t in all_team_ids if t not in teams_in_gw.get(gw, set())]
        if missing:
            bgw[gw] = missing
    return bgw


def get_sim_start_gw(fixtures, upcoming):
    """
    Determine the real GW number for simulation GW1.
    Use the minimum GW in player_upcoming_fixtures (cast to int),
    falling back to first unfinished fixture GW.
    """
    if len(upcoming) > 0:
        valid_up = upcoming.dropna(subset=["gameweek"])
        if len(valid_up) > 0:
            return int(valid_up["gameweek"].min())
    # fallback: first unfinished fixture
    valid_fix = fixtures.dropna(subset=["gameweek"])
    unfinished = valid_fix[valid_fix.get("finished", False) == False]
    if len(unfinished) > 0:
        return int(unfinished["gameweek"].min())
    return 1


# ---------------------------------------------------------------------------
# Build player pool
# ---------------------------------------------------------------------------
def build_player_pool(players, train_dfs, upcoming, teams, sim_start_gw=1):
    """Build a feature-vector dict for every active player."""

    # --- filter active players ---
    active = players.copy()
    n_start = len(active)
    if "status" in active.columns:
        active = active[~active["status"].isin(["i", "d", "u", "s"])]
        print(f"  Status filter: {n_start} -> {len(active)} ({n_start - len(active)} removed)")
    if "chance_of_playing_next_round" in active.columns:
        n_before = len(active)
        active = active[
            active["chance_of_playing_next_round"].isna()
            | (active["chance_of_playing_next_round"] >= 50)
        ]
        print(f"  Availability filter: {n_before} -> {len(active)} ({n_before - len(active)} removed)")
    active = active[active["now_cost"] > 0].copy()
    print(f"  Cost filter: {len(active)} players with now_cost > 0")
    active["value"] = active["now_cost"] / 10.0

    # team lookup maps
    team_map       = dict(zip(teams["id"], teams["name"]))
    team_short_map = dict(zip(teams["id"], teams["short_name"]))

    active["team_name_str"] = active["team"].map(team_map).fillna("Unknown")
    active["team_short"]    = active["team"].map(team_short_map).fillna("?")

    # display name: use second_name (shorter), fall back to first+second
    if "web_name" in active.columns:
        active["display_name"] = active["web_name"]
    else:
        active["display_name"] = active["second_name"].str.strip()

    active["full_name"] = (
        active["first_name"].str.strip() + " " + active["second_name"].str.strip()
    )

    # normalise upcoming gameweek to int
    upcoming = upcoming.copy()
    upcoming = upcoming.dropna(subset=["gameweek"])
    upcoming["gameweek"] = upcoming["gameweek"].astype(int)

    min_upcoming_gw = sim_start_gw

    player_pool = []
    min_rel_filtered = 0
    snap_2425 = 0
    snap_older = 0
    snap_pos_avg = 0

    SNAPSHOT_SEASONS = ["2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20"]

    for pos_id in [1, 2, 3, 4]:
        pos_players = active[active["element_type"] == pos_id].copy()
        train_df    = train_dfs[pos_id]
        feat_cols   = get_feature_cols(train_df)

        # position-average fallback
        pos_avg = train_df[feat_cols].mean()

        for _, row in pos_players.iterrows():
            pid      = int(row["id"])
            full_nm  = row["full_name"]
            last_nm  = row["second_name"].strip()
            disp_nm  = row["display_name"]

            # --- name-based history lookup ---
            hist = train_df[train_df["name"] == full_nm]
            if len(hist) == 0:
                hist = train_df[
                    train_df["name"].str.lower() == full_nm.lower()
                ]
            if len(hist) == 0:
                # partial match on last name (exact word boundary)
                mask = train_df["name"].str.split().apply(
                    lambda parts: last_nm.lower() in [p.lower() for p in parts]
                )
                hist = train_df[mask]

            if len(hist) > 0:
                # PRIMARY: use full 2024-25 season average
                # FALLBACK: walk back through older seasons until data found
                feat_vec = None
                used_season = None
                for season in SNAPSHOT_SEASONS:
                    season_hist = hist[hist["season"] == season]
                    if len(season_hist) >= 1:
                        feat_vec = season_hist[feat_cols].mean()
                        used_season = season
                        break
                if feat_vec is None:
                    # no season data at all — use position average
                    feat_vec = pos_avg.copy()
                    snap_pos_avg += 1
                elif used_season == "2024-25":
                    snap_2425 += 1
                else:
                    snap_older += 1
            else:
                feat_vec = pos_avg.copy()
                snap_pos_avg += 1

            # --- override with live 2025-26 values ---
            if "value" in feat_cols:
                feat_vec["value"] = row["value"]
            if "selected" in feat_cols:
                sp = row.get("selected_by_percent", 0)
                feat_vec["selected"] = float(sp) if pd.notna(sp) else 0.0
            if "transfers_in" in feat_cols:
                feat_vec["transfers_in"] = float(
                    row.get("transfers_in", 0) or 0
                )
            if "transfers_out" in feat_cols:
                feat_vec["transfers_out"] = float(
                    row.get("transfers_out", 0) or 0
                )

            # home_advantage for GW1 (use upcoming fixture is_home)
            if "home_advantage" in feat_cols:
                up_row = upcoming[
                    (upcoming["player_id"] == pid)
                    & (upcoming["gameweek"] == min_upcoming_gw)
                ]
                if len(up_row) > 0:
                    feat_vec["home_advantage"] = int(
                        bool(up_row.iloc[0]["is_home"])
                    )

            # FDR for GW1
            if "current_gw_fdr" in feat_cols:
                up_row = upcoming[
                    (upcoming["player_id"] == pid)
                    & (upcoming["gameweek"] == min_upcoming_gw)
                ]
                if len(up_row) > 0:
                    feat_vec["current_gw_fdr"] = float(
                        up_row.iloc[0]["difficulty"]
                    )

            # --- filter: low minutes reliability ---
            min_rel = feat_vec.get("minutes_reliability_season", 1.0)
            if min_rel < 0.4:
                min_rel_filtered += 1
                continue

            player_pool.append({
                "player_id":  pid,
                "name":       full_nm,
                "web_name":   disp_nm,
                "pos_id":     pos_id,
                "pos_name":   POSITIONS[pos_id],
                "team_id":    int(row["team"]),
                "team_name":  row["team_name_str"],
                "team_short": row["team_short"],
                "value":      row["value"],
                "feat_vec":   feat_vec.to_dict(),
                "feat_cols":  feat_cols,
            })

    if min_rel_filtered > 0:
        print(f"  Minutes reliability filter: {min_rel_filtered} players removed (< 0.4)")
    print(f"  Feature snapshots: {snap_2425} from 2024-25 avg, {snap_older} from older seasons")
    print(f"  Players with no historical data (position avg fallback): {snap_pos_avg}")
    print(f"\nPlayer pool built: {len(player_pool)} active players")
    return player_pool


# ---------------------------------------------------------------------------
# XGBoost prediction
# ---------------------------------------------------------------------------
def _make_X(p, model_fcols, pos_id):
    """Build feature DataFrame for one player, respecting model feature order."""
    feat_cols = model_fcols.get(pos_id) if model_fcols else None
    if feat_cols is None:
        feat_cols = p["feat_cols"]
    # Build row; fill missing cols with 0
    row = {c: p["feat_vec"].get(c, 0) for c in feat_cols}
    return pd.DataFrame([row])[feat_cols].fillna(0)


def predict_gw0(player_pool, models, model_fcols=None):
    """Predict points for current GW using base feature vectors."""
    model_fcols = model_fcols or {}
    preds = {}
    for p in player_pool:
        pid    = p["player_id"]
        pos_id = p["pos_id"]
        model  = models[pos_id]
        X      = _make_X(p, model_fcols, pos_id)
        pred   = float(model.predict(X)[0])
        preds[pid] = max(0.0, pred)
    return preds


def predict_horizon_gw(player_pool, models, gw_offset,
                       real_gw, upcoming_df, dgw_gws, bgw_teams_by_gw,
                       model_fcols=None):
    """
    Predict points for real_gw + gw_offset with FDR/DGW/BGW adjustments.
    real_gw is the actual season GW number (not simulation index).
    """
    model_fcols = model_fcols or {}
    target_gw   = real_gw + gw_offset
    preds = {}

    for p in player_pool:
        pid    = p["player_id"]
        pos_id = p["pos_id"]
        model  = models[pos_id]

        X    = _make_X(p, model_fcols, pos_id)
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)

        if gw_offset > 0:
            team_id = p["team_id"]

            # BGW: blank gameweek -> 0 points
            if team_id in bgw_teams_by_gw.get(target_gw, []):
                pred = 0.0
            else:
                # DGW bonus
                if team_id in dgw_gws.get(target_gw, []):
                    pred *= 1.8

                # FDR adjustment from upcoming fixtures
                up_row = upcoming_df[
                    (upcoming_df["player_id"] == pid)
                    & (upcoming_df["gameweek"] == target_gw)
                ]
                if len(up_row) > 0:
                    fdr = float(up_row.iloc[0]["difficulty"])
                    adj = 1.0 - 0.03 * (fdr - 3.0)
                    pred *= max(0.5, adj)

        preds[pid] = pred
    return preds


def predict_horizon(player_pool, models, real_gw,
                    upcoming_df, dgw_gws, bgw_teams_by_gw, model_fcols=None):
    """Return discounted 3-GW horizon score per player."""
    model_fcols = model_fcols or {}
    horizon = {}
    for offset, weight in enumerate(HORIZON_WEIGHTS):
        gw_preds = predict_horizon_gw(
            player_pool, models, offset,
            real_gw, upcoming_df, dgw_gws, bgw_teams_by_gw,
            model_fcols=model_fcols,
        )
        for pid, pred in gw_preds.items():
            horizon[pid] = horizon.get(pid, 0.0) + weight * pred
    return horizon


# ---------------------------------------------------------------------------
# ILP Optimizer
# ---------------------------------------------------------------------------
def run_ilp(player_pool, horizon_scores, pred_gw0_scores,
            budget=BUDGET, prev_squad=None, free_transfers=1,
            is_wildcard=False, is_freehit=False, max_transfers=None):
    """
    Run PuLP ILP to select optimal squad.

    Returns dict with keys:
      squad, xi, captain, vice,
      transfers_in, transfers_out, penalty
    or None if infeasible.
    """
    try:
        import pulp
    except ImportError:
        raise ImportError("PuLP not installed. Run: pip install pulp")

    players = player_pool
    n       = len(players)
    ids     = [p["player_id"] for p in players]

    h_scores = [horizon_scores.get(pid, 0.0)    for pid in ids]
    g0       = [pred_gw0_scores.get(pid, 0.0)   for pid in ids]

    is_gk  = [1 if p["pos_id"] == 1 else 0 for p in players]
    is_def = [1 if p["pos_id"] == 2 else 0 for p in players]
    is_mid = [1 if p["pos_id"] == 3 else 0 for p in players]
    is_fwd = [1 if p["pos_id"] == 4 else 0 for p in players]
    costs  = [p["value"] for p in players]
    team_ids = [p["team_id"] for p in players]

    prev_ids = set(p["player_id"] for p in prev_squad) if prev_squad else set()
    x_prev   = [1 if pid in prev_ids else 0 for pid in ids]

    use_transfers = (prev_squad is not None) and (not is_wildcard) and (not is_freehit)

    # Pre-computed arrays for bench quality constraints
    min_rel = [
        p["feat_vec"].get("minutes_reliability_season", 1.0) for p in players
    ]

    def build_and_solve(budget_limit):
        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

        x  = [pulp.LpVariable(f"x_{i}",  cat="Binary") for i in range(n)]
        s  = [pulp.LpVariable(f"s_{i}",  cat="Binary") for i in range(n)]
        c  = [pulp.LpVariable(f"c_{i}",  cat="Binary") for i in range(n)]
        vc = [pulp.LpVariable(f"vc_{i}", cat="Binary") for i in range(n)]

        if use_transfers:
            t_in  = [pulp.LpVariable(f"tin_{i}",  cat="Binary") for i in range(n)]
            t_out = [pulp.LpVariable(f"tout_{i}", cat="Binary") for i in range(n)]
            ts = [pulp.LpVariable(f"ts_{i}", cat="Binary") for i in range(n)]
        else:
            t_in = t_out = ts = None

        pen = pulp.LpVariable("penalty", lowBound=0, cat="Integer")

        TRANSFER_START_BONUS = 0.0

        # --- Objective (includes bench weight for auto-sub value) ---
        transfer_start_term = (
            pulp.lpSum(TRANSFER_START_BONUS * ts[i] for i in range(n))
            if use_transfers else 0
        )
        prob += (
            pulp.lpSum(h_scores[i] * s[i] for i in range(n))
            + pulp.lpSum(g0[i] * c[i]           for i in range(n))
            + pulp.lpSum(g0[i] * 0.5 * vc[i]    for i in range(n))
            + pulp.lpSum(0.25 * g0[i] * (x[i] - s[i])
                         for i in range(n) if not is_gk[i])
            + transfer_start_term
            - 4.0 * pen
        )

        # --- Squad structure ---
        prob += pulp.lpSum(x) == 15
        prob += pulp.lpSum(is_gk[i]  * x[i] for i in range(n)) == 2
        prob += pulp.lpSum(is_def[i] * x[i] for i in range(n)) == 5
        prob += pulp.lpSum(is_mid[i] * x[i] for i in range(n)) == 5
        prob += pulp.lpSum(is_fwd[i] * x[i] for i in range(n)) == 3

        # Budget
        prob += pulp.lpSum(costs[i] * x[i] for i in range(n)) <= budget_limit

        # Club limit
        for tid in set(team_ids):
            prob += pulp.lpSum(x[i] for i in range(n) if team_ids[i] == tid) <= MAX_CLUB

        # --- XI structure ---
        prob += pulp.lpSum(s) == 11
        prob += pulp.lpSum(is_gk[i]  * s[i] for i in range(n)) == 1
        prob += pulp.lpSum(is_def[i] * s[i] for i in range(n)) >= 3
        prob += pulp.lpSum(is_def[i] * s[i] for i in range(n)) <= 4
        prob += pulp.lpSum(is_mid[i] * s[i] for i in range(n)) >= 2
        prob += pulp.lpSum(is_mid[i] * s[i] for i in range(n)) <= 5
        prob += pulp.lpSum(is_fwd[i] * s[i] for i in range(n)) >= 1
        prob += pulp.lpSum(is_fwd[i] * s[i] for i in range(n)) <= 3

        for i in range(n):
            prob += s[i] <= x[i]

        # --- Captain / Vice ---
        prob += pulp.lpSum(c)  == 1
        prob += pulp.lpSum(vc) == 1
        for i in range(n):
            prob += c[i]  <= s[i]
            prob += vc[i] <= s[i]
            prob += c[i] + vc[i] <= 1

        # --- Transfer constraints ---
        if use_transfers:
            for i in range(n):
                prob += x[i] == x_prev[i] + t_in[i] - t_out[i]
                # ts[i] = t_in[i] AND s[i] (linearized product)
                prob += ts[i] <= t_in[i]
                prob += ts[i] <= s[i]
                prob += ts[i] >= t_in[i] + s[i] - 1
            n_in  = pulp.lpSum(t_in)
            n_out = pulp.lpSum(t_out)
            prob += n_in == n_out
            prob += pen >= n_in - free_transfers
            prob += pen >= 0
            if max_transfers is not None:
                prob += n_in <= max_transfers
        else:
            prob += pen == 0

        # --- Bench quality constraints ---
        # (1) Outfield players with pred < 2.5 cannot be benched
        for i in range(n):
            if not is_gk[i] and g0[i] < 2.5:
                prob += x[i] <= s[i]

        # (2) Outfield players with minutes_reliability <= 0.5 cannot be benched
        for i in range(n):
            if not is_gk[i] and min_rel[i] <= 0.5:
                prob += x[i] <= s[i]

        # (3) Combined bench outfield predicted pts >= 7.0
        prob += (
            pulp.lpSum(g0[i] * (x[i] - s[i]) for i in range(n) if not is_gk[i])
            >= 7.0
        )

        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            return None

        x_sol  = [round(pulp.value(x[i])  or 0) for i in range(n)]
        s_sol  = [round(pulp.value(s[i])  or 0) for i in range(n)]
        c_sol  = [round(pulp.value(c[i])  or 0) for i in range(n)]
        vc_sol = [round(pulp.value(vc[i]) or 0) for i in range(n)]

        tin_sol  = [round(pulp.value(t_in[i])  or 0) for i in range(n)] if t_in  else [0]*n
        tout_sol = [round(pulp.value(t_out[i]) or 0) for i in range(n)] if t_out else [0]*n
        pen_val  = int(round(pulp.value(pen) or 0))

        return {
            "squad":         [players[i] for i in range(n) if x_sol[i]  == 1],
            "xi":            [players[i] for i in range(n) if s_sol[i]  == 1],
            "captain":       next((players[i] for i in range(n) if c_sol[i]  == 1), None),
            "vice":          next((players[i] for i in range(n) if vc_sol[i] == 1), None),
            "transfers_in":  [players[i] for i in range(n) if tin_sol[i]  == 1],
            "transfers_out": [players[i] for i in range(n) if tout_sol[i] == 1],
            "penalty":       pen_val,
        }

    # Retry with relaxed budget if infeasible
    for attempt in range(4):
        result = build_and_solve(budget + attempt * 0.5)
        if result is not None:
            if attempt > 0:
                print(f"  ILP solved with budget relaxation +{attempt*0.5:.1f}m")
            # Override captain/vice with positional bias
            cap, vc = select_captain(result["xi"], pred_gw0_scores)
            result["captain"] = cap
            result["vice"]    = vc
            return result

    print("WARNING: ILP infeasible after 4 attempts. Skipping GW.")
    return None


# ---------------------------------------------------------------------------
# Bench ordering
# ---------------------------------------------------------------------------
def split_bench(squad, xi, pred_gw0_scores):
    xi_ids    = {p["player_id"] for p in xi}
    bench     = [p for p in squad if p["player_id"] not in xi_ids]
    bench_gk  = [p for p in bench if p["pos_id"] == 1]
    bench_out = sorted(
        [p for p in bench if p["pos_id"] != 1],
        key=lambda p: pred_gw0_scores.get(p["player_id"], 0),
        reverse=True,
    )
    return bench_gk, bench_out


# ---------------------------------------------------------------------------
# Captain selection with positional bias
# ---------------------------------------------------------------------------
CAP_POS_MULTIPLIERS = {1: 0.0, 2: 0.75, 3: 1.15, 4: 1.25}


def select_captain(xi, pred_gw0_scores):
    """
    Post-process captain/vice with positional bias multipliers.
    Hard rules: never captain GK; DEF only if adj > 12 AND form_last3 > 8.
    Prints top-5 candidates with raw -> adjusted scores.
    Returns (captain, vice) player dicts.
    """
    candidates = []
    for p in xi:
        pid  = p["player_id"]
        raw  = pred_gw0_scores.get(pid, 0)
        mult = CAP_POS_MULTIPLIERS.get(p["pos_id"], 1.0)
        adj  = raw * mult
        if p["pos_id"] == 1:          # GK: never captain
            continue
        if p["pos_id"] == 2:          # DEF: hard threshold
            form = p["feat_vec"].get("form_last3", 0)
            if adj <= 12 or form <= 8:
                continue
        candidates.append((adj, raw, p))

    if not candidates:
        # fallback: best MID/FWD by raw pred
        fallback = sorted(
            [p for p in xi if p["pos_id"] in (3, 4)],
            key=lambda p: pred_gw0_scores.get(p["player_id"], 0),
            reverse=True,
        )
        if fallback:
            return fallback[0], fallback[1] if len(fallback) > 1 else fallback[0]
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    print("  Captain candidates (top 5):")
    for adj, raw, p in candidates[:5]:
        print(f"    {p['web_name']:<22} ({p['pos_name']:<3})  raw:{raw:5.1f}  adj:{adj:5.1f}")

    captain = candidates[0][2]
    vice    = candidates[1][2] if len(candidates) > 1 else candidates[0][2]
    return captain, vice


# ---------------------------------------------------------------------------
# Chip logic
# ---------------------------------------------------------------------------
def check_chips(current_gw, dgw_gws, chips_used,
                current_squad, pred_gw0_scores, bench_out):
    """
    Returns the chip to activate this GW, or None.
    Priority: freehit > wildcard > bench_boost > triple_captain
    """

    # 1. Freehit: current GW is a DGW
    if "freehit" not in chips_used:
        if current_gw in dgw_gws and dgw_gws[current_gw]:
            return "freehit"

    # 2. Wildcard: >= 4 squad members below position average
    half     = 1 if current_gw <= 19 else 2
    wc_key   = f"wildcard_{half}"
    if wc_key not in chips_used and current_squad:
        pos_avgs = {}
        for pos_id in [1, 2, 3, 4]:
            pos_preds = [
                pred_gw0_scores.get(p["player_id"], 0)
                for p in current_squad if p["pos_id"] == pos_id
            ]
            pos_avgs[pos_id] = np.mean(pos_preds) if pos_preds else 0.0

        underperformers = sum(
            1 for p in current_squad
            if pred_gw0_scores.get(p["player_id"], 0) < pos_avgs.get(p["pos_id"], 0)
        )
        if underperformers >= 4:
            return wc_key

    # 3. Bench boost: top-3 bench outfield pts > 18
    if "bench_boost" not in chips_used and bench_out:
        bench_pts = sum(pred_gw0_scores.get(p["player_id"], 0) for p in bench_out[:3])
        if bench_pts > 18:
            return "bench_boost"

    # 4. Triple captain: best player pred > 12 AND at home
    if "triple_captain" not in chips_used and current_squad:
        best  = max(current_squad, key=lambda p: pred_gw0_scores.get(p["player_id"], 0))
        b_pred = pred_gw0_scores.get(best["player_id"], 0)
        home   = best["feat_vec"].get("home_advantage", 0)
        if b_pred > 12 and home == 1:
            return "triple_captain"

    return None


# ---------------------------------------------------------------------------
# Online retraining
# ---------------------------------------------------------------------------
def simulate_actuals(player_pool, pred_gw0_scores, current_gw):
    """Simulate actual GW points: Normal(pred, 2.5) clipped to [-2, 26]."""
    rng = np.random.RandomState(SEED + current_gw)
    actuals = {}
    for p in player_pool:
        pid  = p["player_id"]
        pred = pred_gw0_scores.get(pid, 0.0)
        actuals[pid] = float(np.clip(rng.normal(pred, 2.5), -2, 26))
    return actuals


def retrain_models(train_dfs, new_rows_by_pos, best_params, prev_model_fcols=None):
    """
    Retrain all 4 XGBoost models with simulated actuals appended.
    Returns (new_models, new_model_fcols).
    """
    from xgboost import XGBRegressor

    new_models      = {}
    new_model_fcols = {}

    for pos_id in [1, 2, 3, 4]:
        df = train_dfs[pos_id]
        if new_rows_by_pos.get(pos_id):
            new_df    = pd.DataFrame(new_rows_by_pos[pos_id])
            feat_cols = get_feature_cols(df)
            for c in feat_cols:
                if c not in new_df.columns:
                    new_df[c] = 0.0
            if TARGET_COL not in new_df.columns:
                new_df[TARGET_COL] = 0.0
            df = pd.concat([df, new_df], ignore_index=True)
            train_dfs[pos_id] = df

        feat_cols = get_feature_cols(df)
        X = df[feat_cols].fillna(0)
        y = df[TARGET_COL].fillna(0)

        params = {k: v for k, v in best_params.get(pos_id, {}).items()}
        model  = XGBRegressor(**params, random_state=SEED, verbosity=0)
        model.fit(X, y)
        new_models[pos_id]      = model
        new_model_fcols[pos_id] = feat_cols

    print("  Models retrained.")
    return new_models, new_model_fcols


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
SEP = "=" * 60

def print_gw_block(sim_gw, real_gw, chip, result, pred_gw0_scores, dgw_gws):
    squad   = result["squad"]
    xi      = result["xi"]
    captain = result["captain"]
    vice    = result["vice"]

    bench_gk, bench_out = split_bench(squad, xi, pred_gw0_scores)

    cap_mult = 3 if chip == "triple_captain" else 2

    gw = sim_gw  # use simulation GW for display
    print()
    print(SEP)
    print(f"  GAMEWEEK {gw}  [CHIP: {chip or 'None'}]")
    print(SEP)

    xi_gk  = [p for p in xi if p["pos_id"] == 1]
    xi_def = [p for p in xi if p["pos_id"] == 2]
    xi_mid = [p for p in xi if p["pos_id"] == 3]
    xi_fwd = [p for p in xi if p["pos_id"] == 4]

    cap_id = captain["player_id"] if captain else -1
    vic_id = vice["player_id"]    if vice    else -1
    dgw_teams_now = set(dgw_gws.get(real_gw, []))

    def fmt(p, prefix):
        pid     = p["player_id"]
        pred    = pred_gw0_scores.get(pid, 0)
        c_str   = " [C]"   if pid == cap_id else ""
        v_str   = " [V]"   if pid == vic_id else ""
        d_str   = " [DGW]" if p["team_id"] in dgw_teams_now else ""
        return (
            f"    {prefix}{p['web_name']:<22} ({p['team_short']:<3})"
            f"  GBP{p['value']:.1f}m  pred:{pred:5.1f} pts"
            f"{c_str}{v_str}{d_str}"
        )

    print("  STARTING XI:")
    for p in xi_gk:  print(fmt(p, "GK:  "))
    for p in xi_def: print(fmt(p, "DEF: "))
    for p in xi_mid: print(fmt(p, "MID: "))
    for p in xi_fwd: print(fmt(p, "FWD: "))

    if captain:
        cp   = pred_gw0_scores.get(captain["player_id"], 0)
        home = "home" if captain["feat_vec"].get("home_advantage", 0) == 1 else "away"
        print(f"    * Captain: {captain['web_name']}  pred:{cp:5.1f} pts  ({home})  x{cap_mult}")
    if vice:
        vp = pred_gw0_scores.get(vice["player_id"], 0)
        print(f"    ~ Vice:    {vice['web_name']}  pred:{vp:5.1f} pts")

    print()
    print("  BENCH:")
    for p in bench_gk:
        print(f"    GK:  {p['web_name']:<22} ({p['team_short']:<3})  GBP{p['value']:.1f}m")
    labels = ["1st", "2nd", "3rd"]
    for rank, p in enumerate(bench_out[:3], 0):
        pred = pred_gw0_scores.get(p["player_id"], 0)
        print(
            f"    {labels[rank]}: {p['web_name']:<22} ({p['team_short']:<3})"
            f"  GBP{p['value']:.1f}m  pred:{pred:5.1f} pts"
        )

    # Bench swap warnings
    xi_by_pos = {}
    for p in xi:
        xi_by_pos.setdefault(p["pos_id"], []).append(p)
    for bench_p in bench_out:
        b_pred = pred_gw0_scores.get(bench_p["player_id"], 0)
        for starter in xi_by_pos.get(bench_p["pos_id"], []):
            s_pred = pred_gw0_scores.get(starter["player_id"], 0)
            if b_pred > s_pred:
                print(
                    f"  WARNING: {bench_p['web_name']} (bench, pred {b_pred:.1f})"
                    f" > {starter['web_name']} (XI, pred {s_pred:.1f}) -- consider swapping"
                )

    print()
    print("  TRANSFERS:")
    if result["transfers_in"]:
        for p in result["transfers_in"]:
            print(f"    IN:  {p['web_name']:<22} ({p['team_short']})  GBP{p['value']:.1f}m")
        for p in result["transfers_out"]:
            print(f"    OUT: {p['web_name']:<22} ({p['team_short']})  GBP{p['value']:.1f}m")
        pen     = result["penalty"]
        n_tr    = len(result["transfers_in"])
        cost_str = f"-{pen*4} pts" if pen > 0 else "Free"
        print(f"    Cost: {cost_str}  ({n_tr} transfer(s), penalty={pen})")
    else:
        if sim_gw == 1:
            print("    None  (GW1 fresh squad)")
        else:
            print("    None  (no transfers)")

    squad_val = sum(p["value"] for p in squad)
    xi_preds  = [pred_gw0_scores.get(p["player_id"], 0) for p in xi]
    cap_bonus = pred_gw0_scores.get(cap_id, 0) if cap_id != -1 else 0
    vc_bonus  = pred_gw0_scores.get(vic_id, 0) * 0.5 if vic_id != -1 else 0
    bench_pts = sum(pred_gw0_scores.get(p["player_id"], 0) for p in bench_gk + bench_out) \
                if chip == "bench_boost" else 0
    gw_pred   = sum(xi_preds) + cap_bonus + vc_bonus + bench_pts - result["penalty"] * 4

    print()
    print(f"  Squad value: GBP{squad_val:.1f}m")
    print(f"  Predicted GW score: {gw_pred:.1f} pts")
    print(SEP)

    return gw_pred


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  FPL AI Stage 8 -- ILP Squad Optimizer")
    print("=" * 60)

    print("\nLoading models...")
    models, model_fcols = load_models()
    best_params = load_best_params()

    print("\nLoading training data...")
    train_dfs = load_training_data()

    print("\nLoading FPL API data...")
    players, ph, fixtures, fdr_df, upcoming, teams = load_fpl_data()

    print("\nDetecting DGWs and BGWs...")
    dgw_gws      = detect_dgw(fixtures)
    all_team_ids = list(teams["id"].unique())
    all_gws      = list(range(1, 39))
    bgw_by_gw    = detect_bgw(fixtures, all_gws, all_team_ids)
    print(f"  DGW gameweeks: {sorted(dgw_gws.keys()) or 'None found'}")

    # Determine real GW offset for simulation
    sim_start_gw = get_sim_start_gw(fixtures, upcoming)
    print(f"  Simulation starts at real GW{sim_start_gw} (maps to sim GW1)")

    print("\nBuilding player pool...")
    player_pool = build_player_pool(players, train_dfs, upcoming, teams,
                                    sim_start_gw=sim_start_gw)

    # normalise upcoming gameweek column to int (for lookup)
    upcoming = upcoming.dropna(subset=["gameweek"]).copy()
    upcoming["gameweek"] = upcoming["gameweek"].astype(int)

    # --- Simulation state ---
    current_squad    = None
    free_transfers   = 1
    chips_used       = set()
    chips_log        = []
    gw_scores        = []
    squad_values     = []
    penalties_total  = 0
    captain_counts   = defaultdict(int)
    best_gw_info     = {"gw": None, "score": -999, "squad": []}
    new_rows_by_pos  = defaultdict(list)

    sim_log = {
        "gameweeks":              [],
        "total_predicted_points": 0,
        "total_penalties":        0,
        "chips_used":             [],
        "best_gw":                None,
        "most_captained":         None,
        "squad_value_trajectory": [],
    }

    for gw in range(1, 11):
        print(f"\n{'-'*60}")
        print(f"  Processing GW{gw}...")
        print(f"{'-'*60}")

        real_gw = sim_start_gw + (gw - 1)  # map sim GW to real season GW

        # --- Predict GW0 ---
        pred_gw0_scores = predict_gw0(player_pool, models, model_fcols)

        # --- 3-GW horizon ---
        horizon_scores = predict_horizon(
            player_pool, models, real_gw=real_gw,
            upcoming_df=upcoming, dgw_gws=dgw_gws, bgw_teams_by_gw=bgw_by_gw,
            model_fcols=model_fcols,
        )

        # --- Chip check (requires existing squad) ---
        chip       = None
        is_wildcard = False
        is_freehit  = False

        if current_squad is not None:
            # Estimate bench for bench_boost check (use current squad)
            xi_proxy       = sorted(
                [p for p in current_squad if p["pos_id"] != 1],
                key=lambda p: pred_gw0_scores.get(p["player_id"], 0),
                reverse=True,
            )[:10]
            xi_proxy_ids   = {p["player_id"] for p in xi_proxy}
            bench_out_proxy = [
                p for p in current_squad
                if p["player_id"] not in xi_proxy_ids and p["pos_id"] != 1
            ]

            chip_raw = check_chips(
                real_gw, dgw_gws, chips_used,
                current_squad, pred_gw0_scores, bench_out_proxy,
            )

            if chip_raw == "freehit":
                chip         = "freehit"
                is_freehit   = True
                chips_used.add("freehit")
                chips_log.append({"chip": "freehit", "gw": gw})
                free_transfers = 15  # unlimited (no cost)

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

        # --- ILP ---
        result = run_ilp(
            player_pool       = player_pool,
            horizon_scores    = horizon_scores,
            pred_gw0_scores   = pred_gw0_scores,
            budget            = BUDGET,
            prev_squad        = current_squad if gw > 1 else None,
            free_transfers    = free_transfers if gw > 1 else 1,
            is_wildcard       = is_wildcard,
            is_freehit        = is_freehit,
        )

        if result is None:
            print(f"  GW{gw}: ILP failed, skipping.")
            continue

        # Freehit: squad reverts next GW (don't update current_squad permanently)
        if not is_freehit:
            current_squad = result["squad"]

        # Captain tracking
        if result["captain"]:
            captain_counts[result["captain"]["web_name"]] += 1

        # --- Print GW block ---
        gw_score = print_gw_block(gw, real_gw, chip, result, pred_gw0_scores, dgw_gws)

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

        sim_log["gameweeks"].append({
            "gw":           gw,
            "chip":         chip,
            "predicted_score": gw_score,
            "squad_value":  squad_val,
            "transfers_in":  [p["web_name"] for p in result["transfers_in"]],
            "transfers_out": [p["web_name"] for p in result["transfers_out"]],
            "penalty_pts":  result["penalty"] * 4,
            "captain":      result["captain"]["web_name"] if result["captain"] else None,
            "xi":           [p["web_name"] for p in result["xi"]],
        })

        # --- Online retraining (for next GW) ---
        if gw < 10:
            print(f"\n  Simulating actuals + retraining for GW{gw+1}...")
            actuals = simulate_actuals(player_pool, pred_gw0_scores, gw)

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

        # --- Update free transfers ---
        if is_wildcard or is_freehit:
            free_transfers = 1  # reset after chip
        else:
            n_tr = len(result["transfers_in"])
            if n_tr <= free_transfers:
                # unused free transfer rolls over (max 2)
                free_transfers = min(2, free_transfers - n_tr + 1)
            else:
                free_transfers = 1

    # -----------------------------------------------------------------------
    # Season Summary
    # -----------------------------------------------------------------------
    total_pred    = sum(gw_scores)
    most_captained = (
        max(captain_counts, key=captain_counts.get) if captain_counts else "N/A"
    )

    print()
    print(SEP)
    print("  SEASON SUMMARY  (GW1-GW10)")
    print(SEP)
    print(f"  Total predicted points   : {total_pred:.1f}")
    print(f"  Total transfer penalties : -{penalties_total} pts")
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
    print(SEP)

    # --- Save JSON log ---
    sim_log["total_predicted_points"] = total_pred
    sim_log["total_penalties"]        = penalties_total
    sim_log["chips_used"]             = chips_log
    sim_log["best_gw"]                = best_gw_info
    sim_log["most_captained"]         = most_captained
    sim_log["squad_value_trajectory"] = [
        {"gw": i + 1, "value": v} for i, v in enumerate(squad_values)
    ]

    out_path = os.path.join(MODELS_DIR, "stage8_simulation.json")
    with open(out_path, "w") as f:
        json.dump(sim_log, f, indent=2)
    print(f"\nSimulation log saved -> {out_path}")


if __name__ == "__main__":
    main()
