"""
GW28 Single-Gameweek Optimizer
================================
Completely standalone from Stage 8 / Intel 06.

Objective: pick the single best 15-man squad + starting XI to maximise
predicted points in GW28 only.  No future horizon.  No transfer costs.
No chip logic.  Just the highest possible predicted GW28 score.

Pipeline:
  1. Load player_history.csv (2025-26 GW1-29) + supporting FPL data
  2. Build per-player, per-GW rolling features from GW1-27 actuals
  3. Train 4 fresh XGBoost models (GK/DEF/MID/FWD) on those features
  4. Snapshot GW28 features for every player (using GW1-27 history)
  5. Apply intel adjustments: FDR adj, zero-minutes filter, ownership boost
  6. ILP optimizer -> best 15-man squad + XI
  7. Print squad report including actual GW28 points alongside predictions
  8. Hindsight section: ILP on actual points -> theoretical upper bound
"""

import os
import warnings

import numpy as np
import pandas as pd
import pulp
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw", "fpl_api")

TARGET_GW      = 28
TRAIN_UP_TO_GW = TARGET_GW - 1   # 28

BUDGET           = 100.0
MAX_CLUB         = 3
SEED             = 42
FDR_MULT         = 0.03    # pred *= (1 - FDR_MULT * (fdr - 3))  clipped at 0.5
OWNERSHIP_WEIGHT = 0.02    # pred += selected_by_percent * OWNERSHIP_WEIGHT

POSITIONS    = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
CAP_POS_MULT = {1: 0.0, 2: 0.75, 3: 1.15, 4: 1.25}

np.random.seed(SEED)

# Feature columns used for each position.
# saves_per_game is meaningful only for GK; outfield gets it too but ~0.
FEAT_COLS = [
    "form_last3",
    "form_last5",
    "avg_ppg",
    "minutes_reliability",
    "goals_per_game",
    "assists_per_game",
    "clean_sheet_rate",
    "bonus_per_game",
    "saves_per_game",
    "value",
    "was_home",
    "fdr",
]


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data():
    ph       = pd.read_csv(os.path.join(DATA_RAW, "player_history.csv"))
    players  = pd.read_csv(os.path.join(DATA_RAW, "players_raw.csv"))
    fixtures = pd.read_csv(os.path.join(DATA_RAW, "fixtures_raw.csv"))
    teams    = pd.read_csv(os.path.join(DATA_RAW, "teams_raw.csv"))

    # Ensure was_home is boolean-safe
    ph["was_home"] = ph["was_home"].astype(bool)

    # Player lookup maps
    pid_to_team  = dict(zip(players["id"].astype(int), players["team"].astype(int)))
    pid_to_pos   = dict(zip(players["id"].astype(int), players["element_type"].astype(int)))
    pid_to_name  = dict(zip(
        players["id"].astype(int),
        players["first_name"].str.strip() + " " + players["second_name"].str.strip()
    ))
    pid_to_price = dict(zip(players["id"].astype(int), players["now_cost"].astype(float) / 10.0))
    pid_to_ownership = dict(zip(
        players["id"].astype(int),
        players["selected_by_percent"].fillna(0.0).astype(float)
    ))

    # Team lookup maps
    team_id_to_name  = dict(zip(teams["id"].astype(int), teams["name"]))
    team_id_to_short = dict(zip(teams["id"].astype(int), teams["short_name"]))

    # (team_id, gw) -> fdr lookup from fixtures
    fx = fixtures.dropna(subset=["gameweek"]).copy()
    fx["gameweek"] = fx["gameweek"].astype(int)
    team_gw_fdr = {}
    for _, row in fx.iterrows():
        gw  = int(row["gameweek"])
        th  = int(row["team_h"])
        ta  = int(row["team_a"])
        team_gw_fdr[(th, gw)] = float(row["team_h_difficulty"])
        team_gw_fdr[(ta, gw)] = float(row["team_a_difficulty"])

    # GW29 teams & home/away
    gw29_fx        = fx[fx["gameweek"] == TARGET_GW]
    gw29_home_teams = set(gw29_fx["team_h"].astype(int))
    gw29_away_teams = set(gw29_fx["team_a"].astype(int))
    gw29_teams      = gw29_home_teams | gw29_away_teams

    print(f"Loaded {len(ph)} player-GW rows | {len(players)} players | {len(gw29_fx)} GW{TARGET_GW} fixtures")
    print(f"GW{TARGET_GW} has {len(gw29_teams)} teams playing (no BGW)")

    return dict(
        ph=ph,
        players=players,
        pid_to_team=pid_to_team,
        pid_to_pos=pid_to_pos,
        pid_to_name=pid_to_name,
        pid_to_price=pid_to_price,
        pid_to_ownership=pid_to_ownership,
        team_id_to_name=team_id_to_name,
        team_id_to_short=team_id_to_short,
        team_gw_fdr=team_gw_fdr,
        gw29_home_teams=gw29_home_teams,
        gw29_teams=gw29_teams,
    )


# ---------------------------------------------------------------------------
# 2. Build training dataset  (GW2 to TRAIN_UP_TO_GW per player)
#
#    For each player, for each GW t in [2, TRAIN_UP_TO_GW]:
#      features  = rolling stats computed from GW1..(t-1) history
#                  + current-GW was_home / value / fdr
#      target    = total_points at GW t
# ---------------------------------------------------------------------------
def build_training_data(data):
    ph          = data["ph"]
    pid_to_team = data["pid_to_team"]
    pid_to_pos  = data["pid_to_pos"]
    team_gw_fdr = data["team_gw_fdr"]

    ph_train = ph[ph["gameweek"] <= TRAIN_UP_TO_GW].copy()

    rows_by_pos = {1: [], 2: [], 3: [], 4: []}

    for pid_raw, grp in ph_train.groupby("player_id"):
        pid    = int(pid_raw)
        pos_id = pid_to_pos.get(pid)
        if pos_id not in POSITIONS:
            continue
        team_id = pid_to_team.get(pid)
        if team_id is None:
            continue

        grp = grp.sort_values("gameweek").reset_index(drop=True)

        for i in range(1, len(grp)):
            hist = grp.iloc[:i]     # prior GWs (features)
            cur  = grp.iloc[i]      # current GW (target + fixture info)
            gw   = int(cur["gameweek"])

            pts     = hist["total_points"].values.astype(float)
            mins    = hist["minutes"].values.astype(float)
            goals   = hist["goals_scored"].values.astype(float)
            assists = hist["assists"].values.astype(float)
            cs      = hist["clean_sheets"].values.astype(float)
            saves   = hist["saves"].values.astype(float)
            bonus   = hist["bonus"].values.astype(float)
            n       = len(pts)

            row = {
                "form_last3":        float(np.mean(pts[-3:])),
                "form_last5":        float(np.mean(pts[-5:])),
                "avg_ppg":           float(np.mean(pts)),
                "minutes_reliability": float(np.sum(mins)) / (n * 90.0),
                "goals_per_game":    float(np.mean(goals)),
                "assists_per_game":  float(np.mean(assists)),
                "clean_sheet_rate":  float(np.mean(cs)),
                "bonus_per_game":    float(np.mean(bonus)),
                "saves_per_game":    float(np.mean(saves)),
                # value in player_history is already in £m units
                "value":             float(cur["value"]),
                "was_home":          int(bool(cur["was_home"])),
                "fdr":               float(team_gw_fdr.get((team_id, gw), 3.0)),
                "total_points":      float(cur["total_points"]),
            }
            rows_by_pos[pos_id].append(row)

    train_dfs = {}
    for pos_id in [1, 2, 3, 4]:
        df = pd.DataFrame(rows_by_pos[pos_id])
        train_dfs[pos_id] = df
        print(f"  {POSITIONS[pos_id]:<3}: {len(df):>5} training rows")

    return train_dfs


# ---------------------------------------------------------------------------
# 3. Train XGBoost models  (one per position)
# ---------------------------------------------------------------------------
def train_models(train_dfs):
    models = {}
    for pos_id in [1, 2, 3, 4]:
        df = train_dfs[pos_id]
        X  = df[FEAT_COLS].fillna(0.0)
        y  = df["total_points"].fillna(0.0)

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=SEED,
            verbosity=0,
        )
        model.fit(X, y)
        models[pos_id] = model
        print(f"  {POSITIONS[pos_id]:<3}: trained on {len(X):>5} rows | {len(FEAT_COLS)} features")

    return models


# ---------------------------------------------------------------------------
# 4. Build GW29 feature snapshot  (features from full GW1-28 history)
# ---------------------------------------------------------------------------
def build_gw29_features(data, models):
    ph               = data["ph"]
    pid_to_team      = data["pid_to_team"]
    pid_to_pos       = data["pid_to_pos"]
    pid_to_name      = data["pid_to_name"]
    pid_to_price     = data["pid_to_price"]
    pid_to_ownership = data["pid_to_ownership"]
    team_gw_fdr      = data["team_gw_fdr"]
    gw29_home_teams  = data["gw29_home_teams"]
    gw29_teams       = data["gw29_teams"]
    team_id_to_name  = data["team_id_to_name"]
    team_id_to_short = data["team_id_to_short"]

    ph_history = ph[ph["gameweek"] <= TRAIN_UP_TO_GW].copy()

    # GW29 actuals (all 820 players have a row; minutes=0 if didn't play)
    ph_gw29      = ph[ph["gameweek"] == TARGET_GW].copy()
    gw29_actuals = dict(zip(
        ph_gw29["player_id"].astype(int),
        ph_gw29["total_points"].astype(float)
    ))

    player_pool         = []
    skipped_zero_mins   = 0
    skipped_no_fixture  = 0
    skipped_unknown_pos = 0

    for pid_raw, grp in ph_history.groupby("player_id"):
        pid    = int(pid_raw)
        pos_id = pid_to_pos.get(pid)
        if pos_id not in POSITIONS:
            skipped_unknown_pos += 1
            continue

        team_id = pid_to_team.get(pid)
        if team_id is None:
            continue

        # BGW filter (no GW29 fixture -> 0 pts expected)
        if team_id not in gw29_teams:
            skipped_no_fixture += 1
            continue

        grp           = grp.sort_values("gameweek").reset_index(drop=True)
        total_minutes = float(grp["minutes"].sum())

        # Zero-minutes filter: no playing time all season -> exclude
        if total_minutes == 0:
            skipped_zero_mins += 1
            continue

        pts     = grp["total_points"].values.astype(float)
        mins    = grp["minutes"].values.astype(float)
        goals   = grp["goals_scored"].values.astype(float)
        assists = grp["assists"].values.astype(float)
        cs      = grp["clean_sheets"].values.astype(float)
        saves   = grp["saves"].values.astype(float)
        bonus   = grp["bonus"].values.astype(float)
        n       = len(pts)

        was_home_29 = 1 if team_id in gw29_home_teams else 0
        fdr_29      = float(team_gw_fdr.get((team_id, TARGET_GW), 3.0))
        price       = pid_to_price.get(pid, 5.0)   # current price from players_raw

        feats = {
            "form_last3":        float(np.mean(pts[-3:])) if n >= 3 else float(np.mean(pts)),
            "form_last5":        float(np.mean(pts[-5:])) if n >= 5 else float(np.mean(pts)),
            "avg_ppg":           float(np.mean(pts)),
            "minutes_reliability": float(np.sum(mins)) / (n * 90.0),
            "goals_per_game":    float(np.mean(goals)),
            "assists_per_game":  float(np.mean(assists)),
            "clean_sheet_rate":  float(np.mean(cs)),
            "bonus_per_game":    float(np.mean(bonus)),
            "saves_per_game":    float(np.mean(saves)),
            "value":             price,
            "was_home":          was_home_29,
            "fdr":               fdr_29,
        }

        X        = pd.DataFrame([feats])[FEAT_COLS].fillna(0.0)
        raw_pred = max(0.0, float(models[pos_id].predict(X)[0]))

        player_pool.append({
            "player_id":   pid,
            "name":        pid_to_name.get(pid, f"Player {pid}"),
            "pos_id":      pos_id,
            "pos_name":    POSITIONS[pos_id],
            "team_id":     team_id,
            "team_name":   team_id_to_name.get(team_id, "Unknown"),
            "team_short":  team_id_to_short.get(team_id, "???"),
            "value":       price,
            "was_home":    was_home_29,
            "fdr":         fdr_29,
            "ownership":   pid_to_ownership.get(pid, 0.0),
            "raw_pred":    raw_pred,
            "gw29_actual": gw29_actuals.get(pid, 0.0),
        })

    print(f"\nPlayer pool: {len(player_pool)} players")
    print(f"  Skipped (0 total minutes GW1-28): {skipped_zero_mins}")
    print(f"  Skipped (no GW29 fixture):        {skipped_no_fixture}")

    return player_pool, gw29_actuals


# ---------------------------------------------------------------------------
# 5. Apply intel adjustments
#    (a) FDR adjustment  (b) ownership boost
#    Zero-minutes players already removed in step 4
# ---------------------------------------------------------------------------
def apply_intel(player_pool):
    for p in player_pool:
        pred    = p["raw_pred"]
        fdr_adj = max(0.5, 1.0 - FDR_MULT * (p["fdr"] - 3.0))
        pred   *= fdr_adj
        pred   += p["ownership"] * OWNERSHIP_WEIGHT
        p["pred"] = max(0.0, pred)
    return player_pool


# ---------------------------------------------------------------------------
# 6. ILP optimizer
#    score_key: which column in player_pool dicts to use as the objective
# ---------------------------------------------------------------------------
def run_ilp(player_pool, score_key="pred", budget=BUDGET, label=""):
    n         = len(player_pool)
    ids       = [p["player_id"]          for p in player_pool]
    scores    = [p.get(score_key, 0.0)   for p in player_pool]
    costs     = [p["value"]              for p in player_pool]
    team_ids  = [p["team_id"]            for p in player_pool]

    is_gk  = [1 if p["pos_id"] == 1 else 0 for p in player_pool]
    is_def = [1 if p["pos_id"] == 2 else 0 for p in player_pool]
    is_mid = [1 if p["pos_id"] == 3 else 0 for p in player_pool]
    is_fwd = [1 if p["pos_id"] == 4 else 0 for p in player_pool]

    def solve(budget_limit):
        prob = pulp.LpProblem(f"FPL_GW{TARGET_GW}{label}", pulp.LpMaximize)

        x  = [pulp.LpVariable(f"x_{i}",  cat="Binary") for i in range(n)]
        s  = [pulp.LpVariable(f"s_{i}",  cat="Binary") for i in range(n)]
        c  = [pulp.LpVariable(f"c_{i}",  cat="Binary") for i in range(n)]
        vc = [pulp.LpVariable(f"vc_{i}", cat="Binary") for i in range(n)]

        # Objective:
        #   sum(score * starter) + score_captain * 1.0 (extra for captaincy = 2×)
        #   + 0.5 * score_vc (partial VC credit to break ties sensibly)
        prob += (
            pulp.lpSum(scores[i] * s[i]         for i in range(n))
            + pulp.lpSum(scores[i] * c[i]       for i in range(n))
            + pulp.lpSum(scores[i] * 0.5 * vc[i] for i in range(n))
        )

        # --- Squad structure ---
        prob += pulp.lpSum(x) == 15
        prob += pulp.lpSum(is_gk[i]  * x[i] for i in range(n)) == 2
        prob += pulp.lpSum(is_def[i] * x[i] for i in range(n)) == 5
        prob += pulp.lpSum(is_mid[i] * x[i] for i in range(n)) == 5
        prob += pulp.lpSum(is_fwd[i] * x[i] for i in range(n)) == 3

        # Budget
        prob += pulp.lpSum(costs[i] * x[i] for i in range(n)) <= budget_limit

        # Max 3 per club
        for tid in set(team_ids):
            prob += pulp.lpSum(x[i] for i in range(n) if team_ids[i] == tid) <= MAX_CLUB

        # --- XI structure (standard FPL formations: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD) ---
        prob += pulp.lpSum(s) == 11
        prob += pulp.lpSum(is_gk[i]  * s[i] for i in range(n)) == 1
        prob += pulp.lpSum(is_def[i] * s[i] for i in range(n)) >= 3
        prob += pulp.lpSum(is_def[i] * s[i] for i in range(n)) <= 5
        prob += pulp.lpSum(is_mid[i] * s[i] for i in range(n)) >= 2
        prob += pulp.lpSum(is_mid[i] * s[i] for i in range(n)) <= 5
        prob += pulp.lpSum(is_fwd[i] * s[i] for i in range(n)) >= 1
        prob += pulp.lpSum(is_fwd[i] * s[i] for i in range(n)) <= 3

        # Can only start if in squad
        for i in range(n):
            prob += s[i] <= x[i]

        # --- Captain / Vice in XI, not same player ---
        prob += pulp.lpSum(c)  == 1
        prob += pulp.lpSum(vc) == 1
        for i in range(n):
            prob += c[i]  <= s[i]
            prob += vc[i] <= s[i]
            prob += c[i] + vc[i] <= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            return None

        def val(v):
            return round(pulp.value(v) or 0)

        return {
            "squad":   [player_pool[i] for i in range(n) if val(x[i])  == 1],
            "xi":      [player_pool[i] for i in range(n) if val(s[i])  == 1],
            "captain": next((player_pool[i] for i in range(n) if val(c[i])  == 1), None),
            "vice":    next((player_pool[i] for i in range(n) if val(vc[i]) == 1), None),
        }

    for attempt in range(4):
        result = solve(budget + attempt * 0.5)
        if result is not None:
            if attempt > 0:
                print(f"  ILP solved with budget relaxation +{attempt * 0.5:.1f}m")
            return result

    print("WARNING: ILP infeasible after 4 attempts.")
    return None


# ---------------------------------------------------------------------------
# 7. Captain selection  (post-process with positional bias)
# ---------------------------------------------------------------------------
def select_captain(xi, score_key="pred"):
    """
    Apply CAP_POS_MULT to each XI player's score.
    GK: never captain.  DEF: only if adj > 12 AND form > 8.
    Returns (captain, vice, candidates_top5).
    """
    candidates = []
    for p in xi:
        raw  = p.get(score_key, 0.0)
        mult = CAP_POS_MULT.get(p["pos_id"], 1.0)
        adj  = raw * mult
        if p["pos_id"] == 1:
            continue
        if p["pos_id"] == 2 and (adj <= 12 or p.get("form_last3", 0) <= 8):
            continue
        candidates.append((adj, raw, p))

    if not candidates:
        fallback = sorted(
            [p for p in xi if p["pos_id"] in (3, 4)],
            key=lambda p: p.get(score_key, 0.0),
            reverse=True,
        )
        if len(fallback) >= 2:
            return fallback[0], fallback[1], []
        if len(fallback) == 1:
            return fallback[0], fallback[0], []
        return None, None, []

    candidates.sort(key=lambda x: x[0], reverse=True)
    cap  = candidates[0][2]
    vice = candidates[1][2] if len(candidates) > 1 else candidates[0][2]
    return cap, vice, candidates[:5]


# ---------------------------------------------------------------------------
# 8. Print main squad report
# ---------------------------------------------------------------------------
def _ha(was_home):
    return "H" if was_home else "A"


def print_squad_report(result, score_key="pred"):
    if result is None:
        print("No valid squad found.")
        return 0.0

    xi     = result["xi"]
    squad  = result["squad"]
    cap    = result["captain"]
    vc     = result["vice"]

    cap_id = cap["player_id"] if cap else -1
    vc_id  = vc["player_id"]  if vc else -1

    xi_ids       = {p["player_id"] for p in xi}
    bench        = [p for p in squad if p["player_id"] not in xi_ids]
    pos_order    = {1: 0, 2: 1, 3: 2, 4: 3}
    xi_sorted    = sorted(xi,    key=lambda p: (pos_order[p["pos_id"]], -p.get(score_key, 0.0)))
    bench_gk     = [p for p in bench if p["pos_id"] == 1]
    bench_out    = sorted([p for p in bench if p["pos_id"] != 1],
                          key=lambda p: -p.get(score_key, 0.0))
    bench_sorted = bench_gk + bench_out

    total_cost = sum(p["value"] for p in squad)

    W  = 74
    print("\n" + "=" * W)
    print(f"  GW{TARGET_GW} PREDICTED OPTIMAL SQUAD  "
          f"(trained on GW1-{TRAIN_UP_TO_GW} actuals)")
    print("=" * W)
    print(f"  {'Player':<26} {'Pos':<4} {'Club':<12} {'£':>5} "
          f"{'Pred':>6} {'Actual':>7}  {'H/A'} FDR")
    print("-" * W)

    xi_pred_total   = 0.0
    xi_actual_total = 0.0

    for p in xi_sorted:
        pid     = p["player_id"]
        marker  = " [C]" if pid == cap_id else " [V]" if pid == vc_id else ""
        pred    = p.get(score_key, 0.0)
        actual  = p.get("gw29_actual", 0.0)
        ha      = _ha(p["was_home"])
        cap_mul = 2 if pid == cap_id else 1

        xi_pred_total   += pred   * cap_mul
        xi_actual_total += actual * cap_mul

        name_field = (p["name"][:22] + marker)
        print(f"  {name_field:<26} {p['pos_name']:<4} {p['team_short']:<12} "
              f"{p['value']:>5.1f} {pred:>6.2f} {actual:>7.0f}   {ha}  {p['fdr']:.0f}")

    print("-" * W)
    print(f"  {'XI Total':<26} {'':4} {'':12} {'':5} "
          f"{xi_pred_total:>6.2f} {xi_actual_total:>7.0f}")

    print(f"\n  {'--- BENCH ---'}")
    bench_actual = 0.0
    for i, p in enumerate(bench_sorted):
        pred   = p.get(score_key, 0.0)
        actual = p.get("gw29_actual", 0.0)
        bench_actual += actual
        print(f"  {i+1}. {p['name']:<24} {p['pos_name']:<4} "
              f"{p['team_short']:<12} £{p['value']:.1f}  "
              f"pred {pred:.2f}  actual {actual:.0f}")

    print()
    print(f"  Squad cost :         £{total_cost:.1f}m  "
          f"(remaining £{BUDGET - total_cost:.1f}m)")
    print(f"  Predicted XI total:  {xi_pred_total:.2f} pts  (with 2x captain)")
    print(f"  Actual XI total:     {xi_actual_total:.0f} pts  (with 2x captain)")
    print(f"  Actual bench total:  {bench_actual:.0f} pts")

    # --- Top 5 captain candidates ---
    _, _, top5 = select_captain(xi, score_key=score_key)
    if top5:
        print(f"\n  TOP 5 CAPTAIN CANDIDATES")
        print(f"  {'Player':<26} {'Pos':<4} {'Raw Pred':>9} {'Adj Pred':>9}  Actual")
        print("  " + "-" * 55)
        for adj, raw, p in top5:
            marker = " *" if p["player_id"] == cap_id else ""
            actual = p.get("gw29_actual", 0.0)
            print(f"  {(p['name'][:24] + marker):<26} {p['pos_name']:<4} "
                  f"{raw:>9.2f} {adj:>9.2f}  {actual:.0f}")

    print("=" * W)
    return xi_actual_total


# ---------------------------------------------------------------------------
# 9. Hindsight optimal  — best possible XI using actual GW29 points
# ---------------------------------------------------------------------------
def hindsight_optimal(player_pool, gw29_actuals):
    W = 74
    print()
    print("=" * W)
    print(f"  WHAT ACTUALLY WON GW{TARGET_GW} -- HINDSIGHT OPTIMAL SQUAD")
    print("=" * W)
    print("  Best possible 15-man squad + XI given actual GW29 scores,")
    print("  subject to the same budget / formation / club constraints.")
    print("  This is the theoretical ceiling -- the perfect hindsight squad.")
    print()

    # Inject actual scores as the ILP objective
    pool = []
    for p in player_pool:
        pc = dict(p)
        pc["hindsight_score"] = float(gw29_actuals.get(p["player_id"], 0.0))
        pool.append(pc)

    result = run_ilp(pool, score_key="hindsight_score", label="_hindsight")

    if result is None:
        print("Could not compute hindsight optimal squad.")
        return 0.0

    xi     = result["xi"]
    squad  = result["squad"]
    cap    = result["captain"]
    vc     = result["vice"]

    cap_id = cap["player_id"] if cap else -1
    vc_id  = vc["player_id"]  if vc else -1

    xi_ids       = {p["player_id"] for p in xi}
    bench        = [p for p in squad if p["player_id"] not in xi_ids]
    pos_order    = {1: 0, 2: 1, 3: 2, 4: 3}
    xi_sorted    = sorted(xi,    key=lambda p: (pos_order[p["pos_id"]], -p.get("hindsight_score", 0.0)))
    bench_gk     = [p for p in bench if p["pos_id"] == 1]
    bench_out    = sorted([p for p in bench if p["pos_id"] != 1],
                          key=lambda p: -p.get("hindsight_score", 0.0))
    bench_sorted = bench_gk + bench_out

    total_cost = sum(p["value"] for p in squad)

    print(f"  {'Player':<26} {'Pos':<4} {'Club':<12} {'£':>5} {'Actual':>7}  {'H/A'} FDR")
    print("-" * W)

    xi_actual_total = 0.0
    for p in xi_sorted:
        pid     = p["player_id"]
        actual  = p.get("hindsight_score", 0.0)
        marker  = " [C]" if pid == cap_id else " [V]" if pid == vc_id else ""
        ha      = _ha(p["was_home"])
        cap_mul = 2 if pid == cap_id else 1

        xi_actual_total += actual * cap_mul

        name_field = (p["name"][:22] + marker)
        print(f"  {name_field:<26} {p['pos_name']:<4} {p['team_short']:<12} "
              f"{p['value']:>5.1f} {actual:>7.0f}   {ha}  {p['fdr']:.0f}")

    print("-" * W)
    print(f"  {'XI Total':<26} {'':4} {'':12} {total_cost:>5.1f} {xi_actual_total:>7.0f}")

    print(f"\n  {'--- BENCH ---'}")
    bench_actual = 0.0
    for i, p in enumerate(bench_sorted):
        actual = p.get("hindsight_score", 0.0)
        bench_actual += actual
        print(f"  {i+1}. {p['name']:<24} {p['pos_name']:<4} "
              f"{p['team_short']:<12} £{p['value']:.1f}  actual {actual:.0f}")

    print()
    print(f"  Squad cost:               £{total_cost:.1f}m  "
          f"(remaining £{BUDGET - total_cost:.1f}m)")
    print(f"  Hindsight XI total:        {xi_actual_total:.0f} pts  (with 2x captain)")
    print(f"  Hindsight bench total:     {bench_actual:.0f} pts")
    print("=" * W)

    return xi_actual_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    W = 74
    print("=" * W)
    print(f"  FPL GW{TARGET_GW} STANDALONE OPTIMIZER")
    print(f"  Train on GW1-{TRAIN_UP_TO_GW} actuals  |  Predict GW{TARGET_GW}")
    print(f"  Budget £{BUDGET:.1f}m  |  Max {MAX_CLUB} per club  |  No chips / transfer costs")
    print("=" * W)

    print(f"\n[1/6] Loading FPL data...")
    data = load_data()

    print(f"\n[2/6] Building training dataset (GW2-{TRAIN_UP_TO_GW} rolling features)...")
    train_dfs = build_training_data(data)

    print(f"\n[3/6] Training XGBoost models (one per position)...")
    models = train_models(train_dfs)

    print(f"\n[4/6] Building GW{TARGET_GW} feature snapshot from GW1-{TRAIN_UP_TO_GW} history...")
    player_pool, gw29_actuals = build_gw29_features(data, models)

    print(f"\n[5/6] Applying intel adjustments "
          f"(FDR mult={FDR_MULT}, ownership weight={OWNERSHIP_WEIGHT})...")
    player_pool = apply_intel(player_pool)

    # Summary of predictions before ILP
    preds = sorted(player_pool, key=lambda p: -p["pred"])
    print(f"  Top predicted players:")
    for p in preds[:5]:
        print(f"    {p['name']:<30} {p['pos_name']:<4} pred={p['pred']:.2f}  "
              f"actual={p['gw29_actual']:.0f}  £{p['value']:.1f}m")

    print(f"\n[6/6] Running ILP optimizer...")
    result = run_ilp(player_pool, score_key="pred")

    # Post-process captain using positional bias
    if result:
        cap, vice, _ = select_captain(result["xi"], score_key="pred")
        result["captain"] = cap
        result["vice"]    = vice

    our_score = print_squad_report(result, score_key="pred")

    print(f"\n[BONUS] Computing hindsight optimal (actual GW{TARGET_GW} points as objective)...")
    hindsight_score = hindsight_optimal(player_pool, gw29_actuals)

    # --- Final comparison ---
    print()
    print("=" * W)
    print("  PREDICTION vs HINDSIGHT COMPARISON")
    print("=" * W)
    print(f"  Our optimizer (actual):    {our_score:.0f} pts  (with 2x captain)")
    print(f"  Hindsight optimal:         {hindsight_score:.0f} pts  (theoretical max)")
    if hindsight_score > 0:
        efficiency = our_score / hindsight_score * 100
        gap        = hindsight_score - our_score
        print(f"  Efficiency:                {efficiency:.1f}% of theoretical maximum")
        print(f"  Pts left on table:         {gap:.0f} pts")
    print("=" * W)


if __name__ == "__main__":
    main()
