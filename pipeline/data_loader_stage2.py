"""
Stage 2: Vaastav Historical Data Loader
Loads, cleans and engineers features from 6 seasons of FPL historical GW data.
Seasons: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25
2025-26 is EXCLUDED (live season — reserved for GW34 demo only).
Usage: python pipeline/data_loader_stage2.py
"""

import os
import sys
import io
import pandas as pd
import numpy as np

# Force UTF-8 stdout on Windows to handle accented player names in print output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAASTAV_DIR = os.path.join(BASE_DIR, "Fantasy-Premier-League", "data")
FPL_API_DIR = os.path.join(BASE_DIR, "data", "raw", "fpl_api")
OUT_DIR = os.path.join(BASE_DIR, "data", "raw", "vaastav")
os.makedirs(OUT_DIR, exist_ok=True)

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
SEASON_YEAR = {"2019-20": 2020, "2020-21": 2021, "2021-22": 2022,
               "2022-23": 2023, "2023-24": 2024, "2024-25": 2025}

# 2025-26 is the live current season — NEVER load it here
BLOCKED_SEASONS = {"2025-26"}

# Columns we want to extract (handles missing gracefully)
DESIRED_COLS = [
    "name", "position", "team", "GW", "opponent_team", "was_home",
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "influence", "creativity", "threat", "ict_index",
    "transfers_in", "transfers_out", "selected", "value",
]

POSITION_MAP = {
    1: "GK", 2: "DEF", 3: "MID", 4: "FWD",
    "Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD",
    "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD",
}

# ---------------------------------------------------------------------------
# Team name mapping: vaastav names -> canonical (FPL API style)
# The vaastav teams.csv already uses the same names as the FPL API for
# historically present teams. We load teams_raw.csv to build the canonical
# set and flag anything that differs.
# ---------------------------------------------------------------------------

def build_team_name_mapping():
    """Build canonical name mapping from API teams and vaastav season teams."""
    api_teams_path = os.path.join(FPL_API_DIR, "teams_raw.csv")
    api_teams = read_csv_safe(api_teams_path)
    canonical_names = set(api_teams["name"].tolist())

    # Collect all team names from vaastav teams.csv files
    vaastav_names = set()
    for season in SEASONS:
        teams_csv = os.path.join(VAASTAV_DIR, season, "teams.csv")
        if os.path.exists(teams_csv):
            t = read_csv_safe(teams_csv)
            vaastav_names.update(t["name"].tolist())

    # Manual overrides for known divergences
    manual_map = {
        "Man City":       "Man City",
        "Man Utd":        "Man Utd",
        "Nott'm Forest":  "Nott'm Forest",
        "Spurs":          "Spurs",
        "West Brom":      "West Brom",
        "Sheffield Utd":  "Sheffield Utd",
        "Norwich":        "Norwich",
        "Watford":        "Watford",
        "Burnley":        "Burnley",
        "Leicester":      "Leicester",
        "Southampton":    "Southampton",
        "Leeds":          "Leeds",
        "Luton":          "Luton",
        "Brentford":      "Brentford",
        "Fulham":         "Fulham",
        "Ipswich":        "Ipswich",
    }

    mapping = {}
    unmapped = []
    for name in sorted(vaastav_names):
        if name in canonical_names:
            mapping[name] = name
        elif name in manual_map:
            mapping[name] = manual_map[name]
        else:
            mapping[name] = name  # keep as-is, flag below
            unmapped.append(name)

    return mapping, unmapped


def read_csv_safe(path):
    """Read a CSV trying UTF-8 first, falling back to latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def build_season_team_id_map(season):
    """Return dict: team_id -> team_name for a given season's teams.csv."""
    teams_csv = os.path.join(VAASTAV_DIR, season, "teams.csv")
    t = read_csv_safe(teams_csv)
    return dict(zip(t["id"].astype(int), t["name"]))


# ---------------------------------------------------------------------------
# Step 1: Load a single season
# ---------------------------------------------------------------------------

def load_season(season, team_name_map, season_team_id_map):
    """Load merged_gw.csv for one season and return a cleaned DataFrame."""
    gw_path = os.path.join(VAASTAV_DIR, season, "gws", "merged_gw.csv")
    df = read_csv_safe(gw_path)

    raw_row_count = len(df)
    missing_cols = []

    # --- Handle 2019-20 which lacks position and team columns ---
    if "position" not in df.columns or "team" not in df.columns:
        players_raw_path = os.path.join(VAASTAV_DIR, season, "players_raw.csv")
        players_raw = read_csv_safe(players_raw_path)

        # Build lookup: player_id -> (position_code, team_id)
        player_lookup = players_raw[["id", "element_type", "team"]].copy()
        player_lookup.columns = ["element", "element_type", "team_id"]

        # Merge on element (player ID)
        df = df.merge(player_lookup, on="element", how="left")

        # Map element_type (1-4) to position string
        df["position"] = df["element_type"].map(POSITION_MAP)

        # Map team_id to team name
        df["team"] = df["team_id"].map(season_team_id_map)

        # Clean up name: "First_Last_ID" -> "First Last"
        df["name"] = df["name"].str.rsplit("_", n=1).str[0].str.replace("_", " ")

    # --- Normalize GW column (some files use 'round', some 'GW') ---
    if "GW" not in df.columns and "round" in df.columns:
        df["GW"] = df["round"]

    # --- Map opponent_team from ID to name ---
    if df["opponent_team"].dtype != object:
        df["opponent_team"] = df["opponent_team"].astype(int).map(season_team_id_map)

    # --- Extract desired columns, filling missing with NaN/0 ---
    for col in DESIRED_COLS:
        if col not in df.columns:
            missing_cols.append(f"{season}: {col}")
            df[col] = np.nan

    result = df[DESIRED_COLS].copy()

    # --- Add season metadata ---
    result["season"] = season
    result["season_year"] = SEASON_YEAR[season]

    return result, raw_row_count, missing_cols


# ---------------------------------------------------------------------------
# Step 2: Cleaning
# ---------------------------------------------------------------------------

def clean_data(df):
    # Remove rows where player played fewer than 30 minutes (no meaningful signal)
    df = df[df["minutes"] >= 30].copy()

    # Fix price: value stored as integer (55 = £5.5m)
    df["value"] = pd.to_numeric(df["value"], errors="coerce") / 10.0

    # Standardize position labels
    df["position"] = df["position"].map(lambda x: POSITION_MAP.get(x, x) if pd.notna(x) else x)

    # Deduplicate on name + team + GW + season
    df = df.drop_duplicates(subset=["name", "team", "GW", "season"])

    # Ensure numeric columns are numeric
    numeric_cols = [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
        "yellow_cards", "red_cards", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "transfers_in", "transfers_out", "selected", "value",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Step 3: Rolling form features (within season only)
# ---------------------------------------------------------------------------

def weighted_rolling(series, weights, n):
    """Apply a weighted rolling average over last n values (most recent last)."""
    w = np.array(weights)
    result = np.zeros(len(series))
    arr = series.values
    for i in range(len(arr)):
        # Gather up to n previous values (not including current)
        start = max(0, i - n)
        prev = arr[start:i]  # values before current row
        if len(prev) == 0:
            result[i] = 0.0
        else:
            # Align weights: weight[0] is most recent (prev[-1])
            k = len(prev)
            w_slice = w[:k]
            # Reverse prev so prev[-1] matches w[0]
            vals = prev[::-1]
            result[i] = np.dot(vals[:k], w_slice[:k]) / w_slice[:k].sum()
    return result


def engineer_rolling_features(df):
    """Calculate rolling features within each season for each player."""
    df = df.sort_values(["season", "name", "team", "GW"]).reset_index(drop=True)

    out_parts = []

    for (season, player, team), grp in df.groupby(["season", "name", "team"], sort=False):
        grp = grp.sort_values("GW").reset_index(drop=True)
        pts = grp["total_points"].fillna(0).values
        mins = grp["minutes"].fillna(0).values
        goals = grp["goals_scored"].fillna(0).values
        ast = grp["assists"].fillna(0).values
        cs = grp["clean_sheets"].fillna(0).values
        saves = grp["saves"].fillna(0).values
        val = grp["value"]

        n = len(grp)

        # form_last3: weights [0.5, 0.3, 0.2] most-recent-first
        form3 = weighted_rolling(pd.Series(pts), [0.5, 0.3, 0.2], 3)

        # form_last5: weights [0.35, 0.25, 0.20, 0.12, 0.08] most-recent-first
        form5 = weighted_rolling(pd.Series(pts), [0.35, 0.25, 0.20, 0.12, 0.08], 5)

        # Cumulative counters (use cumsum up to and including current row,
        # but exclude current for ratios to avoid leakage — spec says "so far
        # that season" which includes the current GW per standard FPL approach)
        games_played = np.arange(1, n + 1)
        cum_pts = np.cumsum(pts)
        cum_goals = np.cumsum(goals)
        cum_ast = np.cumsum(ast)
        cum_cs = np.cumsum(cs)
        cum_saves = np.cumsum(saves)

        # minutes_reliability: % of games so far with >= 60 mins
        high_min = (mins >= 60).astype(int)
        cum_high_min = np.cumsum(high_min)
        minutes_reliability = cum_high_min / games_played

        # avg_points_per_game
        avg_ppg = cum_pts / games_played

        # goals/assists/cs/saves per game
        goals_pg = cum_goals / games_played
        ast_pg = cum_ast / games_played
        cs_rate = cum_cs / games_played
        saves_pg = cum_saves / games_played

        # Mask saves for non-GKs
        position = grp["position"].iloc[0]
        if position != "GK":
            saves_pg = np.zeros(n)

        # points_per_million
        val_arr = val.values
        ppm = np.where(val_arr > 0, pts / val_arr, np.nan)

        grp = grp.copy()
        grp["form_last3"] = np.round(form3, 4)
        grp["form_last5"] = np.round(form5, 4)
        grp["minutes_reliability_season"] = np.round(minutes_reliability, 4)
        grp["cumulative_points_season"] = cum_pts
        grp["avg_points_per_game_season"] = np.round(avg_ppg, 4)
        grp["goals_per_game_season"] = np.round(goals_pg, 4)
        grp["assists_per_game_season"] = np.round(ast_pg, 4)
        grp["clean_sheet_rate_season"] = np.round(cs_rate, 4)
        grp["saves_per_game_season"] = np.round(saves_pg, 4)
        grp["points_per_million"] = np.round(ppm, 4)

        out_parts.append(grp)

    return pd.concat(out_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4: Save
# ---------------------------------------------------------------------------

def save_outputs(df):
    master_path = os.path.join(OUT_DIR, "historical_gw_data.csv")
    df.to_csv(master_path, index=False)

    counts = {}
    for pos, label in [("GK", "historical_gk"), ("DEF", "historical_def"),
                       ("MID", "historical_mid"), ("FWD", "historical_fwd")]:
        sub = df[df["position"] == pos]
        path = os.path.join(OUT_DIR, f"{label}.csv")
        sub.to_csv(path, index=False)
        counts[pos] = len(sub)

    return counts


# ---------------------------------------------------------------------------
# Step 5: Validation report
# ---------------------------------------------------------------------------

def print_validation_report(raw_counts, df_clean, pos_file_counts,
                             team_name_map, unmapped_teams, missing_cols_list):
    print("\n=== STAGE 2 VALIDATION REPORT ===\n")

    print("ROWS LOADED PER SEASON (before cleaning):")
    total_raw = 0
    for s, c in raw_counts.items():
        print(f"  {s}: {c:,} rows")
        total_raw += c
    print(f"  TOTAL:   {total_raw:,} rows")

    print("\nROWS AFTER CLEANING (minutes >= 30, deduped):")
    total_clean = 0
    for s in SEASONS:
        c = len(df_clean[df_clean["season"] == s])
        print(f"  {s}: {c:,} rows")
        total_clean += c
    print(f"  TOTAL:   {total_clean:,} rows")

    print("\nROWS PER POSITION:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"  {pos}:  {len(df_clean[df_clean['position'] == pos]):,} rows")

    print("\nAVERAGE TOTAL_POINTS PER POSITION PER SEASON:")
    pivot = df_clean.groupby(["position", "season"])["total_points"].mean().unstack("season")
    pivot = pivot.reindex(["GK", "DEF", "MID", "FWD"])
    print(pivot.round(2).to_string())

    print("\nTOP 10 HIGHEST SCORING PLAYERS (total points across all 5 seasons):")
    top = (df_clean.groupby("name")["total_points"].sum()
           .sort_values(ascending=False).head(10))
    for i, (name, pts) in enumerate(top.items(), 1):
        print(f"  {i:2}. {name}: {int(pts)} pts")

    print("\nTEAM NAME MAPPING:")
    mapped = {k: v for k, v in team_name_map.items()}
    print(f"  Successfully mapped: {len(mapped)} teams")
    if unmapped_teams:
        print(f"  Could not map (kept as-is): {unmapped_teams}")
    else:
        print("  Could not map: none — all resolved")

    print("\nCOLUMNS MISSING PER SEASON:")
    if missing_cols_list:
        for m in missing_cols_list:
            print(f"  {m}")
    else:
        print("  None — all columns present in all seasons")

    print("\nNULL VALUES AFTER CLEANING:")
    null_counts = df_clean.isnull().sum()
    null_pct = (null_counts / len(df_clean) * 100).round(1)
    flagged = False
    for col, cnt in null_counts.items():
        if cnt > 0:
            flag = " <-- FLAG (>5%)" if null_pct[col] > 5 else ""
            print(f"  {col}: {cnt:,} nulls ({null_pct[col]}%){flag}")
            flagged = True
    if not flagged:
        print("  None")

    print("\nSAMPLE ROWS (5 per position file):")
    sample_cols = ["name", "team", "season", "GW", "position",
                   "total_points", "minutes", "value", "form_last3", "form_last5"]
    for pos, label in [("GK", "historical_gk"), ("DEF", "historical_def"),
                       ("MID", "historical_mid"), ("FWD", "historical_fwd")]:
        sub = df_clean[df_clean["position"] == pos]
        print(f"\n  [{label}.csv]")
        sample = sub.sample(min(5, len(sub)), random_state=42)[sample_cols]
        print(sample.to_string(index=False))

    print("\nFILES SAVED:")
    print(f"  historical_gw_data.csv -- {total_clean:,} rows")
    for pos, label in [("GK", "historical_gk"), ("DEF", "historical_def"),
                       ("MID", "historical_mid"), ("FWD", "historical_fwd")]:
        print(f"  {label}.csv      -- {pos_file_counts[pos]:,} rows")

    print("\n=== END REPORT ===\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Stage 2: Loading vaastav historical data...")
    print(f"  Loading: {', '.join(SEASONS)}")
    print(f"  Blocked: {', '.join(sorted(BLOCKED_SEASONS))} (live season -- never train on this)")

    # Safety check — abort if someone tries to add a blocked season to SEASONS
    available = [d for d in os.listdir(VAASTAV_DIR) if os.path.isdir(os.path.join(VAASTAV_DIR, d))]
    for blocked in BLOCKED_SEASONS:
        if blocked in SEASONS:
            raise RuntimeError(f"FATAL: {blocked} is in SEASONS list — remove it immediately!")
        if blocked in available:
            print(f"  [OK] {blocked} folder exists in repo but is blocked — will not be loaded")

    # Build team name mapping
    team_name_map, unmapped_teams = build_team_name_mapping()
    if unmapped_teams:
        print(f"  WARNING: Could not map these team names: {unmapped_teams}")

    # Load all seasons
    all_dfs = []
    raw_counts = {}
    all_missing_cols = []

    for season in SEASONS:
        print(f"  Loading {season}...")
        season_team_id_map = build_season_team_id_map(season)
        df_s, raw_count, missing = load_season(season, team_name_map, season_team_id_map)
        raw_counts[season] = raw_count
        all_missing_cols.extend(missing)
        all_dfs.append(df_s)

    df_raw = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total rows loaded: {len(df_raw):,}")

    # Clean
    print("  Cleaning data...")
    df_clean = clean_data(df_raw)
    print(f"  Rows after cleaning: {len(df_clean):,}")

    # Apply team name mapping to team and opponent_team columns
    df_clean["team"] = df_clean["team"].map(lambda x: team_name_map.get(x, x))
    df_clean["opponent_team"] = df_clean["opponent_team"].map(lambda x: team_name_map.get(x, x))

    # Engineer rolling features
    print("  Engineering rolling features...")
    df_final = engineer_rolling_features(df_clean)

    # Save
    print("  Saving output files...")
    pos_file_counts = save_outputs(df_final)

    # Report
    print_validation_report(raw_counts, df_final, pos_file_counts,
                             team_name_map, unmapped_teams, all_missing_cols)


if __name__ == "__main__":
    main()
