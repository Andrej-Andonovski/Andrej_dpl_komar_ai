"""
FPL AI - Stage 3: Team Form Dataset
Builds team-level form + xG features from vaastav historical data + Understat.

Parts:
  A - Team form aggregated from vaastav player-GW data
  B - xG / xGA scraped from Understat (cached to raw_matches.json)
  C - Merge into final team_form.csv
  D - Validation report

Run: python pipeline/team_form_stage3.py
"""

import os
import sys
import io
import json
import math
import time
import warnings
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrameGroupBy.apply")

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAASTAV_DIR  = os.path.join(BASE_DIR, "data", "raw", "vaastav")
FPL_API_DIR  = os.path.join(BASE_DIR, "data", "raw", "fpl_api")
UNDERSTAT_DIR = os.path.join(BASE_DIR, "data", "raw", "understat")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(UNDERSTAT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
BLOCKED  = {"2025-26"}

# Understat uses the start year of the season as the identifier
UNDERSTAT_YEARS = {
    "2019-20": "2019",
    "2020-21": "2020",
    "2021-22": "2021",
    "2022-23": "2022",
    "2023-24": "2023",
    "2024-25": "2024",
}

# Understat team name -> canonical vaastav/FPL name
UNDERSTAT_TEAM_MAP = {
    "Manchester City":       "Man City",
    "Manchester United":     "Man Utd",
    "Nottingham Forest":     "Nott'm Forest",
    "Tottenham":             "Spurs",
    "West Bromwich Albion":  "West Brom",
    "Sheffield United":      "Sheffield Utd",
    "Leeds United":          "Leeds",
    "Newcastle United":      "Newcastle",
    "Brighton":              "Brighton",
    "Arsenal":               "Arsenal",
    "Chelsea":               "Chelsea",
    "Liverpool":             "Liverpool",
    "Aston Villa":           "Aston Villa",
    "Everton":               "Everton",
    "West Ham":              "West Ham",
    "Leicester":             "Leicester",
    "Wolverhampton Wanderers": "Wolves",
    "Crystal Palace":        "Crystal Palace",
    "Burnley":               "Burnley",
    "Southampton":           "Southampton",
    "Watford":               "Watford",
    "Norwich":               "Norwich",
    "Fulham":                "Fulham",
    "Brentford":             "Brentford",
    "Bournemouth":           "Bournemouth",
    "Luton":                 "Luton",
    "Ipswich":               "Ipswich",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FPL-AI-Research/1.0)"}


# ===========================================================================
# PART A: Team form from vaastav player-GW data
# ===========================================================================

def build_team_form_vaastav():
    """
    Aggregate vaastav historical_gw_data.csv to team-level per GW per season.
    Compute rolling form features without crossing season boundaries.
    """
    print("\nPart A: Building team form from vaastav data...")

    gw_path = os.path.join(VAASTAV_DIR, "historical_gw_data.csv")
    df = pd.read_csv(gw_path, low_memory=False)
    print(f"  Loaded {len(df):,} player-GW rows")

    # Normalise types
    df["was_home"] = (
        df["was_home"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    df["goals_scored"]   = pd.to_numeric(df["goals_scored"],   errors="coerce").fillna(0)
    df["goals_conceded"] = pd.to_numeric(df["goals_conceded"], errors="coerce").fillna(0)
    df["GW"]             = pd.to_numeric(df["GW"],             errors="coerce").astype(int)

    # Aggregate to one row per team + GW + season
    # goals_scored_for : sum of all players' goals
    # goals_conceded   : max (the team total is the same for every player, so max == team total)
    team_gw = (
        df.groupby(["team", "season", "GW"])
        .agg(
            goals_scored_for=("goals_scored",   "sum"),
            goals_conceded   =("goals_conceded",  "max"),
            was_home         =("was_home",         "first"),
            opponent         =("opponent_team",    "first"),
        )
        .reset_index()
    )

    team_gw["clean_sheet"]   = (team_gw["goals_conceded"] == 0).astype(int)
    team_gw["win"]           = (team_gw["goals_scored_for"] > team_gw["goals_conceded"]).astype(int)
    team_gw["draw"]          = (team_gw["goals_scored_for"] == team_gw["goals_conceded"]).astype(int)
    team_gw["points_earned"] = team_gw["win"] * 3 + team_gw["draw"]

    team_gw = team_gw.sort_values(["team", "season", "GW"]).reset_index(drop=True)

    # ── Rolling features, reset at each season boundary ──────────────────────
    print("  Calculating rolling form features (no cross-season bleed)...")

    def rolling_team_features(group):
        g = group.sort_values("GW").copy()
        n = len(g)

        # Shift-1 rolling (current game excluded from its own feature)
        s_gs  = g["goals_scored_for"].shift(1)
        s_gc  = g["goals_conceded"].shift(1)
        s_cs  = g["clean_sheet"].shift(1)
        s_win = g["win"].shift(1)
        s_pts = g["points_earned"].shift(1)

        g["goals_scored_last5"]     = s_gs.rolling(5, min_periods=1).mean()
        g["goals_conceded_last5"]   = s_gc.rolling(5, min_periods=1).mean()
        g["clean_sheet_rate_last5"] = s_cs.rolling(5, min_periods=1).mean()
        g["wins_last5"]             = s_win.rolling(5, min_periods=1).sum()
        g["form_points_last5"]      = s_pts.rolling(5, min_periods=1).sum()
        g["goals_scored_last3"]     = s_gs.rolling(3, min_periods=1).mean()
        g["goals_conceded_last3"]   = s_gc.rolling(3, min_periods=1).mean()

        # Home / away season averages (cumulative, excluding current game)
        is_home    = g["was_home"].values.astype(bool)
        goals_for  = g["goals_scored_for"].values.astype(float)
        goals_vs   = g["goals_conceded"].values.astype(float)

        home_gs_season = np.full(n, np.nan)
        away_gs_season = np.full(n, np.nan)
        home_gc_season = np.full(n, np.nan)
        away_gc_season = np.full(n, np.nan)

        hgs_sum = hgc_sum = ags_sum = agc_sum = 0.0
        hcnt = acnt = 0

        for i in range(n):
            if hcnt > 0:
                home_gs_season[i] = hgs_sum / hcnt
                home_gc_season[i] = hgc_sum / hcnt
            if acnt > 0:
                away_gs_season[i] = ags_sum / acnt
                away_gc_season[i] = agc_sum / acnt
            # Accumulate current game
            if is_home[i]:
                hgs_sum += goals_for[i]; hgc_sum += goals_vs[i]; hcnt += 1
            else:
                ags_sum += goals_for[i]; agc_sum += goals_vs[i]; acnt += 1

        g["home_goals_scored_season"]   = home_gs_season
        g["away_goals_scored_season"]   = away_gs_season
        g["home_goals_conceded_season"] = home_gc_season
        g["away_goals_conceded_season"] = away_gc_season

        return g

    team_gw = (
        team_gw.groupby(["team", "season"], group_keys=False)
        .apply(rolling_team_features)
    )

    # ── Normalise attacking / defensive strength across all teams per GW ──────
    print("  Normalising attacking / defensive strength per GW...")

    def normalise_gw(group):
        lo_att = group["goals_scored_last5"].min()
        hi_att = group["goals_scored_last5"].max()
        rng_att = hi_att - lo_att
        group["attacking_strength"] = (
            (group["goals_scored_last5"] - lo_att) / rng_att if rng_att > 0 else 0.5
        )

        lo_def = group["goals_conceded_last5"].min()
        hi_def = group["goals_conceded_last5"].max()
        rng_def = hi_def - lo_def
        group["defensive_strength"] = (
            1.0 - (group["goals_conceded_last5"] - lo_def) / rng_def if rng_def > 0 else 0.5
        )
        return group

    team_gw = (
        team_gw.groupby(["season", "GW"], group_keys=False)
        .apply(normalise_gw)
    )

    # Fill first-GW NaN values (no prior data)
    rolling_cols = [
        "goals_scored_last5", "goals_conceded_last5", "clean_sheet_rate_last5",
        "wins_last5", "form_points_last5", "goals_scored_last3", "goals_conceded_last3",
        "home_goals_scored_season", "away_goals_scored_season",
        "home_goals_conceded_season", "away_goals_conceded_season",
        "attacking_strength", "defensive_strength",
    ]
    team_gw[rolling_cols] = team_gw[rolling_cols].fillna(0)

    out_path = os.path.join(VAASTAV_DIR, "team_form_vaastav.csv")
    team_gw.to_csv(out_path, index=False)
    print(f"  Saved: team_form_vaastav.csv ({len(team_gw):,} rows)")
    return team_gw


# ===========================================================================
# PART B: Scrape xG / xGA from Understat
# ===========================================================================

def fetch_understat_season(year, retries=2):
    """
    Fetch EPL match data for one season from Understat's JSON API.
    Returns list of match dicts (the 'dates' array), or None on failure.
    Understat API: GET /getLeagueData/EPL/{year}
    """
    url = f"https://understat.com/getLeagueData/EPL/{year}"
    api_headers = {**HEADERS, "X-Requested-With": "XMLHttpRequest",
                   "Referer": f"https://understat.com/league/EPL/{year}"}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=api_headers, timeout=40)
            r.raise_for_status()
            data = r.json()
            if "dates" in data:
                return data["dates"]
            print(f"  [WARN] 'dates' key missing in Understat response for year {year}")
            return None
        except Exception as e:
            if attempt < retries:
                print(f"  [WARN] Understat fetch failed for {year} (attempt {attempt+1}): {e}")
                time.sleep(5)
            else:
                print(f"  [ERROR] Understat fetch failed for {year} after {retries+1} attempts: {e}")
                return None


def build_understat_xg():
    """
    Fetch Understat match xG data for all seasons, cache to JSON, and build
    per-team per-match rolling xG features.
    Returns: (team_matches_df, matches_per_season_dict, fetch_failures_list)
    """
    print("\nPart B: Fetching xG data from Understat...")

    cache_path = os.path.join(UNDERSTAT_DIR, "raw_matches.json")

    # Load existing cache
    if os.path.exists(cache_path):
        print(f"  Loading cached Understat data: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            all_raw = json.load(f)
    else:
        all_raw = {}

    fetch_failures = []

    for season, year in UNDERSTAT_YEARS.items():
        if year in all_raw:
            print(f"  {season}: cached ({len(all_raw[year])} matches in response)")
            continue
        print(f"  Fetching {season} (year={year}) from Understat...")
        data = fetch_understat_season(year)
        if data is None:
            fetch_failures.append(season)
            print(f"  [WARN] {season}: fetch failed - will fallback to vaastav goals")
        else:
            all_raw[year] = data
            print(f"  {season}: fetched {len(data)} entries")
        time.sleep(2)

    # Always resave cache (even if nothing new was fetched)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_raw, f, ensure_ascii=False, indent=2)
    print(f"  Cache saved: raw_matches.json")

    # Parse into flat records
    records = []
    matches_per_season = {}

    for season, year in UNDERSTAT_YEARS.items():
        if year not in all_raw:
            matches_per_season[season] = 0
            continue

        season_count = 0
        for m in all_raw[year]:
            # isResult may be bool True or string "1"
            is_result = m.get("isResult")
            if not is_result or is_result == "0":
                continue
            try:
                h_team  = UNDERSTAT_TEAM_MAP.get(m["h"]["title"], m["h"]["title"])
                a_team  = UNDERSTAT_TEAM_MAP.get(m["a"]["title"], m["a"]["title"])
                h_xg    = float(m["xG"]["h"])
                a_xg    = float(m["xG"]["a"])
                h_goals = int(m["goals"]["h"])
                a_goals = int(m["goals"]["a"])
                # datetime: "2019-08-11 17:00:00" or "2019-08-11T17:00:00"
                date_str = m["datetime"][:10]
                records.append({
                    "season":     season,
                    "date":       date_str,
                    "home_team":  h_team,
                    "away_team":  a_team,
                    "home_xG":    h_xg,
                    "away_xG":    a_xg,
                    "home_goals": h_goals,
                    "away_goals": a_goals,
                })
                season_count += 1
            except (KeyError, ValueError, TypeError):
                pass
        matches_per_season[season] = season_count

    if not records:
        print("  [ERROR] No Understat records parsed — will use vaastav-only fallback")
        return None, matches_per_season, fetch_failures

    matches_df = pd.DataFrame(records)
    matches_df["date"] = pd.to_datetime(matches_df["date"])

    # Expand to one row per team per match
    home_df = matches_df.rename(columns={
        "home_team": "team",   "away_team": "opponent",
        "home_xG":   "xG",    "away_xG":   "xGA",
        "home_goals": "goals_scored_for", "away_goals": "goals_conceded",
    })[["season", "date", "team", "opponent", "xG", "xGA",
        "goals_scored_for", "goals_conceded"]].copy()
    home_df["was_home"] = True

    away_df = matches_df.rename(columns={
        "away_team": "team",   "home_team": "opponent",
        "away_xG":   "xG",    "home_xG":   "xGA",
        "away_goals": "goals_scored_for", "home_goals": "goals_conceded",
    })[["season", "date", "team", "opponent", "xG", "xGA",
        "goals_scored_for", "goals_conceded"]].copy()
    away_df["was_home"] = False

    team_matches = pd.concat([home_df, away_df], ignore_index=True)
    team_matches = team_matches.sort_values(["team", "season", "date"]).reset_index(drop=True)

    # ── Rolling xG features (reset at each season boundary) ──────────────────
    print("  Calculating rolling xG features (no cross-season bleed)...")

    def rolling_xg_features(group):
        g = group.sort_values("date").copy()
        n = len(g)

        s_xg  = g["xG"].shift(1)
        s_xga = g["xGA"].shift(1)

        g["xG_last5"]  = s_xg.rolling(5, min_periods=1).mean()
        g["xGA_last5"] = s_xga.rolling(5, min_periods=1).mean()
        g["xG_last3"]  = s_xg.rolling(3, min_periods=1).mean()
        g["xGA_last3"] = s_xga.rolling(3, min_periods=1).mean()

        g["xG_season_avg"]  = s_xg.expanding().mean()
        g["xGA_season_avg"] = s_xga.expanding().mean()

        # Home / away season averages (cumulative, excluding current game)
        is_home = g["was_home"].values.astype(bool)
        xg_vals  = g["xG"].values.astype(float)
        xga_vals = g["xGA"].values.astype(float)

        hxg_season  = np.full(n, np.nan)
        axg_season  = np.full(n, np.nan)
        hxga_season = np.full(n, np.nan)
        axga_season = np.full(n, np.nan)

        hxg_sum = hxga_sum = axg_sum = axga_sum = 0.0
        hcnt = acnt = 0

        for i in range(n):
            if hcnt > 0:
                hxg_season[i]  = hxg_sum  / hcnt
                hxga_season[i] = hxga_sum / hcnt
            if acnt > 0:
                axg_season[i]  = axg_sum  / acnt
                axga_season[i] = axga_sum / acnt
            if is_home[i]:
                hxg_sum  += xg_vals[i];  hxga_sum += xga_vals[i]; hcnt += 1
            else:
                axg_sum  += xg_vals[i];  axga_sum += xga_vals[i]; acnt += 1

        g["home_xG_season"]  = hxg_season
        g["away_xG_season"]  = axg_season
        g["home_xGA_season"] = hxga_season
        g["away_xGA_season"] = axga_season

        # Poisson clean sheet probability: P(X=0 | lambda=xGA_last5) = e^(-lambda)
        g["clean_sheet_probability"] = g["xGA_last5"].apply(
            lambda x: math.exp(-x) if pd.notna(x) and x > 0 else (1.0 if x == 0 else np.nan)
        )

        return g

    team_matches = (
        team_matches.groupby(["team", "season"], group_keys=False)
        .apply(rolling_xg_features)
    )

    # Fill first-game NaN values
    xg_rolling_cols = [
        "xG_last5", "xGA_last5", "xG_last3", "xGA_last3",
        "xG_season_avg", "xGA_season_avg",
        "home_xG_season", "away_xG_season",
        "home_xGA_season", "away_xGA_season",
        "clean_sheet_probability",
    ]
    team_matches[xg_rolling_cols] = team_matches[xg_rolling_cols].fillna(0)

    out_path = os.path.join(UNDERSTAT_DIR, "team_xg_data.csv")
    team_matches.to_csv(out_path, index=False)
    print(f"  Saved: team_xg_data.csv ({len(team_matches):,} rows)")

    return team_matches, matches_per_season, fetch_failures


# ===========================================================================
# PART C: Merge into final team_form.csv
# ===========================================================================

def merge_team_form(team_form_vaastav, team_xg):
    """
    Merge vaastav form with Understat xG data on (team, season, opponent, was_home).
    This maps each xG rolling snapshot to the correct GW.
    """
    print("\nPart C: Merging vaastav form + Understat xG data...")

    XG_COLS = [
        "xG_last5", "xGA_last5", "xG_last3", "xGA_last3",
        "xG_season_avg", "xGA_season_avg",
        "home_xG_season", "away_xG_season",
        "home_xGA_season", "away_xGA_season",
        "clean_sheet_probability",
    ]

    if team_xg is None:
        print("  [WARN] No xG data — saving vaastav-only form as team_form.csv")
        # Add placeholder xG columns filled with 0
        for col in XG_COLS:
            team_form_vaastav[col] = 0.0
        # Fallback Poisson probability from goals_conceded_last5
        team_form_vaastav["clean_sheet_probability"] = (
            team_form_vaastav["goals_conceded_last5"]
            .apply(lambda x: math.exp(-x) if x > 0 else 1.0)
        )
        out_path = os.path.join(PROCESSED_DIR, "team_form.csv")
        team_form_vaastav.to_csv(out_path, index=False)
        print(f"  Saved: team_form.csv ({len(team_form_vaastav):,} rows) [vaastav-only]")
        return team_form_vaastav

    # Prepare join: keep xG rolling features + join keys
    xg_join = team_xg[["season", "team", "opponent", "was_home"] + XG_COLS].copy()

    # Deduplicate join side (guard against any parsing duplicates)
    before = len(xg_join)
    xg_join = xg_join.drop_duplicates(subset=["season", "team", "opponent", "was_home"])
    after = len(xg_join)
    if before != after:
        print(f"  [INFO] Removed {before - after} duplicate rows from xG join table")

    # Ensure was_home is bool in both tables
    team_form_vaastav["was_home"] = team_form_vaastav["was_home"].astype(bool)
    xg_join["was_home"] = xg_join["was_home"].astype(bool)

    merged = team_form_vaastav.merge(
        xg_join,
        on=["season", "team", "opponent", "was_home"],
        how="left",
    )

    # Fill unmatched xG with 0
    fill_cols = [c for c in XG_COLS if c != "clean_sheet_probability"]
    merged[fill_cols] = merged[fill_cols].fillna(0)

    # Fallback clean_sheet_probability from goals_conceded if xG join missed
    merged["clean_sheet_probability"] = merged["clean_sheet_probability"].fillna(
        merged["goals_conceded_last5"].apply(lambda x: math.exp(-max(x, 0)))
    )

    out_path = os.path.join(PROCESSED_DIR, "team_form.csv")
    merged.to_csv(out_path, index=False)
    print(f"  Saved: team_form.csv ({len(merged):,} rows)")
    return merged


# ===========================================================================
# PART D: Validation report
# ===========================================================================

def print_validation_report(team_form_vaastav, team_xg, merged, matches_per_season, fetch_failures):
    print("\n" + "=" * 60)
    print("=== STAGE 3 VALIDATION REPORT ===")
    print("=" * 60)

    # ── Vaastav form ──────────────────────────────────────────────────────────
    print("\nTEAM FORM FROM VAASTAV:")
    print(f"  Rows: {len(team_form_vaastav):,}  (expected ~4,200)")
    seasons_present = sorted(team_form_vaastav["season"].unique())
    print(f"  Seasons covered: {seasons_present}")

    gw_counts = team_form_vaastav.groupby(["team", "season"])["GW"].count()
    sparse = gw_counts[gw_counts < 30]
    if len(sparse) > 0:
        print(f"  [WARN] Teams with < 30 GWs in a season:")
        for (team, season), cnt in sparse.items():
            print(f"    {team} {season}: {cnt} GWs")
    else:
        print("  All teams have >= 30 GWs per season -- OK")

    # ── Understat ─────────────────────────────────────────────────────────────
    print("\nXG DATA FROM UNDERSTAT:")
    per_season_str = ", ".join(
        f"{s}: {matches_per_season.get(s, 0)}" + ("" if matches_per_season.get(s, 0) >= 370 else " [WARN]")
        for s in SEASONS
    )
    print(f"  Matches fetched per season: {per_season_str}")
    with_data = [s for s, c in matches_per_season.items() if c > 0]
    print(f"  Seasons covered: {with_data}")
    if fetch_failures:
        print(f"  Any fetch failures: YES — {fetch_failures}")
        print("    Fallback: Poisson probability derived from vaastav goals_conceded")
    else:
        print("  Any fetch failures: None")

    # ── Merged ────────────────────────────────────────────────────────────────
    print("\nMERGED TEAM FORM:")
    print(f"  Total rows: {len(merged):,}")
    null_counts = merged.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        print("  Null values per column:")
        for col, cnt in null_counts.items():
            print(f"    {col}: {cnt}")
    else:
        print("  Null values: None")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\nSANITY CHECKS:")
    d2425 = merged[merged["season"] == "2024-25"]

    if "xG_last5" in merged.columns and d2425["xG_last5"].sum() > 0:
        top_xg = (
            d2425.groupby("team")["xG_last5"].mean()
            .sort_values(ascending=False).head(5)
        )
        print("  Top 5 teams by avg xG last 5 GWs in 2024-25 (should be Man City, Liverpool, Arsenal etc):")
        for i, (team, val) in enumerate(top_xg.items(), 1):
            print(f"    {i}. {team}: {val:.2f}")

        top_def = (
            d2425.groupby("team")["xGA_last5"].mean()
            .sort_values(ascending=True).head(5)
        )
        print("  Top 5 teams by avg xGA last 5 GWs in 2024-25 (lowest = best defense, should be top clubs):")
        for i, (team, val) in enumerate(top_def.items(), 1):
            print(f"    {i}. {team}: {val:.2f}")
    else:
        print("  [INFO] xG data not available -- using vaastav goals for sanity check")

    cs_by_team = d2425.groupby("team")["clean_sheet"].mean().sort_values(ascending=False)
    best_cs_team = cs_by_team.index[0]
    best_cs_val  = cs_by_team.iloc[0]
    print(f"  Team with highest clean sheet rate 2024-25: {best_cs_team} ({best_cs_val:.1%})"
          " [should be a top 6 club]")

    # ── Sample rows ───────────────────────────────────────────────────────────
    print("\nSAMPLE ROWS (5 rows from team_form.csv):")
    show_cols = [
        "team", "season", "GW", "was_home",
        "goals_scored_for", "goals_conceded", "clean_sheet",
        "goals_scored_last5", "goals_conceded_last5",
        "attacking_strength", "defensive_strength",
        "clean_sheet_probability",
    ]
    show_cols = [c for c in show_cols if c in merged.columns]
    sample = merged[show_cols].sample(5, random_state=42)
    print(sample.to_string(index=False))

    # ── Files saved ───────────────────────────────────────────────────────────
    print("\nFILES SAVED:")
    vf_rows = len(team_form_vaastav) if team_form_vaastav is not None else 0
    xf_rows = len(team_xg)           if team_xg           is not None else 0
    mf_rows = len(merged)            if merged             is not None else 0
    print(f"  team_form_vaastav.csv  - {vf_rows:,} rows")
    print(f"  team_xg_data.csv       - {xf_rows:,} rows")
    print(f"  team_form.csv          - {mf_rows:,} rows (final merged)")

    print("\n=== END REPORT ===")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Stage 3: Building team form dataset...")
    print(f"  Seasons: {SEASONS}")
    print(f"  Blocked: {sorted(BLOCKED)} (live season -- never use)")

    # Part A
    team_form_vaastav = build_team_form_vaastav()

    # Part B
    team_xg, matches_per_season, fetch_failures = build_understat_xg()

    # Part C
    merged = merge_team_form(team_form_vaastav, team_xg)

    # Part D
    print_validation_report(
        team_form_vaastav, team_xg, merged, matches_per_season, fetch_failures
    )
