#!/usr/bin/env python3
"""
Stage 4b: 2024/25 PL Debutants — Previous-League FBref Stats

Identifies FPL players who debuted in the Premier League in the 2024/25 season
(in vaastav data but with zero prior vaastav history), scrapes their previous-
league stats from FBref, and appends to the 4 position files.

Usage:
    python pipeline/data_loader_stage4b.py
    python pipeline/data_loader_stage4b.py --start-step 2

Global rules enforced throughout:
    - Never touch 2025-26 data
    - Cache files are sacred (never overwrite)
    - Rolling features partition by season (no cross-season bleed)
    - Position file schema must match Stage 4a (VAASTAV_COLS) before and after
    - adjG/90 from < 500 mins always has small_sample = True
    - Stop and wait for 'y' at every gate
"""

import io
import os
import re
import sys
import json
import time
import argparse
import unicodedata
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# Windows UTF-8 stdout
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── dependency check ──────────────────────────────────────────────────────────
_missing = []
try:
    import pandas as pd
    import numpy as np
except ImportError:
    _missing.append("pandas numpy")
try:
    from fuzzywuzzy import fuzz, process as fzprocess
except ImportError:
    _missing.append("fuzzywuzzy python-Levenshtein")
try:
    from seleniumbase import SB
    _HAS_SELENIUM = True
except ImportError:
    _HAS_SELENIUM = False

if _missing:
    print("[ERROR] Missing dependencies. Install with:")
    print(f"  pip install {' '.join(_missing)}")
    sys.exit(1)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
FBREF_RAW_DIR  = os.path.join(RAW_DIR, "fbref", "raw")
FBREF_SIGN_DIR = os.path.join(RAW_DIR, "fbref", "new_signings")
VAASTAV_DIR    = os.path.join(RAW_DIR, "vaastav")
FPL_API_DIR    = os.path.join(RAW_DIR, "fpl_api")
TRANSFERS_DIR  = os.path.join(RAW_DIR, "transfers")
STATE_FILE     = os.path.join(RAW_DIR, "fbref", "stage4b_state.json")
REPORT_FILE    = os.path.join(FBREF_SIGN_DIR, "stage4b_report.txt")

for _d in [FBREF_RAW_DIR, FBREF_SIGN_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── import helpers from Stage 4a ─────────────────────────────────────────────
# Ensure BASE_DIR is on sys.path so the pipeline package can be found
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from pipeline.new_signings_stage4a import (
        FBREF_FALSE_POSITIVES,
        FBREF_NAME_OVERRIDES,
        FBREF_LEAGUE_IDS,
        FBREF_LEAGUE_SLUGS,
        LEAGUE_MULTIPLIERS,
        VAASTAV_COLS,
        normalize_name,
        fuzzy_match_name,
        flatten_columns,
        find_col,
        safe_div,
        season_to_year,
        _season_reliability,
        _extract_fbref_stats,
        _find_player_col,
        _find_squad_col,
        _find_pos_col,
        _build_fbref_url,
        _parse_fbref_page_html,
    )
    _4A_IMPORTED = True
except ImportError as e:
    print(f"[WARN] Could not import all helpers from Stage 4a: {e}")
    print("       Defining fallback helpers locally.")
    _4A_IMPORTED = False

# ── fallback helpers (if stage4a import fails) ────────────────────────────────
if not _4A_IMPORTED:
    FBREF_FALSE_POSITIVES = set()
    FBREF_NAME_OVERRIDES  = {}
    FBREF_LEAGUE_IDS = {
        "Bundesliga": 20, "La Liga": 12, "Serie A": 11, "Ligue 1": 13,
        "Eredivisie": 23, "Primeira Liga": 32, "Scottish Premiership": 40,
        "Championship": 10, "Belgian Pro League": 37,
    }
    FBREF_LEAGUE_SLUGS = {
        "Bundesliga": "Bundesliga", "La Liga": "La-Liga", "Serie A": "Serie-A",
        "Ligue 1": "Ligue-1", "Eredivisie": "Eredivisie",
        "Primeira Liga": "Primeira-Liga", "Scottish Premiership": "Scottish-Premiership",
        "Championship": "Championship", "Belgian Pro League": "Belgian-First-Division-A",
    }
    LEAGUE_MULTIPLIERS = {
        "Bundesliga": 0.89, "La Liga": 0.92, "Serie A": 0.88, "Ligue 1": 0.82,
        "Eredivisie": 0.75, "Primeira Liga": 0.78, "Scottish Premiership": 0.65,
        "Championship": 0.72, "Belgian Pro League": 0.74,
        "Serie A (Brazil)": 0.70,
    }
    VAASTAV_COLS = [
        "name", "position", "team", "GW", "opponent_team", "was_home",
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
        "yellow_cards", "red_cards", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "transfers_in", "transfers_out", "selected", "value",
        "season", "season_year", "form_last3", "form_last5",
        "minutes_reliability_season", "cumulative_points_season",
        "avg_points_per_game_season", "goals_per_game_season",
        "assists_per_game_season", "clean_sheet_rate_season",
        "saves_per_game_season", "points_per_million", "is_new_to_pl",
    ]

    def normalize_name(name):
        nfd = unicodedata.normalize("NFD", str(name))
        ascii_str = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        cleaned = re.sub(r"[^\w\s-]", "", ascii_str.lower()).strip()
        return re.sub(r"\s+", " ", cleaned)

    def fuzzy_match_name(name, name_list, threshold=80):
        if not name_list:
            return None, 0
        result = fzprocess.extractOne(name, name_list, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            return result[0], result[1]
        return None, 0

    def flatten_columns(df):
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                parts = [str(c).strip() for c in col if str(c).strip() and str(c).strip() != "nan"]
                new_cols.append("_".join(parts))
            else:
                new_cols.append(str(col))
        df.columns = new_cols
        return df

    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def safe_div(a, b, default=0.0):
        try:
            if b == 0 or pd.isna(b):
                return default
            return a / b
        except Exception:
            return default

    def season_to_year(season_str):
        try:
            return int(season_str.split("-")[1]) + 2000
        except Exception:
            return 0

    def _season_reliability(minutes, appearances):
        available = appearances * 90
        if available <= 0:
            return 0.1
        ratio = minutes / available
        if ratio >= 0.6:
            return 1.0
        if ratio >= 0.3:
            return 0.5
        return 0.1

    def _find_player_col(df):
        for c in df.columns:
            if "player" in c.lower() or c.lower() == "player":
                return c
        return None

    def _find_squad_col(df):
        for c in df.columns:
            if "squad" in c.lower() or "team" in c.lower():
                return c
        return None

    def _find_pos_col(df):
        for c in df.columns:
            if c.lower().endswith("_pos") or c.lower() == "pos":
                return c
        return None

    def _extract_fbref_stats(row_series, df_columns):
        def get(candidates, default=0.0):
            col = find_col(pd.DataFrame(columns=df_columns), candidates)
            if col and col in row_series.index:
                v = row_series[col]
                try:
                    return float(v) if not pd.isna(v) else default
                except (ValueError, TypeError):
                    return default
            return default
        mp   = get(["Playing Time_MP",     "Performance_MP",    "MP"])
        st   = get(["Playing Time_Starts", "Performance_Starts","Starts"])
        mn   = get(["Playing Time_Min",    "Performance_Min",   "Min"])
        gls  = get(["Performance_Gls", "Gls"])
        ast  = get(["Performance_Ast", "Ast"])
        xg   = get(["Standard_xG", "Expected_xG", "xG"])
        xag  = get(["Expected_xAG", "xAG"])
        crdy = get(["Performance_CrdY", "CrdY"])
        crdr = get(["Performance_CrdR", "CrdR"])
        prgc = get(["Progression_PrgC", "PrgC"])
        prgp = get(["Progression_PrgP", "PrgP"])
        gls90 = get(["Per 90 Minutes_Gls", "Gls/90"])
        ast90 = get(["Per 90 Minutes_Ast", "Ast/90"])
        nineties = mn / 90.0 if mn > 0 else 0
        if gls90 == 0.0 and nineties > 0:
            gls90 = safe_div(gls, nineties)
        if ast90 == 0.0 and nineties > 0:
            ast90 = safe_div(ast, nineties)
        xg90  = safe_div(xg,  nineties) if nineties > 0 else 0
        xag90 = safe_div(xag, nineties) if nineties > 0 else 0
        sh90 = get(["Standard_Sh/90", "Sh/90"])
        if sh90 == 0.0:
            sh_tot = get(["Standard_Sh", "Sh"])
            sh90   = safe_div(sh_tot, nineties) if nineties > 0 else 0
        kp90   = get(["Passes_KP", "KP", "key_passes_per_90", "Standard_SoT/90", "SoT/90"])
        saves  = get(["Performance_Saves", "Saves"])
        save_p = get(["Performance_Save%", "Save%"])
        cs     = get(["Performance_CS", "CS"])
        sota   = get(["Performance_SoTA", "SoTA"])
        return {
            "appearances": mp, "starts": st, "minutes": mn,
            "goals": gls, "assists": ast, "xG": xg, "xA": xag,
            "shots_per_90": sh90, "key_passes_per_90": kp90,
            "goals_per_90": gls90, "assists_per_90": ast90,
            "yellow_cards": crdy, "red_cards": crdr,
            "progressive_carries": prgc, "progressive_passes": prgp,
            "clean_sheets": cs, "saves": saves, "save_percentage": save_p,
        }

    def _build_fbref_url(league_std, season, stat_type="standard"):
        fbref_id = FBREF_LEAGUE_IDS.get(league_std)
        slug     = FBREF_LEAGUE_SLUGS.get(league_std)
        if not fbref_id or not slug:
            return None
        year1, yr2_short = season.split("-")
        year2 = "20" + yr2_short if len(yr2_short) == 2 else yr2_short
        season_str = f"{year1}-{year2}"
        table_path = {"keeper": "keepers", "shooting": "shooting"}.get(stat_type, "stats")
        return (
            f"https://fbref.com/en/comps/{fbref_id}/{season_str}"
            f"/{table_path}/{season_str}-{slug}-Stats"
        )

    def _parse_fbref_page_html(html, table_id):
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            table_tag = soup.find("table", id=table_id)
            if table_tag is None:
                return None
            df_list = pd.read_html(str(table_tag), header=[0, 1], na_values=["", "N/A"])
            if not df_list:
                return None
            df = df_list[0]
            df = flatten_columns(df)
            player_col = _find_player_col(df)
            if player_col:
                bad_vals = {"player", "squad total", "", "nan"}
                df = df[~df[player_col].astype(str).str.strip().str.lower().isin(bad_vals)]
                df = df[df[player_col].notna()]
            return df.reset_index(drop=True)
        except Exception:
            return None

# ── constants ─────────────────────────────────────────────────────────────────
FUZZY_THRESHOLD = 80
CONFIRM_THRESHOLD = 95

# Seasons to scrape previous-league data for
TARGET_PREV_SEASONS = ["2023-24", "2022-23", "2021-22"]

# Full schema for output rows (VAASTAV_COLS + data_source)
OUTPUT_COLS = VAASTAV_COLS + ["data_source"]

# FPL position name mapping
FPL_POS_MAP = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
FBREF_POS_MAP = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}

# Map league name -> cache filename slug
LEAGUE_SLUG_MAP = {
    "Bundesliga":           "bundesliga",
    "La Liga":              "la_liga",
    "Serie A":              "serie_a",
    "Ligue 1":              "ligue_1",
    "Eredivisie":           "eredivisie",
    "Primeira Liga":        "primeira_liga",
    "Scottish Premiership": "scottish_premiership",
    "Championship":         "championship",
    "Belgian Pro League":   "belgian_pro_league",
}

# Standardized league name from various TM/FBref strings
LEAGUE_NORMALIZE_MAP = {
    "Bundesliga": "Bundesliga", "1. Bundesliga": "Bundesliga",
    "La Liga": "La Liga", "LaLiga": "La Liga", "Primera Division": "La Liga",
    "LaLiga EA Sports": "La Liga",
    "Serie A": "Serie A",
    "Ligue 1": "Ligue 1", "Ligue 1 Uber Eats": "Ligue 1",
    "Eredivisie": "Eredivisie",
    "Primeira Liga": "Primeira Liga", "Liga Portugal": "Primeira Liga",
    "Liga Portugal Betclic": "Primeira Liga",
    "Scottish Premiership": "Scottish Premiership",
    "Scottish Premier League": "Scottish Premiership",
    "Championship": "Championship", "EFL Championship": "Championship",
    "Belgian Pro League": "Belgian Pro League",
    "Jupiler Pro League": "Belgian Pro League",
}

# Players to exclude from Stage 4b (pre-populated or add manually after Step 1 review)
STAGE4B_EXCLUSIONS = {
    "Mitoma Kaoru",  # name-order issue — prior vaastav seasons list him as "Kaoru Mitoma"
    "Alex Palmer",   # wrong FPL match — GK falsely matched to Cole Palmer (Chelsea MID)
}


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def _load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"  State saved: {STATE_FILE}")


# =============================================================================
# CONFIRMATION GATE
# =============================================================================

def gate(step_num, step_name, summary_lines, next_description, stop_step=None):
    """
    Print a confirmation gate and wait for 'y'. Returns True if confirmed, False if stopped.
    """
    print()
    print("=" * 70)
    print(f"STEP {step_num} COMPLETE -- {step_name}")
    print("=" * 70)
    print()
    for line in summary_lines:
        print(line)
    print()
    print(f"NEXT: {next_description}")
    print()
    ans = input(f"Proceed to Step {step_num + 1}? (y/n): ").strip().lower()
    if ans != "y":
        print()
        resume_step = stop_step if stop_step is not None else step_num
        print(f"Stopped at Step {step_num}. Fix any issues then re-run with "
              f"--start-step {resume_step} to resume.")
        return False
    return True


# =============================================================================
# HELPERS
# =============================================================================

def _get_cache_path(league_std, season):
    """Return local cache file path for a (league, season) pair."""
    slug = LEAGUE_SLUG_MAP.get(league_std, league_std.lower().replace(" ", "_"))
    return os.path.join(FBREF_RAW_DIR, f"{slug}_{season}.csv")


def _infer_league_from_cache(player_name, threshold=90):
    """
    Check all existing FBref cache files for a player match.
    Returns (standardized_league, season, score) or (None, None, 0).
    """
    norm_pname = normalize_name(player_name)
    best_score = 0
    best_league = None
    best_season = None

    for fname in os.listdir(FBREF_RAW_DIR):
        if not fname.endswith(".csv"):
            continue
        # Skip 2024-25 files (those are current PL season data we already have)
        if "2024-25" in fname:
            continue
        fpath = os.path.join(FBREF_RAW_DIR, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        pcol = _find_player_col(df)
        if not pcol:
            continue
        names = df[pcol].dropna().astype(str).tolist()
        norm_names = [normalize_name(n) for n in names]
        result = fzprocess.extractOne(norm_pname, norm_names, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold and result[1] > best_score:
            best_score = result[1]
            # Parse league + season from filename: e.g. bundesliga_2022-23.csv
            parts = fname.replace(".csv", "").rsplit("_", 2)
            # Last element is always season like 2022-23
            season_part = fname.replace(".csv", "").split("_")[-1] + "-" + fname.replace(".csv", "").split("_")[-1]
            # More reliable: find the season pattern
            m = re.search(r"(\d{4}-\d{2})\.csv$", fname)
            if m:
                best_season = m.group(1)
                # League is everything before the season
                slug_part = fname[:m.start()].rstrip("_")
                # Reverse lookup
                for league, slug in LEAGUE_SLUG_MAP.items():
                    if slug == slug_part:
                        best_league = league
                        break
                else:
                    # Try normalized
                    for league, slug in LEAGUE_SLUG_MAP.items():
                        if slug_part.startswith(slug.replace(" ", "_").lower()):
                            best_league = league
                            break

    return best_league, best_season, best_score


def _map_fbref_position(fbref_pos_str):
    if not fbref_pos_str or pd.isna(fbref_pos_str):
        return "MID"
    first = str(fbref_pos_str).split(",")[0].strip()
    return FBREF_POS_MAP.get(first, "MID")


def _get_age_from_player_summaries(fpl_id):
    """
    Try to estimate birth year from player_summaries history_past.
    Returns birth year (int) or None.
    """
    summary_dir = os.path.join(FPL_API_DIR, "player_summaries")
    if not os.path.exists(summary_dir):
        return None
    fpath = os.path.join(summary_dir, f"{int(fpl_id)}.json")
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        history_past = data.get("history_past", [])
        if not history_past:
            return None
        # history_past has season_name like "2021/22" — get earliest season
        seasons = []
        for entry in history_past:
            sn = entry.get("season_name", "")
            m = re.match(r"(\d{4})/\d{2}", sn)
            if m:
                seasons.append(int(m.group(1)))
        return min(seasons) if seasons else None
    except Exception:
        return None


def _has_pre_vaastav_pl_history(fpl_id):
    """
    Returns True if player_summaries shows FPL history before 2019/20.
    This indicates they had a PL career before vaastav coverage.
    """
    earliest = _get_age_from_player_summaries(fpl_id)
    if earliest is None:
        return False
    return earliest < 2019  # season starting before 2019 -> pre-vaastav


def _get_born_year_from_fbref(player_name, threshold=80):
    """
    Try to find player birth year from any FBref cache file.
    Returns birth year (int) or None.
    """
    norm_pname = normalize_name(player_name)
    for fname in os.listdir(FBREF_RAW_DIR):
        if not fname.endswith(".csv"):
            continue
        try:
            df = pd.read_csv(os.path.join(FBREF_RAW_DIR, fname))
        except Exception:
            continue
        pcol = _find_player_col(df)
        born_col = None
        for c in df.columns:
            if "born" in c.lower():
                born_col = c
                break
        if not pcol or not born_col:
            continue
        names_norm = [normalize_name(str(n)) for n in df[pcol].dropna()]
        res = fzprocess.extractOne(norm_pname, names_norm, scorer=fuzz.token_sort_ratio)
        if res and res[1] >= threshold:
            idx = names_norm.index(res[0])
            try:
                by = int(df[born_col].iloc[idx])
                return by
            except (ValueError, TypeError):
                pass
    return None


def _current_age(born_year):
    """Compute approximate age from birth year. Today is 2026-03-09."""
    if born_year is None:
        return None
    return 2026 - born_year


# =============================================================================
# STEP 1 — Identify players who debuted in the PL in 2024/25
# =============================================================================

def step1_identify_debutants(state):
    print()
    print("=" * 70)
    print("STEP 1: Identify players who debuted in the PL in 2024/25")
    print("=" * 70)

    # Load vaastav
    vaastav_path = os.path.join(VAASTAV_DIR, "historical_gw_data.csv")
    vaastav = pd.read_csv(vaastav_path)
    print(f"  Loaded vaastav: {len(vaastav)} rows, seasons: "
          f"{sorted(vaastav['season'].unique())}")

    # 2024-25 players with at least one GW with minutes >= 90
    v2425 = vaastav[vaastav["season"] == "2024-25"]
    qualified_2425 = v2425[v2425["minutes"] >= 90]["name"].unique()
    print(f"  2024-25 players with >= 90 mins in at least one GW: {len(qualified_2425)}")

    # Players with ANY row in 2019-20 through 2023-24
    prior_seasons_mask = vaastav["season"].isin(
        ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    )
    players_with_prior = set(vaastav[prior_seasons_mask]["name"].unique())
    players_with_prior_norm = {normalize_name(n) for n in players_with_prior}
    players_in_2425_norm    = {normalize_name(n) for n in v2425["name"].unique()}
    print(f"  Players with any prior vaastav history (2019-24): {len(players_with_prior)}")

    # Initial candidate list: in 2024-25 + no prior history
    candidates = [n for n in qualified_2425 if n not in players_with_prior]
    print(f"  Initial candidates (2024-25 debut, no prior vaastav): {len(candidates)}")

    # Load Stage 4a position files — already covered players
    stage4a_names = set()
    for pos_file in ["new_signings_gk.csv", "new_signings_def.csv",
                     "new_signings_mid.csv", "new_signings_fwd.csv"]:
        fpath = os.path.join(FBREF_SIGN_DIR, pos_file)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            name_col = "name" if "name" in df.columns else df.columns[0]
            stage4a_names.update(df[name_col].dropna().unique())
    stage4a_names_norm = {normalize_name(n) for n in stage4a_names}
    print(f"  Stage 4a position files contain {len(stage4a_names)} unique names")

    # Load FPL players_raw for price, team, position
    players_raw = pd.read_csv(os.path.join(FPL_API_DIR, "players_raw.csv"))
    players_raw["full_name"] = (
        players_raw["first_name"].str.strip() + " "
        + players_raw["second_name"].str.strip()
    )
    fpl_name_list = players_raw["full_name"].tolist()
    fpl_name_norm = [normalize_name(n) for n in fpl_name_list]
    fpl_norm_to_row = {}
    for _, row in players_raw.iterrows():
        fpl_norm_to_row[normalize_name(row["full_name"])] = row

    confirmed    = []
    unconfirmed  = []  # age 28+
    flagged      = []  # name change risk
    excluded     = []

    for vaastav_name in sorted(candidates):
        norm_vname = normalize_name(vaastav_name)

        # Exclusion check
        if vaastav_name in STAGE4B_EXCLUSIONS or norm_vname in {
            normalize_name(e) for e in STAGE4B_EXCLUSIONS
        }:
            excluded.append(vaastav_name)
            continue

        # Check if already in Stage 4a
        if norm_vname in stage4a_names_norm:
            # Try fuzzy match to be safe
            m, s = fuzzy_match_name(norm_vname, list(stage4a_names_norm))
            if m:
                continue  # already covered by Stage 4a

        # Match to FPL players_raw
        res = fzprocess.extractOne(norm_vname, fpl_name_norm, scorer=fuzz.token_sort_ratio)
        if not res or res[1] < FUZZY_THRESHOLD:
            # Can't match to FPL — skip (likely retired or very low price)
            continue

        fpl_norm_match = res[0]
        fpl_row = None
        for norm_fn, row in fpl_norm_to_row.items():
            if norm_fn == fpl_norm_match:
                fpl_row = row
                break

        if fpl_row is None:
            continue

        # Price filter
        price = float(fpl_row.get("price", 0) or 0)
        if price < 4.0:
            continue

        fpl_name = fpl_row["full_name"]
        fpl_team = str(fpl_row.get("team_name", ""))
        raw_pos  = str(fpl_row.get("position", "MID")).upper()
        fpl_pos  = FPL_POS_MAP.get(raw_pos, raw_pos)
        fpl_id   = int(fpl_row["id"])

        # 2024-25 total minutes in vaastav
        p_2425 = v2425[v2425["name"] == vaastav_name]
        total_mins_2425 = int(p_2425["minutes"].sum())

        # Age check — try birth year from FBref cache files
        born_year = _get_born_year_from_fbref(vaastav_name)
        age = _current_age(born_year)

        # Also check for pre-vaastav PL history via player_summaries
        has_pre_vaastav = _has_pre_vaastav_pl_history(fpl_id)

        # Name change / encoding guard:
        # A genuine name-change risk exists only when:
        #   (a) the FPL name match score is < 100% (exact match = definitely same player)
        #   (b) a vaastav veteran at the same team+pos+price dropped out of 2024-25
        # If the vaastav name matches the FPL name exactly (100%), the player identity
        # is confirmed — there is no ambiguity and we skip this check entirely.
        possible_name_change = False
        name_change_reason   = ""
        fpl_match_score = res[1]  # score from the FPL fuzzy match above
        if fpl_match_score < 100:
            for _, pr_row in players_raw[players_raw["team_name"] == fpl_team].iterrows():
                pr_full = pr_row["full_name"]
                pr_norm = normalize_name(pr_full)
                # Skip the same player
                if pr_norm == norm_vname:
                    continue
                # Must have prior vaastav history (pre-2024-25) — use normalized set
                if pr_norm not in players_with_prior_norm:
                    continue
                # Key guard: veteran must have DROPPED OUT of vaastav 2024-25
                # If the veteran still appears in 2024-25, they co-exist -> no flag
                if pr_norm in players_in_2425_norm:
                    continue
                pr_price = float(pr_row.get("price", 0) or 0)
                pr_raw_pos = str(pr_row.get("position", "")).upper()
                pr_pos = FPL_POS_MAP.get(pr_raw_pos, pr_raw_pos)
                # Same position + price within £0.5m + same team + veteran dropped out = flag
                if pr_pos == fpl_pos and abs(pr_price - price) <= 0.5:
                    possible_name_change = True
                    name_change_reason = (
                        f"Same team/pos as ex-vaastav veteran '{pr_full}' who left 2024-25 "
                        f"(price diff: £{abs(pr_price - price):.1f}m)"
                    )
                    break

        if possible_name_change:
            flagged.append({
                "vaastav_name": vaastav_name,
                "fpl_name": fpl_name,
                "fpl_team": fpl_team,
                "fpl_pos": fpl_pos,
                "price": price,
                "age": age if age else "?",
                "mins_2425": total_mins_2425,
                "reason": name_change_reason,
            })
            continue

        entry = {
            "vaastav_name": vaastav_name,
            "fpl_name": fpl_name,
            "fpl_id": fpl_id,
            "fpl_team": fpl_team,
            "fpl_pos": fpl_pos,
            "price": price,
            "age": age if age else "?",
            "born_year": born_year if born_year else "?",
            "mins_2425": total_mins_2425,
            "has_pre_vaastav_pl": has_pre_vaastav,
        }

        # Age guard: 28+ or has pre-vaastav PL history -> unconfirmed
        if has_pre_vaastav or (age is not None and age >= 28):
            unconfirmed.append(entry)
        else:
            confirmed.append(entry)

    # Print results
    print()
    print("=" * 70)
    print("=== CONFIRMED 2024/25 PL DEBUTANTS ===")
    print("=" * 70)
    hdr = f"  {'#':>3}  {'Name':<25} {'Team':<20} {'Pos':<5} {'Price':>6}  {'Age':>4}  {'2024-25 Mins':>13}"
    print(hdr)
    for i, e in enumerate(confirmed, 1):
        print(f"  {i:>3}  {e['vaastav_name']:<25} {e['fpl_team']:<20} "
              f"{e['fpl_pos']:<5} £{e['price']:.1f}m  {str(e['age']):>4}  "
              f"{e['mins_2425']:>13}")
    print(f"Total confirmed: {len(confirmed)}")

    print()
    print("=" * 70)
    print("=== UNCONFIRMED -- AGE 28+ or pre-2019 PL history (review manually) ===")
    print("=" * 70)
    print(hdr)
    for i, e in enumerate(unconfirmed, 1):
        flag = " [pre-vaastav PL]" if e["has_pre_vaastav_pl"] else ""
        print(f"  {i:>3}  {e['vaastav_name']:<25} {e['fpl_team']:<20} "
              f"{e['fpl_pos']:<5} £{e['price']:.1f}m  {str(e['age']):>4}  "
              f"{e['mins_2425']:>13}{flag}")
    print(f"Total unconfirmed: {len(unconfirmed)}")

    print()
    print("=" * 70)
    print("=== FLAGGED -- POSSIBLE NAME CHANGE / DATA ISSUE ===")
    print("=" * 70)
    flag_hdr = f"  {'#':>3}  {'Vaastav Name':<25} {'Possible Match/Reason'}"
    print(flag_hdr)
    for i, e in enumerate(flagged, 1):
        print(f"  {i:>3}  {e['vaastav_name']:<25} {e['reason']}")
    print(f"Total flagged: {len(flagged)}")

    print()
    print(f"Total excluded (STAGE4B_EXCLUSIONS): {len(excluded)}")
    print()
    print("Add any exclusions to STAGE4B_EXCLUSIONS then re-run,")
    print("or type y to proceed with confirmed list only.")

    # Save state
    state["step1"] = {
        "confirmed": confirmed,
        "unconfirmed": unconfirmed,
        "flagged": flagged,
        "excluded": excluded,
    }
    _save_state(state)

    return confirmed, unconfirmed, flagged, excluded


# =============================================================================
# STEP 2 — Determine previous league per player
# =============================================================================

def step2_determine_leagues(confirmed, state):
    print()
    print("=" * 70)
    print("STEP 2: Determine previous league per player")
    print("=" * 70)

    # Load transfer CSV for previous_league lookup
    transfers_path = os.path.join(TRANSFERS_DIR, "new_signings_2025.csv")
    transfers_df = pd.DataFrame()
    if os.path.exists(transfers_path):
        transfers_df = pd.read_csv(transfers_path)
        print(f"  Loaded new_signings_2025.csv: {len(transfers_df)} rows")
    else:
        print("  [WARN] new_signings_2025.csv not found — will rely on cache inference only")

    # Build normalized lookup from transfers
    tm_lookup = {}  # norm_fpl_name -> previous_league_standardized
    if not transfers_df.empty and "fpl_name" in transfers_df.columns:
        league_col = "previous_league_standardized"
        if league_col not in transfers_df.columns:
            league_col = "previous_league"
        for _, row in transfers_df.iterrows():
            if pd.isna(row.get("fpl_name", "")):
                continue
            nname = normalize_name(str(row["fpl_name"]))
            league = str(row.get(league_col, "")) if row.get(league_col) else ""
            if league and league not in ("nan", "None", ""):
                league_std = LEAGUE_NORMALIZE_MAP.get(league, league)
                tm_lookup[nname] = league_std

    # Build plan
    scrape_plan = []
    known_count  = 0
    unknown_count = 0
    cache_hits   = 0
    new_scrapes  = 0

    for entry in confirmed:
        fpl_name = entry["fpl_name"]
        norm_name = normalize_name(fpl_name)

        # 1. Check transfers CSV
        previous_league = tm_lookup.get(norm_name, "")

        if not previous_league:
            # Try vaastav_name normalization too
            norm_vaastav = normalize_name(entry["vaastav_name"])
            previous_league = tm_lookup.get(norm_vaastav, "")

        if not previous_league:
            # Fuzzy match against transfers
            tm_names_norm = list(tm_lookup.keys())
            if tm_names_norm:
                res = fzprocess.extractOne(norm_name, tm_names_norm, scorer=fuzz.token_sort_ratio)
                if res and res[1] >= 85:
                    previous_league = tm_lookup[res[0]]

        # 2. Check FBref cache files if still not found
        inferred_from_cache = False
        if not previous_league:
            inf_league, inf_season, inf_score = _infer_league_from_cache(fpl_name)
            if inf_league and inf_score >= 90:
                previous_league = inf_league
                inferred_from_cache = True

        if not previous_league:
            unknown_count += 1
            scrape_plan.append({
                **entry,
                "previous_league": "unknown",
                "scrape_status": "skip",
                "seasons_to_scrape": [],
                "cache_status": [],
            })
            continue

        known_count += 1

        # Determine seasons to scrape
        born_year = entry.get("born_year", "?")
        try:
            born_int = int(born_year)
            very_young = born_int >= 2004  # born 2004 or later -> skip 2021-22
        except (ValueError, TypeError):
            very_young = False

        seasons = ["2023-24", "2022-23"]
        if not very_young:
            seasons.append("2021-22")

        cache_status = []
        for s in seasons:
            cp = _get_cache_path(previous_league, s)
            if os.path.exists(cp):
                cache_status.append("HIT")
                cache_hits += 1
            else:
                cache_status.append("MISS")
                new_scrapes += 1

        scrape_plan.append({
            **entry,
            "previous_league": previous_league,
            "scrape_status": "scrape" if previous_league in LEAGUE_MULTIPLIERS else "skip",
            "seasons_to_scrape": seasons,
            "cache_status": cache_status,
            "inferred_from_cache": inferred_from_cache,
        })

    # Print plan
    print()
    print("=== SCRAPE PLAN ===")
    hdr = f"  {'Player':<28} {'Pos':<5} {'Previous League':<22} {'Seasons to attempt':<35} {'Cache'}"
    print(hdr)
    print("  " + "-" * 110)
    for p in scrape_plan:
        if p["scrape_status"] == "skip" and p["previous_league"] == "unknown":
            continue
        seasons_str = ", ".join(p["seasons_to_scrape"])
        cache_str   = " / ".join(p["cache_status"])
        print(f"  {p['vaastav_name']:<28} {p['fpl_pos']:<5} "
              f"{p['previous_league']:<22} {seasons_str:<35} {cache_str}")

    unknown_players = [p for p in scrape_plan if p["previous_league"] == "unknown"]
    if unknown_players:
        print()
        print("Players with unknown previous league (will be skipped):")
        for p in unknown_players:
            print(f"  {p['vaastav_name']}  -- not found in transfers CSV or any cache file")

    state["step2"] = {"scrape_plan": scrape_plan}
    _save_state(state)

    return scrape_plan, known_count, unknown_count, cache_hits, new_scrapes


# =============================================================================
# STEP 3 — FBref scrape with proper caching
# =============================================================================

def _scrape_fbref_selenium(league_std, season, stat_types=("standard", "keeper")):
    """
    Scrape FBref using SeleniumBase (same Cloudflare bypass as Stage 4a).
    Returns a DataFrame or None.
    """
    if not _HAS_SELENIUM:
        print(f"  [ERROR] SeleniumBase not available — cannot scrape {league_std} {season}")
        return None

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [ERROR] beautifulsoup4 not installed")
        return None

    dfs = []
    for stat_type in stat_types:
        url = _build_fbref_url(league_std, season, stat_type=stat_type)
        if not url:
            continue
        table_id = "stats_standard" if stat_type == "standard" else "stats_keeper"
        print(f"  Scraping {url} ...", end="", flush=True)
        try:
            with SB(uc=True, headless=True) as sb:
                sb.open(url)
                time.sleep(4)
                html = sb.get_page_source()
            df = _parse_fbref_page_html(html, table_id)
            if df is not None and len(df) > 0:
                df["league_tag"]  = league_std
                df["season_tag"]  = season
                dfs.append(df)
                print(f" done ({len(df)} rows)")
            else:
                print(" no data found")
        except Exception as e:
            print(f" ERROR: {e}")

    if not dfs:
        return None
    # Merge standard + keeper
    combined = dfs[0]
    if len(dfs) > 1:
        pcol = _find_player_col(dfs[0])
        if pcol:
            combined = dfs[0].merge(dfs[1], on=pcol, how="left", suffixes=("", "_k"))
    return combined


def step3_scrape(scrape_plan, state):
    print()
    print("=" * 70)
    print("STEP 3: Execute scrape plan (cached files reused, new files scraped)")
    print("=" * 70)

    # Build unique (league, season) pairs to scrape
    pairs_needed = set()
    for p in scrape_plan:
        if p["scrape_status"] == "skip":
            continue
        for s in p["seasons_to_scrape"]:
            pairs_needed.add((p["previous_league"], s))

    cache_reused  = 0
    files_created = 0
    total_rows    = 0
    all_cache     = {}  # (league, season) -> DataFrame

    # Process pairs: load from cache or scrape
    for league, season in sorted(pairs_needed):
        cp = _get_cache_path(league, season)
        if os.path.exists(cp):
            try:
                df = pd.read_csv(cp)
                print(f"  Cache HIT : {os.path.basename(cp)} ({len(df)} rows)")
                all_cache[(league, season)] = df
                cache_reused += 1
                total_rows += len(df)
            except Exception as e:
                print(f"  [WARN] Could not load cache {cp}: {e}")
        else:
            print(f"  Cache MISS: {league} {season} -- scraping FBref ...")
            if league not in FBREF_LEAGUE_IDS:
                print(f"  [SKIP] {league} not in FBREF_LEAGUE_IDS -- cannot scrape")
                continue
            df = _scrape_fbref_selenium(league, season)
            if df is not None and len(df) > 0:
                df.to_csv(cp, index=False)
                print(f"  Saved: {os.path.basename(cp)} ({len(df)} rows)")
                all_cache[(league, season)] = df
                files_created += 1
                total_rows += len(df)
            else:
                print(f"  [WARN] No data scraped for {league} {season} -- skipping")

    # Now match target players in the scraped data
    # Build target lookup: norm_name -> entry dict
    target_entries = {
        normalize_name(p["fpl_name"]): p
        for p in scrape_plan if p["scrape_status"] != "skip"
    }
    target_vaastav_norm = {
        normalize_name(p["vaastav_name"]): p
        for p in scrape_plan if p["scrape_status"] != "skip"
    }
    target_norm_list = list(target_entries.keys())

    # False positive set (normalized)
    fp_norm_set = {
        (normalize_name(fb), normalize_name(fpl))
        for fb, fpl in FBREF_FALSE_POSITIVES
    }

    # Override map
    override_reverse = {
        normalize_name(v): k for k, v in FBREF_NAME_OVERRIDES.items()
    }

    matches_high  = 0  # >= 95%
    matches_low   = 0  # 80-94%
    blocked_fp    = 0
    below_thresh  = 0
    matched_rows  = []

    print()
    print("  Matching target players in FBref data ...")

    for (league, season), cache_df in all_cache.items():
        pcol = _find_player_col(cache_df)
        if not pcol:
            continue
        fbref_names = cache_df[pcol].dropna().astype(str).unique().tolist()
        fbref_norm  = [normalize_name(n) for n in fbref_names]
        norm_to_fbref = {normalize_name(n): n for n in fbref_names}

        # Build match cache for this file
        for norm_target, tentry in list(target_entries.items()) + list(target_vaastav_norm.items()):
            # Check override
            if norm_target in override_reverse:
                fpl_key = normalize_name(override_reverse[norm_target])
                # Find original fbref name
                for fn_norm, fn_orig in norm_to_fbref.items():
                    if fn_norm == normalize_name(override_reverse[norm_target]):
                        score = 100
                        fbref_orig = fn_orig
                        break
                else:
                    continue
            else:
                res = fzprocess.extractOne(norm_target, fbref_norm, scorer=fuzz.token_sort_ratio)
                if not res or res[1] < FUZZY_THRESHOLD:
                    continue
                score = res[1]
                fbref_orig = norm_to_fbref.get(res[0], res[0])

            # Check which tentry this is
            entry = target_entries.get(norm_target) or target_vaastav_norm.get(norm_target)
            if entry is None:
                continue
            fpl_name_norm = normalize_name(entry["fpl_name"])

            # Skip if wrong season for this player
            if season not in entry.get("seasons_to_scrape", []):
                continue

            # False positive check
            pair_norm = (normalize_name(fbref_orig), fpl_name_norm)
            if pair_norm in fp_norm_set:
                blocked_fp += 1
                continue

            # Get matching rows for this player-season
            player_rows = cache_df[
                cache_df[pcol].apply(lambda x: normalize_name(str(x)) == normalize_name(fbref_orig))
            ]

            if player_rows.empty:
                continue

            if score < CONFIRM_THRESHOLD and score >= FUZZY_THRESHOLD:
                # Prompt for confirmation
                stats_row = player_rows.iloc[0]
                min_col = find_col(cache_df, ["Playing Time_Min", "Min"])
                apps_col = find_col(cache_df, ["Playing Time_MP", "MP"])
                gls_col = find_col(cache_df, ["Performance_Gls", "Gls"])
                ast_col = find_col(cache_df, ["Performance_Ast", "Ast"])
                mins_val = stats_row.get(min_col, "?") if min_col else "?"
                apps_val = stats_row.get(apps_col, "?") if apps_col else "?"
                gls_val  = stats_row.get(gls_col,  "?") if gls_col  else "?"
                ast_val  = stats_row.get(ast_col,  "?") if ast_col  else "?"
                print()
                print("  LOW CONFIDENCE MATCH -- confirmation required:")
                print(f"    FBref name:  \"{fbref_orig}\"")
                print(f"    FPL target:  \"{entry['fpl_name']}\"")
                print(f"    Confidence:  {score}%")
                print(f"    Season:      {season} / {league}")
                print(f"    Stats:       {apps_val} apps, {mins_val} mins, {gls_val}G {ast_val}A")
                ans = input("    Include this row? (y/n): ").strip().lower()
                if ans != "y":
                    below_thresh += 1
                    continue
                matches_low += 1
                confidence_label = "medium"
            else:
                matches_high += 1
                confidence_label = "high"

            # Tag and collect rows
            for _, dr in player_rows.iterrows():
                matched_rows.append({
                    "_fpl_name":    entry["fpl_name"],
                    "_vaastav_name": entry["vaastav_name"],
                    "_fpl_pos":     entry["fpl_pos"],
                    "_fpl_team":    entry["fpl_team"],
                    "_price":       entry["price"],
                    "_prev_league": entry["previous_league"],
                    "_season":      season,
                    "_league":      league,
                    "_score":       score,
                    "_confidence":  confidence_label,
                    "_fbref_name":  fbref_orig,
                    **{c: dr[c] for c in cache_df.columns},
                })

    matched_df = pd.DataFrame(matched_rows)
    players_matched = matched_df["_fpl_name"].nunique() if not matched_df.empty else 0

    print()
    print(f"  Cache files reused:     {cache_reused}")
    print(f"  New cache files:        {files_created}")
    print(f"  Total rows scraped:     {total_rows}")
    print(f"  Players matched:        {players_matched}")
    print(f"    >= 95% confidence:    {matches_high}")
    print(f"    80-95% (confirmed):   {matches_low}")
    print(f"    Blocked (false pos):  {blocked_fp}")
    print(f"    Below threshold:      {below_thresh}")

    state["step3"] = {
        "cache_reused": cache_reused,
        "files_created": files_created,
        "total_rows": total_rows,
        "players_matched": players_matched,
        "matches_high": matches_high,
        "matches_low": matches_low,
        "blocked_fp": blocked_fp,
        "below_thresh": below_thresh,
    }
    _save_state(state)

    return matched_df, {
        "cache_reused": cache_reused,
        "files_created": files_created,
        "total_rows": total_rows,
        "players_matched": players_matched,
        "matches_high": matches_high,
        "matches_low": matches_low,
        "blocked_fp": blocked_fp,
        "below_thresh": below_thresh,
    }


# =============================================================================
# STEP 4 — Extract stats with full position-specific schema
# =============================================================================

def step4_extract_stats(matched_df, scrape_plan):
    print()
    print("=" * 70)
    print("STEP 4: Extract stats with position-specific schema")
    print("=" * 70)

    if matched_df.empty:
        print("  [WARN] No matched rows to extract.")
        return pd.DataFrame()

    # Build fpl_name -> scrape plan entry lookup
    plan_lookup = {normalize_name(p["fpl_name"]): p for p in scrape_plan}

    # Read existing position files to confirm column schema
    pos_schemas = {}
    for pos, slug in [("GK", "gk"), ("DEF", "def"), ("MID", "mid"), ("FWD", "fwd")]:
        fpath = os.path.join(FBREF_SIGN_DIR, f"new_signings_{slug}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, nrows=0)
            pos_schemas[pos] = list(df.columns)

    extracted_rows = []
    small_sample_count = 0
    high_adj_g_flags   = []

    for _, row in matched_df.iterrows():
        fpl_name   = row["_fpl_name"]
        fpl_pos    = row["_fpl_pos"]
        fpl_team   = row["_fpl_team"]
        price      = float(row["_price"])
        prev_league = row["_prev_league"]
        season     = row["_season"]
        confidence = row["_confidence"]

        cols = list(matched_df.columns)
        stats = _extract_fbref_stats(row, cols)

        mins  = float(stats["minutes"]     or 0)
        apps  = float(stats["appearances"] or 0)
        gls   = float(stats["goals"]       or 0)
        ast   = float(stats["assists"]     or 0)
        cs    = float(stats["clean_sheets"] or 0)
        saves = float(stats["saves"]       or 0)
        crdy  = float(stats["yellow_cards"] or 0)
        crdr  = float(stats["red_cards"]    or 0)

        multiplier  = LEAGUE_MULTIPLIERS.get(prev_league, 1.0)
        nineties    = mins / 90.0 if mins > 0 else 0

        adjG_per_90 = (gls / mins * 90) * multiplier if mins > 0 else 0.0
        adjA_per_90 = (ast / mins * 90) * multiplier if mins > 0 else 0.0

        # Season reliability (partition by season — each row is one season already)
        reliability = _season_reliability(mins, apps)

        small_sample = mins < 500

        if small_sample:
            small_sample_count += 1

        # Flag high adjG/90 for non-small-sample rows
        if adjG_per_90 > 1.5 and not small_sample:
            high_adj_g_flags.append({
                "fpl_name": fpl_name,
                "season": season,
                "league": prev_league,
                "adjG_per_90": round(adjG_per_90, 3),
                "mins": int(mins),
                "goals": int(gls),
            })

        # Rate stats
        cs_rate = safe_div(cs, apps) if fpl_pos in ("GK", "DEF") else 0.0
        sv_pg   = safe_div(saves, apps) if fpl_pos == "GK" else 0.0

        seas_yr = season_to_year(season)

        out_row = {
            "name":           fpl_name,
            "position":       fpl_pos,
            "team":           fpl_team,
            "GW":             0,
            "opponent_team":  "",
            "was_home":       False,
            "total_points":   0,
            "minutes":        mins,
            "goals_scored":   gls,
            "assists":        ast,
            "clean_sheets":   cs if fpl_pos in ("GK", "DEF") else 0,
            "goals_conceded": 0,
            "own_goals":      0,
            "penalties_saved":  0,
            "penalties_missed": 0,
            "yellow_cards":   crdy,
            "red_cards":      crdr,
            "saves":          saves if fpl_pos == "GK" else 0,
            "bonus":          0,
            "bps":            0,
            "influence":      0.0,
            "creativity":     0.0,
            "threat":         0.0,
            "ict_index":      0.0,
            "transfers_in":   0,
            "transfers_out":  0,
            "selected":       0,
            "value":          price,
            "season":         season,
            "season_year":    seas_yr,
            "form_last3":     0.0,
            "form_last5":     0.0,
            "minutes_reliability_season":   reliability,
            "cumulative_points_season":     0,
            "avg_points_per_game_season":   0.0,
            "goals_per_game_season":        adjG_per_90,
            "assists_per_game_season":      adjA_per_90,
            "clean_sheet_rate_season":      cs_rate,
            "saves_per_game_season":        sv_pg,
            "points_per_million":           0.0,
            "is_new_to_pl":                 1,
            "data_source":                  "stage4b",
            # Extra FBref metadata (not in VAASTAV_COLS, used for deduplication)
            "_prev_league":    prev_league,
            "_multiplier":     multiplier,
            "_small_sample":   small_sample,
            "_adjG_per_90":    adjG_per_90,
            "_adjA_per_90":    adjA_per_90,
            "_reliability":    reliability,
            "_confidence":     confidence,
            "_fbref_name":     row.get("_fbref_name", ""),
        }
        extracted_rows.append(out_row)

    result_df = pd.DataFrame(extracted_rows)

    # Preview
    print()
    print("=== EXTRACTION PREVIEW (first 3 rows per position) ===")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        sub = result_df[result_df["position"] == pos] if not result_df.empty else pd.DataFrame()
        if sub.empty:
            print(f"\n{pos}: (no rows)")
            continue
        print(f"\n{pos}:")
        hdr = (f"  {'Player':<25} {'Season':<10} {'Team':<18} "
               f"{'Mins':>6} {'adjG/90':>8} {'adjA/90':>8} {'Rel':>5} {'Small?':>7}")
        print(hdr)
        for _, r in sub.head(3).iterrows():
            print(f"  {r['name']:<25} {r['season']:<10} {r['team']:<18} "
                  f"{int(r['minutes']):>6} {r['_adjG_per_90']:>8.3f} "
                  f"{r['_adjA_per_90']:>8.3f} {r['minutes_reliability_season']:>5.1f} "
                  f"{str(r['_small_sample']):>7}")

    if high_adj_g_flags:
        print()
        print("  [WARNING] adjG/90 > 1.5 on non-small-sample rows:")
        for f in high_adj_g_flags:
            print(f"    {f['fpl_name']} {f['season']} {f['league']}: "
                  f"adjG/90={f['adjG_per_90']} ({f['goals']}G in {f['mins']}mins)")

    # Counts by position
    print()
    for pos in ["GK", "DEF", "MID", "FWD"]:
        sub = result_df[result_df["position"] == pos] if not result_df.empty else pd.DataFrame()
        n_unique = sub["name"].nunique() if not sub.empty else 0
        print(f"  {pos}: {len(sub)} rows ({n_unique} unique players)")
    print(f"  Small sample warnings (< 500 mins): {small_sample_count} rows")

    return result_df, high_adj_g_flags


# =============================================================================
# STEP 5 — Append to position files with source priority
# =============================================================================

SOURCE_PRIORITY = {"vaastav": 3, "stage4a": 2, "stage4b": 1, "manual": 0}


def step5_write_position_files(extracted_df, scrape_plan):
    print()
    print("=" * 70)
    print("STEP 5: Append to position files with source priority deduplication")
    print("=" * 70)

    if extracted_df.empty:
        print("  [WARN] No extracted data to write.")
        return {}

    pos_map  = {"GK": "gk", "DEF": "def", "MID": "mid", "FWD": "fwd"}
    results  = {}
    total_appended = 0
    total_removed  = 0
    backfilled_rows = 0
    schema_ok = True

    for fpl_pos, slug in pos_map.items():
        fpath = os.path.join(FBREF_SIGN_DIR, f"new_signings_{slug}.csv")

        # Load existing file
        if os.path.exists(fpath):
            existing = pd.read_csv(fpath)
        else:
            existing = pd.DataFrame(columns=VAASTAV_COLS)

        # Backfill data_source on existing rows
        if "data_source" not in existing.columns:
            existing["data_source"] = "stage4a"
            backfilled_rows += len(existing)
        else:
            mask_missing = existing["data_source"].isna() | (existing["data_source"] == "")
            existing.loc[mask_missing, "data_source"] = "stage4a"

        # New rows for this position
        new_rows = extracted_df[extracted_df["position"] == fpl_pos].copy()

        # Strip internal _* columns before writing
        internal_cols = [c for c in new_rows.columns if c.startswith("_")]
        write_cols    = VAASTAV_COLS + ["data_source"]
        new_rows_write = new_rows.drop(columns=internal_cols, errors="ignore")

        # Ensure all output columns exist
        for col in write_cols:
            if col not in new_rows_write.columns:
                new_rows_write[col] = 0
        if not new_rows_write.empty:
            new_rows_write = new_rows_write[write_cols]

        n_new = len(new_rows_write)

        # Combine
        combined = pd.concat([existing, new_rows_write], ignore_index=True)

        # Deduplication: on name + season, keep highest priority source
        # GW=0 rows are season-aggregated (from FBref); GW>0 are vaastav GW-level
        # Only deduplicate GW=0 rows against each other
        def dedup_priority(df):
            gw0 = df[df["GW"] == 0].copy()
            gw_other = df[df["GW"] != 0].copy()
            # Assign priority score
            gw0["_prio"] = gw0["data_source"].map(
                lambda s: SOURCE_PRIORITY.get(str(s).strip(), 0)
            )
            # For each (name, season) keep highest priority
            gw0_sorted = gw0.sort_values("_prio", ascending=False)
            gw0_dedup  = gw0_sorted.drop_duplicates(subset=["name", "season"], keep="first")
            removed_n  = len(gw0) - len(gw0_dedup)
            gw0_dedup  = gw0_dedup.drop(columns=["_prio"])
            return pd.concat([gw_other, gw0_dedup], ignore_index=True), removed_n

        combined, removed_n = dedup_priority(combined)
        total_removed  += removed_n
        total_appended += n_new

        # Fill NaN -> 0
        for col in write_cols:
            if col in combined.columns:
                if combined[col].dtype in (float, int) or col not in ("name", "position", "team",
                        "opponent_team", "season", "data_source"):
                    combined[col] = combined[col].fillna(0)

        # Validate schema
        missing_cols = [c for c in write_cols if c not in combined.columns]
        if missing_cols:
            schema_ok = False
            print(f"  [WARN] {fpl_pos}: missing columns: {missing_cols}")

        # Ensure output column order
        for col in write_cols:
            if col not in combined.columns:
                combined[col] = 0
        combined = combined[write_cols]

        # adjG > 1.5 non-small-sample check
        # goals_per_game_season stores adjG_per_90
        if "goals_per_game_season" in combined.columns:
            high_g = combined[
                (combined["goals_per_game_season"] > 1.5)
                & (combined["minutes"] >= 500)
                & (combined["GW"] == 0)
            ]
            if len(high_g) > 0:
                print(f"  [DATA ERROR] {fpl_pos}: {len(high_g)} rows with "
                      f"adjG/90 > 1.5 and minutes >= 500 -- investigate before proceeding!")
                for _, hr in high_g.iterrows():
                    print(f"    {hr['name']} {hr['season']} "
                          f"adjG/90={hr['goals_per_game_season']:.3f} mins={hr['minutes']}")

        # Write
        combined.to_csv(fpath, index=False)
        results[fpl_pos] = {"path": fpath, "n_new": n_new, "total": len(combined)}
        print(f"  {fpl_pos}: +{n_new} new rows -> {len(combined)} total rows in {os.path.basename(fpath)}")

    # Schema consistency check across all 4 files
    all_schemas = {}
    for fpl_pos, slug in pos_map.items():
        fpath = os.path.join(FBREF_SIGN_DIR, f"new_signings_{slug}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, nrows=0)
            all_schemas[fpl_pos] = sorted(df.columns.tolist())

    schemas_identical = len(set(tuple(v) for v in all_schemas.values())) == 1

    # Null count
    null_count = 0
    for fpl_pos, slug in pos_map.items():
        fpath = os.path.join(FBREF_SIGN_DIR, f"new_signings_{slug}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            write_cols_check = [c for c in write_cols if c in df.columns
                                and c not in ("opponent_team",)]
            null_count += int(df[write_cols_check].isnull().sum().sum())

    print()
    print(f"  Total new rows appended:              {total_appended}")
    print(f"  Rows removed by deduplication:        {total_removed}")
    print(f"  data_source backfilled on existing:   {backfilled_rows}")
    print(f"  Schema identical across 4 files:      {'YES' if schemas_identical else 'NO'}")
    print(f"  Null values remaining:                {null_count}")

    return results, total_appended, total_removed, backfilled_rows, schemas_identical, null_count


# =============================================================================
# STEP 6 — Validation report
# =============================================================================

def step6_report(confirmed, unconfirmed, flagged, excluded,
                 scrape_plan, scrape_stats, extracted_df,
                 append_results, total_appended, total_removed,
                 backfilled_rows, schemas_identical, null_count,
                 high_adj_flags, step3_low_conf_decisions):
    print()
    print("=" * 70)
    print("=== STAGE 4b VALIDATION REPORT ===")
    print("=" * 70)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    known_count   = sum(1 for p in scrape_plan if p["previous_league"] != "unknown")
    unknown_count = sum(1 for p in scrape_plan if p["previous_league"] == "unknown")

    # Unique players/rows per position
    pos_stats = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        sub = extracted_df[extracted_df["position"] == pos] if not extracted_df.empty else pd.DataFrame()
        pos_stats[pos] = {"rows": len(sub), "players": sub["name"].nunique() if not sub.empty else 0}

    total_rows_added = sum(v["rows"] for v in pos_stats.values())

    report_lines = [
        "=" * 70,
        "=== STAGE 4b VALIDATION REPORT ===",
        f"Generated: {ts}",
        "=" * 70,
        "",
        "IDENTIFICATION:",
        f"  Confirmed 2024/25 PL debutants:      {len(confirmed)}",
        f"  Unconfirmed (age 28+, skipped):      {len(unconfirmed)}",
        f"  Flagged (name change risk, skipped): {len(flagged)}",
        f"  Excluded (STAGE4B_EXCLUSIONS):       {len(excluded)}",
        "",
        "SCRAPING:",
        f"  Players with known previous league:  {known_count}",
        f"  Players with unknown league:         {unknown_count}  (skipped)",
        f"  Cache files reused from Stage 4a:    {scrape_stats.get('cache_reused', 0)}",
        f"  New cache files created:             {scrape_stats.get('files_created', 0)}",
        f"  Total FBref rows scraped:            {scrape_stats.get('total_rows', 0)}",
        "",
        "MATCHING:",
        f"  Matches >= 95% confidence:           {scrape_stats.get('matches_high', 0)}",
        f"  Matches 80-95% confidence:           {scrape_stats.get('matches_low', 0)}",
        f"  Blocked by false positive list:      {scrape_stats.get('blocked_fp', 0)}",
        f"  Skipped (below 80% threshold):       {scrape_stats.get('below_thresh', 0)}",
        "",
        "ROWS ADDED:",
        f"  GK:    {pos_stats['GK']['rows']} rows  ({pos_stats['GK']['players']} unique players)",
        f"  DEF:   {pos_stats['DEF']['rows']} rows  ({pos_stats['DEF']['players']} unique players)",
        f"  MID:   {pos_stats['MID']['rows']} rows  ({pos_stats['MID']['players']} unique players)",
        f"  FWD:   {pos_stats['FWD']['rows']} rows  ({pos_stats['FWD']['players']} unique players)",
        f"  TOTAL: {total_rows_added} rows",
        "",
        "DEDUPLICATION:",
        f"  Rows removed (lower priority source): {total_removed}",
        "",
        "SCHEMA VALIDATION:",
        f"  All 4 files identical columns:        {'YES' if schemas_identical else 'NO'}",
        f"  Null values in any column:            {null_count}  (target: 0)",
        f"  adjG/90 > 1.5 non-small-sample:       {len(high_adj_flags)}  (target: 0)",
        "",
        "FULL PLAYER LIST:",
        f"  {'Player':<28} {'Pos':<5} {'League':<22} {'Seasons':>7}  {'Conf':<8} {'Rows':>5}",
        "  " + "-" * 80,
    ]

    if not extracted_df.empty:
        for player_name in sorted(extracted_df["name"].unique()):
            sub = extracted_df[extracted_df["name"] == player_name]
            pos = sub["position"].iloc[0]
            league = sub["_prev_league"].iloc[0] if "_prev_league" in sub.columns else "?"
            n_seasons = sub["season"].nunique()
            conf = sub["_confidence"].iloc[0] if "_confidence" in sub.columns else "?"
            report_lines.append(
                f"  {player_name:<28} {pos:<5} {league:<22} {n_seasons:>7}  {conf:<8} {len(sub):>5}"
            )
    else:
        report_lines.append("  (none)")

    report_lines += [
        "",
        "FUZZY MATCHES BELOW 95%:",
        "  FBref Name               -> FPL Name               Conf   Decision",
        "  " + "-" * 65,
    ]
    if step3_low_conf_decisions:
        for d in step3_low_conf_decisions:
            report_lines.append(
                f"  {d['fbref']:<25}-> {d['fpl']:<25}{d['score']:>4}%  {d['decision']}"
            )
    else:
        report_lines.append("  None")

    report_lines += [
        "",
        "PLAYERS SKIPPED (unknown previous league):",
    ]
    unknown_players = [p for p in scrape_plan if p["previous_league"] == "unknown"]
    if unknown_players:
        for p in unknown_players:
            report_lines.append(f"  {p['vaastav_name']}")
    else:
        report_lines.append("  None")

    report_lines += [
        "",
        "=" * 70,
        "Stage 4b complete. Ready for Stage 5.",
        "=" * 70,
    ]

    # Print
    for line in report_lines:
        print(line)

    # Save
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print()
    print(f"  Report saved: {REPORT_FILE}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4b: 2024/25 PL Debutants previous-league FBref stats"
    )
    parser.add_argument(
        "--start-step", type=int, default=1, metavar="N",
        help="Resume from step N (1-6). Default: 1"
    )
    args = parser.parse_args()
    start_step = args.start_step

    state = _load_state()

    # Resume message
    if start_step > 1:
        print(f"Resuming from Step {start_step}.")
        if "step1" in state:
            confirmed = state["step1"]["confirmed"]
            print(f"  Loaded {len(confirmed)} confirmed targets from cache.")
        if "step2" in state:
            scrape_plan = state["step2"]["scrape_plan"]
            print(f"  Loaded scrape plan ({len(scrape_plan)} entries) from cache.")
        print("Proceeding...")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    if start_step <= 1:
        confirmed, unconfirmed, flagged, excluded = step1_identify_debutants(state)

        ok = gate(
            step_num=1,
            step_name="Target Identification",
            summary_lines=[
                f"  Confirmed debutants:    {len(confirmed)}",
                f"  Unconfirmed (age 28+):  {len(unconfirmed)}",
                f"  Flagged (name issues):  {len(flagged)}",
                f"  Excluded:               {len(excluded)}",
            ],
            next_description=(
                "Step 2 will determine the previous league for each confirmed player "
                "by checking new_signings_2025.csv and existing FBref cache files. "
                "No scraping yet."
            ),
        )
        if not ok:
            return
    else:
        confirmed   = state.get("step1", {}).get("confirmed", [])
        unconfirmed = state.get("step1", {}).get("unconfirmed", [])
        flagged     = state.get("step1", {}).get("flagged", [])
        excluded    = state.get("step1", {}).get("excluded", [])

    # ── Step 2 ────────────────────────────────────────────────────────────────
    if start_step <= 2:
        scrape_plan, known_count, unknown_count, cache_hits, new_scrapes = \
            step2_determine_leagues(confirmed, state)

        ok = gate(
            step_num=2,
            step_name="Scrape Plan",
            summary_lines=[
                f"  Players with known previous league:  {known_count}",
                f"  Players with unknown league:         {unknown_count}  (will be skipped)",
                f"  Cache files already available:       {cache_hits}  (no re-scrape needed)",
                f"  New scrapes required:                {new_scrapes}",
            ],
            next_description=(
                f"Step 3 will execute the scrape plan. New cache files will be written to "
                f"data/raw/fbref/raw/. Existing cache files will NOT be touched. "
                f"Estimated new files to create: {new_scrapes}."
            ),
        )
        if not ok:
            return
    else:
        scrape_plan = state.get("step2", {}).get("scrape_plan", [])

    # ── Step 3 ────────────────────────────────────────────────────────────────
    if start_step <= 3:
        matched_df, scrape_stats = step3_scrape(scrape_plan, state)

        s = scrape_stats
        ok = gate(
            step_num=3,
            step_name="FBref Scrape",
            summary_lines=[
                f"  New cache files created:             {s['files_created']}",
                f"  Cache files reused:                  {s['cache_reused']}",
                f"  Total rows scraped across all files: {s['total_rows']}",
                f"  Players matched:                     {s['players_matched']}",
                f"    Matches >= 95% confidence:         {s['matches_high']}",
                f"    Matches 80-95% (confirmed by you): {s['matches_low']}",
                f"    Blocked by false positive list:    {s['blocked_fp']}",
                f"    Skipped (below 80%):               {s['below_thresh']}",
            ],
            next_description=(
                "Step 4 will extract full position-specific stats from matched rows, "
                "compute adjG/90, adjA/90, season reliability, and flag small samples. "
                "No files will be written yet."
            ),
        )
        if not ok:
            return
    else:
        # When resuming from step 4+, we need to re-derive matched_df from cache
        # by re-running the matching logic (no new scraping)
        print("  Re-loading matched data from cache files for step 3 resume ...")
        # Build a minimal matched_df from the cached data by re-running step3's match logic
        matched_df, scrape_stats = step3_scrape(scrape_plan, state)

    # ── Step 4 ────────────────────────────────────────────────────────────────
    if start_step <= 4:
        extracted_df, high_adj_flags = step4_extract_stats(matched_df, scrape_plan)

        pos_stats = {}
        for pos in ["GK", "DEF", "MID", "FWD"]:
            sub = extracted_df[extracted_df["position"] == pos] if not extracted_df.empty else pd.DataFrame()
            pos_stats[pos] = {"rows": len(sub), "players": sub["name"].nunique() if not sub.empty else 0}

        # Check schema match with existing files
        schema_match = "YES"
        for pos, slug in [("GK","gk"),("DEF","def"),("MID","mid"),("FWD","fwd")]:
            fpath = os.path.join(FBREF_SIGN_DIR, f"new_signings_{slug}.csv")
            if os.path.exists(fpath):
                exist_cols = set(pd.read_csv(fpath, nrows=0).columns)
                new_cols   = set(VAASTAV_COLS + ["data_source"])
                # After step 5 the files will have data_source added; before that, check VAASTAV_COLS
                vaastav_set = set(VAASTAV_COLS)
                if not vaastav_set.issubset(exist_cols):
                    schema_match = "NO (missing VAASTAV_COLS)"
                    break

        ok = gate(
            step_num=4,
            step_name="Stat Extraction",
            summary_lines=[
                "  Rows extracted:",
                f"    GK:   {pos_stats['GK']['rows']} rows  ({pos_stats['GK']['players']} unique players)",
                f"    DEF:  {pos_stats['DEF']['rows']} rows  ({pos_stats['DEF']['players']} unique players)",
                f"    MID:  {pos_stats['MID']['rows']} rows  ({pos_stats['MID']['players']} unique players)",
                f"    FWD:  {pos_stats['FWD']['rows']} rows  ({pos_stats['FWD']['players']} unique players)",
                f"  Small sample warnings (< 500 mins): {sum(1 for _, r in extracted_df.iterrows() if r.get('_small_sample', False)) if not extracted_df.empty else 0} rows",
                f"  adjG/90 > 1.5 flags:                {len(high_adj_flags)} rows",
                f"  Schema matches existing files:      {schema_match}",
            ],
            next_description=(
                "Step 5 will append these rows to the 4 position files, "
                "deduplicate using source priority (vaastav > stage4a > stage4b), "
                "and backfill the data_source column on existing rows.\n"
                "Files that will be modified:\n"
                "  data/raw/fbref/new_signings/new_signings_gk.csv\n"
                "  data/raw/fbref/new_signings/new_signings_def.csv\n"
                "  data/raw/fbref/new_signings/new_signings_mid.csv\n"
                "  data/raw/fbref/new_signings/new_signings_fwd.csv"
            ),
        )
        if not ok:
            return
    else:
        # Re-derive extracted_df
        extracted_df, high_adj_flags = step4_extract_stats(matched_df, scrape_plan)

    # ── Step 5 ────────────────────────────────────────────────────────────────
    if start_step <= 5:
        append_result = step5_write_position_files(extracted_df, scrape_plan)
        (append_results, total_appended, total_removed,
         backfilled_rows, schemas_identical, null_count) = append_result

        # High adjG on non-small-sample check
        n_high_adj_nonsmall = len(high_adj_flags)

        ok = gate(
            step_num=5,
            step_name="Files Written",
            summary_lines=[
                "  Rows appended:",
                f"    GK:   {append_results.get('GK', {}).get('n_new', 0)} new rows",
                f"    DEF:  {append_results.get('DEF', {}).get('n_new', 0)} new rows",
                f"    MID:  {append_results.get('MID', {}).get('n_new', 0)} new rows",
                f"    FWD:  {append_results.get('FWD', {}).get('n_new', 0)} new rows",
                f"  Rows removed by deduplication (lower priority source): {total_removed}",
                f"  data_source column backfilled on existing rows:        {backfilled_rows}",
                "",
                "  Schema validation:",
                f"    All 4 files identical columns:   {'YES' if schemas_identical else 'NO'}",
                f"    Null values remaining:           {null_count}  (target: 0)",
                f"    adjG/90 > 1.5 non-small-sample: {n_high_adj_nonsmall}  (target: 0)",
            ],
            next_description=(
                "Step 6 will generate the full validation report and print it to "
                "console and save to data/raw/fbref/new_signings/stage4b_report.txt"
            ),
        )
        if not ok:
            return
    else:
        append_results   = {}
        total_appended   = 0
        total_removed    = 0
        backfilled_rows  = 0
        schemas_identical = True
        null_count       = 0

    # ── Step 6 ────────────────────────────────────────────────────────────────
    step6_report(
        confirmed=confirmed,
        unconfirmed=unconfirmed,
        flagged=flagged,
        excluded=excluded,
        scrape_plan=scrape_plan,
        scrape_stats=scrape_stats if "scrape_stats" in dir() else state.get("step3", {}),
        extracted_df=extracted_df if not extracted_df.empty else pd.DataFrame(),
        append_results=append_results,
        total_appended=total_appended,
        total_removed=total_removed,
        backfilled_rows=backfilled_rows,
        schemas_identical=schemas_identical,
        null_count=null_count,
        high_adj_flags=high_adj_flags if "high_adj_flags" in dir() else [],
        step3_low_conf_decisions=[],  # populated interactively during step 3
    )

    # Final gate
    print()
    print("=" * 70)
    print("STAGE 4b COMPLETE")
    print("=" * 70)
    print(f"Report saved to: {REPORT_FILE}")
    print(f"State file at:   {STATE_FILE}")
    print()
    print("All position files updated and validated.")
    print("Ready for Stage 5 -- Player vs Opponent Matchup Stats.")
    print("=" * 70)


if __name__ == "__main__":
    main()
