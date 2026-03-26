#!/usr/bin/env python3
"""
Stage 4a: New Premier League Signings Data

Identifies FPL players with zero vaastav history, scrapes their previous-league
stats from FBref (via soccerdata), applies league difficulty multipliers, and
builds position-specific files structurally identical to the vaastav position files.

Usage: python pipeline/new_signings_stage4a.py
"""

import io
import os
import re
import sys
import time
import unicodedata
import warnings
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
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    _missing.append("requests beautifulsoup4")
try:
    from fuzzywuzzy import fuzz, process as fzprocess
except ImportError:
    _missing.append("fuzzywuzzy python-Levenshtein")
try:
    from seleniumbase import SB
    _HAS_SELENIUM = True
except ImportError:
    _HAS_SELENIUM = False
    _missing.append("seleniumbase")

if _missing:
    print("[ERROR] Missing dependencies. Install with:")
    print(f"  pip install {' '.join(_missing)}")
    sys.exit(1)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
TRANSFERS_DIR  = os.path.join(RAW_DIR, "transfers")
HTML_CACHE_DIR = os.path.join(TRANSFERS_DIR, "raw_html_cache")
FBREF_RAW_DIR  = os.path.join(RAW_DIR, "fbref", "raw")
FBREF_SIGN_DIR = os.path.join(RAW_DIR, "fbref", "new_signings")
VAASTAV_DIR    = os.path.join(RAW_DIR, "vaastav")
FPL_API_DIR    = os.path.join(RAW_DIR, "fpl_api")

for _d in [TRANSFERS_DIR, HTML_CACHE_DIR, FBREF_RAW_DIR, FBREF_SIGN_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
FUZZY_THRESHOLD = 80
SEASONS         = ["2022-23", "2023-24", "2024-25"]

TM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com",
}

TM_BASE = (
    "https://www.transfermarkt.com/premier-league/neuimland/wettbewerb/GB1"
    "/saison_id/2025/land_id//ausrichtung//spielerposition_id//altersklasse//leihe//w_s//plus/1"
)
TM_URLS = [
    TM_BASE,
    TM_BASE + "/page/2",
    TM_BASE + "/page/3",
    TM_BASE + "/page/4",
]

# Transfermarkt league name -> standardized name
LEAGUE_MAP = {
    "Bundesliga":              "Bundesliga",
    "1. Bundesliga":           "Bundesliga",
    "LaLiga":                  "La Liga",
    "La Liga":                 "La Liga",
    "Primera Division":        "La Liga",
    "LaLiga EA Sports":        "La Liga",
    "Serie A":                 "Serie A",
    "Ligue 1":                 "Ligue 1",
    "Ligue 1 Uber Eats":       "Ligue 1",
    "Eredivisie":              "Eredivisie",
    "Primeira Liga":           "Primeira Liga",
    "Liga Portugal":           "Primeira Liga",
    "Liga Portugal Betclic":   "Primeira Liga",
    "Scottish Premiership":    "Scottish Premiership",
    "Scottish Premier League": "Scottish Premiership",
    "Championship":            "Championship",
    "EFL Championship":        "Championship",
    "Belgian Pro League":      "Belgian Pro League",
    "Jupiler Pro League":      "Belgian Pro League",
    "Premier League":          "Premier League",
    "Super League":            "Other",
    "Süper Lig":               "Other",
    "Ekstraklasa":             "Other",
    "MLS":                     "Other",
}

LEAGUE_MULTIPLIERS = {
    "Bundesliga":           0.89,
    "La Liga":              0.92,
    "Serie A":              0.88,
    "Ligue 1":              0.82,
    "Eredivisie":           0.75,
    "Primeira Liga":        0.78,
    "Scottish Premiership": 0.65,
    "Championship":         0.72,
    "Belgian Pro League":   0.74,
}

# FBref league numeric IDs (used in URL construction)
FBREF_LEAGUE_IDS = {
    "Bundesliga":           20,
    "La Liga":              12,
    "Serie A":              11,
    "Ligue 1":              13,
    "Eredivisie":           23,
    "Primeira Liga":        32,
    "Scottish Premiership": 40,
    "Championship":         10,
    "Belgian Pro League":   37,
}

# FBref URL league name slugs
FBREF_LEAGUE_SLUGS = {
    "Bundesliga":           "Bundesliga",
    "La Liga":              "La-Liga",
    "Serie A":              "Serie-A",
    "Ligue 1":              "Ligue-1",
    "Eredivisie":           "Eredivisie",
    "Primeira Liga":        "Primeira-Liga",
    "Scottish Premiership": "Scottish-Premiership",
    "Championship":         "Championship",
    "Belgian Pro League":   "Belgian-First-Division-A",
}

# Manual TM name corrections for unmatched players (name as it appears on TM -> FPL name fragment)
TM_MANUAL_FIXES = {
    "Estevaao":         "Estevao",        # Estêvão -> Estevao Almeida
    "Estavao":          "Estevao",
    "Rayan":            "Rayan Cherki",
    "Igor Jesus":       "Igor Jesus Maciel da Cruz",
    "Jocelin Ta Bi":    "Djiamgone Jocelin Ta Bi",
    "Yeremy Pino":      "Yeremy Pino",
    "Fer Lopez":        "Fer Lopez Gonzalez",
    "Alex Jimenez":     "Alex Jimenez",
    "Diego Leon":       "Diego Leon",
    "Kendry Paez":      "Kendry Paez",
}
# TM names that are fringe/unresolvable — skip FBref for these
TM_UNRESOLVED = {
    "Kevin", "Pablo", "Souza", "Alysson", "Cuiabano",
    "John Victor", "Jair Cunha", "Antonito Cordero", "Do-young Yoon",
}

# FPL full names that fuzzy-matched a WRONG vaastav entry — force them into the new-to-PL list
VAASTAV_FALSE_POSITIVES = {
    "Andrew Moran",
    "Mamadou Sarr",
    "Divine Mukasa",
    "Charlie Crew",
    "Anthony Patterson",
    "Steven Benda",
    "Viktor Gyokeres",   # encoding-safe version; also handled via normalize_name
}

# Manual previous-league overrides for players TM missed (fpl_name -> standardized league)
MANUAL_PLAYER_LEAGUES = {
    "Viktor Gyokeres": "Primeira Liga",   # Sporting CP; TM didn't list him on new-arrivals page
}

# FBref rows that are WRONG players matched to a new signing — block them entirely.
# Format: (fbref_player_name, fpl_target_name) — normalized comparison used at runtime.
FBREF_FALSE_POSITIVES = {
    ("Mamadou Sylla",     "Mamadou Sarr"),       # different player — Guinean striker
    ("Mamadou Sakho",     "Mamadou Sarr"),       # veteran ex-Liverpool CB
    ("Mamadou Sangare",   "Mamadou Sarr"),       # Malian CDM
    ("Mamadou Traore",    "Mamadou Sarr"),       # Ivorian winger (accent-stripped)
    ("Mouhamadou Sarr",   "Mamadou Sarr"),       # different Senegalese player
    ("Anderson",          "Joe Anderson"),       # generic single-name Brazilian
    ("Anderson Jesus",    "Joe Anderson"),       # Brazilian striker
    ("Nito Gomes",        "Toti Gomes"),         # different Portuguese player
    ("Vitor Gomes",       "Toti Gomes"),         # different Portuguese player (accent-stripped)
    ("Matheus Reis",      "Remi Matthews"),      # Brazilian LB, not English GK
    ("Benjamin Leroy",    "Benjamin Lecomte"),   # different Belgian player
    ("Steven Baseya",     "Steven Benda"),       # Belgian winger, not Swiss GK
    ("Charlie Cresswell", "Charlie Crew"),       # Leeds CB, not young Welsh MID
    ("Yacine Adli",       "Amine Adli"),         # Algerian mid, not Moroccan winger
    ("Giovanni Simeone",  "Giovanni Leoni"),     # Argentine striker, not Italian CB
}

# Correct FBref names for players whose automatic match failed.
# Key = FPL name, Value = exact FBref player name to look for in cached CSVs.
# Populated manually after inspecting FBref data.
FBREF_NAME_OVERRIDES = {
    "Mamadou Sarr":    "Mamadou Sarr",
    "Benjamin Lecomte": "Benjamin Lecomte",
    "Amine Adli":      "Amine Adli",
    "Giovanni Leoni":  "Giovanni Leoni",
}

# Players confirmed to have no FBref top-5 league data — get zero-stat rows with data_confidence=low.
SKIP_FBREF = {
    "Joe Anderson",   # Sunderland youth, Championship only
    "Toti Gomes",     # may have PL history; no top-5 non-PL FBref data
    "Remi Matthews",  # English GK, Championship only
    "Steven Benda",   # Swiss GK, Championship only
    "Charlie Crew",   # 17yo Welsh youth player
}

# FBref raw position -> FPL position (take first token)
FBREF_POS_MAP = {
    "GK":    "GK",
    "DF":    "DEF",
    "MF":    "MID",
    "FW":    "FWD",
}

# Vaastav column order (exactly matching historical_*.csv) + is_new_to_pl
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


# =============================================================================
# HELPERS
# =============================================================================

def find_col(df, candidates):
    """Return first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_div(a, b, default=0.0):
    """Safe division, returns default when b is 0 or None."""
    try:
        if b == 0 or pd.isna(b):
            return default
        return a / b
    except Exception:
        return default


def flatten_columns(df):
    """
    Flatten a MultiIndex column DataFrame by joining levels with '_'.
    Strips trailing/leading underscores and handles non-tuple columns.
    """
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(c).strip() for c in col if str(c).strip() and str(c).strip() != "nan"]
            new_cols.append("_".join(parts))
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df


def normalize_name(name):
    """Strip accents, lowercase, remove punctuation (keep spaces/hyphens)."""
    nfd = unicodedata.normalize("NFD", str(name))
    ascii_str = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    cleaned = re.sub(r"[^\w\s-]", "", ascii_str.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def fuzzy_match_name(name, name_list, threshold=FUZZY_THRESHOLD):
    """Return (matched_name, score) or (None, 0) if no match above threshold."""
    if not name_list:
        return None, 0
    result = fzprocess.extractOne(name, name_list, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0], result[1]
    return None, 0


def _match_name_multi_strategy(fpl_full, fpl_first, fpl_second,
                                vaastav_names, vaastav_names_norm,
                                norm_to_orig, threshold=FUZZY_THRESHOLD):
    """
    Try multiple strategies to match an FPL player to a vaastav name.
    Returns (original_vaastav_name, score, strategy) or (None, 0, None).

    Strategies (in order, with separate thresholds to avoid false positives):
      1. Normalized full name  -> normalized vaastav names  (threshold)
      2. second_name normalized -> normalized vaastav names  (threshold+10, avoids "Alex X" -> "Alex Y")
      3. first + first-4-of-last -> normalized vaastav       (threshold+12, very strict)
    """
    norm_full  = normalize_name(fpl_full)
    norm_last  = normalize_name(fpl_second)
    last_first4 = norm_last.split()[0][:4] if norm_last.split() else ""
    first_plus4 = (normalize_name(fpl_first) + " " + last_first4).strip() if last_first4 else ""

    strategies = [
        (norm_full,   "full",    threshold),
        (norm_last,   "last",    min(threshold + 10, 95)),
        (first_plus4, "first+4", min(threshold + 12, 95)),
    ]

    for query, label, thresh in strategies:
        if not query.strip():
            continue
        res = fzprocess.extractOne(query, vaastav_names_norm, scorer=fuzz.token_sort_ratio)
        if res and res[1] >= thresh:
            orig = norm_to_orig.get(res[0], res[0])
            return orig, res[1], label

    return None, 0, None


def season_to_year(season_str):
    """'2022-23' -> 2023"""
    try:
        return int(season_str.split("-")[1]) + 2000
    except Exception:
        return 0


# =============================================================================
# STEP 1 — Identify new-to-PL players
# =============================================================================

def step1_identify_new_signings():
    print("\n" + "=" * 60)
    print("STEP 1: Identifying new-to-PL players")
    print("=" * 60)

    fpl = pd.read_csv(os.path.join(FPL_API_DIR, "players_raw.csv"))
    fpl["full_name"] = fpl["first_name"].str.strip() + " " + fpl["second_name"].str.strip()

    vaastav = pd.read_csv(os.path.join(VAASTAV_DIR, "historical_gw_data.csv"))
    vaastav_names = vaastav["name"].unique().tolist()

    # Build normalized vaastav index for accent-insensitive matching
    vaastav_names_norm = [normalize_name(n) for n in vaastav_names]
    norm_to_orig = {normalize_name(n): n for n in vaastav_names}

    print(f"  FPL players loaded:            {len(fpl)}")
    print(f"  Vaastav unique player names:   {len(vaastav_names)}")
    print(f"  Matching threshold:            {FUZZY_THRESHOLD}% (flag <90% for review)")

    new_rows  = []
    has_hist  = 0
    low_conf  = []   # matches 80-89%
    mid_conf  = []   # matches 90-94%

    for _, player in fpl.iterrows():
        fname  = player["first_name"].strip()
        sname  = player["second_name"].strip()
        full   = player["full_name"]

        orig_match, score, strategy = _match_name_multi_strategy(
            full, fname, sname,
            vaastav_names, vaastav_names_norm, norm_to_orig,
            threshold=FUZZY_THRESHOLD,
        )

        # Force false positives (wrongly matched to wrong vaastav name) into new list
        # Match on normalized name so diacritics (Gyökeres) are handled correctly
        _fp_norms = {normalize_name(n) for n in VAASTAV_FALSE_POSITIVES}
        if orig_match and normalize_name(full) in _fp_norms:
            orig_match = None

        if orig_match:
            has_hist += 1
            if score < 90:
                low_conf.append((full, orig_match, score, strategy))
            elif score < 95:
                mid_conf.append((full, orig_match, score, strategy))
        else:
            new_rows.append({
                "fpl_id":           player["id"],
                "fpl_name":         full,
                "fpl_team":         player["team_name"],
                "fpl_position":     player["position"],
                "fpl_price":        player["price"],
                "vaastav_match":    None,
                "match_confidence": 0,
            })

    df = pd.DataFrame(new_rows)

    if low_conf:
        print(f"\n  [MANUAL REVIEW REQUIRED] {len(low_conf)} matches at 80-89% — may be wrong:")
        for fpl_n, vas_n, sc, strat in sorted(low_conf, key=lambda x: x[2]):
            print(f"    {sc:3d}% [{strat}]  '{fpl_n}'  ->  '{vas_n}'")

    if mid_conf:
        print(f"\n  [REVIEW] {len(mid_conf)} matches at 90-94%:")
        for fpl_n, vas_n, sc, strat in sorted(mid_conf, key=lambda x: x[2]):
            print(f"    {sc:3d}% [{strat}]  '{fpl_n}'  ->  '{vas_n}'")

    print(f"\n  Players with vaastav history:            {has_hist}")
    print(f"  Players with NO vaastav history (new):   {len(df)}")
    print()
    print(f"  {'ID':<6} {'Name':<32} {'Team':<22} {'Pos':<5} {'Price'}")
    print(f"  {'-'*6} {'-'*32} {'-'*22} {'-'*5} {'-'*5}")
    for _, r in df.iterrows():
        print(f"  {int(r['fpl_id']):<6} {r['fpl_name']:<32} {r['fpl_team']:<22} {r['fpl_position']:<5} {r['fpl_price']:.1f}")

    out = os.path.join(TRANSFERS_DIR, "new_signings_2025.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved: new_signings_2025.csv ({len(df)} rows)")
    return df


# =============================================================================
# STEP 2 — Scrape Transfermarkt for previous club / league
# =============================================================================

def _fetch_tm_page(url, page_num):
    """Fetch a Transfermarkt page with caching. Returns HTML string."""
    cache_path = os.path.join(HTML_CACHE_DIR, f"tm_page_{page_num}.html")
    if os.path.exists(cache_path):
        print(f"    Page {page_num}: loaded from cache")
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    print(f"    Page {page_num}: fetching {url}")
    try:
        r = requests.get(url, headers=TM_HEADERS, timeout=20)
        r.raise_for_status()
        html = r.text
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"    Page {page_num}: cached ({len(html):,} bytes)")
        return html
    except Exception as e:
        print(f"    Page {page_num}: [ERROR] {e}")
        return ""


def _parse_tm_table(html, page_num):
    """
    Parse a Transfermarkt neuimland (new arrivals) page.
    Returns list of dicts with: player_name, position, age, previous_club,
    previous_league, new_pl_club, transfer_type.

    Verified TM column layout (18 tds per player row):
      td[0]  number
      td[1]  player photo + name anchor
      td[2]  player photo
      td[3]  hauptlink  = player name text
      td[4]  position text
      td[5]  age (zentriert, 2-digit integer)
      td[6]  date
      td[7]  market value
      td[8]  nationality flags
      td[9]  from-club block (anchors: ['', 'Club', 'League'])
      td[10] from-club photo
      td[11] hauptlink = from-club short name
      td[12] from-league  (anchor text = e.g. 'Bundesliga')
      td[13] to-club block  (anchors: ['', 'PLClub', 'Premier League'])
      td[14] to-club photo
      td[15] hauptlink = to-club short name
      td[16] to-league  (text = 'Premier League')
      td[17] fee (rechts hauptlink)
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="items")
    if table is None:
        print(f"    Page {page_num}: [WARN] Could not find items table")
        return []

    records = []
    for row in table.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 12:
            continue  # header or short rows

        try:
            # ── Player name (td[3], hauptlink) ──────────────────────────────
            player_name = ""
            # Find the first hauptlink td with a non-empty anchor
            for td in tds:
                if "hauptlink" in td.get("class", []):
                    a = td.find("a")
                    if a and a.get_text(strip=True):
                        player_name = a.get_text(strip=True)
                        break
            if not player_name:
                continue

            # ── Position: td[4] text (index after player hauptlink) ──────────
            position = ""
            try:
                hl_idx = next(
                    i for i, td in enumerate(tds)
                    if "hauptlink" in td.get("class", [])
                    and td.find("a")
                    and td.find("a").get_text(strip=True) == player_name
                )
                if hl_idx + 1 < len(tds):
                    position = tds[hl_idx + 1].get_text(strip=True)
            except StopIteration:
                pass

            # ── Age: first zentriert td whose text is a valid age integer ────
            age = 0
            for td in tds:
                txt = td.get_text(strip=True)
                if "zentriert" in " ".join(td.get("class", [])) and txt.isdigit():
                    v = int(txt)
                    if 15 <= v <= 45:
                        age = v
                        break

            # ── Use the full 18-td structure for clubs / league / fee ────────
            if len(tds) >= 18:
                # From-club (short name) at td[11], from-league at td[12]
                from_club_td    = tds[11]
                from_league_td  = tds[12]
                to_club_td      = tds[15]
                fee_td          = tds[17]

                from_a = from_club_td.find("a")
                previous_club = from_a.get_text(strip=True) if from_a else from_club_td.get_text(strip=True)

                # From-league: get first anchor text in td[12]
                anchors_12 = [a.get_text(strip=True) for a in from_league_td.find_all("a") if a.get_text(strip=True)]
                previous_league = anchors_12[0] if anchors_12 else from_league_td.get_text(strip=True)

                to_a = to_club_td.find("a")
                new_pl_club = to_a.get_text(strip=True) if to_a else to_club_td.get_text(strip=True)

                fee_text = fee_td.get_text(strip=True)
            else:
                # Fallback: extract from hauptlinks list
                hauptlinks = [td for td in tds if "hauptlink" in td.get("class", [])]
                previous_club = ""
                previous_league = ""
                new_pl_club = ""
                fee_text = ""
                if len(hauptlinks) >= 2:
                    from_a = hauptlinks[1].find("a")
                    previous_club = from_a.get_text(strip=True) if from_a else ""
                if len(hauptlinks) >= 3:
                    to_a = hauptlinks[2].find("a")
                    new_pl_club = to_a.get_text(strip=True) if to_a else ""
                # Fee: last cell
                fee_text = tds[-1].get_text(strip=True)

            fee_lower     = fee_text.lower()
            transfer_type = "loan" if any(w in fee_lower for w in ["loan", "leihe", "leih"]) else "permanent"

            records.append({
                "player_name":     player_name,
                "position":        position,
                "age":             age,
                "previous_club":   previous_club,
                "previous_league": previous_league,
                "new_pl_club":     new_pl_club,
                "transfer_type":   transfer_type,
                "fee_text":        fee_text,
            })

        except Exception:
            continue  # Skip malformed rows silently

    return records


def step2_scrape_transfermarkt(new_signings_df):
    print("\n" + "=" * 60)
    print("STEP 2: Scraping Transfermarkt for previous club/league")
    print("=" * 60)

    all_tm = []
    for page_num, url in enumerate(TM_URLS, start=1):
        html = _fetch_tm_page(url, page_num)
        records = _parse_tm_table(html, page_num)
        print(f"    Page {page_num}: parsed {len(records)} player rows")
        all_tm.extend(records)
        if page_num < len(TM_URLS):
            time.sleep(3)

    if not all_tm:
        print("  [WARN] No Transfermarkt data found. HTML cache may be empty or page structure changed.")
        print("  Continuing without TM data — previous_league will be unknown.")
        # Add blank columns and return
        for col in ["previous_club", "previous_league", "previous_league_standardized",
                    "new_pl_club_tm", "transfer_type", "tm_match", "tm_confidence"]:
            new_signings_df[col] = None
        return new_signings_df

    tm_df = pd.DataFrame(all_tm)
    print(f"\n  Total TM rows scraped: {len(tm_df)}")

    # Standardize previous_league
    tm_df["previous_league_standardized"] = (
        tm_df["previous_league"]
        .str.strip()
        .map(LEAGUE_MAP)
        .fillna(tm_df["previous_league"].str.strip())
    )

    # ── Fuzzy-join TM players -> new_signings list ────────────────────────────
    tm_names  = tm_df["player_name"].tolist()
    ns_names  = new_signings_df["fpl_name"].tolist()

    tm_match_col = []
    tm_conf_col  = []

    for tm_name in tm_names:
        # Skip known unresolvable fringe names
        norm_tm = normalize_name(tm_name)
        if norm_tm in {normalize_name(u) for u in TM_UNRESOLVED}:
            tm_match_col.append(None)
            tm_conf_col.append(0)
            continue
        # Apply manual fixes first
        resolved = tm_name
        for raw_fix, fpl_fragment in TM_MANUAL_FIXES.items():
            if normalize_name(tm_name) == normalize_name(raw_fix):
                resolved = fpl_fragment
                break
        # Fuzzy match (using normalized names for accent-insensitive matching)
        norm_ns = [normalize_name(n) for n in ns_names]
        norm_to_ns = {normalize_name(n): n for n in ns_names}
        norm_resolved = normalize_name(resolved)
        res = fzprocess.extractOne(norm_resolved, norm_ns, scorer=fuzz.token_sort_ratio)
        if res and res[1] >= FUZZY_THRESHOLD:
            orig = norm_to_ns.get(res[0], res[0])
            tm_match_col.append(orig)
            tm_conf_col.append(res[1])
        else:
            tm_match_col.append(None)
            tm_conf_col.append(0)

    tm_df["fpl_match"]      = tm_match_col
    tm_df["tm_confidence"]  = tm_conf_col
    tm_df["matched_to_fpl"] = tm_df["fpl_match"].notna()

    # Keep only TM rows that matched a new signing
    tm_matched = tm_df[tm_df["matched_to_fpl"]].copy()
    print(f"  TM rows matched to new signings: {len(tm_matched)}")
    unmatched_tm = tm_df[~tm_df["matched_to_fpl"]]["player_name"].tolist()
    if unmatched_tm:
        print(f"  TM rows NOT matched to FPL list ({len(unmatched_tm)}): {unmatched_tm[:20]}")

    # Print low-confidence TM matches (deduplicated)
    low_tm = tm_matched[tm_matched["tm_confidence"] < 95].drop_duplicates("fpl_match")
    if len(low_tm):
        print(f"\n  [REVIEW] {len(low_tm)} TM->FPL matches below 95%:")
        for _, r in low_tm.iterrows():
            print(f"    {r['tm_confidence']:3d}%  '{r['player_name']}'  ->  '{r['fpl_match']}'")

    # ── Merge back into new_signings_df ──────────────────────────────────────
    tm_slim = tm_matched[[
        "fpl_match", "previous_club", "previous_league",
        "previous_league_standardized", "new_pl_club", "transfer_type", "tm_confidence"
    ]].rename(columns={
        "fpl_match":    "fpl_name",
        "new_pl_club":  "new_pl_club_tm",
    })

    # Drop duplicate fpl_name matches (keep best confidence)
    tm_slim = tm_slim.sort_values("tm_confidence", ascending=False).drop_duplicates("fpl_name")

    merged = new_signings_df.merge(tm_slim, on="fpl_name", how="left")

    # League breakdown
    league_counts = (
        merged["previous_league_standardized"]
        .value_counts()
        .rename_axis("league")
        .reset_index(name="count")
    )
    print("\n  Breakdown by previous league:")
    for _, r in league_counts.iterrows():
        print(f"    {r['league']:<30} {r['count']}")

    # Transfer type breakdown
    perm  = merged["transfer_type"].eq("permanent").sum()
    loan  = merged["transfer_type"].eq("loan").sum()
    unkn  = merged["transfer_type"].isna().sum()
    print(f"\n  Transfer type: {int(perm)} permanent / {int(loan)} loan / {int(unkn)} unknown")

    # Apply manual league overrides for players TM missed
    for fpl_n, league_std in MANUAL_PLAYER_LEAGUES.items():
        norm_target = normalize_name(fpl_n)
        mask = merged["fpl_name"].apply(lambda x: normalize_name(str(x)) == norm_target)
        if mask.any():
            merged.loc[mask, "previous_league_standardized"] = league_std
            print(f"  [MANUAL] Set {fpl_n} previous_league -> {league_std}")

    # Save updated CSV
    out = os.path.join(TRANSFERS_DIR, "new_signings_2025.csv")
    merged.to_csv(out, index=False)
    print(f"\n  Saved: new_signings_2025.csv ({len(merged)} rows, with TM data)")
    return merged, len(tm_df), unmatched_tm


# =============================================================================
# STEP 3 — Scrape FBref via Selenium (Cloudflare bypass)
# =============================================================================

def _get_fbref_cache_path(league_std, season):
    slug = league_std.lower().replace(" ", "_")
    return os.path.join(FBREF_RAW_DIR, f"{slug}_{season}.csv")


def _build_fbref_url(league_std, season, stat_type="standard"):
    """
    Build the correct FBref URL for a league+season+stat_type.
    URL format: /en/comps/{id}/{year1}-{year2}/{table}/{year1}-{year2}-{slug}-Stats

    stat_type: 'standard' | 'keeper'
    """
    fbref_id  = FBREF_LEAGUE_IDS.get(league_std)
    slug      = FBREF_LEAGUE_SLUGS.get(league_std)
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
    """
    Parse a FBref stats table from page HTML using pandas read_html.
    FBref uses a 2-row thead: category (over_header) + stat names.
    Returns a flat DataFrame with '_'-joined column names, or None.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        table_tag = soup.find("table", id=table_id)
        if table_tag is None:
            return None

        # pandas read_html with 2-level header
        df_list = pd.read_html(str(table_tag), header=[0, 1], na_values=["", "N/A"])
        if not df_list:
            return None

        df = df_list[0]
        df = flatten_columns(df)

        # Use pattern-based player column detection (handles Unnamed: N_level_0_Player)
        # Must be done after flatten_columns
        player_col = _find_player_col(df)
        if player_col:
            # Filter out repeated header rows ("Player") and squad-total rows
            bad_vals = {"player", "squad total", "", "nan"}
            df = df[~df[player_col].astype(str).str.strip().str.lower().isin(bad_vals)]
            df = df[df[player_col].notna()]

        return df.reset_index(drop=True)

    except Exception as e:
        return None


def _find_player_col(df):
    """Find the player name column. FBref uses 'Unnamed: N_level_0_Player' pattern."""
    for c in df.columns:
        if c.endswith("_Player") or c == "Player":
            return c
    return None


def _find_squad_col(df):
    """Find the squad/team column."""
    for c in df.columns:
        if c.endswith("_Squad") or c == "Squad":
            return c
    return None


def _find_pos_col(df):
    """Find the position column."""
    for c in df.columns:
        if c.endswith("_Pos") or c == "Pos":
            return c
    return None


def _fetch_fbref_table(sb, url, table_id, label):
    """
    Open a FBref URL and parse the specified table. Sets a 1920px viewport
    to ensure all columns (including Expected xG) are rendered.
    Returns DataFrame or None.
    """
    try:
        sb.open(url)
        # Wide viewport ensures FBref renders all stat columns (xG etc.)
        sb.execute_script("window.resizeTo(1920, 1080);")
        time.sleep(5)
        html = sb.get_page_source()
        df = _parse_fbref_page_html(html, table_id)
        if df is not None:
            print(f"      {label}: {len(df)} rows | cols: {len(df.columns)}")
        else:
            print(f"      [WARN] {label}: table '{table_id}' not found at {url}")
        return df
    except Exception as e:
        print(f"      [ERROR] {label}: {e}")
        return None


def _merge_fbref_tables(base_df, extra_df, extra_keep_cols_filter):
    """
    Left-merge extra_df into base_df on Player+Squad columns.
    Only keeps columns from extra_df matching extra_keep_cols_filter keywords.
    """
    if extra_df is None or base_df is None:
        return base_df
    bp  = _find_player_col(base_df)
    bs  = _find_squad_col(base_df)
    ep  = _find_player_col(extra_df)
    es  = _find_squad_col(extra_df)
    if not bp or not ep:
        return base_df
    keep = [
        c for c in extra_df.columns
        if any(k in c for k in extra_keep_cols_filter)
        and c not in [ep, es]
    ]
    if not keep:
        return base_df
    join_l = [bp] + ([bs] if bs and es else [])
    join_r = [ep] + ([es] if bs and es else [])
    sel = [ep] + ([es] if es else []) + keep
    return base_df.merge(extra_df[sel], left_on=join_l, right_on=join_r,
                         how="left", suffixes=("", "_x"))


def _scrape_fbref_league_season_selenium(sb, league_std, season):
    """
    Fetch FBref standard + shooting + keeper stats for a league+season.
    Uses 3 separate page fetches to get all columns including xG.
    Returns merged DataFrame or None.
    """
    cache_path = _get_fbref_cache_path(league_std, season)
    if os.path.exists(cache_path):
        print(f"      {league_std} {season}: loaded from cache")
        return pd.read_csv(cache_path)

    fbref_id = FBREF_LEAGUE_IDS.get(league_std)
    if not fbref_id:
        print(f"      {league_std}: [WARN] No FBref ID — skipping")
        return None

    print(f"      {league_std} {season}: fetching from FBref...")

    # ── Standard stats ───────────────────────────────────────────────────────
    std_url = _build_fbref_url(league_std, season, "standard")
    std_df  = _fetch_fbref_table(sb, std_url, "stats_standard",
                                 f"Standard ({league_std} {season})")
    if std_df is None:
        return None
    std_df["league_tag"] = league_std
    std_df["season_tag"] = season
    time.sleep(4)

    # ── Shooting stats (provides xG, Sh/90, SoT/90) ─────────────────────────
    shoot_url = _build_fbref_url(league_std, season, "shooting")
    shoot_df  = _fetch_fbref_table(sb, shoot_url, "stats_shooting",
                                   f"Shooting ({league_std} {season})")
    std_df = _merge_fbref_tables(std_df, shoot_df,
                                 ["xG", "npxG", "Sh", "SoT", "Dist"])
    time.sleep(4)

    # ── Keeper stats ─────────────────────────────────────────────────────────
    keep_url = _build_fbref_url(league_std, season, "keeper")
    keep_df  = _fetch_fbref_table(sb, keep_url, "stats_keeper",
                                  f"Keeper ({league_std} {season})")
    std_df = _merge_fbref_tables(std_df, keep_df,
                                 ["GA", "Save", "SoTA", "CS", "Saves", "PSxG"])
    time.sleep(4)

    std_df.to_csv(cache_path, index=False)
    print(f"      Saved cache: {cache_path} ({len(std_df)} rows, {len(std_df.columns)} cols)")
    return std_df


def step3_scrape_fbref(new_signings_df):
    print("\n" + "=" * 60)
    print("STEP 3: Scraping FBref (Selenium) for last 3 seasons")
    print("=" * 60)

    if not _HAS_SELENIUM:
        print("  [ERROR] seleniumbase not installed. Run: pip install seleniumbase")
        return pd.DataFrame()

    # Determine which leagues actually appear in the data
    if "previous_league_standardized" in new_signings_df.columns:
        leagues_needed = (
            new_signings_df["previous_league_standardized"]
            .dropna()
            .unique()
            .tolist()
        )
    else:
        leagues_needed = list(FBREF_LEAGUE_IDS.keys())

    leagues_to_scrape = [lg for lg in leagues_needed if lg in FBREF_LEAGUE_IDS]
    leagues_skipped   = [lg for lg in leagues_needed if lg not in FBREF_LEAGUE_IDS]

    # Also check for already-cached leagues (no browser needed for those)
    leagues_need_browser = []
    for lg in leagues_to_scrape:
        needs_fetch = any(
            not os.path.exists(_get_fbref_cache_path(lg, s))
            for s in SEASONS
        )
        if needs_fetch:
            leagues_need_browser.append(lg)

    print(f"  Leagues to scrape:          {leagues_to_scrape}")
    if leagues_skipped:
        print(f"  Leagues skipped (no ID):    {leagues_skipped}")
    print(f"  Leagues needing browser:    {leagues_need_browser}")

    all_frames = []
    fetch_failures = []

    # ── Load cached leagues first (no browser needed) ──────────────────────
    for league in leagues_to_scrape:
        for season in SEASONS:
            cache = _get_fbref_cache_path(league, season)
            if os.path.exists(cache):
                df = pd.read_csv(cache)
                df["league_tag"] = league
                df["season_tag"] = season
                all_frames.append(df)
                print(f"    {league} {season}: loaded from cache ({len(df)} rows)")

    # ── Open browser once for all uncached league+seasons ──────────────────
    if leagues_need_browser:
        print(f"\n  Opening browser (UC mode) for {len(leagues_need_browser)} league(s)...")
        try:
            with SB(uc=True, headless=True) as sb:
                for league in leagues_need_browser:
                    print(f"\n  League: {league}")
                    for season in SEASONS:
                        if os.path.exists(_get_fbref_cache_path(league, season)):
                            continue  # already handled above
                        df = _scrape_fbref_league_season_selenium(sb, league, season)
                        if df is not None and len(df) > 0:
                            df["league_tag"] = league
                            df["season_tag"] = season
                            all_frames.append(df)
                        else:
                            fetch_failures.append((league, season))
        except Exception as e:
            print(f"  [ERROR] Browser failed: {e}")

    if not all_frames:
        print("\n  [WARN] No FBref data scraped.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n  Total FBref rows combined: {len(combined)}")
    if fetch_failures:
        print(f"  Fetch failures: {fetch_failures}")

    return combined


# =============================================================================
# STEP 4 — Filter to new signings + apply league multipliers
# =============================================================================

def _extract_fbref_stats(row_series, df_columns):
    """
    Given a row from the merged FBref DataFrame, extract all needed stats
    using flexible column-name lookup. Returns a dict of raw stats.
    """
    def get(candidates, default=0.0):
        col = find_col(pd.DataFrame(columns=df_columns), candidates)
        if col and col in row_series.index:
            v = row_series[col]
            try:
                return float(v) if not pd.isna(v) else default
            except (ValueError, TypeError):
                return default
        return default

    # Appearances / minutes
    mp  = get(["Playing Time_MP",     "Performance_MP",    "MP"])
    st  = get(["Playing Time_Starts", "Performance_Starts","Starts"])
    mn  = get(["Playing Time_Min",    "Performance_Min",   "Min"])

    # Goals / assists
    gls = get(["Performance_Gls", "Gls"])
    ast = get(["Performance_Ast", "Ast"])

    # Expected (from shooting page merge: Standard_xG or Expected_xG)
    xg  = get(["Standard_xG", "Expected_xG", "xG", "xG_x"])
    xag = get(["Expected_xAG", "xAG", "xAG_x"])

    # Cards
    crdy = get(["Performance_CrdY", "CrdY"])
    crdr = get(["Performance_CrdR", "CrdR"])

    # Progressive
    prgc = get(["Progression_PrgC", "PrgC"])
    prgp = get(["Progression_PrgP", "PrgP"])

    # Per-90 from standard page
    gls90 = get(["Per 90 Minutes_Gls", "Gls/90"])
    ast90 = get(["Per 90 Minutes_Ast", "Ast/90"])

    # Compute per-90 from totals if not directly available
    nineties = mn / 90.0 if mn > 0 else 0
    if gls90 == 0.0 and nineties > 0:
        gls90 = safe_div(gls, nineties)
    if ast90 == 0.0 and nineties > 0:
        ast90 = safe_div(ast, nineties)

    xg90  = safe_div(xg, nineties)  if nineties > 0 else 0
    xag90 = safe_div(xag, nineties) if nineties > 0 else 0

    # Shots per 90 (from shooting page: Standard_Sh/90 or computed)
    sh90 = get(["Standard_Sh/90", "Sh/90"])
    if sh90 == 0.0:
        sh_tot = get(["Standard_Sh", "Sh"])
        sh90   = safe_div(sh_tot, nineties) if nineties > 0 else 0

    # Key passes — proxy from shooting page SoT/90 if not available
    kp90 = get(["Passes_KP", "KP", "key_passes_per_90", "Standard_SoT/90", "SoT/90"])

    # Keeper stats (from keeper page merge)
    saves  = get(["Performance_Saves", "Saves"])
    save_p = get(["Performance_Save%", "Save%"])
    cs     = get(["Performance_CS",    "CS"])
    sota   = get(["Performance_SoTA",  "SoTA"])

    return {
        "appearances": mp,
        "starts":      st,
        "minutes":     mn,
        "goals":       gls,
        "assists":     ast,
        "xG":          xg,
        "xA":          xag,
        "shots_per_90": sh90,
        "key_passes_per_90": kp90,
        "goals_per_90":   gls90,
        "assists_per_90": ast90,
        "yellow_cards":   crdy,
        "red_cards":      crdr,
        "progressive_carries": prgc,
        "progressive_passes":  prgp,
        "clean_sheets":    cs,
        "saves":           saves,
        "save_percentage": save_p,
    }


def _season_reliability(minutes, appearances):
    """1.0 / 0.5 / 0.1 based on minutes vs available (appearances x 90)."""
    available = appearances * 90
    if available <= 0:
        return 0.1
    ratio = minutes / available
    if ratio >= 0.6:
        return 1.0
    if ratio >= 0.3:
        return 0.5
    return 0.1


def step4_filter_and_multiply(fbref_combined, new_signings_df):
    print("\n" + "=" * 60)
    print("STEP 4: Filtering to new signings + applying league multipliers")
    print("=" * 60)

    if fbref_combined.empty:
        print("  [WARN] No FBref data to filter.")
        return pd.DataFrame()

    # Build lookup: new signing name -> (fpl_name, fpl_team, fpl_position, fpl_price, prev_league)
    ns_lookup = {}
    for _, r in new_signings_df.iterrows():
        ns_lookup[r["fpl_name"]] = r

    ns_fpl_names = list(ns_lookup.keys())

    # Identify player column in fbref_combined
    player_col = _find_player_col(fbref_combined)
    team_col   = _find_squad_col(fbref_combined)
    pos_col    = _find_pos_col(fbref_combined)
    season_col = find_col(fbref_combined, ["season_tag", "season", "Season"])
    league_col = find_col(fbref_combined, ["league_tag", "league", "League"])

    if not player_col:
        print("  [ERROR] Cannot find player name column in FBref data.")
        print(f"  Available columns: {list(fbref_combined.columns)[:40]}")
        return pd.DataFrame()

    print(f"  FBref columns: player='{player_col}', team='{team_col}', "
          f"pos='{pos_col}', season='{season_col}', league='{league_col}'")

    fbref_player_names = fbref_combined[player_col].dropna().unique().tolist()

    matched_rows = []
    low_conf     = []
    no_match     = []

    # Match each FBref row to a new signing
    fbref_combined = fbref_combined.copy()
    fbref_combined["_ns_match"] = None
    fbref_combined["_ns_score"] = 0

    # Build a cache of fbref_name -> best ns match to avoid re-computing.
    # Apply FBREF_NAME_OVERRIDES first: inject exact lookups so fuzzy never has to guess.
    fbref_to_ns = {}
    override_reverse = {normalize_name(v): k for k, v in FBREF_NAME_OVERRIDES.items()}
    for fname in fbref_player_names:
        norm_fname = normalize_name(str(fname))
        if norm_fname in override_reverse:
            fpl_target = override_reverse[norm_fname]
            fbref_to_ns[fname] = (fpl_target, 100)
        else:
            m, s = fuzzy_match_name(fname, ns_fpl_names)
            fbref_to_ns[fname] = (m, s)

    # Build normalised false-positive set for fast lookup at row-apply time.
    _fp_norm_set = {
        (normalize_name(fb), normalize_name(fpl))
        for fb, fpl in FBREF_FALSE_POSITIVES
    }

    # Apply match to rows, skipping false positives
    for idx, row in fbref_combined.iterrows():
        pname = row.get(player_col, "")
        if pd.isna(pname) or not pname:
            continue
        m, s = fbref_to_ns.get(str(pname), (None, 0))
        if m:
            pair = (normalize_name(str(pname)), normalize_name(str(m)))
            if pair in _fp_norm_set:
                continue  # blocked false positive
            fbref_combined.at[idx, "_ns_match"] = m
            fbref_combined.at[idx, "_ns_score"] = s

    matched_df = fbref_combined[fbref_combined["_ns_match"].notna()].copy()
    print(f"\n  FBref rows matched to new signings: {len(matched_df)}")

    # Low confidence
    low_df = matched_df[matched_df["_ns_score"] < 95]
    if len(low_df) > 0:
        pairs_seen = set()
        print(f"\n  [REVIEW] FBref->FPL matches below 95%:")
        for _, r in low_df.iterrows():
            pair = (r[player_col], r["_ns_match"])
            if pair not in pairs_seen:
                print(f"    {int(r['_ns_score']):3d}%  '{r[player_col]}'  ->  '{r['_ns_match']}'")
                pairs_seen.add(pair)

    # Players in new_signings with NO fbref rows at all
    matched_ns_names = matched_df["_ns_match"].unique().tolist()
    no_fbref = [n for n in ns_fpl_names if n not in matched_ns_names]
    if no_fbref:
        print(f"\n  [FLAG] {len(no_fbref)} new signings with no FBref data found:")
        for n in no_fbref:
            pl_team = ns_lookup.get(n, {}).get("fpl_team", "?")
            pl_pos  = ns_lookup.get(n, {}).get("fpl_position", "?")
            print(f"    - {n} ({pl_team}, {pl_pos})")

    # ── Apply multipliers ─────────────────────────────────────────────────────
    result_rows = []
    for _, row in matched_df.iterrows():
        ns_name   = row["_ns_match"]
        ns_data   = ns_lookup.get(ns_name, {})
        league    = row.get(league_col, "") if league_col else ""
        season    = row.get(season_col, "") if season_col else ""
        fbref_pos = row.get(pos_col, "") if pos_col else ""

        # Get multiplier from the new_signings previous_league
        prev_league = ""
        if hasattr(ns_data, "get"):
            raw = ns_data.get("previous_league_standardized", "")
            # Guard against NaN (float) which is truthy in Python
            if raw and not (isinstance(raw, float) and pd.isna(raw)):
                prev_league = str(raw)
        if not prev_league and league:
            prev_league = str(league)  # fall back to FBref league_tag

        multiplier = LEAGUE_MULTIPLIERS.get(str(prev_league), 1.0)

        stats = _extract_fbref_stats(row, list(fbref_combined.columns))

        adj_goals    = stats["goals"]    * multiplier
        adj_assists  = stats["assists"]  * multiplier
        adj_xG       = stats["xG"]       * multiplier
        adj_xA       = stats["xA"]       * multiplier
        adj_gls90    = stats["goals_per_90"]   * multiplier
        adj_ast90    = stats["assists_per_90"] * multiplier

        reliability = _season_reliability(stats["minutes"], stats["appearances"])

        result_rows.append({
            # Identity
            "fbref_name":      row.get(player_col, ""),
            "fpl_name":        ns_name,
            "previous_league": prev_league,
            "multiplier":      multiplier,
            "fbref_season":    season,
            "fbref_team":      row.get(team_col, "") if team_col else "",
            "fbref_position":  fbref_pos,
            # Raw stats
            **stats,
            # Adjusted
            "adjusted_goals":          adj_goals,
            "adjusted_assists":        adj_assists,
            "adjusted_xG":             adj_xG,
            "adjusted_xA":             adj_xA,
            "adjusted_goals_per_90":   adj_gls90,
            "adjusted_assists_per_90": adj_ast90,
            # Reliability
            "season_reliability": reliability,
            "is_new_to_pl":       1,
        })

    result_df = pd.DataFrame(result_rows)
    print(f"\n  Rows after filtering + multipliers: {len(result_df)}")

    # ── Zero-stats fallback for players with non-scrapeable leagues ───────────
    # Find confirmed players (price>=4.5 + has TM league, or VAASTAV_FALSE_POSITIVES)
    # who still have no FBref rows — add them with 0 stats + data_confidence=low
    _fp_norms = {normalize_name(n) for n in VAASTAV_FALSE_POSITIVES}
    matched_ns_names = set(result_df["fpl_name"].tolist()) if not result_df.empty else set()
    zero_rows = []
    for _, ns_row in new_signings_df.iterrows():
        ns_name   = ns_row["fpl_name"]
        ns_norm   = normalize_name(str(ns_name))
        has_league = (
            "previous_league_standardized" in ns_row.index
            and ns_row["previous_league_standardized"]
            and not (isinstance(ns_row["previous_league_standardized"], float)
                     and pd.isna(ns_row["previous_league_standardized"]))
        )
        is_fp  = ns_norm in _fp_norms
        is_conf = (ns_row.get("fpl_price", 0) >= 4.5 and has_league) or is_fp

        if not is_conf:
            continue
        if ns_name in matched_ns_names:
            continue  # already has FBref data

        prev_league = str(ns_row.get("previous_league_standardized", "")) if has_league else "unknown"
        no_fbref.append(ns_name)
        zero_rows.append({
            "fbref_name":            ns_name,
            "fpl_name":              ns_name,
            "previous_league":       prev_league,
            "multiplier":            LEAGUE_MULTIPLIERS.get(prev_league, 1.0),
            "fbref_season":          "N/A",
            "fbref_team":            ns_row.get("fpl_team", ""),
            "fbref_position":        "",
            "data_confidence":       "low",
            "appearances": 0, "starts": 0, "minutes": 0,
            "goals": 0, "assists": 0, "xG": 0, "xA": 0,
            "shots_per_90": 0, "key_passes_per_90": 0,
            "goals_per_90": 0, "assists_per_90": 0,
            "yellow_cards": 0, "red_cards": 0,
            "progressive_carries": 0, "progressive_passes": 0,
            "clean_sheets": 0, "saves": 0, "save_percentage": 0,
            "adjusted_goals": 0, "adjusted_assists": 0,
            "adjusted_xG": 0, "adjusted_xA": 0,
            "adjusted_goals_per_90": 0, "adjusted_assists_per_90": 0,
            "season_reliability": 0.1,
            "is_new_to_pl": 1,
        })

    # ── SKIP_FBREF zero-stat rows ──────────────────────────────────────────────
    # These players are confirmed to have no top-5 league FBref data.
    skip_zero_rows = []
    _skip_norms = {normalize_name(n) for n in SKIP_FBREF}
    for _, ns_row in new_signings_df.iterrows():
        ns_name = ns_row["fpl_name"]
        if normalize_name(str(ns_name)) not in _skip_norms:
            continue
        if ns_name in matched_ns_names:
            continue  # already has some data (shouldn't happen, but guard)
        prev_league = str(ns_row.get("previous_league_standardized", "unknown"))
        skip_zero_rows.append({
            "fbref_name":      ns_name,
            "fpl_name":        ns_name,
            "previous_league": prev_league,
            "multiplier":      LEAGUE_MULTIPLIERS.get(prev_league, 1.0),
            "fbref_season":    "N/A",
            "fbref_team":      ns_row.get("fpl_team", ""),
            "fbref_position":  "",
            "data_confidence": "low",
            "appearances": 0, "starts": 0, "minutes": 0,
            "goals": 0, "assists": 0, "xG": 0, "xA": 0,
            "shots_per_90": 0, "key_passes_per_90": 0,
            "goals_per_90": 0, "assists_per_90": 0,
            "yellow_cards": 0, "red_cards": 0,
            "progressive_carries": 0, "progressive_passes": 0,
            "clean_sheets": 0, "saves": 0, "save_percentage": 0,
            "adjusted_goals": 0, "adjusted_assists": 0,
            "adjusted_xG": 0, "adjusted_xA": 0,
            "adjusted_goals_per_90": 0, "adjusted_assists_per_90": 0,
            "season_reliability": 0.1,
            "is_new_to_pl": 1,
        })

    if zero_rows:
        print(f"\n  Adding {len(zero_rows)} zero-stat rows for non-scrapeable leagues:")
        for zr in zero_rows:
            print(f"    - {zr['fpl_name']} ({zr['previous_league']})")
        result_df = pd.concat(
            [result_df, pd.DataFrame(zero_rows)], ignore_index=True
        )

    if skip_zero_rows:
        print(f"\n  Adding {len(skip_zero_rows)} zero-stat rows for SKIP_FBREF players:")
        for zr in skip_zero_rows:
            print(f"    - {zr['fpl_name']} (no top-5 league data)")
        result_df = pd.concat(
            [result_df, pd.DataFrame(skip_zero_rows)], ignore_index=True
        )

    rel_counts = result_df["season_reliability"].value_counts().to_dict()
    print(f"  Season reliability: 1.0={rel_counts.get(1.0,0)}, "
          f"0.5={rel_counts.get(0.5,0)}, 0.1={rel_counts.get(0.1,0)}")

    return result_df, no_fbref


# =============================================================================
# STEP 5 — Build position-specific files matching vaastav structure
# =============================================================================

def _map_fbref_position(fbref_pos_str):
    """Map FBref position string (e.g. 'DF,MF') to FPL position (DEF/MID/FWD/GK)."""
    if not fbref_pos_str or pd.isna(fbref_pos_str):
        return "MID"  # default
    first = str(fbref_pos_str).split(",")[0].strip()
    return FBREF_POS_MAP.get(first, "MID")


def step5_build_position_files(result_df, new_signings_df):
    print("\n" + "=" * 60)
    print("STEP 5: Building position-specific files (vaastav structure)")
    print("=" * 60)

    if result_df.empty:
        print("  [WARN] No data to write.")
        return {}, {}

    # Build FPL lookup: fpl_name -> {fpl_team, fpl_position, fpl_price, ...}
    fpl_lookup = {}
    for _, r in new_signings_df.iterrows():
        fpl_lookup[r["fpl_name"]] = r

    out_rows = []

    for _, r in result_df.iterrows():
        ns_name  = r.get("fpl_name", "")
        fpl_data = fpl_lookup.get(ns_name, {})

        # Position: prefer FPL API position, fall back to FBref
        fpl_pos = fpl_data.get("fpl_position", "") if hasattr(fpl_data, "get") else ""
        if not fpl_pos or pd.isna(fpl_pos):
            fpl_pos = _map_fbref_position(r.get("fbref_position", ""))
        else:
            fpl_pos = str(fpl_pos).upper()
            if fpl_pos == "GKP":
                fpl_pos = "GK"

        # FPL price -> value (in £, same as vaastav tenths/10 format)
        fpl_price = fpl_data.get("fpl_price", 5.0) if hasattr(fpl_data, "get") else 5.0
        try:
            fpl_price = float(fpl_price)
        except (ValueError, TypeError):
            fpl_price = 5.0

        fpl_team = fpl_data.get("fpl_team", r.get("fbref_team", "Unknown"))
        if hasattr(fpl_team, "get"):
            fpl_team = "Unknown"

        season   = str(r.get("fbref_season", "2024-25"))
        seas_yr  = season_to_year(season)

        mins     = float(r.get("minutes",  0) or 0)
        goals    = float(r.get("goals",    0) or 0)
        assists  = float(r.get("assists",  0) or 0)
        cs       = float(r.get("clean_sheets", 0) or 0)
        saves    = float(r.get("saves",    0) or 0)
        crdy     = float(r.get("yellow_cards", 0) or 0)
        crdr     = float(r.get("red_cards",    0) or 0)
        apps     = float(r.get("appearances",  0) or 0)
        adj_g90  = float(r.get("adjusted_goals_per_90",   0) or 0)
        adj_a90  = float(r.get("adjusted_assists_per_90", 0) or 0)
        rel      = float(r.get("season_reliability", 0.1) or 0.1)

        cs_rate  = safe_div(cs, apps) if fpl_pos in ("GK", "DEF") else 0.0
        sv_pg    = safe_div(saves, apps) if fpl_pos == "GK" else 0.0

        row_out = {
            "name":           ns_name,
            "position":       fpl_pos,
            "team":           fpl_team,
            "GW":             0,
            "opponent_team":  "",
            "was_home":       False,
            "total_points":   0,
            "minutes":        mins,
            "goals_scored":   goals,
            "assists":        assists,
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
            "value":          fpl_price,
            "season":         season,
            "season_year":    seas_yr,
            "form_last3":     0.0,
            "form_last5":     0.0,
            "minutes_reliability_season":   rel,
            "cumulative_points_season":     0,
            "avg_points_per_game_season":   0.0,
            "goals_per_game_season":        adj_g90,
            "assists_per_game_season":      adj_a90,
            "clean_sheet_rate_season":      cs_rate,
            "saves_per_game_season":        sv_pg,
            "points_per_million":           0.0,
            "is_new_to_pl":                 1,
        }
        out_rows.append(row_out)

    out_df = pd.DataFrame(out_rows, columns=VAASTAV_COLS)

    # ── Also save extended intermediate (for verify_stage4a.py) ──────────────
    # Attach extra columns from result_df that aren't in VAASTAV_COLS
    ext_cols = [
        "fbref_name", "fpl_name", "previous_league", "multiplier",
        "fbref_season", "fbref_team", "fbref_position",
        "appearances", "goals", "assists", "xG", "xA",
        "shots_per_90", "goals_per_90", "assists_per_90",
        "yellow_cards", "red_cards", "clean_sheets", "saves", "save_percentage",
        "adjusted_goals", "adjusted_assists", "adjusted_goals_per_90", "adjusted_assists_per_90",
        "season_reliability", "data_confidence", "is_new_to_pl",
    ]
    ext_keep = [c for c in ext_cols if c in result_df.columns]
    ext_df = result_df[ext_keep].copy()
    # Add FPL position from out_df (aligned by index)
    ext_df = ext_df.reset_index(drop=True)
    out_df_reset = out_df.reset_index(drop=True)
    ext_df["fpl_position"] = out_df_reset["position"]
    ext_df["fpl_team"]     = out_df_reset["team"]
    ext_df["fpl_price"]    = out_df_reset["value"]
    ext_path = os.path.join(FBREF_SIGN_DIR, "result_extended.csv")
    ext_df.to_csv(ext_path, index=False)
    print(f"  Saved: result_extended.csv ({len(ext_df)} rows, extended cols)")

    # ── Split by position ─────────────────────────────────────────────────────
    pos_map = {"GK": "gk", "DEF": "def", "MID": "mid", "FWD": "fwd"}
    saved_files = {}

    for fpl_pos, slug in pos_map.items():
        pos_df = out_df[out_df["position"] == fpl_pos].copy()
        fname  = f"new_signings_{slug}.csv"
        fpath  = os.path.join(FBREF_SIGN_DIR, fname)
        pos_df.to_csv(fpath, index=False)
        saved_files[fpl_pos] = (fpath, len(pos_df))
        print(f"  Saved: {fname} ({len(pos_df)} rows)")

    return out_df, saved_files


# =============================================================================
# STEP 6 — Validation report
# =============================================================================

def step6_validation_report(
    new_signings_df, tm_total, tm_unmatched,
    fbref_combined, result_df_or_tuple,
    saved_files, no_fbref_list
):
    print("\n" + "=" * 60)
    print("=== STAGE 4a VALIDATION REPORT ===")
    print("=" * 60)

    # Unpack result_df if it came as a tuple (result_df, saved_files)
    result_df = result_df_or_tuple
    if isinstance(result_df_or_tuple, tuple):
        result_df = result_df_or_tuple[0]

    total_fpl = pd.read_csv(os.path.join(FPL_API_DIR, "players_raw.csv")).shape[0]
    n_new = len(new_signings_df)
    n_hist = total_fpl - n_new

    print(f"\nNEW TO PL PLAYERS IDENTIFIED:")
    print(f"  Total FPL players:                 {total_fpl}")
    print(f"  Players with vaastav history:      {n_hist}")
    print(f"  Players new to PL (no vaastav):    {n_new}")

    print(f"\nTRANSFERMARKT SCRAPE:")
    tm_matched = new_signings_df["previous_league_standardized"].notna().sum() \
        if "previous_league_standardized" in new_signings_df.columns else 0
    print(f"  Players found on Transfermarkt:    {tm_total}")
    print(f"  Matched to FPL API list:           {tm_matched}")
    if tm_unmatched:
        print(f"  Could not match to FPL API ({len(tm_unmatched)}): {tm_unmatched}")

    if "previous_league_standardized" in new_signings_df.columns:
        lc = (
            new_signings_df["previous_league_standardized"]
            .value_counts()
            .rename_axis("League")
            .reset_index(name="Count")
        )
        print(f"\n  Breakdown by previous league:")
        for _, r in lc.iterrows():
            print(f"    {r['League']:<30} {r['Count']}")

    if "transfer_type" in new_signings_df.columns:
        perm = new_signings_df["transfer_type"].eq("permanent").sum()
        loan = new_signings_df["transfer_type"].eq("loan").sum()
        print(f"\n  Breakdown by transfer type: {perm} permanent / {loan} loan")

    print(f"\nFBREF SCRAPE:")
    if not fbref_combined.empty:
        player_col = find_col(fbref_combined, ["player", "Player"])
        league_col = find_col(fbref_combined, ["league_tag", "league"])
        season_col = find_col(fbref_combined, ["season_tag", "season"])
        leagues_scraped = fbref_combined[league_col].unique().tolist() if league_col else []
        seasons_scraped = fbref_combined[season_col].unique().tolist() if season_col else []
        print(f"  Leagues scraped: {leagues_scraped}")
        print(f"  Seasons:         {seasons_scraped}")
        if player_col:
            print(f"  Total FBref rows: {len(fbref_combined)}")
        matched_players = result_df["fpl_name"].nunique() if not result_df.empty else 0
        print(f"  Players successfully matched: {matched_players}")
    else:
        print("  [No FBref data]")

    if no_fbref_list:
        print(f"\n  [FLAG] Players with no FBref data found ({len(no_fbref_list)}):")
        for n in no_fbref_list:
            print(f"    - {n}")

    if not result_df.empty and "adjusted_goals_per_90" in result_df.columns:
        print(f"\nTOP 10 NEW SIGNINGS BY ADJUSTED GOALS PER 90:")
        top = (
            result_df[result_df["adjusted_goals_per_90"] > 0]
            .sort_values("adjusted_goals_per_90", ascending=False)
            .drop_duplicates("fpl_name")
            .head(10)
        )
        for i, (_, r) in enumerate(top.iterrows(), 1):
            print(f"  {i:2d}. {r['fpl_name']:<28} {r['adjusted_goals_per_90']:.3f}  "
                  f"({r.get('previous_league','?')}, {r.get('fbref_season','?')})")

    if not result_df.empty and "season_reliability" in result_df.columns:
        print(f"\nSEASON RELIABILITY BREAKDOWN:")
        rc = result_df["season_reliability"].value_counts().to_dict()
        print(f"  1.0 (fully reliable): {rc.get(1.0, 0)} player-seasons")
        print(f"  0.5 (partial):        {rc.get(0.5, 0)} player-seasons")
        print(f"  0.1 (injury affected): {rc.get(0.1, 0)} player-seasons")

    print(f"\nFILES SAVED:")
    ns_csv = os.path.join(TRANSFERS_DIR, "new_signings_2025.csv")
    print(f"  new_signings_2025.csv    -- {len(new_signings_df)} rows")
    for pos, (fpath, n) in saved_files.items():
        slug = {"GK": "gk", "DEF": "def", "MID": "mid", "FWD": "fwd"}[pos]
        print(f"  new_signings_{slug}.csv     -- {n} rows")

    print("\n=== END REPORT ===")


# =============================================================================
# MAIN
# =============================================================================

def _print_confirmed_list(df):
    """
    Print the relevance-filtered new signings list grouped by position, sorted by price.
    Kept: price >= 4.5 AND has TM previous_league  OR  in VAASTAV_FALSE_POSITIVES.
    """
    has_league  = df["previous_league_standardized"].notna()
    high_price  = df["fpl_price"] >= 4.5
    force_in    = df["fpl_name"].isin(VAASTAV_FALSE_POSITIVES)
    confirmed   = df[(has_league & high_price) | force_in].copy()

    print("\n" + "=" * 70)
    print("CONFIRMED NEW-TO-PL SIGNINGS (price >= 4.5 + TM match, or override)")
    print(f"Total: {len(confirmed)} players")
    print("=" * 70)

    for pos in ["GK", "DEF", "MID", "FWD"]:
        subset = confirmed[confirmed["fpl_position"] == pos].sort_values(
            "fpl_price", ascending=False
        )
        if subset.empty:
            continue
        print(f"\n  {pos} ({len(subset)})")
        print(f"  {'Name':<32} {'Club':<22} {'Price':>5}  Previous League")
        print(f"  {'-'*32} {'-'*22} {'-'*5}  {'-'*22}")
        for _, r in subset.iterrows():
            league = r.get("previous_league_standardized", "")
            if not league or (isinstance(league, float) and pd.isna(league)):
                league = "(unknown)"
            print(f"  {r['fpl_name']:<32} {r['fpl_team']:<22} {r['fpl_price']:>5.1f}  {league}")

    print("\n" + "=" * 70)
    print("Run with --full-run to proceed to FBref scraping.")
    print("=" * 70)
    return confirmed


def main():
    full_run   = "--full-run"   in sys.argv
    step4_only = "--step4-only" in sys.argv

    print("Stage 4a: New Premier League Signings Data")
    print("Seasons for FBref: " + str(SEASONS))
    print("Blocked: 2025-26 (live season -- never use)")

    if step4_only:
        # ── Fast re-run: load cached outputs from Steps 1-3 ──────────────────
        print("\n[--step4-only] Loading cached Step 1 + Step 3 outputs...")
        ns_path = os.path.join(TRANSFERS_DIR, "new_signings_2025.csv")
        if not os.path.exists(ns_path):
            print(f"  [ERROR] Missing {ns_path} — run without --step4-only first.")
            sys.exit(1)
        new_signings_df = pd.read_csv(ns_path)
        tm_total    = int(new_signings_df["previous_league_standardized"].notna().sum())
        tm_unmatched = []

        # Load all cached FBref CSVs and concatenate.
        # Cache files are saved as flat CSVs (single-row header) by step3.
        fbref_frames = []
        for league_name in FBREF_LEAGUE_IDS:
            slug = league_name.lower().replace(" ", "_")
            for season in SEASONS:
                cache_path = os.path.join(FBREF_RAW_DIR, f"{slug}_{season}.csv")
                if os.path.exists(cache_path):
                    df = pd.read_csv(cache_path, low_memory=False)
                    # league_tag / season_tag should already be in the saved CSV;
                    # overwrite to be safe.
                    df["league_tag"] = league_name
                    df["season_tag"] = season
                    fbref_frames.append(df)
        fbref_combined = pd.concat(fbref_frames, ignore_index=True) if fbref_frames else pd.DataFrame()
        print(f"  new_signings_2025.csv: {len(new_signings_df)} rows")
        print(f"  FBref cache: {len(fbref_frames)} files, {len(fbref_combined)} rows")
    else:
        # ── Normal flow: Steps 1 → 2 ─────────────────────────────────────────
        # Step 1
        new_signings_df = step1_identify_new_signings()

        # Step 2
        tm_total, tm_unmatched = 0, []
        step2_result = step2_scrape_transfermarkt(new_signings_df)
        if isinstance(step2_result, tuple):
            new_signings_df, tm_total, tm_unmatched = step2_result
        else:
            new_signings_df = step2_result

        # Print confirmed list and stop unless --full-run
        _print_confirmed_list(new_signings_df)
        if not full_run:
            print("\n[STOPPED] Confirm the list above, then re-run with --full-run to continue.")
            return

        # Step 3
        fbref_combined = step3_scrape_fbref(new_signings_df)

    # Step 4
    no_fbref_list = []
    result_df     = pd.DataFrame()
    no_fbref_list = []

    if fbref_combined.empty:
        pass  # result_df stays empty
    else:
        result_tuple = step4_filter_and_multiply(fbref_combined, new_signings_df)
        if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
            result_df, no_fbref_list = result_tuple
        else:
            result_df = result_tuple

    # Step 5
    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
        out_df, saved_files = step5_build_position_files(result_df, new_signings_df)
    else:
        out_df = pd.DataFrame()
        saved_files = {}

    # Step 6
    step6_validation_report(
        new_signings_df,
        tm_total,
        tm_unmatched,
        fbref_combined if not fbref_combined.empty else pd.DataFrame(),
        result_df if isinstance(result_df, pd.DataFrame) else pd.DataFrame(),
        saved_files,
        no_fbref_list,
    )


if __name__ == "__main__":
    main()
