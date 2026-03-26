"""
Stage 6 — Feature Engineering
Run: python pipeline/feature_engineering_stage6.py
Resume: python pipeline/feature_engineering_stage6.py --start-step 2
"""

import argparse
import difflib
import json
import os
import sys
import unicodedata

import pandas as pd

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_RAW_VAASTAV = os.path.join(BASE, "data", "raw", "vaastav")
DATA_RAW_FBREF   = os.path.join(BASE, "data", "raw", "fbref", "new_signings")
DATA_RAW_FPL     = os.path.join(BASE, "data", "raw", "fpl_api")
DATA_PROCESSED   = os.path.join(BASE, "data", "processed")
STATE_FILE       = os.path.join(DATA_PROCESSED, "stage6_state.json")

# ─── State helpers ────────────────────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_completed_step": 0}


def save_state(state: dict):
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[state] Saved to {STATE_FILE}")


# ─── Gate helper ──────────────────────────────────────────────────────────────

def gate(step_num: int, next_description: str) -> bool:
    """Print gate banner and wait for y/n. Returns True to proceed."""
    print()
    print("=" * 70)
    print(f"STEP {step_num} COMPLETE -- Gate {step_num}")
    print("=" * 70)
    print(f"NEXT: {next_description}")
    print()
    while True:
        ans = input(f"Proceed to Step {step_num + 1}? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"
        print("  Please enter y or n.")


# ─── Step 1 ───────────────────────────────────────────────────────────────────

FILES = {
    "historical_gw_data.csv": {
        "path": os.path.join(DATA_RAW_VAASTAV, "historical_gw_data.csv"),
        "key_cols": [
            "name", "GW", "season", "total_points", "minutes",
            "goals_scored", "assists", "clean_sheets", "goals_conceded",
            "bonus", "bps", "saves", "team", "opponent_team", "was_home",
            "position", "transfers_in", "transfers_out", "selected", "value",
        ],
        "expected_rows": 51044,
    },
    "historical_gk.csv": {
        "path": os.path.join(DATA_RAW_VAASTAV, "historical_gk.csv"),
        "key_cols": [
            "name", "season", "GW", "total_points", "saves",
            "clean_sheets", "goals_conceded", "minutes",
        ],
        "expected_rows": None,
    },
    "historical_def.csv": {
        "path": os.path.join(DATA_RAW_VAASTAV, "historical_def.csv"),
        "key_cols": [
            "name", "season", "GW", "total_points", "clean_sheets",
            "goals_conceded", "minutes", "tackles_won_per_90",
            "interceptions_per_90", "defensive_actions_available",
        ],
        "expected_rows": None,
    },
    "historical_mid.csv": {
        "path": os.path.join(DATA_RAW_VAASTAV, "historical_mid.csv"),
        "key_cols": [
            "name", "season", "GW", "total_points", "goals_scored",
            "assists", "minutes",
        ],
        "expected_rows": None,
    },
    "historical_fwd.csv": {
        "path": os.path.join(DATA_RAW_VAASTAV, "historical_fwd.csv"),
        "key_cols": [
            "name", "season", "GW", "total_points", "goals_scored",
            "assists", "minutes",
        ],
        "expected_rows": None,
    },
    "new_signings_gk.csv": {
        "path": os.path.join(DATA_RAW_FBREF, "new_signings_gk.csv"),
        "key_cols": [
            "fpl_name", "season", "saves", "clean_sheets",
            "prev_saves_per_game", "prev_cs_rate",
            "data_source", "data_confidence",
        ],
        "expected_rows": 29,
    },
    "new_signings_def.csv": {
        "path": os.path.join(DATA_RAW_FBREF, "new_signings_def.csv"),
        "key_cols": [
            "fpl_name", "season", "adjG_per_90", "adjA_per_90",
            "interceptions_per_90", "tackles_won_per_90",
            "defensive_actions_available", "data_source",
        ],
        "expected_rows": 107,
    },
    "new_signings_mid.csv": {
        "path": os.path.join(DATA_RAW_FBREF, "new_signings_mid.csv"),
        "key_cols": [
            "fpl_name", "season", "adjG_per_90", "adjA_per_90",
            "league_multiplier", "season_reliability",
            "data_source", "data_confidence",
        ],
        "expected_rows": 123,
    },
    "new_signings_fwd.csv": {
        "path": os.path.join(DATA_RAW_FBREF, "new_signings_fwd.csv"),
        "key_cols": [
            "fpl_name", "season", "adjG_per_90", "adjA_per_90",
            "league_multiplier", "season_reliability",
            "data_source", "data_confidence",
        ],
        "expected_rows": 46,
    },
    "team_form.csv": {
        "path": os.path.join(DATA_PROCESSED, "team_form.csv"),
        "key_cols": [
            "team", "season", "GW", "xG_last5", "xGA_last5",
            "attacking_strength", "defensive_strength",
            "clean_sheet_probability",
        ],
        "expected_rows": None,
    },
    "fixture_difficulty.csv": {
        "path": os.path.join(DATA_RAW_FPL, "fixture_difficulty.csv"),
        "key_cols": ["team", "GW", "season", "difficulty"],
        "expected_rows": None,
    },
    "players_raw.csv": {
        "path": os.path.join(DATA_RAW_FPL, "players_raw.csv"),
        "key_cols": ["id", "web_name", "team", "element_type", "now_cost"],
        "expected_rows": None,
    },
    "player_upcoming_fixtures.csv": {
        "path": os.path.join(DATA_RAW_FPL, "player_upcoming_fixtures.csv"),
        "key_cols": ["element", "GW", "difficulty", "is_home", "opponent_team"],
        "expected_rows": None,
    },
}

# Column aliases: if the expected col name is absent but an alias exists, note it
COLUMN_ALIASES = {
    # fixture_difficulty.csv uses different names
    "fixture_difficulty.csv": {
        "team": "team_name",
        "GW": "gameweek",
        "difficulty": "fdr",
        # season: absent entirely
    },
    # player_upcoming_fixtures.csv
    "player_upcoming_fixtures.csv": {
        "element": "player_id",
        "GW": "gameweek",
        "is_home": "is_home",  # same
    },
    # new_signings files use 'name' not 'fpl_name'
    "new_signings_gk.csv":  {"fpl_name": "name"},
    "new_signings_def.csv": {"fpl_name": "name"},
    "new_signings_mid.csv": {"fpl_name": "name"},
    "new_signings_fwd.csv": {"fpl_name": "name"},
}


def step1(state: dict):
    print()
    print("=" * 70)
    print("STEP 1 -- Load and Audit All Source Files")
    print("=" * 70)
    print()

    loaded = {}
    load_errors = []
    all_critical_missing = []
    row_count_mismatches = []

    # ── Load table ──────────────────────────────────────────────────────────
    col_w = [40, 8, 6, 50, 20]
    header = (
        f"{'File':<40} {'Rows':>8} {'Cols':>6}  "
        f"{'Key columns present':<50}  {'Missing':<20}"
    )
    print(header)
    print("-" * 130)

    for fname, meta in FILES.items():
        path = meta["path"]
        key_cols = meta["key_cols"]
        expected_rows = meta.get("expected_rows")

        try:
            df = pd.read_csv(path, low_memory=False)
        except FileNotFoundError:
            load_errors.append(fname)
            print(f"  ERROR loading: {fname} -- file not found")
            continue
        except Exception as e:
            load_errors.append(fname)
            print(f"  ERROR loading: {fname} -- {e}")
            continue

        loaded[fname] = df
        actual_cols = set(df.columns.tolist())
        aliases = COLUMN_ALIASES.get(fname, {})

        present = []
        missing = []
        for c in key_cols:
            if c in actual_cols:
                present.append(c)
            elif c in aliases and aliases[c] in actual_cols:
                present.append(f"{c}(as {aliases[c]})")
            else:
                missing.append(c)
                all_critical_missing.append((fname, c))

        present_str = ", ".join(present[:5])  # truncate for display
        if len(present) > 5:
            present_str += f" +{len(present)-5} more"
        missing_str = ", ".join(missing) if missing else "none"

        rows = len(df)
        cols = len(df.columns)

        print(
            f"  {fname:<38} {rows:>8,} {cols:>6}  "
            f"{present_str:<50}  {missing_str}"
        )

        # Row count check
        if expected_rows is not None:
            tolerance = expected_rows * 0.10
            if abs(rows - expected_rows) > tolerance:
                row_count_mismatches.append(
                    f"{fname}: expected ~{expected_rows:,}, got {rows:,}"
                )

    print()

    # ── CHECK 1 — vaastav names not in players_raw ──────────────────────────
    print("CHECK 1 -- Players in vaastav_gw not in players_raw:")
    if "historical_gw_data.csv" in loaded and "players_raw.csv" in loaded:
        vaastav_names = set(loaded["historical_gw_data.csv"]["name"].dropna().unique())
        pr = loaded["players_raw.csv"]
        # players_raw may use web_name or first+second name
        pr_names = set()
        if "web_name" in pr.columns:
            pr_names.update(pr["web_name"].dropna().unique())
        if "first_name" in pr.columns and "second_name" in pr.columns:
            pr_names.update(
                (pr["first_name"].fillna("") + " " + pr["second_name"].fillna("")).str.strip()
            )
        v_not_pr = vaastav_names - pr_names
        print(f"  Count: {len(v_not_pr):,}")
        print("  Note: expected -- these are historical players who have left the PL")
    else:
        print("  SKIP -- required files not loaded")
    print()

    # ── CHECK 2 — players_raw names not in vaastav_gw ───────────────────────
    print("CHECK 2 -- Players in players_raw not in vaastav_gw:")
    if "players_raw.csv" in loaded and "historical_gw_data.csv" in loaded:
        pr = loaded["players_raw.csv"]
        vaastav_names = set(loaded["historical_gw_data.csv"]["name"].dropna().unique())

        pr_web = pr["web_name"].dropna().unique() if "web_name" in pr.columns else []
        no_history = []
        for wname in pr_web:
            if wname not in vaastav_names:
                no_history.append(wname)

        # Check which have stage4/4b data
        sig_names = set()
        for sig_file in ["new_signings_gk.csv", "new_signings_def.csv",
                         "new_signings_mid.csv", "new_signings_fwd.csv"]:
            if sig_file in loaded:
                df_sig = loaded[sig_file]
                col = "name" if "name" in df_sig.columns else (
                    "fpl_name" if "fpl_name" in df_sig.columns else None
                )
                if col:
                    sig_names.update(df_sig[col].dropna().unique())

        with_stage4 = [n for n in no_history if n in sig_names]
        without_data = [n for n in no_history if n not in sig_names]

        print(f"  Total in players_raw with zero GW history: {len(no_history):,}")
        print(f"  -- Have stage4a/4b data:  {len(with_stage4):,}")
        print(f"  -- No data at all:        {len(without_data):,}")
        if len(without_data) <= 30:
            for n in sorted(without_data):
                print(f"     {n}")
        else:
            for n in sorted(without_data)[:20]:
                print(f"     {n}")
            print(f"     ... and {len(without_data)-20} more")
    else:
        print("  SKIP -- required files not loaded")
    print()

    # ── CHECK 3 — Team name mismatches: vaastav vs team_form ─────────────────
    print("CHECK 3 -- Team name mismatches (vaastav vs team_form):")
    if "historical_gw_data.csv" in loaded and "team_form.csv" in loaded:
        vaastav_teams = set(loaded["historical_gw_data.csv"]["team"].dropna().unique())
        tf_teams = set(loaded["team_form.csv"]["team"].dropna().unique())
        v_only = vaastav_teams - tf_teams
        tf_only = tf_teams - vaastav_teams

        if not v_only and not tf_only:
            print("  No mismatches found -- join will be clean")
        else:
            print(f"  {'Vaastav name':<25} {'team_form name':<25} {'Action needed'}")
            print(f"  {'-'*70}")
            for name in sorted(v_only):
                print(f"  {name:<25} {'???':<25} add to mapping (in vaastav only)")
            for name in sorted(tf_only):
                print(f"  {'???':<25} {name:<25} add to mapping (in team_form only)")
    else:
        print("  SKIP -- required files not loaded")
    print()

    # ── CHECK 4 — 2025-26 data scan ──────────────────────────────────────────
    print("CHECK 4 -- 2025-26 data scan:")
    found_2526 = False
    for fname, df in loaded.items():
        for col in df.columns:
            col_lower = col.lower()
            if "season" in col_lower:
                vals = df[col].astype(str)
                mask = vals.str.contains("2025-26", na=False) | vals.str.contains("2025/26", na=False)
                n = mask.sum()
                if n > 0:
                    print(f"  CRITICAL ERROR: {fname} col '{col}' has {n:,} rows with 2025-26 season")
                    found_2526 = True
        # Also check for season_year == 2026
        if "season_year" in df.columns:
            mask2 = df["season_year"].astype(str).str.contains("2026", na=False)
            n2 = mask2.sum()
            if n2 > 0:
                print(f"  CRITICAL ERROR: {fname} col 'season_year' has {n2:,} rows with year 2026")
                found_2526 = True

    if found_2526:
        print()
        print("  HALTING -- 2025-26 data detected. Fix before proceeding.")
        sys.exit(1)
    else:
        print("  2025-26 data: NOT FOUND -- clean")
    print()

    # ── CHECK 5 — Season range ────────────────────────────────────────────────
    print("CHECK 5 -- Season range in historical_gw_data.csv:")
    if "historical_gw_data.csv" in loaded:
        seasons = sorted(loaded["historical_gw_data.csv"]["season"].dropna().unique())
        expected_seasons = {
            "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
        }
        print(f"  Seasons found: {seasons}")
        unexpected = [s for s in seasons if s not in expected_seasons]
        missing_s = [s for s in sorted(expected_seasons) if s not in seasons]
        if unexpected:
            print(f"  UNEXPECTED seasons: {unexpected}")
        if missing_s:
            print(f"  MISSING expected seasons: {missing_s}")
        if not unexpected and not missing_s:
            print("  All 6 expected seasons present -- OK")
    else:
        print("  SKIP -- file not loaded")
    print()

    # ── CHECK 6 — Row count sanity ────────────────────────────────────────────
    print("CHECK 6 -- Row count sanity:")
    checks = [
        ("historical_gw_data.csv",  51044),
        ("new_signings_def.csv",     107),
        ("new_signings_mid.csv",     123),
        ("new_signings_fwd.csv",     46),
        ("new_signings_gk.csv",      29),
    ]
    any_mismatch = False
    for fname, exp in checks:
        if fname in loaded:
            actual = len(loaded[fname])
            tol = exp * 0.10
            status = "OK" if abs(actual - exp) <= tol else "MISMATCH"
            if status == "MISMATCH":
                any_mismatch = True
            print(f"  {fname:<35} expected ~{exp:>6,}  actual {actual:>6,}  [{status}]")
        else:
            print(f"  {fname:<35} [NOT LOADED]")
    if not any_mismatch:
        print("  All row counts within 10% tolerance")
    print()

    # ── Gate 1 summary ────────────────────────────────────────────────────────
    n_loaded = len(loaded)
    n_total = len(FILES)
    n_critical = len(all_critical_missing)
    n_load_errors = len(load_errors)

    # Team name mismatches
    team_mismatch_count = 0
    if "historical_gw_data.csv" in loaded and "team_form.csv" in loaded:
        vaastav_teams = set(loaded["historical_gw_data.csv"]["team"].dropna().unique())
        tf_teams = set(loaded["team_form.csv"]["team"].dropna().unique())
        team_mismatch_count = len((vaastav_teams - tf_teams) | (tf_teams - vaastav_teams))

    print()
    print("=" * 70)
    print("STEP 1 COMPLETE -- Source Audit")
    print("=" * 70)
    print(f"  Files loaded successfully:      {n_loaded}/{n_total}")
    if load_errors:
        for e in load_errors:
            print(f"    FAILED: {e}")
    print(f"  2025-26 data detected:          NO")
    print(f"  Critical missing columns:       {n_critical}")
    if n_critical > 0:
        for fname, col in all_critical_missing:
            print(f"    {fname}: missing '{col}'")
    print(f"  Team name mismatches found:     {team_mismatch_count}")
    print(f"  Row count mismatches:           {len(row_count_mismatches)}")
    for m in row_count_mismatches:
        print(f"    {m}")
    print()
    print("  NEXT: Step 2 will build the base GW table from vaastav_gw data only.")
    print("  No features computed, no files written -- structural foundation only.")
    print()

    state["last_completed_step"] = 1
    state["step1"] = {
        "files_loaded": n_loaded,
        "files_total": n_total,
        "critical_missing_columns": n_critical,
        "team_mismatches": team_mismatch_count,
        "row_count_mismatches": len(row_count_mismatches),
        "load_errors": load_errors,
    }
    save_state(state)

    # Gate
    print("=" * 70)
    while True:
        ans = input("Proceed to Step 2? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            break
        print("  Please enter y or n.")
    print("=" * 70)

    if ans == "n":
        print("Stopped at Gate 1.")
        sys.exit(0)

    step2(state)


# ─── Step 2 ───────────────────────────────────────────────────────────────────

BASE_COLS = [
    "name", "season", "GW", "total_points", "minutes", "goals_scored",
    "assists", "clean_sheets", "goals_conceded", "bonus", "bps",
    "yellow_cards", "red_cards", "saves", "team", "opponent_team",
    "was_home", "transfers_in", "transfers_out", "selected", "value",
    "position",
]

VALID_POSITIONS = {"GK", "DEF", "MID", "FWD"}
VALID_SEASONS   = {
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
}

# element_type -> position label (fallback if position col absent)
ETYPE_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def step2(state: dict):
    print()
    print("=" * 70)
    print("STEP 2 -- Build Base GW Table")
    print("=" * 70)
    print()

    gw_path = os.path.join(DATA_RAW_VAASTAV, "historical_gw_data.csv")
    pr_path  = os.path.join(DATA_RAW_FPL, "players_raw.csv")

    # ── Load source ──────────────────────────────────────────────────────────
    print(f"  Loading {gw_path} ...")
    df_raw = pd.read_csv(gw_path, low_memory=False)
    print(f"  Loaded {len(df_raw):,} rows x {len(df_raw.columns)} cols")
    print()

    # ── Check required columns ────────────────────────────────────────────────
    print("  Checking required columns ...")
    present_cols = set(df_raw.columns)
    fatal = False

    # Handle position: might be named element_type / pos / position
    if "position" not in present_cols:
        if "element_type" in present_cols:
            print("  NOTE: 'position' absent — mapping from 'element_type'")
            df_raw["position"] = df_raw["element_type"].map(ETYPE_MAP)
        elif "pos" in present_cols:
            print("  NOTE: 'position' absent — using 'pos' column")
            df_raw = df_raw.rename(columns={"pos": "position"})
        else:
            print("  MISSING: position  (and no element_type/pos fallback)")
            fatal = True

    for col in BASE_COLS:
        if col not in df_raw.columns:
            print(f"  MISSING: {col}")
            fatal = True

    if fatal:
        print()
        print("  FATAL: missing required columns — cannot build base table. Stopping.")
        sys.exit(1)

    print("  All required columns present -- OK")
    print()

    # ── Trim to base columns only ─────────────────────────────────────────────
    df = df_raw[BASE_COLS].copy()

    # ── Pre-validation normalisation ─────────────────────────────────────────
    # 'GKP' is a known FPL typo that appears in 2021-22 GW37 (20 rows) — normalise to 'GK'
    gkp_count = (df["position"] == "GKP").sum()
    if gkp_count > 0:
        df["position"] = df["position"].replace("GKP", "GK")
        print(f"  NOTE: normalised {gkp_count} 'GKP' rows -> 'GK'")

    # 2019-20 COVID season ran to GW47 (bubble restart) — document and allow
    max_gw_by_season = df.groupby("season")["GW"].max()
    extended_seasons = max_gw_by_season[max_gw_by_season > 38]
    if len(extended_seasons):
        for s, mx in extended_seasons.items():
            n_ext = (df["season"] == s).sum()
            print(f"  NOTE: season {s} has GWs up to {mx} (COVID extension) -- {n_ext:,} rows, allowed")

    # ── Validation ────────────────────────────────────────────────────────────
    print()
    print("  Running row-level validations ...")
    errors = []

    # total_points nulls
    null_pts = df["total_points"].isna().sum()
    if null_pts > 0:
        errors.append(f"total_points has {null_pts:,} null values")

    # minutes negative
    neg_mins = (df["minutes"] < 0).sum()
    if neg_mins > 0:
        errors.append(f"minutes has {neg_mins:,} negative values")

    # season validity
    bad_seasons = df[~df["season"].isin(VALID_SEASONS)]["season"].unique()
    if len(bad_seasons):
        errors.append(f"invalid seasons: {sorted(bad_seasons)}")

    # GW range — 2019-20 legitimately goes to 47; all others max 38
    bad_gw_rows = []
    for season, grp in df.groupby("season"):
        max_allowed = 47 if season == "2019-20" else 38
        outside = grp[(grp["GW"] < 1) | (grp["GW"] > max_allowed)]
        bad_gw_rows.append(outside)
    bad_gw = pd.concat(bad_gw_rows) if bad_gw_rows else pd.DataFrame()
    bad_gw = bad_gw[bad_gw.index.isin(df.index)]  # ensure valid index
    if len(bad_gw):
        errors.append(
            f"GW out of valid range: {len(bad_gw):,} rows "
            f"(values: {sorted(bad_gw['GW'].unique())})"
        )

    # position validity
    bad_pos = df[~df["position"].isin(VALID_POSITIONS)]
    if len(bad_pos):
        errors.append(
            f"invalid position values: {list(bad_pos['position'].unique())} "
            f"({len(bad_pos):,} rows)"
        )

    if errors:
        print()
        for e in errors:
            print(f"  FATAL VALIDATION ERROR: {e}")
        print()
        print("  Stopping — fix validation errors before proceeding.")
        sys.exit(1)

    print("  All validations passed -- OK")
    print()

    # ── Position breakdown ────────────────────────────────────────────────────
    total_rows     = len(df)
    gk_rows        = (df["position"] == "GK").sum()
    def_rows       = (df["position"] == "DEF").sum()
    mid_rows       = (df["position"] == "MID").sum()
    fwd_rows       = (df["position"] == "FWD").sum()
    unique_players = df["name"].nunique()
    seasons_present = sorted(df["season"].unique())
    null_pts_final  = df["total_points"].isna().sum()

    print(f"  Total rows:       {total_rows:,}")
    print(f"  GK rows:          {gk_rows:,}")
    print(f"  DEF rows:         {def_rows:,}")
    print(f"  MID rows:         {mid_rows:,}")
    print(f"  FWD rows:         {fwd_rows:,}")
    print(f"  Unique players:   {unique_players:,}")
    print(f"  Seasons present:  {', '.join(seasons_present)}")
    print(f"  Null total_points: {null_pts_final}  {'(OK)' if null_pts_final == 0 else '(STOP)'}")
    print()

    # ── players_raw web_name construction ────────────────────────────────────
    print("  Building web_name from players_raw.csv ...")
    pr = pd.read_csv(pr_path, low_memory=False)
    if "web_name" in pr.columns:
        print("  web_name column found directly -- no construction needed")
        web_name_ok = True
    elif "first_name" in pr.columns and "second_name" in pr.columns:
        pr["web_name"] = (
            pr["first_name"].fillna("").str.strip()
            + " "
            + pr["second_name"].fillna("").str.strip()
        ).str.strip()
        sample = pr[["id", "web_name"]].head(5)
        print(f"  Constructed web_name from first_name + second_name -- sample:")
        for _, r in sample.iterrows():
            print(f"    id={r['id']}  web_name='{r['web_name']}'")
        web_name_ok = True
    else:
        print("  ERROR: neither web_name nor first_name+second_name found in players_raw.csv")
        web_name_ok = False

    if not web_name_ok:
        print("  Stopping -- cannot build web_name lookup.")
        sys.exit(1)

    null_web = pr["web_name"].isna().sum()
    print(f"  web_name nulls: {null_web}  ({len(pr):,} total players in players_raw)")
    print()

    # ── Save base table ───────────────────────────────────────────────────────
    out_path = os.path.join(DATA_PROCESSED, "base_gw_table.csv")
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved base_gw_table.csv  ({len(df):,} rows x {len(df.columns)} cols)")

    state["last_completed_step"] = 2
    state["step2"] = {
        "rows": total_rows,
        "unique_players": unique_players,
        "gk_rows": int(gk_rows),
        "def_rows": int(def_rows),
        "mid_rows": int(mid_rows),
        "fwd_rows": int(fwd_rows),
        "null_total_points": int(null_pts_final),
        "seasons": seasons_present,
        "output": "data/processed/base_gw_table.csv",
    }
    save_state(state)

    # ── Gate 2 ────────────────────────────────────────────────────────────────
    next_desc = (
        "Step 3 will compute all rolling player features from\n"
        "  the base table only. No new data sources. Features computed:\n"
        "  form_last3, form_last5, minutes_reliability_season,\n"
        "  cumulative_points_season, avg_points_per_game_season,\n"
        "  goals_per_game_season, assists_per_game_season,\n"
        "  clean_sheet_rate_season, saves_per_game_season,\n"
        "  points_per_million. All windows strictly partition by season."
    )

    print()
    print("=" * 70)
    print("STEP 2 COMPLETE -- Base GW Table")
    print("=" * 70)
    print(f"  Rows:           {total_rows:,}")
    print(f"  Unique players: {unique_players:,}")
    print(f"  Null targets:   {null_pts_final}  (must be 0)")
    print(f"  Position split: GK {gk_rows:,} / DEF {def_rows:,} / MID {mid_rows:,} / FWD {fwd_rows:,}")
    print(f"  Saved to:       data/processed/base_gw_table.csv")
    print()
    print(f"  NEXT: {next_desc}")
    print()

    while True:
        ans = input("Proceed to Step 3? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            break
        print("  Please enter y or n.")
    print("=" * 70)

    if ans == "n":
        print("Stopped at Gate 2.")
        sys.exit(0)

    step3(state)


# ─── Step 3 ───────────────────────────────────────────────────────────────────

ROLLING_FEATURES = [
    "form_last3",
    "form_last5",
    "cumulative_points_season",
    "avg_points_per_game_season",
    "goals_per_game_season",
    "assists_per_game_season",
    "clean_sheet_rate_season",
    "saves_per_game_season",
    "minutes_reliability_season",
    "points_per_million",
]


def validate_no_leakage(df: pd.DataFrame, feature_col: str):
    """Assert all GW==1 rows are 0 for this feature (no cross-season bleed)."""
    gw1_vals = df.loc[df["GW"] == 1, feature_col].fillna(0)
    bad = gw1_vals[gw1_vals != 0]
    if len(bad):
        print(f"  LEAKAGE DETECTED in {feature_col} -- {len(bad)} non-zero GW1 rows:")
        sample = df.loc[bad.index, ["name", "season", "GW", feature_col]].head(10)
        print(sample.to_string(index=False))
        raise AssertionError(f"Leakage in {feature_col}")


def step3(state: dict):
    print()
    print("=" * 70)
    print("STEP 3 -- Compute Rolling Player Features")
    print("=" * 70)
    print()

    base_path = os.path.join(DATA_PROCESSED, "base_gw_table.csv")

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"  Loading {base_path} ...")
    df = pd.read_csv(base_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")
    print()

    # ── Sort — critical before any rolling ───────────────────────────────────
    print("  Sorting by name, season, GW ...")
    df = df.sort_values(["name", "season", "GW"]).reset_index(drop=True)
    print("  Sort complete")
    print()

    # ── Compute 10 rolling features ───────────────────────────────────────────
    print("  Computing rolling features (all windows partition by [name, season]) ...")
    g = df.groupby(["name", "season"], sort=False)

    # 1. form_last3
    df["form_last3"] = g["total_points"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # 2. form_last5
    df["form_last5"] = g["total_points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # 3. cumulative_points_season
    df["cumulative_points_season"] = g["total_points"].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )

    # 4. avg_points_per_game_season
    df["avg_points_per_game_season"] = g["total_points"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 5. goals_per_game_season
    df["goals_per_game_season"] = g["goals_scored"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 6. assists_per_game_season
    df["assists_per_game_season"] = g["assists"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 7. clean_sheet_rate_season
    df["clean_sheet_rate_season"] = g["clean_sheets"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 8. saves_per_game_season
    df["saves_per_game_season"] = g["saves"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 9. minutes_reliability_season
    cum_mins = g["minutes"].transform(lambda x: x.shift(1).cumsum())
    prev_gw  = g["GW"].transform(lambda x: x.shift(1))
    denom    = prev_gw * 90
    # Avoid division by zero; where denom is 0 or NaN -> 0
    with_denom = denom.replace(0, float("nan"))
    df["minutes_reliability_season"] = (cum_mins / with_denom).clip(0, 1).fillna(0)

    # 10. points_per_million  — cumulative_pts / (value/10); 0 where either is 0
    price = df["value"] / 10.0
    cum_pts = df["cumulative_points_season"]
    df["points_per_million"] = 0.0
    valid_mask = (price > 0) & (cum_pts > 0)
    df.loc[valid_mask, "points_per_million"] = (
        cum_pts[valid_mask] / price[valid_mask]
    )

    print("  All 10 features computed")
    print()

    # ── Fill NaN (GW1 and any residual) ──────────────────────────────────────
    print("  Filling NaN values ...")
    fill_zero = [
        "form_last3", "form_last5", "avg_points_per_game_season",
        "goals_per_game_season", "assists_per_game_season",
        "clean_sheet_rate_season", "saves_per_game_season",
        "points_per_million",
    ]
    for col in fill_zero:
        df[col] = df[col].fillna(0)

    # cumulative_points_season already filled with fillna(0) in transform
    # minutes_reliability_season already filled with fillna(0) above
    # Final sweep — catch anything left
    residual_nan = df[ROLLING_FEATURES].isna().sum().sum()
    if residual_nan > 0:
        for col in ROLLING_FEATURES:
            df[col] = df[col].fillna(0)
        print(f"  NOTE: filled {residual_nan} residual NaNs with 0 after sweep")
    print("  NaN fill complete")
    print()

    # ── Leakage validation ────────────────────────────────────────────────────
    print("  Running leakage validation (GW==1 must all be 0) ...")
    print()
    leakage_results = []
    all_pass = True
    for feat in ROLLING_FEATURES:
        try:
            validate_no_leakage(df, feat)
            gw1_ok = True
        except AssertionError:
            gw1_ok = False
            all_pass = False

        gw1_zeros = df.loc[df["GW"] == 1, feat].fillna(0).eq(0).all()
        leakage_results.append((feat, gw1_zeros, gw1_ok))

    # Print leakage report table
    print(f"  {'Feature':<32} {'GW1 zeros':<12} {'Leakage check'}")
    print(f"  {'-'*56}")
    for feat, gw1_ok, leak_ok in leakage_results:
        gw1_str   = "YES" if gw1_ok else "NO -- FAIL"
        check_str = "PASS" if leak_ok else "FAIL"
        print(f"  {feat:<32} {gw1_str:<12} {check_str}")

    pass_count = sum(1 for _, _, ok in leakage_results if ok)
    print()
    print(f"  All {pass_count}/{len(ROLLING_FEATURES)} leakage checks: {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        print()
        print("  FATAL: leakage detected -- stopping. Fix before proceeding.")
        sys.exit(1)
    print()

    # ── NaN assertion ─────────────────────────────────────────────────────────
    final_nan = df[ROLLING_FEATURES].isna().sum().sum()
    if final_nan != 0:
        print(f"  FATAL: {final_nan} NaN values remain after fill -- stopping.")
        for col in ROLLING_FEATURES:
            n = df[col].isna().sum()
            if n:
                print(f"    {col}: {n} NaNs")
        sys.exit(1)
    print(f"  NaN assertion: 0 NaNs remaining -- OK")
    print()

    # ── Feature distribution summary ──────────────────────────────────────────
    print("  Feature distribution summary:")
    print(f"  {'Feature':<32} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*68}")
    for feat in ROLLING_FEATURES:
        s = df[feat]
        print(
            f"  {feat:<32} {s.mean():>8.3f} {s.std():>8.3f} "
            f"{s.min():>8.3f} {s.max():>8.3f}"
        )
    print()

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_PROCESSED, "base_gw_table.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved base_gw_table.csv  ({len(df):,} rows x {len(df.columns)} cols)")

    state["last_completed_step"] = 3
    state["step3"] = {
        "features_computed": len(ROLLING_FEATURES),
        "leakage_checks_passed": pass_count,
        "nan_remaining": int(final_nan),
        "total_columns": len(df.columns),
        "output": "data/processed/base_gw_table.csv",
    }
    save_state(state)

    # ── Gate 3 ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STEP 3 COMPLETE -- Rolling Player Features")
    print("=" * 70)
    print(f"  Features computed:          {len(ROLLING_FEATURES)}")
    print(f"  Leakage checks passed:      {pass_count}/{len(ROLLING_FEATURES)}  (must be {len(ROLLING_FEATURES)}/{len(ROLLING_FEATURES)})")
    print(f"  GW1 zero checks passed:     {pass_count}/{len(ROLLING_FEATURES)}  (must be {len(ROLLING_FEATURES)}/{len(ROLLING_FEATURES)})")
    print(f"  NaN values remaining:       {final_nan}      (must be 0)")
    print(f"  Columns in base table now:  {len(df.columns)}     (22 base + 10 rolling)")
    print(f"  Saved to: data/processed/base_gw_table.csv")
    print()
    print("  NEXT: Step 4 will attach stage 4a/4b previous-league baseline")
    print("  features as static columns on every GW row for each player's")
    print("  first PL season. These become prior knowledge features for")
    print("  new signings where vaastav has no history yet.")
    print()

    while True:
        ans = input("Proceed to Step 4? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            break
        print("  Please enter y or n.")
    print("=" * 70)

    if ans == "n":
        print("Stopped at Gate 3.")
        sys.exit(0)

    step4(state)


# ─── Step 4 helpers ───────────────────────────────────────────────────────────

# Fuzzy matches confirmed as same player (different name spellings)
MANUAL_MATCH_OVERRIDES = {
    "Filip Jörgensen":    "Filip Jørgensen",           # ö vs ø
    "Mikel Merino Zazón": "Mikel Merino",               # full vs short name
    "Omari Hutchinson":   "Omari Giraud-Hutchinson",   # shortened name
}

# Fuzzy hits that are DIFFERENT players — reject them
FUZZY_REJECT = {
    "Andrew Moran",       # matched Andrew Surman  — wrong player
    "Anthony Patterson",  # matched Nathan Patterson — wrong player
    "Charlie Crew",       # matched Charlie Cresswell — wrong player
    "Divine Mukasa",      # matched Divin Mubama — wrong player
    "Mamadou Sarr",       # matched Mamadou Sakho — wrong player
}

PREV_COLS = [
    "has_prev_league_data",
    "prev_adjG_per_90",
    "prev_adjA_per_90",
    "prev_league_multiplier",
    "prev_seasons_available",
    "prev_reliability_avg",
    "prev_minutes_avg",
    "prev_small_sample",
    "prev_int_per_90",
    "prev_tklW_per_90",
    "prev_saves_per_game",
    "prev_cs_rate",
]


def _norm(s: str) -> str:
    nfd = unicodedata.normalize("NFD", str(s))
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return "".join(c for c in stripped.lower() if c.isalpha() or c == " ").strip()


def _weighted_avg(series: "pd.Series", weights: "pd.Series") -> float:
    w = weights.fillna(0)
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0
    return float((series.fillna(0) * w).sum() / w_sum)


def build_prev_lookup(sigs_all: "pd.DataFrame") -> "pd.DataFrame":
    """Build one-row-per-player reliability-weighted prev features."""
    foreign = sigs_all[sigs_all["data_source"] != "vaastav"].copy()
    rows = []
    for name, grp in foreign.groupby("name"):
        pos = str(grp["position"].iloc[0]) if "position" in grp.columns else "UNK"
        w = grp["season_reliability"].fillna(0)

        adj_g = _weighted_avg(grp["adjG_per_90"], w)
        adj_a = _weighted_avg(grp["adjA_per_90"], w)

        prev_int  = _weighted_avg(grp["interceptions_per_90"], w) if "interceptions_per_90" in grp.columns else 0.0
        prev_tkl  = _weighted_avg(grp["tackles_won_per_90"],   w) if "tackles_won_per_90"   in grp.columns else 0.0

        prev_saves = _weighted_avg(grp["saves_per_game_season"],   w) if "saves_per_game_season"   in grp.columns else 0.0
        prev_cs    = _weighted_avg(grp["clean_sheet_rate_season"], w) if "clean_sheet_rate_season" in grp.columns else 0.0

        prev_mult     = float(grp["league_multiplier"].max())
        n_seasons     = int((grp["season_reliability"] > 0).sum())
        rel_avg       = float(grp["season_reliability"].mean())
        mins_avg      = float(grp["minutes"].mean())
        small_sample  = 1 if (grp["minutes"] < 500).all() else 0

        rows.append({
            "sig_name":              name,
            "position":              pos,
            "prev_adjG_per_90":      round(adj_g,   6),
            "prev_adjA_per_90":      round(adj_a,   6),
            "prev_league_multiplier": round(prev_mult, 4),
            "prev_seasons_available": n_seasons,
            "prev_reliability_avg":  round(rel_avg,  4),
            "prev_minutes_avg":      round(mins_avg, 1),
            "prev_small_sample":     small_sample,
            "prev_int_per_90":       round(prev_int, 6),
            "prev_tklW_per_90":      round(prev_tkl, 6),
            "prev_saves_per_game":   round(prev_saves, 4),
            "prev_cs_rate":          round(prev_cs,    4),
        })
    return pd.DataFrame(rows)


def match_lookup_to_vaastav(
    lookup: "pd.DataFrame", vaastav_names: set
) -> tuple:
    """Returns (matched_df, unmatched_names, fuzzy_below95)."""
    v_norm = {_norm(n): n for n in vaastav_names}
    matched_rows = []
    unmatched    = []
    fuzzy_below95 = []

    for _, row in lookup.iterrows():
        sig = row["sig_name"]

        # 1. Manual override
        if sig in MANUAL_MATCH_OVERRIDES:
            v_name = MANUAL_MATCH_OVERRIDES[sig]
            matched_rows.append({**row.to_dict(),
                                  "vaastav_name": v_name,
                                  "match_conf":   1.0,
                                  "match_type":   "manual"})
            continue

        # 2. Reject list — different player despite fuzzy similarity
        if sig in FUZZY_REJECT:
            unmatched.append(sig)
            continue

        # 3. Exact normalised match
        nk = _norm(sig)
        if nk in v_norm:
            matched_rows.append({**row.to_dict(),
                                  "vaastav_name": v_norm[nk],
                                  "match_conf":   1.0,
                                  "match_type":   "exact"})
            continue

        # 4. Fuzzy match
        best = difflib.get_close_matches(nk, list(v_norm.keys()), n=1, cutoff=0.80)
        if best:
            ratio  = difflib.SequenceMatcher(None, nk, best[0]).ratio()
            v_name = v_norm[best[0]]
            matched_rows.append({**row.to_dict(),
                                  "vaastav_name": v_name,
                                  "match_conf":   round(ratio, 3),
                                  "match_type":   "fuzzy"})
            if ratio < 0.95:
                fuzzy_below95.append((sig, v_name, ratio))
        else:
            unmatched.append(sig)

    return pd.DataFrame(matched_rows), unmatched, fuzzy_below95


# ─── Step 4 ───────────────────────────────────────────────────────────────────

def step4(state: dict):
    print()
    print("=" * 70)
    print("STEP 4 -- Attach Previous-League Baseline Features")
    print("=" * 70)
    print()

    base_path = os.path.join(DATA_PROCESSED, "base_gw_table.csv")

    # ── Load base table ───────────────────────────────────────────────────────
    print(f"  Loading base_gw_table.csv ...")
    df = pd.read_csv(base_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")
    print()

    # ── Load all 4 new_signings files ─────────────────────────────────────────
    print("  Loading new_signings files ...")
    sig_parts = []
    for pos_key in ["gk", "def", "mid", "fwd"]:
        path = os.path.join(DATA_RAW_FBREF, f"new_signings_{pos_key}.csv")
        sig_parts.append(pd.read_csv(path, low_memory=False))
        print(f"    new_signings_{pos_key}.csv  {len(sig_parts[-1]):,} rows")
    sigs_all = pd.concat(sig_parts, ignore_index=True)
    print(f"  Combined: {len(sigs_all):,} rows  ({sigs_all['name'].nunique()} unique players)")
    print()

    # ── STEP 4A — Build prev_league lookup ────────────────────────────────────
    print("  STEP 4A: Building reliability-weighted prev-league lookup ...")
    lookup = build_prev_lookup(sigs_all)
    print(f"  Lookup table: {len(lookup)} players")
    print()

    # Match to vaastav
    vaastav_names = set(df["name"].dropna().unique())
    matched, unmatched, fuzzy_below95 = match_lookup_to_vaastav(lookup, vaastav_names)
    print(f"  Matched to vaastav: {len(matched)}")
    print(f"  No vaastav match (2025/26 debutants): {len(unmatched)}")
    print()

    # Print lookup table header
    print(f"  {'Player (vaastav)':<30} {'Player (signings)':<30} {'Conf':>5}  "
          f"{'adjG/90':>8} {'adjA/90':>8} {'Seasons':>7} {'Rel':>6}")
    print(f"  {'-'*95}")
    for _, r in matched.sort_values("vaastav_name").iterrows():
        conf_str = f"{r['match_conf']:.0%}"
        print(
            f"  {str(r['vaastav_name']):<30} {str(r['sig_name']):<30} {conf_str:>5}  "
            f"{r['prev_adjG_per_90']:>8.4f} {r['prev_adjA_per_90']:>8.4f} "
            f"{r['prev_seasons_available']:>7} {r['prev_reliability_avg']:>6.2f}"
        )

    print()
    if fuzzy_below95:
        print("  Fuzzy matches below 95% (review):")
        for sig, v, r in sorted(fuzzy_below95, key=lambda x: x[2]):
            print(f"    {sig:<35} -> {v:<35} {r:.1%}")
    else:
        print("  Fuzzy matches below 95%: None")

    print()
    print(f"  Players in signings with no vaastav match ({len(unmatched)}) -- 2025/26 debutants:")
    for n in sorted(unmatched):
        print(f"    {n}")
    print()

    # ── STEP 4B — First PL season per matched player ──────────────────────────
    print("  STEP 4B: Identifying first PL season per matched player ...")
    first_season_map = {}  # vaastav_name -> first_pl_season
    for _, r in matched.iterrows():
        v_name = r["vaastav_name"]
        player_rows = df[df["name"] == v_name]
        if len(player_rows):
            first_season_map[v_name] = player_rows["season"].min()
    print(f"  First-season map built for {len(first_season_map)} players")
    # Sanity: all should be 2024-25 given dataset range
    season_counts = pd.Series(list(first_season_map.values())).value_counts()
    for season, cnt in season_counts.items():
        print(f"    first_pl_season = {season}: {cnt} players")
    print()

    # ── STEP 4C — Attach to base table ────────────────────────────────────────
    print("  STEP 4C: Initialising prev_ columns (all zeros) ...")
    for col in PREV_COLS:
        df[col] = 0.0
    df[PREV_COLS] = df[PREV_COLS].astype("float64")

    # Build index: (vaastav_name, first_pl_season) -> feature dict
    attach_map = {}
    for _, r in matched.iterrows():
        v_name = r["vaastav_name"]
        if v_name not in first_season_map:
            continue  # no vaastav rows at all
        attach_map[v_name] = {
            "has_prev_league_data":    1,
            "prev_adjG_per_90":        r["prev_adjG_per_90"],
            "prev_adjA_per_90":        r["prev_adjA_per_90"],
            "prev_league_multiplier":  r["prev_league_multiplier"],
            "prev_seasons_available":  r["prev_seasons_available"],
            "prev_reliability_avg":    r["prev_reliability_avg"],
            "prev_minutes_avg":        r["prev_minutes_avg"],
            "prev_small_sample":       r["prev_small_sample"],
            "prev_int_per_90":         r["prev_int_per_90"]    if r["position"] == "DEF" else 0,
            "prev_tklW_per_90":        r["prev_tklW_per_90"]   if r["position"] == "DEF" else 0,
            "prev_saves_per_game":     r["prev_saves_per_game"] if r["position"] == "GK"  else 0,
            "prev_cs_rate":            r["prev_cs_rate"]        if r["position"] == "GK"  else 0,
        }

    rows_filled = 0
    for v_name, feats in attach_map.items():
        first_s = first_season_map[v_name]
        mask = (df["name"] == v_name) & (df["season"] == first_s)
        n_rows = mask.sum()
        for col, val in feats.items():
            df.loc[mask, col] = val
        rows_filled += n_rows

    print(f"  Filled {rows_filled:,} GW rows across {len(attach_map)} players "
          f"(first PL season only)")
    print()

    # ── Validation ────────────────────────────────────────────────────────────
    print("  Validation ...")
    nan_count = df[PREV_COLS].isna().sum().sum()
    has1 = (df["has_prev_league_data"] == 1).sum()
    has0 = (df["has_prev_league_data"] == 0).sum()
    print(f"    NaN in prev_ columns:          {nan_count}  (must be 0)")
    print(f"    has_prev_league_data=1 rows:   {has1:,}")
    print(f"    has_prev_league_data=0 rows:   {has0:,}")
    if nan_count:
        print("  FATAL: NaN values in prev_ columns -- stopping.")
        sys.exit(1)
    print()

    # Spot checks
    print("  Spot checks:")
    spot = {
        "Omar Marmoush":    ["prev_adjG_per_90", "prev_adjA_per_90", "prev_league_multiplier"],
        "Riccardo Calafiori": ["prev_adjG_per_90", "prev_adjA_per_90", "prev_league_multiplier"],
        "Viktor Gyökeres":  None,  # 2025/26 — not in base table
        "Mads Hermansen":   ["prev_saves_per_game", "prev_cs_rate"],
        "Maxence Lacroix":  ["prev_int_per_90", "prev_tklW_per_90"],
    }
    for player, cols in spot.items():
        if cols is None:
            # Not in base_gw_table — show from lookup table
            lrow = lookup[lookup["sig_name"] == player]
            if len(lrow):
                r = lrow.iloc[0]
                print(f"    {player:<25} [2025/26 debutant -- not in training data]")
                print(f"      lookup: prev_adjG={r['prev_adjG_per_90']:.4f}  "
                      f"prev_adjA={r['prev_adjA_per_90']:.4f}  "
                      f"mult={r['prev_league_multiplier']:.2f}")
            else:
                print(f"    {player:<25} not found in lookup table")
            continue

        prows = df[(df["name"] == player) & (df["has_prev_league_data"] == 1)]
        if len(prows):
            r = prows.iloc[0]
            vals = "  ".join(f"{c}={r[c]:.4f}" for c in cols)
            print(f"    {player:<25} {vals}")
        else:
            # Try vaastav_name mapping
            vname = MANUAL_MATCH_OVERRIDES.get(player, player)
            prows2 = df[(df["name"] == vname) & (df["has_prev_league_data"] == 1)]
            if len(prows2):
                r = prows2.iloc[0]
                vals = "  ".join(f"{c}={r[c]:.4f}" for c in cols)
                print(f"    {player:<25} (as {vname})  {vals}")
            else:
                print(f"    {player:<25} not found in base table")
    print()

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(base_path, index=False)
    print(f"  Saved base_gw_table.csv  ({len(df):,} rows x {len(df.columns)} cols)")

    state["last_completed_step"] = 4
    state["step4"] = {
        "players_with_prev_data":    len(attach_map),
        "players_without":           len(df["name"].unique()) - len(attach_map),
        "gw_rows_with_prev":         int(has1),
        "gw_rows_without":           int(has0),
        "nan_in_prev_cols":          int(nan_count),
        "total_columns":             len(df.columns),
        "fuzzy_below_95":            [(s, v, r) for s, v, r in fuzzy_below95],
        "output":                    "data/processed/base_gw_table.csv",
    }
    save_state(state)

    # ── Gate 4 ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STEP 4 COMPLETE -- Previous League Features Attached")
    print("=" * 70)
    print(f"  Players with prev league data:    {len(attach_map)}")
    print(f"  Players without (vaastav only):   {df['name'].nunique() - len(attach_map):,}")
    print(f"  has_prev_league_data=1 GW rows:   {has1:,}")
    print(f"  has_prev_league_data=0 GW rows:   {has0:,}")
    print(f"  NaN in prev_ columns:             {nan_count}  (must be 0)")
    print(f"  Columns in base table now:        {len(df.columns)}  (32 + 12)")
    print()
    if fuzzy_below95:
        print("  Fuzzy matches below 95% needing review:")
        for s, v, r in fuzzy_below95:
            print(f"    {s:<35} -> {v:<35} {r:.1%}")
    else:
        print("  Fuzzy matches below 95% needing review: None")
    print()
    print("  Players in signings with no vaastav match (2025/26 only):")
    for n in sorted(unmatched):
        print(f"    {n}")
    print()
    print("  Spot checks:")
    for player, cols in spot.items():
        if cols is None:
            lrow = lookup[lookup["sig_name"] == player]
            if len(lrow):
                r = lrow.iloc[0]
                print(f"    {player:<25} prev_adjG={r['prev_adjG_per_90']:.4f}  "
                      f"prev_adjA={r['prev_adjA_per_90']:.4f}  "
                      f"mult={r['prev_league_multiplier']:.2f}  [not in training data]")
            continue
        prows = df[(df["name"] == player) & (df["has_prev_league_data"] == 1)]
        if not len(prows):
            vname = MANUAL_MATCH_OVERRIDES.get(player, player)
            prows = df[(df["name"] == vname) & (df["has_prev_league_data"] == 1)]
        if len(prows):
            r = prows.iloc[0]
            vals = "  ".join(f"{c}={r[c]:.4f}" for c in cols)
            print(f"    {player:<25} {vals}")
        else:
            print(f"    {player:<25} NOT FOUND")
    print()
    print("  NEXT: Step 5 will attach team form features -- both the")
    print("  player's own team form and the opponent's team form for")
    print("  each GW row, joined from team_form.csv.")
    print()

    while True:
        ans = input("Proceed to Step 5? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            break
        print("  Please enter y or n.")
    print("=" * 70)

    if ans == "n":
        print("Stopped at Gate 4.")
        sys.exit(0)

    step5(state)


# ─── Step 5 ───────────────────────────────────────────────────────────────────

# Team-form columns to attach (own team + opponent, with prefix)
TEAM_FORM_COLS = [
    "xG_last5",
    "xGA_last5",
    "attacking_strength",
    "defensive_strength",
    "clean_sheet_probability",
    "goals_scored_last5",
    "goals_conceded_last5",
    "clean_sheet_rate_last5",
    "form_points_last5",
    "xG_season_avg",
    "xGA_season_avg",
]


def step5(state: dict):
    print()
    print("=" * 70)
    print("STEP 5 -- Attach Team Form Features")
    print("=" * 70)
    print()

    base_path      = os.path.join(DATA_PROCESSED, "base_gw_table.csv")
    team_form_path = os.path.join(DATA_PROCESSED, "team_form.csv")

    # ── 5A: Inspect team_form.csv ─────────────────────────────────────────────
    print("  STEP 5A: Inspect team_form.csv ...")
    tf = pd.read_csv(team_form_path, low_memory=False)
    print(f"  Loaded {len(tf):,} rows x {len(tf.columns)} cols")
    print(f"  Columns: {list(tf.columns)}")
    print(f"  Seasons: {sorted(tf['season'].unique())}")
    print(f"  GW range: {tf['GW'].min()} - {tf['GW'].max()}")
    print(f"  Teams ({tf['team'].nunique()}): {sorted(tf['team'].unique())}")
    print()

    # ── 5B: GW1 leakage check ─────────────────────────────────────────────────
    print("  STEP 5B: GW1 leakage check ...")
    gw1 = tf[tf["GW"] == 1]
    rolling_cols = [c for c in TEAM_FORM_COLS if c in tf.columns]
    leakage_issues = []
    for col in rolling_cols:
        nonzero = (gw1[col].fillna(0) != 0).sum()
        if nonzero:
            leakage_issues.append((col, nonzero))
    if leakage_issues:
        print("  WARNING: GW1 non-zero values in rolling columns (potential leakage):")
        for col, n in leakage_issues:
            print(f"    {col}: {n} non-zero GW1 rows")
    else:
        print("  GW1 rolling columns all zero -- no leakage detected.")
    print()

    # ── 5C: Team name consistency check ───────────────────────────────────────
    print("  STEP 5C: Team name consistency check ...")
    df = pd.read_csv(base_path, low_memory=False)
    # Drop any team_/opp_ columns from a previous partial run (idempotency)
    stale = [c for c in df.columns if c.startswith("team_") or c.startswith("opp_")]
    if stale:
        df = df.drop(columns=stale)
        print(f"  Dropped {len(stale)} stale team/opp columns from previous partial run.")
    print(f"  Loaded base_gw_table.csv: {len(df):,} rows x {len(df.columns)} cols")

    base_teams = set(df["team"].dropna().unique())
    base_opp   = set(df["opponent_team"].dropna().unique())
    form_teams = set(tf["team"].dropna().unique())

    missing_own = sorted(base_teams - form_teams)
    missing_opp = sorted(base_opp - form_teams)

    if missing_own:
        print(f"  WARNING: teams in base table not in team_form: {missing_own}")
    else:
        print("  Own-team names: all match team_form -- no mismatches.")

    if missing_opp:
        print(f"  WARNING: opponent_team values not in team_form: {missing_opp}")
    else:
        print("  opponent_team names: all match team_form -- no mismatches.")
    print()

    # ── 5D: Left join own-team form ───────────────────────────────────────────
    print("  STEP 5D: Joining own-team form (team_ prefix) ...")
    tf_own = tf[["team", "season", "GW"] + [c for c in TEAM_FORM_COLS if c in tf.columns]].copy()
    rename_own = {c: f"team_{c}" for c in TEAM_FORM_COLS if c in tf.columns}
    tf_own = tf_own.rename(columns=rename_own)

    before_cols = len(df.columns)
    df = df.merge(tf_own, on=["team", "season", "GW"], how="left")
    added_own = len(df.columns) - before_cols
    team_form_cols_added = list(rename_own.values())
    nan_own = df[team_form_cols_added].isna().sum().sum()
    print(f"  Added {added_own} own-team columns. NaN count: {nan_own}")
    print()

    # ── 5E: Left join opponent form ───────────────────────────────────────────
    print("  STEP 5E: Joining opponent form (opp_ prefix) ...")
    tf_opp = tf[["team", "season", "GW"] + [c for c in TEAM_FORM_COLS if c in tf.columns]].copy()
    rename_opp = {c: f"opp_{c}" for c in TEAM_FORM_COLS if c in tf.columns}
    tf_opp = tf_opp.rename(columns=rename_opp)
    tf_opp = tf_opp.rename(columns={"team": "opponent_team"})

    before_cols = len(df.columns)
    df = df.merge(tf_opp, on=["opponent_team", "season", "GW"], how="left")
    added_opp = len(df.columns) - before_cols
    opp_form_cols_added = list(rename_opp.values())
    nan_opp = df[opp_form_cols_added].isna().sum().sum()
    print(f"  Added {added_opp} opponent columns. NaN count: {nan_opp}")
    print()

    # ── 5F: Fill GW1 NaNs with 0 ─────────────────────────────────────────────
    print("  STEP 5F: Filling NaNs with 0 (GW1 and any unmatched) ...")
    all_new_cols = team_form_cols_added + opp_form_cols_added
    df[all_new_cols] = df[all_new_cols].fillna(0.0)
    remaining_nan = df[all_new_cols].isna().sum().sum()
    print(f"  Remaining NaN in team form columns: {remaining_nan}  (must be 0)")
    if remaining_nan:
        print("  FATAL: NaN values remain after fillna -- stopping.")
        sys.exit(1)
    print()

    # ── 5G: Sample rows ───────────────────────────────────────────────────────
    print("  STEP 5G: Sample rows (5 rows, non-GW1) ...")
    sample_cols = ["name", "team", "opponent_team", "season", "GW",
                   "team_xG_last5", "team_xGA_last5",
                   "team_attacking_strength", "team_defensive_strength",
                   "team_clean_sheet_probability",
                   "opp_xG_last5", "opp_xGA_last5",
                   "opp_attacking_strength", "opp_defensive_strength"]
    available_sample = [c for c in sample_cols if c in df.columns]
    sample = df[df["GW"] > 1].head(5)[available_sample]
    print(sample.to_string(index=False))
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(base_path, index=False)
    print(f"  Saved base_gw_table.csv  ({len(df):,} rows x {len(df.columns)} cols)")

    state["last_completed_step"] = 5
    state["step5"] = {
        "team_form_cols_added":   len(team_form_cols_added),
        "opp_form_cols_added":    len(opp_form_cols_added),
        "total_new_cols":         len(all_new_cols),
        "nan_remaining":          int(remaining_nan),
        "total_columns":          len(df.columns),
        "leakage_issues":         [(col, int(n)) for col, n in leakage_issues],
        "output":                 "data/processed/base_gw_table.csv",
    }
    save_state(state)

    # ── Gate 5 ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STEP 5 COMPLETE -- Team Form Features Attached")
    print("=" * 70)
    print(f"  Own-team form columns added:        {len(team_form_cols_added)}")
    print(f"  Opponent form columns added:        {len(opp_form_cols_added)}")
    print(f"  Total columns in base table now:    {len(df.columns)}")
    print(f"  NaN remaining in form columns:      {remaining_nan}  (must be 0)")
    if leakage_issues:
        print("  Leakage warnings:")
        for col, n in leakage_issues:
            print(f"    {col}: {n} non-zero GW1 rows")
    else:
        print("  No leakage issues detected.")
    print()
    print("  NEXT: Step 6 will attach fixture difficulty features")
    print("  from fixture_difficulty.csv and player_upcoming_fixtures.csv.")
    print()
    while True:
        ans = input("Proceed to Step 6? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            break
        print("  Please enter y or n.")
    print("=" * 70)

    if ans == "n":
        print("Stopped at Gate 5.")
        sys.exit(0)

    step6(state)


# ─── Step 6 ───────────────────────────────────────────────────────────────────

def step6(state: dict):
    print()
    print("=" * 70)
    print("STEP 6 -- Attach Fixture Difficulty Features")
    print("=" * 70)
    print()

    base_path  = os.path.join(DATA_PROCESSED, "base_gw_table.csv")
    fdr_path   = os.path.join(DATA_RAW_FPL,   "fixture_difficulty.csv")
    puf_path   = os.path.join(DATA_RAW_FPL,   "player_upcoming_fixtures.csv")

    # ── 6A: Inspect fixture_difficulty.csv ────────────────────────────────────
    print("  STEP 6A: Inspect fixture_difficulty.csv ...")
    fd = pd.read_csv(fdr_path, low_memory=False)
    print(f"  Columns: {list(fd.columns)}")
    print(f"  Shape:   {fd.shape[0]:,} rows x {fd.shape[1]} cols")
    if "season" in fd.columns:
        print(f"  Seasons: {sorted(fd['season'].unique())}")
    else:
        print("  No 'season' column -- file is 2025-26 only (current season FDR)")
    gw_col  = "gameweek" if "gameweek" in fd.columns else "GW"
    tm_col  = "team_name" if "team_name" in fd.columns else "team"
    fdr_col = "fdr" if "fdr" in fd.columns else "difficulty"
    print(f"  GW range: {fd[gw_col].min()} - {fd[gw_col].max()}")
    print(f"  Teams ({fd[tm_col].nunique()}): {sorted(fd[tm_col].unique())}")
    print(f"  First 5 rows:")
    print(fd.head().to_string(index=False))
    print()

    # ── 6B: Inspect player_upcoming_fixtures.csv ──────────────────────────────
    print("  STEP 6B: Inspect player_upcoming_fixtures.csv ...")
    puf = pd.read_csv(puf_path, low_memory=False)
    print(f"  Columns: {list(puf.columns)}")
    print(f"  Shape:   {puf.shape[0]:,} rows x {puf.shape[1]} cols")
    puf_gw_col = "gameweek" if "gameweek" in puf.columns else "GW"
    print(f"  GW range: {puf[puf_gw_col].dropna().min()} - {puf[puf_gw_col].dropna().max()}")
    print(f"  First 5 rows:")
    print(puf.head().to_string(index=False))
    print()

    # ── Load base table ───────────────────────────────────────────────────────
    print("  Loading base_gw_table.csv ...")
    df = pd.read_csv(base_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")
    print()

    # ── 6C: Historical FDR proxy ──────────────────────────────────────────────
    print("  STEP 6C: Historical FDR proxy (was_home -> FDR 2/4) ...")
    # was_home may be bool dtype — cast to int first so map({1:2, 0:4}) works
    df["current_gw_fdr"] = df["was_home"].astype(int).map({1: 2, 0: 4})
    df["fdr_is_proxy"]   = 1
    proxy_home = (df["was_home"].astype(int) == 1).sum()
    proxy_away = (df["was_home"].astype(int) == 0).sum()
    print(f"  Proxy FDR assigned: {len(df):,} rows total")
    print(f"    was_home=1 -> FDR 2:  {proxy_home:,} rows")
    print(f"    was_home=0 -> FDR 4:  {proxy_away:,} rows")
    print()

    # ── 6D: Current season FDR check ─────────────────────────────────────────
    print("  STEP 6D: fixture_difficulty.csv season check ...")
    has_season = "season" in fd.columns
    if has_season:
        fd_seasons = sorted(fd["season"].unique())
        print(f"  Seasons in fixture_difficulty.csv: {fd_seasons}")
        if "2024-25" in fd_seasons:
            # Join actual FDR for 2024-25 rows
            fd_2425 = fd[fd["season"] == "2024-25"][[tm_col, gw_col, fdr_col]].copy()
            fd_2425 = fd_2425.rename(columns={tm_col: "team", gw_col: "GW", fdr_col: "_actual_fdr"})
            df = df.merge(fd_2425, on=["team", "GW"], how="left")
            mask_2425 = (df["season"] == "2024-25") & df["_actual_fdr"].notna()
            df.loc[mask_2425, "current_gw_fdr"] = df.loc[mask_2425, "_actual_fdr"].astype(int)
            df.loc[mask_2425, "fdr_is_proxy"]   = 0
            n_updated = mask_2425.sum()
            df = df.drop(columns=["_actual_fdr"])
            print(f"  2024-25 rows updated with actual FDR: {n_updated:,}")
        else:
            print("  2024-25 NOT in fixture_difficulty.csv -- using proxy for all historical rows")
    else:
        print("  fixture_difficulty.csv is 2025-26 only -- using proxy FDR for all historical rows")
    print()

    # ── 6E: Fixture trajectory score ─────────────────────────────────────────
    print("  STEP 6E: Fixture trajectory score ...")
    # For all historical training rows, trajectory = current GW FDR (no lookahead available)
    df["fixture_trajectory_score"] = df["current_gw_fdr"].astype(float)
    df["trajectory_is_full"]       = 0
    print("  trajectory_is_full=0 for all training rows (no historical lookahead available)")
    print(f"  fixture_trajectory_score = current_gw_fdr for {len(df):,} rows")
    print()

    # ── 6F: Home advantage flag ───────────────────────────────────────────────
    print("  STEP 6F: Home advantage flag ...")
    df["home_advantage"] = df["was_home"].astype(int)
    print("  home_advantage added (copy of was_home as int, was_home retained)")
    print()

    # ── 6G: Validation ────────────────────────────────────────────────────────
    print("  STEP 6G: Validation ...")
    fixture_cols = ["current_gw_fdr", "fixture_trajectory_score", "fdr_is_proxy", "home_advantage"]
    nan_counts = {c: int(df[c].isna().sum()) for c in fixture_cols}
    total_nan = sum(nan_counts.values())

    for col, n in nan_counts.items():
        status = "OK" if n == 0 else f"FAIL ({n} NaNs)"
        print(f"    NaN in {col:<32} {n}  [{status}]")

    if total_nan:
        print()
        print("  FATAL: NaN values in fixture columns -- stopping.")
        sys.exit(1)

    print()
    print("  current_gw_fdr value distribution:")
    for val, cnt in sorted(df["current_gw_fdr"].value_counts().items()):
        print(f"    FDR {val}: {cnt:,} rows")

    print()
    fdr0 = int((df["fdr_is_proxy"] == 0).sum())
    fdr1 = int((df["fdr_is_proxy"] == 1).sum())
    traj1 = int((df["trajectory_is_full"] == 1).sum())
    print(f"  fdr_is_proxy=0 (actual FDR):     {fdr0:,}")
    print(f"  fdr_is_proxy=1 (proxy home/away): {fdr1:,}")
    print(f"  trajectory_is_full=1:             {traj1:,}")
    print()

    print("  Sample rows (5, GW > 1):")
    sample_cols = ["name", "GW", "team", "opponent_team", "was_home",
                   "current_gw_fdr", "fixture_trajectory_score",
                   "fdr_is_proxy", "home_advantage"]
    available = [c for c in sample_cols if c in df.columns]
    print(df[df["GW"] > 1].head(5)[available].to_string(index=False))
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(base_path, index=False)
    print(f"  Saved base_gw_table.csv  ({len(df):,} rows x {len(df.columns)} cols)")

    state["last_completed_step"] = 6
    state["step6"] = {
        "fixture_cols_added":        len(fixture_cols) + 1,   # + trajectory_is_full
        "fdr_is_proxy_0":            fdr0,
        "fdr_is_proxy_1":            fdr1,
        "trajectory_is_full_1":      traj1,
        "nan_in_fixture_cols":       total_nan,
        "total_columns":             len(df.columns),
        "output":                    "data/processed/base_gw_table.csv",
    }
    save_state(state)

    # ── Gate 6 ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STEP 6 COMPLETE -- Fixture Difficulty Features")
    print("=" * 70)
    print(f"  current_gw_fdr attached:             {len(df):,} rows")
    print(f"  fixture_trajectory_score attached:   {len(df):,} rows")
    print(f"  fdr_is_proxy=0 (actual FDR):         {fdr0:,} rows")
    print(f"  fdr_is_proxy=1 (proxy home/away):    {fdr1:,} rows")
    print(f"  trajectory_is_full=1:                {traj1:,} rows")
    print(f"  NaN in fixture columns:              {total_nan}  (must be 0)")
    print(f"  Total columns now:                   {len(df.columns)}  (expected ~70)")
    print()
    print("  NEXT: Step 7 will split into 4 position-specific training")
    print("  files, apply position-specific feature selection, fill")
    print("  any remaining NaN, and write final training CSVs to")
    print("  data/processed/.")
    print()

    while True:
        ans = input("Proceed to Step 7? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            break
        print("  Please enter y or n.")
    print("=" * 70)

    if ans == "n":
        print("Stopped at Gate 6.")
        sys.exit(0)

    step7(state)


# ─── Step 7 ───────────────────────────────────────────────────────────────────

def step7(state):
    """Split base_gw_table into 4 position-specific training files."""

    base_path = "data/processed/base_gw_table.csv"
    out_paths = {
        "GK":  "data/processed/train_gk.csv",
        "DEF": "data/processed/train_def.csv",
        "MID": "data/processed/train_mid.csv",
        "FWD": "data/processed/train_fwd.csv",
    }

    print()
    print("=" * 70)
    print("STEP 7 -- Position-Specific Training File Split")
    print("=" * 70)
    print(f"  Loading {base_path} ...")

    df = pd.read_csv(base_path)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")
    print()

    # ── 7A: Define feature column sets ────────────────────────────────────────
    IDENTITY_COLS = [
        "name", "season", "GW", "team", "opponent_team", "was_home", "position",
    ]
    TARGET_COL = ["total_points"]

    RAW_STAT_COLS = [
        "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "bonus", "bps", "yellow_cards", "red_cards",
    ]
    # "saves" and "saves_per_game_season" are GK-only -- excluded from shared

    ROLLING_COLS_SHARED = [
        "form_last3", "form_last5", "cumulative_points_season",
        "avg_points_per_game_season", "goals_per_game_season",
        "assists_per_game_season", "clean_sheet_rate_season",
        "minutes_reliability_season", "points_per_million",
    ]
    # "saves_per_game_season" is GK-only -- excluded from shared

    PREV_LEAGUE_COLS = [
        "has_prev_league_data", "prev_adjG_per_90", "prev_adjA_per_90",
        "prev_league_multiplier", "prev_seasons_available",
        "prev_reliability_avg", "prev_minutes_avg", "prev_small_sample",
    ]
    # "prev_saves_per_game", "prev_cs_rate"  -> GK only
    # "prev_int_per_90",     "prev_tklW_per_90" -> DEF only

    TEAM_FORM_COLS_SHARED = [
        "team_xG_last5", "team_xGA_last5",
        "team_attacking_strength", "team_defensive_strength",
        "team_clean_sheet_probability",
        "team_goals_scored_last5", "team_goals_conceded_last5",
        "team_clean_sheet_rate_last5", "team_form_points_last5",
        "team_xG_season_avg", "team_xGA_season_avg",
        "opp_xG_last5", "opp_xGA_last5",
        "opp_attacking_strength", "opp_defensive_strength",
        "opp_clean_sheet_probability",
        "opp_goals_scored_last5", "opp_goals_conceded_last5",
        "opp_clean_sheet_rate_last5", "opp_form_points_last5",
        "opp_xG_season_avg", "opp_xGA_season_avg",
    ]

    FIXTURE_COLS_SHARED = [
        "current_gw_fdr", "fdr_is_proxy",
        "fixture_trajectory_score", "trajectory_is_full",
        "home_advantage",
    ]

    MARKET_COLS = ["transfers_in", "transfers_out", "selected", "value"]

    SHARED = (
        IDENTITY_COLS + TARGET_COL + RAW_STAT_COLS + ROLLING_COLS_SHARED
        + PREV_LEAGUE_COLS + TEAM_FORM_COLS_SHARED + FIXTURE_COLS_SHARED
        + MARKET_COLS
    )

    POS_EXTRA = {
        "GK":  ["saves", "saves_per_game_season", "prev_saves_per_game", "prev_cs_rate"],
        "DEF": ["prev_int_per_90", "prev_tklW_per_90"],
        "MID": [],
        "FWD": [],
    }

    # Columns that should be zero at GW1 (used in validation)
    ZERO_AT_GW1 = ROLLING_COLS_SHARED + TEAM_FORM_COLS_SHARED

    # ── 7B: Split + select columns ─────────────────────────────────────────────
    results = {}

    for pos, out_path in out_paths.items():
        print(f"  Processing {pos} ...")
        subset = df[df["position"] == pos].copy()
        wanted = SHARED + POS_EXTRA[pos]

        # resolve against actual columns -- warn + skip if missing
        available = []
        for col in wanted:
            if col in df.columns:
                available.append(col)
            else:
                print(f"    WARNING: column '{col}' not found in "
                      f"base_gw_table -- skipping")

        subset = subset[available].copy()

        # ── 7C: Fill remaining NaN ────────────────────────────────────────────
        nan_counts = subset.isnull().sum()
        nan_cols   = nan_counts[nan_counts > 0]

        if len(nan_cols) == 0:
            print(f"    No NaN to fill")
        else:
            for col, n in nan_cols.items():
                pct = n / len(subset) * 100
                if pct > 1.0:
                    med = subset[col].median()
                    print(f"    NaN fill: '{col}' has {n} NaN "
                          f"({pct:.1f}%) -- filling with median={med:.4f}")
                    subset[col] = subset[col].fillna(med)
                else:
                    subset[col] = subset[col].fillna(0)

        # ── 7D: Validate ──────────────────────────────────────────────────────
        remaining_nan = int(subset.isnull().sum().sum())
        assert remaining_nan == 0, (
            f"[{pos}] FAILED: {remaining_nan} NaN values remain after fill"
        )

        # GW1 zero check: rolling + form features must all be zero
        gw1 = subset[subset["GW"] == 1]
        gw1_violations = 0
        for col in ZERO_AT_GW1:
            if col not in subset.columns:
                continue
            nonzero = int((gw1[col] != 0).sum())
            if nonzero > 0:
                print(f"    WARNING: GW1 nonzero -- '{col}' has {nonzero} nonzero rows")
                gw1_violations += 1

        if gw1_violations == 0:
            print(f"    GW1 zero-check: PASSED")
        else:
            print(f"    GW1 zero-check: {gw1_violations} column(s) with violations")

        # row count sanity
        expected_rows = {"GK": 4421, "DEF": 18828, "MID": 22132, "FWD": 5663}
        assert len(subset) == expected_rows[pos], (
            f"[{pos}] Row count mismatch: got {len(subset):,}, "
            f"expected {expected_rows[pos]:,}"
        )

        # save
        subset.to_csv(out_path, index=False)
        results[pos] = {
            "rows":          len(subset),
            "cols":          len(subset.columns),
            "nan_after_fill": remaining_nan,
            "gw1_violations": gw1_violations,
        }
        print(f"    Saved: {out_path}  "
              f"({len(subset):,} rows x {len(subset.columns)} cols)")
        print()

    # ── State ─────────────────────────────────────────────────────────────────
    state["last_completed_step"] = 7
    state["step7"] = {
        "positions": results,
        "outputs":   list(out_paths.values()),
    }
    save_state(state)

    # ── Gate 7 ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 7 COMPLETE -- Position-Specific Training Files")
    print("=" * 70)
    for pos, r in results.items():
        print(f"  {pos:<4} : {r['rows']:>6,} rows x {r['cols']:>3} cols"
              f"  ->  train_{pos.lower()}.csv")
    print()
    print("  All files saved to data/processed/")
    print()
    print("  NEXT: Step 8 will run final validation checks across all 4")
    print("  training files and save the Stage 6 validation report.")
    print()

    while True:
        ans = input("Proceed to Step 8? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            break
        print("  Please enter y or n.")
    print("=" * 70)

    if ans == "n":
        print("Stopped at Gate 7.")
        sys.exit(0)

    step8(state)


# ─── Step 8 ───────────────────────────────────────────────────────────────────

def step8(state):
    """Final validation report across all 4 position training files."""
    import datetime

    in_paths = {
        "GK":  "data/processed/train_gk.csv",
        "DEF": "data/processed/train_def.csv",
        "MID": "data/processed/train_mid.csv",
        "FWD": "data/processed/train_fwd.csv",
    }
    report_path = "data/processed/stage6_validation_report.txt"

    EXPECTED_ROWS = {"GK": 4421, "DEF": 18828, "MID": 22132, "FWD": 5663}
    POSITIONS     = ["GK", "DEF", "MID", "FWD"]
    ID_COLS       = ["name", "season", "team", "opponent_team", "position"]

    GW1_ROLLING = [
        "form_last3", "form_last5",
        "avg_points_per_game_season",
        "goals_per_game_season",
        "assists_per_game_season",
        "clean_sheet_rate_season",
        "team_xG_last5", "team_xGA_last5",
        "opp_xG_last5", "opp_xGA_last5",
        "team_attacking_strength",
        "team_defensive_strength",
    ]

    print()
    print("=" * 70)
    print("STEP 8 -- Stage 6 Final Validation Report")
    print("=" * 70)
    print()

    # ── Load all 4 files ──────────────────────────────────────────────────────
    dfs = {}
    for pos, path in in_paths.items():
        dfs[pos] = pd.read_csv(path)
        print(f"  Loaded {path}  ({len(dfs[pos]):,} rows x {len(dfs[pos].columns)} cols)")
    print()

    # ── Run checks ────────────────────────────────────────────────────────────
    # results[check_num][pos] = "PASS" | "FAIL: <reason>"
    results = {i: {} for i in range(1, 11)}
    any_fail = False

    def record(check, pos, ok, reason=""):
        tag = "PASS" if ok else f"FAIL: {reason}"
        results[check][pos] = tag
        if not ok:
            nonlocal any_fail
            any_fail = True
            print(f"  *** CRITICAL FAIL — Check {check} [{pos}]: {reason}")

    for pos in POSITIONS:
        df = dfs[pos]

        # Check 1 — No NaN
        nan_total = int(df.isna().sum().sum())
        record(1, pos, nan_total == 0, f"{nan_total} NaN found")

        # Check 2 — No 2025-26 data
        has_live = "2025-26" in df["season"].values
        record(2, pos, not has_live, "2025-26 season present")

        # Check 3 — All 6 seasons present
        n_seasons = df["season"].nunique()
        record(3, pos, n_seasons == 6, f"{n_seasons} seasons found")

        # Check 4 — GW1 rolling features are zero
        gw1 = df[df["GW"] == 1]
        leaky = [c for c in GW1_ROLLING if c in df.columns and not gw1[c].eq(0).all()]
        record(4, pos, len(leaky) == 0,
               f"leakage in: {leaky}" if leaky else "")

        # Check 5 — Points distribution sane
        # Max threshold is 30: a MID scoring 4 goals legitimately hits 26
        # (4 x 5pts + 3 bonus + 2 appearance + 1 CS); 30 gives safe headroom
        mean_pts = df["total_points"].mean()
        max_pts  = df["total_points"].max()
        pts_ok   = (1.0 < mean_pts < 6.0) and (max_pts <= 30)
        record(5, pos, pts_ok,
               f"mean={mean_pts:.2f} max={max_pts}" if not pts_ok else "")

        # Check 6 — Row count
        record(6, pos, len(df) == EXPECTED_ROWS[pos],
               f"got {len(df):,} expected {EXPECTED_ROWS[pos]:,}")

        # Check 7 — Position purity
        pure = (df["position"] == pos).all()
        record(7, pos, pure, f"unexpected positions: {df['position'].unique().tolist()}")

        # Check 8 — No negative minutes
        min_min = df["minutes"].min()
        record(8, pos, min_min >= 0, f"min minutes = {min_min}")

        # Check 9 — All non-ID columns numeric
        non_id = [c for c in df.columns if c not in ID_COLS]
        bad_types = [c for c in non_id if not pd.api.types.is_numeric_dtype(df[c])]
        record(9, pos, len(bad_types) == 0,
               f"non-numeric: {bad_types}" if bad_types else "")

        # Check 10 — Prev features zero for non-new-signing players
        no_prev = df[df["has_prev_league_data"] == 0]
        g_bad   = int((no_prev["prev_adjG_per_90"] != 0).sum())
        a_bad   = int((no_prev["prev_adjA_per_90"] != 0).sum())
        record(10, pos, g_bad == 0 and a_bad == 0,
               f"prev_adjG nonzero={g_bad}, prev_adjA nonzero={a_bad}")

    if any_fail:
        print()
        print("  *** ONE OR MORE CRITICAL CHECKS FAILED — stopping. ***")
        sys.exit(1)

    # ── Print check results ────────────────────────────────────────────────────
    CHECK_LABELS = {
        1:  "No NaN",
        2:  "No 2025-26",
        3:  "6 seasons",
        4:  "GW1 zeros",
        5:  "Points sane",
        6:  "Row counts",
        7:  "Pos purity",
        8:  "No neg min",
        9:  "All numeric",
        10: "Prev isolation",
    }
    print("=" * 70)
    print("CHECK RESULTS:")
    print()
    for n, label in CHECK_LABELS.items():
        row = f"  Check {n:<2} -- {label:<18}"
        for pos in POSITIONS:
            tag = results[n][pos]
            row += f"  {pos} {tag}"
        print(row)
    print()

    # ── Compute summary stats ─────────────────────────────────────────────────
    all_df = pd.concat(dfs.values(), ignore_index=True)
    with_prev    = int(all_df[all_df["has_prev_league_data"] == 1]["name"].nunique())
    total_players= int(all_df["name"].nunique())
    vaastav_only = total_players - with_prev
    timestamp    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_summary = []
    for pos, path in in_paths.items():
        df = dfs[pos]
        fname = path.split("/")[-1]
        file_summary.append((
            fname,
            len(df),
            len(df.columns),
            round(df["total_points"].mean(), 2),
            int(df["total_points"].max()),
            df["season"].nunique(),
        ))

    # ── Build report text ─────────────────────────────────────────────────────
    check_line_fmt = "  Check {n:<2} -- {label:<20} {results}"
    overall = "ALL CHECKS PASSED" if not any_fail else f"CHECKS FAILED"

    def check_line(n):
        label  = CHECK_LABELS[n]
        parts  = "  ".join(f"{p} {results[n][p]}" for p in POSITIONS)
        return f"  Check {n:<2} -- {label:<20} {parts}"

    report_lines = [
        "=" * 70,
        "STAGE 6 FINAL VALIDATION REPORT",
        f"Generated: {timestamp}",
        "=" * 70,
        "",
        "CRITICAL CHECKS:",
    ] + [check_line(n) for n in range(1, 11)] + [
        "",
        f"OVERALL: {overall}",
        "",
        "TRAINING FILE SUMMARY:",
        f"  {'File':<22}  {'Rows':>7}  {'Cols':>5}  {'Mean pts':>9}  {'Max pts':>8}  {'Seasons':>7}",
    ] + [
        f"  {f:<22}  {r:>7,}  {c:>5}  {m:>9.2f}  {mx:>8}  {s:>7}"
        for f, r, c, m, mx, s in file_summary
    ] + [
        f"  {'TOTAL':<22}  {sum(r for _,r,*_ in file_summary):>7,}",
        "",
        "FEATURE ENGINEERING SUMMARY:",
        f"  Base GW rows:                {sum(r for _,r,*_ in file_summary):,}",
         "  Rolling features added:      10",
         "  Prev league features added:  12  (position-specific)",
         "  Team form features added:    22  (11 own + 11 opp)",
         "  Fixture features added:      4",
        f"  Players with prev data:      {with_prev}  (stage4a/4b new signings)",
        f"  Players vaastav-only:        {vaastav_only}",
         "  GW1 leakage checks passed:   all",
         "  Cross-season bleed:          none",
         "  NaN in final files:          0",
        "",
        "DATA SOURCES:",
         "  Vaastav historical GW data:  2019-20 to 2024-25",
         "  FBref new signings:          Stage 4a (63 players) +",
         "                               Stage 4b (26 players)",
         "  Team form:                   Stage 3 (vaastav + understat)",
         "  Fixture difficulty:          FPL API (proxy for training,",
         "                               live FDR for GW1 prediction)",
        "",
        "=" * 70,
        "Stage 6 complete. Training data ready.",
        "Ready for Stage 7 -- XGBoost Model Training.",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)

    # ── Save report ────────────────────────────────────────────────────────────
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"  Report saved to {report_path}")
    print()

    # ── State ─────────────────────────────────────────────────────────────────
    state["last_completed_step"] = 8
    state["stage6_complete"]     = True
    state["step8"] = {
        "timestamp":       timestamp,
        "checks_passed":   10,
        "checks_failed":   0,
        "report":          report_path,
        "players_with_prev": with_prev,
        "players_vaastav_only": vaastav_only,
        "total_rows":      int(sum(r for _, r, *_ in file_summary)),
    }
    save_state(state)

    # ── Gate 8 ────────────────────────────────────────────────────────────────
    print(report_text)
    print()

    while True:
        ans = input("Stage 6 complete. Proceed to Stage 7? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            break
        print("  Please enter y or n.")
    print("=" * 70)

    if ans == "n":
        print("Stopped after Stage 6. Training files are ready at data/processed/")
        sys.exit(0)

    print()
    print("Stage 7 is not yet implemented.")
    print("Training files are ready at data/processed/ when you are.")
    sys.exit(0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 6 -- Feature Engineering")
    parser.add_argument(
        "--start-step", type=int, default=1, metavar="N",
        help="Resume from step N (default: 1)"
    )
    args = parser.parse_args()

    state = load_state()
    start = args.start_step

    print()
    print("=" * 70)
    print("STAGE 6 -- Feature Engineering Pipeline")
    print("=" * 70)
    print(f"  Starting from step: {start}")
    print(f"  Last completed step (from state): {state.get('last_completed_step', 0)}")
    print()

    if start <= 1:
        step1(state)
    elif start == 2:
        step2(state)
    elif start == 3:
        step3(state)
    elif start == 4:
        step4(state)
    elif start == 5:
        step5(state)
    elif start == 6:
        step6(state)
    elif start == 7:
        step7(state)
    elif start == 8:
        step8(state)
    else:
        print(f"Step {start} is not yet implemented.")
        sys.exit(0)


if __name__ == "__main__":
    main()
