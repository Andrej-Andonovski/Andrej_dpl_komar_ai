"""
patch_signings_columns.py
Add missing league-adjusted columns to all 4 new_signings files.

Run: python pipeline/patch_signings_columns.py
"""

import json
import os
import sys
import unicodedata

import pandas as pd

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.join(os.path.dirname(__file__), "..")

TRANSFERS_CSV   = os.path.join(BASE, "data", "raw", "transfers", "new_signings_2025.csv")
STATE4B_JSON    = os.path.join(BASE, "data", "raw", "fbref", "stage4b_state.json")
SIGNINGS_DIR    = os.path.join(BASE, "data", "raw", "fbref", "new_signings")

SIGNINGS_FILES = {
    "gk":  os.path.join(SIGNINGS_DIR, "new_signings_gk.csv"),
    "def": os.path.join(SIGNINGS_DIR, "new_signings_def.csv"),
    "mid": os.path.join(SIGNINGS_DIR, "new_signings_mid.csv"),
    "fwd": os.path.join(SIGNINGS_DIR, "new_signings_fwd.csv"),
}

# Stage-level multipliers (same as Stage 4b constants)
LEAGUE_MULTIPLIERS = {
    "La Liga":              0.92,
    "Serie A":              0.88,
    "Bundesliga":           0.89,
    "Ligue 1":              0.82,
    "Eredivisie":           0.75,
    "Primeira Liga":        0.78,
    "Scottish Premiership": 0.65,
    "Championship":         0.72,
    "Belgian Pro League":   0.74,
    "Serie A (Brazil)":     0.70,   # Série A (Brazil) normalized
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_name(s: str) -> str:
    """NFD decompose, strip combining chars, lowercase, remove non-alpha."""
    if not isinstance(s, str):
        return ""
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return "".join(c for c in stripped.lower() if c.isalpha())


# ─── STEP 1 — Build player -> league mapping ─────────────────────────────────

def build_league_mapping() -> dict:
    print()
    print("=" * 70)
    print("STEP 1 -- Build player-to-league mapping")
    print("=" * 70)

    # ── Transfers CSV ────────────────────────────────────────────────────────
    df_tr = pd.read_csv(TRANSFERS_CSV)
    # Use previous_league_standardized (already cleaned in Stage 4a)
    mask = df_tr["previous_league_standardized"].notna()
    df_tr = df_tr[mask].copy()

    # Build exact + normalized lookup from transfers
    exact_map: dict[str, str] = {}
    norm_map: dict[str, str] = {}
    for _, row in df_tr.iterrows():
        name = str(row["fpl_name"])
        league = str(row["previous_league_standardized"]).strip()
        if league and league != "nan":
            exact_map[name] = league
            norm_map[normalize_name(name)] = league

    # ── Stage4b state ────────────────────────────────────────────────────────
    # Supplement with any confirmed players that have league info via fbref match
    try:
        with open(STATE4B_JSON, encoding="utf-8") as f:
            state4b = json.load(f)
        # stage4b state stores confirmed players; league info is in transfers csv
        # so we don't need extra data from here, but we could cross-check names
        n_confirmed = len(state4b.get("step1", {}).get("confirmed", []))
        print(f"  stage4b_state.json loaded — {n_confirmed} confirmed players")
    except FileNotFoundError:
        print("  stage4b_state.json not found — using transfers CSV only")

    print(f"  Transfers CSV: {len(exact_map)} players with known previous league")
    print()

    # Print full mapping
    print(f"  {'Player':<45} {'Previous League'}")
    print(f"  {'-'*65}")
    for name, league in sorted(exact_map.items()):
        print(f"  {name:<45} {league}")

    print()
    return exact_map, norm_map


def lookup_league(name: str, exact_map: dict, norm_map: dict) -> str:
    """Return standardized previous league, or empty string if unknown."""
    if name in exact_map:
        return exact_map[name]
    key = normalize_name(name)
    if key in norm_map:
        return norm_map[key]
    return ""


# ─── STEP 2 — Add columns to all 4 files ──────────────────────────────────────

def add_columns(pos_key: str, df: pd.DataFrame,
                exact_map: dict, norm_map: dict) -> pd.DataFrame:
    df = df.copy()

    # ── season_reliability (alias for minutes_reliability_season) ────────────
    if "season_reliability" not in df.columns:
        if "minutes_reliability_season" in df.columns:
            df["season_reliability"] = df["minutes_reliability_season"]
        else:
            df["season_reliability"] = 0.0

    # ── league_multiplier ────────────────────────────────────────────────────
    def get_multiplier(row) -> float:
        src = str(row.get("data_source", "")).strip().lower()
        if src == "vaastav":
            return 1.0
        name = str(row.get("name", ""))
        league = lookup_league(name, exact_map, norm_map)
        if not league:
            return 1.0  # unknown — conservative
        # Normalise 'Serie A (Brazil)' / 'Série A' aliases
        if "rie" in league.lower() and "bra" in league.lower():
            league = "Serie A (Brazil)"
        elif league.lower().startswith("s") and "rie" in league.lower() and "brazil" not in league.lower():
            league = "Serie A"
        return LEAGUE_MULTIPLIERS.get(league, 1.0)

    df["league_multiplier"] = df.apply(get_multiplier, axis=1)

    # ── adjG_per_90 ───────────────────────────────────────────────────────────
    def get_adjG(row) -> float:
        src = str(row.get("data_source", "")).strip().lower()
        if src == "vaastav":
            return float(row.get("goals_per_game_season", 0) or 0)
        mins = float(row.get("minutes", 0) or 0)
        goals = float(row.get("goals_scored", 0) or 0)
        mult = float(row.get("league_multiplier", 1.0) or 1.0)
        if mins > 0:
            return round((goals / mins * 90) * mult, 6)
        return 0.0

    df["adjG_per_90"] = df.apply(get_adjG, axis=1)

    # ── adjA_per_90 ───────────────────────────────────────────────────────────
    def get_adjA(row) -> float:
        src = str(row.get("data_source", "")).strip().lower()
        if src == "vaastav":
            return float(row.get("assists_per_game_season", 0) or 0)
        mins = float(row.get("minutes", 0) or 0)
        assists = float(row.get("assists", 0) or 0)
        mult = float(row.get("league_multiplier", 1.0) or 1.0)
        if mins > 0:
            return round((assists / mins * 90) * mult, 6)
        return 0.0

    df["adjA_per_90"] = df.apply(get_adjA, axis=1)

    # ── data_confidence ───────────────────────────────────────────────────────
    def get_confidence(row) -> str:
        src = str(row.get("data_source", "")).strip().lower()
        if src == "vaastav":
            return "high"
        rel = float(row.get("season_reliability", 0) or 0)
        if rel >= 0.6:
            return "high"
        if rel >= 0.3:
            return "medium"
        if rel > 0:
            return "low"
        return "none"

    df["data_confidence"] = df.apply(get_confidence, axis=1)

    # ── GK-specific ───────────────────────────────────────────────────────────
    if pos_key == "gk":
        if "prev_saves_per_game" not in df.columns:
            df["prev_saves_per_game"] = df["saves_per_game_season"] if "saves_per_game_season" in df.columns else 0.0
        if "prev_cs_rate" not in df.columns:
            df["prev_cs_rate"] = df["clean_sheet_rate_season"] if "clean_sheet_rate_season" in df.columns else 0.0

    return df


def validate_and_report(pos_key: str, df: pd.DataFrame):
    fname = f"new_signings_{pos_key}.csv"
    print(f"\n  {fname}")
    print(f"  {'-'*60}")

    new_cols = ["adjG_per_90", "adjA_per_90", "league_multiplier",
                "season_reliability", "data_confidence"]
    if pos_key == "gk":
        new_cols += ["prev_saves_per_game", "prev_cs_rate"]

    for col in new_cols:
        if col not in df.columns:
            print(f"  ERROR: {col} missing!")
            continue
        null_count = df[col].isna().sum()
        status = "OK" if null_count == 0 else f"WARNING: {null_count} nulls"
        print(f"    {col:<30} null count: {null_count:>4}  [{status}]")

    # Outlier check
    high_g = (df["adjG_per_90"] > 1.5).sum()
    if high_g:
        print(f"    adjG_per_90 > 1.5 count: {high_g}  (flag — small sample outliers)")
    else:
        print(f"    adjG_per_90 > 1.5 count: 0  [OK]")

    # Top 10 by adjG_per_90 (non-zero minutes)
    top = (df[df["minutes"] > 0]
           .nlargest(10, "adjG_per_90")
           [["name", "league_multiplier", "adjG_per_90",
             "season_reliability", "minutes"]]
           .copy())

    if len(top):
        # Look up previous league for display
        print()
        print(f"    {'Player':<35} {'Mult':>5}  {'adjG/90':>8}  {'Rel':>6}  {'Mins':>6}")
        print(f"    {'-'*65}")
        for _, r in top.iterrows():
            print(
                f"    {str(r['name']):<35} {r['league_multiplier']:>5.2f}  "
                f"{r['adjG_per_90']:>8.4f}  {r['season_reliability']:>6.2f}  "
                f"{int(r['minutes'] or 0):>6}"
            )


# ─── STEP 3 — Cross-position summary ─────────────────────────────────────────

def cross_position_summary(dfs: dict):
    print()
    print("=" * 70)
    print("STEP 3 -- Sanity check cross-position")
    print("=" * 70)
    print()
    print(f"  {'File':<22} {'Players w/ adj data':>20} {'Avg adjG/90':>12} {'Max adjG/90':>12}")
    print(f"  {'-'*70}")
    for key, df in dfs.items():
        fname = f"new_signings_{key}"
        has_data = df[df["minutes"] > 0]
        n = len(has_data)
        avg_g = has_data["adjG_per_90"].mean() if n else 0.0
        max_g = has_data["adjG_per_90"].max() if n else 0.0
        print(f"  {fname:<22} {n:>20,} {avg_g:>12.4f} {max_g:>12.4f}")


# ─── STEP 4 — Overwrite files ─────────────────────────────────────────────────

def overwrite_files(dfs: dict, orig_col_counts: dict):
    print()
    print("=" * 70)
    print("STEP 4 -- Overwrite files")
    print("=" * 70)
    print()
    for key, df in dfs.items():
        path = SIGNINGS_FILES[key]
        df.to_csv(path, index=False)
        was = orig_col_counts[key]
        now = len(df.columns)
        fname = f"new_signings_{key}.csv"
        print(f"  {fname:<25} patched -- now {now} cols  (was {was})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    exact_map, norm_map = build_league_mapping()

    # Count unknown players
    all_names = set()
    for key, path in SIGNINGS_FILES.items():
        df = pd.read_csv(path, low_memory=False)
        all_names.update(df["name"].dropna().unique())

    unknown = [n for n in sorted(all_names) if not lookup_league(n, exact_map, norm_map)]
    print(f"  Players with unknown previous league: {len(unknown)}  (will get multiplier=1.0)")
    if unknown:
        print("  Unknown players (multiplier will be 1.0):")
        for n in unknown:
            print(f"    {n}")

    # ── STEP 2 ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STEP 2 -- Add missing columns to all 4 files")
    print("=" * 70)

    dfs = {}
    orig_col_counts = {}
    for key, path in SIGNINGS_FILES.items():
        df = pd.read_csv(path, low_memory=False)
        orig_col_counts[key] = len(df.columns)
        dfs[key] = df

    patched = {}
    for key, df in dfs.items():
        patched[key] = add_columns(key, df, exact_map, norm_map)
        validate_and_report(key, patched[key])

    # ── STEP 3 ────────────────────────────────────────────────────────────────
    cross_position_summary(patched)

    # ── STEP 4 ────────────────────────────────────────────────────────────────
    overwrite_files(patched, orig_col_counts)

    print()
    print("Patch complete. Re-run Stage 6 Step 1 to verify.")


if __name__ == "__main__":
    main()
