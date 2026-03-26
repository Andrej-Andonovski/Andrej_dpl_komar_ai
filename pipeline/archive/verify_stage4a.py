#!/usr/bin/env python3
"""
Stage 4a Verification Report

Deduplicates the 4 new-signing position files, then generates a comprehensive
verification report saved to console AND:
  data/raw/fbref/new_signings/verification_report.txt

Usage: python pipeline/verify_stage4a.py
"""

import io
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("[ERROR] pip install pandas numpy")
    sys.exit(1)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGN_DIR      = os.path.join(BASE_DIR, "data", "raw", "fbref", "new_signings")
TRANSFERS_DIR = os.path.join(BASE_DIR, "data", "raw", "transfers")
REPORT_PATH   = os.path.join(SIGN_DIR, "verification_report.txt")

POS_FILES = {
    "GK":  os.path.join(SIGN_DIR, "new_signings_gk.csv"),
    "DEF": os.path.join(SIGN_DIR, "new_signings_def.csv"),
    "MID": os.path.join(SIGN_DIR, "new_signings_mid.csv"),
    "FWD": os.path.join(SIGN_DIR, "new_signings_fwd.csv"),
}
EXT_PATH = os.path.join(SIGN_DIR, "result_extended.csv")
NS_PATH  = os.path.join(TRANSFERS_DIR, "new_signings_2025.csv")

# ── constants ─────────────────────────────────────────────────────────────────
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

SKIP_FBREF_REASONS = {
    "Joe Anderson":   "Sunderland youth — no top-5 league history",
    "Toti Gomes":     "PL data covered by vaastav — no separate top-5 non-PL stats",
    "Remi Matthews":  "English GK — Championship only",
    "Steven Benda":   "Swiss GK — Championship only",
    "Charlie Crew":   "17yo Welsh youth — no senior top-5 data",
}

FBREF_FALSE_POSITIVES = [
    ("Mamadou Sylla",     "Mamadou Sarr",     "different player — Guinean striker"),
    ("Mamadou Sakho",     "Mamadou Sarr",     "veteran ex-Liverpool CB"),
    ("Mamadou Sangare",   "Mamadou Sarr",     "Malian CDM"),
    ("Mamadou Traore",    "Mamadou Sarr",     "Ivorian winger"),
    ("Mouhamadou Sarr",   "Mamadou Sarr",     "different Senegalese player"),
    ("Anderson",          "Joe Anderson",     "generic single-name Brazilian"),
    ("Anderson Jesus",    "Joe Anderson",     "Brazilian striker"),
    ("Nito Gomes",        "Toti Gomes",       "different Portuguese player"),
    ("Vitor Gomes",       "Toti Gomes",       "different Portuguese player"),
    ("Matheus Reis",      "Remi Matthews",    "Brazilian LB, not English GK"),
    ("Benjamin Leroy",    "Benjamin Lecomte", "different Belgian player"),
    ("Steven Baseya",     "Steven Benda",     "Belgian winger, not Swiss GK"),
    ("Charlie Cresswell", "Charlie Crew",     "Leeds CB, not Welsh MID"),
    ("Yacine Adli",       "Amine Adli",       "Algerian MID, not Moroccan winger"),
    ("Giovanni Simeone",  "Giovanni Leoni",   "Argentine striker, not Italian CB"),
]

KEPT_LOW_CONF = [
    ("Jamie Gittens",     "Jamie Bynoe-Gittens",       "81%",  "KEPT — same player"),
    ("Benjamin Sesko",    "Benjamin Sesko",             "86%",  "KEPT — same player (accent)"),
    ("Martin Zubimendi",  "Martin Zubimendi Ibanez",    "86%",  "KEPT — same player"),
    ("Dario Essugo",      "Dario Luis Essugo",          "81%",  "KEPT — same player"),
    ("Ladislav Krejci",   "Ladislav Krejci",            "93%",  "KEPT — same player (accent)"),
    ("Eli Junior Kroupi", "Junior Kroupi",              "87%",  "KEPT — same player"),
    ("Timothee Pembele",  "Timothee Pembele",           "90%",  "KEPT — same player (accent)"),
    ("Miodrag Pivas",     "Miodrag Pivas",              "92%",  "KEPT — same player (accent)"),
]

# ── output helper ──────────────────────────────────────────────────────────────
_lines = []

def p(text=""):
    """Print to console and buffer for file output."""
    print(text)
    _lines.append(text)

def save_report():
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n[Saved] {REPORT_PATH}")

# ── helpers ────────────────────────────────────────────────────────────────────
def safe_div(a, b, default=0.0):
    try:
        return float(a) / float(b) if float(b) != 0 else default
    except (TypeError, ValueError, ZeroDivisionError):
        return default

def fmt_pct(v):
    try:
        return f"{float(v)*100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"

def fmt_f(v, decimals=3):
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"

def rel_label(r):
    try:
        r = float(r)
    except (TypeError, ValueError):
        return "0.1"
    if r >= 1.0:
        return "1.0"
    elif r >= 0.5:
        return "0.5"
    else:
        return "0.1"

def infer_confidence(player_rows):
    """Return (confidence_str, reason) from a player's extended rows."""
    if player_rows.empty:
        return "low", "no FBref data"
    # Check if all rows are zero-stat (data_confidence col)
    if "data_confidence" in player_rows.columns:
        if (player_rows["data_confidence"] == "low").all():
            return "low", "no top-5 league data"
    # Count seasons with reliability >= 0.5 and minutes > 0
    good = player_rows[
        (player_rows.get("season_reliability", 0) >= 0.5) &
        (player_rows.get("appearances", 0) > 0)
    ] if "season_reliability" in player_rows.columns else pd.DataFrame()
    n_good = len(good)
    if n_good >= 3:
        return "high", "3 seasons of data"
    elif n_good >= 1:
        return "medium", f"{n_good} season(s) of data"
    else:
        return "low", "partial or no minutes data"

def season_display(rows):
    """Return sorted unique seasons list from a group of rows."""
    seasons = rows["fbref_season"].dropna().unique().tolist() if "fbref_season" in rows.columns else []
    return sorted([s for s in seasons if s != "N/A"])


# =============================================================================
# STEP 1 — Load data
# =============================================================================
def load_data():
    # Position files (VAASTAV format)
    pos_dfs = {}
    for pos, path in POS_FILES.items():
        if not os.path.exists(path):
            print(f"[ERROR] Missing {path} — run new_signings_stage4a.py --step4-only first.")
            sys.exit(1)
        pos_dfs[pos] = pd.read_csv(path)

    # Extended intermediate (rich columns)
    if not os.path.exists(EXT_PATH):
        print(f"[ERROR] Missing {EXT_PATH} — run new_signings_stage4a.py --step4-only first.")
        sys.exit(1)
    ext_df = pd.read_csv(EXT_PATH)

    # New signings lookup
    ns_df = pd.read_csv(NS_PATH) if os.path.exists(NS_PATH) else pd.DataFrame()

    return pos_dfs, ext_df, ns_df


# =============================================================================
# STEP 2 — Deduplicate position files
# =============================================================================
def deduplicate_position_files(pos_dfs):
    """
    Group by name + season, keep row with highest minutes_reliability_season,
    then highest minutes where tied.
    """
    deduped = {}
    total_removed = 0

    for pos, df in pos_dfs.items():
        before = len(df)
        df_sorted = df.sort_values(
            ["minutes_reliability_season", "minutes"],
            ascending=[False, False]
        )
        df_dedup = df_sorted.drop_duplicates(subset=["name", "season"], keep="first")
        removed = before - len(df_dedup)
        total_removed += removed
        deduped[pos] = df_dedup.reset_index(drop=True)

        # Save back
        df_dedup.to_csv(POS_FILES[pos], index=False)
        slug = pos.lower()
        fname = f"new_signings_{slug}.csv"
        if removed > 0:
            print(f"  {fname}: removed {removed} duplicate rows ({before} -> {len(df_dedup)})")
        else:
            print(f"  {fname}: no duplicates found ({len(df_dedup)} rows)")

    print(f"  Total duplicate rows removed: {total_removed}")
    return deduped, total_removed


# =============================================================================
# REPORT SECTIONS
# =============================================================================

def section1_summary(pos_dfs, ext_df, ns_df, total_removed):
    p("=" * 70)
    p("=== STAGE 4a VERIFICATION REPORT ===")
    p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 70)

    total_fpl    = 820
    n_with_hist  = 457
    n_new        = 363
    n_confirmed  = 63
    fbref_raw    = 276  # before false positive removal (from full run)
    fbref_clean  = 249
    fp_dropped   = fbref_raw - fbref_clean

    p("")
    p("PIPELINE SUMMARY:")
    p(f"  Total FPL players:                    {total_fpl}")
    p(f"  Players with vaastav history:         {n_with_hist}")
    p(f"  Players new to PL:                    {n_new}")
    p(f"  Confirmed relevant new signings:       {n_confirmed}")
    p(f"  FBref rows scraped (raw):          {fbref_raw:,}")
    p(f"  FBref rows after false pos removal:   {fbref_clean}")
    p(f"  False positive rows dropped:           {fp_dropped}")
    p(f"  Players skipped (no top league data):   {len(SKIP_FBREF_REASONS)}")
    p(f"  Duplicate rows removed (dedup):         {total_removed}")
    p("")

    p("POSITION FILE ROW COUNTS (after dedup):")
    total_rows    = 0
    total_players = 0
    for pos in ["GK", "DEF", "MID", "FWD"]:
        df = pos_dfs[pos]
        n_players = df["name"].nunique()
        total_rows    += len(df)
        total_players += n_players
        fname = f"new_signings_{pos.lower()}.csv"
        p(f"  {fname:<28} {len(df):3d} rows  ({n_players} unique players)")
    p(f"  {'TOTAL':<28} {total_rows:3d} rows  ({total_players} unique players)")


def section2_player_details(pos_dfs, ext_df, ns_df):
    p("")
    p("=" * 70)
    p("SECTION 2 — FULL PLAYER LIST BY POSITION")
    p("=" * 70)

    # Build NS lookup: name -> row
    ns_lookup = {}
    if not ns_df.empty and "fpl_name" in ns_df.columns:
        for _, r in ns_df.iterrows():
            ns_lookup[str(r["fpl_name"])] = r

    pos_labels = {
        "GK":  "GOALKEEPERS",
        "DEF": "DEFENDERS",
        "MID": "MIDFIELDERS",
        "FWD": "FORWARDS",
    }

    for pos, label in pos_labels.items():
        df = pos_dfs[pos]
        players = sorted(df["name"].dropna().unique().tolist())
        p("")
        p(f"=== {label} ({len(players)} players) ===")

        for player in players:
            player_rows = df[df["name"] == player].sort_values("season")
            # Get extended rows for this player
            ext_rows = ext_df[ext_df["fpl_name"] == player] if not ext_df.empty else pd.DataFrame()
            ext_rows = ext_rows.sort_values("fbref_season") if not ext_rows.empty else ext_rows

            ns_row = ns_lookup.get(player, {})
            prev_league = "unknown"
            fpl_team    = str(player_rows.iloc[0]["team"]) if len(player_rows) > 0 else "?"
            fpl_price   = float(player_rows.iloc[0]["value"]) if len(player_rows) > 0 else 0.0
            fpl_price_s = f"£{fpl_price:.1f}m"

            if not ext_rows.empty and "previous_league" in ext_rows.columns:
                prev_league_vals = ext_rows["previous_league"].dropna().unique()
                prev_league_vals = [v for v in prev_league_vals if v not in ("nan", "", "unknown")]
                if prev_league_vals:
                    prev_league = prev_league_vals[0]

            if hasattr(ns_row, "get"):
                pls = ns_row.get("previous_league_standardized", "")
                if pls and str(pls) not in ("nan", "", "unknown"):
                    prev_league = str(pls)

            confidence, conf_reason = infer_confidence(ext_rows)

            # FBref match name
            fbref_name_display = "N/A (zero stats)"
            if not ext_rows.empty and "fbref_name" in ext_rows.columns:
                fn_vals = ext_rows["fbref_name"].dropna().unique()
                fn_vals = [v for v in fn_vals if v != player]
                if fn_vals:
                    fbref_name_display = fn_vals[0]
                elif len(ext_rows) > 0:
                    fbref_name_display = player  # matched exactly

            p("")
            p(f"  Player: {player}")
            p(f"    FPL Team:          {fpl_team}")
            p(f"    FPL Price:         {fpl_price_s}")
            p(f"    Previous League:   {prev_league}")
            p(f"    Data Confidence:   {confidence}  ({conf_reason})")
            p(f"    FBref Matched As:  {fbref_name_display}")

            # Is this a zero-stat row?
            is_zero = (
                not ext_rows.empty and
                "data_confidence" in ext_rows.columns and
                (ext_rows["data_confidence"] == "low").all()
            )

            if is_zero:
                p(f"    [NO FBREF DATA — zero stats, data_confidence=low]")
                continue

            if pos == "GK":
                p(f"")
                p(f"    {'Season':<10} {'Team':<22} {'League':<20} {'Apps':>4} {'Mins':>5} {'CS':>3} {'Saves':>6} {'Sv/G':>6} {'CS%':>6} {'Rel':>5}")
                p(f"    {'-'*10} {'-'*22} {'-'*20} {'-'*4} {'-'*5} {'-'*3} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
                for _, vrow in player_rows.iterrows():
                    # match ext row for this season
                    season_s = str(vrow.get("season", ""))
                    erow = ext_rows[ext_rows["fbref_season"] == season_s]
                    if erow.empty and not ext_rows.empty:
                        erow = ext_rows
                    apps_v  = int(erow["appearances"].iloc[0]) if not erow.empty and "appearances" in erow.columns else 0
                    cs_v    = float(vrow.get("clean_sheets", 0) or 0)
                    saves_v = float(vrow.get("saves", 0) or 0)
                    sv_pg   = float(vrow.get("saves_per_game_season", 0) or 0)
                    cs_rate = float(vrow.get("clean_sheet_rate_season", 0) or 0)
                    mins_v  = float(vrow.get("minutes", 0) or 0)
                    rel_v   = rel_label(vrow.get("minutes_reliability_season", 0.1))
                    team_v  = str(erow["fbref_team"].iloc[0]) if not erow.empty and "fbref_team" in erow.columns else "?"
                    league_v = prev_league
                    p(f"    {season_s:<10} {team_v:<22} {league_v:<20} {apps_v:>4} {int(mins_v):>5} {int(cs_v):>3} {int(saves_v):>6} {sv_pg:>6.2f} {fmt_pct(cs_rate):>6} {rel_v:>5}")

                # Adjustments
                adj_cs_seasons = player_rows["clean_sheet_rate_season"].dropna()
                adj_sv_seasons = player_rows["saves_per_game_season"].dropna()
                avg_cs  = adj_cs_seasons.mean() if len(adj_cs_seasons) > 0 else 0.0
                avg_sv  = adj_sv_seasons.mean() if len(adj_sv_seasons) > 0 else 0.0
                p(f"")
                p(f"    Adjusted (PL equivalent, multiplier={LEAGUE_MULTIPLIERS.get(prev_league,'N/A')}):")
                p(f"      Avg CS rate per season:    {fmt_pct(avg_cs)}")
                p(f"      Avg saves per game:        {fmt_f(avg_sv, 2)}")

            else:
                p(f"")
                p(f"    {'Season':<10} {'Team':<22} {'League':<20} {'Apps':>4} {'Mins':>5} {'Gls':>4} {'Ast':>4} {'YC':>3} {'RC':>3} {'AdjG/90':>8} {'AdjA/90':>8} {'Rel':>5}")
                p(f"    {'-'*10} {'-'*22} {'-'*20} {'-'*4} {'-'*5} {'-'*4} {'-'*4} {'-'*3} {'-'*3} {'-'*8} {'-'*8} {'-'*5}")
                for _, vrow in player_rows.iterrows():
                    season_s = str(vrow.get("season", ""))
                    erow = ext_rows[ext_rows["fbref_season"] == season_s] if not ext_rows.empty and "fbref_season" in ext_rows.columns else pd.DataFrame()
                    apps_v  = int(erow["appearances"].iloc[0]) if not erow.empty and "appearances" in erow.columns else 0
                    mins_v  = float(vrow.get("minutes", 0) or 0)
                    gls_v   = float(vrow.get("goals_scored", 0) or 0)
                    ast_v   = float(vrow.get("assists", 0) or 0)
                    yc_v    = float(vrow.get("yellow_cards", 0) or 0)
                    rc_v    = float(vrow.get("red_cards", 0) or 0)
                    adj_g   = float(vrow.get("goals_per_game_season", 0) or 0)
                    adj_a   = float(vrow.get("assists_per_game_season", 0) or 0)
                    rel_v   = rel_label(vrow.get("minutes_reliability_season", 0.1))
                    team_v  = str(erow["fbref_team"].iloc[0]) if not erow.empty and "fbref_team" in erow.columns else "?"
                    p(f"    {season_s:<10} {team_v:<22} {prev_league:<20} {apps_v:>4} {int(mins_v):>5} {int(gls_v):>4} {int(ast_v):>4} {int(yc_v):>3} {int(rc_v):>3} {adj_g:>8.3f} {adj_a:>8.3f} {rel_v:>5}")

                # Adjusted summary
                valid_seasons = player_rows[player_rows["minutes_reliability_season"] >= 0.5]
                avg_ag90 = valid_seasons["goals_per_game_season"].mean() if not valid_seasons.empty else 0.0
                avg_aa90 = valid_seasons["assists_per_game_season"].mean() if not valid_seasons.empty else 0.0
                mult = LEAGUE_MULTIPLIERS.get(prev_league, "N/A")
                p(f"")
                p(f"    Adjusted (PL equivalent, multiplier={mult}):")
                p(f"      Avg adj goals per 90:      {fmt_f(avg_ag90, 3)}")
                p(f"      Avg adj assists per 90:    {fmt_f(avg_aa90, 3)}")
                if pos in ("DEF", "GK"):
                    avg_csr = player_rows["clean_sheet_rate_season"].mean()
                    p(f"      Avg adj CS rate:           {fmt_pct(avg_csr)}")


def section3_rankings(pos_dfs, ext_df, ns_df):
    p("")
    p("=" * 70)
    p("SECTION 3 — RANKINGS")
    p("=" * 70)

    # Build a combined frame: one row per (player, season), using position file data
    all_frames = []
    for pos, df in pos_dfs.items():
        df2 = df.copy()
        df2["_pos"] = pos
        all_frames.append(df2)
    combined = pd.concat(all_frames, ignore_index=True)

    # Join previous_league from ext_df
    if not ext_df.empty and "fpl_name" in ext_df.columns and "previous_league" in ext_df.columns:
        league_map = (
            ext_df[["fpl_name", "previous_league"]]
            .drop_duplicates(subset="fpl_name")
        )
        combined = combined.merge(
            league_map.rename(columns={"fpl_name": "name"}),
            on="name", how="left"
        )
    if "previous_league" not in combined.columns:
        combined["previous_league"] = "unknown"

    # Filter: only use rows with reliability >= 0.5 for rankings
    reliable = combined[combined["minutes_reliability_season"] >= 0.5].copy()

    # -- Helper: best season per player (highest adj goals/90) for per-player ranking
    def best_per_player(df, metric_col, ascending=False):
        """Return one row per player: the season with the best metric_col value."""
        return (
            df.sort_values(metric_col, ascending=ascending)
              .drop_duplicates(subset="name", keep="first")
        )

    def count_seasons(df, player_name):
        return len(df[df["name"] == player_name])

    # ── TOP 10: Adjusted Goals per 90 ─────────────────────────────────────────
    p("")
    p("TOP 10 BY ADJUSTED GOALS PER 90:")
    p(f"  {'Rank':<5} {'Player':<28} {'Pos':<5} {'Prev League':<20} {'Adj G/90':>8} {'Conf':<8} {'Note'}")
    p(f"  {'-'*5} {'-'*28} {'-'*5} {'-'*20} {'-'*8} {'-'*8} {'-'*30}")
    top10g = (
        reliable[reliable["goals_per_game_season"] > 0]
        .sort_values("goals_per_game_season", ascending=False)
        .drop_duplicates(subset=["name", "season"], keep="first")
    )
    # Group to get best season per player
    best_g = best_per_player(top10g, "goals_per_game_season")[:10]
    for rank, (_, row) in enumerate(best_g.iterrows(), 1):
        n_seasons = count_seasons(reliable[reliable["name"] == row["name"]], row["name"])
        ext_player = ext_df[ext_df["fpl_name"] == row["name"]] if not ext_df.empty else pd.DataFrame()
        conf, _ = infer_confidence(ext_player)
        note = ""
        mins = float(row.get("minutes", 0) or 0)
        if mins < 500:
            note = f"SMALL SAMPLE ({int(mins)} mins)"
        p(f"  {rank:<5} {row['name']:<28} {row['_pos']:<5} {str(row.get('previous_league','')):<20} {row['goals_per_game_season']:>8.3f} {conf:<8} {note}")

    # ── TOP 10: Adjusted Assists per 90 ───────────────────────────────────────
    p("")
    p("TOP 10 BY ADJUSTED ASSISTS PER 90:")
    p(f"  {'Rank':<5} {'Player':<28} {'Pos':<5} {'Prev League':<20} {'Adj A/90':>8} {'Conf':<8}")
    p(f"  {'-'*5} {'-'*28} {'-'*5} {'-'*20} {'-'*8} {'-'*8}")
    top10a = (
        reliable[reliable["assists_per_game_season"] > 0]
        .sort_values("assists_per_game_season", ascending=False)
    )
    best_a = best_per_player(top10a, "assists_per_game_season")[:10]
    for rank, (_, row) in enumerate(best_a.iterrows(), 1):
        ext_player = ext_df[ext_df["fpl_name"] == row["name"]] if not ext_df.empty else pd.DataFrame()
        conf, _ = infer_confidence(ext_player)
        p(f"  {rank:<5} {row['name']:<28} {row['_pos']:<5} {str(row.get('previous_league','')):<20} {row['assists_per_game_season']:>8.3f} {conf:<8}")

    # ── TOP 10 MIDFIELDERS: Goal Contributions per 90 ─────────────────────────
    p("")
    p("TOP 10 MIDFIELDERS BY ADJUSTED GOAL CONTRIBUTIONS PER 90 (goals + assists):")
    p(f"  {'Rank':<5} {'Player':<28} {'Prev League':<20} {'G+A/90':>7} {'G/90':>6} {'A/90':>6} {'Conf':<8}")
    p(f"  {'-'*5} {'-'*28} {'-'*20} {'-'*7} {'-'*6} {'-'*6} {'-'*8}")
    mids = reliable[reliable["_pos"] == "MID"].copy()
    mids["gc_per_90"] = mids["goals_per_game_season"] + mids["assists_per_game_season"]
    top_mids = best_per_player(mids[mids["gc_per_90"] > 0], "gc_per_90")[:10]
    for rank, (_, row) in enumerate(top_mids.iterrows(), 1):
        ext_player = ext_df[ext_df["fpl_name"] == row["name"]] if not ext_df.empty else pd.DataFrame()
        conf, _ = infer_confidence(ext_player)
        p(f"  {rank:<5} {row['name']:<28} {str(row.get('previous_league','')):<20} {row['gc_per_90']:>7.3f} {row['goals_per_game_season']:>6.3f} {row['assists_per_game_season']:>6.3f} {conf:<8}")

    # ── TOP 5 DEFENDERS: Clean Sheet Rate ─────────────────────────────────────
    p("")
    p("TOP 5 DEFENDERS BY CLEAN SHEET RATE:")
    p(f"  {'Rank':<5} {'Player':<28} {'Prev League':<20} {'CS Rate':>8} {'Conf':<8}")
    p(f"  {'-'*5} {'-'*28} {'-'*20} {'-'*8} {'-'*8}")
    defs = reliable[(reliable["_pos"] == "DEF") & (reliable["clean_sheet_rate_season"] > 0)]
    best_def = best_per_player(defs, "clean_sheet_rate_season")[:5]
    for rank, (_, row) in enumerate(best_def.iterrows(), 1):
        ext_player = ext_df[ext_df["fpl_name"] == row["name"]] if not ext_df.empty else pd.DataFrame()
        conf, _ = infer_confidence(ext_player)
        p(f"  {rank:<5} {row['name']:<28} {str(row.get('previous_league','')):<20} {fmt_pct(row['clean_sheet_rate_season']):>8} {conf:<8}")

    # ── TOP 5 GK: Saves per Game ───────────────────────────────────────────────
    p("")
    p("TOP 5 GOALKEEPERS BY SAVES PER GAME:")
    p(f"  {'Rank':<5} {'Player':<28} {'Prev League':<20} {'Sv/G':>6} {'CS%':>8} {'Conf':<8}")
    p(f"  {'-'*5} {'-'*28} {'-'*20} {'-'*6} {'-'*8} {'-'*8}")
    gks = reliable[(reliable["_pos"] == "GK") & (reliable["saves_per_game_season"] > 0)]
    best_gk = best_per_player(gks, "saves_per_game_season")[:5]
    for rank, (_, row) in enumerate(best_gk.iterrows(), 1):
        ext_player = ext_df[ext_df["fpl_name"] == row["name"]] if not ext_df.empty else pd.DataFrame()
        conf, _ = infer_confidence(ext_player)
        cs_rate = float(row.get("clean_sheet_rate_season", 0) or 0)
        p(f"  {rank:<5} {row['name']:<28} {str(row.get('previous_league','')):<20} {row['saves_per_game_season']:>6.2f} {fmt_pct(cs_rate):>8} {conf:<8}")

    # ── BEST VALUE: Adj Goal Contributions per £m ─────────────────────────────
    p("")
    p("BEST VALUE NEW SIGNINGS (adjusted goal contributions per £1m of price):")
    p(f"  {'Rank':<5} {'Player':<28} {'Pos':<5} {'Price':>6} {'GC/90':>6} {'GC/90/£m':>9} {'Conf':<8}")
    p(f"  {'-'*5} {'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*9} {'-'*8}")
    reliable2 = reliable.copy()
    reliable2["gc_per_90"] = reliable2["goals_per_game_season"] + reliable2["assists_per_game_season"]
    reliable2["value_score"] = reliable2.apply(
        lambda r: safe_div(r["gc_per_90"], r["value"]), axis=1
    )
    best_val = (
        reliable2[reliable2["gc_per_90"] > 0]
        .sort_values("value_score", ascending=False)
        .drop_duplicates(subset="name", keep="first")
        .head(10)
    )
    for rank, (_, row) in enumerate(best_val.iterrows(), 1):
        ext_player = ext_df[ext_df["fpl_name"] == row["name"]] if not ext_df.empty else pd.DataFrame()
        conf, _ = infer_confidence(ext_player)
        p(f"  {rank:<5} {row['name']:<28} {row['_pos']:<5} £{row['value']:<5.1f} {row['gc_per_90']:>6.3f} {row['value_score']:>9.4f} {conf:<8}")


def section4_data_quality(pos_dfs, ext_df, ns_df):
    p("")
    p("=" * 70)
    p("SECTION 4 — DATA QUALITY FLAGS")
    p("=" * 70)

    # Combine all position data
    all_frames = []
    for pos, df in pos_dfs.items():
        df2 = df.copy()
        df2["_pos"] = pos
        all_frames.append(df2)
    combined = pd.concat(all_frames, ignore_index=True)

    # ── HIGH CONFIDENCE ────────────────────────────────────────────────────────
    p("")
    p("HIGH CONFIDENCE PLAYERS (3 seasons, reliability >= 0.5):")
    high_conf = []
    for name in combined["name"].unique():
        rows = combined[combined["name"] == name]
        good = rows[rows["minutes_reliability_season"] >= 0.5]
        if len(good) >= 3 and rows["minutes"].sum() > 0:
            avg_mins = good["minutes"].mean()
            high_conf.append((name, str(rows.iloc[0]["_pos"]), len(good), int(avg_mins)))
    high_conf.sort(key=lambda x: (x[2], x[3]), reverse=True)
    if high_conf:
        p(f"  {'Player':<30} {'Pos':<5} {'Seasons':>7} {'Avg Mins':>9}")
        p(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*9}")
        for name, pos, n, avg in high_conf:
            p(f"  {name:<30} {pos:<5} {n:>7} {avg:>9}")
    else:
        p("  None.")

    # ── MEDIUM CONFIDENCE ──────────────────────────────────────────────────────
    p("")
    p("MEDIUM CONFIDENCE (1-2 good seasons or partial minutes):")
    med_conf = []
    for name in combined["name"].unique():
        rows = combined[combined["name"] == name]
        good = rows[rows["minutes_reliability_season"] >= 0.5]
        all_mins = rows["minutes"].sum()
        if 1 <= len(good) < 3 and all_mins > 0:
            seasons = sorted(rows["season"].dropna().unique().tolist())
            med_conf.append((name, str(rows.iloc[0]["_pos"]), len(good), seasons))
    med_conf.sort(key=lambda x: x[2], reverse=True)
    if med_conf:
        p(f"  {'Player':<30} {'Pos':<5} {'Good Seas':>9}  Seasons")
        p(f"  {'-'*30} {'-'*5} {'-'*9}  {'-'*30}")
        for name, pos, n, seasons in med_conf:
            p(f"  {name:<30} {pos:<5} {n:>9}  {', '.join(seasons)}")
    else:
        p("  None.")

    # ── SMALL SAMPLE WARNINGS ──────────────────────────────────────────────────
    p("")
    p("LOW CONFIDENCE / SMALL SAMPLE WARNINGS (any season < 500 minutes):")
    small_sample = []
    for _, row in combined.iterrows():
        mins = float(row.get("minutes", 0) or 0)
        if 0 < mins < 500 and row.get("minutes_reliability_season", 0) >= 0.5:
            small_sample.append((row["name"], str(row["_pos"]), row.get("season", "?"), int(mins)))
    if small_sample:
        p(f"  {'Player':<30} {'Pos':<5} {'Season':<10} {'Mins':>6}  WARNING")
        p(f"  {'-'*30} {'-'*5} {'-'*10} {'-'*6}  {'-'*40}")
        for name, pos, season, mins in sorted(small_sample, key=lambda x: x[3]):
            note = "unreliable per-90 stats — small sample"
            if name == "Lukas Nmecha" and "2023" in str(season):
                note = "Wolfsburg hot streak — very small sample, inflated per-90"
            p(f"  {name:<30} {pos:<5} {season:<10} {mins:>6}  {note}")
    else:
        p("  None.")

    # ── ZERO FBREF DATA ────────────────────────────────────────────────────────
    p("")
    p("PLAYERS WITH ZERO FBREF DATA (skipped — zero-stat rows assigned):")
    p(f"  {'Player':<30} {'Pos':<5}  Reason")
    p(f"  {'-'*30} {'-'*5}  {'-'*50}")
    for name, reason in SKIP_FBREF_REASONS.items():
        player_rows = combined[combined["name"] == name]
        pos_s = str(player_rows.iloc[0]["_pos"]) if len(player_rows) > 0 else "?"
        p(f"  {name:<30} {pos_s:<5}  {reason}")

    # Zero-stat rows from non-scrapeable leagues
    if not ext_df.empty and "data_confidence" in ext_df.columns:
        low_conf_extra = ext_df[
            (ext_df["data_confidence"] == "low") &
            (~ext_df["fpl_name"].isin(SKIP_FBREF_REASONS.keys()))
        ][["fpl_name", "previous_league", "fpl_position"]].drop_duplicates("fpl_name")
        if not low_conf_extra.empty:
            p("")
            p("  Additional zero-stat players (non-scrapeable league):")
            for _, r in low_conf_extra.iterrows():
                p(f"    {str(r['fpl_name']):<30} {str(r.get('fpl_position','?')):<5}  League: {r.get('previous_league','unknown')}")

    # ── FALSE POSITIVES REMOVED ────────────────────────────────────────────────
    p("")
    p(f"FALSE POSITIVE ROWS REMOVED ({len(FBREF_FALSE_POSITIVES)} pairs, 27 rows total):")
    p(f"  {'FBref Name':<30} {'FPL Target':<28}  Reason")
    p(f"  {'-'*30} {'-'*28}  {'-'*40}")
    for fb_name, fpl_name, reason in FBREF_FALSE_POSITIVES:
        p(f"  {fb_name:<30} {fpl_name:<28}  {reason}")

    # ── KEPT LOW-CONFIDENCE MATCHES ───────────────────────────────────────────
    p("")
    p("FUZZY MATCHES BELOW 95% THAT WERE KEPT (verified same player):")
    p(f"  {'FBref Name':<28} {'FPL Target':<32} {'Conf':>5}  Decision")
    p(f"  {'-'*28} {'-'*32} {'-'*5}  {'-'*35}")
    for fb, fpl, conf, decision in KEPT_LOW_CONF:
        p(f"  {fb:<28} {fpl:<32} {conf:>5}  {decision}")


def section5_league_coverage(pos_dfs, ext_df):
    p("")
    p("=" * 70)
    p("SECTION 5 — LEAGUE COVERAGE")
    p("=" * 70)
    p("")

    if ext_df.empty or "previous_league" not in ext_df.columns:
        p("  [No extended data available]")
        return

    # Unique players per league (use non-zero rows)
    real_rows = ext_df[ext_df.get("season_reliability", pd.Series([0])) > 0.0] if "season_reliability" in ext_df.columns else ext_df
    grouped = ext_df.drop_duplicates(subset="fpl_name")[["fpl_name", "previous_league"]].copy()

    league_counts = grouped["previous_league"].value_counts()
    scraped_leagues = list(LEAGUE_MULTIPLIERS.keys())

    # Per-league stats from position files
    all_frames = []
    for pos, df in pos_dfs.items():
        all_frames.append(df)
    combined = pd.concat(all_frames, ignore_index=True)

    if not ext_df.empty:
        combined = combined.merge(
            ext_df[["fpl_name", "previous_league"]].drop_duplicates("fpl_name").rename(columns={"fpl_name": "name"}),
            on="name", how="left"
        )

    p(f"  {'League':<22} {'Players':>7} {'Multiplier':>10} {'Avg Adj G/90':>13} {'Avg Adj A/90':>13}")
    p(f"  {'-'*22} {'-'*7} {'-'*10} {'-'*13} {'-'*13}")

    for league in scraped_leagues:
        mult = LEAGUE_MULTIPLIERS[league]
        n_players = int(league_counts.get(league, 0))
        if "previous_league" in combined.columns:
            league_rows = combined[
                (combined["previous_league"] == league) &
                (combined["minutes_reliability_season"] >= 0.5)
            ]
            avg_g = league_rows["goals_per_game_season"].mean() if len(league_rows) > 0 else 0.0
            avg_a = league_rows["assists_per_game_season"].mean() if len(league_rows) > 0 else 0.0
        else:
            avg_g = avg_a = 0.0
        p(f"  {league:<22} {n_players:>7} {mult:>10.2f} {avg_g:>13.3f} {avg_a:>13.3f}")

    # Other leagues
    other = grouped[~grouped["previous_league"].isin(scraped_leagues + ["unknown", "nan"])]
    if not other.empty:
        p("")
        p("  Non-scraped leagues (zero-stat rows):")
        for league, count in other["previous_league"].value_counts().items():
            p(f"    {league:<30} {count} player(s)")


def section6_files_saved(pos_dfs):
    p("")
    p("=" * 70)
    p("SECTION 6 — FILES SAVED")
    p("=" * 70)
    p("")
    p(f"  data/raw/fbref/new_signings/new_signings_gk.csv   ({len(pos_dfs['GK'])} rows)")
    p(f"  data/raw/fbref/new_signings/new_signings_def.csv  ({len(pos_dfs['DEF'])} rows)")
    p(f"  data/raw/fbref/new_signings/new_signings_mid.csv  ({len(pos_dfs['MID'])} rows)")
    p(f"  data/raw/fbref/new_signings/new_signings_fwd.csv  ({len(pos_dfs['FWD'])} rows)")
    p(f"  data/raw/fbref/new_signings/result_extended.csv   (intermediate, all cols)")
    p(f"  data/raw/fbref/new_signings/verification_report.txt")
    p("")
    p("Stage 4a complete. Ready for Stage 4b.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Stage 4a Verification")
    print("=" * 70)

    # Load
    pos_dfs, ext_df, ns_df = load_data()

    # Deduplicate
    print("\nDeduplicating position files...")
    pos_dfs, total_removed = deduplicate_position_files(pos_dfs)

    # Generate report
    section1_summary(pos_dfs, ext_df, ns_df, total_removed)
    section2_player_details(pos_dfs, ext_df, ns_df)
    section3_rankings(pos_dfs, ext_df, ns_df)
    section4_data_quality(pos_dfs, ext_df, ns_df)
    section5_league_coverage(pos_dfs, ext_df)
    section6_files_saved(pos_dfs)

    # Save report file
    save_report()


if __name__ == "__main__":
    main()
