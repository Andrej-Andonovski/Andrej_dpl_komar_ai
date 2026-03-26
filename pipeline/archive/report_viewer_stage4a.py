"""
FPL AI Stage 4a - Full Report Printer
Prints everything: all players, all seasons, all rankings, all flags.
Usage: python pipeline/report_viewer_stage4a.py
"""

import io
import os
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    G = Fore.GREEN
    Y = Fore.YELLOW
    R = Fore.RED
    C = Fore.CYAN
    W = Fore.WHITE
    B = Style.BRIGHT
    RST = Style.RESET_ALL
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    G = Fore.GREEN; Y = Fore.YELLOW; R = Fore.RED; C = Fore.CYAN
    W = Fore.WHITE; B = Style.BRIGHT; RST = Style.RESET_ALL

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGN_DIR = os.path.join(ROOT, "data", "raw", "fbref", "new_signings")
TRANS_DIR= os.path.join(ROOT, "data", "raw", "transfers")

POS_FILES = {
    "GK":  os.path.join(SIGN_DIR, "new_signings_gk.csv"),
    "DEF": os.path.join(SIGN_DIR, "new_signings_def.csv"),
    "MID": os.path.join(SIGN_DIR, "new_signings_mid.csv"),
    "FWD": os.path.join(SIGN_DIR, "new_signings_fwd.csv"),
}
EXT_FILE  = os.path.join(SIGN_DIR, "result_extended.csv")
NS_FILE   = os.path.join(TRANS_DIR, "new_signings_2025.csv")

# ── helpers ──────────────────────────────────────────────────────────────────
def hdr(title, char="=", width=80):
    print(f"\n{B}{C}{char * width}{RST}")
    print(f"{B}{C}  {title}{RST}")
    print(f"{B}{C}{char * width}{RST}")

def sub(title, width=80):
    print(f"\n{B}{W}  -- {title} --{RST}")
    print(f"  {'-' * (width - 4)}")

def bar(r, filled="#", empty="."):
    try: r = float(r)
    except: r = 0.0
    n = min(5, max(0, round(r / 0.2)))
    return filled * n + empty * (5 - n)

def conf_color(c_str):
    if c_str == "high":   return G
    if c_str == "medium": return Y
    return R

def infer_conf(rows):
    if rows.empty: return "low"
    if "data_confidence" in rows.columns and (rows["data_confidence"] == "low").all():
        return "low"
    if "season_reliability" in rows.columns:
        good = rows[rows["season_reliability"].fillna(0) >= 0.5]
        if len(good) >= 3: return "high"
        if len(good) >= 1: return "medium"
    return "low"

def load():
    pos_dfs = {}
    for pos, path in POS_FILES.items():
        pos_dfs[pos] = pd.read_csv(path, low_memory=False) if os.path.exists(path) else pd.DataFrame()
    ext = pd.read_csv(EXT_FILE, low_memory=False) if os.path.exists(EXT_FILE) else pd.DataFrame()
    ns  = pd.read_csv(NS_FILE,  low_memory=False) if os.path.exists(NS_FILE)  else pd.DataFrame()
    return pos_dfs, ext, ns

# ── SECTION 1: Pipeline Summary ───────────────────────────────────────────────
def section1(pos_dfs, ext, ns):
    hdr("SECTION 1 — PIPELINE SUMMARY")
    all_pos = pd.concat([df for df in pos_dfs.values() if not df.empty], ignore_index=True)
    total_rows    = len(all_pos)
    total_players = all_pos["name"].nunique() if not all_pos.empty else 0

    print(f"\n  Total position rows : {B}{total_rows}{RST}")
    print(f"  Unique players      : {B}{total_players}{RST}")
    print(f"\n  {'Position':<10} {'Players':>10} {'Rows':>8}")
    print(f"  {'-'*30}")
    for pos, df in pos_dfs.items():
        players = df["name"].nunique() if not df.empty else 0
        print(f"  {pos:<10} {players:>10} {len(df):>8}")

    if not ext.empty and "data_confidence" in ext.columns:
        print(f"\n  {'Confidence':<10} {'Rows':>6}")
        print(f"  {'-'*20}")
        for label, col in [("high", G), ("medium", Y), ("low", R)]:
            cnt = (ext["data_confidence"].fillna("high") == label).sum()
            print(f"  {col}{label:<10}{RST} {cnt:>6}")

    if not ns.empty:
        print(f"\n  New-to-PL candidates (FPL API): {len(ns)}")
        print(f"  Matched via Transfermarkt      : {ns['previous_league_standardized'].notna().sum()}")


# ── SECTION 2: All Players by Position ───────────────────────────────────────
def section2(pos_dfs, ext):
    hdr("SECTION 2 — ALL PLAYERS BY POSITION")

    for pos, df in pos_dfs.items():
        if df.empty:
            continue
        sub(f"{pos} — {df['name'].nunique()} players, {len(df)} season rows")

        players = sorted(df["name"].dropna().unique())
        for name in players:
            p_ext  = ext[ext["fpl_name"] == name].copy() if not ext.empty else pd.DataFrame()
            conf   = infer_conf(p_ext)
            col    = conf_color(conf)
            p_rows = df[df["name"] == name]

            team  = p_rows["team"].iloc[0]  if not p_rows.empty else "?"
            value = p_rows["value"].iloc[0] if not p_rows.empty else "?"

            prev_league = "?"
            multiplier  = "?"
            if not p_ext.empty:
                prev_league = p_ext["previous_league"].iloc[0] if "previous_league" in p_ext.columns else "?"
                multiplier  = p_ext["multiplier"].iloc[0]      if "multiplier"       in p_ext.columns else "?"

            # confidence badge + reliability across seasons
            best_rel = p_ext["season_reliability"].fillna(0).max() if not p_ext.empty and "season_reliability" in p_ext.columns else 0
            b = bar(best_rel)
            n_seasons = p_ext["fbref_season"].nunique() if not p_ext.empty and "fbref_season" in p_ext.columns else 0

            print(f"\n  {col}{B}{name}{RST}  [{pos}]  {team}  £{value}m")
            print(f"  League: {prev_league}  Mult: {multiplier}  Conf: {col}{conf}{RST}  Seasons: {n_seasons}  Rel: {b}")

            # per-season rows from ext
            if not p_ext.empty and "fbref_season" in p_ext.columns:
                p_ext_show = p_ext[p_ext["fbref_season"].notna() & (p_ext["fbref_season"].astype(str) != "nan")]
                if not p_ext_show.empty:
                    print(f"  {'Season':<10} {'Team':<20} {'Apps':>5} {'Min':>6} {'Gls':>5} {'Ast':>5} {'xG':>6} {'xA':>6} {'adjG/90':>8} {'adjA/90':>8} {'YC':>3} {'RC':>3} {'CS':>4} {'Sv':>5} {'Rel':>5} {'Bar'}")
                    print(f"  {'-'*110}")
                    for _, row in p_ext_show.sort_values("fbref_season").iterrows():
                        season   = row.get("fbref_season", "?")
                        team_fb  = str(row.get("fbref_team", "?"))[:18]
                        apps     = row.get("appearances", 0)
                        # minutes from position file for this season
                        s_pos = p_rows[p_rows["season"] == season] if "season" in p_rows.columns else pd.DataFrame()
                        mins = int(s_pos["minutes"].iloc[0]) if not s_pos.empty else "?"
                        gls  = row.get("goals", 0)
                        ast  = row.get("assists", 0)
                        xg   = row.get("xG", 0)
                        xa   = row.get("xA", 0)
                        ag90 = row.get("adjusted_goals_per_90", 0)
                        aa90 = row.get("adjusted_assists_per_90", 0)
                        yc   = row.get("yellow_cards", 0)
                        rc   = row.get("red_cards", 0)
                        cs   = row.get("clean_sheets", 0)
                        sv   = row.get("saves", 0)
                        rel  = row.get("season_reliability", 0)
                        b2   = bar(rel)
                        low  = mins != "?" and int(mins) < 500

                        line = (f"  {season:<10} {team_fb:<20} {int(apps) if apps else '?':>5} "
                                f"{str(mins):>6} {float(gls):.1f} {float(ast):.1f} "
                                f"{float(xg):.2f} {float(xa):.2f} "
                                f"{float(ag90):.4f} {float(aa90):.4f} "
                                f"{int(yc):>3} {int(rc):>3} {float(cs):>4.0f} {float(sv):>5.0f} "
                                f"{float(rel):.1f} {b2}")
                        if low:
                            print(f"{R}{line}  << <500 min{RST}")
                        else:
                            print(line)
                else:
                    print(f"  {R}  No FBref season data (SKIP_FBREF / zero-stat player){RST}")
            else:
                print(f"  {R}  No FBref extended data{RST}")


# ── SECTION 3: Rankings ───────────────────────────────────────────────────────
def section3(pos_dfs, ext):
    hdr("SECTION 3 — RANKINGS")

    def best_per_player(col):
        if col not in ext.columns: return pd.DataFrame()
        idx = ext[ext[col].fillna(0) > 0].groupby("fpl_name")[col].idxmax()
        return ext.loc[idx].sort_values(col, ascending=False)

    # 3.1 Top 20 goals/90
    sub("Top 20: Adjusted Goals per 90 (best season per player)")
    df = best_per_player("adjusted_goals_per_90").head(20)
    print(f"  {'#':>3}  {'Player':<32} {'AdjG/90':>9} {'Season':<10} {'League':<22} {'Apps':>5} {'Min':>6}")
    print(f"  {'-'*90}")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        name = row.get("fpl_name", "?")
        p_rows = pos_dfs.get(row.get("fpl_position", ""), pd.DataFrame())
        mins = "?"
        if not p_rows.empty and "season" in p_rows.columns:
            s_rows = p_rows[(p_rows["name"] == name) & (p_rows["season"] == row.get("fbref_season",""))]
            if not s_rows.empty: mins = int(s_rows["minutes"].iloc[0])
        low = mins != "?" and int(mins) < 500
        line = (f"  {i:>3}  {name:<32} {float(row.get('adjusted_goals_per_90',0)):>9.4f} "
                f"{str(row.get('fbref_season','?')):<10} {str(row.get('previous_league','?')):<22} "
                f"{int(row.get('appearances',0)):>5} {str(mins):>6}")
        print(f"{R}{line}  <{RST}" if low else line)

    # 3.2 Top 20 assists/90
    sub("Top 20: Adjusted Assists per 90 (best season per player)")
    df = best_per_player("adjusted_assists_per_90").head(20)
    print(f"  {'#':>3}  {'Player':<32} {'AdjA/90':>9} {'Season':<10} {'League':<22}")
    print(f"  {'-'*80}")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"  {i:>3}  {str(row.get('fpl_name','?')):<32} {float(row.get('adjusted_assists_per_90',0)):>9.4f} "
              f"{str(row.get('fbref_season','?')):<10} {str(row.get('previous_league','?')):<22}")

    # 3.3 Top 20 MID G+A/90
    sub("Top 20 MID: Adjusted G+A per 90 (best season per player)")
    mid_names = set(pos_dfs.get("MID", pd.DataFrame()).get("name", pd.Series()).dropna().unique())
    mid_ext = ext[ext["fpl_name"].isin(mid_names)].copy()
    if not mid_ext.empty:
        mid_ext["ga90"] = mid_ext["adjusted_goals_per_90"].fillna(0) + mid_ext["adjusted_assists_per_90"].fillna(0)
        idx = mid_ext[mid_ext["ga90"] > 0].groupby("fpl_name")["ga90"].idxmax()
        df = mid_ext.loc[idx].sort_values("ga90", ascending=False).head(20)
        print(f"  {'#':>3}  {'Player':<32} {'G+A/90':>8} {'G/90':>8} {'A/90':>8} {'Season':<10} {'League':<22}")
        print(f"  {'-'*90}")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"  {i:>3}  {str(row.get('fpl_name','?')):<32} {float(row.get('ga90',0)):>8.4f} "
                  f"{float(row.get('adjusted_goals_per_90',0)):>8.4f} {float(row.get('adjusted_assists_per_90',0)):>8.4f} "
                  f"{str(row.get('fbref_season','?')):<10} {str(row.get('previous_league','?')):<22}")

    # 3.4 Top 10 GK saves/game
    sub("Top 10 GK: Saves per Game")
    gk = pos_dfs.get("GK", pd.DataFrame())
    if not gk.empty and "saves_per_game_season" in gk.columns:
        idx = gk[gk["saves_per_game_season"] > 0].groupby("name")["saves_per_game_season"].idxmax()
        df = gk.loc[idx].sort_values("saves_per_game_season", ascending=False).head(10)
        print(f"  {'#':>3}  {'Player':<32} {'Sv/G':>8} {'CS':>6} {'Season':<10} {'Team':<20}")
        print(f"  {'-'*80}")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"  {i:>3}  {str(row.get('name','?')):<32} {float(row.get('saves_per_game_season',0)):>8.3f} "
                  f"{float(row.get('clean_sheets',0)):>6.0f} {str(row.get('season','?')):<10} {str(row.get('team','?')):<20}")

    # 3.5 Top 10 DEF CS rate (from ext — use CS/appearances)
    sub("Top 10 DEF: Clean Sheets per Game (from FBref)")
    def_names = set(pos_dfs.get("DEF", pd.DataFrame()).get("name", pd.Series()).dropna().unique())
    def_ext = ext[ext["fpl_name"].isin(def_names)].copy()
    if not def_ext.empty and "clean_sheets" in def_ext.columns and "appearances" in def_ext.columns:
        def_ext = def_ext[def_ext["appearances"].fillna(0) > 0].copy()
        def_ext["cs_rate"] = def_ext["clean_sheets"].fillna(0) / def_ext["appearances"]
        idx = def_ext[def_ext["cs_rate"] > 0].groupby("fpl_name")["cs_rate"].idxmax()
        df = def_ext.loc[idx].sort_values("cs_rate", ascending=False).head(10)
        print(f"  {'#':>3}  {'Player':<32} {'CS/G':>8} {'CS':>5} {'Apps':>5} {'Season':<10} {'League':<22}")
        print(f"  {'-'*85}")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"  {i:>3}  {str(row.get('fpl_name','?')):<32} {float(row.get('cs_rate',0)):>8.3f} "
                  f"{float(row.get('clean_sheets',0)):>5.0f} {int(row.get('appearances',0)):>5} "
                  f"{str(row.get('fbref_season','?')):<10} {str(row.get('previous_league','?')):<22}")
    else:
        print("  No DEF clean sheet data available.")

    # 3.6 Best value (adj_g90 / price)
    sub("Top 20: Best Value (AdjG/90 per price £m)")
    if "fpl_price" in ext.columns:
        val_ext = ext[ext["fpl_price"].fillna(0) > 0].copy()
        val_ext["value_ratio"] = val_ext["adjusted_goals_per_90"].fillna(0) / val_ext["fpl_price"]
        idx = val_ext[val_ext["value_ratio"] > 0].groupby("fpl_name")["value_ratio"].idxmax()
        df = val_ext.loc[idx].sort_values("value_ratio", ascending=False).head(20)
        print(f"  {'#':>3}  {'Player':<32} {'G90/£':>9} {'AdjG/90':>9} {'Price':>7} {'Pos':<5} {'Season':<10}")
        print(f"  {'-'*85}")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"  {i:>3}  {str(row.get('fpl_name','?')):<32} {float(row.get('value_ratio',0)):>9.5f} "
                  f"{float(row.get('adjusted_goals_per_90',0)):>9.4f} {float(row.get('fpl_price',0)):>7.1f} "
                  f"{str(row.get('fpl_position','?')):<5} {str(row.get('fbref_season','?')):<10}")


# ── SECTION 4: Data Quality ───────────────────────────────────────────────────
def section4(pos_dfs, ext):
    hdr("SECTION 4 — DATA QUALITY FLAGS")

    # Confidence breakdown
    sub("Confidence by player")
    all_names = ext["fpl_name"].dropna().unique() if not ext.empty else []
    high, medium, low = [], [], []
    for name in all_names:
        conf = infer_conf(ext[ext["fpl_name"] == name])
        if conf == "high":   high.append(name)
        elif conf == "medium": medium.append(name)
        else: low.append(name)

    print(f"  {G}HIGH   ({len(high):>3}){RST}: {', '.join(sorted(high))}")
    print(f"\n  {Y}MEDIUM ({len(medium):>3}){RST}: {', '.join(sorted(medium))}")
    print(f"\n  {R}LOW    ({len(low):>3}){RST}: {', '.join(sorted(low))}")

    # Small samples
    sub("Small samples: < 500 minutes in any season row")
    all_pos = pd.concat([df for df in pos_dfs.values() if not df.empty], ignore_index=True)
    if not all_pos.empty and "minutes" in all_pos.columns:
        small = all_pos[all_pos["minutes"].fillna(0) < 500][["name","season","minutes","position"]].sort_values("minutes")
        if small.empty:
            print("  None.")
        else:
            print(f"  {'Player':<32} {'Season':<10} {'Min':>6} {'Pos':<5}")
            print(f"  {'-'*55}")
            for _, row in small.iterrows():
                print(f"  {R}{str(row['name']):<32} {str(row['season']):<10} {int(row['minutes']):>6} {str(row['position']):<5}{RST}")

    # Zero-data players
    sub("Zero / low-data players (SKIP_FBREF or no top-5 match)")
    if not ext.empty and "data_confidence" in ext.columns:
        zero = sorted(ext[ext["data_confidence"] == "low"]["fpl_name"].dropna().unique())
        if zero:
            for n in zero: print(f"  {R}{n}{RST}")
        else:
            print("  None.")


# ── SECTION 5: League Coverage ────────────────────────────────────────────────
def section5(ext, ns):
    hdr("SECTION 5 — LEAGUE COVERAGE")

    if not ext.empty and "previous_league" in ext.columns:
        sub("FBref extended data — rows and players per league")
        vc = ext["previous_league"].fillna("unknown").value_counts()
        print(f"  {'League':<28} {'Rows':>6} {'Players':>9}")
        print(f"  {'-'*46}")
        for league, cnt in vc.items():
            players_cnt = ext[ext["previous_league"] == league]["fpl_name"].nunique()
            print(f"  {league:<28} {cnt:>6} {players_cnt:>9}")

    if not ns.empty and "previous_league_standardized" in ns.columns:
        sub("Transfermarkt source — players per league")
        vc2 = ns["previous_league_standardized"].fillna("(not matched)").value_counts()
        print(f"  {'League':<28} {'Players':>9}")
        print(f"  {'-'*40}")
        for league, cnt in vc2.items():
            print(f"  {league:<28} {cnt:>9}")


# ── SECTION 6: Files ──────────────────────────────────────────────────────────
def section6():
    hdr("SECTION 6 — FILES SAVED")
    files = [
        (os.path.join(SIGN_DIR, "new_signings_gk.csv"),   "GK position file (vaastav format)"),
        (os.path.join(SIGN_DIR, "new_signings_def.csv"),  "DEF position file"),
        (os.path.join(SIGN_DIR, "new_signings_mid.csv"),  "MID position file"),
        (os.path.join(SIGN_DIR, "new_signings_fwd.csv"),  "FWD position file"),
        (EXT_FILE,                                         "Extended result (all FBref cols)"),
        (NS_FILE,                                          "New signings TM data"),
        (os.path.join(SIGN_DIR, "verification_report.txt"), "Full text report"),
    ]
    print(f"\n  {'Path':<60} {'Size':>10}  Status")
    print(f"  {'-'*80}")
    for path, desc in files:
        rel = os.path.relpath(path, ROOT)
        if os.path.exists(path):
            sz = os.path.getsize(path)
            size_str = f"{sz // 1024} KB" if sz >= 1024 else f"{sz} B"
            print(f"  {G}{rel:<60}{RST} {size_str:>10}  OK   ({desc})")
        else:
            print(f"  {R}{rel:<60}{'MISSING':>10}  !!{RST}  ({desc})")


# ── tee: write to file AND console ───────────────────────────────────────────
import re as _re

_ANSI_ESCAPE = _re.compile(r'\x1b\[[0-9;]*m')
_file_lines = []
_orig_print = print

def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    text = sep.join(str(a) for a in args) + end
    _file_lines.append(_ANSI_ESCAPE.sub("", text))

def save_report():
    out_path = os.path.join(SIGN_DIR, "stage4a_full_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(_file_lines)
    _orig_print(f"\n{G}{B}Report saved -> {out_path}{RST}\n")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading data...", end=" ", flush=True)
    pos_dfs, ext, ns = load()
    print("done.\n")

    section1(pos_dfs, ext, ns)
    section2(pos_dfs, ext)
    section3(pos_dfs, ext)
    section4(pos_dfs, ext)
    section5(ext, ns)
    section6()

    print(f"\nDone. Scroll up to read the full report.")
    save_report()

if __name__ == "__main__":
    main()
