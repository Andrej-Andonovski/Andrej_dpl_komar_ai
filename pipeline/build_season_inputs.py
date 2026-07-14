"""
pipeline/build_season_inputs.py
Cross-season harness, step 1 (fixing stage — docs/phase4_report.md backlog).

Converts TRUE per-fixture vaastav repo data (data/raw/vaastav_repo/data/<S>/)
into the exact input schema season_simulator.py expects:

  data/raw/seasons/<SEASON>/player_history.csv   (per-fixture rows: DGWs = 2)
  data/raw/seasons/<SEASON>/players_raw.csv      (real ids, real ownership)
  data/raw/seasons/<SEASON>/fixtures_raw.csv     (real FDR)

Notes:
  - merged_gw 'value' is in £0.1m units -> divided by 10 (£m, matching the
    2025-26 player_history schema)
  - players_raw price is set to each player's GW1 value from history (the
    repo's now_cost is an end-of-season snapshot); GW1 squad costs are
    therefore exact — slightly MORE correct than the 2025-26 run's own
    snapshot behaviour (documented)
  - real selected_by_percent is passed through (GW1 community prior)

Usage:
  docker run --rm -v "<repo>:/app" -w /app fpl-sim \
      python pipeline/build_season_inputs.py 2023-24 2024-25
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "..", "data")
REPO = os.path.join(DATA, "raw", "vaastav_repo", "data")
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def build_season(season):
    print(f"\n=== {season} ===")
    src = os.path.join(REPO, season)
    gw = pd.read_csv(os.path.join(src, "gws", "merged_gw.csv"),
                     encoding="utf-8-sig")
    fixtures = pd.read_csv(os.path.join(src, "fixtures.csv"))
    praw = pd.read_csv(os.path.join(src, "players_raw.csv"),
                       encoding="utf-8-sig")

    # 2024-25 introduced element_type 5 (assistant managers) — not squad
    # players; drop them everywhere
    praw = praw[praw["element_type"].isin([1, 2, 3, 4])].copy()
    real_ids = set(praw["id"].astype(int))
    gw = gw[gw["element"].astype(int).isin(real_ids)].copy()

    gw = gw.dropna(subset=["GW"]).copy()
    gw["GW"] = gw["GW"].astype(int)
    gw["was_home"] = gw["was_home"].astype(str).str.lower().isin(
        ("true", "1"))
    out_dir = os.path.join(DATA, "raw", "seasons", season)
    os.makedirs(out_dir, exist_ok=True)

    # ── player_history.csv (per-fixture; DGW players get 2 rows) ────────────
    ph = pd.DataFrame({
        "player_id":        gw["element"].astype(int),
        "gameweek":         gw["GW"],
        "total_points":     gw["total_points"],
        "minutes":          gw["minutes"],
        "goals_scored":     gw["goals_scored"],
        "assists":          gw["assists"],
        "clean_sheets":     gw["clean_sheets"],
        "saves":            gw["saves"],
        "bonus":            gw["bonus"],
        "value":            gw["value"] / 10.0,
        "was_home":         gw["was_home"].astype(int),
        "transfers_in":     gw["transfers_in"],
        "transfers_out":    gw["transfers_out"],
        "opponent_team_id": gw["opponent_team"].astype(int),
    })
    ph.to_csv(os.path.join(out_dir, "player_history.csv"), index=False)
    n_dgw_rows = ph.duplicated(subset=["player_id", "gameweek"]).sum()
    over90 = (ph.groupby(["player_id", "gameweek"])["minutes"].sum() > 90).sum()
    print(f"  player_history: {len(ph)} rows | DGW extra rows: {n_dgw_rows} "
          f"| player-GWs >90min: {over90}")

    # ── players_raw.csv (real ids/ownership; price = GW1 value) ─────────────
    gw1_price = (ph.sort_values("gameweek").groupby("player_id")["value"]
                   .first())
    players = pd.DataFrame({
        "id":            praw["id"].astype(int),
        "web_name":      praw["web_name"],
        "first_name":    praw["first_name"],
        "second_name":   praw["second_name"],
        "element_type":  praw["element_type"].astype(int),
        "position":      praw["element_type"].map(POS_MAP),
        "team":          praw["team"].astype(int),
        "selected_by_percent": praw["selected_by_percent"],
    })
    players["price"] = players["id"].map(gw1_price)
    # players with no history rows (never in a squad): end-season cost
    players["price"] = players["price"].fillna(praw["now_cost"] / 10.0)
    players["now_cost"] = (players["price"] * 10).round().astype(int)
    players.to_csv(os.path.join(out_dir, "players_raw.csv"), index=False)
    print(f"  players_raw: {len(players)} rows "
          f"({players['price'].isna().sum()} priceless)")

    # ── fixtures_raw.csv (real FDR) ──────────────────────────────────────────
    fx = fixtures.dropna(subset=["event"]).copy()
    fdf = pd.DataFrame({
        "gameweek":          fx["event"].astype(int),
        "team_h":            fx["team_h"].astype(int),
        "team_a":            fx["team_a"].astype(int),
        "team_h_difficulty": fx["team_h_difficulty"].astype(int),
        "team_a_difficulty": fx["team_a_difficulty"].astype(int),
    }).sort_values(["gameweek", "team_h"])
    fdf.to_csv(os.path.join(out_dir, "fixtures_raw.csv"), index=False)
    counts = pd.concat([fdf.groupby(["gameweek", "team_h"]).size(),
                        fdf.groupby(["gameweek", "team_a"]).size()])
    counts = counts.groupby(level=[0, 1]).sum()
    dgw = sorted({g for (g, _), c in counts.items() if c > 1})
    blanks = sorted({g for g in range(1, 39)
                     if len(set(fdf[fdf["gameweek"] == g][["team_h", "team_a"]]
                                .values.ravel())) < 20})
    print(f"  fixtures: {len(fdf)} | DGWs: {dgw} | blank GWs: {blanks}")


if __name__ == "__main__":
    seasons = sys.argv[1:] or ["2023-24", "2024-25"]
    for s in seasons:
        build_season(s)
    print("\ndone")
