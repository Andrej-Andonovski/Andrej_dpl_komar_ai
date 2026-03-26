"""
FPL AI — Stage 1: FPL API Data Fetcher
Pulls all necessary data from the official FPL API and saves as clean CSVs.
Run: python pipeline/data_fetcher_stage1.py
"""

import os
import json
import time
import requests
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "fpl_api")
SUMMARIES_DIR = os.path.join(RAW_DIR, "player_summaries")
FAILED_LOG = os.path.join(RAW_DIR, "failed_requests.txt")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# ── FPL API base ───────────────────────────────────────────────────────────────

FPL_BASE = "https://fantasy.premierleague.com/api"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FPL-AI-Research/1.0)"
}

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get(url: str, retries: int = 1) -> dict | None:
    """GET request with one retry on failure."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries:
                print(f"  [WARN] Request failed ({e}), retrying in 5s...")
                time.sleep(5)
            else:
                print(f"  [ERROR] Request failed after {retries + 1} attempts: {url} — {e}")
                return None


def log_failed(player_id: int):
    with open(FAILED_LOG, "a") as f:
        f.write(f"{player_id}\n")


# ── STEP 1 — Bootstrap Static ─────────────────────────────────────────────────

def fetch_bootstrap():
    print("\n=== STEP 1: Fetching bootstrap-static ===")
    data = get(f"{FPL_BASE}/bootstrap-static/")
    if data is None:
        raise RuntimeError("Failed to fetch bootstrap-static — cannot continue.")

    # Build lookup dicts
    team_lookup = {t["id"]: t["name"] for t in data["teams"]}

    # ── Teams ──────────────────────────────────────────────────────────────────
    team_fields = [
        "id", "name", "short_name", "strength",
        "strength_overall_home", "strength_overall_away",
        "strength_attack_home", "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ]
    teams_df = pd.DataFrame(data["teams"])[team_fields]
    teams_path = os.path.join(RAW_DIR, "teams_raw.csv")
    teams_df.to_csv(teams_path, index=False)
    print(f"  Saved {len(teams_df)} teams -> {teams_path}")

    # ── Players ────────────────────────────────────────────────────────────────
    player_fields = [
        "id", "first_name", "second_name", "team", "element_type",
        "now_cost", "selected_by_percent", "transfers_in", "transfers_out",
        "total_points", "points_per_game", "minutes",
        "goals_scored", "assists", "clean_sheets", "goals_conceded",
        "yellow_cards", "red_cards", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "form", "value_form", "value_season",
        "corners_and_indirect_freekicks_order",
        "direct_freekicks_order",
        "penalties_order",
    ]
    players_df = pd.DataFrame(data["elements"])[player_fields].copy()

    # Map team ID → name, position int → string, cost → £m
    players_df["team_name"] = players_df["team"].map(team_lookup)
    players_df["position"] = players_df["element_type"].map(POSITION_MAP)
    players_df["price"] = players_df["now_cost"] / 10.0

    # Set piece engineering
    players_df["is_corner_taker"] = (
        players_df["corners_and_indirect_freekicks_order"] == 1
    ).astype(int)
    players_df["is_freekick_taker"] = (
        players_df["direct_freekicks_order"] == 1
    ).astype(int)
    players_df["is_penalty_taker"] = (
        players_df["penalties_order"] == 1
    ).astype(int)
    players_df["is_set_piece_taker"] = (
        (players_df["is_corner_taker"] == 1)
        | (players_df["is_freekick_taker"] == 1)
        | (players_df["is_penalty_taker"] == 1)
    ).astype(int)

    players_path = os.path.join(RAW_DIR, "players_raw.csv")
    players_df.to_csv(players_path, index=False)
    print(f"  Saved {len(players_df)} players -> {players_path}")

    return players_df, teams_df, team_lookup, data


# ── STEP 2 — Fixtures ──────────────────────────────────────────────────────────

def fetch_fixtures(team_lookup: dict):
    print("\n=== STEP 2: Fetching fixtures ===")
    data = get(f"{FPL_BASE}/fixtures/")
    if data is None:
        raise RuntimeError("Failed to fetch fixtures — cannot continue.")

    rows = []
    for f in data:
        rows.append({
            "id": f["id"],
            "gameweek": f.get("event"),
            "team_h": f["team_h"],
            "team_a": f["team_a"],
            "team_h_name": team_lookup.get(f["team_h"]),
            "team_a_name": team_lookup.get(f["team_a"]),
            "team_h_difficulty": f.get("team_h_difficulty"),
            "team_a_difficulty": f.get("team_a_difficulty"),
            "finished": f.get("finished"),
            "kickoff_time": f.get("kickoff_time"),
        })

    fixtures_df = pd.DataFrame(rows)
    path = os.path.join(RAW_DIR, "fixtures_raw.csv")
    fixtures_df.to_csv(path, index=False)
    print(f"  Saved {len(fixtures_df)} fixtures -> {path}")
    return fixtures_df


# ── STEP 3 — Fixture Difficulty Table + Trajectory ────────────────────────────

def build_fixture_difficulty(fixtures_df: pd.DataFrame):
    print("\n=== STEP 3: Building fixture difficulty table ===")

    rows = []
    # Only include scheduled fixtures (gameweek not null)
    sched = fixtures_df.dropna(subset=["gameweek"]).copy()
    sched["gameweek"] = sched["gameweek"].astype(int)

    for _, row in sched.iterrows():
        gw = row["gameweek"]
        # Home team perspective
        rows.append({
            "team_name": row["team_h_name"],
            "gameweek": gw,
            "opponent": row["team_a_name"],
            "was_home": True,
            "fdr": row["team_h_difficulty"],
        })
        # Away team perspective
        rows.append({
            "team_name": row["team_a_name"],
            "gameweek": gw,
            "opponent": row["team_h_name"],
            "was_home": False,
            "fdr": row["team_a_difficulty"],
        })

    fdr_df = pd.DataFrame(rows).sort_values(["team_name", "gameweek"]).reset_index(drop=True)

    # ── Fixture trajectory score ───────────────────────────────────────────────
    # For each (team, gameweek) compute weighted FDR sum over next 5 GWs
    weights = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
    all_gws = sorted(fdr_df["gameweek"].unique())

    trajectory_rows = []
    for team in fdr_df["team_name"].unique():
        team_df = fdr_df[fdr_df["team_name"] == team].set_index("gameweek")
        for gw in all_gws:
            weighted_sum = 0.0
            count = 0
            for offset, w in weights.items():
                future_gw = gw + offset
                if future_gw in team_df.index:
                    fdr_val = team_df.loc[future_gw, "fdr"]
                    # Double gameweek: multiple rows → take mean FDR
                    if hasattr(fdr_val, "__len__"):
                        fdr_val = float(fdr_val.mean())
                    else:
                        fdr_val = float(fdr_val)
                    weighted_sum += fdr_val * w
                    count += 1
            trajectory_rows.append({
                "team_name": team,
                "gameweek": gw,
                "trajectory_raw": weighted_sum if count > 0 else float("nan"),
            })

    traj_df = pd.DataFrame(trajectory_rows)

    # Normalize per gameweek then invert (lower FDR = easier = higher score)
    for gw in traj_df["gameweek"].unique():
        mask = traj_df["gameweek"] == gw
        col = traj_df.loc[mask, "trajectory_raw"]
        min_v, max_v = col.min(), col.max()
        if max_v > min_v:
            traj_df.loc[mask, "trajectory_score"] = 1 - (col - min_v) / (max_v - min_v)
        else:
            traj_df.loc[mask, "trajectory_score"] = 0.5

    # Merge trajectory back onto fdr_df
    fdr_df = fdr_df.merge(
        traj_df[["team_name", "gameweek", "trajectory_raw", "trajectory_score"]],
        on=["team_name", "gameweek"],
        how="left",
    )

    path = os.path.join(RAW_DIR, "fixture_difficulty.csv")
    fdr_df.to_csv(path, index=False)
    print(f"  Saved {len(fdr_df)} rows -> {path}")
    return fdr_df


# ── STEP 4 — Per-Player Summaries ─────────────────────────────────────────────

def fetch_player_summaries(players_df: pd.DataFrame, team_lookup: dict):
    print("\n=== STEP 4: Fetching per-player summaries ===")
    player_ids = players_df["id"].tolist()
    total = len(player_ids)

    upcoming_rows = []
    history_rows = []
    failed = []

    for i, pid in enumerate(player_ids, 1):
        cache_path = os.path.join(SUMMARIES_DIR, f"{pid}.json")

        # Use cache if available
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                summary = json.load(f)
            if i % 100 == 0 or i == total:
                print(f"  [{i}/{total}] Player {pid} — loaded from cache")
        else:
            print(f"  [{i}/{total}] Fetching player {pid}...", end=" ", flush=True)
            summary = get(f"{FPL_BASE}/element-summary/{pid}/")
            if summary is None:
                print("FAILED")
                failed.append(pid)
                log_failed(pid)
                continue
            with open(cache_path, "w") as f:
                json.dump(summary, f)
            print("OK")
            time.sleep(0.5)

        # ── Upcoming fixtures ──────────────────────────────────────────────────
        for fix in summary.get("fixtures", []):
            upcoming_rows.append({
                "player_id": pid,
                "gameweek": fix.get("event"),
                "opponent_team_id": fix.get("opponent_team"),
                "opponent_team": team_lookup.get(fix.get("opponent_team")),
                "is_home": fix.get("is_home"),
                "difficulty": fix.get("difficulty"),
            })

        # ── Historical gameweek data ───────────────────────────────────────────
        for h in summary.get("history", []):
            history_rows.append({
                "player_id": pid,
                "gameweek": h.get("round"),
                "opponent_team_id": h.get("opponent_team"),
                "opponent_team": team_lookup.get(h.get("opponent_team")),
                "was_home": h.get("was_home"),
                "total_points": h.get("total_points"),
                "minutes": h.get("minutes"),
                "goals_scored": h.get("goals_scored"),
                "assists": h.get("assists"),
                "clean_sheets": h.get("clean_sheets"),
                "saves": h.get("saves"),
                "bonus": h.get("bonus"),
                "value": h.get("value", 0) / 10.0,
            })

    # Save upcoming fixtures
    upcoming_df = pd.DataFrame(upcoming_rows)
    upcoming_path = os.path.join(RAW_DIR, "player_upcoming_fixtures.csv")
    upcoming_df.to_csv(upcoming_path, index=False)
    print(f"\n  Saved {len(upcoming_df)} upcoming fixture rows -> {upcoming_path}")

    # Save history
    history_df = pd.DataFrame(history_rows)
    history_path = os.path.join(RAW_DIR, "player_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"  Saved {len(history_df)} history rows -> {history_path}")

    if failed:
        print(f"  [WARN] {len(failed)} players failed: {failed}")

    return upcoming_df, history_df


# ── STEP 5 — Validation Report ─────────────────────────────────────────────────

def validate(players_df, teams_df, fixtures_df, fdr_df, upcoming_df, history_df):
    print("\n=== STEP 5: Validation Report ===")
    sep = "-" * 50

    print(sep)
    print(f"Total players fetched:       {len(players_df)}")

    pos_counts = players_df["position"].value_counts()
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"  {pos:<5} {pos_counts.get(pos, 0)}")

    print(sep)
    print(f"Primary penalty takers:      {players_df['is_penalty_taker'].sum()}")
    print(f"Primary corner takers:       {players_df['is_corner_taker'].sum()}")
    print(f"Primary free kick takers:    {players_df['is_freekick_taker'].sum()}")

    print(sep)
    print(f"Total fixtures fetched:      {len(fixtures_df)}")
    gws = fixtures_df["gameweek"].dropna().unique()
    print(f"Gameweeks covered:           {int(min(gws))}–{int(max(gws))} ({len(gws)} GWs)")

    print(sep)
    missing_cost = players_df["now_cost"].isna().sum()
    missing_pos  = players_df["element_type"].isna().sum()
    print(f"Players missing now_cost:    {missing_cost}")
    print(f"Players missing element_type:{missing_pos}")

    print(sep)
    print("Top 10 most owned players:")
    top_owned = (
        players_df[["first_name", "second_name", "team_name", "position", "selected_by_percent"]]
        .sort_values("selected_by_percent", ascending=False)
        .head(10)
    )
    for _, r in top_owned.iterrows():
        print(f"  {r['first_name']} {r['second_name']:<20} {r['team_name']:<20} {r['position']}  {r['selected_by_percent']}%")

    print(sep)
    print("Top 10 highest priced players:")
    top_price = (
        players_df[["first_name", "second_name", "team_name", "position", "price"]]
        .sort_values("price", ascending=False)
        .head(10)
    )
    for _, r in top_price.iterrows():
        print(f"  {r['first_name']} {r['second_name']:<20} {r['team_name']:<20} {r['position']}  £{r['price']}m")

    print(sep)
    print("Sample rows from each CSV:")
    print("\n  players_raw.csv:")
    print(players_df.head(1).to_string(index=False))
    print("\n  teams_raw.csv:")
    print(teams_df.head(1).to_string(index=False))
    print("\n  fixtures_raw.csv:")
    print(fixtures_df.head(1).to_string(index=False))
    print("\n  fixture_difficulty.csv:")
    print(fdr_df.head(1).to_string(index=False))
    print("\n  player_upcoming_fixtures.csv:")
    if len(upcoming_df):
        print(upcoming_df.head(1).to_string(index=False))
    print("\n  player_history.csv:")
    if len(history_df):
        print(history_df.head(1).to_string(index=False))

    print(sep)
    print("Validation complete.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    players_df, teams_df, team_lookup, bootstrap_data = fetch_bootstrap()
    fixtures_df = fetch_fixtures(team_lookup)
    fdr_df = build_fixture_difficulty(fixtures_df)
    upcoming_df, history_df = fetch_player_summaries(players_df, team_lookup)
    validate(players_df, teams_df, fixtures_df, fdr_df, upcoming_df, history_df)
    print("\nAll done. Data saved to data/raw/fpl_api/")
