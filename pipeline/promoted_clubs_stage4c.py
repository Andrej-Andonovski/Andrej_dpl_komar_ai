#!/usr/bin/env python3
"""
Stage 4c: Newly-Promoted Clubs — Championship Previous-Season Stats

Companion to Stage 4a (foreign/domestic transfers via Transfermarkt) for the
one case Transfermarkt's "new arrivals" page structurally cannot see: a
promoted club's RETAINED squad. Those players didn't transfer anywhere —
their club moved divisions — so their previous league (Championship) and
season (whatever the promoted club's finishing season was) are already known
outright; no Transfermarkt lookup is needed.

Reuses Stage 4a's FBref scraping/matching machinery (step3_scrape_fbref,
step4_filter_and_multiply) with a synthetic new_signings_df that has
previous_league_standardized="Championship" preset for the whole promoted
roster (minus anyone Stage 4a's real-transfer path already caught — e.g. a
promoted club's own summer signing from abroad).

Stage 4a's step5_build_position_files OVERWRITES new_signings_{pos}.csv, so
this script captures the existing contents first (written by Stage 4a for
this cycle's real transfers) and re-merges them back in after, deduped by
name (existing real-transfer rows take priority over a promoted-club row for
the same name — should never collide, but keep the same precedence policy
used elsewhere in this pipeline: more specific data wins).

Usage: python pipeline/promoted_clubs_stage4c.py --clubs "Coventry City,Hull City,Ipswich Town"
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# new_signings_stage4a wraps sys.stdout for UTF-8 on Windows as an import
# side effect -- must not be re-wrapped here (double-wrap closes the buffer).
from pipeline.new_signings_stage4a import (
    FPL_API_DIR, TRANSFERS_DIR, FBREF_SIGN_DIR,
    step3_scrape_fbref, step4_filter_and_multiply, step5_build_position_files,
)

DEFAULT_CLUBS = ["Coventry City", "Hull City", "Ipswich Town"]
PREV_LEAGUE = "Championship"


def build_promoted_signings_df(clubs):
    fpl = pd.read_csv(os.path.join(FPL_API_DIR, "players_raw.csv"))
    fpl["full_name"] = fpl["first_name"].str.strip() + " " + fpl["second_name"].str.strip()
    roster = fpl[fpl["team_name"].isin(clubs)].copy()
    print(f"  Promoted-club roster ({', '.join(clubs)}): {len(roster)} players")

    # Exclude anyone Stage 4a's Transfermarkt path already resolved to a real
    # previous league this cycle (e.g. a promoted club's own summer signing).
    ns_path = os.path.join(TRANSFERS_DIR, "new_signings_2025.csv")
    already_resolved = set()
    if os.path.exists(ns_path):
        ns = pd.read_csv(ns_path)
        if "previous_league_standardized" in ns.columns:
            resolved = ns[ns["previous_league_standardized"].notna()]
            already_resolved = set(resolved["fpl_id"].tolist())

    before = len(roster)
    roster = roster[~roster["id"].isin(already_resolved)]
    skipped = before - len(roster)
    if skipped:
        names = fpl[fpl["id"].isin(already_resolved) & fpl["team_name"].isin(clubs)]["full_name"].tolist()
        print(f"  Excluding {skipped} already resolved via real transfer this cycle: {names}")

    rows = []
    for _, p in roster.iterrows():
        rows.append({
            "fpl_id":           p["id"],
            "fpl_name":         p["full_name"],
            "fpl_team":         p["team_name"],
            "fpl_position":     p["position"],
            "fpl_price":        p["price"],
            "vaastav_match":    None,
            "match_confidence": 0,
            "previous_club":    p["team_name"],
            "previous_league":  PREV_LEAGUE,
            "previous_league_standardized": PREV_LEAGUE,
            "transfer_type":    "promotion",
        })
    df = pd.DataFrame(rows)
    print(f"  Synthetic new_signings_df: {len(df)} rows, previous_league={PREV_LEAGUE}")
    return df


def merge_back(existing_by_pos, saved_files):
    """After step5 overwrote new_signings_{pos}.csv with ONLY promoted-club
    rows, re-merge the pre-existing (real-transfer) rows back in.

    IMPORTANT: do NOT dedupe by player name. build_prev_lookup (Stage 6)
    expects MULTIPLE rows per player -- one per scraped season -- and does
    a reliability-weighted average across them (prev_seasons_available,
    prev_reliability_avg). Collapsing to one row per name silently destroys
    that multi-season signal. Only guard against a literal exact-duplicate
    row (all columns identical), which would only happen on an accidental
    re-run of this script.
    """
    pos_map = {"GK": "gk", "DEF": "def", "MID": "mid", "FWD": "fwd"}
    for fpl_pos, slug in pos_map.items():
        fpath = os.path.join(FBREF_SIGN_DIR, f"new_signings_{slug}.csv")
        new_part = pd.read_csv(fpath) if os.path.exists(fpath) else pd.DataFrame()
        old_part = existing_by_pos.get(fpl_pos, pd.DataFrame())
        if old_part.empty and new_part.empty:
            continue
        combined = pd.concat([old_part, new_part], ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(keep="first")  # exact full-row dupes only
        after = len(combined)
        combined.to_csv(fpath, index=False)
        dedup_note = f" (removed {before - after} exact-duplicate rows)" if before != after else ""
        print(f"  Merged {fpath}: {len(old_part)} existing + {len(new_part)} promoted-club "
              f"-> {after} rows{dedup_note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clubs", type=str, default=",".join(DEFAULT_CLUBS))
    ap.add_argument("--full-run", action="store_true")
    args = ap.parse_args()
    clubs = [c.strip() for c in args.clubs.split(",")]

    print("Stage 4c: Newly-Promoted Clubs -- Championship Previous-Season Stats")
    print(f"Clubs: {clubs}")
    print(f"Previous league: {PREV_LEAGUE} (SEASONS window set in new_signings_stage4a.SEASONS)")

    signings_df = build_promoted_signings_df(clubs)

    print("\n" + "=" * 70)
    print("PREVIEW -- players to scrape Championship stats for:")
    print("=" * 70)
    for pos in ["GK", "DEF", "MID", "FWD"]:
        subset = signings_df[signings_df["fpl_position"] == pos]
        if subset.empty:
            continue
        print(f"\n  {pos} ({len(subset)})")
        for _, r in subset.sort_values("fpl_team").iterrows():
            print(f"    {r['fpl_name']:<32} {r['fpl_team']:<16} {r['fpl_price']:.1f}")

    if not args.full_run:
        print("\n[STOPPED] Confirm the list above, then re-run with --full-run to continue.")
        return

    # Capture existing position-file contents (this cycle's real transfers,
    # just written by Stage 4a) before step5 overwrites them.
    pos_map = {"GK": "gk", "DEF": "def", "MID": "mid", "FWD": "fwd"}
    existing_by_pos = {}
    for fpl_pos, slug in pos_map.items():
        fpath = os.path.join(FBREF_SIGN_DIR, f"new_signings_{slug}.csv")
        existing_by_pos[fpl_pos] = pd.read_csv(fpath) if os.path.exists(fpath) else pd.DataFrame()

    fbref_combined = step3_scrape_fbref(signings_df)
    if fbref_combined.empty:
        print("\n[ERROR] No FBref data scraped -- aborting, existing files untouched.")
        return

    result_tuple = step4_filter_and_multiply(fbref_combined, signings_df)
    result_df = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
    if result_df.empty:
        print("\n[WARN] No matches found -- existing files untouched.")
        return

    step5_build_position_files(result_df, signings_df)
    merge_back(existing_by_pos, None)

    print("\n=== END: Stage 4c ===")


if __name__ == "__main__":
    main()
