"""
pipeline/archive/add_penalty_feature.py
Stage 10 LSTM fix 3 — add `penalty_taker` to the training files + base table.

*** RULED OUT 2026-09-07 — kept for reference only. ***
Measured: gate-2 MAE regressed (mean +0.026 -> +0.016, the 28th feature
dimension degraded the thin early folds) AND it worsened the captain channel —
PK takers are high-ceiling but blank-prone, the q90 head already over-promotes
high-variance players, and flagging PK takers amplified that (2025-26
captain-regret proxy +1.24). Penalty-taker ceiling belongs in the fix-6 EV
captain objective, not as a mean-prediction feature. See
docs/stage10_phase1_report.md §12.


`penalty_taker` = 1.0 if the player is their team's designated FIRST-choice
penalty taker for that season, else 0.0. Penalties are the single biggest
captaincy-ceiling driver in FPL (a PK ~ 0.76 xG) and no existing feature
captures set-piece role.

Source: vaastav per-season players_raw.csv `penalties_order` (== 1). Joined to
the training rows by season-local player id via merged_gw.csv (name<->element),
then name<->name (the merged_gw name is the same string base_gw_table uses).
Available 2021-22..2025-26; 2019-20/2020-21 predate the FPL field -> 0.

Limitation (disclosed): `penalties_order` in players_raw is an END-OF-SEASON
snapshot, not per-GW. Set-piece roles are mostly stable within a season but a
mid-season change (transfer, manager) is attributed to the whole season. Mild
look-ahead on a categorical status feature; acceptable, documented.

Run once after a vaastav_repo sparse-checkout of the relevant seasons:
    python pipeline/add_penalty_feature.py
Idempotent — rewrites the column if already present.
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
REPO = os.path.join(_ROOT, "data", "raw", "vaastav_repo", "data")
PROC = os.path.join(_ROOT, "data", "processed")

TRAIN_FILES = ["train_gk.csv", "train_def.csv", "train_mid.csv", "train_fwd.csv"]
BASE = "base_gw_table.csv"
COL = "penalty_taker"


def build_lookup():
    """{(name, season): 1.0/0.0}."""
    lut = {}
    for season in sorted(os.listdir(REPO)):
        sdir = os.path.join(REPO, season)
        pr_p = os.path.join(sdir, "players_raw.csv")
        mg_p = os.path.join(sdir, "gws", "merged_gw.csv")
        if not (os.path.exists(pr_p) and os.path.exists(mg_p)):
            continue
        pr = pd.read_csv(pr_p)
        if "penalties_order" not in pr.columns:
            print(f"  {season}: no penalties_order — skip (all 0)")
            continue
        mg = pd.read_csv(mg_p, encoding="utf-8-sig",
                         usecols=["name", "element"]).drop_duplicates("element")
        j = mg.merge(pr[["id", "penalties_order"]], left_on="element",
                     right_on="id", how="left")
        n_pk = 0
        for r in j.itertuples(index=False):
            v = 1.0 if r.penalties_order == 1 else 0.0
            lut[(r.name, season)] = v
            n_pk += int(v)
        print(f"  {season}: {len(j)} players, {n_pk} first-PK takers")
    return lut


def patch(path, lut):
    df = pd.read_csv(path)
    if {"name", "season"} - set(df.columns):
        print(f"  {os.path.basename(path)}: no name/season — skip")
        return
    df[COL] = [lut.get((n, s), 0.0) for n, s in zip(df["name"], df["season"])]
    df.to_csv(path, index=False)
    cov = (df[COL] > 0).sum()
    print(f"  {os.path.basename(path)}: {len(df)} rows, {cov} penalty-taker rows "
          f"({cov / len(df) * 100:.1f}%), {df[COL].isna().sum()} NaN")


if __name__ == "__main__":
    print("Building penalty_taker lookup from vaastav_repo ...")
    lut = build_lookup()
    if not lut:
        sys.exit("no vaastav seasons found under data/raw/vaastav_repo/data/ — "
                 "sparse-checkout first")
    print(f"\nlookup: {len(lut)} (name, season) keys, "
          f"{sum(v for v in lut.values()):.0f} first-PK\n")
    for f in TRAIN_FILES + [BASE]:
        patch(os.path.join(PROC, f), lut)
