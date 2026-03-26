"""
patch_multipliers.py
Targeted fix: update league_multiplier for players whose previous league
was known from Stage 4b but missing from the transfers CSV.
Recomputes adjG_per_90 and adjA_per_90 for affected rows only.

Run: python pipeline/patch_multipliers.py
"""

import os
import sys
import unicodedata

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.join(os.path.dirname(__file__), "..")
SIGNINGS_DIR = os.path.join(BASE, "data", "raw", "fbref", "new_signings")

SIGNINGS_FILES = {
    "gk":  os.path.join(SIGNINGS_DIR, "new_signings_gk.csv"),
    "def": os.path.join(SIGNINGS_DIR, "new_signings_def.csv"),
    "mid": os.path.join(SIGNINGS_DIR, "new_signings_mid.csv"),
    "fwd": os.path.join(SIGNINGS_DIR, "new_signings_fwd.csv"),
}

# Manual overrides: normalized_name -> (display_league, multiplier)
# Normalized = NFD strip + lowercase + alpha only, same as normalize_name()
OVERRIDES = {
    # GKs
    "giorgi mamardashvili":  ("La Liga",              0.92),
    "mads hermansen":        ("Championship",         0.72),
    "filip jorgensen":       ("La Liga",              0.92),
    # DEFs
    "jeremie frimpong":      ("Bundesliga",           0.89),
    "maxence lacroix":       ("Bundesliga",           0.89),
    "matthijs de ligt":      ("Bundesliga",           0.89),
    "sepp van den berg":     ("Bundesliga",           0.89),
    "jean-clair todibo":     ("Ligue 1",              0.82),
    "leny yoro":             ("Ligue 1",              0.82),
    "jake o'brien":          ("Ligue 1",              0.82),
    "abdukodir khusanov":    ("Ligue 1",              0.82),
    "djed spence":           ("Ligue 1",              0.82),
    "riccardo calafiori":    ("Serie A",              0.88),
    "radu dragusin":         ("Serie A",              0.88),
    "michael kayode":        ("Serie A",              0.88),
    "patrick dorgu":         ("Serie A",              0.88),
    # MIDs
    "omar marmoush":         ("Bundesliga",           0.89),
    "mathys tel":            ("Bundesliga",           0.89),
    "iliman ndiaye":         ("Ligue 1",              0.82),
    "archie gray":           ("Championship",         0.72),
    "omari hutchinson":      ("Championship",         0.72),
    "mats wieffer":          ("Eredivisie",           0.75),
    "yankuba minteh":        ("Eredivisie",           0.75),
    "jorgen strand larsen":  ("Eredivisie",           0.75),
    "jorgenstrandlarsen":    ("Eredivisie",           0.75),  # alt without space
    "mikel merino zazon":    ("La Liga",              0.92),
    "savio moreira":         ("La Liga",              0.92),
    "carlos alcaraz duran":  ("Serie A",              0.88),
    "francisco evanilson":   ("Primeira Liga",        0.78),
    "matt o'riley":          ("Scottish Premiership", 0.65),
    # FWDs
    "joshua zirkzee":        ("Serie A",              0.88),
    "jorgenstrandlarsen":    ("Eredivisie",           0.75),
}


def normalize(s: str) -> str:
    """NFD + strip combining marks + lowercase + alpha-only."""
    nfd = unicodedata.normalize("NFD", str(s))
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return "".join(c for c in stripped.lower() if c.isalpha() or c in " '-")


def norm_key(s: str) -> str:
    """Normalize and collapse for lookup (alpha + space only)."""
    nfd = unicodedata.normalize("NFD", str(s))
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return "".join(c for c in stripped.lower() if c.isalpha() or c == " ").strip()


def lookup_override(name: str):
    """Return (league, multiplier) or None."""
    key = norm_key(name)
    if key in OVERRIDES:
        return OVERRIDES[key]
    # Also try alpha-only (no spaces) for names like Jorgen Strand Larsen / Jorgenstrandlarsen
    key_nospace = key.replace(" ", "")
    for ok, ov in OVERRIDES.items():
        if ok.replace(" ", "") == key_nospace:
            return ov
    return None


def recompute_adj(row) -> tuple:
    """Return (adjG_per_90, adjA_per_90) using updated multiplier."""
    src = str(row.get("data_source", "")).strip().lower()
    mult = float(row.get("league_multiplier", 1.0) or 1.0)
    if src == "vaastav":
        return (
            float(row.get("goals_per_game_season", 0) or 0),
            float(row.get("assists_per_game_season", 0) or 0),
        )
    mins = float(row.get("minutes", 0) or 0)
    goals = float(row.get("goals_scored", 0) or 0)
    assists = float(row.get("assists", 0) or 0)
    if mins > 0:
        adj_g = round((goals / mins * 90) * mult, 6)
        adj_a = round((assists / mins * 90) * mult, 6)
    else:
        adj_g = 0.0
        adj_a = 0.0
    return adj_g, adj_a


def process_file(key: str, path: str) -> tuple[pd.DataFrame, list]:
    """Returns (patched_df, change_log)."""
    df = pd.read_csv(path, low_memory=False)
    change_log = []

    for idx in df.index:
        name = str(df.at[idx, "name"])
        override = lookup_override(name)
        if override is None:
            continue
        league, new_mult = override
        old_mult = float(df.at[idx, "league_multiplier"])
        if abs(old_mult - new_mult) < 1e-9:
            continue  # Already correct — skip

        old_adjG = float(df.at[idx, "adjG_per_90"])
        old_adjA = float(df.at[idx, "adjA_per_90"])

        # Apply new multiplier
        df.at[idx, "league_multiplier"] = new_mult

        # Recompute adj columns for this row
        new_adjG, new_adjA = recompute_adj(df.loc[idx])
        df.at[idx, "adjG_per_90"] = new_adjG
        df.at[idx, "adjA_per_90"] = new_adjA

        change_log.append({
            "name": name,
            "league": league,
            "old_mult": old_mult,
            "new_mult": new_mult,
            "old_adjG": old_adjG,
            "new_adjG": new_adjG,
            "old_adjA": old_adjA,
            "new_adjA": new_adjA,
            "minutes": float(df.at[idx, "minutes"] or 0),
        })

    return df, change_log


def main():
    print()
    print("=" * 70)
    print("Multiplier Patch -- Targeted league_multiplier corrections")
    print("=" * 70)

    all_changes = []
    patched_dfs = {}

    for key, path in SIGNINGS_FILES.items():
        df, changes = process_file(key, path)
        patched_dfs[key] = df
        all_changes.extend(changes)

        if changes:
            print(f"\n  {key.upper()} file — {len(changes)} row(s) updated")
        else:
            print(f"\n  {key.upper()} file — no changes needed")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)

    # Unique players updated
    updated_names = sorted({c["name"] for c in all_changes})
    print(f"  Players updated:  {len(updated_names)}")

    # Count still at 1.0
    still_1 = set()
    for key, df in patched_dfs.items():
        mask = (df["league_multiplier"] - 1.0).abs() < 1e-9
        for n in df.loc[mask, "name"].dropna().unique():
            still_1.add(n)
    print(f"  Players still with multiplier=1.0 (genuinely unknown): {len(still_1)}")
    print()

    # ── Recomputed adjG report ────────────────────────────────────────────────
    # Aggregate per player (pick the row with most minutes for clean display)
    seen = {}
    for c in all_changes:
        name = c["name"]
        if name not in seen or c["minutes"] > seen[name]["minutes"]:
            seen[name] = c

    print(f"  {'Player':<35} {'Old adjG/90':>12} {'New adjG/90':>12} {'Multiplier':>12}")
    print(f"  {'-'*72}")
    for name in sorted(seen):
        c = seen[name]
        print(
            f"  {name:<35} {c['old_adjG']:>12.4f} {c['new_adjG']:>12.4f} "
            f"  {c['old_mult']:.2f} -> {c['new_mult']:.2f} ({c['league']})"
        )

    # ── Outlier check after patch ─────────────────────────────────────────────
    print()
    outliers = []
    for key, df in patched_dfs.items():
        big = df[(df["adjG_per_90"] > 1.5) & (df["minutes"] >= 500)]
        for _, r in big.iterrows():
            outliers.append((r["name"], key, r["adjG_per_90"], r["minutes"]))

    print(f"  adjG/90 > 1.5 with >= 500 mins (non-small-sample warnings): {len(outliers)}")
    if outliers:
        for name, key, g, m in sorted(outliers, key=lambda x: -x[2]):
            print(f"    {name} ({key}): adjG/90={g:.4f}, mins={m:.0f}")

    # ── Overwrite ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Overwriting files...")
    for key, df in patched_dfs.items():
        path = SIGNINGS_FILES[key]
        df.to_csv(path, index=False)
        fname = f"new_signings_{key}.csv"
        total_updated = sum(1 for c in all_changes
                            if c["name"] in df["name"].values)
        print(f"  {fname:<25} written  ({len(df)} rows, {len(df.columns)} cols)")

    print()
    print("Multiplier patch complete.")


if __name__ == "__main__":
    main()
