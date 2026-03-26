#!/usr/bin/env python3
"""
Stage 4b verification script.

Cross-checks the Step 1 confirmed/unconfirmed/flagged lists against
multiple ground-truth sources so you can decide what (if anything) to
add to STAGE4B_EXCLUSIONS before typing y at Gate 1.

Checks performed:
  1. Does the vaastav_name actually match the FPL player shown?
     (re-runs the fuzzy match and shows the score + FPL name)
  2. Is the player genuinely new to vaastav?
     (shows earliest vaastav season, if any)
  3. Does Stage 4a already cover this player?
     (checks all 4 position files)
  4. Is the player in new_signings_2025.csv (Stage 4a targets)?
     (those should NOT appear here)
  5. What is the player's birth year from FBref cache?
  6. Spot-check a few flagged players — are they really name-change risks?

Usage:
    python pipeline/verify_stage4b.py
"""

import io
import json
import os
import re
import sys
import unicodedata
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
try:
    from fuzzywuzzy import fuzz, process as fzprocess
except ImportError:
    print("[ERROR] fuzzywuzzy not installed: pip install fuzzywuzzy")
    sys.exit(1)

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
VAASTAV_DIR   = os.path.join(RAW_DIR, "vaastav")
FPL_API_DIR   = os.path.join(RAW_DIR, "fpl_api")
FBREF_RAW_DIR = os.path.join(RAW_DIR, "fbref", "raw")
SIGN_DIR      = os.path.join(RAW_DIR, "fbref", "new_signings")
TRANSFERS_DIR = os.path.join(RAW_DIR, "transfers")
STATE_FILE    = os.path.join(RAW_DIR, "fbref", "stage4b_state.json")


def normalize(name):
    nfd = unicodedata.normalize("NFD", str(name))
    ascii_str = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    cleaned = re.sub(r"[^\w\s-]", "", ascii_str.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


# ── Load state ────────────────────────────────────────────────────────────────
if not os.path.exists(STATE_FILE):
    print("[ERROR] State file not found. Run Stage 4b Step 1 first.")
    sys.exit(1)

with open(STATE_FILE, "r", encoding="utf-8") as f:
    state = json.load(f)

confirmed   = state["step1"]["confirmed"]
unconfirmed = state["step1"]["unconfirmed"]
flagged     = state["step1"]["flagged"]
excluded    = state["step1"]["excluded"]

# ── Load reference data ───────────────────────────────────────────────────────
vaastav = pd.read_csv(os.path.join(VAASTAV_DIR, "historical_gw_data.csv"))
fpl_raw = pd.read_csv(os.path.join(FPL_API_DIR, "players_raw.csv"))
fpl_raw["full_name"] = fpl_raw["first_name"].str.strip() + " " + fpl_raw["second_name"].str.strip()
fpl_norm_list = [normalize(n) for n in fpl_raw["full_name"]]
fpl_norm_to_row = {normalize(r["full_name"]): r for _, r in fpl_raw.iterrows()}

# Stage 4a coverage
stage4a_names_norm = set()
for slug in ["gk", "def", "mid", "fwd"]:
    fpath = os.path.join(SIGN_DIR, f"new_signings_{slug}.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        stage4a_names_norm.update(normalize(n) for n in df["name"].dropna())

# new_signings_2025.csv
transfers_path = os.path.join(TRANSFERS_DIR, "new_signings_2025.csv")
transfers_norm = set()
if os.path.exists(transfers_path):
    tf = pd.read_csv(transfers_path)
    transfers_norm = {normalize(n) for n in tf["fpl_name"].dropna()}

# FBref birth years from cache files
born_map = {}  # norm_player_name -> born_year
for fname in os.listdir(FBREF_RAW_DIR):
    if not fname.endswith(".csv"):
        continue
    try:
        df = pd.read_csv(os.path.join(FBREF_RAW_DIR, fname))
    except Exception:
        continue
    # Find player and born columns
    pcol   = next((c for c in df.columns if "player" in c.lower()), None)
    bcol   = next((c for c in df.columns if "born"   in c.lower()), None)
    if not pcol or not bcol:
        continue
    for _, row in df.iterrows():
        pname = str(row.get(pcol, ""))
        if not pname or pname == "nan":
            continue
        try:
            by = int(float(row[bcol]))
            born_map[normalize(pname)] = by
        except (ValueError, TypeError):
            pass


def get_born(player_name):
    return born_map.get(normalize(player_name), None)


def age_from_born(by):
    return 2026 - by if by else None


def vaastav_seasons(name):
    rows = vaastav[vaastav["name"] == name]
    return sorted(rows["season"].unique().tolist())


def best_fpl_match(vaastav_name):
    norm = normalize(vaastav_name)
    res = fzprocess.extractOne(norm, fpl_norm_list, scorer=fuzz.token_sort_ratio)
    if res and res[1] >= 70:
        row = fpl_norm_to_row.get(res[0])
        return row["full_name"] if row is not None else res[0], res[1]
    return None, 0


def in_stage4a(fpl_name):
    return normalize(fpl_name) in stage4a_names_norm


def in_transfers_csv(fpl_name):
    return normalize(fpl_name) in transfers_norm


# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 80

print(SEP)
print("STAGE 4b STEP 1 VERIFICATION")
print(SEP)
print(f"State file: {STATE_FILE}")
print(f"Confirmed: {len(confirmed)}  |  Unconfirmed: {len(unconfirmed)}  "
      f"|  Flagged: {len(flagged)}  |  Excluded: {len(excluded)}")


# ── CHECK 1: CONFIRMED list ───────────────────────────────────────────────────
print()
print(SEP)
print("CHECK 1 — CONFIRMED 2024/25 PL DEBUTANTS")
print("Columns: #  Vaastav name  |  FPL match (score%)  |  FPL pos/price  |")
print("         Born  Age  |  Vaastav seasons  |  In4a?  InTF?")
print(SEP)

problems = []
for i, e in enumerate(confirmed, 1):
    vn        = e["vaastav_name"]
    fpl_name  = e["fpl_name"]
    fpl_pos   = e["fpl_pos"]
    price     = e["price"]
    fpl_team  = e["fpl_team"]

    # Re-check fuzzy match
    best_match, score = best_fpl_match(vn)

    # Vaastav seasons (should be only 2024-25)
    seasons = vaastav_seasons(vn)

    # Birth year
    by  = get_born(fpl_name) or get_born(vn)
    age = age_from_born(by) if by else None

    # Stage 4a / transfers
    in4a = in_stage4a(fpl_name)
    intf = in_transfers_csv(fpl_name)

    # Flag problems
    flags = []
    if best_match and normalize(best_match) != normalize(fpl_name):
        flags.append(f"FPL match mismatch: fuzzy says '{best_match}' ({score}%), "
                     f"state says '{fpl_name}'")
    if score < 80:
        flags.append(f"LOW match score ({score}%) — possible wrong player")
    # First-name overlap check: if NONE of the significant tokens in vaastav_name
    # appear in fpl_name, it's likely a wrong match (e.g. Alex -> Cole)
    vn_tokens  = {t for t in normalize(vn).split() if len(t) >= 3}
    fn_tokens  = {t for t in normalize(fpl_name).split() if len(t) >= 3}
    overlap    = vn_tokens & fn_tokens
    if vn_tokens and not overlap:
        flags.append(
            f"NAME TOKEN MISMATCH: vaastav='{vn}' shares NO tokens with FPL='{fpl_name}' "
            f"-- almost certainly the wrong player"
        )
    if "2023-24" in seasons or "2022-23" in seasons:
        flags.append(f"Has PRIOR vaastav seasons: {seasons} -- should NOT be confirmed!")
    # Name-order / encoding guard: check if a TOKEN-REVERSED version of vaastav_name
    # appears in prior vaastav seasons (catches "Mitoma Kaoru" vs "Kaoru Mitoma")
    vn_parts   = vn.split()
    reversed_vn = " ".join(reversed(vn_parts)) if len(vn_parts) >= 2 else ""
    if reversed_vn and reversed_vn != vn:
        prior_vaastav_names = set(
            vaastav[vaastav["season"].isin(["2019-20","2020-21","2021-22","2022-23","2023-24"])]["name"].unique()
        )
        if reversed_vn in prior_vaastav_names or normalize(reversed_vn) in {normalize(n) for n in prior_vaastav_names}:
            flags.append(
                f"NAME ORDER ISSUE: '{reversed_vn}' exists in prior vaastav seasons "
                f"-- this is NOT a 2024-25 debutant, it's the same player listed differently"
            )
    if in4a:
        flags.append("ALREADY IN Stage 4a position files -- should be excluded!")
    if intf:
        flags.append("In new_signings_2025.csv -- already a Stage 4a target!")

    status = "  [FLAG]" if flags else "      OK"
    born_str = str(by) if by else "?"
    age_str  = str(age) if age else "?"

    print(f"{status} {i:>2}. {vn:<32} -> {fpl_name:<32} ({score}%)")
    print(f"         Pos={fpl_pos}  Price=£{price:.1f}m  Team={fpl_team}")
    print(f"         Born={born_str}  Age={age_str}  "
          f"Vaastav seasons={seasons}  In4a={in4a}  InTransfers={intf}")
    for fl in flags:
        print(f"         *** {fl}")
    if flags:
        problems.append((i, vn, fpl_name, flags))

print()
if problems:
    print(f"  PROBLEMS FOUND in confirmed list ({len(problems)}):")
    for idx, vn, fn, flgs in problems:
        print(f"    #{idx} {vn} -> {fn}")
        for fl in flgs:
            print(f"         {fl}")
    print()
    print("  ACTION: Add these vaastav names to STAGE4B_EXCLUSIONS in the script")
    print("          or verify they are correct before proceeding.")
else:
    print("  All confirmed players look clean.")


# ── CHECK 2: UNCONFIRMED list ─────────────────────────────────────────────────
print()
print(SEP)
print("CHECK 2 — UNCONFIRMED (age 28+ or pre-2019 PL history)")
print(SEP)
for i, e in enumerate(unconfirmed, 1):
    vn   = e["vaastav_name"]
    fn   = e["fpl_name"]
    by   = get_born(fn) or get_born(vn)
    age  = age_from_born(by) if by else "?"
    pre  = e.get("has_pre_vaastav_pl", False)
    seasons = vaastav_seasons(vn)
    best_match, score = best_fpl_match(vn)
    print(f"  {i:>2}. {vn:<32} -> {fn}")
    print(f"       Born={by or '?'}  Age={age}  pre-vaastav={pre}  seasons={seasons}")
    print(f"       FPL match score={score}%")


# ── CHECK 3: FLAGGED list (name change suspects) ──────────────────────────────
print()
print(SEP)
print("CHECK 3 — FLAGGED (possible name change / data issue)")
print("These were excluded from confirmed. Review to see if any are genuine debutants")
print("that should be added to confirmed (by removing from flag logic).")
print(SEP)
for i, e in enumerate(flagged, 1):
    vn     = e["vaastav_name"]
    fn     = e["fpl_name"]
    reason = e["reason"]
    by     = get_born(fn) or get_born(vn)
    age    = age_from_born(by) if by else "?"
    seasons = vaastav_seasons(vn)
    best_match, score = best_fpl_match(vn)
    in4a   = in_stage4a(fn)
    intf   = in_transfers_csv(fn)
    print(f"  {i:>2}. {vn:<32} -> {fn}")
    print(f"       Reason: {reason}")
    print(f"       Born={by or '?'}  Age={age}  seasons={seasons}  "
          f"match={score}%  In4a={in4a}  InTransfers={intf}")


# ── CHECK 4: Stage 4a coverage spot-check ─────────────────────────────────────
print()
print(SEP)
print("CHECK 4 — Stage 4a position file coverage")
print(SEP)
for slug in ["gk", "def", "mid", "fwd"]:
    fpath = os.path.join(SIGN_DIR, f"new_signings_{slug}.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        seasons = sorted(df["season"].dropna().unique().tolist()) if "season" in df.columns else []
        print(f"  new_signings_{slug}.csv: {len(df)} rows, seasons: {seasons[:6]}")
    else:
        print(f"  new_signings_{slug}.csv: NOT FOUND")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print(SEP)
print("SUMMARY")
print(SEP)
total_confirmed_clean = len(confirmed) - len(problems)
print(f"  Confirmed (clean):      {total_confirmed_clean}")
print(f"  Confirmed (problems):   {len(problems)}")
print(f"  Unconfirmed (age 28+):  {len(unconfirmed)}")
print(f"  Flagged (name change):  {len(flagged)}")
print()
if problems:
    print("  RECOMMENDED STAGE4B_EXCLUSIONS additions:")
    for _, vn, fn, _ in problems:
        print(f'    "{vn}",  # fuzzy-matched to wrong FPL player or already in Stage 4a')
    print()
    print("  Add these to STAGE4B_EXCLUSIONS in pipeline/data_loader_stage4b.py")
    print("  then re-run with: python pipeline/data_loader_stage4b.py")
else:
    print("  Confirmed list is clean. Proceed with: python pipeline/data_loader_stage4b.py")
    print("  Type y at Gate 1.")
print(SEP)
