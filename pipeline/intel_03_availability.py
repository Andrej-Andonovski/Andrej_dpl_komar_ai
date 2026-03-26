"""
Intel 03: Player Availability Confidence Score
Merges FPL live status (intel_01) + press conference data (intel_02)
into a per-player 0-100 availability score for each GW (GW1-28).

Reads:  data/intel/fpl_live.json
        data/intel/press_conferences.json
Writes: data/intel/availability.json
"""

import sys
import os
import json
import re
import unicodedata
from datetime import datetime, timezone
from collections import defaultdict, Counter

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEL_DIR  = os.path.join(ROOT_DIR, "data", "intel")
LIVE_PATH  = os.path.join(INTEL_DIR, "fpl_live.json")
PRESS_PATH = os.path.join(INTEL_DIR, "press_conferences.json")
OUT_PATH   = os.path.join(INTEL_DIR, "availability.json")

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# FPL status -> base score (None = use chance_next)
FPL_STATUS_SCORES = {
    "a": 100,
    "d": None,   # use chance_next if available, else 50
    "i": 5,
    "s": 0,
    "u": 0,
}

PRESS_AVAIL_SCORES = {
    "available":  95,
    "doubtful":   40,
    "out":        5,
    "suspended":  0,
    "unknown":    50,
}

# (threshold, tier_label) — first match wins (high to low)
AVAILABILITY_TIERS = [
    (80, "available"),
    (60, "probable"),
    (30, "doubtful"),
    (10, "unlikely"),
    (0,  "out"),
]

# Manual alias table: press conf display name -> lookup name
ALIASES = {
    "Son":             "Son Heung-min",
    "B.Fernandes":     "Bruno Fernandes",
    "De Gea":          "David de Gea",
}

# Press conference club names -> likely fragment of FPL team name
# Used for fuzzy club matching when exact match fails
PRESS_CLUB_HINTS = {
    "Newcastle":         "newcastle",
    "Tottenham":         "spurs",
    "Wolves":            "wolverhampton",
    "West Ham":          "west ham",
    "Nottingham Forest": "nott",
    "Manchester City":   "man city",
    "Manchester United": "man utd",
    "Crystal Palace":    "crystal palace",
    "Aston Villa":       "aston villa",
    "Brighton":          "brighton",
}

# Clubs present in GW1-10 press conferences but absent from current FPL data (relegated)
RELEGATED_CLUBS = {"Ipswich", "Leicester", "Southampton"}


# ===========================================================================
# Scoring helpers
# ===========================================================================

def fpl_score(status: str, chance_next) -> int:
    """Convert FPL status + chance_next to 0-100 score."""
    if status == "d":
        if chance_next is not None:
            return int(chance_next)
        return 50
    return FPL_STATUS_SCORES.get(status, 100)


def press_score_from_avail(availability: str) -> int:
    """Convert press conf availability string to 0-100 score."""
    return PRESS_AVAIL_SCORES.get(availability, 50)


def availability_tier(pct: int) -> str:
    """Map 0-100 score to tier label."""
    for threshold, label in AVAILABILITY_TIERS:
        if pct >= threshold:
            return label
    return "out"


# ===========================================================================
# Name normalisation
# ===========================================================================

def normalize_name(name: str) -> str:
    """
    Lowercase, strip diacritics, remove all non-alphanumeric characters.
    e.g. "Sørloth" -> "sorloth", "O'Riley" -> "oriley", "Van Dijk" -> "vandijk"
    """
    # NFD decomposition strips most accents
    nfd = unicodedata.normalize("NFD", name)
    cleaned = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Handle characters not covered by NFD
    cleaned = (cleaned
               .replace("\u0141", "l").replace("\u0142", "l")   # Polish L/l
               .replace("\u0131", "i")                          # Turkish dotless i
               .replace("\u00df", "ss"))                        # German sz
    # Strip everything except a-z 0-9
    return re.sub(r"[^a-z0-9]", "", cleaned.lower())


def last_name(name: str) -> str:
    """Return the last whitespace-separated token of a name."""
    parts = name.strip().split()
    return parts[-1] if parts else name


# ===========================================================================
# Club mapping
# ===========================================================================

def build_club_to_short(teams: dict) -> dict:
    """
    Map press conference club names -> FPL team_short.

    FPL teams dict: tid_str -> {"name": "Arsenal", "short_name": "ARS", ...}
    Press clubs use the PL_CLUBS list from intel_02.
    """
    # Build fpl name (lowercase) -> short_name
    fpl_lower_to_short = {v["name"].lower(): v["short_name"] for v in teams.values()}

    press_clubs = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
        "Leeds", "Leicester", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle", "Nottingham Forest", "Southampton", "Sunderland",
        "Tottenham", "West Ham", "Wolves",
    ]

    result = {}
    for pc in press_clubs:
        pc_lower = pc.lower()

        # 1. Exact match
        if pc_lower in fpl_lower_to_short:
            result[pc] = fpl_lower_to_short[pc_lower]
            continue

        # 2. Hint-based partial match
        hint = PRESS_CLUB_HINTS.get(pc, pc_lower)
        found = None
        for fpl_nm, short in fpl_lower_to_short.items():
            if hint in fpl_nm or fpl_nm in pc_lower:
                found = short
                break
        if found:
            result[pc] = found
            continue

        # 3. Fallback: first word of press club in FPL name
        first_word = pc_lower.split()[0]
        if len(first_word) >= 4:
            for fpl_nm, short in fpl_lower_to_short.items():
                if first_word in fpl_nm:
                    result[pc] = short
                    break

    return result  # press_club_name -> team_short


# ===========================================================================
# Name matching
# ===========================================================================

def match_player(press_name: str, press_club: str,
                 club_to_short: dict,
                 short_to_players: dict) -> dict | None:
    """
    Find the FPL player dict that corresponds to a press conference entry.

    Matching order:
      1. Exact normalised web_name within club
      2. Exact normalised full_name within club
      3. Press name (normalised) is a substring of normalised full_name (>=4 chars)
      4. Last-name match within club
      5. Last-name of press name appears anywhere in normalised full_name
    """
    # Apply known aliases first
    display = ALIASES.get(press_name, press_name)

    team_short = club_to_short.get(press_club)
    if team_short is None:
        return None

    candidates = short_to_players.get(team_short, [])
    if not candidates:
        return None

    norm_press = normalize_name(display)
    if len(norm_press) < 3:
        return None

    # 1. Exact web_name within club
    for p in candidates:
        if normalize_name(p["web_name"]) == norm_press:
            return p

    # 2. Exact full_name within club
    for p in candidates:
        if normalize_name(p["full_name"]) == norm_press:
            return p

    # 3. Press name is a contiguous substring of full_name (min 4 chars)
    if len(norm_press) >= 4:
        for p in candidates:
            norm_full = normalize_name(p["full_name"])
            if norm_press in norm_full:
                return p

    # 4. Last-name exact match within club
    press_last = normalize_name(last_name(display))
    if len(press_last) >= 4:
        for p in candidates:
            if normalize_name(last_name(p["full_name"])) == press_last:
                return p

    # 5. Last-name of press name appears in full_name within club
    if len(press_last) >= 4:
        for p in candidates:
            if press_last in normalize_name(p["full_name"]):
                return p

    return None


# ===========================================================================
# Core availability computation
# ===========================================================================

def compute_availability(player_fpl: dict, press_entry: dict | None,
                          is_backtest: bool = False) -> dict:
    """
    Merge FPL and press conference signals into one availability record.

    is_backtest=True: press conference is the sole scoring signal.
      FPL live fields (status, chance, transfers) are GW29 snapshots and
      must not contaminate historical GW scores.
    is_backtest=False (live mode): both sources are current; blend them.

    Returns the full availability dict for this player.
    """
    status      = player_fpl.get("status", "a")
    chance_next = player_fpl.get("chance_next")
    net_tr      = player_fpl.get("net_transfers", 0) or 0
    tp          = player_fpl.get("transfer_pressure", "stable")

    fpl_s = fpl_score(status, chance_next)

    flags = []
    press_s      = None
    press_avail  = None
    press_injury = None
    press_quote  = None

    if press_entry:
        press_avail  = press_entry.get("availability", "unknown")
        press_injury = (press_entry.get("injury") or "").strip() or None
        press_quote  = (press_entry.get("news") or "").strip() or None
        press_s      = press_score_from_avail(press_avail)

    # --- Merge ---
    if is_backtest:
        # Press conf is the only accurate per-GW source; FPL data is GW29 snapshot
        merged = press_s if press_s is not None else 95
    else:
        if press_s is not None:
            merged = round(0.65 * press_s + 0.35 * fpl_s)
        else:
            merged = fpl_s

    # --- Source agreement (live mode only — backtest FPL data is off-GW) ---
    sources_agree = None
    if not is_backtest and press_s is not None:
        fpl_avail_flag    = (fpl_s  >= 80)
        press_avail_flag  = (press_s >= 80)
        fpl_out_flag      = (fpl_s  <= 9)
        press_out_flag    = (press_s <= 9)

        if (fpl_avail_flag and press_avail_flag) or (fpl_out_flag and press_out_flag):
            sources_agree = True
            merged = min(100, merged + 5)
        elif (fpl_avail_flag and press_out_flag) or (fpl_out_flag and press_avail_flag):
            sources_agree = False
            flags.append("conflicting_sources")
        else:
            sources_agree = True

    # --- Crowd wisdom modifier (live mode only — transfers are GW29 in backtest) ---
    if not is_backtest and tp == "selling" and net_tr < -100_000:
        merged = max(5, merged - 10)
        flags.append("mass_sell")

    # --- Returning from injury flag (live mode only) ---
    if not is_backtest and status == "i" and press_s is not None and press_s >= 80:
        flags.append("returning_from_injury")

    merged = max(0, min(100, merged))

    return {
        "availability_pct":  merged,
        "availability_tier": availability_tier(merged),
        "fpl_status":        status,
        "fpl_chance":        chance_next,
        "press_conf_status": press_avail,
        "press_conf_injury": press_injury,
        "press_conf_quote":  press_quote,
        "sources_agree":     sources_agree,
        "crowd_signal":      tp,
        "flags":             flags,
    }


# ===========================================================================
# Per-GW processing
# ===========================================================================

def process_gw(gw: int, gw_press: dict,
               fpl_players: dict,
               club_to_short: dict,
               short_to_players: dict,
               is_backtest: bool = True) -> dict:
    """
    Score all 820 players for one GW.
    is_backtest=True: press conf is sole scoring signal (FPL data = identification only).
    Returns dict with tier_counts, flagged players, and stats.
    """
    all_player_news = gw_press.get("all_player_news", [])

    # --- Build press lookup: player_id_str -> press entry ---
    # If a player appears multiple times, keep the worst (lowest) availability score
    press_by_pid: dict[str, dict] = {}
    matched_count   = 0
    unmatched_count = 0

    for entry in all_player_news:
        press_name = entry.get("player", "")
        press_club = entry.get("club", "")

        p_match = match_player(press_name, press_club, club_to_short, short_to_players)
        if p_match:
            pid_str = str(p_match["player_id"])
            if pid_str not in press_by_pid:
                press_by_pid[pid_str] = entry
            else:
                # Keep the worse (lower) availability for this GW
                existing_s = press_score_from_avail(
                    press_by_pid[pid_str].get("availability", "unknown"))
                new_s = press_score_from_avail(entry.get("availability", "unknown"))
                if new_s < existing_s:
                    press_by_pid[pid_str] = entry
            matched_count += 1
        else:
            unmatched_count += 1

    # --- Score every player ---
    flagged_players: dict[str, dict] = {}
    fully_available_count = 0
    tier_counts: Counter = Counter()

    for pid_str, player in fpl_players.items():
        press_entry = press_by_pid.get(pid_str)
        result = compute_availability(player, press_entry, is_backtest=is_backtest)

        pct = result["availability_pct"]
        tier_counts[result["availability_tier"]] += 1

        # Only flag players with some concern (< 95% OR any flags OR press mention)
        if pct >= 95 and not result["flags"] and press_entry is None:
            fully_available_count += 1
        else:
            flagged_players[pid_str] = {
                "player_id": int(pid_str),
                "web_name":  player["web_name"],
                "full_name": player["full_name"],
                "team_short": player["team_short"],
                "position": player["position"],
                **result,
            }

    mode = "backtest" if is_backtest else "live"
    return {
        "gw":                     gw,
        "mode":                   mode,
        "total_players_assessed": len(fpl_players),
        "flagged_players":        len(flagged_players),
        "fully_available_count":  fully_available_count,
        "press_matches":          matched_count,
        "press_unmatched":        unmatched_count,
        "tier_counts":            dict(tier_counts),
        "players":                flagged_players,
    }


# ===========================================================================
# Console report helpers
# ===========================================================================

def print_gw_report(gw: int, gw_result: dict) -> None:
    tc = gw_result["tier_counts"]
    print(f"  Assessed : {gw_result['total_players_assessed']} players  |  "
          f"Flagged: {gw_result['flagged_players']}  |  "
          f"Fully available: {gw_result['fully_available_count']}")
    print(f"  Press    : {gw_result['press_matches']} matched  |  "
          f"{gw_result['press_unmatched']} unmatched")
    print(f"  Tiers    : "
          f"available={tc.get('available', 0)}  "
          f"probable={tc.get('probable', 0)}  "
          f"doubtful={tc.get('doubtful', 0)}  "
          f"unlikely={tc.get('unlikely', 0)}  "
          f"out={tc.get('out', 0)}")

    # Players below 80%
    below80 = sorted(
        [d for d in gw_result["players"].values() if d["availability_pct"] < 80],
        key=lambda d: d["availability_pct"],
    )
    if below80:
        print()
        print(f"  Players below 80% availability ({len(below80)} total):")
        for d in below80[:25]:
            pct   = d["availability_pct"]
            tier  = d["availability_tier"]
            wn    = d["web_name"]
            inj   = d.get("press_conf_injury") or ""
            flags = ", ".join(d["flags"]) if d["flags"] else ""
            line  = f"    [{pct:3d}%] {wn:<20} ({d['team_short']:<3}) {tier}"
            if inj:
                line += f"  [{inj}]"
            if flags:
                line += f"  ({flags})"
            print(line)
        if len(below80) > 25:
            print(f"    ... +{len(below80) - 25} more")

    # Conflicting sources
    conflicts = [d["web_name"] for d in gw_result["players"].values()
                 if "conflicting_sources" in d.get("flags", [])]
    if conflicts:
        print()
        print(f"  Conflicting sources ({len(conflicts)}): "
              + ", ".join(conflicts[:12])
              + (f" +{len(conflicts)-12} more" if len(conflicts) > 12 else ""))


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(INTEL_DIR, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   FPL AVAILABILITY INTEL -- Intel 03                ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # --- Load inputs ---
    print("Loading fpl_live.json...", end=" ", flush=True)
    with open(LIVE_PATH, "r", encoding="utf-8") as f:
        live_data = json.load(f)
    fpl_players = live_data["players"]   # pid_str -> dict
    teams_data  = live_data["teams"]     # tid_str -> dict
    current_gw  = live_data["current_gw"]
    print(f"{len(fpl_players)} players  |  current GW: {current_gw}")

    print("Loading press_conferences.json...", end=" ", flush=True)
    with open(PRESS_PATH, "r", encoding="utf-8") as f:
        press_data = json.load(f)
    gws_scraped = press_data.get("gws_scraped", [])
    print(f"GWs scraped: {gws_scraped}")
    print()

    # Add player_id to each player dict for convenience during matching
    for pid_str, p in fpl_players.items():
        p["player_id"] = int(pid_str)

    # --- Build club/team mappings ---
    club_to_short = build_club_to_short(teams_data)
    # short_to_players: team_short -> list of player dicts (same team)
    short_to_players: dict[str, list] = defaultdict(list)
    for p in fpl_players.values():
        ts = p.get("team_short", "")
        if ts:
            short_to_players[ts].append(p)

    # Debug: show club mapping (helps spot unmapped clubs)
    all_press_clubs = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
        "Leeds", "Leicester", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle", "Nottingham Forest", "Southampton", "Sunderland",
        "Tottenham", "West Ham", "Wolves",
    ]
    missing_clubs  = [pc for pc in all_press_clubs if pc not in club_to_short]
    relegated      = [pc for pc in missing_clubs if pc in RELEGATED_CLUBS]
    truly_missing  = [pc for pc in missing_clubs if pc not in RELEGATED_CLUBS]
    if relegated:
        print(f"  [INFO] Clubs not in current FPL data (relegated): {', '.join(relegated)}")
    if truly_missing:
        print(f"  [WARN] Unexpected club mapping failures: {truly_missing}")

    # --- Process each GW ---
    gameweeks_out: dict[str, dict] = {}
    injury_frequency: Counter = Counter()   # player_name -> GWs flagged below 80%

    press_gws = press_data.get("gameweeks", {})
    for gw_str in sorted(press_gws.keys(), key=int):
        gw       = int(gw_str)
        gw_press = press_gws[gw_str]

        print(f"── GW{gw} {'─' * 47}")

        if gw_press.get("status") != "success":
            print(f"  [SKIP] status = {gw_press.get('status')}")
            print()
            continue

        is_backtest = (gw != current_gw)
        gw_result = process_gw(gw, gw_press, fpl_players, club_to_short, short_to_players,
                               is_backtest=is_backtest)
        gameweeks_out[gw_str] = gw_result

        print_gw_report(gw, gw_result)
        print()

        # Accumulate cross-GW injury tracking
        for d in gw_result["players"].values():
            if d["availability_pct"] < 80:
                injury_frequency[d["web_name"]] += 1

    # --- Cross-GW summary ---
    total_flagged = sum(v["flagged_players"] for v in gameweeks_out.values())

    print("╔══════════════════════════════════════════════════════╗")
    print("║   CROSS-GW SUMMARY                                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  GWs processed:                {len(gameweeks_out)}")
    print(f"  Total flagged entries:         {total_flagged}")

    if injury_frequency:
        print()
        print("  Most frequently below 80% availability (across GW1-28):")
        for name, count in injury_frequency.most_common(12):
            print(f"    {name:<24} {count} GW(s)")

    # Aggregate availability distribution
    avail_dist: Counter = Counter()
    for gv in gameweeks_out.values():
        for k, v in gv.get("tier_counts", {}).items():
            avail_dist[k] += v

    print()
    print("  Tier distribution (summed across all GWs):")
    for tier, _ in AVAILABILITY_TIERS:
        label = availability_tier(tier)
        print(f"    {label:<12}: {avail_dist.get(label, 0)}")

    # --- Save output ---
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "backtest",
        "note": (
            "GW1-28 scored using press conference data only. "
            "FPL live fields (fpl_status, fpl_chance, crowd_signal) "
            "reflect the GW29 snapshot and are included for reference only, "
            "not used in scoring."
        ),
        "sources": {
            "fpl_live":          LIVE_PATH,
            "press_conferences": PRESS_PATH,
        },
        "current_gw": current_gw,
        "gameweeks":  gameweeks_out,
        "summary": {
            "total_flagged_across_gws":  total_flagged,
            "most_frequently_flagged":   [n for n, _ in injury_frequency.most_common(10)],
            "availability_distribution": dict(avail_dist),
        },
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"  Saved -> data/intel/availability.json")
    print()


if __name__ == "__main__":
    main()
