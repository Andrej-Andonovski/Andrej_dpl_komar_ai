"""
Intel 04: Player Rotation Risk Rating
Computes a per-player tactical rotation risk score (0-100) for each GW.
Uses per-GW historical minutes from player_history.csv, press conference
rotation signals, and availability data from intel_03.

Reads:  data/intel/fpl_live.json
        data/intel/press_conferences.json
        data/intel/availability.json
        data/raw/fpl_api/player_history.csv
Writes: data/intel/rotation_risk.json
"""

import sys
import os
import csv
import json
import re
import math
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
AVAIL_PATH = os.path.join(INTEL_DIR, "availability.json")
HIST_PATH  = os.path.join(ROOT_DIR, "data", "raw", "fpl_api", "player_history.csv")
OUT_PATH   = os.path.join(INTEL_DIR, "rotation_risk.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# (threshold, tier_label) — first match wins (high to low)
ROTATION_TIERS = [
    (76,  "bench"),
    (56,  "heavy_rotation"),
    (36,  "rotation_risk"),
    (16,  "likely"),
    (0,   "nailed"),
]

# Default baseline score when no prior GW data is available, by position
POSITION_BASELINE = {
    "GK":  30,
    "DEF": 40,
    "MID": 40,
    "FWD": 40,
}

# Internal S1:S2:S3:S4 proportions (sum = 90, matching base weights in spec)
_S_WEIGHTS = [30 / 90, 25 / 90, 20 / 90, 15 / 90]

# Press conference keyword lists (checked in priority order: HIGH > MOD > LOW)
PRESS_HIGH_RISK = [
    "rotate", "rested", "from the bench", "not in the squad",
    "drop", "dropped",
]
PRESS_MOD_RISK = [
    "competition", "assess", "decision to make", "not sure",
    "could start", "fighting fit", "in contention", "doubt",
]
PRESS_LOW_RISK = [
    "will start", "nailed", "first choice", "always plays",
    "guaranteed", "key player",
]

# Players with availability_pct below this threshold skip rotation scoring
AVAIL_THRESHOLD = 60

# Players with rotation_risk >= this OR a rotation press mention are included
# in the output; the rest are counted in low_risk_count
OUTPUT_THRESHOLD = 20

# ---------------------------------------------------------------------------
# Name normalisation (identical to intel_03)
# ---------------------------------------------------------------------------

ALIASES = {
    "Son":         "Son Heung-min",
    "B.Fernandes": "Bruno Fernandes",
    "De Gea":      "David de Gea",
}

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

RELEGATED_CLUBS = {"Ipswich", "Leicester", "Southampton"}


def normalize_name(name: str) -> str:
    nfd = unicodedata.normalize("NFD", name)
    cleaned = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    cleaned = (cleaned
               .replace("\u0141", "l").replace("\u0142", "l")
               .replace("\u0131", "i")
               .replace("\u00df", "ss"))
    return re.sub(r"[^a-z0-9]", "", cleaned.lower())


def last_name(name: str) -> str:
    parts = name.strip().split()
    return parts[-1] if parts else name


# ---------------------------------------------------------------------------
# Club / player lookup (identical pattern to intel_03)
# ---------------------------------------------------------------------------

def build_club_to_short(teams: dict) -> dict:
    fpl_lower = {v["name"].lower(): v["short_name"] for v in teams.values()}
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
        if pc_lower in fpl_lower:
            result[pc] = fpl_lower[pc_lower]
            continue
        hint = PRESS_CLUB_HINTS.get(pc, pc_lower)
        found = next(
            (short for nm, short in fpl_lower.items() if hint in nm or nm in pc_lower),
            None,
        )
        if found:
            result[pc] = found
            continue
        first = pc_lower.split()[0]
        if len(first) >= 4:
            found = next((short for nm, short in fpl_lower.items() if first in nm), None)
            if found:
                result[pc] = found
    return result


def match_player(press_name: str, press_club: str,
                 club_to_short: dict, short_to_players: dict) -> dict | None:
    display = ALIASES.get(press_name, press_name)
    team_short = club_to_short.get(press_club)
    if not team_short:
        return None
    candidates = short_to_players.get(team_short, [])
    if not candidates:
        return None
    norm = normalize_name(display)
    if len(norm) < 3:
        return None
    for p in candidates:
        if normalize_name(p["web_name"]) == norm:
            return p
    for p in candidates:
        if normalize_name(p["full_name"]) == norm:
            return p
    if len(norm) >= 4:
        for p in candidates:
            if norm in normalize_name(p["full_name"]):
                return p
    press_last = normalize_name(last_name(display))
    if len(press_last) >= 4:
        for p in candidates:
            if normalize_name(last_name(p["full_name"])) == press_last:
                return p
        for p in candidates:
            if press_last in normalize_name(p["full_name"]):
                return p
    return None


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def get_weights(prior_gws: int) -> tuple[float, float, float]:
    """Return (minutes_weight, press_weight, baseline_weight) for given data volume."""
    if prior_gws == 0:
        return 0.00, 0.60, 0.40
    if prior_gws == 1:
        return 0.40, 0.40, 0.20
    if prior_gws <= 3:
        return 0.70, 0.20, 0.10
    return 0.90, 0.10, 0.00  # 4+ GWs


# ---------------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------------

def signal_start_rate(minutes_list: list) -> float:
    """S1: proportion of GWs where player played 60+ min (0=always starts, 100=never)."""
    if not minutes_list:
        return 50.0
    starts = sum(1 for m in minutes_list if m >= 60)
    return (1.0 - starts / len(minutes_list)) * 100.0


def signal_volatility(minutes_list: list) -> float:
    """S2: minutes standard deviation, normalised to 0-100 (stdev >= 40 -> 100)."""
    if len(minutes_list) < 2:
        return 30.0  # small-sample neutral
    mean = sum(minutes_list) / len(minutes_list)
    variance = sum((m - mean) ** 2 for m in minutes_list) / len(minutes_list)
    return min(100.0, (math.sqrt(variance) / 40.0) * 100.0)


def signal_bench_rate(minutes_list: list) -> float:
    """S3: proportion of GWs with 0 minutes (0=never benched, 100=always benched)."""
    if not minutes_list:
        return 30.0
    benched = sum(1 for m in minutes_list if m == 0)
    return (benched / len(minutes_list)) * 100.0


def signal_recent_trend(minutes_list: list) -> float:
    """S4: compare last-2 vs earlier GWs. Declining -> higher risk (> 50). (0-100)."""
    if len(minutes_list) < 3:
        return 50.0
    recent = minutes_list[-2:]
    prior  = minutes_list[:-2]
    recent_avg = sum(recent) / len(recent)
    prior_avg  = sum(prior)  / len(prior)
    # Each 30-min swing in either direction counts as ±25 risk points
    adjustment = -(recent_avg - prior_avg) / 30.0 * 25.0
    return max(0.0, min(100.0, 50.0 + adjustment))


def signal_press(quotes: list) -> tuple:
    """S5: scan press quotes for rotation keywords. Returns (score 0-100, phrase|None)."""
    if not quotes:
        return 50.0, None
    text = " ".join(quotes).lower()
    for phrase in PRESS_HIGH_RISK:
        if phrase in text:
            return 85.0, phrase
    for phrase in PRESS_MOD_RISK:
        if phrase in text:
            return 60.0, phrase
    for phrase in PRESS_LOW_RISK:
        if phrase in text:
            return 15.0, phrase
    return 50.0, None


def _recent_trend_label(minutes_list: list) -> str:
    if len(minutes_list) < 3:
        return None
    recent_avg = sum(minutes_list[-2:]) / 2
    prior_avg  = sum(minutes_list[:-2]) / len(minutes_list[:-2])
    diff = recent_avg - prior_avg
    if diff >= 15:
        return "improving"
    if diff <= -15:
        return "declining"
    return "stable"


def rotation_tier(score: float) -> str:
    for threshold, label in ROTATION_TIERS:
        if score >= threshold:
            return label
    return "nailed"


# ---------------------------------------------------------------------------
# Per-player rotation risk computation
# ---------------------------------------------------------------------------

def compute_rotation_risk(player_fpl: dict,
                          prior_minutes: list,
                          quotes: list,
                          availability_pct: int) -> dict:
    """
    Merge all signals into one rotation risk record for a player/GW pair.

    prior_minutes : ordered list of minute values for GWs before this GW.
    quotes        : list of press conference quote strings mentioning this player.
    """
    position = player_fpl.get("position", "MID")

    # Unavailable players: skip rotation scoring
    if availability_pct < AVAIL_THRESHOLD:
        return {
            "rotation_risk":       None,
            "rotation_tier":       "unavailable",
            "start_rate":          None,
            "bench_rate":          None,
            "minutes_avg":         None,
            "minutes_stdev":       None,
            "recent_trend":        None,
            "press_signal":        None,
            "contributing_factors": ["availability too low to assess rotation"],
        }

    n = len(prior_minutes)
    w_min, w_press, w_base = get_weights(n)

    # --- Compute individual signals ---
    s1 = signal_start_rate(prior_minutes)
    s2 = signal_volatility(prior_minutes)
    s3 = signal_bench_rate(prior_minutes)
    s4 = signal_recent_trend(prior_minutes)
    s5, press_phrase = signal_press(quotes)

    # --- Combine ---
    baseline = float(POSITION_BASELINE.get(position, 40))
    min_score = (
        _S_WEIGHTS[0] * s1 +
        _S_WEIGHTS[1] * s2 +
        _S_WEIGHTS[2] * s3 +
        _S_WEIGHTS[3] * s4
    )
    final = w_min * min_score + w_press * s5 + w_base * baseline
    final = max(0.0, min(100.0, final))

    # GW1 special case: no minutes data AND no press mention -> assume likely starter.
    # The neutral math (0.60*50 + 0.40*baseline) gives every player 42-46, which is
    # uninformative noise. Only a real press signal should elevate risk at GW1.
    if n == 0 and press_phrase is None:
        final = 15.0

    risk  = round(final, 1)

    # --- Descriptive stats ---
    avg_min   = round(sum(prior_minutes) / n, 1) if n > 0 else None
    stdev_min = None
    if n >= 2:
        mean = sum(prior_minutes) / n
        stdev_min = round(
            math.sqrt(sum((m - mean) ** 2 for m in prior_minutes) / n), 1
        )
    start_rate_val = round(sum(1 for m in prior_minutes if m >= 60) / n, 2) if n > 0 else None
    bench_rate_val = round(sum(1 for m in prior_minutes if m == 0) / n, 2) if n > 0 else None

    # --- Contributing factors (human-readable) ---
    factors = []
    if n == 0 and press_phrase is None:
        factors.append("no prior data and no press signal -- assumed likely starter")
    elif n == 0:
        factors.append("no prior GW data -- score driven by press conf signal")
    else:
        if start_rate_val is not None and start_rate_val >= 0.9:
            factors.append(f"{int(start_rate_val * 100)}% start rate")
        if bench_rate_val is not None and bench_rate_val >= 0.3:
            factors.append(f"{int(bench_rate_val * 100)}% bench rate")
        if stdev_min is not None and stdev_min >= 25:
            factors.append(f"high minutes volatility (stdev={stdev_min})")
        trend = _recent_trend_label(prior_minutes)
        if trend and trend != "stable":
            factors.append(f"recent trend: {trend}")
    if press_phrase:
        factors.append(f"press conf: '{press_phrase}'")
    if not factors:
        if risk <= 15:
            factors.append("consistent minutes, no concerns")
        else:
            factors.append("moderate rotation pattern")

    return {
        "rotation_risk":       risk,
        "rotation_tier":       rotation_tier(risk),
        "start_rate":          start_rate_val,
        "bench_rate":          bench_rate_val,
        "minutes_avg":         avg_min,
        "minutes_stdev":       stdev_min,
        "recent_trend":        _recent_trend_label(prior_minutes),
        "press_signal":        press_phrase,
        "contributing_factors": factors,
    }


# ---------------------------------------------------------------------------
# Per-GW processing
# ---------------------------------------------------------------------------

def process_gw(gw: int,
               gw_press: dict,
               fpl_players: dict,
               club_to_short: dict,
               short_to_players: dict,
               availability_players: dict,
               history_by_pid: dict) -> dict:
    """
    Score all players for one GW.

    availability_players : pid_str -> availability record (from intel_03).
                           Players absent from this dict are assumed 95% available.
    history_by_pid       : pid_int -> sorted list of (gw_num, minutes).
    """
    all_news = gw_press.get("all_player_news", [])

    # Build press quotes lookup: pid_str -> list[str]
    press_quotes: dict[str, list] = defaultdict(list)
    press_matched   = 0
    press_unmatched = 0

    for entry in all_news:
        press_name = entry.get("player", "")
        press_club = entry.get("club", "")
        news_text  = (entry.get("news") or "").strip()
        if not news_text:
            continue
        p_match = match_player(press_name, press_club, club_to_short, short_to_players)
        if p_match:
            press_quotes[str(p_match["player_id"])].append(news_text)
            press_matched += 1
        else:
            press_unmatched += 1

    prior_gw_list  = list(range(1, gw))
    flagged        = {}
    low_risk_count = 0
    unavail_count  = 0
    fringe_count   = 0
    tier_counts    = Counter()

    for pid_str, player in fpl_players.items():
        pid_int = int(pid_str)

        # Availability for this GW (players absent from dict are 95%+ available)
        avail_entry      = availability_players.get(pid_str, {})
        availability_pct = avail_entry.get("availability_pct", 95)

        # Prior minutes (strictly before this GW, in chronological order)
        hist = history_by_pid.get(pid_int, [])
        prior_minutes = [m for gw_num, m in hist if gw_num < gw]

        quotes = press_quotes.get(pid_str, [])

        # Fringe player filter: never played, zero FPL points, zero form, no press mention.
        # These are reserves / youth players who were never rotation candidates.
        # Skip them entirely — they are not counted in tier_counts or flagged.
        all_zero = all(m == 0 for m in prior_minutes) if prior_minutes else True
        if (all_zero
                and player.get("total_points", 0) == 0
                and player.get("form", 0.0) == 0.0
                and not quotes):
            fringe_count += 1
            continue

        result = compute_rotation_risk(
            player_fpl       = player,
            prior_minutes    = prior_minutes,
            quotes           = quotes,
            availability_pct = availability_pct,
        )

        tier = result["rotation_tier"]
        tier_counts[tier] += 1

        if tier == "unavailable":
            unavail_count += 1
            continue

        risk = result["rotation_risk"]
        has_rotation_press = any(ph in " ".join(quotes).lower()
                                 for ph in PRESS_HIGH_RISK + PRESS_MOD_RISK) if quotes else False

        if risk is not None and (risk >= OUTPUT_THRESHOLD or has_rotation_press):
            flagged[pid_str] = {
                "player_id":       pid_int,
                "web_name":        player["web_name"],
                "full_name":       player["full_name"],
                "team_short":      player["team_short"],
                "position":        player["position"],
                "availability_pct": availability_pct,
                **result,
            }
        else:
            low_risk_count += 1

    return {
        "gw":                  gw,
        "prior_gws_used":      prior_gw_list,
        "players_assessed":    len(fpl_players),
        "unavailable_skipped": unavail_count,
        "fringe_skipped":      fringe_count,
        "low_risk_count":      low_risk_count,
        "flagged_count":       len(flagged),
        "press_matched":       press_matched,
        "press_unmatched":     press_unmatched,
        "tier_counts":         dict(tier_counts),
        "players":             flagged,
    }


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_gw_report(gw: int, gw_result: dict) -> None:
    tc = gw_result["tier_counts"]
    print(f"  Assessed : {gw_result['players_assessed']}  |  "
          f"Fringe skipped: {gw_result['fringe_skipped']}  |  "
          f"Unavailable: {gw_result['unavailable_skipped']}  |  "
          f"Flagged: {gw_result['flagged_count']}  |  "
          f"Low risk: {gw_result['low_risk_count']}")
    print(f"  Prior GWs: {len(gw_result['prior_gws_used'])}  |  "
          f"Press matched: {gw_result['press_matched']}  |  "
          f"Press unmatched: {gw_result['press_unmatched']}")
    print(f"  Tiers    : "
          f"nailed={tc.get('nailed', 0)}  "
          f"likely={tc.get('likely', 0)}  "
          f"rotation_risk={tc.get('rotation_risk', 0)}  "
          f"heavy_rotation={tc.get('heavy_rotation', 0)}  "
          f"bench={tc.get('bench', 0)}")

    # Top 10 highest rotation risk
    by_risk = sorted(
        [p for p in gw_result["players"].values() if p.get("rotation_risk") is not None],
        key=lambda p: p["rotation_risk"],
        reverse=True,
    )
    if by_risk:
        print()
        print("  Top 10 highest rotation risk:")
        for p in by_risk[:10]:
            risk  = p["rotation_risk"]
            tier  = p["rotation_tier"]
            wn    = p["web_name"]
            ts    = p["team_short"]
            pos   = p["position"]
            sr    = p.get("start_rate")
            sr_s  = f"SR={sr:.0%}" if sr is not None else "SR=n/a"
            trend = p.get("recent_trend") or "--"
            press = p.get("press_signal") or ""
            line  = f"    [{risk:5.1f}] {wn:<20} ({ts:<3} {pos}) {tier:<16} {sr_s}  trend={trend}"
            if press:
                line += f"  press='{press}'"
            print(line)

    # Press rotation signals
    press_flagged = [p for p in gw_result["players"].values() if p.get("press_signal")]
    if press_flagged:
        print()
        print(f"  Press rotation signals ({len(press_flagged)}):")
        for p in press_flagged[:8]:
            print(f"    {p['web_name']:<20} ({p['team_short']}) "
                  f"risk={p['rotation_risk']}  phrase='{p['press_signal']}'")
        if len(press_flagged) > 8:
            print(f"    ... +{len(press_flagged) - 8} more")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(gameweeks_out: dict) -> dict:
    # Key by (pid, web_name, team_short) to avoid web_name collisions
    risk_by_pid:   dict[int, list]  = defaultdict(list)
    pid_meta:      dict[int, tuple] = {}   # pid -> (web_name, team_short)
    risk_by_team:  dict[str, list]  = defaultdict(list)
    rotation_freq: Counter          = Counter()  # pid -> count of GWs at risk>=36

    for gv in gameweeks_out.values():
        for pid_str, p in gv["players"].items():
            risk = p.get("rotation_risk")
            if risk is None:
                continue
            pid = int(pid_str)
            wn  = p["web_name"]
            ts  = p["team_short"]
            risk_by_pid[pid].append(risk)
            pid_meta[pid] = (wn, ts)
            risk_by_team[ts].append(risk)
            if risk >= 36:
                rotation_freq[pid] += 1

    avg_by_pid  = {pid: sum(v) / len(v) for pid, v in risk_by_pid.items()}
    avg_by_team = {t: round(sum(v) / len(v), 1) for t, v in risk_by_team.items()}

    sorted_pids_desc = sorted(avg_by_pid, key=avg_by_pid.get, reverse=True)
    sorted_pids_asc  = sorted(avg_by_pid, key=avg_by_pid.get)

    most_rotated = [
        f"{pid_meta[pid][0]} ({pid_meta[pid][1]})"
        for pid in sorted_pids_desc[:10]
    ]
    most_nailed = [
        f"{pid_meta[pid][0]} ({pid_meta[pid][1]})"
        for pid in sorted_pids_asc[:10]
    ]
    chronic_rotators = [
        f"{pid_meta[pid][0]} ({pid_meta[pid][1]})"
        for pid, _ in rotation_freq.most_common(10)
    ]

    return {
        "most_rotated_players": most_rotated,
        "most_nailed_players":  most_nailed,
        "rotation_by_team":     dict(sorted(avg_by_team.items(), key=lambda x: x[1], reverse=True)),
        "chronic_rotators":     chronic_rotators,
        # Internal data for console report
        "_pid_avg":   {pid: round(avg_by_pid[pid], 1) for pid in sorted_pids_desc[:15]},
        "_pid_meta":  {pid: pid_meta[pid] for pid in
                       set(sorted_pids_desc[:15]) | set(sorted_pids_asc[:8])
                       | set(pid for pid, _ in rotation_freq.most_common(10))},
        "_rot_freq":  {pid: cnt for pid, cnt in rotation_freq.most_common(10)},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(INTEL_DIR, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   FPL ROTATION RISK INTEL -- Intel 04               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Load inputs ──────────────────────────────────────────────────────────
    print("Loading fpl_live.json...", end=" ", flush=True)
    with open(LIVE_PATH, "r", encoding="utf-8") as f:
        live_data = json.load(f)
    fpl_players = live_data["players"]   # pid_str -> player dict
    teams_data  = live_data["teams"]     # tid_str -> team dict
    current_gw  = live_data["current_gw"]
    print(f"{len(fpl_players)} players  |  current GW: {current_gw}")

    print("Loading press_conferences.json...", end=" ", flush=True)
    with open(PRESS_PATH, "r", encoding="utf-8") as f:
        press_data = json.load(f)
    gws_scraped = press_data.get("gws_scraped", [])
    print(f"GWs scraped: {gws_scraped}")

    print("Loading availability.json...", end=" ", flush=True)
    with open(AVAIL_PATH, "r", encoding="utf-8") as f:
        avail_data = json.load(f)
    avail_gws = avail_data.get("gameweeks", {})
    print(f"GWs: {sorted(avail_gws.keys(), key=int)}")

    print("Loading player_history.csv...", end=" ", flush=True)
    history_by_pid: dict[int, list] = defaultdict(list)
    row_count = 0
    with open(HIST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid  = int(row["player_id"])
            gw_n = int(row["gameweek"])
            mins = int(row["minutes"])
            history_by_pid[pid].append((gw_n, mins))
            row_count += 1
    for pid in history_by_pid:
        history_by_pid[pid].sort(key=lambda x: x[0])
    print(f"{row_count} rows, {len(history_by_pid)} players")
    print()

    # ── Build lookup structures ───────────────────────────────────────────────
    for pid_str, p in fpl_players.items():
        p["player_id"] = int(pid_str)

    club_to_short: dict[str, str] = build_club_to_short(teams_data)

    short_to_players: dict[str, list] = defaultdict(list)
    for p in fpl_players.values():
        ts = p.get("team_short", "")
        if ts:
            short_to_players[ts].append(p)

    # Warn about unexpected club mapping failures
    all_press_clubs = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
        "Leeds", "Leicester", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle", "Nottingham Forest", "Southampton", "Sunderland",
        "Tottenham", "West Ham", "Wolves",
    ]
    missing = [c for c in all_press_clubs if c not in club_to_short]
    truly_missing = [c for c in missing if c not in RELEGATED_CLUBS]
    if truly_missing:
        print(f"  [WARN] Club mapping failures: {truly_missing}")

    # ── Process each GW ───────────────────────────────────────────────────────
    press_gws       = press_data.get("gameweeks", {})
    gameweeks_out   = {}

    for gw_str in sorted(press_gws.keys(), key=int):
        gw       = int(gw_str)
        gw_press = press_gws[gw_str]

        print(f"-- GW{gw} {'-' * 47}")

        if gw_press.get("status") != "success":
            print(f"  [SKIP] status = {gw_press.get('status')}")
            print()
            continue

        avail_players = avail_gws.get(gw_str, {}).get("players", {})

        gw_result = process_gw(
            gw                  = gw,
            gw_press            = gw_press,
            fpl_players         = fpl_players,
            club_to_short       = club_to_short,
            short_to_players    = short_to_players,
            availability_players = avail_players,
            history_by_pid      = history_by_pid,
        )
        gameweeks_out[gw_str] = gw_result

        print_gw_report(gw, gw_result)
        print()

    # ── Cross-GW summary ─────────────────────────────────────────────────────
    summary = build_summary(gameweeks_out)

    print("╔══════════════════════════════════════════════════════╗")
    print("║   CROSS-GW SUMMARY                                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  GWs processed: {len(gameweeks_out)}")

    print()
    print("  Most rotated players (avg risk across all flagged GWs):")
    for label in summary["most_rotated_players"][:8]:
        print(f"    {label}")

    print()
    print("  Most nailed players (avg risk of flagged entries, ascending):")
    for label in summary["most_nailed_players"][:8]:
        print(f"    {label}")

    print()
    print("  Rotation by team (avg risk of flagged players, descending):")
    for team, avg in list(summary["rotation_by_team"].items())[:10]:
        print(f"    {team:<6} {avg}")

    print()
    print("  Chronic rotators (rotation_risk/bench tier in most GWs):")
    for label, count in zip(summary["chronic_rotators"],
                            summary["_rot_freq"].values()):
        print(f"    {label:<32} {count} GW(s)")

    # ── Save output ───────────────────────────────────────────────────────────
    # Strip internal console-only keys before saving
    clean_summary = {k: v for k, v in summary.items() if not k.startswith("_")}

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode":         "backtest",
        "note": (
            "GW1-28 rotation risk scored using minutes history (player_history.csv), "
            "press conference quotes, and availability from intel_03. "
            "Players with availability_pct < 60 are skipped (rotation_tier='unavailable'). "
            "Players with rotation_risk < 20 and no rotation press mention are omitted "
            "from per-GW player lists to keep output manageable."
        ),
        "sources": {
            "fpl_live":          LIVE_PATH,
            "press_conferences": PRESS_PATH,
            "availability":      AVAIL_PATH,
            "player_history":    HIST_PATH,
        },
        "current_gw":  current_gw,
        "gameweeks":   gameweeks_out,
        "summary":     clean_summary,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("  Saved -> data/intel/rotation_risk.json")
    print()


if __name__ == "__main__":
    main()
