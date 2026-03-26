"""
Stage 9: LLM Agent Layer
FPL AI Thesis -- Claude API integration wrapping Stage 8 ILP optimizer

Two-call LLM pipeline per gameweek:
  Call 1 (Pre-ILP):  flag injury/rotation risks, apply prediction penalties
  Call 2 (Post-ILP): override captain, optimise bench order, produce report

Total API calls: 2 per GW x 10 GWs = 20 calls maximum.
Falls back to Stage 8 decisions silently on any API failure.
"""

import os
import sys
import json
import time
import re
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for player names)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW  = os.path.join(ROOT, "data", "raw", "fpl_api")
DATA_PROC = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")

# ---------------------------------------------------------------------------
# Import Stage 8 (module-level constants and functions, no main() executed)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import ilp_optimizer_stage8 as stage8

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID   = "claude-sonnet-4-20250514"
MAX_TOKENS = 2000
TEMP       = 0   # deterministic

# Stage 8 baseline (from models/stage8_backtest_patched.json)
STAGE8_GW_ACTUALS = [39, 51, 56, 86, 57, 47, 81, 61, 30, 71]
STAGE8_CAPTAINS   = ["Isak", "Haaland", "Haaland", "Haaland", "Haaland",
                     "Haaland", "Haaland", "Haaland", "Haaland", "Haaland"]

CALL1_POOL_SIZE = 120   # top-N players sent to Call 1 (by predicted pts)
AVG_MANAGER_PTS = 57.0  # approximate 2025-26 season avg pts/GW

SEP = "=" * 54


# ===========================================================================
# Anthropic client
# ===========================================================================
def get_client():
    """Return Anthropic() client using ANTHROPIC_API_KEY env var."""
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        raise ImportError("anthropic not installed. Run: pip install anthropic")


# ===========================================================================
# Parse <decisions> block
# ===========================================================================
def parse_decisions(text, fallback=None):
    """
    Extract JSON from <decisions>...</decisions>.
    Returns parsed dict or fallback on any failure.
    """
    try:
        m = re.search(r"<decisions>(.*?)</decisions>", text, re.DOTALL)
        if not m:
            print("  [WARN] No <decisions> block found")
            return fallback
        return json.loads(m.group(1).strip())
    except Exception as e:
        print(f"  [WARN] JSON parse failed: {e}")
        return fallback


# ===========================================================================
# FPL Bootstrap injury data (live, no scraping)
# ===========================================================================
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

def fetch_fpl_injury_news(timeout=10):
    """
    Fetch live player injury/availability data from the FPL bootstrap API.
    Returns dict keyed by player_id:
      { player_id: {"status": str, "chance": int|None, "news": str, "news_added": str} }
    Falls back to empty dict on any failure.
    """
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(BOOTSTRAP_URL, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        injury_news = {}
        for p in data.get("elements", []):
            pid    = int(p["id"])
            status = str(p.get("status", "a"))
            chance = p.get("chance_of_playing_next_round", None)
            news   = str(p.get("news", "") or "").strip()
            added  = str(p.get("news_added", "") or "").strip()
            injury_news[pid] = {
                "status":     status,
                "chance":     int(chance) if chance is not None else None,
                "news":       news,
                "news_added": added,
            }

        # Print summary
        with_news    = sum(1 for v in injury_news.values() if v["news"])
        unavailable  = sum(1 for v in injury_news.values() if v["status"] in ("i","d","u","s"))
        print(f"  Live FPL injury data loaded: {with_news} players with news")
        print(f"  Doubtful/injured players: {unavailable}")

        # Top 10 with news
        news_players = [
            (pid, v) for pid, v in injury_news.items() if v["news"]
        ]
        # Sort by status severity then alphabetically
        status_order = {"i": 0, "s": 1, "u": 2, "d": 3, "a": 4}
        news_players.sort(key=lambda x: status_order.get(x[1]["status"], 5))
        print("  Top injury news:")
        shown = 0
        for pid, v in news_players[:10]:
            print(f"    [{v['status'].upper()}] id={pid}: {v['news'][:80]}")
            shown += 1

        return injury_news

    except Exception as e:
        print(f"  [WARN] Bootstrap API failed ({e}) -- no live injury data")
        return {}


def fallback_google_search(player_name, timeout=5):
    """
    Per-player Google fallback (used only if bootstrap API fails).
    Returns first 300 chars of stripped text, or None.
    """
    try:
        import requests
        url    = "https://www.google.com/search"
        params = {"q": f"{player_name} injury news site:bbc.co.uk"}
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            text = re.sub(r"<[^>]+>", " ", resp.text)
            return re.sub(r"\s+", " ", text).strip()[:300]
    except Exception:
        pass
    return None


# ===========================================================================
# Build context package
# ===========================================================================
def build_context_package(player_pool, pred_gw0, upcoming_df, players_raw,
                           real_gw, top_n=CALL1_POOL_SIZE, injury_news=None):
    """
    Build per-player context dict for the top-N players by predicted pts.
    """
    # Status / chance-of-playing from players_raw
    status_map = {}
    cop_map    = {}
    if players_raw is not None and len(players_raw) > 0:
        for _, row in players_raw.iterrows():
            pid = int(row.get("id", 0))
            status_map[pid] = str(row.get("status", "a"))
            cop = row.get("chance_of_playing_next_round", None)
            cop_map[pid] = None if pd.isna(cop) else int(cop)

    # Sort by predicted pts descending
    sorted_pool = sorted(
        player_pool,
        key=lambda p: pred_gw0.get(p["player_id"], 0),
        reverse=True,
    )[:top_n]

    up = upcoming_df.copy()
    up["gameweek"] = up["gameweek"].astype(int)
    up_cols = set(up.columns)

    context = []
    for p in sorted_pool:
        pid = p["player_id"]
        fv  = p["feat_vec"]

        # Upcoming fixtures (next 3 GWs)
        fixtures_info = []
        for offset in range(3):
            tgw = real_gw + offset
            rows = up[(up["player_id"] == pid) & (up["gameweek"] == tgw)]
            if len(rows) > 0:
                r = rows.iloc[0]
                opp = "?"
                for col in ("opponent_short_name", "opponent_team_short",
                            "opponent", "team_short"):
                    if col in up_cols and pd.notna(r.get(col)):
                        opp = str(r[col])
                        break
                fixtures_info.append({
                    "gw":       tgw,
                    "fdr":      int(r.get("difficulty", 3)),
                    "home_away": "H" if bool(r.get("is_home", True)) else "A",
                    "opponent": opp,
                })

        live = (injury_news or {}).get(pid, {})
        # Prefer live bootstrap data over players_raw where available
        live_status = live.get("status") or status_map.get(pid, "a")
        live_chance = live.get("chance") if live.get("chance") is not None else cop_map.get(pid, None)
        live_news   = live.get("news", "")

        context.append({
            "name":             p["web_name"],
            "team":             p["team_short"],
            "position":         p["pos_name"],
            "price":            round(float(p["value"]), 1),
            "xgb_predicted_pts": round(float(pred_gw0.get(pid, 0)), 1),
            "status":           live_status,
            "chance_of_playing": live_chance,
            "news":             live_news,
            "form_last3":       round(float(fv.get("form_last3", 0)), 1),
            "current_gw_fdr":   int(fv.get("current_gw_fdr", 3)),
            "upcoming_fixtures": fixtures_info,
        })

    return context


# ===========================================================================
# LLM Call 1 -- Pre-ILP risk filter
# ===========================================================================
SYS_CALL1 = (
    "You are an expert FPL analyst working on the 2025/26 Premier League season. "
    "Your job is to identify players who should be excluded from squad selection "
    "this gameweek due to injury, suspension, or rotation risk. "
    "Be conservative -- only flag players with clear evidence "
    "of risk. Return your analysis as prose followed by a "
    "JSON block wrapped in <decisions> tags."
)


def llm_call1_risk_filter(client, gw, real_gw, context_package, current_squad_names=None):
    """
    Call 1: identify risky players and optional web-search queries.
    Returns (raw_text, decisions_dict).
    Falls back to empty flags on any API error.
    """
    squad_line = ""
    if current_squad_names:
        squad_line = "Current squad: " + ", ".join(current_squad_names) + "\n\n"

    # Compact player table
    rows = [
        f"{'Name':<20} {'Pos':<4} {'Team':<5} {'GBP':<5} "
        f"{'Pred':<6} {'St':<3} {'CoP':<5} {'F3':<5} "
        f"{'FDR':<4} {'Fixtures':<20} News"
    ]
    rows.append("-" * 110)
    for p in context_package:
        cop_str = f"{p['chance_of_playing']}%" if p["chance_of_playing"] is not None else "OK"
        fix_str = " ".join(
            f"GW{f['gw']}:{f['fdr']}:{f['home_away']}"
            for f in p["upcoming_fixtures"]
        )
        news_str = p.get("news", "")[:60]
        rows.append(
            f"{p['name']:<20} {p['position']:<4} {p['team']:<5} "
            f"{p['price']:<5} {p['xgb_predicted_pts']:<6} "
            f"{p['status']:<3} {cop_str:<5} {p['form_last3']:<5} "
            f"{p['current_gw_fdr']:<4} {fix_str:<20} {news_str}"
        )

    user_msg = (
        f"Season: 2025/26 Premier League\n"
        f"Gameweek: {real_gw} (simulation step {gw})\n\n"
        + squad_line
        + "Player pool (top players by XGBoost prediction):\n"
        + "\n".join(rows)
        + "\n\nBase your analysis ONLY on the status and chance_of_playing data provided above. "
        "Do NOT suggest web searches -- all injury/availability data is already included.\n"
        "Status codes: a=available, d=doubtful, i=injured, s=suspended, u=unavailable.\n"
        "chance_of_playing: percentage chance of playing (blank=100%).\n"
        "Flag players with status d/i/s/u, or chance_of_playing <= 75, or very low form.\n\n"
        "Return prose analysis then:\n"
        "<decisions>\n"
        "{\n"
        '  "flagged_players": [\n'
        '    {"name": "Player Name", "risk_type": "injury|rotation|suspension|form",\n'
        '     "confidence": "high|medium|low", "reason": "one sentence"}\n'
        "  ]\n"
        "}\n"
        "</decisions>"
    )

    fallback = {"flagged_players": []}

    try:
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            temperature=TEMP,
            system=SYS_CALL1,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        print("  LLM Call 1 complete")
        return raw, parse_decisions(raw, fallback=fallback)
    except Exception as e:
        print(f"  LLM Call 1 FAILED ({e}) -- using Stage 8 fallback")
        return "", fallback


# ===========================================================================
# Apply risk penalties to predictions
# ===========================================================================
def apply_risk_penalties(player_pool, pred_gw0, horizon, flagged_players):
    """
    Return (penalized_pred, penalized_horizon) with risk discounts applied.
      high confidence -> 50% penalty
      medium          -> 25% penalty
      low             -> no penalty (warning only)
    Name matching: web_name substring check (case-insensitive).
    """
    # Build penalty dict: lowercased name fragment -> penalty factor
    penalty_map = {}
    for fp in flagged_players:
        conf = fp.get("confidence", "low").lower()
        name = fp.get("name", "").lower().strip()
        if not name:
            continue
        if conf == "high":
            penalty_map[name] = 0.50
        elif conf == "medium":
            penalty_map[name] = 0.25
        # low = 0 penalty

    new_pred    = dict(pred_gw0)
    new_horizon = dict(horizon)

    for p in player_pool:
        pid      = p["player_id"]
        web_low  = p["web_name"].lower()
        full_low = p["name"].lower()

        best_pen = 0.0
        for name_key, pen in penalty_map.items():
            # match if either name contains the key or vice versa
            if (name_key in web_low or name_key in full_low
                    or web_low in name_key or full_low in name_key):
                best_pen = max(best_pen, pen)

        if best_pen > 0:
            new_pred[pid]    = pred_gw0.get(pid, 0)    * (1.0 - best_pen)
            new_horizon[pid] = horizon.get(pid, 0)      * (1.0 - best_pen)

    return new_pred, new_horizon


# ===========================================================================
# LLM Call 2 -- Post-ILP captain + bench + report
# ===========================================================================
SYS_CALL2 = (
    "You are an expert FPL manager producing a gameweek "
    "analysis report. You have received an optimised squad "
    "from a mathematical ILP optimizer. Your job is to: "
    "1. Review and potentially override the captain pick. "
    "2. Optimise bench order for fixture-based rotation. "
    "3. Flag any last-minute concerns. "
    "4. Produce a full FPL manager report. "
    "IMPORTANT -- captain override rule: The ILP captain already "
    "accounts for fixture difficulty via horizon scoring (shown as "
    "'adj' in the XI table). Only override the ILP captain if "
    "(a) an alternative player's adj score is at least 0.5 pts "
    "higher, OR (b) you have strong qualitative reason such as "
    "confirmed rotation risk, active injury doubt flagged in the "
    "news field, or a known favourable double gameweek. "
    "Do NOT override for marginal differences or vague upside. "
    "When in doubt, keep the ILP captain. "
    "Return your decisions as prose analysis with a JSON "
    "block wrapped in <decisions> tags at the end."
)


def llm_call2_report(client, gw, result, pred_gw0, flagged_players,
                     chips_used, total_pts_so_far, search_snippets=None,
                     penalized_horizon=None):
    """
    Call 2: captain override, bench reorder, full report.
    Returns (raw_text, decisions_dict).
    penalized_horizon: FDR-adjusted horizon scores (5-GW outlook) for XI players.
    """
    squad   = result["squad"]
    xi      = result["xi"]
    captain = result["captain"]
    vice    = result["vice"]

    bench_gk_list, bench_out_list = stage8.split_bench(squad, xi, pred_gw0)
    xi_ids = {p["player_id"] for p in xi}

    # Format XI table — include adj (FDR-adjusted horizon) alongside raw pred
    xi_lines = []
    for p in sorted(xi, key=lambda x: x["pos_id"]):
        pid   = p["player_id"]
        pred  = round(pred_gw0.get(pid, 0), 1)
        adj   = round(penalized_horizon.get(pid, pred), 1) if penalized_horizon else pred
        fv    = p["feat_vec"]
        f3    = round(float(fv.get("form_last3", 0)), 1)
        ha    = "H" if fv.get("home_advantage", 0) == 1 else "A"
        fdr   = int(fv.get("current_gw_fdr", 3))
        mark  = ""
        if captain and pid == captain["player_id"]:
            mark = " [C]"
        elif vice and pid == vice["player_id"]:
            mark = " [V]"
        xi_lines.append(
            f"  {p['pos_name']:<4} {p['web_name']:<20} {p['team_short']:<5} "
            f"GBP{p['value']:.1f}  pred:{pred}  adj:{adj}  form3:{f3}  {ha} FDR:{fdr}{mark}"
        )

    # Format bench
    bench_lines = []
    for p in bench_gk_list:
        bench_lines.append(
            f"  GK:  {p['web_name']:<20} {p['team_short']:<5} "
            f"pred:{round(pred_gw0.get(p['player_id'],0),1)}"
        )
    labels = ["1st", "2nd", "3rd"]
    for i, p in enumerate(bench_out_list[:3]):
        bench_lines.append(
            f"  {labels[i]}: {p['web_name']:<20} {p['team_short']:<5} "
            f"pred:{round(pred_gw0.get(p['player_id'],0),1)}"
        )

    # Captain alternatives (top-5 MID/FWD in XI) — ranked by adj score if available
    alt_caps = sorted(
        [p for p in xi if p["pos_id"] in (3, 4)],
        key=lambda p: (
            penalized_horizon.get(p["player_id"], pred_gw0.get(p["player_id"], 0))
            if penalized_horizon else pred_gw0.get(p["player_id"], 0)
        ),
        reverse=True,
    )[:5]
    alts_str = ", ".join(
        f"{p['web_name']} (pred:{round(pred_gw0.get(p['player_id'],0),1)}"
        f" adj:{round((penalized_horizon or pred_gw0).get(p['player_id'],0),1)})"
        for p in alt_caps
    )

    # Flagged players
    flag_block = ""
    if flagged_players:
        flag_block = "Flagged players from Call 1:\n" + "\n".join(
            f"  - {fp.get('name','?')}: {fp.get('risk_type','?')} "
            f"({fp.get('confidence','?')}) -- {fp.get('reason','')}"
            for fp in flagged_players
        ) + "\n"

    # Chips remaining
    all_chips = {"wildcard_1", "wildcard_2", "freehit", "bench_boost", "triple_captain"}
    remaining_chips = ", ".join(sorted(all_chips - chips_used)) or "None"

    # Web search snippets
    search_block = ""
    if search_snippets:
        search_block = "\nAdditional context (web search):\n"
        for q, snip in search_snippets.items():
            if snip:
                search_block += f"  [{q}]: {snip[:300]}\n"

    # Default fallback decisions
    cap_name = captain["web_name"] if captain else "None"
    vc_name  = vice["web_name"]    if vice    else "None"
    bench_defaults = (
        [bench_gk_list[0]["web_name"] if bench_gk_list else "None"]
        + [p["web_name"] for p in bench_out_list[:3]]
    )
    while len(bench_defaults) < 4:
        bench_defaults.append("None")

    fallback = {
        "captain": {
            "name": cap_name, "override": False,
            "reasoning": "ILP selection maintained (API fallback).",
        },
        "vice_captain": {
            "name": vc_name,
            "reasoning": "ILP selection maintained.",
        },
        "bench_order": {
            "bench_gk": bench_defaults[0],
            "bench_1":  bench_defaults[1],
            "bench_2":  bench_defaults[2],
            "bench_3":  bench_defaults[3],
            "reasoning": "Default order by predicted points.",
        },
        "transfer_suggestions": [],
        "chip_advice": {
            "play_chip": "none",
            "reasoning": "No chip recommended.",
        },
        "risk_flags": [],
    }

    cap_pred = round(pred_gw0.get(captain["player_id"], 0), 1) if captain else 0
    cap_adj  = round(
        (penalized_horizon or pred_gw0).get(captain["player_id"] if captain else -1, cap_pred), 1
    )
    user_msg = (
        f"Gameweek: {gw}\n"
        f"Season points so far (Stage 9): {total_pts_so_far:.0f}\n"
        f"Average manager benchmark: {AVG_MANAGER_PTS} pts/GW\n\n"
        f"ILP captain suggestion: {cap_name} "
        f"(pred:{cap_pred}  adj:{cap_adj})\n"
        f"  adj = FDR-adjusted 5-GW horizon score. Only override if alternative adj >= {cap_adj + 0.5:.1f}.\n"
        f"Captain alternatives (MID/FWD by adj): {alts_str}\n\n"
        "STARTING XI (pred=GW prediction, adj=FDR-adjusted horizon):\n"
        + "\n".join(xi_lines) + "\n\n"
        "BENCH:\n" + "\n".join(bench_lines) + "\n\n"
        + flag_block
        + f"Chips remaining: {remaining_chips}\n"
        + search_block
        + "\nProduce the full FPL manager report with your reasoning.\n\n"
        "<decisions>\n"
        "{\n"
        '  "captain": {"name": "Player Name", "override": false, "reasoning": "2-3 sentences"},\n'
        '  "vice_captain": {"name": "Player Name", "reasoning": "one sentence"},\n'
        '  "bench_order": {\n'
        '    "bench_gk": "Name",\n'
        '    "bench_1": "Name", "bench_2": "Name", "bench_3": "Name",\n'
        '    "reasoning": "one sentence"\n'
        "  },\n"
        '  "transfer_suggestions": [\n'
        '    {"out": "Name", "in": "Name", "priority": "urgent|recommended|optional", '
        '"reasoning": "one sentence"}\n'
        "  ],\n"
        '  "chip_advice": {"play_chip": "none|wildcard|freehit|bench_boost|triple_captain", '
        '"reasoning": "one sentence"},\n'
        '  "risk_flags": ["warning1", "warning2"]\n'
        "}\n"
        "</decisions>"
    )

    try:
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            temperature=TEMP,
            system=SYS_CALL2,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        print("  LLM Call 2 complete")
        return raw, parse_decisions(raw, fallback=fallback)
    except Exception as e:
        print(f"  LLM Call 2 FAILED ({e}) -- using Stage 8 fallback")
        return "", fallback


# ===========================================================================
# Print full GW report (ASCII only -- Windows console safe)
# ===========================================================================
def _wrap(text, width=72, indent="  "):
    """Simple word-wrap helper."""
    words = text.split()
    lines = []
    line  = ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            if line:
                lines.append(indent + line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(indent + line)
    return "\n".join(lines)


def print_gw_report(gw, result, pred_gw0, decisions_call1,
                    decisions_call2, prose_call2, gw_pred):
    """Print the full FPL manager report for a gameweek."""
    xi      = result["xi"]
    squad   = result["squad"]
    captain = result["captain"]
    vice    = result["vice"]

    d_cap  = decisions_call2.get("captain",     {})
    d_vc   = decisions_call2.get("vice_captain",{})
    d_bench = decisions_call2.get("bench_order", {})
    d_xfer  = decisions_call2.get("transfer_suggestions", [])
    d_chip  = decisions_call2.get("chip_advice", {})
    d_flags = decisions_call2.get("risk_flags",  [])

    cap_name = d_cap.get("name", captain["web_name"] if captain else "?")
    vc_name  = d_vc.get("name",  vice["web_name"]    if vice    else "?")

    bench_gk_list, bench_out_list = stage8.split_bench(squad, xi, pred_gw0)

    print()
    print("+" + "=" * 52 + "+")
    print(f"|   FPL AI GAMEWEEK {gw:<2} -- MANAGER REPORT           |")
    print("+" + "=" * 52 + "+")

    # --- Prose analysis ---
    prose = prose_call2 or ""
    decisions_idx = prose.find("<decisions>")
    if decisions_idx > 0:
        prose = prose[:decisions_idx].strip()
    if prose:
        print()
        print("SQUAD ANALYSIS")
        # Limit to ~5 sentences
        sentences = re.split(r"(?<=[.!?])\s+", prose)
        short_prose = " ".join(sentences[:5])
        print(_wrap(short_prose))

    # --- Starting XI ---
    print()
    print("STARTING XI")
    print(f"  {'Pos':<4} {'Player':<20} {'Club':<5} {'GBP':<5} "
          f"{'Pred':<6} {'Form3':<6} Fix")
    print("  " + "-" * 62)

    for pos_id in [1, 2, 3, 4]:
        label = stage8.POSITIONS[pos_id]
        for p in [pp for pp in xi if pp["pos_id"] == pos_id]:
            pid  = p["player_id"]
            pred = round(pred_gw0.get(pid, 0), 1)
            fv   = p["feat_vec"]
            f3   = round(float(fv.get("form_last3", 0)), 1)
            ha   = "H" if fv.get("home_advantage", 0) == 1 else "A"
            fdr  = int(fv.get("current_gw_fdr", 3))
            mark = ""
            if p["web_name"] == cap_name or p["name"] == cap_name:
                mark = " [C]"
            elif p["web_name"] == vc_name or p["name"] == vc_name:
                mark = " [V]"
            print(
                f"  {label:<4} {p['web_name']:<20} {p['team_short']:<5} "
                f"{p['value']:<5.1f} {pred:<6.1f} {f3:<6.1f} "
                f"{ha} FDR:{fdr}{mark}"
            )

    print(f"  Captain: {cap_name}")
    cap_reason = d_cap.get("reasoning", "")
    if cap_reason:
        print(_wrap(cap_reason[:120], indent="    "))
    print(f"  Vice-C:  {vc_name}")
    vc_reason = d_vc.get("reasoning", "")
    if vc_reason:
        print(_wrap(vc_reason[:100], indent="    "))

    # --- Bench ---
    print()
    print("BENCH")
    bench_names = [
        d_bench.get("bench_gk", bench_gk_list[0]["web_name"] if bench_gk_list else "?"),
        d_bench.get("bench_1",  bench_out_list[0]["web_name"] if len(bench_out_list) > 0 else "?"),
        d_bench.get("bench_2",  bench_out_list[1]["web_name"] if len(bench_out_list) > 1 else "?"),
        d_bench.get("bench_3",  bench_out_list[2]["web_name"] if len(bench_out_list) > 2 else "?"),
    ]
    bench_map = {}
    for p in squad:
        if p["player_id"] not in {pp["player_id"] for pp in xi}:
            bench_map[p["web_name"]] = p
            bench_map[p["name"]]     = p

    b_labels = ["GK:  ", "1st: ", "2nd: ", "3rd: "]
    for lbl, bname in zip(b_labels, bench_names):
        bp = bench_map.get(bname)
        if bp:
            pred = round(pred_gw0.get(bp["player_id"], 0), 1)
            print(f"  {lbl}{bname:<22} pred:{pred:.1f}")
        else:
            print(f"  {lbl}{bname}")
    bench_reason = d_bench.get("reasoning", "")
    if bench_reason:
        print(_wrap(bench_reason[:100]))

    # --- Transfers ---
    print()
    print("TRANSFERS")
    if result["transfers_in"]:
        for p in result["transfers_in"]:
            print(f"  IN:  {p['web_name']:<22} ({p['team_short']})  GBP{p['value']:.1f}m")
        for p in result["transfers_out"]:
            print(f"  OUT: {p['web_name']:<22} ({p['team_short']})  GBP{p['value']:.1f}m")
        pen  = result["penalty"]
        cost = f"-{pen*4} pts" if pen > 0 else "Free"
        print(f"  Cost: {cost}  ({len(result['transfers_in'])} transfer(s))")
    else:
        print("  None" if gw > 1 else "  None (GW1 fresh squad)")

    if d_xfer:
        print("  LLM transfer suggestions:")
        for t in d_xfer:
            print(
                f"    OUT: {t.get('out','?')} -> IN: {t.get('in','?')}  "
                f"[{t.get('priority','?')}]"
            )
            reason = t.get("reasoning", "")
            if reason:
                print(f"      {reason[:80]}")

    # --- Chip advice ---
    print()
    print("CHIP ADVICE")
    chip_play   = d_chip.get("play_chip", "none")
    chip_reason = d_chip.get("reasoning", "")
    chip_label  = chip_play.upper() if chip_play != "none" else "No chip this week"
    print(f"  {chip_label}")
    if chip_reason:
        print(_wrap(chip_reason[:100]))

    # --- Risk flags (Call 1 + Call 2) ---
    call1_flags = decisions_call1.get("flagged_players", [])
    all_flags   = [
        f"[{fp.get('confidence','?').upper()}] "
        f"{fp.get('name','?')}: {fp.get('reason','')[:70]}"
        for fp in call1_flags if fp.get("confidence", "low") in ("high", "medium")
    ] + [str(f)[:80] for f in d_flags]

    if all_flags:
        print()
        print("RISK FLAGS")
        for flag in all_flags:
            print(f"  [!] {flag}")

    # --- Predicted score ---
    print()
    print(f"  PREDICTED SCORE: {gw_pred:.1f} pts")
    print("-" * 54)


# ===========================================================================
# Comparison summary table
# ===========================================================================
def print_comparison_table(gw_log, s8_actuals=STAGE8_GW_ACTUALS,
                            s8_caps=STAGE8_CAPTAINS):
    n  = len(gw_log)
    s8 = s8_actuals[:n]

    s8_total = sum(s8)
    s9_total = sum(e.get("simulated_actual", e["predicted_score"]) for e in gw_log)
    s8_avg   = s8_total / n if n else 0
    s9_avg   = s9_total / n if n else 0
    diff     = s9_total - s8_total

    print()
    print(SEP)
    print("  STAGE 8 vs STAGE 9 COMPARISON")
    print(SEP)
    print(f"  {'Stage':<12} {'Total Pts':<12} {'Avg/GW':<10} vs Stage 8")
    print(f"  {'-'*11:<12} {'-'*10:<12} {'-'*8:<10} {'-'*10}")
    print(f"  {'Stage 8':<12} {s8_total:<12} {s8_avg:<10.1f} baseline")
    diff_str = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
    print(f"  {'Stage 9':<12} {s9_total:<12.0f} {s9_avg:<10.1f} {diff_str} pts")

    print()
    print(f"  {'GW':<4} {'S8 Pts':<8} {'S9 Pts':<8} {'Diff':<8} "
          f"{'Cap S8':<15} {'Cap S9':<15} Override?")
    print(f"  {'-'*4:<4} {'-'*6:<8} {'-'*6:<8} {'-'*6:<8} "
          f"{'-'*13:<15} {'-'*13:<15} {'-'*8}")

    for i, entry in enumerate(gw_log):
        gw_n  = entry["gw"]
        s8_v  = s8[i] if i < len(s8) else "?"
        s9_v  = entry.get("simulated_actual", entry["predicted_score"])
        d     = (s9_v - s8_v) if isinstance(s8_v, (int, float)) else 0
        d_str = f"+{d:.0f}" if d >= 0 else f"{d:.0f}"
        cap8  = s8_caps[i] if i < len(s8_caps) else "?"
        cap9  = entry.get("captain_final", entry.get("captain_ilp", "?"))
        ov    = "Yes" if entry.get("captain_override", False) else "No"
        print(f"  {gw_n:<4} {s8_v:<8} {s9_v:<8.0f} {d_str:<8} {cap8:<15} {cap9:<15} {ov}")

    print(SEP)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(SEP)
    print("  FPL AI Stage 9 -- LLM Agent Layer")
    print(SEP)

    # --- Anthropic client ---
    print("\nInitialising Anthropic client...")
    client = get_client()
    print(f"  Client ready  model: {MODEL_ID}")

    # --- Stage 8 infrastructure ---
    print("\nLoading Stage 8 models...")
    models, model_fcols = stage8.load_models()
    best_params = stage8.load_best_params()

    print("\nLoading training data...")
    train_dfs = stage8.load_training_data()

    print("\nLoading FPL API data...")
    players_raw, ph, fixtures, fdr_df, upcoming, teams = stage8.load_fpl_data()

    print("\nDetecting DGWs / BGWs...")
    dgw_gws      = stage8.detect_dgw(fixtures)
    all_team_ids = list(teams["id"].unique())
    all_gws      = list(range(1, 39))
    bgw_by_gw    = stage8.detect_bgw(fixtures, all_gws, all_team_ids)
    print(f"  DGW gameweeks: {sorted(dgw_gws.keys()) or 'None'}")

    sim_start_gw = 1  # Force GW1 so we can compare against real player_history.csv
    print(f"  Simulation starts at real GW{sim_start_gw}")

    print("\nFetching live FPL injury data...")
    injury_news = fetch_fpl_injury_news()
    if not injury_news:
        print("  Bootstrap failed -- falling back to per-player Google search for flagged players")

    print("\nBuilding player pool...")
    player_pool = stage8.build_player_pool(
        players_raw, train_dfs, upcoming, teams,
        sim_start_gw=sim_start_gw,
    )

    # Normalise upcoming gameweek column
    upcoming_norm = upcoming.dropna(subset=["gameweek"]).copy()
    upcoming_norm["gameweek"] = upcoming_norm["gameweek"].astype(int)

    # --- Simulation state ---
    current_squad   = None
    free_transfers  = 1
    chips_used      = set()
    total_pts       = 0.0
    new_rows_by_pos = defaultdict(list)

    stage9_results = {
        "model":        MODEL_ID,
        "sim_start_gw": sim_start_gw,
        "gameweeks":    [],
        "total_predicted":  0.0,
        "total_simulated":  0.0,
    }
    gw_log = []

    # =========================================================
    for gw in range(1, 11):
        print(f"\n{'-'*54}")
        print(f"  GW{gw}  (real GW{sim_start_gw + gw - 1})")
        print(f"{'-'*54}")

        real_gw = sim_start_gw + (gw - 1)

        # ---- Stage 8 base predictions ----
        pred_gw0   = stage8.predict_gw0(player_pool, models, model_fcols)
        horizon    = stage8.predict_horizon(
            player_pool, models, real_gw=real_gw,
            upcoming_df=upcoming_norm,
            dgw_gws=dgw_gws, bgw_teams_by_gw=bgw_by_gw,
            model_fcols=model_fcols,
        )

        # ---- Build context package ----
        ctx = build_context_package(
            player_pool, pred_gw0, upcoming_norm,
            players_raw, real_gw, top_n=CALL1_POOL_SIZE,
            injury_news=injury_news,
        )

        # ---- LLM Call 1: Pre-ILP risk filter ----
        squad_names = [p["web_name"] for p in current_squad] if current_squad else None
        raw1, d1    = llm_call1_risk_filter(client, gw, real_gw, ctx, squad_names)
        time.sleep(2)

        # Per-player fallback search only if bootstrap API failed
        search_snippets = {}
        if not injury_news:
            flagged_so_far = d1.get("flagged_players", [])
            for fp in flagged_so_far[:5]:
                name = fp.get("name", "")
                if name:
                    print(f"  Fallback search: {name}...")
                    fallback_google_search(name)  # result unused but logged

        # ---- Apply risk penalties ----
        flagged = d1.get("flagged_players", [])
        penalized_pred, penalized_horizon = apply_risk_penalties(
            player_pool, pred_gw0, horizon, flagged
        )

        n_high   = sum(1 for f in flagged if f.get("confidence") == "high")
        n_medium = sum(1 for f in flagged if f.get("confidence") == "medium")
        if flagged:
            print(f"  Flagged: {len(flagged)} players  "
                  f"({n_high} high-conf -> -50%,  {n_medium} medium -> -25%)")
            for f in flagged:
                if f.get("confidence") in ("high", "medium"):
                    print(f"    {f['name']}: {f['risk_type']} [{f['confidence']}]")

        # ---- Chip check (uses penalized predictions) ----
        chip = is_wildcard = is_freehit = None
        is_wildcard = False
        is_freehit  = False

        if current_squad is not None:
            proxy_xi = sorted(
                [p for p in current_squad if p["pos_id"] != 1],
                key=lambda p: penalized_pred.get(p["player_id"], 0),
                reverse=True,
            )[:10]
            proxy_xi_ids   = {p["player_id"] for p in proxy_xi}
            bench_proxy    = [
                p for p in current_squad
                if p["player_id"] not in proxy_xi_ids and p["pos_id"] != 1
            ]
            chip_raw = stage8.check_chips(
                real_gw, dgw_gws, chips_used,
                current_squad, penalized_pred, bench_proxy,
            )
            if chip_raw == "freehit":
                chip          = "freehit"
                is_freehit    = True
                chips_used.add("freehit")
                free_transfers = 15
            elif chip_raw and chip_raw.startswith("wildcard"):
                chip          = chip_raw
                is_wildcard   = True
                chips_used.add(chip_raw)
                free_transfers = 15
            elif chip_raw == "bench_boost":
                chip = "bench_boost"
                chips_used.add("bench_boost")
            elif chip_raw == "triple_captain":
                chip = "triple_captain"
                chips_used.add("triple_captain")

        # ---- ILP with penalized scores ----
        result = stage8.run_ilp(
            player_pool    = player_pool,
            horizon_scores = penalized_horizon,
            pred_gw0_scores= penalized_pred,
            budget         = stage8.BUDGET,
            prev_squad     = current_squad if gw > 1 else None,
            free_transfers = free_transfers if gw > 1 else 1,
            is_wildcard    = is_wildcard,
            is_freehit     = is_freehit,
        )

        if result is None:
            print(f"  GW{gw}: ILP failed, skipping.")
            continue

        if not is_freehit:
            current_squad = result["squad"]

        cap_ilp = result["captain"]["web_name"] if result["captain"] else "None"

        # ---- LLM Call 2: Post-ILP captain + bench + report ----
        raw2, d2 = llm_call2_report(
            client, gw, result, penalized_pred,
            flagged_players   = flagged,
            chips_used        = chips_used,
            total_pts_so_far  = total_pts,
            search_snippets   = search_snippets if search_snippets else None,
            penalized_horizon = penalized_horizon,
        )
        time.sleep(2)

        # ---- Captain override ----
        cap_final_name = d2.get("captain", {}).get("name", cap_ilp)
        cap_override   = d2.get("captain", {}).get("override", False)
        captain_override_applied = False

        if cap_override and cap_final_name != cap_ilp:
            xi_by_name = {p["web_name"]: p for p in result["xi"]}
            xi_by_name.update({p["name"]: p for p in result["xi"]})
            new_cap = xi_by_name.get(cap_final_name)
            if new_cap:
                result["captain"]        = new_cap
                captain_override_applied = True
                print(f"  Captain override: {cap_ilp} -> {cap_final_name}")
            else:
                print(f"  Captain override '{cap_final_name}' not in XI, keeping {cap_ilp}")
                cap_final_name = cap_ilp

        # ---- Predicted GW score ----
        bench_gk_l, bench_out_l = stage8.split_bench(
            result["squad"], result["xi"], penalized_pred
        )
        cap_id   = result["captain"]["player_id"] if result["captain"] else -1
        vic_id   = result["vice"]["player_id"]    if result["vice"]    else -1
        cap_mult = 3 if chip == "triple_captain" else 2

        xi_sum    = sum(penalized_pred.get(p["player_id"], 0) for p in result["xi"])
        cap_bonus = penalized_pred.get(cap_id, 0) * (cap_mult - 1)  # extra beyond starter
        vc_bonus  = penalized_pred.get(vic_id, 0) * 0.5
        bench_sum = (
            sum(penalized_pred.get(p["player_id"], 0)
                for p in bench_gk_l + bench_out_l)
            if chip == "bench_boost" else 0
        )
        gw_pred  = xi_sum + cap_bonus + vc_bonus + bench_sum - result["penalty"] * 4

        # ---- Print full report ----
        print_gw_report(
            gw, result, penalized_pred,
            d1, d2, raw2, gw_pred,
        )

        # ---- Simulate actual score (deterministic -- mirrors Stage 8 RNG) ----
        rng = np.random.RandomState(stage8.SEED + gw)
        actuals = {}
        for p in player_pool:
            pid  = p["player_id"]
            pred = pred_gw0.get(pid, 0.0)     # use original (not penalized)
            actuals[pid] = float(np.clip(rng.normal(pred, 2.5), -2, 26))

        actual_xi   = sum(actuals.get(p["player_id"], 0) for p in result["xi"])
        actual_cap  = actuals.get(cap_id, 0) * (cap_mult - 1)
        actual_vc   = actuals.get(vic_id, 0) * 0.5
        actual_bench = (
            sum(actuals.get(p["player_id"], 0) for p in bench_gk_l + bench_out_l)
            if chip == "bench_boost" else 0
        )
        gw_actual = actual_xi + actual_cap + actual_vc + actual_bench - result["penalty"] * 4
        total_pts += gw_actual

        # ---- Look up REAL points from player_history.csv ----
        ph_gw = ph[ph["gameweek"] == real_gw] if ph is not None else None
        real_pts_by_pid = {}
        if ph_gw is not None and len(ph_gw) > 0:
            for _, row in ph_gw.iterrows():
                real_pts_by_pid[int(row["player_id"])] = int(row["total_points"])

        xi_player_rows = []
        for p in result["xi"]:
            pid  = p["player_id"]
            pred = round(penalized_pred.get(pid, 0.0), 1)
            real = real_pts_by_pid.get(pid, None)
            cap_mark = " [C]" if pid == cap_id else (" [V]" if pid == vic_id else "")
            xi_player_rows.append({
                "name":  p["web_name"] + cap_mark,
                "pid":   pid,
                "pred":  pred,
                "real":  real,
            })

        real_xi_total = sum(r["real"] for r in xi_player_rows if r["real"] is not None)
        cap_real = real_pts_by_pid.get(cap_id, None)
        real_cap_bonus = cap_real * (cap_mult - 1) if cap_real is not None else 0
        vc_real  = real_pts_by_pid.get(vic_id, None)
        real_vc_bonus  = vc_real * 0.5 if vc_real is not None else 0
        real_total = real_xi_total + real_cap_bonus + real_vc_bonus - result["penalty"] * 4

        print(f"\n  --- REAL vs PREDICTED (GW{real_gw}) ---")
        print(f"  {'Player':<26} {'Pred':>6}  {'Real':>6}  {'Diff':>6}")
        print(f"  {'-'*50}")
        for r in xi_player_rows:
            real_str = str(r["real"]) if r["real"] is not None else "n/a"
            diff_str = f"{r['real'] - r['pred']:+.1f}" if r["real"] is not None else "n/a"
            print(f"  {r['name']:<26} {r['pred']:>6.1f}  {real_str:>6}  {diff_str:>6}")
        print(f"  {'-'*50}")
        print(f"  {'XI total':<26} {xi_sum:>6.1f}  {real_xi_total:>6}  {real_xi_total - xi_sum:>+6.1f}")
        print(f"  {'GW total (w/ cap bonus)':<26} {gw_pred:>6.1f}  {real_total:>6.1f}  {real_total - gw_pred:>+6.1f}")

        # ---- Log entry ----
        entry = {
            "gw":                gw,
            "real_gw":           real_gw,
            "chip":              chip,
            "predicted_score":   round(gw_pred, 1),
            "simulated_actual":  round(gw_actual, 1),
            "real_total":        round(real_total, 1),
            "captain_ilp":       cap_ilp,
            "captain_final":     cap_final_name,
            "captain_override":  captain_override_applied,
            "vice_captain":      result["vice"]["web_name"] if result["vice"] else "None",
            "penalty":           result["penalty"],
            "n_flagged":         len(flagged),
            "flagged_high":      [f["name"] for f in flagged if f.get("confidence") == "high"],
            "flagged_medium":    [f["name"] for f in flagged if f.get("confidence") == "medium"],
            "xi":                [p["web_name"] for p in result["xi"]],
            "xi_player_rows":    xi_player_rows,
            "transfers_in":      [p["web_name"] for p in result["transfers_in"]],
            "transfers_out":     [p["web_name"] for p in result["transfers_out"]],
            "squad_value":       round(sum(p["value"] for p in result["squad"]), 1),
            "call1_prose":       raw1[:600]   if raw1 else "",
            "call2_prose":       raw2[:1000]  if raw2 else "",
            "decisions_call1":   d1,
            "decisions_call2":   d2,
        }
        gw_log.append(entry)
        stage9_results["gameweeks"].append(entry)

        # ---- Online retraining (mirrors Stage 8 exactly) ----
        if gw < 10:
            print(f"\n  Retraining models for GW{gw+1}...")
            for p in player_pool:
                pid    = p["player_id"]
                pos_id = p["pos_id"]
                row    = p["feat_vec"].copy()
                row[stage8.TARGET_COL] = actuals[pid]
                row["season"] = "2025-26"
                row["GW"]     = gw
                new_rows_by_pos[pos_id].append(row)

            models, model_fcols = stage8.retrain_models(
                train_dfs, dict(new_rows_by_pos), best_params, model_fcols
            )

        # ---- Update free transfers ----
        if is_wildcard or is_freehit:
            free_transfers = 1
        else:
            n_tr = len(result["transfers_in"])
            free_transfers = (
                min(2, free_transfers - n_tr + 1)
                if n_tr <= free_transfers else 1
            )

    # =========================================================
    # Final summary
    stage9_results["total_predicted"] = round(
        sum(e["predicted_score"] for e in gw_log), 1
    )
    stage9_results["total_simulated"] = round(total_pts, 1)
    stage9_results["total_real"] = round(
        sum(e.get("real_total", 0) for e in gw_log), 1
    )

    print_comparison_table(gw_log)

    # --- Real vs Predicted summary table ---
    print("\n" + "="*70)
    print("  REAL vs PREDICTED SUMMARY (GW1-10)")
    print("="*70)
    print(f"  {'GW':<4} {'Predicted':>10} {'Real':>8} {'Diff':>8}  Captain")
    print(f"  {'-'*60}")
    for e in gw_log:
        diff = e.get("real_total", 0) - e["predicted_score"]
        print(f"  GW{e['real_gw']:<3} {e['predicted_score']:>10.1f} "
              f"{e.get('real_total', 0):>8.1f} {diff:>+8.1f}  {e['captain_final']}")
    tot_pred = sum(e["predicted_score"] for e in gw_log)
    tot_real = sum(e.get("real_total", 0) for e in gw_log)
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<4} {tot_pred:>10.1f} {tot_real:>8.1f} {tot_real - tot_pred:>+8.1f}")
    mae = sum(abs(e.get("real_total", 0) - e["predicted_score"]) for e in gw_log) / len(gw_log)
    print(f"  MAE per GW: {mae:.1f} pts")

    # Per-player accuracy across all GWs
    player_stats = defaultdict(lambda: {"preds": [], "reals": []})
    for e in gw_log:
        for row in e.get("xi_player_rows", []):
            name = row["name"].replace(" [C]", "").replace(" [V]", "")
            if row["real"] is not None:
                player_stats[name]["preds"].append(row["pred"])
                player_stats[name]["reals"].append(row["real"])

    print("\n  PER-PLAYER ACCURACY (players with 3+ GW appearances)")
    print(f"  {'Player':<26} {'Apps':>4} {'AvgPred':>8} {'AvgReal':>8} {'MAE':>6} {'Bias':>8}")
    print(f"  {'-'*64}")
    rows_out = []
    for name, d in player_stats.items():
        if len(d["preds"]) >= 3:
            avg_pred = sum(d["preds"]) / len(d["preds"])
            avg_real = sum(d["reals"]) / len(d["reals"])
            mae_p    = sum(abs(r - p) for r, p in zip(d["reals"], d["preds"])) / len(d["preds"])
            bias     = avg_real - avg_pred
            rows_out.append((name, len(d["preds"]), avg_pred, avg_real, mae_p, bias))
    rows_out.sort(key=lambda x: x[4], reverse=True)  # sort by MAE desc
    for name, apps, avg_pred, avg_real, mae_p, bias in rows_out:
        print(f"  {name:<26} {apps:>4} {avg_pred:>8.1f} {avg_real:>8.1f} "
              f"{mae_p:>6.1f} {bias:>+8.1f}")

    # Save full results
    out_path = os.path.join(MODELS_DIR, "stage9_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stage9_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    print(f"Total API calls made: {len(gw_log) * 2} (2 per GW)")


if __name__ == "__main__":
    main()
