"""
Intel 05: LLM-Powered Captain and Transfer Recommendations
FPL AI Thesis -- Gemini 2.5 Flash integration

Reads intel outputs 01-04 plus historical form and fixture data,
calls Gemini 2.5 Flash once per GW (GW 1-10), and produces structured
captain/transfer/risk recommendations.

Output: data/intel/recommendations.json
"""

import os
import sys
import json
import time
import re
import warnings
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(ROOT, "data", "raw", "fpl_api")
DATA_INTEL = os.path.join(ROOT, "data", "intel")

INTEL_LIVE       = os.path.join(DATA_INTEL, "fpl_live.json")
INTEL_PRESS      = os.path.join(DATA_INTEL, "press_conferences.json")
INTEL_AVAIL      = os.path.join(DATA_INTEL, "availability.json")
INTEL_ROT        = os.path.join(DATA_INTEL, "rotation_risk.json")
PLAYER_HISTORY   = os.path.join(DATA_RAW, "player_history.csv")
FIXTURE_DIFF     = os.path.join(DATA_RAW, "fixture_difficulty.csv")
OUTPUT_PATH      = os.path.join(DATA_INTEL, "recommendations.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID         = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 65536
TEMPERATURE      = 0

GW_START = 1
GW_END   = 38
POOL_SIZE = 40   # top-N players by form for LLM context

SEP = "=" * 60

SYSTEM_PROMPT = """You are an expert FPL analyst for the 2025/26 Premier League season.
You receive a pre-deadline intelligence dossier for each gameweek
containing player form, availability, rotation risk, fixtures, and
press conference quotes.

Your job is to produce actionable recommendations:
1. CAPTAIN PICK: Best captain choice with reasoning
2. DIFFERENTIAL PICK: A low-ownership (<10%) player worth considering
3. TRANSFER TARGETS IN: Top 3 players to buy, with priority
4. TRANSFER TARGETS OUT: Top 3 players to sell/avoid, with priority
5. RISK WARNINGS: Any urgent alerts for the gameweek

Base your analysis ONLY on the data provided. Be specific and
reference the numbers (form, FDR, availability %, rotation risk).

IMPORTANT: Output the <decisions> JSON block FIRST, then your prose analysis AFTER.
This ensures the structured data is always complete even if the response is long."""


# ===========================================================================
# Gemini client
# ===========================================================================
def get_client():
    """Return google-genai Client using GEMINI_API_KEY env var."""
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        return genai.Client(api_key=api_key)
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")


def call_llm(client, system_prompt, user_msg):
    """Call Gemini 2.5 Flash with exponential backoff retry on 503/rate-limit errors."""
    from google import genai
    from google.genai import types

    delays = [5, 15, 30]  # seconds between retries
    last_err = None
    for attempt, delay in enumerate([0] + delays):
        if delay:
            print(f"  [RETRY] Waiting {delay}s before attempt {attempt + 1}...")
            time.sleep(delay)
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            return response.text
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"  [WARN] Gemini rate-limit/unavailable (attempt {attempt + 1}): {err_str[:80]}")
                continue  # retry
            raise  # non-retriable error — propagate immediately
    raise last_err


# ===========================================================================
# Parse <decisions> block
# ===========================================================================
def parse_decisions(text, fallback=None):
    """Extract JSON from <decisions>...</decisions> or ```json code blocks."""
    try:
        # Try <decisions> tags first
        m = re.search(r"<decisions>(.*?)</decisions>", text, re.DOTALL)
        if m:
            return json.loads(m.group(1).strip())
        # Fallback: try ```json ... ``` blocks
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1).strip())
        print("  [WARN] No <decisions> block or ```json block found in LLM response")
        return fallback
    except Exception as e:
        print(f"  [WARN] JSON parse failed: {e}")
        return fallback


def normalize_decisions(raw):
    """Normalize Gemini's varying JSON schema to the expected format."""
    if not raw or not isinstance(raw, dict):
        return raw
    inner = raw.get("decisions", raw)

    def _get_player(d):
        """Extract player name from varying key formats, strip team suffix."""
        val = d.get("player_name") or d.get("player") or d.get("name") or "N/A"
        # Strip team in parentheses: "Haaland (MCI)" -> "Haaland"
        val = re.sub(r"\s*\([A-Z]{3}\)\s*$", "", val)
        return val

    def _get_reason(d):
        return d.get("reasoning") or d.get("reason") or ""

    # Captain
    cap_raw = (inner.get("captain_pick") or inner.get("CAPTAIN_PICK")
               or inner.get("captain") or {})
    captain = {
        "name":      _get_player(cap_raw),
        "team":      cap_raw.get("team", "N/A"),
        "position":  cap_raw.get("position", "N/A"),
        "reasoning": _get_reason(cap_raw),
    }
    # Vice captain
    vc_raw = (inner.get("vice_captain_pick") or inner.get("VICE_CAPTAIN_PICK")
              or inner.get("vice_captain") or {})
    vice_captain = {
        "name":      _get_player(vc_raw),
        "team":      vc_raw.get("team", "N/A"),
        "reasoning": _get_reason(vc_raw),
    }
    # Differential
    diff_raw = (inner.get("differential_pick") or inner.get("DIFFERENTIAL_PICK")
                or inner.get("differential") or {})
    differential = {
        "name":          _get_player(diff_raw),
        "team":          diff_raw.get("team", "N/A"),
        "ownership_pct": diff_raw.get("ownership_pct") or diff_raw.get("ownership_percentage") or diff_raw.get("ownership") or 0,
        "reasoning":     _get_reason(diff_raw),
    }
    # Transfers in
    tin_raw = (inner.get("transfer_targets_in") or inner.get("TRANSFER_TARGETS_IN")
               or inner.get("transfers_in") or [])
    transfers_in = []
    for t in tin_raw:
        transfers_in.append({
            "name":      _get_player(t),
            "team":      t.get("team", "?"),
            "position":  t.get("position", "?"),
            "priority":  t.get("priority", "recommended") if isinstance(t.get("priority"), str) else f"#{t.get('priority', '?')}",
            "reasoning": _get_reason(t),
        })
    # Transfers out
    tout_raw = (inner.get("transfer_targets_out") or inner.get("TRANSFER_TARGETS_OUT")
                or inner.get("transfers_out") or [])
    transfers_out = []
    for t in tout_raw:
        transfers_out.append({
            "name":      _get_player(t),
            "team":      t.get("team", "?"),
            "position":  t.get("position", "?"),
            "priority":  t.get("priority", "recommended") if isinstance(t.get("priority"), str) else f"#{t.get('priority', '?')}",
            "reasoning": _get_reason(t),
        })
    # Risk warnings
    risk_warnings = (inner.get("risk_warnings") or inner.get("RISK_WARNINGS") or [])
    # Narrative
    gw_narrative = (inner.get("gw_narrative") or inner.get("GW_NARRATIVE") or "")
    return {
        "captain":       captain,
        "vice_captain":  vice_captain,
        "differential":  differential,
        "transfers_in":  transfers_in,
        "transfers_out": transfers_out,
        "risk_warnings": risk_warnings,
        "gw_narrative":  gw_narrative,
    }


# ===========================================================================
# Data loading
# ===========================================================================
def load_inputs():
    """Load all intel sources and return as a dict."""
    print("Loading intel sources...")

    with open(INTEL_LIVE, encoding="utf-8") as f:
        fpl_live = json.load(f)
    print(f"  fpl_live.json: {len(fpl_live['players'])} players, {len(fpl_live['teams'])} teams")

    with open(INTEL_PRESS, encoding="utf-8") as f:
        press = json.load(f)
    print(f"  press_conferences.json: {len(press['gameweeks'])} GWs")

    with open(INTEL_AVAIL, encoding="utf-8") as f:
        avail = json.load(f)
    print(f"  availability.json: {len(avail['gameweeks'])} GWs")

    with open(INTEL_ROT, encoding="utf-8") as f:
        rot = json.load(f)
    print(f"  rotation_risk.json: {len(rot['gameweeks'])} GWs")

    ph = pd.read_csv(PLAYER_HISTORY)
    print(f"  player_history.csv: {len(ph)} rows, GWs {ph['gameweek'].min()}-{ph['gameweek'].max()}")

    fd = pd.read_csv(FIXTURE_DIFF)
    print(f"  fixture_difficulty.csv: {len(fd)} rows")

    return {
        "fpl_live": fpl_live,
        "press": press,
        "avail": avail,
        "rot": rot,
        "player_history": ph,
        "fixture_diff": fd,
    }


# ===========================================================================
# Team name mapping (short_name -> fixture_difficulty team_name)
# ===========================================================================
def build_team_maps(fpl_live):
    """
    Build two dicts:
      short_to_full:  'ARS' -> 'Arsenal'
      id_to_short:    '1'   -> 'ARS'
    """
    short_to_full = {}
    id_to_short   = {}
    for tid, t in fpl_live["teams"].items():
        short = t["short_name"]
        full  = t["name"]
        id_to_short[str(tid)] = short
        short_to_full[short]  = full
    return short_to_full, id_to_short


# ===========================================================================
# Rolling form computation (backtest-safe)
# ===========================================================================
def compute_all_forms(player_history_df, gw, window=3):
    """
    Returns dict: player_id (int) -> rolling form (avg pts over last `window` GWs).
    Uses only GWs strictly before `gw` (no leakage).
    """
    prior = player_history_df[player_history_df["gameweek"] < gw]
    forms = {}
    for pid, grp in prior.groupby("player_id"):
        grp_sorted = grp.sort_values("gameweek")
        recent = grp_sorted.tail(window)
        forms[int(pid)] = round(recent["total_points"].mean(), 2)
    return forms


def compute_season_stats(player_history_df, gw):
    """
    Returns dict: player_id (int) -> {total_pts, gws_played, ppg}
    Cumulative up to (but not including) current GW.
    """
    prior = player_history_df[player_history_df["gameweek"] < gw]
    stats = {}
    for pid, grp in prior.groupby("player_id"):
        total = int(grp["total_points"].sum())
        n     = len(grp)
        ppg   = round(total / n, 2) if n > 0 else 0.0
        stats[int(pid)] = {"total_pts": total, "gws_played": n, "ppg": ppg}
    return stats


# ===========================================================================
# Fixture lookup
# ===========================================================================
def get_player_fixtures(player_team_short, gw, fixture_diff_df, short_to_full, next_n=3):
    """
    Returns:
      current_fixture: {'opponent': str, 'was_home': bool, 'fdr': int} or None
      next_fixtures:   list of up to next_n dicts (the GWs after current)
    """
    full_name = short_to_full.get(player_team_short)
    if not full_name:
        return None, []

    team_rows = fixture_diff_df[fixture_diff_df["team_name"] == full_name].sort_values("gameweek")

    current = team_rows[team_rows["gameweek"] == gw]
    upcoming = team_rows[team_rows["gameweek"] > gw].head(next_n)

    curr_fix = None
    if not current.empty:
        row = current.iloc[0]
        curr_fix = {
            "opponent": row["opponent"],
            "was_home": bool(row["was_home"]),
            "fdr": int(row["fdr"]),
        }

    next_fixes = []
    for _, row in upcoming.iterrows():
        next_fixes.append({
            "gw": int(row["gameweek"]),
            "opponent": row["opponent"],
            "was_home": bool(row["was_home"]),
            "fdr": int(row["fdr"]),
        })

    return curr_fix, next_fixes


# ===========================================================================
# Build context package for a single GW
# ===========================================================================
def build_context_for_gw(gw, inputs):
    """
    Compile the intelligence dossier for the given GW.
    Returns (user_msg, context_summary) tuple.
    """
    fpl_live     = inputs["fpl_live"]
    press        = inputs["press"]
    avail        = inputs["avail"]
    rot          = inputs["rot"]
    ph           = inputs["player_history"]
    fd           = inputs["fixture_diff"]

    short_to_full, id_to_short = build_team_maps(fpl_live)

    # Compute form and season stats (backtest-safe: use GWs < current_gw)
    gw_is_first = (gw == 1)
    forms       = compute_all_forms(ph, gw) if not gw_is_first else {}
    season_stats = compute_season_stats(ph, gw) if not gw_is_first else {}

    # Get per-GW intel (flagged players only; others default to safe values)
    gw_str = str(gw)
    avail_players = avail["gameweeks"].get(gw_str, {}).get("players", {})
    rot_players   = rot["gameweeks"].get(gw_str, {}).get("players", {})

    # Build enriched player list
    all_players = []
    for pid_str, p in fpl_live["players"].items():
        pid = int(pid_str)

        # Form: rolling 3-GW avg or 0 for GW1
        form = forms.get(pid, 0.0)

        # Season stats
        ss   = season_stats.get(pid, {"total_pts": 0, "gws_played": 0, "ppg": 0.0})

        # Availability
        avail_entry = avail_players.get(pid_str) or avail_players.get(str(pid))
        avail_pct   = avail_entry["availability_pct"] if avail_entry else 95
        avail_tier  = avail_entry.get("availability_tier", "available") if avail_entry else "available"
        avail_quote = avail_entry.get("press_conf_quote", "") if avail_entry else ""

        # Rotation risk
        rot_entry  = rot_players.get(pid_str) or rot_players.get(str(pid))
        rot_risk   = rot_entry["rotation_risk"] if rot_entry else None
        rot_tier   = rot_entry.get("rotation_tier") if rot_entry else None

        # Fixture
        curr_fix, next_fixes = get_player_fixtures(
            p["team_short"], gw, fd, short_to_full
        )

        all_players.append({
            "pid":         pid,
            "name":        p["web_name"],
            "pos":         p["position"],
            "team":        p["team_short"],
            "price":       p["price"],
            "form":        form,
            "total_pts":   ss["total_pts"],
            "ppg":         ss["ppg"],
            "ppg_prev":    p["points_per_game"],   # FPL's GW29 PPG (prior season proxy for GW1)
            "ownership":   p["ownership_pct"],
            "avail_pct":   avail_pct,
            "avail_tier":  avail_tier,
            "avail_quote": avail_quote,
            "rot_risk":    rot_risk,
            "rot_tier":    rot_tier,
            "curr_fix":    curr_fix,
            "next_fixes":  next_fixes,
        })

    # Rank by form (or PPG for GW1), skip GKs for captain context, keep all for transfers
    if gw_is_first:
        all_players.sort(key=lambda x: x["ppg_prev"], reverse=True)
    else:
        all_players.sort(key=lambda x: (x["form"], x["ppg"]), reverse=True)

    # Take top POOL_SIZE
    pool = all_players[:POOL_SIZE]

    # Press conference quotes for this GW
    pc_gw    = press["gameweeks"].get(gw_str, {})
    all_news = pc_gw.get("all_player_news", [])

    # Build player name -> quote map from press
    press_quotes = {}
    for item in all_news:
        pname = item.get("player", "")
        if pname:
            press_quotes[pname] = item.get("news", "")

    # ---------------------------------------------------------------------------
    # Format user message
    # ---------------------------------------------------------------------------
    lines = []
    lines.append(f"Season: 2025/26 Premier League")
    lines.append(f"Gameweek: {gw}")

    if gw_is_first:
        lines.append("\nNote: GW1 -- no current season form data available. PPG column is from prior seasons.")

    lines.append(f"\n=== TOP {POOL_SIZE} PLAYERS BY {'PRIOR PPG' if gw_is_first else 'FORM'} ===")

    header = f"{'Name':<20} {'Pos':<4} {'Team':<5} {'Price':<6} {'Form':<6} {'TotPts':<8} {'PPG':<5} {'Own%':<7} {'Avail%':<8} {'RotRisk':<9} {'Opp':<14} {'H/A':<4} {'FDR':<4} {'Next3'}"
    lines.append(header)
    lines.append("-" * len(header))

    for p in pool:
        fix    = p["curr_fix"]
        opp    = fix["opponent"] if fix else "N/A"
        ha     = "H" if (fix and fix["was_home"]) else "A"
        fdr    = fix["fdr"] if fix else "-"

        # Next 3 fixtures compact string
        nxt3_parts = []
        for nf in p["next_fixes"][:3]:
            ha_n = "H" if nf["was_home"] else "A"
            nxt3_parts.append(f"{nf['opponent'][:6]}({nf['fdr']}){ha_n}")
        next3 = " ".join(nxt3_parts) if nxt3_parts else "N/A"

        rot_str  = f"{p['rot_risk']:.0f}" if p["rot_risk"] is not None else "nailed"
        form_val = p["ppg_prev"] if gw_is_first else p["form"]

        lines.append(
            f"{p['name']:<20} {p['pos']:<4} {p['team']:<5} {p['price']:<6.1f} "
            f"{form_val:<6.1f} {p['total_pts']:<8} {p['ppg']:<5.1f} {p['ownership']:<7.1f} "
            f"{p['avail_pct']:<8} {rot_str:<9} {opp:<14} {ha:<4} {fdr!s:<4} {next3}"
        )

    # Availability concerns
    concerns = [p for p in pool if p["avail_pct"] < 80]
    if concerns:
        lines.append("\n=== AVAILABILITY CONCERNS ===")
        for p in concerns:
            q = p["avail_quote"] or "(no quote)"
            lines.append(f"  {p['name']} ({p['team']}): {p['avail_pct']}% available, {p['avail_tier'].upper()}")
            lines.append(f"    Quote: {q}")

    # Rotation warnings
    rot_warns = [p for p in pool if p["rot_risk"] is not None and p["rot_risk"] >= 40]
    rot_warns.sort(key=lambda x: x["rot_risk"], reverse=True)
    if rot_warns:
        lines.append("\n=== ROTATION WARNINGS ===")
        for p in rot_warns:
            lines.append(
                f"  {p['name']} ({p['team']}): rotation_risk={p['rot_risk']:.0f}, tier={p['rot_tier']}"
            )

    # Key press quotes (players in pool)
    pool_names = {p["name"] for p in pool}
    relevant_quotes = {n: q for n, q in press_quotes.items()
                       if any(n.lower() in pn.lower() or pn.lower() in n.lower()
                              for pn in pool_names)
                       if q}
    if relevant_quotes:
        lines.append("\n=== KEY PRESS CONFERENCE QUOTES ===")
        for name, quote in list(relevant_quotes.items())[:10]:
            lines.append(f"  {name}: {quote}")

    user_msg = "\n".join(lines)

    # Context summary for output JSON
    top_player  = pool[0]["name"] if pool else "N/A"
    top_form    = pool[0]["ppg_prev"] if gw_is_first else (pool[0]["form"] if pool else 0)
    worst_avail = min(all_players, key=lambda x: x["avail_pct"])
    top_rot     = max(all_players, key=lambda x: (x["rot_risk"] or 0))

    context_summary = {
        "top_form_player":             f"{top_player} ({top_form:.1f})",
        "most_concerning_availability": f"{worst_avail['name']} ({worst_avail['avail_pct']}%)",
        "highest_rotation_risk":        (
            f"{top_rot['name']} ({top_rot['rot_risk']:.0f}, {top_rot['rot_tier']})"
            if top_rot["rot_risk"] is not None else "N/A"
        ),
    }

    return user_msg, context_summary


# ===========================================================================
# Rule-based fallback captain (if LLM fails)
# ===========================================================================
def fallback_decisions(gw, inputs):
    """Simple rule-based fallback: best available non-rotating player."""
    fpl_live = inputs["fpl_live"]
    ph       = inputs["player_history"]
    avail    = inputs["avail"]
    rot      = inputs["rot"]

    forms        = compute_all_forms(ph, gw) if gw > 1 else {}
    gw_str       = str(gw)
    avail_players = avail["gameweeks"].get(gw_str, {}).get("players", {})
    rot_players   = rot["gameweeks"].get(gw_str, {}).get("players", {})

    candidates = []
    for pid_str, p in fpl_live["players"].items():
        pid = int(pid_str)
        av  = avail_players.get(pid_str, {}).get("availability_pct", 95)
        rr  = rot_players.get(pid_str, {}).get("rotation_risk")
        if av >= 80 and (rr is None or rr < 40):
            if p["position"] == "GK":
                continue
            if p.get("ownership_pct", 0) < 1.0:
                continue
            form = forms.get(pid, p["points_per_game"])
            candidates.append((form, p["web_name"], p["team_short"], p["position"]))

    candidates.sort(reverse=True)
    top = candidates[0] if candidates else (0, "Unknown", "N/A", "N/A")

    return {
        "captain": {
            "name": top[1],
            "team": top[2],
            "position": top[3],
            "reasoning": "Rule-based fallback: highest form, availability >= 80, rotation risk < 40",
        },
        "vice_captain":   {"name": "N/A", "team": "N/A", "reasoning": "Fallback"},
        "differential":   {"name": "N/A", "team": "N/A", "ownership_pct": 0, "reasoning": "Fallback"},
        "transfers_in":   [],
        "transfers_out":  [],
        "risk_warnings":  ["LLM call failed -- using rule-based fallback"],
        "gw_narrative":   "LLM unavailable. Rule-based fallback applied.",
    }


# ===========================================================================
# Process a single GW
# ===========================================================================
def process_gw(gw, inputs, client):
    """Build context, call LLM, parse decisions. Returns result dict."""
    print(f"\nGW{gw}: building context...", end=" ", flush=True)
    user_msg, ctx_summary = build_context_for_gw(gw, inputs)

    print(f"calling {MODEL_ID}...", end=" ", flush=True)
    t0 = time.time()

    llm_success  = False
    raw_prose    = ""
    decisions    = None

    try:
        raw_prose   = call_llm(client, SYSTEM_PROMPT, user_msg)
        decisions   = parse_decisions(raw_prose)
        if decisions is not None:
            decisions = normalize_decisions(decisions)
        llm_success = decisions is not None
        elapsed     = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED ({elapsed:.1f}s): {e}")

    if not llm_success:
        print(f"  [FALLBACK] Using rule-based decisions for GW{gw}")
        decisions = fallback_decisions(gw, inputs)

    return {
        "gw":               gw,
        "pool_size":        POOL_SIZE,
        "llm_call_success": llm_success,
        "decisions":        decisions,
        "raw_prose":        raw_prose,
        "context_summary":  ctx_summary,
    }


# ===========================================================================
# Cross-GW summary
# ===========================================================================
def build_summary(gw_results):
    """Build cross-GW summary statistics."""
    captain_picks = []
    transfer_in_counts  = defaultdict(int)
    transfer_out_counts = defaultdict(int)

    for gw_str, res in gw_results.items():
        gw  = res["gw"]
        dec = res.get("decisions") or {}

        cap = dec.get("captain", {}).get("name", "N/A")
        captain_picks.append(f"GW{gw}: {cap}")

        for t in dec.get("transfers_in", []):
            transfer_in_counts[t.get("name", "?")] += 1
        for t in dec.get("transfers_out", []):
            transfer_out_counts[t.get("name", "?")] += 1

    top_in  = sorted(transfer_in_counts.items(),  key=lambda x: -x[1])[:10]
    top_out = sorted(transfer_out_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "captain_picks_across_gws":      captain_picks,
        "most_recommended_transfers_in":  [f"{n} ({c} GWs)" for n, c in top_in],
        "most_recommended_transfers_out": [f"{n} ({c} GWs)" for n, c in top_out],
    }


# ===========================================================================
# Console report
# ===========================================================================
def print_gw_report(gw, result):
    """Print compact per-GW report to console."""
    dec = result.get("decisions") or {}
    cap = dec.get("captain", {})
    vc  = dec.get("vice_captain", {})
    diff = dec.get("differential", {})
    t_in  = dec.get("transfers_in", [])
    t_out = dec.get("transfers_out", [])
    warns = dec.get("risk_warnings", [])

    print(f"\n  GW{gw} Results:")
    print(f"    Captain:    {cap.get('name','?')} ({cap.get('team','?')}) -- {cap.get('reasoning','')[:80]}")
    print(f"    Vice-cap:   {vc.get('name','?')}")
    print(f"    Diff pick:  {diff.get('name','?')} ({diff.get('ownership_pct','?')}% own) -- {diff.get('reasoning','')[:60]}")

    if t_in:
        print("    Transfers IN:")
        for t in t_in[:3]:
            print(f"      + {t.get('name','?')} ({t.get('position','?')}/{t.get('team','?')}) [{t.get('priority','?')}]")
    if t_out:
        print("    Transfers OUT:")
        for t in t_out[:3]:
            print(f"      - {t.get('name','?')} ({t.get('position','?')}/{t.get('team','?')}) [{t.get('priority','?')}]")
    if warns:
        print("    Warnings:")
        for w in warns[:3]:
            print(f"      ! {w}")


def print_final_report(output):
    """Print cross-GW summary."""
    print(f"\n{SEP}")
    print("INTEL 05: CROSS-GW SUMMARY")
    print(SEP)

    summ = output.get("summary", {})

    print("\nCaptain picks:")
    for line in summ.get("captain_picks_across_gws", []):
        print(f"  {line}")

    top_in = summ.get("most_recommended_transfers_in", [])
    if top_in:
        print("\nMost recommended transfers IN:")
        for t in top_in[:5]:
            print(f"  {t}")

    top_out = summ.get("most_recommended_transfers_out", [])
    if top_out:
        print("\nMost recommended transfers OUT:")
        for t in top_out[:5]:
            print(f"  {t}")

    # API stats
    gw_results = output.get("gameweeks", {})
    success_ct = sum(1 for r in gw_results.values() if r.get("llm_call_success"))
    total_ct   = len(gw_results)
    print(f"\nAPI calls: {success_ct}/{total_ct} successful")
    print(f"Model: {output.get('model', MODEL_ID)}")
    print(f"Output: {OUTPUT_PATH}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(SEP)
    print("INTEL 05: LLM-Powered Recommendations")
    print(f"Model: {MODEL_ID}  |  GW {GW_START}-{GW_END}")
    print(SEP)

    # Load all inputs
    inputs = load_inputs()

    # Initialise Gemini client
    print("\nInitialising Gemini client...")
    client = get_client()
    print("  OK")

    # Load existing results for merge (skip already-processed GWs)
    existing_results = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_results = existing_data.get("gameweeks", {})
            already_done = [k for k, v in existing_results.items() if v.get("llm_call_success")]
            print(f"\n  Existing recommendations.json: {len(already_done)} GWs already done "
                  f"({', '.join(sorted(already_done, key=int))})")
            print("  Skipping those — only processing missing/failed GWs.")
        except Exception as e:
            print(f"\n  Could not load existing recommendations.json: {e} — starting fresh")
            existing_results = {}

    # Process each GW (skip already successful)
    gw_results = dict(existing_results)
    for gw in range(GW_START, GW_END + 1):
        gw_str = str(gw)
        if gw_results.get(gw_str, {}).get("llm_call_success"):
            print(f"\n  GW{gw}: already processed — skipping")
            continue
        result = process_gw(gw, inputs, client)
        gw_results[gw_str] = result
        print_gw_report(gw, result)
        time.sleep(1)  # polite rate-limiting between calls

    # Build output JSON
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode":         "backtest",
        "model":        MODEL_ID,
        "sources": {
            "fpl_live":       INTEL_LIVE,
            "press":          INTEL_PRESS,
            "availability":   INTEL_AVAIL,
            "rotation_risk":  INTEL_ROT,
            "player_history": PLAYER_HISTORY,
            "fixture_diff":   FIXTURE_DIFF,
        },
        "gameweeks": gw_results,
        "summary":   build_summary(gw_results),
    }

    # Save
    os.makedirs(DATA_INTEL, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print_final_report(output)
    print(f"\nDone. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
