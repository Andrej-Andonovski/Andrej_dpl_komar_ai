"""
Stage 9: LLM Agent Layer
FPL AI Thesis -- Claude API post-analysis of the finished season simulation.

Reads data/intel/season_simulation.json (produced by season_simulator.py)
and calls Claude ONCE per gameweek to explain why each player was picked.
No decisions are made -- the simulation results are read-only.

Total API calls: 1 per GW (38 calls for a full season).
Saves explanations to models/stage9_explanations.json.
"""

import os
import sys
import json
import time
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(ROOT, "data", "raw", "fpl_api")
DATA_INTEL = os.path.join(ROOT, "data", "intel")
MODELS_DIR = os.path.join(ROOT, "models")

SIM_JSON   = os.path.join(DATA_INTEL, "season_simulation.json")
OUT_JSON   = os.path.join(MODELS_DIR, "stage9_explanations.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID   = "claude-sonnet-4-6"
MAX_TOKENS = 2500
TEMP       = 0
DELAY_S    = 1.2

SEP = "=" * 60

# ---------------------------------------------------------------------------
# System prompt — demand specific, data-driven analysis
# ---------------------------------------------------------------------------
SYS_PROMPT = (
    "You are an expert FPL analyst explaining the decisions of a LightGBM + ILP "
    "squad optimisation system for the 2025/26 Premier League season. "
    "Your job is to explain WHY the model ranked each player highly and WHY the "
    "ILP optimizer picked them. Stick strictly to the numbers in the data: "
    "predicted points, position group ranking, price, value ratio (£/pred), "
    "and comparison to bench alternatives. "
    "Write like a data analyst — reference the actual numbers. Examples: "
    "'Ranked 1st among midfielders at 8.1 predicted pts, £1.05/pred — the best "
    "value MID in the squad, and he delivered 11 actual pts.' or "
    "'At £5.1m he was the cheapest defender, predicted 4.8 pts (just above the "
    "DEF group average of 4.3), freeing budget for premium attackers.' "
    "For captaincy, compare the chosen captain directly to the listed alternatives "
    "using their predicted and actual scores. "
    "Return valid JSON only."
)


# ===========================================================================
# Load fixture context: gw -> team_id -> {opponent, is_home, fdr, opp_name}
# ===========================================================================
def load_fixture_context():
    """
    Returns dict: fixture_ctx[gw][team_id] = {
        opponent_id, opponent_name, is_home, fdr
    }
    Built from fixtures_raw.csv (2025-26 season).
    FDR: 1=very easy, 2=easy, 3=medium, 4=hard, 5=very hard (from that team's perspective).
    """
    fix_path = os.path.join(DATA_RAW, "fixtures_raw.csv")
    ctx = {}
    try:
        import pandas as pd
        df = pd.read_csv(fix_path)
        for _, row in df.iterrows():
            gw = int(row["gameweek"])
            h_id = int(row["team_h"])
            a_id = int(row["team_a"])
            h_name = str(row.get("team_h_name", h_id))
            a_name = str(row.get("team_a_name", a_id))
            h_fdr = int(row.get("team_h_difficulty", 3))
            a_fdr = int(row.get("team_a_difficulty", 3))
            ctx.setdefault(gw, {})
            ctx[gw][h_id] = {"opponent_id": a_id, "opponent_name": a_name,
                              "is_home": True,  "fdr": h_fdr}
            ctx[gw][a_id] = {"opponent_id": h_id, "opponent_name": h_name,
                              "is_home": False, "fdr": a_fdr}
    except Exception as e:
        print(f"  [WARN] Could not load fixture context: {e}")
    return ctx


# ===========================================================================
# Load team id -> short name
# ===========================================================================
def load_team_map():
    teams_path = os.path.join(DATA_RAW, "teams_raw.csv")
    try:
        import pandas as pd
        df = pd.read_csv(teams_path)
        col = "short_name" if "short_name" in df.columns else "name"
        return {int(row["id"]): str(row[col]) for _, row in df.iterrows()}
    except Exception:
        return {}


# ===========================================================================
# Load intel: availability + rotation risk
# ===========================================================================
def load_intel():
    """
    Returns nested dict: intel[gw][player_id] = {
        avail_pct, avail_tier, rotation_risk, rot_tier, start_rate,
        factors, press_quote
    }
    """
    intel = {}

    avail_path = os.path.join(DATA_INTEL, "availability.json")
    rot_path   = os.path.join(DATA_INTEL, "rotation_risk.json")

    if os.path.exists(avail_path):
        with open(avail_path, encoding="utf-8") as f:
            avail_data = json.load(f)
        for gw_str, gw_data in avail_data.get("gameweeks", {}).items():
            gw = int(gw_str)
            intel.setdefault(gw, {})
            for pid_str, p in gw_data.get("players", {}).items():
                pid = int(pid_str)
                intel[gw].setdefault(pid, {})
                intel[gw][pid]["avail_pct"]  = p.get("availability_pct", 100)
                intel[gw][pid]["avail_tier"] = p.get("availability_tier", "available")
                quote = p.get("press_conf_quote", "")
                if quote and p.get("availability_pct", 100) < 95:
                    intel[gw][pid]["press_quote"] = quote[:150]

    if os.path.exists(rot_path):
        with open(rot_path, encoding="utf-8") as f:
            rot_data = json.load(f)
        for gw_str, gw_data in rot_data.get("gameweeks", {}).items():
            gw = int(gw_str)
            intel.setdefault(gw, {})
            for pid_str, p in gw_data.get("players", {}).items():
                pid = int(pid_str)
                intel[gw].setdefault(pid, {})
                intel[gw][pid]["rotation_risk"] = p.get("rotation_risk", 0.0)
                intel[gw][pid]["rot_tier"]      = p.get("rotation_tier", "low_risk")
                intel[gw][pid]["start_rate"]    = p.get("start_rate")
                factors = p.get("contributing_factors", [])
                if factors:
                    intel[gw][pid]["factors"] = "; ".join(factors[:2])

    return intel


# ===========================================================================
# Build prompt for a single GW
# ===========================================================================
def build_prompt(gw_entry, team_map, intel, running_actual, running_predicted, fixture_ctx=None):
    gw            = gw_entry["gw"]
    chip          = gw_entry.get("chip")
    penalty       = gw_entry.get("penalty_pts", 0)
    transfers_in  = gw_entry.get("transfers_in", [])
    transfers_out = gw_entry.get("transfers_out", [])
    bank          = gw_entry.get("bank", 0)
    squad_value   = gw_entry.get("squad_value", 0)
    free_transfers = gw_entry.get("free_transfers", 1)

    xi    = gw_entry.get("xi", [])
    bench = gw_entry.get("bench", [])

    gw_actual    = gw_entry.get("actual_total",
        gw_entry.get("actual_pts",
            sum(p.get("pts_counted", 0) for p in xi) + (penalty or 0)))
    gw_predicted = gw_entry.get("predicted_total",
        sum(p.get("predicted_pts", 0) for p in xi))

    # Intel for this GW
    gw_intel = intel.get(gw, {})

    # --- Squad context line
    squad_ctx = (
        f"Bank: £{bank:.1f}m  |  Squad value: £{squad_value:.1f}m  "
        f"|  Free transfers: {free_transfers}"
    )

    # --- Chip line
    chip_line = f"Chip played: {chip.upper()}" if chip else "No chip"

    # --- Transfers block
    if transfers_in:
        ins  = ", ".join(transfers_in)
        outs = ", ".join(transfers_out)
        pen_str = f" (-{abs(penalty)} pts hit)" if penalty else " (free)"
        transfer_line = f"IN: {ins}  /  OUT: {outs}{pen_str}"
    else:
        transfer_line = "No transfers" if gw > 1 else "Fresh squad (GW1)"

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    gw_fix = (fixture_ctx or {}).get(gw, {})

    def fdr_label(fdr):
        return {1: "very easy", 2: "easy", 3: "medium", 4: "hard", 5: "very hard"}.get(fdr, "?")

    # --- XI table with fixture context
    header  = f"{'Pos':<4} {'Player':<22} {'Club':<5} {'£m':>4}  {'Pred':>5}  {'Act':>5}  {'Fixture':<26}  Role"
    divider = "-" * 82

    rows = []
    for p in sorted(xi, key=lambda x: pos_order.get(x["pos"], 9)):
        team_str = team_map.get(p.get("team", 0), f"t{p.get('team','?')}")
        role = ""
        cap_mult = p.get("captain_multiplier", 1)
        if cap_mult == 3:   role = "[TC]"
        elif p.get("is_captain"): role = "[C]"
        pred  = p.get("predicted_pts", 0)
        price = p.get("price", 0)

        fix = gw_fix.get(p.get("team", 0), {})
        opp  = fix.get("opponent_name", "?")
        home = "H" if fix.get("is_home") else "A"
        fdr  = fix.get("fdr", 3)
        fix_str = f"vs {opp} ({home}) FDR:{fdr}/5 {fdr_label(fdr)}"

        rows.append(
            f"{p['pos']:<4} {p['web_name']:<22} {team_str:<5} "
            f"{price:>4.1f}  {pred:>5.1f}  {p.get('actual_pts',0):>5}  "
            f"{fix_str:<26}  {role}"
        )

    # --- Bench (for comparison — why didn't these start?)
    bench_sorted = sorted(bench, key=lambda x: x.get("predicted_pts", 0), reverse=True)
    bench_lines = []
    for p in bench_sorted:
        pred  = p.get("predicted_pts", 0)
        price = p.get("price", 0)
        bench_lines.append(
            f"  {p['pos']:<4} {p['web_name']:<22} £{price:.1f}m  pred:{pred:.1f}  actual:{p.get('actual_pts',0)}"
        )

    # --- Position group averages (for context)
    pos_groups = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in xi:
        pos_groups[p["pos"]].append(p.get("predicted_pts", 0))
    pos_avg_lines = []
    for pos, vals in pos_groups.items():
        if vals:
            pos_avg_lines.append(f"  {pos}: avg predicted {sum(vals)/len(vals):.1f} pts (n={len(vals)})")

    # --- Captain alternatives (top 4 by model prediction)
    xi_by_pred = sorted(xi, key=lambda x: x.get("predicted_pts", 0), reverse=True)
    alt_lines = []
    for rank, p in enumerate(xi_by_pred[:4], 1):
        marker = "[TC]" if p.get("captain_multiplier", 1) == 3 else \
                 "[C]"  if p.get("is_captain") else ""
        alt_lines.append(
            f"  {rank}. {p['web_name']} {marker:<5} £{p.get('price',0):.1f}m  "
            f"pred:{p.get('predicted_pts',0):.1f}  actual:{p.get('actual_pts',0)}"
        )

    # --- Intel flags ONLY for genuinely notable signals
    flags = []
    for p in xi + bench:
        pid     = p.get("player_id", 0)
        p_intel = gw_intel.get(pid, {})
        avail   = p_intel.get("avail_pct", 100)
        rot     = p_intel.get("rotation_risk", 0.0)
        tier    = p_intel.get("avail_tier", "available")
        quote   = p_intel.get("press_quote", "")
        if avail < 90 or tier in ("doubtful", "likely_out", "unavailable"):
            flag = f"  {p['web_name']}: {avail}% availability ({tier})"
            if quote:
                flag += f' — "{quote[:100]}"'
            flags.append(flag)
        elif rot > 40:
            factors = p_intel.get("factors", "")
            flag = f"  {p['web_name']}: {rot:.0f}% rotation risk"
            if factors:
                flag += f" ({factors})"
            flags.append(flag)

    intel_block = "\n".join(flags) if flags else "  None."

    return (
        f"Season: 2025/26 Premier League  |  Gameweek: {gw}\n"
        f"{chip_line}  |  GW score: {gw_actual} pts (model predicted: {gw_predicted:.1f})\n"
        f"Season running total (before this GW): {running_actual} pts\n"
        f"Squad: bank £{bank:.1f}m  |  value £{squad_value:.1f}m  |  FT: {free_transfers}\n\n"
        f"TRANSFERS\n{transfer_line}\n\n"
        f"STARTING XI — ILP optimizer selection\n"
        f"(£/pred = price divided by predicted pts — lower is better value)\n"
        f"{header}\n{divider}\n"
        + "\n".join(rows)
        + f"\n\nBENCH (did not start — lower model predictions)\n"
        + "\n".join(bench_lines)
        + f"\n\nPOSITION GROUP AVERAGES (predicted pts)\n"
        + "\n".join(pos_avg_lines)
        + f"\n\nCAPTAIN DECISION — top 4 by model prediction:\n"
        + "\n".join(alt_lines)
        + (f"\n\nINTEL FLAGS (injury/rotation concerns this GW):\n" + intel_block if flags else "")
        + "\n\n---\n"
        "TASK: For each of the 11 starting players, write 2-3 sentences explaining "
        "why the LightGBM model ranked them highly and why the ILP optimizer picked them. "
        "Use the data in the tables: predicted pts rank within their position group, "
        "price, fixture (opponent, home/away, FDR). For GK and DEF, reference the "
        "fixture difficulty — an easy fixture (FDR 1-2) means a higher clean sheet "
        "probability which boosts their predicted score. For MID and FWD, mention if "
        "they face a weak defence (low FDR opponent) that inflates scoring probability. "
        "Also compare to bench players at the same position where relevant. "
        "Then write 2-3 sentences on the captain/TC choice vs the listed alternatives.\n\n"
        "Respond with ONLY valid JSON:\n"
        "{\n"
        '  "players": {\n'
        '    "ExactWebName": "2-3 sentences.",\n'
        '    ...\n'
        "  },\n"
        '  "captain_note": "2-3 sentences."\n'
        "}\n"
        "Use exact web_name values from the table above. Cover all 11 XI players."
    )


# ===========================================================================
# LLM call (with retry)
# ===========================================================================
def explain_gw(client, gw_entry, team_map, intel, running_actual, running_predicted, fixture_ctx=None):
    """Call Claude to explain picks for one GW. Returns dict with players + captain_note."""
    user_msg = build_prompt(gw_entry, team_map, intel, running_actual, running_predicted, fixture_ctx)
    fallback = {"players": {}, "captain_note": ""}

    delays = [0, 5, 15, 30]
    last_err = None

    for attempt, delay in enumerate(delays):
        if delay:
            print(f"  [RETRY] Waiting {delay}s before attempt {attempt + 1}...")
            time.sleep(delay)
        try:
            resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                temperature=TEMP,
                system=SYS_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            # Normalise key: accept both "captain" and "captain_note"
            if "captain" in parsed and "captain_note" not in parsed:
                parsed["captain_note"] = parsed.pop("captain")
            return parsed
        except Exception as e:
            last_err = e
            err_str = str(e)
            retryable = any(code in err_str for code in ["529", "503", "429", "overloaded"])
            if retryable and attempt < len(delays) - 1:
                print(f"  [WARN] API rate-limit/overloaded (attempt {attempt + 1}): {err_str[:80]}")
                continue
            if "json" in err_str.lower() or "parse" in err_str.lower():
                print(f"  [WARN] JSON parse error: {err_str[:80]}")
                return fallback
            raise

    print(f"  [ERROR] All retries exhausted: {last_err}")
    return fallback


# ===========================================================================
# Print helper
# ===========================================================================
def _wrap(text, width=74, indent="  "):
    words = text.split()
    lines, line = [], ""
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


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(SEP)
    print("  FPL AI Stage 9 -- LLM Pick Explanations (Rich Mode)")
    print(SEP)

    print(f"\nLoading simulation: {SIM_JSON}")
    if not os.path.exists(SIM_JSON):
        print(f"  ERROR: {SIM_JSON} not found. Run season_simulator.py first.")
        sys.exit(1)

    with open(SIM_JSON, encoding="utf-8") as f:
        sim = json.load(f)

    gameweeks       = sim["gameweeks"]
    total_actual    = sim.get("total_actual_pts", 0)
    total_predicted = sim.get("total_predicted_pts", 0.0)
    chips_used      = {c["chip"]: c["gw"] for c in sim.get("chips_used", [])}

    print(f"  Loaded {len(gameweeks)} gameweeks")
    print(f"  Season total: {total_actual} actual pts  ({total_predicted:.1f} predicted)")
    print(f"  Chips used: {chips_used}")

    team_map = load_team_map()
    print(f"  Team map: {len(team_map)} teams loaded")

    print("\nLoading intel (availability + rotation risk)...")
    intel = load_intel()

    print("Loading fixture context (opponents, home/away, FDR)...")
    fixture_ctx = load_fixture_context()
    print(f"  Fixture context loaded for {len(fixture_ctx)} GWs")
    gws_with_intel = sum(1 for g in intel if intel[g])
    print(f"  Intel loaded for {gws_with_intel} GWs")

    print("\nInitialising Anthropic client...")
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        raise ImportError("anthropic not installed. Run: pip install anthropic")
    print(f"  Client ready  model: {MODEL_ID}  max_tokens: {MAX_TOKENS}")

    running_actual    = 0
    running_predicted = 0.0
    results = []

    print(f"\nGenerating explanations ({len(gameweeks)} API calls)...\n")

    for gw_entry in gameweeks:
        gw = gw_entry["gw"]

        gw_actual = gw_entry.get("actual_total",
            gw_entry.get("actual_pts",
                sum(p.get("pts_counted", 0) for p in gw_entry.get("xi", []))
                + (gw_entry.get("penalty_pts", 0) or 0)
            ))
        gw_pred = gw_entry.get("predicted_total",
            sum(p.get("predicted_pts", 0) for p in gw_entry.get("xi", [])))

        print(f"  GW{gw:<2}  actual:{gw_actual}  predicted:{gw_pred:.1f}", end="  ")

        explanation = explain_gw(
            client, gw_entry, team_map, intel, running_actual, running_predicted, fixture_ctx
        )

        player_expl  = explanation.get("players", {})
        captain_note = explanation.get("captain_note", "")

        if player_expl:
            print(f"OK ({len(player_expl)} players)")
        else:
            print("FAILED (no explanation)")

        running_actual    += gw_actual
        running_predicted += gw_pred

        # Print summary
        print()
        print(f"  GW{gw} -- WHY THESE PICKS")
        print("  " + "-" * 56)
        for name, text in player_expl.items():
            print(_wrap(f"{name}: {text}"))
        if captain_note:
            print(_wrap(f"Captain analysis: {captain_note}"))
        print()

        results.append({
            "gw":                  gw,
            "actual_pts":          gw_actual,
            "predicted_pts":       round(gw_pred, 1),
            "chip":                gw_entry.get("chip"),
            "player_explanations": player_expl,
            "captain_note":        captain_note,
        })

        if gw < len(gameweeks):
            time.sleep(DELAY_S)

    output = {
        "model":               MODEL_ID,
        "max_tokens":          MAX_TOKENS,
        "source":              SIM_JSON,
        "total_actual_pts":    total_actual,
        "total_predicted_pts": round(total_predicted, 1),
        "gameweeks":           results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(SEP)
    print(f"  Done. {len(results)} GW explanations saved.")
    print(f"  Output: {OUT_JSON}")
    print(SEP)


if __name__ == "__main__":
    main()
