"""
Stage 9: LLM Agent Layer
FPL AI Thesis -- Claude API post-analysis of the finished season simulation.

Reads data/intel/season_simulation.json (produced by season_simulator.py)
and calls Claude ONCE per gameweek to explain why each player was picked.
No decisions are made -- the simulation results are read-only.

Total API calls: 1 per GW (28 calls for a full season, 10 for GW1-10).
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
MODEL_ID   = "claude-sonnet-4-20250514"
MAX_TOKENS = 1200
TEMP       = 0
DELAY_S    = 1.0   # seconds between API calls

SEP = "=" * 60


# ===========================================================================
# Anthropic client
# ===========================================================================
def get_client():
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        raise ImportError("anthropic not installed. Run: pip install anthropic")


# ===========================================================================
# Load team id -> short name from teams_raw.csv
# ===========================================================================
def load_team_map():
    """Returns dict {team_id: short_name}. Falls back to empty dict."""
    teams_path = os.path.join(DATA_RAW, "teams_raw.csv")
    try:
        import pandas as pd
        df = pd.read_csv(teams_path)
        # short_name column in FPL teams CSV
        col = "short_name" if "short_name" in df.columns else "name"
        return {int(row["id"]): str(row[col]) for _, row in df.iterrows()}
    except Exception:
        return {}


# ===========================================================================
# Build prompt for a single GW
# ===========================================================================
def build_prompt(gw_entry, team_map, total_actual, total_predicted):
    gw          = gw_entry["gw"]
    chip        = gw_entry.get("chip")
    penalty     = gw_entry.get("penalty_pts", 0)
    transfers_in  = gw_entry.get("transfers_in", [])
    transfers_out = gw_entry.get("transfers_out", [])

    xi    = gw_entry.get("xi", [])
    bench = gw_entry.get("bench", [])

    # Build XI table
    lines = [
        f"{'Pos':<4} {'Player':<26} {'Club':<5} {'Price':>5}  {'Pred':>5}  {'Actual':>6}  Role"
    ]
    lines.append("-" * 68)

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    for p in sorted(xi, key=lambda x: pos_order.get(x["pos"], 9)):
        team_str = team_map.get(p.get("team", 0), f"t{p.get('team','?')}")
        role = "[C]" if p.get("is_captain") else ""
        cap_mult = p.get("captain_multiplier", 1)
        if cap_mult == 3:
            role = "[TC]"
        lines.append(
            f"{p['pos']:<4} {p['web_name']:<26} {team_str:<5} "
            f"{p['price']:>5.1f}  {p['predicted_pts']:>5.1f}  "
            f"{p['actual_pts']:>6}  {role}"
        )

    bench_names = ", ".join(p["web_name"] for p in bench)

    # Transfers block
    if transfers_in:
        ins  = ", ".join(transfers_in)
        outs = ", ".join(transfers_out)
        pen_str = f" (-{penalty} pts hit)" if penalty else " (free)"
        transfer_line = f"IN: {ins}  /  OUT: {outs}{pen_str}"
    else:
        transfer_line = "No transfers" if gw > 1 else "Fresh squad (GW1)"

    chip_line = f"Chip played: {chip.upper()}" if chip else "No chip"

    gw_actual    = gw_entry.get("actual_pts", sum(p["pts_counted"] for p in xi) - penalty)
    gw_predicted = gw_entry.get("predicted_pts", sum(p["predicted_pts"] for p in xi))

    return (
        f"Season: 2025/26 Premier League\n"
        f"Gameweek: {gw}\n"
        f"{chip_line}\n"
        f"Transfers: {transfer_line}\n"
        f"GW score: {gw_actual} actual pts  (predicted: {gw_predicted:.1f})\n"
        f"Season total so far: {total_actual} actual pts  (predicted: {total_predicted:.1f})\n\n"
        f"Starting XI selected by the ILP optimizer:\n"
        + "\n".join(lines)
        + f"\n\nBench: {bench_names}\n\n"
        "For each of the 11 starting players, write exactly one sentence explaining "
        "the primary reason the ILP optimizer selected them (predicted score, form, "
        "fixture difficulty, price value, or positional constraint). "
        "Also write one sentence explaining the captain choice. "
        "Keep it factual and concise. Do not suggest any changes.\n\n"
        "Respond with ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "players": {\n'
        '    "ExactWebName": "One sentence.",\n'
        '    ...\n'
        "  },\n"
        '  "captain": "One sentence about the captain choice."\n'
        "}\n"
        "Use the exact web_name values from the table above as keys."
    )


# ===========================================================================
# LLM call
# ===========================================================================
SYS_PROMPT = (
    "You are an expert FPL analyst providing post-hoc explanations of squad "
    "selections made by a mathematical ILP optimizer for the 2025/26 Premier "
    "League season. Your only job is to explain the picks as they stand -- "
    "do not suggest changes, do not critique decisions, just explain the reasoning "
    "behind each selection based on the data provided. "
    "Return your response as valid JSON only, no prose outside the JSON block."
)


def explain_gw(client, gw_entry, team_map, total_actual, total_predicted):
    """
    Call Claude to explain picks for one GW.
    Returns dict: {"players": {web_name: sentence}, "captain": sentence}
    Falls back to empty dict on any API error or parse failure.
    """
    user_msg = build_prompt(gw_entry, team_map, total_actual, total_predicted)
    fallback = {"players": {}, "captain": ""}
    try:
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            temperature=TEMP,
            system=SYS_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"  [WARN] API call failed or parse error: {e}")
        return fallback


# ===========================================================================
# Print helper
# ===========================================================================
def _wrap(text, width=72, indent="  "):
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
    print("  FPL AI Stage 9 -- LLM Pick Explanations")
    print(SEP)

    # Load simulation results
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

    # Load team names
    team_map = load_team_map()
    print(f"  Team map: {len(team_map)} teams loaded")

    # Anthropic client
    print("\nInitialising Anthropic client...")
    client = get_client()
    print(f"  Client ready  model: {MODEL_ID}")

    # Running season totals for context in each prompt
    running_actual    = 0
    running_predicted = 0.0

    results = []

    print(f"\nGenerating explanations ({len(gameweeks)} API calls)...\n")

    for gw_entry in gameweeks:
        gw = gw_entry["gw"]

        # Compute GW totals from xi if not stored directly
        gw_actual = gw_entry.get("actual_pts",
            sum(p["pts_counted"] for p in gw_entry.get("xi", []))
            - gw_entry.get("penalty_pts", 0)
        )
        gw_pred = gw_entry.get("predicted_pts",
            sum(p["predicted_pts"] for p in gw_entry.get("xi", []))
        )

        print(f"  GW{gw:<2}  actual:{gw_actual}  predicted:{gw_pred:.1f}", end="  ")

        explanation = explain_gw(client, gw_entry, team_map, running_actual, running_predicted)

        player_expl  = explanation.get("players", {})
        captain_note = explanation.get("captain", "")

        if player_expl:
            print(f"OK ({len(player_expl)} players)")
        else:
            print("FAILED (no explanation)")

        running_actual    += gw_actual
        running_predicted += gw_pred

        # Print explanation
        print()
        print(f"  GW{gw} -- WHY THESE PICKS")
        print("  " + "-" * 50)
        for name, sentence in player_expl.items():
            print(_wrap(f"{name}: {sentence}"))
        if captain_note:
            print(_wrap(f"Captain: {captain_note}"))
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

    # Save output
    output = {
        "model":            MODEL_ID,
        "source":           SIM_JSON,
        "total_actual_pts": total_actual,
        "total_predicted_pts": round(total_predicted, 1),
        "gameweeks":        results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(SEP)
    print(f"  Done. {len(results)} GW explanations saved.")
    print(f"  Output: {OUT_JSON}")
    print(f"  Total API calls: {len(results)}")
    print(SEP)


if __name__ == "__main__":
    main()
