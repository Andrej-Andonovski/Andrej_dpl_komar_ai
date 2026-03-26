"""
pipeline/season_report.py
Reads data/intel/season_simulation.json and prints a full season report.
Also computes Monte Carlo baseline (3 random squads) and hindsight optimal.
Fully standalone.
"""
import os, json, random, sys
import numpy as np
import pandas as pd
from collections import defaultdict

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                  LpBinary, LpStatus, PULP_CBC_CMD)

random.seed(42)
np.random.seed(42)

_HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(_HERE, "..", "data")
SIM_JSON     = os.path.join(DATA_DIR, "intel", "season_simulation.json")
HIST_CSV     = os.path.join(DATA_DIR, "raw", "fpl_api", "player_history.csv")
PLAYERS_CSV  = os.path.join(DATA_DIR, "raw", "fpl_api", "players_raw.csv")
FIXTURES_CSV = os.path.join(DATA_DIR, "raw", "fpl_api", "fixtures_raw.csv")

BUDGET   = 100.0
MAX_CLUB = 3
MC_N     = 3   # increase to 500 overnight


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sim():
    with open(SIM_JSON) as f:
        return json.load(f)


def load_actuals_and_meta():
    """Returns {pid: {gw: {total_points, minutes, value}}}, players_df, fdr_lookup"""
    hist_df = pd.read_csv(HIST_CSV)
    hist = defaultdict(dict)
    for r in hist_df.itertuples(index=False):
        hist[int(r.player_id)][int(r.gameweek)] = {
            "total_points": float(r.total_points),
            "minutes":      int(r.minutes),
            "value":        float(r.value),
        }

    players_df = pd.read_csv(PLAYERS_CSV)
    if "price" not in players_df.columns:
        players_df["price"] = players_df["now_cost"] / 10.0
    if "position" not in players_df.columns:
        POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        players_df["position"] = players_df["element_type"].map(POS_MAP)
    if "web_name" not in players_df.columns:
        players_df["web_name"] = players_df["second_name"]

    fix_df = pd.read_csv(FIXTURES_CSV)
    fix_df = fix_df.dropna(subset=["gameweek"])
    fix_df["gameweek"] = fix_df["gameweek"].astype(int)
    fdr_lookup  = {}
    home_lookup = {}
    for r in fix_df.itertuples(index=False):
        gw = int(r.gameweek)
        fdr_lookup[(int(r.team_h), gw)] = int(r.team_h_difficulty)
        fdr_lookup[(int(r.team_a), gw)] = int(r.team_a_difficulty)
        home_lookup[(int(r.team_h), gw)] = 1
        home_lookup[(int(r.team_a), gw)] = 0

    return hist, players_df, fdr_lookup, home_lookup


def sep(char="-", n=70):
    print(char * n)


# ── GW-by-GW Breakdown ───────────────────────────────────────────────────────

def print_gw_breakdown(sim):
    for gw_data in sim["gameweeks"]:
        gw    = gw_data["gw"]
        chip  = gw_data.get("chip") or "—"
        ft    = gw_data.get("free_transfers", "?")
        bank  = gw_data.get("bank", 0)
        sqv   = gw_data.get("squad_value", 0)
        pen   = gw_data.get("penalty_pts", 0)
        pred  = gw_data.get("predicted_total", 0)
        act   = gw_data.get("actual_total", 0)
        diff  = act - pred
        t_in  = gw_data.get("transfers_in", [])
        t_out = gw_data.get("transfers_out", [])

        cap_info = gw_data.get("captain", {})
        cap_name = cap_info.get("web_name", gw_data.get("captain_id", "?"))
        cap_pts  = cap_info.get("actual_pts", "?")
        cap_pred = cap_info.get("predicted_pts", "?")
        cap_src  = cap_info.get("source", "")
        cap_src_str = f" [{cap_src}]" if cap_src and cap_src != "ilp" else ""

        sep()
        print(f"GW {gw:2d}  [Chip: {chip:12s}]  FT: {ft}  "
              f"Bank: GBP{bank:.1f}m  Squad val: GBP{sqv:.1f}m")
        print(f"Captain: {cap_name}  (pred: {cap_pred}  actual: {cap_pts}){cap_src_str}")
        print("STARTING XI:")
        for p in gw_data.get("xi", []):
            cap_tag = f" [C x{p['captain_multiplier']}]" if p["is_captain"] else ""
            print(f"  {p['web_name']:<18s} {p['pos']:3s}  "
                  f"GBP{p['price']:.1f}m  "
                  f"pred:{p['predicted_pts']:5.1f}  "
                  f"actual:{p['actual_pts']:3d}  "
                  f"counted:{p['pts_counted']:3d}{cap_tag}")
        print("BENCH:")
        for p in gw_data.get("bench", []):
            print(f"  {p['web_name']:<18s} {p['pos']:3s}  "
                  f"GBP{p['price']:.1f}m  actual:{p['actual_pts']:3d}")

        if gw_data.get("auto_subs"):
            for sub in gw_data["auto_subs"]:
                print(f"  AUTO-SUB: {sub[1]} ON for {sub[0]}")

        t_str = ""
        if t_in or t_out:
            t_str = f"  IN: {t_in}  OUT: {t_out}"
        print(f"Transfers:{t_str}  | Penalty: {pen} pts")
        print(f"Predicted: {pred:.1f}  |  Actual: {act}  |  Diff: {diff:+.1f}")


# ── Season Totals ─────────────────────────────────────────────────────────────

def print_season_totals(sim):
    gws     = sim["gameweeks"]
    total_a = sim["total_actual_pts"]
    total_p = sim["total_predicted_pts"]
    total_n = sim["total_penalties"]

    sep("=")
    print("  SEASON TOTALS")
    sep("=")
    print(f"  Total actual points:     {total_a}")
    print(f"  Total predicted points:  {total_p:.1f}")
    print(f"  Prediction error:        {total_a - total_p:+.1f}")
    print(f"  Total penalties:         {total_n}")
    print(f"  Net total:               {total_a + total_n}")
    print()

    # GW scores table
    print(f"  {'GW':>3}  {'Chip':<16}  {'FT':>2}  {'Pred':>6}  "
          f"{'Actual':>6}  {'Diff':>6}  {'Running':>7}")
    sep("-", 70)
    running = 0
    for gd in gws:
        running += gd["actual_total"]
        chip_s   = gd.get("chip") or "—"
        diff     = gd["actual_total"] - gd["predicted_total"]
        print(f"  {gd['gw']:>3}  {chip_s:<16}  {gd.get('free_transfers',0):>2}  "
              f"{gd['predicted_total']:>6.1f}  {gd['actual_total']:>6}  "
              f"{diff:>+6.1f}  {running:>7}")

    actuals = [gd["actual_total"] for gd in gws]
    best_idx = int(np.argmax(actuals))
    worst_idx = int(np.argmin(actuals))
    print()
    best_gd  = gws[best_idx]
    worst_gd = gws[worst_idx]
    print(f"  Best GW:  GW{best_gd['gw']} — {best_gd['actual_total']} pts"
          f"  [Chip: {best_gd.get('chip') or 'None'}]")
    print(f"  Worst GW: GW{worst_gd['gw']} — {worst_gd['actual_total']} pts")


# ── Chips ─────────────────────────────────────────────────────────────────────

def print_chips(sim):
    print()
    print("  CHIPS USED:")
    used = sim.get("chips_used", [])
    set1 = [c for c in used if c["chip"].endswith("1")]
    set2 = [c for c in used if c["chip"].endswith("2")]
    set1_names = [c["chip"] for c in set1]
    set2_names = [c["chip"] for c in set2]
    gws_played = {gd["gw"] for gd in sim["gameweeks"]}

    all_set1 = ["wc1", "fh1", "tc1", "bb1"]
    wasted   = [c for c in all_set1 if c not in set1_names and max(gws_played) >= 19]

    for c in set1:
        print(f"    Set 1: {c['chip'].upper():<8}  GW{c['gw']}")
    for c in set2:
        print(f"    Set 2: {c['chip'].upper():<8}  GW{c['gw']}")
    if wasted:
        print(f"    Set 1 chips WASTED (unused by GW19): {wasted}")
    else:
        print(f"    Set 1 chips wasted: None")


# ── Learning Curve ────────────────────────────────────────────────────────────

def print_learning_curve(sim):
    print()
    print("  LEARNING CURVE (model MAE per GW):")
    print(f"  {'GW':>4}  {'MAE':>6}  {'vs prev':>8}")
    sep("-", 30)
    curve = sim.get("learning_curve", [])
    prev_mae = None
    for entry in curve:
        gw  = entry["gw"]
        mae = entry["mae"]
        if prev_mae is None:
            delta = "baseline"
        else:
            delta = f"{mae - prev_mae:+.3f}"
        print(f"  {gw:>4}  {mae:>6.3f}  {delta:>8}")
        prev_mae = mae
    if curve:
        total_imp = curve[-1]["mae"] - curve[0]["mae"]
        print(f"  Total MAE change: {total_imp:+.3f}")


# ── Monte Carlo Baseline ──────────────────────────────────────────────────────

def _random_squad(players, available_budget):
    """Pick a random valid 15-man squad within budget and formation rules."""
    POS_SLOTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for _ in range(5000):
        random.shuffle(players)
        squad = []
        counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        club_cnt = defaultdict(int)
        cost = 0.0
        for p in players:
            pos   = p["pos"]
            team  = p["team"]
            price = p["price"]
            if (counts[pos] < POS_SLOTS[pos] and
                    club_cnt[team] < MAX_CLUB and
                    cost + price <= available_budget + 0.01):
                squad.append(p)
                counts[pos] += 1
                club_cnt[team] += 1
                cost += price
            if sum(counts.values()) == 15:
                break
        if sum(counts.values()) == 15:
            return squad
    return None


def _best_xi(squad):
    """Pick best XI by position rules, captain = highest actual pts."""
    by_pos = defaultdict(list)
    for p in squad:
        by_pos[p["pos"]].append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: -x.get("actual_pts", 0))

    xi = []
    # 1 GK
    if by_pos["GK"]:
        xi.append(by_pos["GK"][0])
    # Try 4-4-2 then 3-5-2 then 3-4-3 etc.
    outfield_sorted = sorted(
        [p for pos in ["DEF", "MID", "FWD"] for p in by_pos[pos]],
        key=lambda x: -x.get("actual_pts", 0)
    )
    # Ensure min formation
    selected = set()
    for pos, mn in [("DEF", 3), ("MID", 2), ("FWD", 1)]:
        added = 0
        for p in by_pos[pos]:
            if p["player_id"] not in selected and added < mn:
                xi.append(p); selected.add(p["player_id"]); added += 1

    # Fill remaining 6 spots
    remaining = 11 - len(xi)
    for p in outfield_sorted:
        if remaining <= 0:
            break
        if p["player_id"] not in selected:
            # Check formation still valid
            test_defs = sum(1 for x in xi if x["pos"] == "DEF") + (1 if p["pos"] == "DEF" else 0)
            test_mids = sum(1 for x in xi if x["pos"] == "MID") + (1 if p["pos"] == "MID" else 0)
            test_fwds = sum(1 for x in xi if x["pos"] == "FWD") + (1 if p["pos"] == "FWD" else 0)
            if test_defs <= 5 and test_mids <= 5 and test_fwds <= 3:
                xi.append(p); selected.add(p["player_id"]); remaining -= 1

    if len(xi) < 11:
        return None, None
    captain = max((p for p in xi if p["pos"] != "GK"),
                  key=lambda x: x.get("actual_pts", 0), default=xi[0])
    return xi, captain


def run_monte_carlo(sim, hist, players_df):
    print()
    print(f"  MONTE CARLO BASELINE ({MC_N} random squads per GW):")
    gws = sim["gameweeks"]
    if not gws:
        return

    # Build player pool once (with actual points per GW)
    players_meta = []
    for r in players_df.itertuples(index=False):
        players_meta.append({
            "player_id": int(r.id),
            "pos": str(r.position),
            "team": int(r.team),
            "price": float(r.price),
        })

    mc_totals = [0.0] * MC_N
    for gw_data in gws:
        gw = gw_data["gw"]
        avail = BUDGET

        for pm in players_meta:
            pm["actual_pts"] = hist.get(pm["player_id"], {}).get(gw, {}).get("total_points", 0)
            # Update price
            pm["price"] = hist.get(pm["player_id"], {}).get(max(1, gw - 1), {}).get(
                "value", pm["price"])

        for k in range(MC_N):
            squad = _random_squad(list(players_meta), avail)
            if squad is None:
                continue
            xi, captain = _best_xi(squad)
            if xi is None:
                continue
            gw_score = sum(
                p["actual_pts"] * 2 if p["player_id"] == captain["player_id"]
                else p["actual_pts"]
                for p in xi
            )
            mc_totals[k] += gw_score

    if mc_totals:
        mc_mean = np.mean(mc_totals)
        mc_std  = np.std(mc_totals)
        our     = sim["total_actual_pts"]
        print(f"    Random squads total (mean of {MC_N}): {mc_mean:.0f} pts")
        print(f"    Random std dev:                       {mc_std:.0f} pts")
        print(f"    Our optimizer total:                  {our} pts")
        print(f"    Outperformance vs mean:               {our - mc_mean:+.0f} pts")
        if mc_std > 0:
            pct = sum(1 for t in mc_totals if t <= our) / MC_N * 100
            print(f"    Our rank among {MC_N} random squads:   top {100-pct:.0f}%")


# ── Hindsight Optimal ─────────────────────────────────────────────────────────

def _hindsight_ilp(players, budget):
    """ILP with actual points as objective. Returns score or None."""
    idx   = list(range(len(players)))
    pred  = [p["actual_pts"] for p in players]
    price = [p["price"]      for p in players]
    pos   = [p["element_type"] for p in players]
    team  = [p["team"]       for p in players]

    prob = LpProblem("hindsight", LpMaximize)
    x = [LpVariable(f"x{i}", cat=LpBinary) for i in idx]
    s = [LpVariable(f"s{i}", cat=LpBinary) for i in idx]
    c = [LpVariable(f"c{i}", cat=LpBinary) for i in idx]

    prob += (lpSum(pred[i] * s[i] for i in idx)
             + lpSum(pred[i] * c[i] for i in idx))

    prob += lpSum(x) == 15
    prob += lpSum(x[i] for i in idx if pos[i] == 1) == 2
    prob += lpSum(x[i] for i in idx if pos[i] == 2) == 5
    prob += lpSum(x[i] for i in idx if pos[i] == 3) == 5
    prob += lpSum(x[i] for i in idx if pos[i] == 4) == 3
    prob += lpSum(price[i] * x[i] for i in idx) <= budget

    for cl in set(team):
        prob += lpSum(x[i] for i in idx if team[i] == cl) <= MAX_CLUB

    prob += lpSum(s) == 11
    prob += lpSum(s[i] for i in idx if pos[i] == 1) == 1
    prob += lpSum(s[i] for i in idx if pos[i] == 2) >= 3
    prob += lpSum(s[i] for i in idx if pos[i] == 2) <= 5
    prob += lpSum(s[i] for i in idx if pos[i] == 3) >= 2
    prob += lpSum(s[i] for i in idx if pos[i] == 3) <= 5
    prob += lpSum(s[i] for i in idx if pos[i] == 4) >= 1
    prob += lpSum(s[i] for i in idx if pos[i] == 4) <= 3
    for i in idx:
        prob += s[i] <= x[i]

    prob += lpSum(c) == 1
    prob += lpSum(c[i] for i in idx if pos[i] == 1) == 0
    for i in idx:
        prob += c[i] <= s[i]

    prob.solve(PULP_CBC_CMD(msg=0))
    if LpStatus[prob.status] == "Optimal":
        xi_score = sum(pred[i] * s[i].value() for i in idx if s[i].value() and s[i].value() > 0.5)
        cap_bonus = sum(pred[i] * c[i].value() for i in idx if c[i].value() and c[i].value() > 0.5)
        return xi_score + cap_bonus
    return None


def run_hindsight(sim, hist, players_df):
    print()
    print("  HINDSIGHT OPTIMAL (perfect information, 28 ILP solves):")
    players_meta = []
    for r in players_df.itertuples(index=False):
        players_meta.append({
            "player_id":   int(r.id),
            "element_type": int(r.element_type),
            "team":        int(r.team),
            "price":       float(r.price),
            "actual_pts":  0.0,
        })

    gws = sim["gameweeks"]
    total_hindsight = 0.0
    for gw_data in gws:
        gw = gw_data["gw"]
        # Update prices and actuals
        for pm in players_meta:
            pid = pm["player_id"]
            pm["actual_pts"] = hist.get(pid, {}).get(gw, {}).get("total_points", 0.0)
            prev_val = hist.get(pid, {}).get(max(1, gw - 1), {}).get("value")
            if prev_val:
                pm["price"] = prev_val

        score = _hindsight_ilp(players_meta, BUDGET)
        if score is not None:
            total_hindsight += score
        print(f"    GW{gw:2d} hindsight optimal: {score:.1f}" if score else f"    GW{gw} ILP failed")

    our   = sim["total_actual_pts"]
    eff   = our / total_hindsight * 100 if total_hindsight > 0 else 0
    print(f"\n    Theoretical max total: {total_hindsight:.0f} pts")
    print(f"    Our optimizer total:   {our} pts")
    print(f"    Efficiency:            {eff:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FULL SEASON REPORT — GW1 to GW28")
    print("=" * 70)

    sim = load_sim()
    gws = sim.get("gameweeks", [])
    if not gws:
        print("No gameweek data found in JSON.")
        return

    gw_range = f"GW{gws[0]['gw']} to GW{gws[-1]['gw']}"
    print(f"  Generated: {sim.get('generated_at', 'unknown')}")
    print(f"  GW range:  {gw_range}")
    print()

    print_gw_breakdown(sim)

    sep("=")
    print_season_totals(sim)
    print_chips(sim)
    print_learning_curve(sim)

    # Monte Carlo and hindsight require raw data
    print("\n[LOADING] Raw data for Monte Carlo / Hindsight...")
    hist, players_df, fdr_lookup, home_lookup = load_actuals_and_meta()

    run_monte_carlo(sim, hist, players_df)
    run_hindsight(sim, hist, players_df)

    sep("=")
    print("  END OF REPORT")
    sep("=")


if __name__ == "__main__":
    main()
