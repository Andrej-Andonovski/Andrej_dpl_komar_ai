"""
Full team (XI + bench) and points per player for all 10 GWs.
Uses data/intel/final_squad.json + data/raw/fpl_api/player_history.csv
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

BASE = Path(r"c:\Users\Andrej\Desktop\fpl ai")
with open(BASE / "data/intel/final_squad.json", encoding="utf-8") as f:
    squad = json.load(f)

# player_id -> gw -> total_points
history = {}
with open(BASE / "data/raw/fpl_api/player_history.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid, gw = int(row["player_id"]), int(row["gameweek"])
        history.setdefault(pid, {})[gw] = int(row["total_points"])

# --- Part 1: Full team + points per GW ---
print("=" * 80)
print("  FULL TEAM AND POINTS PER GAMEWEEK (XI + BENCH)")
print("=" * 80)

season_xi_pts = 0
season_bench_pts = 0
season_penalty = 0
season_bb_pts = 0

for gw_data in squad["gameweeks"]:
    gw = gw_data["gw"]
    chip = gw_data.get("chip") or ""
    cap_id = gw_data.get("captain_id")
    pen = gw_data.get("penalty_pts", 0)
    is_tc = chip == "triple_captain"
    is_bb = chip == "bench_boost"
    cap_mult = 3 if is_tc else 2

    xi_names = gw_data["xi_final"]
    xi_ids = gw_data["xi_final_ids"]
    bench_names = gw_data["bench_final"]
    bench_ids = gw_data["bench_final_ids"]

    print(f"\n  --- GW {gw}  [Chip: {chip or 'None'}]  Captain ID: {cap_id} ---")
    print("  STARTING XI (points count; captain x2 or x3 if TC):")
    xi_pts = 0
    for name, pid in zip(xi_names, xi_ids):
        base_pts = history.get(pid, {}).get(gw, 0)
        pts = base_pts * (cap_mult if pid == cap_id else 1)
        xi_pts += pts
        cap_mark = " (C x" + str(cap_mult) + ")" if pid == cap_id else ""
        print(f"    {name:<24}  id {pid:>4}   {base_pts:>2} pts  =>  {pts:>3} pts{cap_mark}")
    print(f"  XI total: {xi_pts} pts")

    print("  BENCH (points count only if Bench Boost played):")
    bench_pts = 0
    for name, pid in zip(bench_names, bench_ids):
        pts = history.get(pid, {}).get(gw, 0)
        bench_pts += pts
        print(f"    {name:<24}  id {pid:>4}   {pts:>2} pts")
    print(f"  Bench total: {bench_pts} pts")

    gw_total = xi_pts - pen + (bench_pts if is_bb else 0)
    season_xi_pts += xi_pts
    season_bench_pts += bench_pts
    season_penalty += pen
    if is_bb:
        season_bb_pts += bench_pts

    print(f"  Penalty: -{pen} pts  |  GW TOTAL: {gw_total} pts")

print("\n" + "=" * 80)
print(f"  SEASON TOTALS:  XI={season_xi_pts}  Bench(played)={season_bb_pts}  Penalties=-{season_penalty}  =>  {season_xi_pts + season_bb_pts - season_penalty} pts")
print("=" * 80)

# --- Part 2: Every player, points in each GW they were in squad ---
# Collect: for each (player_id, gw) the name and points when they were in squad
# and build list of all unique players with a canonical name
id_to_name = {}
player_gw_pts = defaultdict(dict)  # (id, gw) -> pts (base, no captain mult)
player_gw_captain = defaultdict(dict)  # (id, gw) -> True if captain that gw
player_gw_played = defaultdict(dict)   # (id, gw) -> "XI" or "Bench"

for gw_data in squad["gameweeks"]:
    gw = gw_data["gw"]
    cap_id = gw_data.get("captain_id")
    chip = gw_data.get("chip") or ""
    is_tc = chip == "triple_captain"
    is_bb = chip == "bench_boost"
    cap_mult = 3 if is_tc else 2

    for name, pid in zip(gw_data["xi_final"], gw_data["xi_final_ids"]):
        id_to_name[pid] = name
        base = history.get(pid, {}).get(gw, 0)
        player_gw_pts[pid][gw] = base * (cap_mult if pid == cap_id else 1)
        player_gw_captain[pid][gw] = pid == cap_id
        player_gw_played[pid][gw] = "XI"

    for name, pid in zip(gw_data["bench_final"], gw_data["bench_final_ids"]):
        id_to_name[pid] = name
        base = history.get(pid, {}).get(gw, 0)
        player_gw_pts[pid][gw] = base  # actual points (bench never C)
        player_gw_captain[pid][gw] = False
        player_gw_played[pid][gw] = "Bench"

# Table: one row per player (ever in squad), columns GW1..GW10
print("\n" + "=" * 80)
print("  POINTS PER PLAYER FOR ALL 10 GAMEWEEKS (as counted in our total)")
print("  XI = points counted; Bench = points only if Bench Boost that GW; C = captain")
print("=" * 80)

all_ids = sorted(id_to_name.keys(), key=lambda i: id_to_name.get(i, "").lower())
# Header
header = "  Player                    ID   " + "  ".join(f"GW{gw:>2}" for gw in range(1, 11)) + "   Total"
print(header)
print("  " + "-" * (len(header) - 2))

for pid in all_ids:
    name = id_to_name[pid]
    row_pts = []
    total = 0
    for gw in range(1, 11):
        pts = player_gw_pts[pid].get(gw, None)
        if pts is not None:
            row_pts.append(f"{pts:>3}")
            total += pts
        else:
            row_pts.append("  -")
    print(f"  {name:<24}  {pid:>4}   " + "  ".join(row_pts) + f"   {total:>4}")

print("  " + "-" * (len(header) - 2))
print("\n  (Points shown = actual FPL points; captain x2 or x3 when TC. '-' = not in squad that GW.)")
