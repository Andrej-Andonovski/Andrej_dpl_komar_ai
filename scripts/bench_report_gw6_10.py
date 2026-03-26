"""
Bench report GW6-10 from existing final_squad.json + player_history.
We have: bench composition (names, IDs). We do NOT have per-player predicted
points stored — only predicted_score (GW total). Actual points from history.
"""
import json
import csv
from pathlib import Path

BASE = Path(r"c:\Users\Andrej\Desktop\fpl ai")
with open(BASE / "data/intel/final_squad.json", encoding="utf-8") as f:
    squad = json.load(f)

# player_id -> gw -> total_points
history = {}
with open(BASE / "data/raw/fpl_api/player_history.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid, gw = int(row["player_id"]), int(row["gameweek"])
        history.setdefault(pid, {})[gw] = int(row["total_points"])

print("=" * 70)
print("  BENCH COMPOSITION & ACTUALS  (GW6-10)")
print("  Predicted per-player: NOT logged in this run (see below for fix)")
print("=" * 70)

for gw_data in squad["gameweeks"]:
    gw = gw_data["gw"]
    if gw < 6:
        continue
    names = gw_data["bench"]
    ids = gw_data["bench_ids"]
    chip = gw_data.get("chip") or "—"
    gw_pred_total = gw_data.get("predicted_score")
    actuals = [history.get(pid, {}).get(gw, 0) for pid in ids]
    bench_actual_total = sum(actuals)
    print(f"\n  GW{gw}  [chip: {chip}]  predicted_score (GW total): {gw_pred_total:.1f}")
    print(f"  Bench:")
    for name, pid, pts in zip(names, ids, actuals):
        print(f"    {name:<22} (id {pid})  actual: {pts:2d} pts")
    print(f"  Bench actual total: {bench_actual_total} pts")

print("\n" + "=" * 70)
print("  Summary (which GW would have been best for BB by actuals):")
print("=" * 70)
rows = []
for gw_data in squad["gameweeks"]:
    gw = gw_data["gw"]
    if gw < 6:
        continue
    ids = gw_data["bench_ids"]
    total = sum(history.get(pid, {}).get(gw, 0) for pid in ids)
    rows.append((gw, total))
rows.sort(key=lambda x: -x[1])
for gw, total in rows:
    print(f"  GW{gw}: bench actual total = {total} pts")
print("\n  Predicted values were NOT stored. To set threshold from data,")
print("  add to intel_06 log: bench_pred_total + bench_player_preds per GW.")
