"""
Sweep FORM_BLEND_WEIGHT: post-prediction blend of model output with form_last3.
final_pred = (1 - w) * model_pred + w * form_last3
"""
import subprocess, sys, json, re, os

SIM  = os.path.join("pipeline", "season_simulator.py")
VALS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

with open(SIM, encoding="utf-8") as f:
    src_orig = f.read()

# Anchor strings
PARAM_ANCHOR = "FDR_MULT       = 0.025  # best known"
PRED_ANCHOR  = (
    "        # Per-player prediction ceiling\n"
    "        pred = min(max(0.0, pred), PRED_CAP)\n"
)

results = []

for w in VALS:
    src = src_orig

    # Add FORM_BLEND_WEIGHT param
    src = src.replace(
        PARAM_ANCHOR,
        f"{PARAM_ANCHOR}\nFORM_BLEND_WEIGHT = {w}  # blend model pred with form_last3"
    )

    # Add blend after the pred cap line
    src = src.replace(
        PRED_ANCHOR,
        PRED_ANCHOR +
        "        # Blend with recent form so hot streaks aren't ignored\n"
        "        if FORM_BLEND_WEIGHT > 0:\n"
        "            pred = (1.0 - FORM_BLEND_WEIGHT) * pred + FORM_BLEND_WEIGHT * p.get('form_last3', pred)\n"
    )

    with open(SIM, "w", encoding="utf-8") as f:
        f.write(src)

    out = subprocess.run(
        [sys.executable, SIM],
        capture_output=True, text=True, encoding="utf-8"
    ).stdout

    pts   = int(re.search(r"Total actual pts:\s+(\d+)", out).group(1))
    pen   = int(re.search(r"Total penalties:\s+(-?\d+)", out).group(1))
    chips = re.search(r"Chips used:\s+(\[.*?\])", out).group(1)

    results.append((w, pts, pen, chips))
    print(f"  FORM_BLEND={w:.2f}  ->  pts={pts}  pen={pen}  chips={chips}")

with open(SIM, "w", encoding="utf-8") as f:
    f.write(src_orig)
print("\nOriginal restored.")

print("\n=== RESULTS (sorted by pts) ===")
print(f"{'FORM_BLEND':>12}  {'PTS':>6}  {'PEN':>5}")
for w, pts, pen, chips in sorted(results, key=lambda x: -x[1]):
    marker = " <-- BEST" if pts == max(r[1] for r in results) else ""
    print(f"  {w:>10.2f}  {pts:>6}  {pen:>5}{marker}")
