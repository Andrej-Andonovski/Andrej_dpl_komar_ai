"""
Quick sweep of FDR_MULT_DEF — position-specific FDR penalty for GK/DEF.
Tries values from 0.025 to 0.10, reports total actual pts each run.
"""
import subprocess, sys, json, re, os

SIM   = os.path.join("pipeline", "season_simulator.py")
VALS  = [0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

# Read simulator source once
with open(SIM, encoding="utf-8") as f:
    src_orig = f.read()

results = []

for val in VALS:
    # Patch: add FDR_MULT_DEF and swap the FDR block
    src = src_orig

    # Add FDR_MULT_DEF line after FDR_MULT definition
    src = src.replace(
        "FDR_MULT       = 0.025  # best known",
        f"FDR_MULT       = 0.025  # best known\nFDR_MULT_DEF   = {val}   # GK/DEF sweep"
    )

    # Swap FDR adjustment block to use position-specific mult
    src = src.replace(
        "        # FDR adjustment\n"
        "        fdr  = p.get(\"fdr\", 3.0)\n"
        "        pred *= max(0.5, 1.0 - FDR_MULT * (fdr - 3.0))",
        "        # FDR adjustment\n"
        "        fdr      = p.get(\"fdr\", 3.0)\n"
        "        fdr_mult = FDR_MULT_DEF if pos in (\"GK\", \"DEF\") else FDR_MULT\n"
        "        pred *= max(0.5, 1.0 - fdr_mult * (fdr - 3.0))"
    )

    with open(SIM, "w", encoding="utf-8") as f:
        f.write(src)

    out = subprocess.run(
        [sys.executable, SIM],
        capture_output=True, text=True, encoding="utf-8"
    ).stdout

    # Parse total pts and penalties from output
    pts   = int(re.search(r"Total actual pts:\s+(\d+)", out).group(1))
    pen   = int(re.search(r"Total penalties:\s+(-?\d+)", out).group(1))
    chips = re.search(r"Chips used:\s+(\[.*?\])", out).group(1)

    results.append((val, pts, pen, chips))
    print(f"  FDR_MULT_DEF={val:.3f}  ->  pts={pts}  penalties={pen}  chips={chips}")

# Restore original
with open(SIM, "w", encoding="utf-8") as f:
    f.write(src_orig)
print("\nOriginal simulator restored.")

# Summary
print("\n=== SWEEP RESULTS ===")
print(f"{'FDR_MULT_DEF':>14}  {'PTS':>6}  {'PEN':>5}  CHIPS")
for val, pts, pen, chips in sorted(results, key=lambda x: -x[1]):
    print(f"  {val:>12.3f}  {pts:>6}  {pen:>5}  {chips}")
