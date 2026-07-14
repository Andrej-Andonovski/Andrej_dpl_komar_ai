"""Import smoke check for both optimizer modes (no data files touched)."""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pipeline.season_simulator as s
print("legacy import OK | OPTIMIZER =", s.OPTIMIZER, "| out =",
      os.path.basename(s.OUTPUT_JSON))

os.environ["OPTIMIZER"] = "mp"
os.environ["RULES_MODE"] = "corrected"
importlib.reload(s)
print("mp import OK     | OPTIMIZER =", s.OPTIMIZER, "| out =",
      os.path.basename(s.OUTPUT_JSON))
assert s.OPTIMIZER == "mp" and s.OUTPUT_JSON.endswith("_corrected_mp.json")
print("smoke OK")
