---
type: reference
status: active
tags:
  - reference
  - environment
  - docker
---

# Reference: Environment & Docker

Canonical home for how the project is run and why results are environment-bound.
Other notes link here instead of restating these facts.

## Docker is the run environment
There is no local Python on the working machine. All pipeline scripts run in the
`fpl-sim` Docker image (updated 2026-07-14 to add scikit-learn 1.9.0 and highspy):

```
docker run --rm -v "<repo>:/app" -w /app -e KEY=val fpl-sim python -u <script>
```

## The headline score is environment-bound
The same legacy code produces different totals on different library stacks:

| Figure | Meaning |
|--------|---------|
| **2468** | Full-season total on the **original machine** (the thesis number) |
| **2236** | Same legacy code in **Docker**, bit-identical across reruns (deterministic) |
| **2252** | Fair in-Docker baseline under corrected rules |

Consequence: **all comparisons must stay within one environment (Docker).** The
divergence originates at GW1 from a LightGBM stack difference. See
[[phase0_baseline]] and [[HANDOFF]]. Benchmark numbers themselves live in
[[evaluation-metrics-and-results]].

## Data availability on this clone
- `data/raw/` is **gitignored and absent** on the Desktop clone.
- The production run needs `data/raw/fpl_api/{player_history,players_raw,fixtures_raw}.csv`,
  copied from the original machine.
- The **FPL API cannot be re-fetched** — it rolled over to 2026-27.
- Cross-season inputs are regenerated from a Vaastav repo clone via
  `build_season_inputs.py` (see [[cross-season-harness]]).

## Run discipline
Simulator output is written gameweek-by-gameweek and overwrites per config
(`season_simulation[_corrected][_mp][_<season>].json`). **Never compute metrics
mid-run; archive milestone runs to `data/intel/archive/` before rerunning**
([[HANDOFF]]).

## Related Source Files
- `pipeline/season_simulator.py` (env flags)
- `pipeline/build_season_inputs.py`
- `data/intel/archive/` (milestone run backups)

---
Hubs: [[system-overview]] · [[repository-map]]
