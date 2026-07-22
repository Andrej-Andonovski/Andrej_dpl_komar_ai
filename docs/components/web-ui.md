---
type: component
status: active
tags:
  - component
  - ui
---

# Web UI

A small **Flask** application that visualizes a simulated season: the per-gameweek
squad, the LLM narrative, and availability/rotation warnings. It sits at the top
of the stack in [[system-overview]] and consumes, but never produces, decision
data.

## Responsibility
Serve a single-page dashboard (`index.html`) plus a small JSON API over the
existing output files:
- `GET /api/data` — the season simulation (`season_simulation.json`).
- `GET /api/explanations` — Stage 9 narratives (empty object if absent).
- `GET /api/intel` — compact per-GW availability/rotation per player, merged from
  `availability.json` and `rotation_risk.json`, for warning icons.
- `POST /api/run` — launch `season_simulator.py` as a background subprocess
  (single-flight, guarded by a lock).
- `GET /api/status` — whether a run is in progress.

## Why it exists
The simulation output is dense JSON; the thesis needs a legible way to inspect
each gameweek's decisions and the narrative behind them. The UI is a read/trigger
front-end, keeping all logic in the pipeline.

## How it interacts
Reads artifacts produced by the [[season-simulator]] and [[llm-layers]], plus the
[[intelligence-suite]] availability/rotation files. Its `POST /api/run` can
trigger a [[season-simulator]] run. It calls no other component's code directly —
it only reads their output files and spawns the simulator process.

## Depends on
- [[season-simulator]] (`season_simulation.json`; also the process it can launch).
- [[llm-layers]] (`stage9_explanations.json`).
- [[intelligence-suite]] (`availability.json`, `rotation_risk.json`).

## Depended on by
- Nothing — it is the presentation layer.

## Assumptions & limitations
- Serves on `0.0.0.0:5000`, `debug=False`; intended for local/thesis use, not a
  hardened deployment.
- Endpoints degrade gracefully when files are missing (404 for data, empty object
  for explanations), so an unrun pipeline shows an empty dashboard rather than
  crashing.
- The triggered run uses `sys.executable`; on a machine without the required
  Python/data (see [[repository-map]]) the run will fail even though the UI loads.

## Related Source Files
- `ui/server.py`
- `ui/index.html`

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
