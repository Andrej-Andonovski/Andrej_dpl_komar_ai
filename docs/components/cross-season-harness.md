---
type: component
status: in-progress
tags:
  - component
  - validation
  - generalization
---

# Cross-Season Harness

Tooling that runs the system on **past, unseen seasons** (2023-24, 2024-25) to
test whether it generalizes rather than memorizes the 2025-26 calendar. This is
the thesis-critical validation apparatus for the [[milp-optimizer]] redesign.

## Responsibility
Regenerate a season's inputs in the exact schema the [[season-simulator]]
expects, then run the simulator on that season under controlled settings. Two parts:
- `build_season_inputs.py` — converts true per-fixture Vaastav-repo data into
  `data/raw/seasons/<SEASON>/{player_history,players_raw,fixtures_raw}.csv`
  (per-fixture rows so doubles appear twice; real FDR and ownership preserved).
- `SIM_SEASON` handling in the simulator — swaps paths, snapshot behavior, the
  training cut-off, and turns intel off (corrected rules only). `SIM_END_GW`
  bounds the run for quick smoke tests.

## Why it exists
The tuned [[legacy-ilp-optimizer]] scores highly on 2025-26 partly because its
constants encode that calendar. The only honest way to show the [[milp-optimizer]]
travels is to replay neutral seasons it was never tuned on. The result — legacy's
edge is largely memorized while the MILP travels untuned — is the headline of
[[generalization_report]].

## How it interacts
Feeds rebuilt inputs to the [[season-simulator]], typically with `OPTIMIZER=mp`,
and its outputs are compared config-by-config in [[HANDOFF]]. Runs happen in the
Docker `fpl-sim` image (no local Python on this machine).

## Depends on
- A Vaastav repo clone under `data/raw/vaastav_repo/` (gitignored; re-fetchable).
- [[season-simulator]] and [[milp-optimizer]] (the things it exercises).

## Depended on by
- Nothing at runtime; it is a validation/experiment tool.

## Assumptions & limitations
- Requires the Vaastav repo data locally; on this clone it is absent and must be
  re-fetched.
- Input rebuild has documented deliberate choices (e.g. GW1 price taken from
  history rather than the end-of-season `now_cost`; an `element_type=5` AM filter
  for 2024-25) — see the module docstring and [[fixing_backlog]].
- Neutral-season results are Docker-only and not directly comparable to the
  environment-bound 2468 figure.

## Related Source Files
- `pipeline/build_season_inputs.py`
- `pipeline/season_simulator.py` (`SIM_SEASON` / `SIM_END_GW` handling)
- `pipeline/run_variants.py`

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
