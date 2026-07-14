# Phase 4 — Chips as MILP Variables: Report

Status: **BUILD COMPLETE — result regressed, cause identified** (2026-07-14).
Result: **1955** (vs Phase 3's 2156 with external chips, baseline 2252).
Mechanism proven (11/11 `tests/test_milp_chips.py`, 35/35 across MILP
suites); placement quality of *unanchored* chips is the failure, and it is
the top item on the Phase 5/6 fixing backlog.

Artifacts: chip variables + FH shadow blocks + reservation guard in
`solve_horizon` (`pipeline/milp_core.py`), `MP_CHIPS="model"|"legacy"` env
flag, run `data/intel/run_mp4_20260714.log`, metrics
`data/intel/metrics_mp_phase4.json`. Phase 3 run archived at
`data/intel/archive/season_simulation_mp_phase3_2156.json`.

## What worked — the event-anchored chips

| chip | placed | evidence |
|---|---|---|
| fh2 | **GW26 (DGW)** | event-week-only constraint; shadow squad, no hits |
| bb2 | **GW33 (DGW)** | reservation guard HELD it through GW20-32 non-double weeks for the known far double; bench scored 20 |
| fh1 | correctly never played | Set 1 has no event weeks — constraint made it unplayable rather than wasted on a plain week |

The FH shadow-squad block, spacing constraints, one-chip-per-GW, and
per-set ledgers all executed correctly across the season. Zero solver
failures, zero chipped-solve retries.

## What failed — unanchored chips burn at first eligibility

tc1@**GW1**, wc1@**GW2**, wc2@**GW20** (first week of Set 2), bb1@GW17
(Set 1 has no DGW so the guard was inert). Root cause, precisely the
blueprint's R2 beyond what δ_c=0.97 covers: **a chip's opportunity cost
spans the season, but the model prices it only within the horizon.** Any
weakly-positive chip value fires immediately because holding has no
represented value. WC1@GW2 is the historical panic-rebuild disease
reproduced structurally: a full rebuild on one gameweek of data. The
knock-on squad path also broke hit quality (hit ROI mean −6.8 vs Phase 3's
+5.2 on the same H_cap).

The old system prevented this with CHIP_LOCKOUT=4 + tuned bars
(CHIP_BAR_WC=20 etc.) — crude season-level scarcity prices. We deleted them
without a structural replacement; the BB calendar guard was the only
scarcity mechanism, and it was also the only unanchored chip that placed
well when armed.

## Fixing backlog (Phases 5/6 — with cross-season validation)

1. **Chip scarcity pricing** (the fix for this phase's regression), in
   preference order: (a) extend the reservation guard to TC/WC using
   season-remaining value estimates (e.g. hold WC unless rebuild gain
   exceeds the season-median rebuild gain — measurable, not tuned);
   (b) restore CHIP_LOCKOUT (rule-like, defensible); (c) explicit reserve
   values subtracted on firing (one constant per chip, cross-season tuned).
2. Captain channel (~50 pts vs baseline, Phase 2/3 reports).
3. Cross-solve churn / plan-thrash (T4 memory across solves or a
   plan-consistency penalty).
4. w̄ bench pricing outside chip weeks.
5. q90 coverage re-measurement (headroom definition) + π calibration check
   in the next full calibration run.

## Scoreboard after all build phases

| config | pts | note |
|---|---|---|
| fair baseline (legacy, corrected rules) | **2252** | 25 tuned constants |
| mp H=1 (P2) | 2070 | structure only |
| mp H=5 + legacy chips (P3) | **2156** | best mp so far |
| mp H=5 + in-model chips (P4) | 1955 | scarcity gap |

Best current mp configuration: **Phase 3** (H=5, `MP_CHIPS=legacy`). The
build-everything pass is complete; per the project decision, tuning and the
backlog above are the next (fixing) stage, validated on 2023-24 + 2024-25
before 2025-26 is trusted.
