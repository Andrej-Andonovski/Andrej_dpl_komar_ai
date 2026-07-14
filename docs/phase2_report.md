# Phase 2 — Single-GW MILP with the New Objective: Report

Status: **COMPLETE — exit gate NOT met** (2026-07-14). Verdict: proceed to
Phase 3; do NOT hand-tune further at H=1 (rationale §4).

Artifacts: `pipeline/milp_core.py` (12/12 tests, HiGHS), `OPTIMIZER=mp` path
in `season_simulator.py`, runs/metrics in `data/intel/`
(`season_simulation_corrected_mp.json`, `metrics_mp_phase2*.json`,
`run_mp*_20260714.log`).

## 1. What was built

`milp_core.solve_gw` — the H=1 instance of the blueprint §4 program:
XI by matrix μ + captain κ = π·[(1−θ)μ + θ·q90] + vice (γ=0.07) + bench EV
(β = w̄·π·μ, w̄=0.15) − 4·hits (cap 2). Owned players priced at sell value
(exact §5 identity). Vice armband fallback added to scoring (real FPL rule —
legacy never modelled it). Deleted from the mp path: loyalty bonus, bench
bonus, DGW ×2, FDR post-multipliers, ownership boost (GW1 prior retained,
see below), GW1–8 blending, PRED/XI caps, captain streak/blank/form gates,
intel_05 override, MAX_HITS=1.

## 2. Results (corrected rules, legacy chips, Docker — vs 2252 baseline)

| | baseline | mp v1 | mp v2 (final) |
|---|---|---|---|
| **Total** | **2252** | 2031 | **2070 (−8.1%)** |
| GW1+GW2 | 146 | 48 | 134 |
| hits (pts) | 4 | 28 | 8 |
| transfers / short holds ≤2GW | 42 / 7 | 49 / 13 | 44 / 18 |
| captain avg / regret | 6.21 / 6.53 | 5.08 / 6.89 | 5.29 / 7.68 |
| bench waste /GW | 3.31 | 6.42 | 5.86 |
| chips used | 6 | 7 | 7 (wc2 recovered) |
| FT=1 deadlines | 23/30 | 29/34 | 29/34 |

v1→v2 changes (both principled, both kept):
- **GW1 community prior restored** (`OWN_PRIOR_GW1=0.213`, matrix-level,
  t=1 only — blueprint §1.2 sanctioned interim until ownership becomes a
  model feature). Worth **+86 pts across GW1–2** and removed the GW2 panic
  hits. Deleting it in v1 was over-zealous constant-hunting; the evidence
  is unambiguous.
- **q90 v2: empirical headroom** (p90 − mean of own last-10 played scores,
  shrunk to position prior) replaces μ + z·σ. DEF armbands persisted (7) —
  elite-DEF ceilings in 2025-26 are real (attacking returns + CS + bonus),
  not a σ artifact. The legacy ban on DEF captains was a prior, not a truth.

## 3. Honest decomposition of the remaining −182

1. **Captain channel ≈ −35** (5.29 vs 6.21 avg). κ is position-agnostic and
   ceiling-blended; legacy's tuned CAP_MULT + gates ensemble was better *on
   this season*. θ and the κ form go to the Phase 6 cross-season ablation —
   tuning them on 2025-26 alone would recreate the disease.
2. **Bench overspend ≈ −40 to −60** (waste 5.86 vs 3.31/GW). At H=1 the
   bench EV term buys real players whose cost comes out of XI strength, and
   without BB-planning coordination the investment never cashes. w̄ joins
   the Phase 6 sweep; Phase 4's in-model BB gives the bench spend a purpose.
3. **Churn ≈ −40 to −60** (18 short holds; FT=1 at 29/34 deadlines). The
   H=1 objective cannot price "hold the transfer" — banking has zero value
   and sideways moves are free. This is THE Phase 3 deliverable; patching it
   at H=1 means reinventing the loyalty bonus.
4. Hit pricing now guarded (−8 vs v1's −28) but still one-week-gain based —
   also Phase 3.

## 4. Why the gate failure does not block Phase 3

The blueprint's own thesis is that H=1 without the compensating hacks is
*handicapped* — the hacks were doing real work as crude horizon proxies
(loyalty ≈ switching cost, MAX_HITS ≈ hit distrust, bench bonus ≈ BB
coordination). Phase 2 proves the structure works (rules exact, vice real,
chips complete, per-fixture DGW/blank exact, MAE better) and quantifies
precisely what the horizon must recover (~180 pts across churn, bench
coordination, captaincy). The alternative — iterating constants at H=1
until 2184 — is single-season tuning, explicitly rejected. The binding
comparison vs the baseline happens at Phase 3/4 as designed (§9).

## 5. Carry-forward items

- θ, w̄, γ, HEADROOM priors → Phase 6 cross-season Optuna (never 2025-26 alone).
- q90 coverage of the new headroom definition → measure in the Phase 3
  calibration rerun.
- Captain-channel idea for Phase 3: κ already benefits from horizon context
  (a premium anchor's κ compounds over H weeks of ownership, not one).
- Bench EV: consider making β chip-aware only after Phase 4 lands BB in-model.
