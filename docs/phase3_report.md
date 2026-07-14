# Phase 3 — Multi-Period MILP (H=5, chips external): Report

Status: **COMPLETE** (2026-07-14). Result: **2156** (corrected rules, legacy
chip scheduler, Docker) — +86 over Phase 2 (2070), −4.3% vs the 2252
baseline. Zero horizon-solve failures across 38 GWs.

Artifacts: `solve_horizon`/`prune_pool` in `pipeline/milp_core.py`
(12/12 tests `tests/test_milp_horizon.py`), `MP_HORIZON` env flag,
archived run `data/intel/archive/season_simulation_mp_phase3_2156.json`,
metrics `data/intel/metrics_mp_phase3.json`.

## What the horizon changed (vs Phase 2, same rules/chips)

| | P2 (H=1) | P3 (H=5) | baseline |
|---|---|---|---|
| total | 2070 | **2156** | 2252 |
| hit pts / hit-ROI mean | 8 / −0.5 | 64 / **+5.2** | 4 / +39 (1 hit) |
| BB chips gained | 0, +2 | **+25, +16** | +2, +2 |
| transfer 4GW payoff | −0.38 | +0.82 | −0.55 |
| captain avg | 5.29 | 4.82 | 6.21 |
| bench waste /GW | 5.86 | 11.08 | 3.31 |
| buybacks ≤6GW | 10 | 17 | 8 |

- **Bench-boost coordination materialised**: the solver built real benches
  into the legacy scheduler's BB weeks (+41 total vs +4 baseline) — the
  exact chip-transfer coordination the blueprint promised, even with chips
  still external.
- **Hits are now investments**: 16 hits, mean realized ROI +5.2 over 4-GW
  windows (~+67 gained vs 64 paid — net positive incl. one −36 outlier).
  Unit tests prove the discrimination: explosive-now targets justify hits,
  steady upgrades get deferred to accrued FTs
  (`test_steady_upgrade_deferred_not_hit`).
- **Banking works in-plan** (`test_banking_emerges`) but shows up rarely in
  realized FT histograms — each re-solve usually finds a positive week-t
  move. Cross-solve churn is the mechanism (below).

## Open issues carried to Phases 5/6 (per the all-phases-first decision)

1. **Captain channel still degrading** (4.82 vs 6.21) — now the largest
   single gap (~50 pts). κ/θ/q90 need the cross-season ablation, and the
   horizon may warrant an ownership-continuity term for premium anchors.
2. **Cross-solve churn**: the churn guard binds within one solve; weekly
   re-solves re-free everything (17 buybacks). Candidate fix for Phase 6:
   carry a small plan-consistency penalty or extend T4 memory across solves
   (blueprint R5 mitigation).
3. **Bench waste 11.08/GW** is partly deliberate (BB build-up + auto-sub
   inventory: +76 rescued) but w̄=0.15 likely overprices bench outside chip
   weeks — w̄ goes to the Phase 6 sweep.

Phase 3 exit criterion (beats Phase 2 on points and transfer metrics):
points ✓, payoff/ROI ✓, churn ✗ (worse) — accepted with the cross-solve
churn issue logged as the known cause.
