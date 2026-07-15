# FPL AI — Optimizer Redesign: Full Project State (Handoff)

Written 2026-07-15 to bootstrap a fresh session. Single source of truth for
where the redesign stands. Detailed evidence lives in the sibling docs
(phase0-4 reports, generalization_report, chip_scarcity_fix, fixing_backlog).

## 0. Environment facts (read first)

- **All runs happen in Docker** — no local Python. Image `fpl-sim`
  (updated 2026-07-14: + scikit-learn 1.9.0 + highspy, committed).
  `docker run --rm -v "<repo>:/app" -w /app -e KEY=val fpl-sim python -u <script>`
- **2468 (the thesis number) is environment-bound**: produced by the
  original machine's library stack. In Docker the same legacy code gives
  **2236** (bit-identical across reruns). All comparisons are Docker-only.
- Raw 2025-26 data (`data/raw/fpl_api/`) cannot be re-downloaded (FPL API
  rolled over). Cross-season inputs regenerate via
  `pipeline/build_season_inputs.py` from `data/raw/vaastav_repo/`
  (gitignored, re-fetch from GitHub raw if missing).
- Sim outputs overwrite per config (`season_simulation_corrected[_mp][_<season>].json`)
  and are written GW-by-GW — **never run metrics mid-flight; archive
  before reruns** (`data/intel/archive/` has all milestone runs).

## 1. Scoreboard (all corrected rules, Docker)

| Config | 2025-26 | 2023-24 | 2024-25 |
|---|---|---|---|
| legacy optimizer (25 tuned constants) | **2252** (fair baseline) | 2164 | 2359 |
| mp H=1 (Phase 2) | 2070 | — | — |
| mp H=5 + legacy chips (Phase 3) | 2156 | 2162 | 2341 |
| mp H=5 + model chips + scarcity fix | 2029 | **2174** (8/8 chips) | **2410 ← project best** |

Two headline findings (thesis-ready):
1. **Generalization**: legacy's 96-pt home edge collapses to −2/−18 on
   neutral calendars — its advantage is ~85-90% memorized calendar
   (docs/generalization_report.md).
2. **Chips**: with the scarcity fix, mp+model-chips **beats the fully tuned
   legacy system on both neutral seasons** (docs/chip_scarcity_fix.md).
   Legacy chips only win on 2025-26 (their tuning season; its Set 1 is
   eventless so unanchored chips have no calendar anchor there).

## 2. What is built (all tested, all pushed)

- **Phase 0** — corrected FPL rules behind `RULES_MODE=corrected`:
  50% sell-on via purchase-price ledger (owned players priced at sell value
  inside the ILP — exact identity), real FT banking 1..5, no budget
  relaxation (raise), `RULE_EVENTS_FT={15:5}` (2025-26 only).
  `pipeline/fpl_rules.py` + 10/10 tests.
- **Phase 1** — `pipeline/prediction_matrix.py`: per-future-GW feature-swap
  re-prediction, per-fixture DGW sums, hard blank zeros, π (availability ×
  rotation, decays to base rate), φ (confidence — **currently unused, φ≡1**,
  failed its Phase 1 gate), q90 (empirical headroom p90−mean, shrunk).
  12/12 tests. Calibration: MAE rots only +4.1% over 5 GWs ⇒ horizon safe.
  GW1 community prior `OWN_PRIOR_GW1=0.213` (deleting it cost 98 pts).
- **Phases 2-4** — `pipeline/milp_core.py`: `solve_gw` (H=1) and
  `solve_horizon` (H=5): per-week squad/XI/captain/vice/transfers, FT
  banking (≤-recursion, no-phantom-hit proof), churn guard (≤1 in/out per
  player per horizon), bank recursion with sell values, chips as variables
  (FH shadow squads on event weeks only, BB/TC aux at δ_c=0.97, WC hit
  waiver, spacing, one-per-GW, per-set ledger). 12+12+16 tests, HiGHS.
- **Chip scarcity fix** — cold-start lockout (GW≤4), BB/TC held for known
  far DGWs (self-disarms at set deadline), WC gated on ≥`MP_WC_BELOW=4`
  owned players below replacement level (measured from the matrix).
- **Cross-season harness** — `SIM_SEASON` env: swaps inputs to
  `data/raw/seasons/<S>/`, season-cuts training (no leakage), disables
  2025-26-only intel + FT event, derives the GW1 snapshot season.
  `SIM_END_GW` env for quick smokes. 2024-25 needs the element_type-5
  (assistant manager) filter — already in the input builder.
- **Percentile chip bar** (pulled from personal PC 2026-07-15, commit
  a406cd2): `pipeline/chip_percentile.py` ledger — unanchored WC/TC/BB may
  fire on a PLAIN week only if their proxy value clears the q75 of their
  own season history (3-obs warm-up; event weeks exempt; block-and-resolve
  loop in the simulator; state file `chip_percentile_<season>.json`).
  Also added: `MP_THETA` env (captain blend ablation), HiGHS availability
  fallback. 5/5 + 16/16 tests pass. **NOT yet validated full-season.**
- **Metrics**: `pipeline/backtest_metrics.py` (any sim JSON, `--history`
  for transfer counterfactuals). Baseline metric JSONs in `data/intel/`.

## 3. Env-var cheat sheet

```
RULES_MODE=corrected            # mandatory for mp and cross-season
OPTIMIZER=mp                    # "legacy" = old system
MP_HORIZON=5                    # 1 = Phase 2 behaviour
MP_CHIPS=model                  # "legacy" = external scheduler (P3 ablation)
MP_THETA=0.5                    # captain mean<->ceiling blend
MP_WC_BELOW=4                   # WC squad-state gate
MP_CHIP_PERCENTILE_Q=0.75       # percentile bar (WARMUP/RESUME/STATE too)
SIM_SEASON=2023-24|2024-25      # cross-season (default 2025-26)
SIM_END_GW=3                    # smoke runs
CHIP_STRATEGY=legacy            # only matters for OPTIMIZER=legacy runs
```
Tests: `python tests/test_{fpl_rules,prediction_matrix,milp_core,milp_horizon,milp_chips,chip_percentile}.py`
(plain python, no pytest). Import smoke: `tests/smoke_imports.py`.

## 4. IMMEDIATE NEXT TASK — percentile-bar A/B (not yet run)

Rerun `MP_CHIPS=model` full-season on all three calendars with the bar
active (it's on by default now). Targets:
- 2025-26 ≥ 2156 (close the 2029 gap — this is what the bar was built for)
- 2023-24 ≥ 2174 and 2024-25 ≥ 2410 must NOT regress
Watch: proxies are recorded on event weeks too — DGW-inflated values raise
the bar plain weeks must clear; if chips go unused, that's suspect #1.
Archive the three current mp JSONs before launching (see §0).

## 5. Remaining backlog (docs/fixing_backlog.md has full detail)

1. Captain channel (~50 pts vs legacy on 2025-26): MP_THETA sweep now
   trivial; κ variants; π calibration for premiums. Judge by captain-avg /
   regret across all three seasons.
2. Cross-solve churn (plan-consistency memory) — 10-17 buybacks/season.
3. w̄ bench pricing outside chip weeks (waste 11/GW at H=5 vs 3.3 legacy).
4. Phase 6 sweep: Optuna over {H, δ, δ_c, θ, γ, w̄, MP_WC_BELOW, hit_cap,
   percentile q} — train 2 seasons, hold out 1, report the fold gap.
5. Calibration re-measures: q90 coverage (headroom def), π buckets,
   φ retest with μ-matched buckets.
6. Housekeeping: captain/chip columns in backtest_metrics; unpruned sanity
   solve (§7.2 valve); thesis ablation table (numbers all exist in docs/);
   AGENTS.md is a stale CLAUDE.md fork — sync or delete.

## 6. Repo/process notes

- Repo: github.com/Andrej-Andonovski/Andrej_dpl_komar_ai, branch master,
  direct pushes (no co-author lines in commit messages — owner preference).
  Machine credential may need re-auth to the owner account on push (GCM
  popup).
- The legacy system stays runnable end-to-end (`OPTIMIZER=legacy`) — it is
  the thesis ablation baseline and the fallback ladder's last rung.
- Best real-season config: `OPTIMIZER=mp MP_HORIZON=5 MP_CHIPS=model`.
- Live-play prerequisites (future): persist the purchase ledger + percentile
  state weekly (RESUME=1), re-sync prices, and note intel_02/intel_08
  scraper work happening in a parallel workstream.
