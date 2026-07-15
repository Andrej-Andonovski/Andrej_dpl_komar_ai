# FPL AI — Optimizer Redesign: Full Project State (Handoff)

Written 2026-07-15, updated same day after the discipline/chip-bar 2×2 A/B.
Single source of truth for where the redesign stands. Detailed evidence
lives in the sibling docs (phase0-4 reports, generalization_report,
chip_scarcity_fix, fixing_backlog).

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

| Config | 2025-26 | 2023-24 | 2024-25 | Σ | penalties |
|---|---|---|---|---|---|
| legacy optimizer (25 tuned constants) | **2252** (fair baseline) | 2164 | 2359 | 6775 | −4/−8/−4 |
| mp H=1 (Phase 2) | 2070 | — | — | — | — |
| mp H=5 + legacy chips (Phase 3) | 2156 | 2162 | 2341 | 6659 | −64/…/… |
| mp + model chips + scarcity fix | 2029 | 2174 | **2410** | 6613 | −64/−136/−60 |
| mp + percentile bar (q75) | 2152 | 2206 | 2262 | 6620 | −60/−116/−80 |
| **mp + DISCIPLINE (shipped default)** | **2157** | 2186 | 2338 | **6681** | **0/−12/−8** |
| mp + bar + discipline | 2084 | **2224** | 2327 | 6635 | −12/−16/−8 |

Discipline = MP_HIT_COST=8, MP_HIT_BUDGET=4, MP_REBUY_GAP=4 (defaults now).
Percentile bar defaults OFF (MP_CHIP_BAR=0): with discipline on it nets
−46 across the calendars; its q + switch go into the Phase 6 sweep.

Headline findings (thesis-ready):
1. **Generalization**: legacy's 96-pt home edge collapses to −2/−18 on
   neutral calendars — its advantage is ~85-90% memorized calendar
   (docs/generalization_report.md).
2. **Chips**: mp+model-chips beats the fully tuned legacy system on both
   neutral seasons (docs/chip_scarcity_fix.md); legacy only wins on its
   tuning season.
3. **Discipline (2026-07-15 2×2 A/B)**: hit decision-price 2× + season hit
   budget + cross-solve rebuy lock cut penalties from ~−90 avg to ≤−12 avg
   AND raised the 3-season sum by +68. The undisciplined system's hits and
   sell→rebuy churn (40/30/28 buybacks/season) were net losses, exactly as
   the Phase-0/4 transfer-payoff metrics predicted.

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
- **Percentile chip bar** (pulled 2026-07-15, commit a406cd2; reworked same
  day): `pipeline/chip_percentile.py` ledger — unanchored WC/TC/BB may fire
  on a PLAIN week only if their proxy clears q75 of the season's PLAIN-week
  history, keyed by chip KIND (wc/tc/bb — set-2 chips inherit set-1 history,
  no warm-up free pass). Event weeks exempt both ways (bypass + never
  recorded). A set-deadline disarm was tried and REVERTED (−54 on 2025-26).
  Validated full-season on 3 calendars: helps an undisciplined system
  (+123 on 2025-26), nets −46 once discipline is on → **default OFF**
  (`MP_CHIP_BAR=1` re-enables). State file `chip_percentile_<season>.json`.
- **Hit + churn discipline** (2026-07-15, user-driven): `MP_HIT_COST`
  (objective decision-price per hit; real −4 still scored), `MP_HIT_CAP`
  (per-GW), `MP_HIT_BUDGET` (hard season cap, enforced by shrinking the
  per-GW cap at execution), `MP_REBUY_GAP` (cross-solve sold_at ledger →
  no_rebuy constraint; WC weeks exempt via ti≤WC_g, FH untouched, buys
  clear locks). 46+5 tests. Defaults = shipped config (8/2/4/4).
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
MP_HIT_COST=8                   # hit decision price (real -4 still scored)
MP_HIT_CAP=2                    # max paid hits per GW
MP_HIT_BUDGET=4                 # max paid hits per SEASON (-1 = unlimited)
MP_REBUY_GAP=4                  # sold players locked out for GAP GWs (0=off)
MP_CHIP_BAR=0                   # percentile bar OFF by default
MP_CHIP_PERCENTILE_Q=0.75       # bar q if enabled (WARMUP/RESUME/STATE too)
SIM_SEASON=2023-24|2024-25      # cross-season (default 2025-26)
SIM_END_GW=3                    # smoke runs
CHIP_STRATEGY=legacy            # only matters for OPTIMIZER=legacy runs
```
Tests: `python tests/test_{fpl_rules,prediction_matrix,milp_core,milp_horizon,milp_chips,chip_percentile}.py`
(plain python, no pytest). Import smoke: `tests/smoke_imports.py`.

## 4. IMMEDIATE NEXT TASK — captain channel (MP_THETA sweep)

DONE 2026-07-15: percentile-bar A/B (all variants), hit/churn discipline
2×2 — see §1 scoreboard. Shipped default = discipline-only.

Next: the captain channel is the biggest known lever (~50 pts vs legacy on
2025-26; captain regret 6.5-7.3/GW from Phase 0 metrics). MP_THETA sweep
{0.3, 0.7, 1.0} × 3 seasons on the shipped defaults (0.5 = the Wave C
numbers above). Also queued: legacy-optimizer + CHIP_STRATEGY=v2 backtest
(closes docs/chip_strategy_redesign.md §10 for the thesis ablation table —
writes season_simulation_corrected*.json, safe to run alongside mp runs).
Archive mp JSONs before every wave (§0).

## 5. Remaining backlog (docs/fixing_backlog.md has full detail)

1. Captain channel (~50 pts vs legacy on 2025-26): MP_THETA sweep now
   trivial; κ variants; π calibration for premiums. Judge by captain-avg /
   regret across all three seasons.
2. ~~Cross-solve churn~~ DONE 2026-07-15 (MP_REBUY_GAP sold_at ledger).
3. w̄ bench pricing outside chip weeks (waste 11/GW at H=5 vs 3.3 legacy).
4. Phase 6 sweep: Optuna over {H, δ, δ_c, θ, γ, w̄, MP_WC_BELOW, hit cost/
   budget, rebuy gap, chip-bar on/off + q} — train 2 seasons, hold out 1,
   report the fold gap.
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
