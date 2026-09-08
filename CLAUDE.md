
# FPL AI — Claude Code Project Memory

## Project
Fantasy Premier League Predictive Management System (thesis).
3-layer hybrid AI: LightGBM models → ILP optimizer → LLM agent.

## Documentation (canonical architectural reference)
The conceptual knowledge base lives in `docs/` (Obsidian vault) — entry point
`docs/index.md`. It is the canonical architectural reference, organized as
Architecture / Components / Workflows / Decisions / Reference and linked with
Obsidian wikilinks. This file (`CLAUDE.md`) is operational/session memory; the
detailed evidence reports (`docs/HANDOFF.md`, `docs/phase*_report.md`, etc.)
remain the evidence layer that the conceptual notes cite.

Maintenance rules — apply on any significant change:
- Read the relevant note(s) before implementing.
- Update the matching note IN THE SAME CHANGE whenever architecture, workflows,
  external integrations, or design decisions change.
- Prefer updating existing notes over creating new ones; never one note per
  source file, class, or endpoint.
- Keep ONE canonical explanation per concept and link to it — do not duplicate
  (e.g. environment/score caveats → `environment-and-docker`; leakage rules →
  `walkforward-no-leakage`; tuned constants → `tuned-parameters`).
- If code and docs disagree, verify the code first, then update the docs.
Note: `AGENTS.md` is a stale duplicate of this file — do not treat as current.

## Current Status
ALL STAGES COMPLETE + INTEL COMPLETE + FULLY OPTIMIZED + FULL-SEASON LIVE DEMO DONE
System fully built, validated, enhanced with pre-deadline intelligence, and
hyperparameter-tuned. Live demo = full GW1-38 season run end-to-end.
Result: 2468 pts (~64.9/GW) — roughly +400 pts over the average FPL manager,
i.e. top-tier global rank (~64 pts/GW ≈ top ~0.01% / top ~1000 per GiveMeSport).
Thesis (FINKI_Thesis.pdf, MK + EN, Overleaf) written up with these results.
The 2468 headline number reflects the 2025-26 season (now historical, see
below) — it is not re-derived by the 2026-27 refresh described next.

## 2026-27 Season Data Refresh (2026-07-27)
2025-26 completed, so the pipeline was rolled forward one season: 2025-26 is
now historical training data (rule #1's "never train on the live season"
boundary moved from 2025-26 to 2026-27), and fresh 2026-27 squad/fixture data
was pulled for the live blind test (GW1 deadline 2026-08-21). Machine now has
`data/raw/` and native Python (no Docker needed for pipeline dev; Docker
remains the reference env for any score that must match a documented number).
Promotion/relegation vs 2025-26: **out** Burnley, West Ham, Wolves; **in**
Coventry City, Hull City, Ipswich Town.
Done:
- Stage 1 refreshed (`data/raw/fpl_api/`, old 2025-26 snapshot archived to
  `fpl_api_2025-26_final/`); vaastav repo pulled (adds full 2025-26 season);
  Stage 2/3 season lists extended to 7 seasons (2019-20..2025-26), blocked-season
  gate moved to 2026-27.
- Stage 4a (new_signings_stage4a.py): SEASONS window bumped to
  2023-24/2024-25/2025-26, Transfermarkt `saison_id` bumped to 2026 (its HTML
  cache is NOT keyed by season — must be manually archived/cleared on each
  cycle or it silently serves last year's transfer window).
- New `pipeline/promoted_clubs_stage4c.py`: handles the case Transfermarkt's
  "new arrivals" page structurally can't see — a promoted club's retained
  squad (previous league = Championship, not a real transfer). Reuses Stage
  4a's scrape/match machinery with a synthetic signings df. Run per promotion
  cycle: `python pipeline/promoted_clubs_stage4c.py --full-run`.
- Fixed real bugs found along the way (not cosmetic): (1) `new_signings_stage4a.py`'s
  VAASTAV_COLS was missing `adjG_per_90/adjA_per_90/league_multiplier/
  season_reliability/data_source/data_confidence` — Stage 6's
  `build_prev_lookup` hard-requires these and would KeyError; (2)
  `team_form_stage3.py`'s `normalise_gw` defaulted attacking/defensive_strength
  to 0.5 at GW1 (all-NaN group -> degenerate range -> wrong fallback) instead
  of 0 — a systematic ~137-row-per-position GW1 leakage-check violation,
  present since this function was written, now fixed to distinguish
  "no data yet" from "data exists but tied".
- Stage 6 re-run end to end, all 10 validation checks pass. New training
  counts below. `EXPECTED_ROWS`-style checks changed from exact-match to
  minimum-baseline (a new season should only add rows).
- Ad hoc LGBM walk-forward sanity check (production hyperparameters, no
  re-search) extended to a 6th fold (train 2019-24 -> validate 2025-26):
  mean MAE 2.39, squarely inside the 2.34-2.44 range of all prior folds —
  no degradation from the new season/signings/promoted-club data.
Not done / still open: `pipeline/train_xgboost_stage7.py` is XGBoost-only and
superseded (production trains LightGBM online from `data/processed/` every
run — see `season_simulator.py:train_models` — so no separate "retrain"
artifact step is needed for the new data to take effect); a live 2026-27
GW-by-GW run hasn't happened yet (season hasn't started).

## Completed Stages
- Stage 1 ✅ FPL API data
- Stage 2 ✅ Vaastav historical GW data
- Stage 3 ✅ Team form (vaastav + understat xG)
- Stage 4a ✅ New signings FBref scrape
- Stage 4b ✅ Debutant previous-league stats
- Stage 5 ✅ DROPPED — matchup stats not enough signal
- Stage 6 ✅ Feature engineering — training files ready
- Stage 7 ✅ LightGBM model training (walk-forward CV)
- Stage 8 ✅ ILP optimizer (PuLP) + online retraining
- Stage 9 ✅ LLM agent (Claude API) — per-GW narrative explanations
- Stage 10 ✅ LSTM residual layer (Phase 1) — corrects raw GBM μ + calibrated
  captain q90; `STAGE10=off|on` flag (default off = byte-identical no-op).
  Shipped for `OPTIMIZER=mp` (A/B +14/+40/+81 across 2023-24/24-25/25-26).
  See "Stage 10" section below + docs/stage10_phase1_report.md.
- Intel 01-07 ✅ Pre-deadline intelligence suite (see below)
- Random Search ✅ 250-trial joint search — LGBM trial 220 is best baseline
- Optuna Search ✅ GW1-28 Bayesian search — trial 429 (1799 pts GW1-28)
- Optuna GW38 ✅ Full-season search — 341 trials, best 2468 pts (trial 7011)
- Live Demo ✅ Full GW1-38 season run — 2468 pts (season_simulation.json)

## Pipeline Scripts (all complete)
pipeline/data_fetcher_stage1.py
pipeline/data_loader_stage2.py
pipeline/team_form_stage3.py
pipeline/new_signings_stage4a.py
pipeline/data_loader_stage4b.py
pipeline/feature_engineering_stage6.py
pipeline/train_xgboost_stage7.py
pipeline/ilp_optimizer_stage8.py
pipeline/llm_agent_stage9.py
pipeline/intel_01_fpl_live.py
pipeline/intel_02_press_conferences.py
pipeline/intel_03_availability.py
pipeline/intel_04_rotation_risk.py
pipeline/intel_05_recommendations.py
pipeline/intel_06_optimizer.py
pipeline/intel_07_bench.py
pipeline/intel_08_effective_ownership.py  (recommendation-layer §4.1 — top-10k EO scraper; built 2026-07-14)
pipeline/season_simulator.py
pipeline/random_search_full.py
pipeline/optuna_search.py
pipeline/stage10_oof.py         (Stage 10 — walk-forward OOF μ, residual target)
pipeline/stage10_sequence.py    (Stage 10 — leakage-safe GW sequence builder)
pipeline/stage10_model.py       (Stage 10 — torch LSTM + numpy runtime forward)
pipeline/stage10_train.py       (Stage 10 — 5 walk-forward folds, torch, offline)
pipeline/stage10_infer.py       (Stage 10 — torch-free checkpoint loader + q90)
pipeline/stage10_refine.py      (Stage 10 — runtime hook into predict_pool/build_matrix)
pipeline/stage10_finetune.py    (Stage 10 — in-season head fine-tune; DISABLED, measured negative)

## File Structure
data/raw/fpl_api/          — FPL API files
data/raw/vaastav/          — historical GW data
data/raw/vaastav_repo/     — vaastav git repo clone
data/raw/fbref/            — FBref scraped data
data/raw/fbref/new_signings/ — stage 4a/4b position files
data/raw/transfers/        — transfermarkt signings
data/processed/            — training files + team form
data/intel/                — live intel outputs (JSONs + season_simulation.json)
data/intel/archive/        — old sweep results, best_* backups, trial runs
data/intel/random_search_full/ — 250-trial search results (summary.json)
data/intel/optuna_search/  — Optuna GW1-28 results (summary.json, study.db, trial JSONs)
data/intel/optuna_search_gw38/ — Optuna GW1-38 full-season search (341 trials, best 2468)
pipeline/                  — all core pipeline scripts
pipeline/archive/          — dev/one-off scripts (sweeps, patches, verifiers)
scripts/                   — analysis scripts (bench reports, form sweeps)
models/                    — trained models + stage9 results
models/stage10/            — Stage 10 LSTM checkpoints + OOF + calibration JSONs
ui/                        — Flask UI (server.py + index.html)

## Intel Pipeline Architecture (intel_01 through intel_07)
Pre-deadline intelligence suite that gathers real-time data and feeds
it into the season simulator to improve squad decisions.

intel_01: FPL live data fetch (injuries, prices, ownership, transfer pressure)
intel_02: Press conference scraping (Fantasy Football Scout, per-GW articles)
intel_03: Player availability assessment — merges intel_01 + intel_02
          into per-player 0-100 availability score per GW
          Merge: 65% press score + 35% FPL score; +5 if both sources agree
intel_04: Rotation risk scoring (0-100) per player per GW
          Signals: start rate, minutes volatility, bench rate, recent trend, press keywords
intel_05: LLM-powered recommendations (Gemini 2.5 Flash API)
          Outputs: captain pick, differentials, transfer targets, risk warnings
intel_06: Enhanced season simulator — wraps ILP with intel penalties,
          availability/rotation multipliers, chip timing, auto-subs, loyalty bonuses
intel_07: Bench intelligence — lookahead bench boost targeting, bench candidate scoring

## Intel Penalty Formula (intel_06 / season_simulator)
  avail_mult  = availability_pct / 100
  rot_mult    = 0.40 if rotation_risk >= 80
              = 0.60 if rotation_risk >= 60
              = 0.80 if rotation_risk >= 40
              = 1.00 otherwise
  combined    = avail_mult * rot_mult
  adjusted_pred = original_pred * combined

## Season Simulator Best Result (GW1-38 — full-season live demo)
  Best: 2468 pts (~64.9/GW avg) — confirmed from season_simulation.json
  Model: LightGBM
  Chips: tc1 GW6, bb1 GW8, wc1 GW17, bb2 GW21, tc2 GW23, fh2 GW26
  Penalties: -12 pts (one -4 hit region; total_predicted ≈ 2782)
  Context: ~+400 pts over average FPL manager; ~64 pts/GW ≈ top ~0.01%
           (~top 1000 globally per GiveMeSport benchmark).

  GW-by-GW actuals (actual_total, captain):
    GW 1:  85 (Haaland)            GW20:  40 (Calvert-Lewin)
    GW 2:  54 (Haaland)            GW21:  84 (Garner, BB bb2)
    GW 3:  67 (Junqueira de Jesus) GW22:  44 (Nascimento Rodrigues)
    GW 4:  95 (Haaland)            GW23:  71 (Semenyo, TC tc2)
    GW 5:  56 (Borges Fernandes)   GW24:  63 (Borges Fernandes)
    GW 6: 110 (Haaland, TC tc1)    GW25:  56 (Bowen)
    GW 7:  56 (Haaland)            GW26:  62 (dos Santos Magalhães, FH fh2)
    GW 8:  85 (Haaland, BB bb1)    GW27:  45 (Junqueira de Jesus)
    GW 9:  67 (Haaland)            GW28:  70 (Borges Fernandes)
    GW10:  58 (Semenyo)            GW29:  75 (Garner)
    GW11:  39 (Haaland)            GW30:  79 (Borges Fernandes)
    GW12:  46 (Haaland)            GW31:  59 (Borges Fernandes)
    GW13:  51 (Haaland)            GW32:  67 (Borges Fernandes)
    GW14:  76 (Borges Fernandes)   GW33:  93 (Truffert)
    GW15:  58 (Haaland)            GW34:  45 (Borges Fernandes)
    GW16:  79 (Borges Fernandes)   GW35:  57 (Calvert-Lewin)
    GW17:  87 (Haaland, WC wc1)    GW36:  66 (Haaland)
    GW18:  54 (Haaland)            GW37:  74 (Calvert-Lewin)
    GW19:  44 (Haaland)            GW38:  51 (Haaland)

  Improvement history from baseline:
    ~429  Stage 8 baseline (no intel, no snapshots)          [GW1-28]
    ~557  + GW-snapshot features                             [GW1-28]
    ~616  + Intel penalties + FDR + loyalty                  [GW1-28]
    ~654  + Ownership boost + chip lockout (pre-bug-fix)     [GW1-28]
    ~598  + Penalty sign fix (corrected)                     [GW1-28]
    ~629  + Auto-subs + bench weight + BB fix                [GW1-28]
     652  + TC/BB triggers (GW1-10 verified)                 [GW1-28]
    1716  Random search XGB trial 1 baseline                 [GW1-28]
    1760  Switch to LGBM trial 220                           [GW1-28]
    1799  Optuna trial 429                                   [GW1-28]
    2468  Full-season run + GW38 Optuna (341 trials)         [GW1-38] ← current best

## Current Simulator Params (pipeline/season_simulator.py — production, GW1-38 → 2468 pts)
  MODEL_TYPE        = "lgbm"
  SIM_END_GW        = 38          (full-season run)
  FDR_MULT          = 0.0285      (MID/FWD fixture difficulty adjustment)
  FDR_MULT_DEF      = 0.084       (GK/DEF position-specific, more sensitive)
  OWN_BOOST_GW1     = 0.213       (GW1 ownership bonus on prediction)
  TC_THRESH         = 6.17        (captain form threshold to trigger TC)
  TC_FORM_MIN       = 6.0
  TC2_MIN_GW        = 20          (earliest GW for second TC chip)
  FH2_EARLIEST_GW   = 20          (earliest GW for second Free Hit)
  BB_THRESH         = 9.0         (full bench pred threshold for BB)
  BB_MIN_GW         = 8           (earliest GW for BB)
  BB_MAX_GW_SET1/2  = 19 / 38     (BB set deadlines)
  DGW_PRED_MULT     = 2.0         (boost predictions for DGW players)
  BENCH_BONUS_NORMAL = 2.71       (bench candidate prediction boost)
  BENCH_BONUS_BB_GW  = 2.25       (bench boost on BB GW)
  CAP_FORM_GATE     = 6.57        (min form for captain consideration)
  CAP_FORM_PENALTY  = 0.574       (penalty if captain below form gate)
  CAP_STREAK_LIMIT  = 2           (max consecutive GWs same captain)
  CAP_STREAK_MULT   = 0.899       (prediction multiplier after streak)
  CAP_FDR_MULT      = 0.009       (captain FDR adjustment)
  CAP_BLANK_PENALTY = 0.757       (captain prediction penalty on blank GW)
  CAP_BLANK_THRESH  = 4           (FDR threshold to apply blank penalty)
  WC_THRESH         = 5           (squad members below pos avg → trigger WC)
  MC_SQUADS         = 3           (Monte Carlo random squad comparison)
  Loyalty bonus: GW1-5: 10.0, GW6-10: 2.0, GW11+: 1.0
  LGBM: n_estimators=200, max_depth=3, lr=0.0439, subsample=0.932,
        colsample_bytree=0.824, num_leaves=31, min_child_samples=27
  Note: GW38 Optuna trial 7011 ties this at 2468 with different params
        (max_depth=4, lr=0.0156, num_leaves=63) — 2468 is a robust ceiling.

## Chip Strategy v2 (IMPLEMENTED 2026-07-02 — backtest pending)
Design: docs/chip_strategy_redesign.md. Replaces the hardcoded GW17/18/19
chip policy with a rolling-horizon, calendar-agnostic scheduler:
  - CHIP_STRATEGY flag in season_simulator.py: "v2" (default) | "legacy"
    (old policy kept for thesis ablation)
  - No hardcoded GWs beyond FPL set boundaries (Set1 GW1-19, Set2 GW20-38)
  - Per-chip value functions: BB = bench pred sum, TC = captain marginal x1
    (natural trigger unchanged — it delivered both TCs in the 2468 run),
    FH = budget-true ILP temp-XI gain (event weeks only: blanks/doubles),
    WC = rolling ILP rebuild gain summed over WC_HORIZON=5 GWs
  - Event-aware planning: candidate weeks = lookahead(4) + ALL known
    double/blank GWs left in the set (reserves BB/FH for far events)
  - Constraints: one chip/GW; WC<->FH >= SPACING_GAP(4) GWs apart
    (structural fix for the manual FH GW23 -> WC GW24 mistake)
  - Deadline pressure fires best remaining week by value (use-it-or-lose-it)
  - Bars (Optuna-tunable): BB 14, FH 16, WC 20, TC = TC_THRESH
  NOT YET VALIDATED: needs GW1-38 backtest vs 2468 baseline + generalization
  run on 2023-24 / 2024-25 calendars (see design doc §10).

## Optimizer Redesign — CHIP SCARCITY FIX SHIPPED 2026-07-14
docs/chip_scarcity_fix.md — lockout + TC/BB far-DGW guard + WC replacement-
level gate (MP_WC_BELOW=4). A/B (legacy opt / mp+legacy-chips / mp+model-chips):
  2025-26: 2252 / 2156 / 2029   2023-24: 2164 / 2162 / **2174** (8/8 chips)
  2024-25: 2359 / 2341 / **2410** ← PROJECT BEST on any season
On neutral calendars mp+model-chips now beats the tuned legacy system
(mean 2292 vs 2261). Best real-season config: OPTIMIZER=mp MP_HORIZON=5
MP_CHIPS=model. Tests 15/15. Remaining: eventless-set unanchored chips
(percentile bar, Phase 6), captain channel, churn, w̄, final sweep.

## Optimizer Redesign — GENERALIZATION PROVEN 2026-07-14
docs/generalization_report.md — the thesis-critical result:
  legacy vs mp:  2025-26 (home): 2252 vs 2156 (−96)
                 2023-24: 2164 vs 2162 (−2) | 2024-25: 2359 vs 2341 (−18)
Legacy's edge is ~85-90% memorized calendar; mp travels untuned.
Cross-season harness: SIM_SEASON env (paths/snapshot/train-cut/intel-off,
corrected-only) + pipeline/build_season_inputs.py from vaastav_repo per-
fixture data (downloaded 2026-07-14; element_type-5 AM filter for 2024-25).
SIM_END_GW env-overridable for smokes. mp recovered wc2 on both neutral
seasons. Next: chip scarcity fix A/B on all 3 calendars (2023-24 best bed:
Set-1 DGW7 + six blanks), then captain channel, then Phase 6 tuning
(train 2 seasons / hold out 1 — harness ready).

## Optimizer Redesign — Phases 3+4 COMPLETE 2026-07-14 (build-all done)
Reports: docs/phase3_report.md, docs/phase4_report.md. Scoreboard (corrected
rules, Docker): baseline 2252 | P2 H=1 2070 | P3 H=5+legacy-chips **2156**
(best mp) | P4 in-model chips 1955.
Phase 3 (solve_horizon, MP_HORIZON env, 12/12 tests): banking + horizon-priced
hits proven (hit ROI +5.2), BB coordination (+25/+16), zero solve failures.
Phase 4 (chip vars + FH shadow + reservation guard, MP_CHIPS env, 11/11
tests): event-anchored chips place perfectly (fh2@DGW26, bb2@DGW33 held by
guard through 13 weeks); UNANCHORED chips burn at first eligibility
(tc1@GW1, wc1@GW2, wc2@GW20) — chip scarcity beyond horizon unpriced; broke
hit quality downstream (ROI −6.8). Best config: OPTIMIZER=mp MP_HORIZON=5
MP_CHIPS=legacy.
FIXING BACKLOG (Phase 5/6, cross-season 2023-24/2024-25 validation):
1) chip scarcity pricing (guard extension to TC/WC / lockout / reserve
values), 2) captain channel ~50 pts, 3) cross-solve churn, 4) w̄ bench
pricing, 5) q90/π calibration re-measure.

## Optimizer Redesign — Phase 2 COMPLETE 2026-07-14 (docs/phase2_report.md)
Single-GW MILP (pipeline/milp_core.py, 12/12 tests, HiGHS) behind
OPTIMIZER="mp" env flag (requires RULES_MODE=corrected; output
season_simulation_corrected_mp.json). New objective: bench EV + in-ILP
captain/vice (vice armband fallback now in scoring) + sell-value pricing.
RESULT: 2070 vs 2252 baseline (−8.1%, gate −3% NOT met — documented decision:
no H=1 hand-tuning; proceed to Phase 3). Ablation chain: v1 2031 → +GW1
community prior OWN_PRIOR_GW1=0.213 (+86 pts GW1-2, blueprint-sanctioned) +
empirical-headroom q90 → 2070. Remaining gap attributed: captain ~−35,
bench overspend ~−50, churn/hits ~−50 (18 short holds, FT=1 at 29/34
deadlines) — all Phase 3/6 targets. Wins proven: 7/7 chips (wc2 recovered),
hits guarded (−8), rules exact, MAE better (3.21 vs 3.89).
Next: Phase 3 — multi-period H>1 MILP (transfers over horizon, FT banking).

## Optimizer Redesign — Phase 1 COMPLETE 2026-07-14 (docs/phase1_report.md)
Prediction matrix (pipeline/prediction_matrix.py, 12/12 tests) + walk-forward
calibration (pipeline/phase1_calibration.py → data/intel/phase1_calibration.json):
  - HORIZON IS TRUSTWORTHY: MAE +4.1% only from h=0 (2.044) to h=5 (2.128),
    Spearman 0.72→0.63. Risk R1 low; H=5-6 supported; δ=0.84 too pessimistic
    (measured decay supports δ≈0.90-0.97, finalize in Phase 6 ablation)
  - Gate h=0 vs legacy PASS: matrix 2.044/2.843 (top60 BETTER than baseline
    2.040/2.863) — FDR post-multipliers + DGW×2.0 safely deletable
  - Blank/DGW exact: 0 violations, 417 DGW cells per offset (per-fixture sums)
  - φ gate FAIL (magnitude-confounded inversion) → φ≡1 in Phase 2 MILP;
    revisit Phase 5 with μ-matched buckets
  - q90 coverage 0.836 < 0.90 → raise Z90 ≈1.65, re-measure in Phase 3 run
Next: Phase 2 — single-GW MILP with new objective (bench EV, captain/vice
in-ILP, corrected rules, no fudge constants), compare vs 2252 fair baseline.

## Optimizer Redesign (blueprint + Phase 0 — COMPLETE 2026-07-14)
Blueprint: docs/optimizer_redesign.md — multi-period MILP (H=5 rolling),
chips/captain/vice/bench in one program, 8 honest constants. Build plan §9.
Phase 0 RESULTS (docs/phase0_baseline.md — data/raw copied 2026-07-14):
  - FAIR BASELINE = 2252 pts (corrected rules, legacy chips, Docker env)
  - Docker legacy repro = 2236, bit-identical across two runs (deterministic)
  - 2468 is ENVIRONMENT-BOUND: Docker diverges at GW1 (LightGBM stack
    difference on original machine) — all future comparisons in Docker only.
    fpl-sim image updated 2026-07-14: + scikit-learn 1.9.0 + highspy (committed)
  - Corrected beats legacy in-env (+16): sell-on rule costs < 5-FT banking gains
  - Metrics (data/intel/metrics_{legacy_docker,corrected}.json): transfer 4-GW
    payoff ≈ 0 (corrected: −0.55/transfer, 51.7% positive), FT=1 at 23-27 of
    ~30 deadlines, BB chips worth +2..+9, captain regret 6.5-7.3/GW — the
    quantified targets for Phases 2-4.
Phase 0 implementation:
  - pipeline/fpl_rules.py — pure rule accounting (50% sell-on, FT 1..5,
    RULE_EVENTS_FT config) + tests/test_fpl_rules.py (plain python, no pytest)
  - season_simulator.py: RULES_MODE env flag ("legacy" default | "corrected");
    CHIP_STRATEGY now env-overridable. Corrected = purchase-price ledger,
    owned players at sell value in ILP (ilp_price), no budget relaxation
    (raises), real FT banking. Corrected output goes to
    season_simulation_corrected.json (never clobbers production JSON).
  - pipeline/backtest_metrics.py — §10.3 metrics from any sim log
    (add --history player_history.csv for transfer counterfactuals)
Run + exit criteria: docs/phase0_baseline.md. Gate 1: legacy repro must
still total exactly 2468. Gate 2: corrected run = the fair baseline number.

## Data Availability (this machine, updated 2026-07-27)
data/raw/ is present and native Python works (pandas/lightgbm/xgboost/sklearn/
pulp/highspy/optuna all installed) — Docker is no longer required for pipeline
dev, only as the reference env for reproducing a documented score exactly.
FPL API now serves 2026-27 (2025-26 has ended) — re-fetching 2025-26 raw data
is impossible, but the completed season lives on in the vaastav repo and was
folded into training in the 2026-27 refresh (see above).
Pre-v2 production result backed up:
  data/intel/archive/season_simulation_legacy_2468.json

## Hyperparameter Optimization
  Random Search (pipeline/random_search_full.py):
    250 trials, pure random sampling
    Results: data/intel/random_search_full/summary.json
    Search space: MODEL_TYPE + model hyperparams + all optimizer params
    Key finding: LightGBM dominates — 14 of top 15 results are LGBM
    XGBoost best: 1716 pts (trial 1, baseline config)
    LGBM best:    1760 pts (trial 220)

  Optuna Search — GW1-28 (pipeline/optuna_search.py):
    TPE Bayesian sampler (smarter than pure random)
    Trial 1 seeded with random search trial 220 params as warm start
    Results: data/intel/optuna_search/summary.json + study.db
    Best: trial 429 → 1799 pts (GW1-28 scope)

  Optuna Search — GW1-38 full season (data/intel/optuna_search_gw38/):
    341 completed trials, scope "GW1-38 full season"
    Results: data/intel/optuna_search_gw38/summary.json + study.db + trial_*.json
    Best: trial 7011 → 2468 pts, -12 penalties
          chips tc1/6 bb1/8 wc1/17 bb2/21 tc2/23 fh2/26
    This is the full-season ceiling and matches the production run.

## Stage 9 Architecture (LLM Narrative Layer)
Per-GW explanations via Claude API (post-simulation analysis):
  - Reads season_simulation.json GW-by-GW squad decisions
  - Calls Claude once per GW to explain why each player was picked
  - Output: models/stage9_explanations.json (narrative per GW)
  - MODEL_ID: claude-sonnet-4-20250514
  - MAX_TOKENS: 1200, TEMPERATURE: 0
  Note: Stage 9 is explanatory only — decisions are made by intel_06/simulator.

## Stage 10 — LSTM Residual Layer (Phase 1, shipped 2026-09-08)
Stacked correction between the LightGBM μ and the optimizer. Full evidence:
docs/stage10_phase1_report.md (definitive A/B matrix + post-ship refinement log,
fixes 1-7); plan: docs/stage10_phase1_plan.md; component note:
docs/components/stage10-residual-layer.md.
  Flow:  LightGBM μ -> LSTM(μ, GW-sequence 1..t-1) -> (r̂, σ)
         r_applied = confidence_gate(r̂)  added to raw μ  (FDR/intel stay downstream)
         q90       = μ + r_applied + z·σ_eff   -> legacy captain only (mp uses pure EV)
  Flag:  STAGE10=off (default, TRUE no-op — stage10_refine never imported so
         torch never loads; predict_pool/build_matrix byte-identical) | on
         (writes *_s10.json).
  Training:  torch, OFFLINE (stage10_train.py), 5 walk-forward folds
             (val 2021-22..2025-26; 2019-20 has no OOF residual). Runtime
             inference is a dependency-free numpy forward (NumpyResidualLSTM) —
             season_simulator + Optuna loop never import torch. torch==2.14.0
             is a training-only dep (Smart App Control whitelisted the DLLs
             after first run; WSL2 is the fallback if it re-blocks).
  Checkpoints:  models/stage10/pretrain_<S>.{pt,npz} + stage10_config.json +
             stage10_calibration.json. Live 2026-27 loads pretrain_2025-26.npz.
  Gates:  1 (off byte-identical, all 4 configs) PASS; 2 (OOF MAE) mean +0.026,
          all folds positive; 3 (q90 coverage) 0.836 -> 0.890; 4 (A/B) see below;
          8 (determinism) PASS. Tests: test_stage10_{sequence,identity,shapes}.py.
  A/B (definitive 15-run matrix, STAGE10 off -> on, all fixes applied):
             mp  2023-24 +28 | 2024-25 +56 | 2025-26 +81   (MEAN +55/season)
                 decomposition: LSTM residual ~+28 (+37/+33/+14)
                              + pure-EV captain obj (fix 6, MP_THETA 0.3->0) ~+27
             legacy 2023-24 -53 | 2024-25 +0 | 2025-26 +19  (mean -11, inconsistent,
                 one real regression — kept functional, mp recommended)
  SHIPPING CONFIG:  STAGE10=on OPTIMIZER=mp MP_THETA=0 RULES_MODE=corrected.
             select_captain got a CAP_Q90_W=0.35 kappa-analogue blend (legacy
             only; ILP untouched). mp captain = pure EV (MP_THETA=0): removing the
             q90 ceiling tilt is theoretically correct for season-long points max
             and +23..+27/season on the A/B. q90 head now vestigial for mp, kept
             for legacy + Phase 2 GNN.
  Post-ship fixes:  1 (FWD sigma tail) = inference SIGMA_CAP=15 clamp ONLY; the
             training-side tail penalty was REVERTED after it silently cost ~60
             A/B pts (gate-only check missed it) -> checkpoints byte-identical to
             step 3. 2 (o_bps train/serve gap) closed. 3/4/5 (penalty feature,
             cheap ceiling features, pinball q90 head) all RULED OUT with data:
             confirmed 3x that the captain problem is an OBJECTIVE problem not a
             sigma-formulation problem -> motivated fix 6. 6 shipped. 7 (conformal
             cal) pending, low priority.
  In-season head fine-tune (stage10_finetune.py): built, MEASURED NEGATIVE
             (holdout MAE -0.001..-0.054), DISABLED (STAGE10_FT=on to experiment;
             ft_*.npz gitignored).
  Phase 2:  GNN over player/team/fixture nodes on the LSTM embeddings (planned).

## Training Files (data/processed/) — updated 2026-07-27, 7 seasons (2019-20..2025-26)
train_gk.csv   — 5,174  rows  73 cols
train_def.csv  — 22,108 rows  71 cols
train_mid.csv  — 25,975 rows  69 cols
train_fwd.csv  — 6,563  rows  69 cols
TOTAL          — 59,820 rows
Target column: total_points
All validated: 0 NaN, 0 leakage (11/11 Stage 6 checks pass), 0 cross-season bleed
Col counts include the 4 career_* features added 2026-09-04 (docs/player_identity_features.md §2).

## Model Output Paths
models/xgb_gk.pkl   — GK model (contains LightGBM when MODEL_TYPE=lgbm)
models/xgb_def.pkl  — DEF model
models/xgb_mid.pkl  — MID model
models/xgb_fwd.pkl  — FWD model
models/stage7_results.json      — best hyperparams + MAE curves per position
models/stage9_explanations.json — Claude narrative per GW
models/stage10/pretrain_<S>.npz — Stage 10 LSTM checkpoints (numpy, runtime path)
models/stage10/oof_preds.csv    — Stage 10 walk-forward OOF μ + residual target
data/intel/season_simulation*_s10.json — STAGE10=on runs (never clobber baselines)
data/intel/season_simulation.json — Season simulator GW1-38 full-season run (2468 pts)
data/intel/final_squad.json       — Intel 06 GW1-10 simulation log
data/intel/availability.json      — intel_03 output
data/intel/rotation_risk.json     — intel_04 output
data/intel/recommendations.json   — intel_05 output
data/intel/press_conferences.json — intel_02 output
data/intel/fpl_live.json          — intel_01 output
data/intel/effective_ownership.json — intel_08 output (top-10k EO, latest snapshot)
data/intel/eo_history/gw{N}.json  — intel_08 per-GW EO archive (cannot backfill)

## Critical Rules — Never Break
1. GW1 BLIND TEST — zero 2026-27 data in training ever (boundary moves
   forward one season each cycle; 2025-26 is now historical, see above)
2. NO LEAKAGE — all features must be knowable before GW kickoff
3. NO CROSS-SEASON BLEED — rolling windows partition by season
4. 4 SEPARATE MODELS — one per position, never mix
5. WALK-FORWARD VALIDATION — train on seasons 1-N, validate N+1
   Never shuffle. Always respect temporal order.
6. CONFIRMATION GATE after every step — never auto-advance
7. ONLINE RETRAINING — full retrain each GW with actuals appended (not River/incremental)
8. FPL FREE TRANSFER CAP — max 5 banked free transfers (2025-26 rules)
9. PENALTY SUBTRACTION — transfer hits are SUBTRACTED not added

## Validation Strategy (Stage 7)
Walk-forward cross-validation by season:
  Fold 1: train 2019-20        → validate 2020-21
  Fold 2: train 2019-21        → validate 2021-22
  Fold 3: train 2019-22        → validate 2022-23
  Fold 4: train 2019-23        → validate 2023-24
  Fold 5: train 2019-24        → validate 2024-25
Final model: train all 6 seasons → predict GW1 2025-26
Fold weights: [1, 1.5, 2, 2.5, 3] (recent seasons weighted more)

## Evaluation Metrics
Primary:   MAE (mean absolute error on total_points)
Secondary: Top-N accuracy (did top predicted players score well)
Tertiary:  Feature importance plots per position

## Columns to EXCLUDE from model features
name, season, GW, team, opponent_team, position,
was_home, fdr_is_proxy, trajectory_is_full

## Key Feature Groups
Rolling player:  form_last3, form_last5, avg_points_per_game_season,
                 goals_per_game_season, assists_per_game_season,
                 clean_sheet_rate_season, minutes_reliability_season,
                 points_per_million
Prev league:     has_prev_league_data, prev_adjG_per_90,
                 prev_adjA_per_90, prev_league_multiplier,
                 prev_seasons_available, prev_reliability_avg,
                 prev_minutes_avg, prev_small_sample
Career quality:  career_ppg_last_season, career_ppg_last3_seasons,
                 career_minutes_reliability_last_season,
                 career_seasons_established
                 (added 2026-09-04, docs/player_identity_features.md §2 —
                 multi-season prior-PL-season summary; rolling player
                 features above reset every season, so this is the only
                 signal that tells an established player apart from a
                 rookie early in a live season)
Team form:       team_xG_last5, team_xGA_last5, team_xG_season_avg,
                 team_xGA_season_avg, team_attacking_strength,
                 team_defensive_strength, team_cs_probability
Opp form:        opp_xG_last5, opp_xGA_last5, opp_attacking_strength,
                 opp_defensive_strength, opp_cs_probability
Fixture:         current_gw_fdr, fixture_trajectory_score,
                 home_advantage
Market:          transfers_in, transfers_out, selected, value
GK only:         saves, saves_per_game_season,
                 prev_saves_per_game, prev_cs_rate
DEF only:        prev_int_per_90, prev_tklW_per_90

## Bugs Found & Fixed
- No cross-season player-identity signal (2026-09-04): live-run GW3 2026-27
  recommendation for a real squad had the mp optimizer proposing to sell
  Haaland (15 pts across GW1-2, elite fixture) for a cheaper player, because
  every rolling feature (form_last3, avg_points_per_game, etc.) is computed
  from the CURRENT live season's actuals only — at GW3 that's 2 data points
  for a rookie and a 4-season incumbent alike. Root-caused via
  docs/player_identity_features.md; fixed by adding 4 career_* features
  (multi-season prior-PL-season PPG/reliability/established-count) — see
  "Key Feature Groups" above. Confirmed fixed: re-running the same live
  recommendation with retrained models, the optimizer now holds Haaland.
- Penalty sign error: season_simulator was ADDING penalties instead of
  SUBTRACTING — inflated reported scores. Fixed.
- Player name encoding: accented names (Raya Martin, Ekitike) caused
  0-point lookups. Fixed by switching to player_id-based lookups.
- Free transfer cap: was limited to max 2, fixed to max 5 per FPL rules.
- Excessive early transfers: GW2 had 6+ transfers with -16 penalty.
  Fixed with chip lockout (GW1-4), dynamic loyalty bonus, and
  zero-hit enforcement re-run.
- Bench Boost never triggered: BB used result.get("bench_outfield", [])
  but ILP result has no bench_outfield key, so bench pred was ~GK only.
  Fixed by computing bench = [p in squad if p not in xi].
- TC never triggered: trigger used raw pred; captain display uses
  position-adjusted (FWD x1.25). Fixed by using position multipliers
  for TC check.
- DGW actual points undercounted: load_player_history() used dict assignment
  so for DGW weeks (2 rows per player per GW), the second row silently
  overwrote the first. Fixed by accumulating additive stats across both rows.
  Affected GW26 (+10 pts recovered).

## Known Limitations
- intel_02 (press conference scraper) has popularity bias: only scrapes clubs
  that appear in FFS article headers. Newcastle never appeared as a section
  header — their injured players (Bruno Guimaraes, Schar, Livramento, Krafth)
  are mentioned inline under other clubs' sections and go undetected.
  A cross-club name-matching fallback was tested in intel_03 but caused
  cascading squad changes (wildcard at GW12 instead of GW17) that cost ~113 pts
  overall. Left as known limitation — documented for thesis.
- Season simulator does not use intel_03 availability data directly for
  transfer decisions. Injured players not covered by press conferences
  (e.g., Newcastle players) may stay in predicted squad.
- FH trigger only fires on DGWs. Blank GW scenarios (e.g., AFCON) do not
  trigger Free Hit automatically.
- Sell-buyback: ILP has no memory of last week's transfers, so it can
  sell and re-buy the same player in consecutive GWs. A sellback penalty
  was tested but hurt overall score.

## How We Work
- One stage at a time, one step at a time
- Confirmation gate after every step before proceeding
- Full prompts provided — never start a stage without a prompt
- Paste all output back for review before moving on
- Never auto-advance between steps or stages

## Next Steps
- ✅ DONE: Full GW1-38 season live demo (2468 pts)
- ✅ DONE: Thesis write-up (FINKI_Thesis.pdf, MK + EN, Overleaf) with results
- ✅ DONE: Chip strategy v2 implementation (see section above)
- ✅ DONE: Stage 10 Phase 1 (LSTM residual layer) — shipped for OPTIMIZER=mp
- PENDING: Stage 10 Phase 2 (GNN on the LSTM embeddings) — design in
  docs/stage10_phase1_plan.md; held-out hyperparameter fold to de-risk the
  Phase 1 in-sample tuning
- PENDING: v2 GW1-38 backtest vs 2468 (blocked: needs data/raw from original
  machine), then generalization runs + bar re-tune via Optuna
- Remaining polish: final thesis review / defense prep
- Potential new features: minutes_last3, minutes_last5, minutes_trend
  (fatigue proxy) — would require updating feature_engineering_stage6.py
  + retraining models

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

Division of responsibilities (do not conflate):
- `docs/` (Obsidian vault, entry `docs/index.md`) remains the CANONICAL reference for
  architecture, design decisions, workflows, and intent — prefer it over
  `graphify-out/GRAPH_REPORT.md` for the "why" and the conceptual model.
- Graphify (query/path/explain/affected/god-nodes) is the authority for IMPLEMENTATION
  relationships: callers, callees, imports/dependencies, shortest paths, and impact
  analysis ("what breaks if I change X").
- Source code is the FINAL authority. If docs, graph, and code disagree, verify against
  the code, then update the docs (per the maintenance rules above).
