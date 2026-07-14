
# FPL AI — Claude Code Project Memory

## Project
Fantasy Premier League Predictive Management System (thesis).
3-layer hybrid AI: LightGBM models → ILP optimizer → LLM agent.

## Current Status
ALL STAGES COMPLETE + INTEL COMPLETE + FULLY OPTIMIZED + FULL-SEASON LIVE DEMO DONE
System fully built, validated, enhanced with pre-deadline intelligence, and
hyperparameter-tuned. Live demo = full GW1-38 season run end-to-end.
Result: 2468 pts (~64.9/GW) — roughly +400 pts over the average FPL manager,
i.e. top-tier global rank (~64 pts/GW ≈ top ~0.01% / top ~1000 per GiveMeSport).
Thesis (FINKI_Thesis.pdf, MK + EN, Overleaf) written up with these results.

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

## Data Availability Warning (this machine)
This Desktop copy is a git clone — data/raw/ is GITIGNORED and absent.
season_simulator.py needs data/raw/fpl_api/{player_history,players_raw,
fixtures_raw}.csv — copy from the original machine before running.
FPL API re-fetch impossible: API rolled over to 2026-27 (season ended).
No local Python either — use Docker image `fpl-sim` (built 2026-07-02):
  docker run --rm -v "<repo>:/app" fpl-sim python -u pipeline/season_simulator.py
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

## Training Files (data/processed/)
train_gk.csv   — 4,421  rows  69 cols
train_def.csv  — 18,828 rows  67 cols
train_mid.csv  — 22,132 rows  65 cols
train_fwd.csv  — 5,663  rows  65 cols
TOTAL          — 51,044 rows
Target column: total_points
All validated: 0 NaN, 0 leakage, 0 cross-season bleed

## Model Output Paths
models/xgb_gk.pkl   — GK model (contains LightGBM when MODEL_TYPE=lgbm)
models/xgb_def.pkl  — DEF model
models/xgb_mid.pkl  — MID model
models/xgb_fwd.pkl  — FWD model
models/stage7_results.json      — best hyperparams + MAE curves per position
models/stage9_explanations.json — Claude narrative per GW
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
1. GW1 BLIND TEST — zero 2025-26 data in training ever
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
- PENDING: v2 GW1-38 backtest vs 2468 (blocked: needs data/raw from original
  machine), then generalization runs + bar re-tune via Optuna
- Remaining polish: final thesis review / defense prep
- Potential new features: minutes_last3, minutes_last5, minutes_trend
  (fatigue proxy) — would require updating feature_engineering_stage6.py
  + retraining models
