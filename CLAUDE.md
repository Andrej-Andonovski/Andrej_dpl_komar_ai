
# FPL AI — Claude Code Project Memory

## Project
Fantasy Premier League Predictive Management System (thesis).
3-layer hybrid AI: LightGBM models → ILP optimizer → LLM agent.

## Current Status
ALL STAGES COMPLETE + INTEL PIPELINE COMPLETE + FULLY OPTIMIZED
System fully built, validated, enhanced with pre-deadline intelligence,
and hyperparameter-tuned via joint random search (250 trials) + Optuna (100 trials).

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
- Optuna Search ✅ 100-trial Bayesian search — trial 429 is current best

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
data/intel/optuna_search/  — Optuna results (summary.json, study.db, trial JSONs)
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

## Season Simulator Best Result (GW1-28)
  Best: 1799 pts (64.3/GW avg) — confirmed from season_simulation.json
  Model: LightGBM (Optuna trial 429 params)
  Chips: tc1 GW6, bb1 GW8, wc1 GW17, tc2 GW21, bb2 GW23, fh2 GW26
  Penalties: 0 pts

  GW-by-GW actuals:
    GW 1:  79 pts (Haaland C)
    GW 2:  83 pts (Haaland C)
    GW 3:  63 pts (Junqueira de Jesus C)
    GW 4:  98 pts (Haaland C)
    GW 5:  62 pts (Haaland C)
    GW 6: 101 pts (Haaland C, Triple Captain tc1)
    GW 7:  70 pts (Haaland C)
    GW 8:  91 pts (Haaland C, Bench Boost bb1)
    GW 9:  47 pts (Haaland C)
    GW10:  74 pts (Haaland C)
    GW11:  46 pts (Haaland C)
    GW12:  45 pts (Lomba Neto C)
    GW13:  61 pts (Haaland C)
    GW14:  60 pts (Borges Fernandes C)
    GW15:  61 pts (Haaland C)
    GW16:  82 pts (Borges Fernandes C)
    GW17:  65 pts (Borges Fernandes C, Wildcard wc1)
    GW18:  38 pts (Haaland C)
    GW19:  50 pts (Semenyo C)
    GW20:  38 pts (Garner C)
    GW21: 100 pts (Nascimento Rodrigues C, Triple Captain tc2)
    GW22:  42 pts (Garner C)
    GW23:  63 pts (Nascimento Rodrigues C, Bench Boost bb2)
    GW24:  56 pts (Borges Fernandes C)
    GW25:  62 pts (Borges Fernandes C)
    GW26:  60 pts (Borges Fernandes C, Free Hit fh2)
    GW27:  47 pts (Junqueira de Jesus C)
    GW28:  55 pts (Palmer C)

  Improvement history from baseline:
    ~429  Stage 8 baseline (no intel, no snapshots)
    ~557  + GW-snapshot features
    ~616  + Intel penalties + FDR + loyalty
    ~654  + Ownership boost + chip lockout (pre-bug-fix)
    ~598  + Penalty sign fix (corrected)
    ~629  + Auto-subs + bench weight + BB fix
     652  + TC/BB triggers (GW1-10 verified)
    1716  Random search XGB trial 1 baseline (GW1-28)
    1760  Switch to LGBM trial 220
    1799  Optuna trial 429 (current best, GW1-28)

## Current Simulator Params (pipeline/season_simulator.py — Optuna trial 429)
  MODEL_TYPE        = "lgbm"
  FDR_MULT          = 0.028       (MID/FWD fixture difficulty adjustment)
  FDR_MULT_DEF      = 0.084       (GK/DEF position-specific, more sensitive)
  OWN_BOOST_GW1     = 0.213       (GW1 ownership bonus on prediction)
  TC_THRESH         = 6.17        (captain form threshold to trigger TC)
  TC_FORM_MIN       = 6.0
  TC2_MIN_GW        = 20          (earliest GW for second TC chip)
  BB_THRESH         = 9.0         (full bench pred threshold for BB)
  BB_MIN_GW         = 8           (earliest GW for BB)
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
  LGBM: n_estimators=300, max_depth=3, lr=0.03, subsample=0.9,
        colsample=0.6, num_leaves=31, min_child_samples=30

## Hyperparameter Optimization
  Random Search (pipeline/random_search_full.py):
    250 trials, pure random sampling
    Results: data/intel/random_search_full/summary.json
    Search space: MODEL_TYPE + model hyperparams + all optimizer params
    Key finding: LightGBM dominates — 14 of top 15 results are LGBM
    XGBoost best: 1716 pts (trial 1, baseline config)
    LGBM best:    1760 pts (trial 220)

  Optuna Search (pipeline/optuna_search.py):
    100 trials, TPE Bayesian sampler (smarter than pure random)
    Trial 1 seeded with random search trial 220 params as warm start
    Results: data/intel/optuna_search/summary.json + study.db
    Best: trial 429 → 1799 pts (current production params)

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
data/intel/season_simulation.json — Season simulator GW1-28 best run (1799 pts)
data/intel/final_squad.json       — Intel 06 GW1-10 simulation log
data/intel/availability.json      — intel_03 output
data/intel/rotation_risk.json     — intel_04 output
data/intel/recommendations.json   — intel_05 output
data/intel/press_conferences.json — intel_02 output
data/intel/fpl_live.json          — intel_01 output

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
- Thesis write-up / results analysis
- Live GW30+ demo run (re-run sim_start_gw = 30)
- Potential new features: minutes_last3, minutes_last5, minutes_trend
  (fatigue proxy) — would require updating feature_engineering_stage6.py
  + retraining models
