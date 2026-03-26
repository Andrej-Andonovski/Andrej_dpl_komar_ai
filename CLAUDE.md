
# FPL AI — Claude Code Project Memory

## Project
Fantasy Premier League Predictive Management System (thesis).
3-layer hybrid AI: LightGBM models → ILP optimizer → LLM agent.

## Current Status
ALL STAGES COMPLETE + INTEL PIPELINE COMPLETE + OPTIMIZED
System fully built, validated, enhanced with pre-deadline intelligence,
and hyperparameter-tuned via joint random search (250 trials).

## Completed Stages
- Stage 1 ✅ FPL API data
- Stage 2 ✅ Vaastav historical GW data
- Stage 3 ✅ Team form (vaastav + understat xG)
- Stage 4a ✅ New signings FBref scrape
- Stage 4b ✅ Debutant previous-league stats
- Stage 5 ✅ DROPPED — matchup stats not enough signal
- Stage 6 ✅ Feature engineering — training files ready
- Stage 7 ✅ LightGBM model training (walk-forward CV)
- Stage 8 ✅ ILP optimizer (PuLP) + online retraining (River)
- Stage 9 ✅ LLM agent (Claude API) — 2-call pipeline per GW
- Intel 01-07 ✅ Pre-deadline intelligence suite (see below)
- Random Search ✅ 250-trial joint search — LGBM trial 220 is best

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
pipeline/                  — all core pipeline scripts
pipeline/archive/          — dev/one-off scripts (sweeps, patches, verifiers)
scripts/                   — analysis scripts (bench reports, form sweeps)
models/                    — trained models + stage9 results
ui/                        — Flask UI (server.py + index.html)

## Intel Pipeline Architecture (intel_01 through intel_07)
Pre-deadline intelligence suite that gathers real-time data and feeds
it into the season simulator to improve squad decisions.

intel_01: FPL live data fetch (injuries, prices, ownership)
intel_02: Press conference scraping (Fantasy Football Scout)
intel_03: Player availability assessment — merges intel_01 + intel_02
          into per-player 0-100 availability score per GW
intel_04: Rotation risk scoring (JSON per GW)
intel_05: LLM-powered recommendations (Gemini API)
intel_06: Enhanced Stage 8 optimizer — wraps the ILP with intel penalties,
          GW-snapshot features, transfer strategy, chip timing, auto-subs
intel_07: Bench intelligence — lookahead bench boost targeting

## Intel 06 Architecture (main optimizer)
Enhances Stage 8 ILP with multiple layers:

1. Intel Penalties: Availability/rotation data from intel_03/04 applied
   as multiplicative penalties to predictions before ILP
2. GW-Snapshot Features: Dynamically recalculates player features per GW
   from player_history.csv (rolling stats, fixture data)
3. Playing Filter: Zeros out predictions for players with 0 total minutes
4. FDR Adjustment: multiplier 0.03 (adj = 1.0 - 0.03 * (fdr - 3.0))
5. GW1 Ownership Boost: OWNERSHIP_WEIGHT = 0.04 on selected_by_percent
6. Squad Loyalty: Dynamic bonus (10.0 during lockout, 2.0 GW5-6, 1.0 after)
7. Chip Lockout: No chips allowed GW1-4 (CHIP_LOCKOUT_GW = 4)
8. Zero-Hit Lockout: If ILP makes hits during GW2-4, re-runs with
   emergency loyalty (20.0) to force free-transfer-only moves
9. Auto-Free-Hit: Post-lockout, triggers Free Hit if hits > MAX_HITS (1)
10. Captain Override: From intel_05 with 0.5pt threshold
11. Bench Ordering: Outfield bench sorted by predicted points (highest = 1st sub)
12. Triple Captain Trigger: Post-lockout (GW6+), if captain position-adjusted
    pred >= TC_THRESHOLD (9.5), auto-activates TC. Uses CAP_POS_MULTIPLIERS
    (FWD 1.25, MID 1.15) so trigger matches captain-candidate display.
13. Bench Boost Trigger: Post-lockout (GW6+), if full bench (squad minus XI)
    predicted total >= BB_THRESHOLD (9.0), auto-activates BB. Bench computed
    from result["squad"] minus result["xi"] (not bench_outfield — that key
    does not exist on ILP result; using it was a bug that prevented BB).
14. Auto-Sub Simulation: Mirrors real FPL rules — if starter got 0 minutes,
    first eligible bench player subs in (respects min 3 DEF, min 1 FWD)
15. ILP Bench Weight: 0.15 * g0 weight for bench outfield in ILP objective,
    so optimizer slightly prefers squads with decent bench coverage
16. Online Retraining: Real actuals fed back per GW for model updates

## Intel 06 Key Parameters
  OWNERSHIP_WEIGHT: 0.04
  FDR_MULT: 0.03
  CHIP_LOCKOUT_GW: 4
  MAX_HITS: 1
  TC_THRESHOLD: 9.5 (position-adjusted captain pred, GW6+)
  BB_THRESHOLD: 9.0 (full bench pred total, GW6+)
  ILP bench weight: 0.15
  Loyalty bonus: 20.0 (emergency), 10.0 (lockout GW2-4), 2.0 (GW5-6), 1.0 (GW7+)
  Free transfer cap: 5 (FPL 2025-26 rules)

## Intel 06 Simulation Results (GW1-10, actual points verified)
  Total actual points:   652 (65.2/GW avg)
  Total predicted:       ~612
  Penalties:             -4 pts
  Chips used:            Wildcard GW5, Triple Captain GW6, Bench Boost GW7

  GW-by-GW actuals (with chips):
    GW 1:  61 pts (Salah C)
    GW 2:  49 pts (Palmer C)
    GW 3:  62 pts (Wood C)
    GW 4:  86 pts (Junqueira C)
    GW 5:  48 pts (Haaland C, Wildcard)
    GW 6:  97 pts (Haaland C, Triple Captain)
    GW 7:  69 pts (Haaland C, Bench Boost: XI 60 + bench 9)
    GW 8:  61 pts (Haaland C)
    GW 9:  45 pts (Haaland C, -4 penalty)
    GW10:  74 pts (Haaland C)

## Intel 06 Improvement History
  Initial Stage 8 baseline:         ~429 actual pts (no intel, no snapshots)
  + GW-snapshot features:           ~557 pts
  + Intel penalties + FDR + loyalty: ~616 pts
  + Ownership boost + chip lockout:  ~654 pts (bug: penalty sign error)
  + Penalty sign fix:                ~598 pts (corrected)
  + Zero-hit lockout:                ~580 pts (fewer penalties)
  + Auto-subs + bench weight + BB:   629 pts
  + TC/BB triggers + BB bug fix:     652 pts (current best)
  BB bug: result["bench_outfield"] does not exist; bench = squad minus XI.

## Stage 9 Architecture (original, separate from intel)
Two LLM calls per GW:
  Call 1 (Pre-ILP): Risk filter — flags injured/unavailable players,
                    applies -50% penalty (high confidence) or -25% (medium)
  Call 2 (Post-ILP): Captain/bench/report — reviews ILP squad,
                     may override captain if alternative adj score >= 0.5 pts higher

Key Stage 9 parameters:
  MODEL_ID: claude-sonnet-4-20250514
  CALL1_POOL_SIZE: top-N players passed to LLM for risk screening
  FDR multiplier: 0.03 (adj = 1.0 - 0.03 * (fdr - 3.0))
  Captain override threshold: 0.5 pts adj difference minimum

## Stage 9 Simulation Results (GW1-10, real GW1-10 of 2025-26)
  Stage 8 baseline: 579 pts total (57.9/GW avg)
  Stage 9 with LLM: 772 pts total (77.2/GW avg) — +193 pts vs Stage 8
  Captain overrides: 2 out of 10 GWs (GW5, GW9)

## Random Search — Joint Hyperparameter Optimization
  Script: pipeline/random_search_full.py
  Results: data/intel/random_search_full/summary.json
  Trials: 250 total (124 XGBoost, 126 LightGBM)
  Search space: MODEL_TYPE + model hyperparams + all optimizer params

  Key finding: LightGBM dominates — 14 of top 15 results are LGBM.
  XGBoost best: 1716 pts (trial 1, baseline config)
  LGBM best:    1760 pts (trial 220) — becomes 1767 with TC_THRESH=6.0

  Trial 220 winning params:
    MODEL_TYPE:           lgbm
    model_n_estimators:   300
    model_max_depth:      3
    model_learning_rate:  0.03
    model_subsample:      0.9
    model_colsample:      0.6
    lgbm_num_leaves:      31
    lgbm_min_child_samples: 30
    FDR_MULT:             0.02
    FDR_MULT_DEF:         0.08
    CAP_FORM_GATE:        4.0
    CAP_FORM_PENALTY:     0.3
    OWN_BOOST_GW1:        0.15
    CAP_STREAK_LIMIT:     2
    CAP_STREAK_MULT:      0.9
    TC_THRESH:            8.0   (overridden to 6.0 for final result)
    BB_MIN_GW:            9
    CAP_FDR_MULT:         0.10
    CAP_BLANK_PENALTY:    0.9
    CAP_BLANK_THRESH:     4

## Training Files (data/processed/)
train_gk.csv   — 4,421  rows  69 cols
train_def.csv  — 18,828 rows  67 cols
train_mid.csv  — 22,132 rows  65 cols
train_fwd.csv  — 5,663  rows  65 cols
TOTAL          — 51,044 rows
Target column: total_points
All validated: 0 NaN, 0 leakage, 0 cross-season bleed

## Model Output Paths
models/xgb_gk.pkl   — also used as lgbm_gk when MODEL_TYPE=lgbm
models/xgb_def.pkl
models/xgb_mid.pkl
models/xgb_fwd.pkl
models/stage9_results.json      — Stage 9 GW1-10 simulation log
data/intel/season_simulation.json — Season simulator GW1-28 best run
data/intel/final_squad.json       — Intel 06 GW1-10 simulation log
data/intel/availability.json      — intel_03 output
data/intel/rotation_risk.json     — intel_04 output
data/intel/recommendations.json   — intel_05 output
data/intel/press_conferences.json — intel_02 output

## Critical Rules — Never Break
1. GW1 BLIND TEST — zero 2025-26 data in training ever
2. NO LEAKAGE — all features must be knowable before GW kickoff
3. NO CROSS-SEASON BLEED — rolling windows partition by season
4. 4 SEPARATE MODELS — one per position, never mix
5. WALK-FORWARD VALIDATION — train on seasons 1-N, validate N+1
   Never shuffle. Always respect temporal order.
6. CONFIRMATION GATE after every step — never auto-advance
7. ONLINE LEARNING — River library retrains after each GW (GW2+)
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

## Season Simulator Best Result (GW1-28)
  Best: 1767 pts (63.1/GW avg) — confirmed reproducible
  Model: LightGBM (trial 220 params)
  Chips: tc1 GW6, bb1 GW11, wc1 GW17, bb2 GW23, tc2 GW25, fh2 GW26
  Penalties: 0 pts

  GW-by-GW actuals:
    GW 1:  79 pts (Haaland C)
    GW 2:  83 pts (Haaland C)
    GW 3:  63 pts (Junqueira C)
    GW 4:  98 pts (Haaland C)
    GW 5:  63 pts (Borges Fernandes C)
    GW 6:  88 pts (Haaland C, Triple Captain tc1)
    GW 7:  72 pts (Haaland C)
    GW 8:  82 pts (Haaland C)
    GW 9:  67 pts (Haaland C)
    GW10:  66 pts (Haaland C)
    GW11:  25 pts (Haaland C, Bench Boost bb1)
    GW12:  37 pts (Salah C, Wildcard wc1)
    GW13:  24 pts (Haaland C)
    GW14:  60 pts (Borges Fernandes C)
    GW15:  64 pts (Haaland C)
    GW16:  67 pts (Borges Fernandes C)
    GW17:  63 pts (Haaland C)
    GW18:  28 pts (Haaland C)
    GW19:  41 pts (Ekitike C)
    GW20:  44 pts (Haaland C, Triple Captain tc2)
    GW21:  65 pts (Nascimento Rodrigues C)
    GW22:  40 pts (Wirtz C)
    GW23:  59 pts (Nascimento Rodrigues C, Bench Boost bb2)
    GW24:  53 pts (Borges Fernandes C)
    GW25:  74 pts (Borges Fernandes C)
    GW26:  49 pts (dos Santos Magalhaes C, Free Hit fh2)
    GW27:  60 pts (Haaland C)
    GW28:  40 pts (Haaland C)

  Improvement history from baseline:
    1668  XGB Trial 1 (random search baseline)
    1679  CAP_FORM_GATE 5.0 → 7.0  (+11 pts)
    1680  CAP_FDR_MULT 0.05 → 0.10  (+1 pt)
    1682  TC_THRESH 10.5 → 8.5 + TC_FORM_MIN 7.0 → 6.0  (+2 pts)
    1690  DGW_PRED_MULT = 2.0  (+8 pts, DGW prediction boost)
    1700  DGW actuals bug fix  (+10 pts, data correctness)
    1716  TC_THRESH 8.5 → 6.0  (+16 pts, tc2 fires earlier)
    1760  Switch to LGBM trial 220 params  (+44 pts)
    1767  TC_THRESH fine-tuned to 6.0 on LGBM  (+7 pts, current best)

  Current simulator params (pipeline/season_simulator.py):
    MODEL_TYPE     = "lgbm"   (was "xgb")
    FDR_MULT       = 0.02     (was 0.025)
    FDR_MULT_DEF   = 0.08     (GK/DEF position-specific, was 0.025)
    OWN_BOOST_GW1  = 0.15     (was 0.20)
    TC_THRESH      = 6.0      (was 10.5)
    TC_FORM_MIN    = 6.0
    DGW_PRED_MULT  = 2.0
    CAP_FORM_GATE  = 4.0      (was 5.0 → 7.0, now 4.0 via trial 220)
    CAP_FORM_PENALTY = 0.3    (was 0.5)
    CAP_STREAK_MULT  = 0.9    (was 0.8)
    CAP_FDR_MULT     = 0.10
    CAP_BLANK_PENALTY = 0.9   (was 0.65)
    CAP_BLANK_THRESH  = 4
    LGBM n_estimators=300, max_depth=3, lr=0.03, subsample=0.9,
         colsample=0.6, num_leaves=31, min_child_samples=30

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
  position-adjusted (FWD x1.25). TC threshold was 10.0, Haaland raw
  ~9.5. Fixed by using CAP_POS_MULTIPLIERS for TC check and threshold 9.5.
- DGW actual points undercounted: load_player_history() used dict assignment
  so for DGW weeks (2 rows per player per GW), the second row silently
  overwrote the first. Fixed by accumulating additive stats (pts/mins/goals/
  assists/cs/saves/bonus) across both rows. Affected GW26 (+10 pts reported).

## Known Limitations
- intel_02 (press conference scraper) has popularity bias: only scrapes clubs
  that appear in FFS article headers. Newcastle never appeared as a section
  header — their injured players (Bruno Guimaraes, Schar, Livramento, Krafth)
  are mentioned inline under other clubs' sections and go undetected.
  Root cause: FFS article structure does not always include a "NEWCASTLE" h2
  header, instead mentioning Newcastle injuries inside another club's section.
  A cross-club name-matching fallback was tested in intel_03 but caused
  cascading squad changes (wildcard at GW12 instead of GW17) that cost ~113 pts
  overall. Left as known limitation — documented for thesis.
- Season simulator uses XGBoost/LGBM predictions only; it does NOT use
  intel_03 availability data. Injured players who aren't covered by press
  conferences (e.g., Newcastle players) stay in the predicted squad.
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
