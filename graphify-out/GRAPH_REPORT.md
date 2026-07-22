# Graph Report - .  (2026-07-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1244 nodes · 2166 edges · 78 communities (72 shown, 6 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 143 edges (avg confidence: 0.77)
- Token cost: 50,248 input · 1,204 output

## Graph Freshness
- Built from commit: `3b6152e2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Club Identity Registry
- Injury Claim Ledger
- MILP Chip Tests
- FFS Team News Adapter
- FPL Rules Accounting
- Effective Ownership Scraper
- Minutes Model Tests
- LLM Recommendations (Intel 05)
- Intel Optimizer Adjustments
- Press Conference Scraper
- Rotation Risk Scoring
- Defender Stats Patcher
- Debutant Stats Loader (4b)
- Feature Engineering (Stage 6)
- MILP Core Tests
- Season Simulator Setup
- Chip Percentile Gates
- Simulation Run Engine
- New Signings Scraper
- Stage 4a Verification
- Player Availability (Intel 03)
- ILP Optimizer (Stage 8)
- Intel 02 Tier Tests
- Season Report
- Injury Source Adapters
- Vaastav Data Loader (Stage 2)
- Debutant Identification
- Stage 4a Report Viewer
- Scraper Orchestrator
- MILP Core Solver
- Multi-Source Adapters
- XGBoost Training (Stage 7)
- GW28 Optimizer
- GW29 Optimizer
- League Multiplier Patch
- Signings Column Patch
- Scraper Validation
- Chip Decision Scheduler
- ILP Backtest Eval
- Backtest Metrics
- FPL Live Snapshot (Intel 01)
- Guardian News Adapter
- LLM Agent (Stage 9)
- Minutes Play-Probability Model
- FPL API Fetcher (Stage 1)
- Sky Sports Adapter
- Team Form (Stage 3)
- Reach Local Adapter
- Gemini LLM Extraction
- FBref Stats Extraction
- Optuna Search GW1-28
- Historical Model Training
- Flask UI Server
- Intel 06 HPO
- Random Search
- Stage 4b Verification
- Horizon Prediction Matrix
- Bench Intelligence (Intel 07)
- Optuna Full-Season Search
- Cross-Season Optuna (Phase 6)
- Joint Random Search
- Bench Random Search
- LGBM Random Search
- Set Piece Rebuilder
- FBref Table Parsing
- Bench Targeting Helpers
- Thesis PPTX Generator
- Trial Runner
- FBref Column Matching
- Chip Timing Checks
- ILP Squad Selection
- Squad Loyalty Bonus
- Season Input Builder
- Simulator Variant Runner
- Bench Report GW6-10
- Full Team Points Report
- FDR Def Sweep
- Form Blend Sweep

## God Nodes (most connected - your core abstractions)
1. `make_matrix()` - 36 edges
2. `run_simulation()` - 35 edges
3. `reconcile_player()` - 27 edges
4. `base_rows()` - 23 edges
5. `C()` - 20 edges
6. `solve()` - 17 edges
7. `solve()` - 17 edges
8. `solve()` - 17 edges
9. `main()` - 16 edges
10. `make_claim()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `test_sky_and_reach_claims_reconcile_together()` --calls--> `reconcile_player()`  [INFERRED]
  tests/test_intel_02_tier34.py → pipeline/intel_02_ledger.py
- `test_nearest_rank_is_deterministic()` --calls--> `nearest_rank()`  [INFERRED]
  tests/test_chip_percentile.py → pipeline/chip_percentile.py
- `test_anchored_week_bypasses_bar()` --calls--> `ChipPercentileLedger`  [INFERRED]
  tests/test_chip_percentile.py → pipeline/chip_percentile.py
- `test_plain_week_must_clear_prior_percentile()` --calls--> `ChipPercentileLedger`  [INFERRED]
  tests/test_chip_percentile.py → pipeline/chip_percentile.py
- `test_state_round_trips_atomically()` --calls--> `ChipPercentileLedger`  [INFERRED]
  tests/test_chip_percentile.py → pipeline/chip_percentile.py

## Import Cycles
- None detected.

## Communities (78 total, 6 thin omitted)

### Community 0 - "Club Identity Registry"
Cohesion: 0.05
Nodes (32): FplClub, club_key(), club_tokens(), ClubRegistry, _last_name(), load_reference(), normalize_name(), PlayerResolver (+24 more)

### Community 1 - "Injury Claim Ledger"
Cohesion: 0.09
Nodes (41): availability_tier(), ClaimLedger, _effective_ts(), _hours_between(), make_claim(), _parse_ts(), datetime, pipeline/intel_02_ledger.py Claim ledger + reconciler for the multi-source scrap (+33 more)

### Community 2 - "MILP Chip Tests"
Cohesion: 0.11
Nodes (42): _blank_owned_at(), dgw_over(), tests/test_milp_chips.py — Phase 4: chips as MILP variables.  Run:  docker run, Make players double at GW g (mu doubled, n_fix=2)., solve(), test_bb_fires_on_the_double_week(), test_bb_reservation_guard_holds_for_far_dgw(), test_blank_week_rescued_by_a_reset_chip() (+34 more)

### Community 3 - "FFS Team News Adapter"
Cohesion: 0.07
Nodes (32): Pattern, classify_news(), FfsTeamNewsAdapter, _parse_iso(), Stage A availability classification of one liveblog sentence., WP REST date_gmt/modified_gmt ('2026-01-16T13:00:00') -> tz-aware ISO., FFS "FPL Gameweek N team news" liveblog (multi-edition per GW: Thu/Fri,     plus, Return this GW's liveblog edition post dicts (full body included).         after (+24 more)

### Community 4 - "FPL Rules Accounting"
Cohesion: 0.07
Nodes (35): hit_points(), next_free_transfers(), pipeline/fpl_rules.py Pure FPL rule accounting — stdlib only, no pandas/model de, £m float -> integer tenths (5.3 -> 53)., FPL sell rule:       - price fell or unchanged: sell at current market price, Total sell value of a squad.     purchase_prices: {player_id: price_paid}     ma, Free transfers available at GW gw+1 under real 2025-26 rules:       - +1 accrues, Points deducted for transfers beyond the free allowance (-4 each). (+27 more)

### Community 5 - "Effective Ownership Scraper"
Cohesion: 0.07
Nodes (27): _as_float(), build_snapshot(), differentials(), _enrichment(), eo_of(), fetch_bootstrap(), fetch_eo(), _fetch_json() (+19 more)

### Community 6 - "Minutes Model Tests"
Cohesion: 0.11
Nodes (28): ph(), tests/test_minutes_model.py — learned play-probability model + matrix hook.  Run, {gw: {"minutes": m, "total_points": 2}} from a GW1.. sequence., Population of always-starters, rotation players and never-players., synth_history(), test_features_capture_recency(), test_learns_starter_vs_ghost(), test_matrix_pi_override_applies() (+20 more)

### Community 7 - "LLM Recommendations (Intel 05)"
Cohesion: 0.09
Nodes (32): build_context_for_gw(), build_summary(), build_team_maps(), call_llm(), compute_all_forms(), compute_season_stats(), fallback_decisions(), get_client() (+24 more)

### Community 8 - "Intel Optimizer Adjustments"
Cohesion: 0.07
Nodes (29): apply_captain_override(), apply_fdr_adjustment(), apply_gw1_ownership_boost(), apply_intel_penalties(), apply_ongoing_ownership_boost(), apply_playing_filter(), build_fdr_lookup(), build_history_lookup() (+21 more)

### Community 9 - "Press Conference Scraper"
Cohesion: 0.13
Nodes (27): BeautifulSoup, _build_result(), classify(), discover_url(), extract_injury_type(), extract_players_from_elements(), find_article_body(), _is_club_header() (+19 more)

### Community 10 - "Rotation Risk Scoring"
Cohesion: 0.12
Nodes (26): build_club_to_short(), build_summary(), compute_rotation_risk(), get_weights(), last_name(), main(), match_player(), normalize_name() (+18 more)

### Community 11 - "Defender Stats Patcher"
Cohesion: 0.12
Nodes (22): build_fbref_url(), extract_def_stats(), fetch_table(), find_player_col(), flatten_columns(), fuzzy_find(), get_stat(), normalize() (+14 more)

### Community 12 - "Debutant Stats Loader (4b)"
Cohesion: 0.16
Nodes (20): _build_fbref_url(), find_col(), _find_player_col(), flatten_columns(), gate(), _get_cache_path(), _infer_league_from_cache(), _load_state() (+12 more)

### Community 13 - "Feature Engineering (Stage 6)"
Cohesion: 0.18
Nodes (25): build_prev_lookup(), gate(), load_state(), main(), match_lookup_to_vaastav(), _norm(), DataFrame, Stage 6 — Feature Engineering Run: python pipeline/feature_engineering_stage6.p (+17 more)

### Community 14 - "MILP Core Tests"
Cohesion: 0.18
Nodes (24): base_rows(), mk(), tests/test_milp_core.py — Phase 2 MILP core tests on synthetic pools.  Run:  doc, 24-player pool; owned = a legal 2/5/5/3 squad of the x01..x05 ids., solve(), test_bench_ev_prefers_playing_bench(), test_bench_slots_legal_and_devalues_depth(), test_budget_and_sell_value_identity() (+16 more)

### Community 15 - "Season Simulator Setup"
Cohesion: 0.09
Nodes (23): apply_bonus_adjustment(), apply_intel_captain_override(), build_bonus_lookup(), build_gw1_pool(), build_gw1_team_form(), build_lookahead_pool(), build_opponent_lookup(), load_availability() (+15 more)

### Community 16 - "Chip Percentile Gates"
Cohesion: 0.13
Nodes (14): ChipPercentileLedger, nearest_rank(), Persistent percentile gates for unanchored in-model chips.  The ledger records, Return the q-th quantile by deterministic nearest-rank convention., JSON-backed, append-only weekly values for a single simulation season., Whether a chip may fire now; events bypass the seasonal bar., Append this GW's proxy values once for every still-relevant chip., _validate_q() (+6 more)

### Community 17 - "Simulation Run Engine"
Cohesion: 0.08
Nodes (24): apply_auto_subs(), apply_availability_penalties(), build_retrain_rows(), build_rolling_pool(), build_team_form_lookup(), compute_blank_gws(), compute_score(), load_pi_intel() (+16 more)

### Community 18 - "New Signings Scraper"
Cohesion: 0.14
Nodes (25): _build_fbref_url(), _fetch_tm_page(), find_col(), _find_pos_col(), _get_fbref_cache_path(), main(), _match_name_multi_strategy(), normalize_name() (+17 more)

### Community 19 - "Stage 4a Verification"
Cohesion: 0.19
Nodes (21): deduplicate_position_files(), fmt_f(), fmt_pct(), infer_confidence(), load_data(), main(), p(), Return (confidence_str, reason) from a player's extended rows. (+13 more)

### Community 20 - "Player Availability (Intel 03)"
Cohesion: 0.14
Nodes (21): availability_tier(), build_club_to_short(), compute_availability(), fpl_score(), last_name(), main(), match_player(), normalize_name() (+13 more)

### Community 21 - "ILP Optimizer (Stage 8)"
Cohesion: 0.18
Nodes (19): detect_bgw(), detect_dgw(), get_sim_start_gw(), load_best_params(), load_fpl_data(), load_models(), load_training_data(), main() (+11 more)

### Community 22 - "Intel 02 Tier Tests"
Cohesion: 0.12
Nodes (10): _attribute_club(), Headline first (most reliable); else the article's opening sentence.     Ambiguo, tests/test_intel_02_tier34.py — Tier 3/4 adapter tests (scraper redesign step 4,, StubLLM, test_attribute_club_ambiguous_two_clubs_skipped(), test_attribute_club_from_headline(), test_attribute_club_none_when_no_club(), test_llm_gap_detected_and_claims_tagged() (+2 more)

### Community 23 - "Season Report"
Cohesion: 0.19
Nodes (18): _best_xi(), _hindsight_ilp(), load_actuals_and_meta(), load_sim(), main(), print_chips(), print_gw_breakdown(), print_learning_curve() (+10 more)

### Community 24 - "Injury Source Adapters"
Cohesion: 0.13
Nodes (12): FfsInjuriesAdapter, _iso_from_date_only(), _iso_from_daymonth(), _iso_from_ddmmyyyy(), _iso_from_long_date(), _iso_from_yyyymmdd(), KnocksAndBansAdapter, datetime (+4 more)

### Community 25 - "Vaastav Data Loader (Stage 2)"
Cohesion: 0.18
Nodes (17): build_season_team_id_map(), build_team_name_mapping(), clean_data(), engineer_rolling_features(), load_season(), main(), print_validation_report(), Stage 2: Vaastav Historical Data Loader Loads, cleans and engineers features fr (+9 more)

### Community 26 - "Debutant Identification"
Cohesion: 0.17
Nodes (13): _current_age(), fuzzy_match_name(), _get_age_from_player_summaries(), _get_born_year_from_fbref(), _has_pre_vaastav_pl_history(), normalize_name(), Try to estimate birth year from player_summaries history_past.     Returns birt, Returns True if player_summaries shows FPL history before 2019/20.     This ind (+5 more)

### Community 27 - "Stage 4a Report Viewer"
Cohesion: 0.29
Nodes (16): bar(), conf_color(), hdr(), infer_conf(), load(), main(), print(), FPL AI Stage 4a - Full Report Printer Prints everything: all players, all seaso (+8 more)

### Community 28 - "Scraper Orchestrator"
Cohesion: 0.18
Nodes (16): build_compat_gw(), ffs_window(), load_health(), main(), merge_compat_output(), missing_clubs(), pipeline/intel_02_scrape.py Multi-source scraper orchestrator (redesign steps 2-, Reconciled records -> one GW block in press_conferences.json shape. (+8 more)

### Community 29 - "MILP Core Solver"
Cohesion: 0.18
Nodes (16): _bench_term(), _get_solver(), kappa(), prune_pool(), pipeline/milp_core.py Phases 2-4 of the optimizer redesign (docs/optimizer_rede, HiGHS when available (Phase 3+ scale needs it), else CBC., Captain coefficient: play-prob-weighted mean/ceiling blend., rows  : {pid: matrix row} for ONE gameweek — needs mu, pi, q90, price, (+8 more)

### Community 30 - "Multi-Source Adapters"
Cohesion: 0.17
Nodes (15): _extract_stories(), _fold(), _injury_in(), _player_name_tokens(), pipeline/intel_02_sources.py Source adapters for the multi-source scraper (redes, Accent-fold + lowercase but KEEP word boundaries (spaces)., Plain body text of a story dict — pre-extracted `body`, else `html`., Distinctive name forms to scan a story body for (>=4 alpha chars). (+7 more)

### Community 31 - "XGBoost Training (Stage 7)"
Cohesion: 0.22
Nodes (13): ndarray, build_split(), get_feature_cols(), make_objective(), DataFrame, Stage 7 — XGBoost Model Training with Optuna Hyperparameter Tuning ============, Return an Optuna objective function for a given position dataframe., Full train+tune pipeline for one position. Returns summary dict. (+5 more)

### Community 32 - "GW28 Optimizer"
Cohesion: 0.29
Nodes (13): apply_intel(), build_gw29_features(), build_training_data(), _ha(), hindsight_optimal(), load_data(), main(), print_squad_report() (+5 more)

### Community 33 - "GW29 Optimizer"
Cohesion: 0.29
Nodes (13): apply_intel(), build_gw29_features(), build_training_data(), _ha(), hindsight_optimal(), load_data(), main(), print_squad_report() (+5 more)

### Community 34 - "League Multiplier Patch"
Cohesion: 0.20
Nodes (13): lookup_override(), main(), norm_key(), normalize(), process_file(), DataFrame, patch_multipliers.py Targeted fix: update league_multiplier for players whose p, Returns (patched_df, change_log). (+5 more)

### Community 35 - "Signings Column Patch"
Cohesion: 0.27
Nodes (12): add_columns(), build_league_mapping(), cross_position_summary(), lookup_league(), main(), normalize_name(), overwrite_files(), DataFrame (+4 more)

### Community 36 - "Scraper Validation"
Cohesion: 0.21
Nodes (10): Severity is encoded in the inj-type span's class attribute., SportsGamblerAdapter, fetch_snapshot(), load_deadlines(), main(), scripts/validate_scraper_v2.py Off-season validation of the v2 multi-source scra, Return (snapshot_url_id, snapshot_ts_iso, distance_days) for the snapshot     cl, {gw: 'YYYY-MM-DD'} for all 38 GWs of 2025-26.     Priority: local cache -> FPL b (+2 more)

### Community 37 - "Chip Decision Scheduler"
Cohesion: 0.17
Nodes (13): _best_cap_stats(), _best_feasible_assignment(), decide_chip(), decide_chip_v2(), loyalty_bonus(), predict_pool(), Predict points for all players.     gw1_preds: stored after GW1 — blended into G, Return (best_adj, best_form3, best_home) for the best captain candidate.     loy (+5 more)

### Community 38 - "ILP Backtest Eval"
Cohesion: 0.21
Nodes (11): actual_gw_score(), build_actuals_index(), main(), Stage 8 Back-Test: Run ILP optimizer for GW1-10 of 2025-26 season, compare pred, Return dict: (player_id, gameweek) -> total_points, Compute actual FPL points for one GW.     Includes:       - Starting XI points, build_player_pool(), get_feature_cols() (+3 more)

### Community 39 - "Backtest Metrics"
Cohesion: 0.26
Nodes (11): compute_metrics(), index_log(), load_history(), load_log(), main(), print_report(), pipeline/backtest_metrics.py Phase 0 metric suite (docs/optimizer_redesign.md §1, player_history.csv -> {pid: {gw: total_points}} (DGW rows summed). (+3 more)

### Community 40 - "FPL Live Snapshot (Intel 01)"
Cohesion: 0.29
Nodes (11): build_snapshot(), fetch_historical_gws(), fetch_json(), main(), ownership_tier(), price_direction(), print_report(), Intel 01: FPL Live Data Snapshot FPL AI Thesis -- real-time player status, inju (+3 more)

### Community 41 - "Guardian News Adapter"
Cohesion: 0.23
Nodes (6): _finalize(), GuardianAdapter, _new_stats(), _new_story_stats(), Return API JSON with article bodies for the date window., raw: API JSON (search response) OR a single result item dict.

### Community 42 - "LLM Agent (Stage 9)"
Cohesion: 0.26
Nodes (11): build_prompt(), explain_gw(), load_fixture_context(), load_intel(), load_team_map(), main(), Stage 9: LLM Agent Layer FPL AI Thesis -- Claude API post-analysis of the finis, Returns nested dict: intel[gw][player_id] = {         avail_pct, avail_tier, ro (+3 more)

### Community 43 - "Minutes Play-Probability Model"
Cohesion: 0.21
Nodes (8): _label(), MinutesModel, pipeline/minutes_model.py — learned play-probability (π) for the mp matrix.  Rep, Train on all (pid, gw) rows with 2 <= gw < t.          intel = {gw: {pid: (avail, {pid: (p_play, p_start)} for gameweek t; {} when not ready., Features for target gameweek g from GWs strictly before g.      ph = {gw: {"minu, Online-retrained 3-class minutes classifier., _row_features()

### Community 44 - "FPL API Fetcher (Stage 1)"
Cohesion: 0.33
Nodes (10): build_fixture_difficulty(), fetch_bootstrap(), fetch_fixtures(), fetch_player_summaries(), get(), log_failed(), DataFrame, FPL AI — Stage 1: FPL API Data Fetcher Pulls all necessary data from the offici (+2 more)

### Community 45 - "Sky Sports Adapter"
Cohesion: 0.22
Nodes (7): _iso_from_rfc822(), Per-tick: the Google news sitemap (~50 URLs, all <48h old) filtered to     footb, T-24h escalation: fresh team-news items from a club's index page., RSS pubDate 'Tue, 7 Jul 2026 11:01:37 +0000' -> ISO UTC., SkySportsAdapter, test_sky_sitemap_filters_football_and_keywords(), test_sky_sitemap_parses_date_and_title()

### Community 46 - "Team Form (Stage 3)"
Cohesion: 0.20
Nodes (9): build_team_form_vaastav(), build_understat_xg(), fetch_understat_season(), merge_team_form(), FPL AI - Stage 3: Team Form Dataset Builds team-level form + xG features from v, Fetch EPL match data for one season from Understat's JSON API.     Returns list, Fetch Understat match xG data for all seasons, cache to JSON, and build     per, Merge vaastav form with Understat xG data on (team, season, opponent, was_home). (+1 more)

### Community 47 - "Reach Local Adapter"
Cohesion: 0.27
Nodes (6): Escalation-only (Tier 4): per-club Reach RSS → JSON-LD articleBody →     the sha, First application/ld+json node carrying an articleBody., ReachLocalAdapter, test_reach_jsonld_body_extraction(), test_reach_jsonld_graph_wrapped(), test_reach_rss_parse()

### Community 48 - "Gemini LLM Extraction"
Cohesion: 0.31
Nodes (3): GeminiExtractor, pipeline/intel_02_llm_extract.py Stage B LLM extraction for free-text press arti, extract_club() -> list of {player,status,injury,quote}; [] on failure.

### Community 49 - "FBref Stats Extraction"
Cohesion: 0.16
Nodes (15): _extract_fbref_stats(), safe_div(), _season_reliability(), season_to_year(), step4_extract_stats(), _extract_fbref_stats(), _map_fbref_position(), Given a row from the merged FBref DataFrame, extract all needed stats     using (+7 more)

### Community 50 - "Optuna Search GW1-28"
Cohesion: 0.25
Nodes (7): main(), print_report(), pipeline/optuna_search.py Optuna-based (TPE/Bayesian) hyperparameter search — a, Patch season_simulator with params and run it. Returns result dict., Define the Optuna search space and sample one trial's parameters., run_simulation(), suggest_params()

### Community 51 - "Historical Model Training"
Cohesion: 0.22
Nodes (9): build_hist_rows(), build_hist_team_form_lookup(), Train XGBoost or LightGBM per position depending on MODEL_TYPE., Convert a value that may be a Series (duplicate columns) to float., Extract all 6-season historical rows in FEAT_COLS format for retraining., Load team_form_vaastav.csv and build lookup for historical training rows.     Re, _safe_float(), train_gw1_models() (+1 more)

### Community 52 - "Flask UI Server"
Cohesion: 0.25
Nodes (4): get_intel(), Returns compact intel: { gw: { player_id: { avail_pct, rotation_risk } } }, run_sim(), _watch_process()

### Community 53 - "Intel 06 HPO"
Cohesion: 0.39
Nodes (7): clone_data(), make_objective(), Hyperparameter optimisation for intel_06_optimizer. Two demos:   Demo 1: HPO o, Deep-copy mutable parts of data; share read-only parts., run_hpo(), Run the intel-enhanced GW simulation.      Parameters     ----------     dat, run_simulation()

### Community 54 - "Random Search"
Cohesion: 0.36
Nodes (7): main(), make_param_list(), pipeline/random_search.py Random hyperparameter search over season simulator co, Pre-generate ALL param combinations before any simulator imports.     Uses a de, Run one trial with the given parameter set. Returns result dict., run_trial(), save_summary()

### Community 55 - "Stage 4b Verification"
Cohesion: 0.39
Nodes (5): best_fpl_match(), get_born(), in_stage4a(), in_transfers_csv(), normalize()

### Community 56 - "Horizon Prediction Matrix"
Cohesion: 0.25
Nodes (8): _make_X(), predict_gw0(), predict_horizon(), predict_horizon_gw(), Build feature DataFrame for one player, respecting model feature order., Predict points for current GW using base feature vectors., Predict points for real_gw + gw_offset with FDR/DGW/BGW adjustments.     real_g, Return discounted 3-GW horizon score per player.

### Community 57 - "Bench Intelligence (Intel 07)"
Cohesion: 0.32
Nodes (7): find_bb_target_gw(), get_bench_intel(), pipeline/intel_07_bench.py Intel 07: Bench Intelligence  Standalone data-driv, Given bench candidate scores for multiple GWs, find the best BB week.      For, Primary function: call each GW from the simulator.      Parameters:         p, Score all affordable players as bench candidates.      Returns dict: {, score_bench_candidates()

### Community 58 - "Optuna Full-Season Search"
Cohesion: 0.29
Nodes (5): main(), print_report(), pipeline/optuna_search_gw38.py Optuna TPE search over the FULL season (GW1-38)., Patch season_simulator with params and run GW1-38. Returns result dict., run_simulation()

### Community 59 - "Cross-Season Optuna (Phase 6)"
Cohesion: 0.33
Nodes (5): main(), pipeline/optuna_mp_search.py — Phase 6: cross-season Optuna sweep for the mp (mu, One full-season sim as a subprocess; returns the output JSON., run_season(), write_summary()

### Community 60 - "Joint Random Search"
Cohesion: 0.48
Nodes (6): main(), make_param_list(), print_report(), pipeline/random_search_full.py Joint random search over:   - MODEL_TYPE (xgb v, run_trial(), save_summary()

### Community 61 - "Bench Random Search"
Cohesion: 0.53
Nodes (5): main(), make_param_list(), pipeline/random_search_bench25.py Focused random search with BENCH_BONUS_NORMAL, run_trial(), save_summary()

### Community 62 - "LGBM Random Search"
Cohesion: 0.53
Nodes (5): main(), make_param_list(), pipeline/random_search_lgbm.py Random hyperparameter search using LightGBM inst, run_trial(), save_summary()

### Community 63 - "Set Piece Rebuilder"
Cohesion: 0.40
Nodes (5): find_player(), Rebuild all set piece taker columns in players_raw.csv from verified data. Clea, Match name_token against players in club_df.     - Strips accents from both sid, Normalize to ASCII-safe lowercase:     1. Apply explicit substitutions for non-, strip_accents()

### Community 64 - "FBref Table Parsing"
Cohesion: 0.33
Nodes (6): _fetch_fbref_table(), flatten_columns(), _parse_fbref_page_html(), Flatten a MultiIndex column DataFrame by joining levels with '_'.     Strips tr, Parse a FBref stats table from page HTML using pandas read_html.     FBref uses, Open a FBref URL and parse the specified table. Sets a 1920px viewport     to e

### Community 65 - "Bench Targeting Helpers"
Cohesion: 0.33
Nodes (6): _find_bb_target_gw(), get_bench_intel(), Score all affordable players as bench candidates.     Returns dict with by_posit, Find best BB week from a dict of {gw: bench_candidates_result}.     Signal: benc, Intel 07: bench intelligence and BB targeting. Called each GW.      Returns: {, _score_bench_candidates()

### Community 66 - "Thesis PPTX Generator"
Cohesion: 0.60
Nodes (4): add_slide(), main(), Generate FPL AI thesis presentation as .pptx Dark theme: bg #080c14, title gree, set_bg()

### Community 67 - "Trial Runner"
Cohesion: 0.50
Nodes (4): main(), pipeline/trial_runner.py Runs multiple season simulator configurations and comp, Run one trial by patching simulator constants and calling run_simulation., run_trial()

### Community 68 - "FBref Column Matching"
Cohesion: 0.33
Nodes (6): _find_player_col(), _find_squad_col(), _merge_fbref_tables(), Find the player name column. FBref uses 'Unnamed: N_level_0_Player' pattern., Find the squad/team column., Left-merge extra_df into base_df on Player+Squad columns.     Only keeps column

### Community 69 - "Chip Timing Checks"
Cohesion: 0.50
Nodes (4): check_chips(), Returns the chip to activate this GW, or None.     Priority: freehit > wildcard, check_chips_improved(), Wrapper around Stage 8's check_chips with smarter timing:     - Block ALL chips

### Community 70 - "ILP Squad Selection"
Cohesion: 0.50
Nodes (4): Run PuLP ILP to select optimal squad.      Returns dict with keys:       squa, Post-process captain/vice with positional bias multipliers.     Hard rules: nev, run_ilp(), select_captain()

### Community 71 - "Squad Loyalty Bonus"
Cohesion: 0.50
Nodes (4): apply_squad_loyalty(), get_loyalty_bonus(), High loyalty during chip lockout prevents hits and enables FT banking.     gw=0, Add a loyalty bonus to horizon scores for existing squad members.     Bonus is

## Knowledge Gaps
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_story_text()` connect `Multi-Source Adapters` to `Press Conference Scraper`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `_parse_fbref_page_html()` connect `FBref Table Parsing` to `Press Conference Scraper`, `New Signings Scraper`, `Debutant Stats Loader (4b)`, `FBref Column Matching`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `parse_fbref_html()` connect `Defender Stats Patcher` to `Press Conference Scraper`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `make_matrix()` (e.g. with `test_bb_fires_on_the_double_week()` and `test_bb_reservation_guard_holds_for_far_dgw()`) actually correct?**
  _`make_matrix()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `reconcile_player()` (e.g. with `test_thursday_then_friday_edition_last_write_wins()` and `test_adjacent_severities_not_conflict()`) actually correct?**
  _`reconcile_player()` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `base_rows()` (e.g. with `make_matrix()` and `test_gw1_initial_build()`) actually correct?**
  _`base_rows()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Club Identity Registry` be split into smaller, more focused modules?**
  _Cohesion score 0.05137844611528822 - nodes in this community are weakly interconnected._