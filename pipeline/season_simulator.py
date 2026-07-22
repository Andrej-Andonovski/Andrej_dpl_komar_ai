"""
pipeline/season_simulator.py
Fully standalone GW1-38 FPL season simulator.
Only pipeline import: fpl_rules (pure rule accounting, stdlib-only).
"""
import os, json, random, warnings, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import xgboost as xgb
import lightgbm as lgb
from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                  LpBinary, LpInteger, LpStatus, PULP_CBC_CMD)

# Phase 0 (docs/optimizer_redesign.md §9): pure FPL rule accounting
try:
    from pipeline.fpl_rules import sell_value, next_free_transfers
except ImportError:
    from fpl_rules import sell_value, next_free_transfers

# Phase 2/3: prediction matrix + MILP core (OPTIMIZER="mp" path)
try:
    from pipeline import prediction_matrix as pmx
    from pipeline.milp_core import (solve_gw as milp_solve_gw,
                                    solve_horizon as milp_solve_horizon,
                                    kappa as milp_kappa)
    from pipeline.minutes_model import MinutesModel
    from pipeline.chip_percentile import ChipPercentileLedger
except ImportError:
    import prediction_matrix as pmx
    from milp_core import (solve_gw as milp_solve_gw,
                           solve_horizon as milp_solve_horizon,
                           kappa as milp_kappa)
    from minutes_model import MinutesModel
    from chip_percentile import ChipPercentileLedger

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(_HERE, "..", "data")
HIST_CSV     = os.path.join(DATA_DIR, "raw", "fpl_api", "player_history.csv")
PLAYERS_CSV  = os.path.join(DATA_DIR, "raw", "fpl_api", "players_raw.csv")
FIXTURES_CSV = os.path.join(DATA_DIR, "raw", "fpl_api", "fixtures_raw.csv")
TRAIN_DIR    = os.path.join(DATA_DIR, "processed")
OUTPUT_JSON  = os.path.join(DATA_DIR, "intel", "season_simulation.json")
AVAIL_JSON   = os.path.join(DATA_DIR, "intel", "availability.json")
RECS_JSON    = os.path.join(DATA_DIR, "intel", "recommendations.json")

# ── Constants ─────────────────────────────────────────────────────────────────
BUDGET         = 100.0
MAX_CLUB       = 3
SIM_END_GW     = int(os.environ.get("SIM_END_GW", "38"))
CHIP_LOCKOUT   = 4
MAX_HITS       = 1
FDR_MULT        = 0.028451479772615692   # optuna trial 429 best (1799 pts, 0 pen)
FDR_MULT_DEF    = 0.08424155707006356   # optuna trial 429 best
OWN_BOOST_GW1  = 0.212871272538615     # optuna trial 429 best
PRED_CAP       = 20.0   # per-player prediction ceiling
XI_PRED_CAP    = 120.0  # XI predicted total ceiling — keeps predictions calibrated
TC_THRESH      = 6.1714966141844405   # optuna trial 429 best
TC_FORM_MIN    = 6.0    # tuned — enables tc2 at GW25 (+67 pts)
TC2_MIN_GW   = 20     # earliest GW tc2 is allowed to fire (set2 only)
FORCE_TC2_GW   = None   # if set, force tc2 to fire on this exact GW (overrides threshold)
FH2_EARLIEST_GW = 20    # earliest GW fh2 is allowed to fire (set2 only)
TC_FORM_FORCE  = 5.0    # minimum form to use TC even when force-used at GW18
BB_THRESH      = 9.0    # bench outfield total (3 outfield bench combined)
BB_FORCE_MIN   = 10.0   # bench pred floor for force-using BB at GW19
WC_THRESH      = 5      # squad members below pos avg triggers WC
WC17_LOYALTY   = 5.0    # loyalty bonus when force-using WC at GW17
CAP_STREAK_MAX      = 3     # consecutive captain GWs before rotation considered
CAP_FORM_MIN        = 6.0   # form_last3 below this + streak triggers rotation
CAP_FORM_GATE       = 6.565535461604104    # optuna trial 429 best
CAP_FORM_PENALTY    = 0.5736997458448272   # optuna trial 429 best
CAP_STREAK_LIMIT    = 2                    # optuna trial 429 best
CAP_STREAK_FORM     = 6.0                  # streak penalty only if form_last3 below this
CAP_STREAK_MULT     = 0.8988579789528998   # optuna trial 429 best
CAP_HOME_MARGIN     = 1.0                  # prefer home player if top candidate is within this many pts
CAP_MIN_RELIABILITY = 0.4                  # never captain player with minutes_reliability below this
CAP_FORM_GW_MIN     = 2                    # apply form gate from this GW onward
CAP_FDR_MULT        = 0.009167472856084574 # optuna trial 429 best
CAP_BLANK_THRESH    = 4                    # optuna trial 429 best
CAP_BLANK_PENALTY   = 0.7567954078245613   # optuna trial 429 best
BLEND_GWS           = 8.0   # GW1 blending fades to 0 by GW9
MC_SQUADS      = 3      # Monte Carlo random squads
DGW_PRED_MULT  = 2.0   # prediction boost for players with 2 fixtures in DGW weeks

# ── Intel 07 — Bench Intelligence ─────────────────────────────────────────────
MAX_BENCH_PRICE    = 5.5   # max price for bench candidates (£m)
MIN_MINUTES_REL    = 0.5   # must have played 50%+ of available minutes
MIN_FORM_LAST3     = 2.0   # minimum form to be considered
BENCH_BONUS_NORMAL = 2.7084338238469625   # optuna trial 429 best
BENCH_BONUS_BB_GW  = 2.2462419467730097   # optuna trial 429 best
LOOKAHEAD_GWS      = 3     # how many GWs ahead to evaluate for BB window
BB_MIN_GW          = 8   # optuna trial 429 best
BB_MAX_GW_SET1     = 19    # Set 1 BB must fire by GW19
BB_MAX_GW_SET2     = 38    # Set 2 BB must fire by GW38

# ── Chip Strategy v2 — rolling-horizon, calendar-agnostic ─────────────────────
# See docs/chip_strategy_redesign.md. No hardcoded GWs beyond the FPL set
# boundaries (Set 1 = GW1-19, Set 2 = GW20-38). "legacy" keeps the old
# GW17/18/19 force policy — retained for the thesis ablation.
CHIP_STRATEGY   = os.environ.get("CHIP_STRATEGY", "v2")   # "v2" | "legacy"
if CHIP_STRATEGY not in ("v2", "legacy"):
    raise ValueError(f"CHIP_STRATEGY must be 'v2' or 'legacy', got {CHIP_STRATEGY!r}")

# ── Rules mode — Phase 0 fair baseline (docs/optimizer_redesign.md §9) ────────
# "legacy"    — original accounting: sells at full market value, budget
#               relaxation on ILP infeasibility, FT bank capped at 2 pre-GW15.
#               Reproduces the 2468 production run unchanged.
# "corrected" — real FPL rules: 50% sell-on profit via a purchase-price
#               ledger, no budget relaxation (infeasible ⇒ raise), FT bank
#               1..5 from GW1. This is the FAIR BASELINE for the redesign.
RULES_MODE = os.environ.get("RULES_MODE", "legacy")
if RULES_MODE not in ("legacy", "corrected"):
    raise ValueError(f"RULES_MODE must be 'legacy' or 'corrected', got {RULES_MODE!r}")
if RULES_MODE != "legacy":
    # never overwrite the production season_simulation.json
    OUTPUT_JSON = OUTPUT_JSON.replace(".json", f"_{RULES_MODE}.json")

# One-off FT rule events {gw: ft_granted}. {15: 5} mirrors the real 2025-26
# mid-season grant that legacy hardcoded inside next_ft. Config, not code.
RULE_EVENTS_FT = {15: 5}

# ── Optimizer selection — Phase 2 (docs/optimizer_redesign.md §9) ─────────────
# "legacy" — original predict_pool adjustment chain + run_ilp + post-hoc
#            captain heuristics. Unchanged behaviour.
# "mp"     — prediction-matrix predictions (per-fixture DGW sums, blank
#            zeros) + milp_core (bench EV, captain+vice in-ILP). No loyalty
#            bonus, no bench bonus, no FDR post-multipliers, no caps, no
#            ownership boost, no GW1 blending, no captain streak/blank gates.
OPTIMIZER = os.environ.get("OPTIMIZER", "legacy")
if OPTIMIZER not in ("legacy", "mp"):
    raise ValueError(f"OPTIMIZER must be 'legacy' or 'mp', got {OPTIMIZER!r}")
if OPTIMIZER == "mp":
    if RULES_MODE != "corrected":
        raise ValueError("OPTIMIZER=mp requires RULES_MODE=corrected "
                         "(blueprint §5: correct rules throughout)")
    OUTPUT_JSON = OUTPUT_JSON.replace(".json", "_mp.json")

# Phase 3: planning horizon for the mp optimizer (1 = Phase 2 behaviour).
# H-sweep is a Phase 6 ablation; 5 is the blueprint default backed by the
# Phase 1 decay measurement.
MP_HORIZON = int(os.environ.get("MP_HORIZON", "5"))

# Captain mean-to-ceiling blend for the mp objective.  Kept as an environment
# override so ablations do not require source edits; its value is recorded in
# the simulation log below.  Default 0.3 = the 2026-07-15 sweep winner
# (sum 6824 vs 6681 at 0.5, 6766 at 0.1): captaincy pays closer to the
# reliable mean than the q90 ceiling.
MP_THETA = float(os.environ.get("MP_THETA", "0.3"))
if not 0.0 <= MP_THETA <= 1.0:
    raise ValueError(f"MP_THETA must be in [0, 1], got {MP_THETA!r}")
# Phase 6 sweep knobs: the remaining honest constants, env-exposed so the
# Optuna harness can search them without source edits.  Defaults = the
# blueprint values the whole scoreboard was measured at.
MP_DELTA      = float(os.environ.get("MP_DELTA", "0.94"))       # week discount
MP_DELTA_CHIP = float(os.environ.get("MP_DELTA_CHIP", "0.97"))  # chip discount
MP_GAMMA      = float(os.environ.get("MP_GAMMA", "0.07"))       # vice weight
MP_W_BENCH    = float(os.environ.get("MP_W_BENCH", "0.15"))     # bench slot prob

# Phase 4: chip decision source for the mp optimizer.
# "model"  — chips are variables inside the horizon MILP (blueprint §4.7)
# "legacy" — external decide_chip scheduler (Phase 3 behaviour, ablation)
MP_CHIPS = os.environ.get("MP_CHIPS", "model")
if MP_CHIPS not in ("model", "legacy"):
    raise ValueError(f"MP_CHIPS must be 'model' or 'legacy', got {MP_CHIPS!r}")
# WC squad-state gate: fire only when this many owned players sit below
# their position's replacement level (interpretable; Phase 6 sweep candidate)
MP_WC_BELOW = int(os.environ.get("MP_WC_BELOW", "4"))
# Hit + churn discipline — DEFAULTS = the shipped config (2×2 A/B
# 2026-07-15: discipline-only won the 3-season sum 6681 vs 6613 undisciplined,
# penalties 0/-12/-8 vs -64/-136/-60; see docs/HANDOFF.md scoreboard).
# MP_HIT_COST is the MILP's decision price per paid hit — scoring always
# subtracts the real -4.  8 demands a 2x margin over the paper gain (upgrade
# mus are selection-biased upward; undisciplined runs took 34/15 hits per
# season at negative realized ROI).
# Clamped > 0: at 0 the no-phantom-hit proof (milp_core §4.6) breaks.
MP_HIT_COST = max(0.01, float(os.environ.get("MP_HIT_COST", "8")))
MP_HIT_CAP = int(os.environ.get("MP_HIT_CAP", "2"))   # max paid hits per GW
# Cross-solve rebuy lock: a player sold at GW g cannot be rebought before
# GW g+GAP+1 (0 = off).  Kills the memoryless sell->rebuy thrash the rolling
# re-solve produces (40/30/28 buybacks per season measured in the milestone
# runs).  WC weeks are exempt — a wildcard is a legitimate judgment reset —
# and FH weeks never touch the ledger (shadow squad reverts).
MP_REBUY_GAP = int(os.environ.get("MP_REBUY_GAP", "4"))
# Season-level hit budget: hard cap on TOTAL paid hits per season (-1 =
# unlimited).  MP_HIT_COST shifts the decision margin; this BOUNDS the
# damage — MP_HIT_BUDGET=4 means at most -16 in penalties all season.
# Enforced at execution: each GW's per-week cap shrinks to what's left.
MP_HIT_BUDGET = int(os.environ.get("MP_HIT_BUDGET", "4"))
# Transfer friction: objective price per EXECUTED transfer, FT-funded
# included (WC/FH/GW1 rebuilds exempt; real scoring unaffected).  A free
# transfer is not free — banking has option value, and sub-noise mu edges
# otherwise trigger sideways churn (e.g. selling a 15-pt hauler for a
# +0.2/wk paper edge).  The honest version of legacy's tuned loyalty bonus.
# 0 = off until A/B-validated.
MP_FT_VALUE = float(os.environ.get("MP_FT_VALUE", "0"))
# Form hold: selling a player who just hauled >= MP_FORM_HOLD_MIN actual
# points costs MP_FORM_HOLD extra objective points (soft — a genuinely
# better move can still pay it; the solver usually sells someone else
# instead).  Past performance as a strategy signal: a haul is evidence the
# model's mu may be lagging the player's true level.  0 = off.
MP_FORM_HOLD = float(os.environ.get("MP_FORM_HOLD", "0"))
MP_FORM_HOLD_MIN = int(os.environ.get("MP_FORM_HOLD_MIN", "10"))
# Learned minutes model (pipeline/minutes_model.py): replaces the heuristic
# pi (appearances/5) with LightGBM P(play)/P(start) trained online on the
# season's own actuals — better pi feeds captaincy, bench value and
# availability everywhere. 0 = off until A/B-validated.
MP_PI_MODEL = os.environ.get("MP_PI_MODEL", "0") == "1"
# Feed intel_03 availability + intel_04 rotation-risk scores into the
# minutes model as features (2025-26 only — the scraped data's season).
# Requires MP_PI_MODEL=1.
MP_PI_INTEL = os.environ.get("MP_PI_INTEL", "0") == "1"
# Slot-ordered bench pricing: "" = off (flat MP_W_BENCH for all four bench
# bodies). Set e.g. "0.35,0.15,0.05" to price outfield bench slots by
# realistic auto-sub likelihood (first sub >> third) and the bench GK at
# MP_W_BENCH_GK — pushes budget out of sub-fodder into the XI (the mp
# system parks ~11 pred-pts/GW on its bench vs legacy's 3.3).
_slots_env = os.environ.get("MP_BENCH_SLOTS", "").strip()
MP_BENCH_SLOTS = (tuple(float(v) for v in _slots_env.split(","))
                  if _slots_env else None)
MP_W_BENCH_GK = float(os.environ.get("MP_W_BENCH_GK", "0.04"))

# ── Cross-season backtests (fixing stage, docs/phase4_report.md) ─────────────
# SIM_SEASON picks the season to simulate. Default "2025-26" = original
# inputs + intel. Other seasons: inputs from data/raw/seasons/<S>/
# (pipeline/build_season_inputs.py), NO intel data, no GW15 FT event,
# training restricted to seasons strictly before SIM_SEASON (no leakage),
# corrected rules mandatory. Chip/FT rules are applied uniformly (2025-26
# ruleset) — the cross-season test measures CALENDAR generalization, not
# historical rule replay.
SIM_SEASON = os.environ.get("SIM_SEASON", "2025-26")
if SIM_SEASON != "2025-26":
    if RULES_MODE != "corrected":
        raise ValueError("cross-season runs require RULES_MODE=corrected")
    _sdir = os.path.join(DATA_DIR, "raw", "seasons", SIM_SEASON)
    if not os.path.exists(_sdir):
        raise FileNotFoundError(f"{_sdir} — run build_season_inputs.py first")
    HIST_CSV     = os.path.join(_sdir, "player_history.csv")
    PLAYERS_CSV  = os.path.join(_sdir, "players_raw.csv")
    FIXTURES_CSV = os.path.join(_sdir, "fixtures_raw.csv")
    RULE_EVENTS_FT = {}              # the GW15 grant was a 2025-26 event
    OUTPUT_JSON = OUTPUT_JSON.replace(".json", f"_{SIM_SEASON}.json")

# Percentile chip bar: unanchored WC/TC/BB can only fire on a plain week
# when their proxy clears the q-th percentile of the season's earlier plain
# weeks (kind-level series).  DEFAULT OFF — the 2026-07-15 2×2 A/B showed
# that once hit/churn discipline is on, the bar nets −46 across the three
# calendars (−73/+38/−11).  MP_CHIP_BAR=1 re-enables; q + the switch are
# Phase 6 sweep candidates.  Proxies are recorded either way (state file is
# diagnostic).  Simulation runs start fresh by default; set RESUME=1 for a
# live weekly executor to load the prior state file.
MP_CHIP_BAR = os.environ.get("MP_CHIP_BAR", "0") == "1"
MP_CHIP_PERCENTILE_Q = float(os.environ.get("MP_CHIP_PERCENTILE_Q", "0.75"))
MP_CHIP_PERCENTILE_WARMUP = int(os.environ.get("MP_CHIP_PERCENTILE_WARMUP", "3"))
MP_CHIP_PERCENTILE_RESUME = os.environ.get("MP_CHIP_PERCENTILE_RESUME", "0") == "1"
MP_CHIP_PERCENTILE_STATE = os.environ.get(
    "MP_CHIP_PERCENTILE_STATE",
    os.path.join(DATA_DIR, "intel", f"chip_percentile_{SIM_SEASON}.json"),
)

# GW1 feature snapshot = the season before SIM_SEASON
_sy = int(SIM_SEASON[:4])
SNAPSHOT_SEASON = f"{_sy - 1}-{str(_sy)[2:]}"    # 2025-26 -> "2024-25"
CHIP_LOOKAHEAD  = 4      # GWs of forward planning (near candidate window)
SPACING_GAP     = 4      # min GWs between the two reset chips (WC <-> FH)
WC_HORIZON      = 5      # GWs over which a wildcard rebuild gain is summed
CHIP_BAR_BB     = 14.0   # bench must project >= this to fire BB (balanced)
CHIP_BAR_FH     = 16.0   # FH temp XI must beat current squad XI by >= this
CHIP_BAR_WC     = 20.0   # WC rebuild gain over WC_HORIZON must be >= this
# TC keeps the legacy natural trigger: adj >= TC_THRESH, form >= TC_FORM_MIN

CAP_MULT = {1: 0.0, 2: 0.75, 3: 1.15, 4: 1.25}   # by element_type

# Availability tier multipliers — applied to pred before ILP (from intel_03)
AVAIL_MULT = {
    "out":       0.0,    # zero prediction entirely
    "suspended": 0.0,    # zero prediction entirely
    "unlikely":  0.3,    # very doubtful
    "doubtful":  0.5,    # halve prediction
    "unknown":   0.85,   # mild penalty for uncertainty
    "probable":  0.95,   # near-certain, tiny penalty
    "available": 1.0,    # no penalty
}

# Loyalty bonus added to pred for current squad members
def loyalty_bonus(gw):
    if gw <= 5:  return 10.0
    if gw <= 10: return 2.0
    return 1.0

MODEL_TYPE = "lgbm"  # optuna trial 429 best (1799 pts, 0 pen)

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    random_state=42, verbosity=0
)

LGBM_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.04390698211469097,
    num_leaves=31, subsample=0.9321213778928387, colsample_bytree=0.8240721576385306,
    min_child_samples=27, random_state=42, verbosity=-1
)

FEAT_COLS = [
    "form_last3", "form_last5", "avg_points_per_game",
    "minutes_reliability", "goals_per_game", "assists_per_game",
    "clean_sheet_rate", "saves_per_game",
    "value", "was_home", "fdr",
]

# Intel 08 feature flags — kept for trial_runner compatibility; team/opp features
# are NOT in FEAT_COLS so these flags have no effect on predictions.
TEAM_FEATURES = True
OPP_FEATURES  = True

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Column name aliases in training CSVs -> our FEAT_COLS names
_COL_ALIAS = {
    "avg_points_per_game_season": "avg_points_per_game",
    "minutes_reliability_season": "minutes_reliability",
    "goals_per_game_season":      "goals_per_game",
    "assists_per_game_season":    "assists_per_game",
    "clean_sheet_rate_season":    "clean_sheet_rate",
    "saves_per_game_season":      "saves_per_game",
    "current_gw_fdr":             "fdr",
    "home_advantage":             "was_home",
}

TRAIN_FILES = {
    "GK":  "train_gk.csv",
    "DEF": "train_def.csv",
    "MID": "train_mid.csv",
    "FWD": "train_fwd.csv",
}


# ── Data Loaders ──────────────────────────────────────────────────────────────

def load_player_history():
    """Returns {player_id: {gw: {total_points, minutes, goals_scored,
                                  assists, clean_sheets, saves, value, was_home}}}
    DGW: players can have 2 rows per GW — points/minutes/stats are summed."""
    df = pd.read_csv(HIST_CSV)
    hist = defaultdict(dict)
    for r in df.itertuples(index=False):
        pid = int(r.player_id)
        gw  = int(r.gameweek)
        row = {
            "total_points":  float(r.total_points),
            "minutes":       int(r.minutes),
            "goals_scored":  int(r.goals_scored),
            "assists":       int(r.assists),
            "clean_sheets":  int(r.clean_sheets),
            "saves":         int(r.saves),
            "value":         float(r.value),
            "was_home":      int(r.was_home),
            "transfers_in":  int(getattr(r, "transfers_in",  0) or 0),
            "transfers_out": int(getattr(r, "transfers_out", 0) or 0),
            "bonus":         int(getattr(r, "bonus", 0) or 0),
        }
        if gw in hist[pid]:
            # DGW: accumulate additive stats; keep latest value for non-additive
            existing = hist[pid][gw]
            existing["total_points"] += row["total_points"]
            existing["minutes"]      += row["minutes"]
            existing["goals_scored"] += row["goals_scored"]
            existing["assists"]      += row["assists"]
            existing["clean_sheets"] += row["clean_sheets"]
            existing["saves"]        += row["saves"]
            existing["bonus"]        += row["bonus"]
            existing["value"]    = row["value"]
            existing["was_home"] = row["was_home"]
        else:
            hist[pid][gw] = row
    return hist


def load_players_raw():
    df = pd.read_csv(PLAYERS_CSV)
    if "price" not in df.columns:
        df["price"] = df["now_cost"] / 10.0
    if "position" not in df.columns and "element_type" in df.columns:
        df["position"] = df["element_type"].map(POS_MAP)
    if "web_name" not in df.columns:
        df["web_name"] = df.get("second_name", df.get("id").astype(str))
    return df


def load_fixtures():
    """Returns fdr_lookup {(team_id,gw)->fdr}, home_lookup {(team_id,gw)->1/0},
    dgw_gws set, gw_teams {gw: {team_id: count}}"""
    df = pd.read_csv(FIXTURES_CSV)
    df = df.dropna(subset=["gameweek"])
    df["gameweek"] = df["gameweek"].astype(int)
    fdr_lookup  = {}
    home_lookup = {}
    gw_teams    = defaultdict(lambda: defaultdict(int))
    for r in df.itertuples(index=False):
        gw = int(r.gameweek)
        th, ta = int(r.team_h), int(r.team_a)
        fdr_lookup[(th, gw)]  = int(r.team_h_difficulty)
        fdr_lookup[(ta, gw)]  = int(r.team_a_difficulty)
        home_lookup[(th, gw)] = 1
        home_lookup[(ta, gw)] = 0
        gw_teams[gw][th] += 1
        gw_teams[gw][ta] += 1
    dgw_gws = {gw for gw, teams in gw_teams.items()
               if any(c > 1 for c in teams.values())}
    return fdr_lookup, home_lookup, dgw_gws, gw_teams


def load_training_data():
    """Returns {pos: DataFrame} with columns renamed to FEAT_COLS equivalents."""
    train_dfs = {}
    for pos, fname in TRAIN_FILES.items():
        path = os.path.join(TRAIN_DIR, fname)
        if not os.path.exists(path):
            print(f"  [WARN] Missing training file: {path}")
            train_dfs[pos] = pd.DataFrame()
            continue
        df = pd.read_csv(path)
        df = df.rename(columns=_COL_ALIAS)
        df = df.loc[:, ~df.columns.duplicated()]   # drop duplicate cols
        # walk-forward hygiene across seasons: train ONLY on seasons strictly
        # before SIM_SEASON ("YYYY-YY" strings compare correctly). No-op for
        # 2025-26 (training files end at 2024-25).
        if "season" in df.columns:
            n0 = len(df)
            df = df[df["season"] < SIM_SEASON]
            if len(df) < n0:
                print(f"  [TRAIN-CUT] {fname}: {n0} -> {len(df)} rows "
                      f"(seasons < {SIM_SEASON})")
        train_dfs[pos] = df
    return train_dfs


# ── Name normalisation ────────────────────────────────────────────────────────

def _norm(s):
    import unicodedata
    try:
        s = unicodedata.normalize("NFD", str(s))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    except Exception:
        pass
    return "".join(c for c in s.lower() if c.isalpha())


# ── Intel-03 Availability ─────────────────────────────────────────────────────

def load_availability():
    """Load intel_03 availability.json → {gw_str: gw_data}."""
    if SIM_SEASON != "2025-26":
        print(f"  [AVAIL] intel data is 2025-26-only — disabled for {SIM_SEASON}")
        return {}
    if not os.path.exists(AVAIL_JSON):
        print("  [AVAIL] availability.json not found — skipping intel penalties")
        return {}
    with open(AVAIL_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("gameweeks", {})


def load_pi_intel():
    """intel_03 availability_pct + intel_04 rotation_risk as minutes-model
    features: {gw: {pid: (availability_pct, rotation_risk)}} (2025-26 only)."""
    if SIM_SEASON != "2025-26":
        return {}
    out = {}
    for path, field in ((AVAIL_JSON, "availability_pct"),
                        (os.path.join(DATA_DIR, "intel", "rotation_risk.json"),
                         "rotation_risk")):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for gw_str, gd in data.get("gameweeks", {}).items():
            gw_map = out.setdefault(int(gw_str), {})
            for pid_str, p in (gd.get("players") or {}).items():
                v = p.get(field)
                if v is None:
                    continue
                prev = gw_map.get(int(pid_str), (None, None))
                gw_map[int(pid_str)] = ((float(v), prev[1])
                                        if field == "availability_pct"
                                        else (prev[0], float(v)))
    # fill missing halves with the minutes-model neutral defaults
    try:
        from pipeline.minutes_model import (INTEL_AVAIL_DEFAULT,
                                            INTEL_ROT_DEFAULT)
    except ImportError:
        from minutes_model import INTEL_AVAIL_DEFAULT, INTEL_ROT_DEFAULT
    for gw_map in out.values():
        for pid, (a, r) in list(gw_map.items()):
            gw_map[pid] = (a if a is not None else INTEL_AVAIL_DEFAULT,
                           r if r is not None else INTEL_ROT_DEFAULT)
    return out


def load_recommendations():
    """Load intel_05 recommendations.json → {gw_str: gw_data}."""
    if SIM_SEASON != "2025-26":
        print(f"  [RECS] intel data is 2025-26-only — disabled for {SIM_SEASON}")
        return {}
    if not os.path.exists(RECS_JSON):
        print("  [RECS] recommendations.json not found — captain override disabled")
        return {}
    with open(RECS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    gws = data.get("gameweeks", {})
    print(f"  [RECS] recommendations.json: {len(gws)} GWs loaded")
    return gws


CAP_OVERRIDE_THRESH = 0.5  # minimum adj-score advantage for intel_05 override


def apply_intel_captain_override(captain_id, xi_pids, pool_by_pid, recs_gws, gw):
    """
    If intel_05 recommended a different captain AND that player is in the XI
    AND their adj score is >= CAP_OVERRIDE_THRESH pts higher, override.
    Returns (captain_id, source_str).
    """
    gw_recs = recs_gws.get(str(gw), {})
    intel_name = gw_recs.get("decisions", {}).get("captain", {}).get("name")
    if not intel_name:
        return captain_id, "ilp"

    # Find recommended player in XI by web_name
    new_cap_pid = None
    for pid in xi_pids:
        p = pool_by_pid.get(pid, {})
        if p.get("web_name") == intel_name:
            new_cap_pid = pid
            break

    if new_cap_pid is None or new_cap_pid == captain_id:
        return captain_id, "ilp"

    def adj(pid):
        p = pool_by_pid.get(pid, {})
        return p.get("pred", 0.0) * CAP_MULT.get(p.get("element_type", 3), 1.0)

    current_adj = adj(captain_id)
    new_adj     = adj(new_cap_pid)

    if new_adj >= current_adj + CAP_OVERRIDE_THRESH:
        old_name = pool_by_pid.get(captain_id, {}).get("web_name", str(captain_id))
        print(f"  [INTEL-CAP] GW{gw} override: {old_name} -> {intel_name} "
              f"(adj: {current_adj:.1f} -> {new_adj:.1f})")
        return new_cap_pid, "intel_05"

    return captain_id, "ilp"


def apply_availability_penalties(pool, avail_gws, gw):
    """
    Apply AVAIL_MULT to pool predictions using intel_03 tier data (player_id keyed).
    Returns (pool, n_penalized).
    """
    gw_str = str(gw)
    if gw_str not in avail_gws:
        return pool, 0

    gw_avail    = avail_gws[gw_str].get("players", {})
    n_penalized = 0
    out_names   = []
    dbt_names   = []

    for p in pool:
        pid_str = str(p["player_id"])
        if pid_str not in gw_avail:
            continue
        tier = gw_avail[pid_str].get("availability_tier", "available")
        mult = AVAIL_MULT.get(tier, 1.0)
        if mult < 1.0:
            p["pred"] = p["pred"] * mult
            n_penalized += 1
            if mult == 0.0:
                out_names.append(p["web_name"])
            elif mult <= 0.5:
                dbt_names.append(p["web_name"])

    if n_penalized > 0:
        def _fmt(lst): return ", ".join(lst[:10]) + (f" +{len(lst)-10}" if len(lst) > 10 else "")
        print(f"  [AVAIL] GW{gw}: {n_penalized} players penalized")
        if out_names:
            print(f"    OUT/SUSP: {_fmt(out_names)}")
        if dbt_names:
            print(f"    DOUBTFUL: {_fmt(dbt_names)}")

    return pool, n_penalized


# ── Intel 07 — Bench Intelligence ─────────────────────────────────────────────

def _score_bench_candidates(pool, gw):
    """
    Score all affordable players as bench candidates.
    Returns dict with by_position, recommended, all_candidates.
    Each candidate has fdr_adj and bench_score added.
    """
    # GW1: form data is 2024-25 season averages — artificially low, so use looser threshold
    form3_threshold = 0.5 if gw == 1 else MIN_FORM_LAST3

    candidates = []
    for p in pool:
        if p.get("price", 0) > MAX_BENCH_PRICE:
            continue
        if p.get("zero_minutes", False):
            continue
        if p.get("minutes_reliability", 0) < MIN_MINUTES_REL:
            continue
        if p.get("form_last3", 0) < form3_threshold:
            continue

        fdr     = p.get("fdr", 3.0)
        fdr_adj = max(0.5, 1.0 - 0.03 * (fdr - 3.0))
        form3   = p.get("form_last3", 0.0)
        min_rel = p.get("minutes_reliability", 0.0)

        bench_score = form3 * min_rel * fdr_adj

        candidates.append({
            **p,
            "fdr_adj":     fdr_adj,
            "bench_score": bench_score,
        })

    candidates.sort(key=lambda x: x["bench_score"], reverse=True)

    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for c in candidates:
        cpos = c.get("pos", "MID")
        if cpos in by_pos:
            by_pos[cpos].append(c)

    best_gk       = by_pos["GK"][0] if by_pos["GK"] else None
    outfield      = [c for c in candidates if c.get("pos") != "GK"]
    best_outfield = outfield[:3]

    return {
        "by_position":    by_pos,
        "recommended":    {"GK": best_gk, "outfield": best_outfield},
        "all_candidates": candidates[:20],
    }


def _find_bb_target_gw(pools_by_gw, bb_min_gw, bb_max_gw):
    """
    Find best BB week from a dict of {gw: bench_candidates_result}.
    Signal: bench candidates fixture quality score.
    """
    gw_scores = {}
    for g, result in pools_by_gw.items():
        if g < bb_min_gw or g > bb_max_gw:
            continue
        rec      = result.get("recommended", {})
        gk       = rec.get("GK")
        outfield = rec.get("outfield", [])
        bench_score = (gk["bench_score"] if gk else 0.0) + sum(p["bench_score"] for p in outfield[:3])
        gw_scores[g] = bench_score

    if not gw_scores:
        return {
            "target_gw":    bb_min_gw,
            "target_score": 0.0,
            "gw_scores":    {},
            "reasoning":    "No valid GWs found",
        }

    target_gw = max(gw_scores, key=gw_scores.get)
    return {
        "target_gw":    target_gw,
        "target_score": gw_scores[target_gw],
        "gw_scores":    gw_scores,
        "reasoning":    (
            f"GW{target_gw} has best bench score "
            f"{gw_scores[target_gw]:.2f}"
        ),
    }

def get_bench_intel(pool, gw, chips_used, bb1_used, bb2_used,
                    fdr_lookup=None, home_lookup=None):
    """
    Intel 07: bench intelligence and BB targeting. Called each GW.

    Returns: {
        "bench_candidates":  {player_id: bonus_pts},
        "bb_target_gw":      int or None,
        "bb_set":            1 or 2,
        "recommended_bench": [player dicts],
        "is_bb_target_gw":   bool
    }
    """
    use_set1 = (gw <= 19)
    bb_used  = bb1_used if use_set1 else bb2_used

    if bb_used:
        return {
            "bench_candidates":  {},
            "bb_target_gw":      None,
            "bb_set":            1 if use_set1 else 2,
            "recommended_bench": [],
            "is_bb_target_gw":   False,
        }

    current_result = _score_bench_candidates(pool, gw)

    bb_max = BB_MAX_GW_SET1 if use_set1 else BB_MAX_GW_SET2
    bb_min = max(gw, BB_MIN_GW)

    lookahead_pools = {}
    for future_gw in range(bb_min, min(bb_max + 1, gw + LOOKAHEAD_GWS + 1)):
        if fdr_lookup and home_lookup:
            future_pool = []
            for p in pool:
                team = p.get("team")
                fp = dict(p)
                fp["fdr"]      = float(fdr_lookup.get((team, future_gw), 3.0))
                fp["was_home"] = float(home_lookup.get((team, future_gw), 0))
                future_pool.append(fp)
        else:
            future_pool = pool
        lookahead_pools[future_gw] = _score_bench_candidates(future_pool, future_gw)

    bb_target = _find_bb_target_gw(lookahead_pools, bb_min, bb_max)
    target_gw = bb_target["target_gw"]
    is_target = (gw == target_gw)

    bonus = BENCH_BONUS_BB_GW if is_target else BENCH_BONUS_NORMAL

    bench_candidates  = {}
    recommended_bench = []

    rec = current_result["recommended"]
    if rec.get("GK"):
        pid = rec["GK"]["player_id"]
        bench_candidates[pid] = bonus
        recommended_bench.append(rec["GK"])

    for p in rec.get("outfield", []):
        pid = p["player_id"]
        bench_candidates[pid] = bonus
        recommended_bench.append(p)

    print(f"  [BENCH-INTEL] GW{gw} | BB target: GW{target_gw} "
          f"(score: {bb_target['target_score']:.2f}) | "
          f"is_target: {is_target} | bonus: {bonus:.1f}pts")
    print(f"  [BENCH-INTEL] Lookahead scores: " +
          ", ".join(f"GW{g}:{s:.2f}"
                    for g, s in sorted(bb_target["gw_scores"].items())))
    if recommended_bench:
        print(f"  [BENCH-INTEL] Recommended bench: " +
              ", ".join(
                  f"{p['web_name']}(£{p['price']:.1f}m,"
                  f"f3={p['form_last3']:.1f})"
                  for p in recommended_bench
              ))

    return {
        "bench_candidates":  bench_candidates,
        "bb_target_gw":      target_gw,
        "bb_set":            1 if use_set1 else 2,
        "recommended_bench": recommended_bench,
        "is_bb_target_gw":   is_target,
    }


# ── Intel 08: Team Form Lookups ───────────────────────────────────────────────

def build_hist_team_form_lookup():
    """
    Load team_form_vaastav.csv and build lookup for historical training rows.
    Returns {(team_name_lower, season, gw): {team_goals_last3, team_cs_rate_last3}}
    Values are already per-game averages in the vaastav file.
    """
    path = os.path.join(DATA_DIR, "raw", "vaastav", "team_form_vaastav.csv")
    if not os.path.exists(path):
        print("  [WARN] team_form_vaastav.csv not found — team form zeros in hist rows")
        return {}

    df = pd.read_csv(path)
    lookup = {}

    goals_col  = "goals_scored_last3" if "goals_scored_last3" in df.columns else "goals_scored_last5"
    cs_col     = "clean_sheet_rate_last3" if "clean_sheet_rate_last3" in df.columns else "clean_sheet_rate_last5"
    team_col   = "team"   if "team"    in df.columns else "team_name"
    season_col = "season" if "season"  in df.columns else None
    gw_col     = "GW"     if "GW"      in df.columns else "gameweek"

    for _, row in df.iterrows():
        team   = str(row.get(team_col, "")).lower().strip()
        season = str(row.get(season_col, "")) if season_col else "unknown"
        gw     = int(_safe_float(row.get(gw_col, 0)))
        goals  = float(row.get(goals_col, 1.5) or 1.5)
        cs     = float(row.get(cs_col,    0.3) or 0.3)

        # Safety: if value looks like a raw sum rather than per-game avg, normalise
        if goals_col.endswith("last3") and goals > 5:
            goals = goals / 3.0
        elif goals_col.endswith("last5") and goals > 8:
            goals = goals / 5.0

        lookup[(team, season, gw)] = {
            "team_goals_last3":  goals,
            "team_cs_rate_last3": cs,
        }

    print(f"  [HIST TEAM FORM] Loaded {len(lookup)} team-GW entries from vaastav")
    return lookup


def build_team_form_lookup(hist_lookup, players_df, completed_gw):
    """
    Build {(team_id, gw): {team_goals_last3, team_cs_rate_last3}}
    from 2025-26 actuals up to completed_gw.
    team_goals_last3  = mean goals scored by team per GW over last 3 GWs
    team_cs_rate_last3 = fraction of last 3 GWs where team kept a clean sheet
    """
    team_players = defaultdict(list)
    for r in players_df.itertuples(index=False):
        team_players[int(r.team)].append(int(r.id))

    lookup = {}

    for team_id, player_ids in team_players.items():
        team_gw_stats = {}

        for g in range(1, completed_gw + 1):
            team_goals = 0
            had_cs     = False
            any_data   = False

            for pid in player_ids:
                if pid in hist_lookup and g in hist_lookup[pid]:
                    s = hist_lookup[pid][g]
                    if s["minutes"] > 0:
                        team_goals += s["goals_scored"]
                        if s["clean_sheets"] > 0:
                            had_cs = True
                        any_data = True

            if any_data:
                team_gw_stats[g] = {"goals": team_goals, "had_cs": had_cs}

        sorted_gws = sorted(team_gw_stats.keys())

        for target_gw in range(2, completed_gw + 2):
            prior = [g for g in sorted_gws if g < target_gw][-3:]

            if not prior:
                lookup[(team_id, target_gw)] = {
                    "team_goals_last3":  0.0,
                    "team_cs_rate_last3": 0.0,
                }
                continue

            lookup[(team_id, target_gw)] = {
                "team_goals_last3":  float(np.mean([team_gw_stats[g]["goals"]          for g in prior])),
                "team_cs_rate_last3": float(np.mean([1.0 if team_gw_stats[g]["had_cs"] else 0.0
                                                     for g in prior])),
            }

    return lookup



def build_bonus_lookup(hist_lookup, completed_gw):
    """
    Compute per-player bonus consistency from GW1..completed_gw actuals.
    bonus_rate_last5: fraction of last 5 played GWs where player earned bonus > 0
    bonus_avg_last5:  average bonus pts per game over last 5 played GWs
    Returns {player_id: {bonus_rate_last5, bonus_avg_last5}}
    """
    lookup = {}
    for pid, gw_data in hist_lookup.items():
        played_gws = [
            (g, d) for g, d in gw_data.items()
            if g <= completed_gw and d.get("minutes", 0) > 0
        ]
        played_gws.sort(key=lambda x: x[0])
        last5 = played_gws[-5:]
        if not last5:
            lookup[pid] = {"bonus_rate_last5": 0.0, "bonus_avg_last5": 0.0}
            continue
        bonus_vals = [d.get("bonus", 0) for _, d in last5]
        bonus_rate = sum(1 for b in bonus_vals if b > 0) / len(bonus_vals)
        bonus_avg  = sum(bonus_vals) / len(bonus_vals)
        lookup[pid] = {
            "bonus_rate_last5": round(bonus_rate, 3),
            "bonus_avg_last5":  round(bonus_avg, 3),
        }
    return lookup

def build_opponent_lookup(fixtures_csv_path, team_form_lookup):
    """
    Build {(team_id, gw): {opp_goals_last3, opp_cs_rate_last3}}
    by joining each team's opponent's form for that GW.
    """
    df = pd.read_csv(fixtures_csv_path)
    df = df.dropna(subset=["gameweek"])
    df["gameweek"] = df["gameweek"].astype(int)

    opp_lookup = {}
    for _, row in df.iterrows():
        gw = int(row["gameweek"])
        th = int(row["team_h"])
        ta = int(row["team_a"])

        away_form = team_form_lookup.get((ta, gw), {})
        home_form = team_form_lookup.get((th, gw), {})

        opp_lookup[(th, gw)] = {
            "opp_goals_last3":  away_form.get("team_goals_last3",  0.0),
            "opp_cs_rate_last3": away_form.get("team_cs_rate_last3", 0.0),
        }
        opp_lookup[(ta, gw)] = {
            "opp_goals_last3":  home_form.get("team_goals_last3",  0.0),
            "opp_cs_rate_last3": home_form.get("team_cs_rate_last3", 0.0),
        }

    return opp_lookup


def build_gw1_team_form(players_df):
    """
    For GW1, use prior-season averages from team_form.csv as team form prior.
    Returns {team_id: {team_goals_last3, team_cs_rate_last3}}
    """
    if SIM_SEASON != "2025-26":
        # teams_raw.csv maps names to 2025-26 team ids — wrong for other
        # seasons; fall back to the league-average prior
        print(f"  [GW1 TEAM FORM] disabled for {SIM_SEASON} (id mapping is "
              "2025-26-specific) — league-average prior")
        return {}
    team_form_path = os.path.join(DATA_DIR, "processed", "team_form.csv")
    if not os.path.exists(team_form_path):
        print("  [WARN] team_form.csv not found — using league-average GW1 team form")
        return {}

    df = pd.read_csv(team_form_path)
    if "season" in df.columns:
        df = df[df["season"] == SNAPSHOT_SEASON]
    if df.empty:
        return {}

    # Build team name -> FPL team_id mapping via teams_raw.csv
    teams_raw_path = os.path.join(DATA_DIR, "raw", "fpl_api", "teams_raw.csv")
    team_name_to_id = {}
    if os.path.exists(teams_raw_path):
        tdf = pd.read_csv(teams_raw_path)
        for r in tdf.itertuples(index=False):
            name = getattr(r, "name", None)
            tid  = getattr(r, "id",   None)
            if name is not None and tid is not None:
                team_name_to_id[str(name).lower().strip()] = int(tid)

    goals_col = "goals_scored_last5" if "goals_scored_last5" in df.columns else None
    cs_col    = "clean_sheet_rate_last5" if "clean_sheet_rate_last5" in df.columns else None
    if not goals_col or not cs_col:
        return {}

    result = {}
    team_col = "team" if "team" in df.columns else "team_name"
    for team_name, grp in df.groupby(team_col):
        goals   = float(grp[goals_col].mean())
        cs_rate = float(grp[cs_col].mean())
        team_id = team_name_to_id.get(str(team_name).lower().strip())
        if team_id:
            result[team_id] = {
                "team_goals_last3":  goals,   # already per-game avg
                "team_cs_rate_last3": cs_rate,
            }

    print(f"  [GW1 TEAM FORM] Loaded 2024-25 averages for {len(result)} teams")
    return result


# ── GW1 Player Pool ───────────────────────────────────────────────────────────

def build_gw1_pool(players_df, train_dfs, fdr_lookup, home_lookup):
    """Build feature vectors for GW1 from 2024-25 training data averages."""
    # Position averages from 2024-25 (fallback)
    pos_avgs = {}
    for pos, df in train_dfs.items():
        df25 = df[df["season"] == SNAPSHOT_SEASON] if "season" in df.columns else df
        if df25.empty:
            df25 = df
        avail = [c for c in FEAT_COLS if c in df25.columns]
        pos_avgs[pos] = df25[avail].mean().to_dict() if avail else {}

    # Name index from 2024-25: norm_name -> feature dict
    name_idx = {}
    for pos, df in train_dfs.items():
        df25 = df[df["season"] == SNAPSHOT_SEASON] if "season" in df.columns else df
        if df25.empty or "name" not in df25.columns:
            continue
        feat_avail = [c for c in FEAT_COLS if c in df25.columns]
        for nm, grp in df25.groupby("name"):
            name_idx[_norm(nm)] = grp[feat_avail].mean().to_dict()

    pool = []
    stats = {"exact": 0, "partial": 0, "pos_avg": 0}
    for r in players_df.itertuples(index=False):
        pid   = int(r.id)
        pos   = str(r.position)
        etype = int(r.element_type)
        team  = int(r.team)
        web   = str(r.web_name)
        price = float(r.price)
        sbp   = float(r.selected_by_percent) if hasattr(r, "selected_by_percent") else 0.0
        fn    = str(getattr(r, "first_name", ""))
        sn    = str(getattr(r, "second_name", ""))

        feats = None
        for candidate in [web, sn, fn + " " + sn, fn]:
            nc = _norm(candidate)
            if nc and nc in name_idx:
                feats = dict(name_idx[nc])
                stats["exact"] += 1
                break

        if feats is None:
            nw = _norm(web)
            for tn, tf in name_idx.items():
                if nw and len(nw) >= 4 and (nw in tn or tn in nw):
                    feats = dict(tf)
                    stats["partial"] += 1
                    break

        if feats is None:
            feats = dict(pos_avgs.get(pos, {}))
            stats["pos_avg"] += 1

        for f in FEAT_COLS:
            feats.setdefault(f, 0.0)

        feats["value"]    = price
        feats["was_home"] = float(home_lookup.get((team, 1), 0))
        feats["fdr"]      = float(fdr_lookup.get((team, 1), 3.0))

        pool.append({
            "player_id": pid, "web_name": web, "pos": pos,
            "element_type": etype, "team": team,
            "price": price, "sbp": sbp, "zero_minutes": False,
            **{f: feats.get(f, 0.0) for f in FEAT_COLS}
        })

    print(f"  [GW1 pool] {len(pool)} players | "
          f"exact={stats['exact']} partial={stats['partial']} pos_avg={stats['pos_avg']}")

    # Attach team form features (GW1: use 2024-25 historical averages)
    gw1_team_form = build_gw1_team_form(players_df) if TEAM_FEATURES else {}
    for p in pool:
        team_id = p["team"]
        tf = gw1_team_form.get(team_id, {})
        p["team_goals_last3"]  = tf.get("team_goals_last3",  1.5)
        p["team_cs_rate_last3"] = tf.get("team_cs_rate_last3", 0.3)
        p["opp_goals_last3"]   = 1.5   # GW1: no opponent data yet, league average
        p["opp_cs_rate_last3"] = 0.3

    return pool


# ── GW2+ Pool (rolling features from actuals) ─────────────────────────────────

def build_rolling_pool(players_df, hist_lookup, fdr_lookup, home_lookup,
                       completed_gw, team_form_lookup=None, opp_lookup=None):
    """Build feature vectors for predicting completed_gw+1 from actuals 1..completed_gw."""
    next_gw = completed_gw + 1
    pool = []
    for r in players_df.itertuples(index=False):
        pid   = int(r.id)
        pos   = str(r.position)
        etype = int(r.element_type)
        team  = int(r.team)
        web   = str(r.web_name)
        price_static = float(r.price)
        sbp   = float(r.selected_by_percent) if hasattr(r, "selected_by_percent") else 0.0

        ph = hist_lookup.get(pid, {})

        # Latest price
        latest_price = price_static
        for g in range(completed_gw, 0, -1):
            if g in ph:
                latest_price = ph[g]["value"]
                break

        pts_ser, min_ser, g_ser, a_ser, cs_ser, sv_ser = [], [], [], [], [], []
        total_mins = 0
        for g in range(1, completed_gw + 1):
            if g not in ph:
                continue
            h = ph[g]
            pts_ser.append(h["total_points"])
            min_ser.append(h["minutes"])
            g_ser.append(h["goals_scored"])
            a_ser.append(h["assists"])
            cs_ser.append(h["clean_sheets"])
            sv_ser.append(h["saves"])
            total_mins += h["minutes"]

        n = len(pts_ser)
        zero_mins = (total_mins == 0 and n > 0)

        # Per-game stats: use played GWs only (avoids bench-GW dilution)
        played = [(pts_ser[i], g_ser[i], a_ser[i], cs_ser[i], sv_ser[i])
                  for i in range(len(pts_ser)) if min_ser[i] > 0]
        pl_pts = [x[0] for x in played]
        pl_g   = [x[1] for x in played]
        pl_a   = [x[2] for x in played]
        pl_cs  = [x[3] for x in played]
        pl_sv  = [x[4] for x in played]

        tf  = (team_form_lookup or {}).get((team, next_gw), {}) if TEAM_FEATURES else {}
        opp = (opp_lookup       or {}).get((team, next_gw), {}) if OPP_FEATURES  else {}

        pool.append({
            "player_id": pid, "web_name": web, "pos": pos,
            "element_type": etype, "team": team,
            "price": latest_price, "sbp": sbp, "zero_minutes": zero_mins,
            "form_last3":          float(np.mean(pts_ser[-3:])) if pts_ser else 0.0,
            "form_last5":          float(np.mean(pts_ser[-5:])) if pts_ser else 0.0,
            "avg_points_per_game": float(np.mean(pl_pts)) if pl_pts else 0.0,
            "minutes_reliability": total_mins / (completed_gw * 90.0),
            "goals_per_game":      float(np.mean(pl_g))  if pl_g  else 0.0,
            "assists_per_game":    float(np.mean(pl_a))  if pl_a  else 0.0,
            "clean_sheet_rate":    float(np.mean(pl_cs)) if pl_cs else 0.0,
            "saves_per_game":      float(np.mean(pl_sv)) if pl_sv else 0.0,
            "value":               latest_price,
            "was_home":            float(home_lookup.get((team, next_gw), 0)),
            "fdr":                 float(fdr_lookup.get((team, next_gw), 3.0)),
            "team_goals_last3":    tf.get("team_goals_last3",   1.5),
            "team_cs_rate_last3":  tf.get("team_cs_rate_last3", 0.3),
            "opp_goals_last3":     opp.get("opp_goals_last3",   1.5),
            "opp_cs_rate_last3":   opp.get("opp_cs_rate_last3", 0.3),
        })
    return pool


# ── Build retrain rows from accumulated 2025-26 actuals ───────────────────────

def build_retrain_rows(players_df, hist_lookup, fdr_lookup, home_lookup,
                       up_to_gw, team_form_lookup=None, opp_lookup=None):
    """Rows for GW 2..up_to_gw: features from prior GWs, target = actual points."""
    rows_by_pos = defaultdict(list)
    for r in players_df.itertuples(index=False):
        pid  = int(r.id)
        pos  = str(r.position)
        team = int(r.team)
        price_static = float(r.price)
        ph   = hist_lookup.get(pid, {})
        if not ph:
            continue

        for target_gw in range(2, up_to_gw + 1):
            if target_gw not in ph:
                continue
            target_pts = ph[target_gw]["total_points"]
            completed  = target_gw - 1

            pts_ser, min_ser, g_ser, a_ser, cs_ser, sv_ser = [], [], [], [], [], []
            total_mins = 0
            for g in range(1, completed + 1):
                if g not in ph:
                    continue
                h = ph[g]
                pts_ser.append(h["total_points"])
                min_ser.append(h["minutes"])
                g_ser.append(h["goals_scored"])
                a_ser.append(h["assists"])
                cs_ser.append(h["clean_sheets"])
                sv_ser.append(h["saves"])
                total_mins += h["minutes"]

            latest_price = price_static
            for g in range(completed, 0, -1):
                if g in ph:
                    latest_price = ph[g]["value"]
                    break

            # Per-game stats: played GWs only (avoids bench-GW dilution)
            played = [(pts_ser[i], g_ser[i], a_ser[i], cs_ser[i], sv_ser[i])
                      for i in range(len(pts_ser)) if min_ser[i] > 0]
            pl_pts = [x[0] for x in played]
            pl_g   = [x[1] for x in played]
            pl_a   = [x[2] for x in played]
            pl_cs  = [x[3] for x in played]
            pl_sv  = [x[4] for x in played]

            tf  = (team_form_lookup or {}).get((team, target_gw), {}) if TEAM_FEATURES else {}
            opp = (opp_lookup       or {}).get((team, target_gw), {}) if OPP_FEATURES  else {}

            rows_by_pos[pos].append({
                "form_last3":          float(np.mean(pts_ser[-3:])) if pts_ser else 0.0,
                "form_last5":          float(np.mean(pts_ser[-5:])) if pts_ser else 0.0,
                "avg_points_per_game": float(np.mean(pl_pts)) if pl_pts else 0.0,
                "minutes_reliability": total_mins / (completed * 90.0) if completed else 0.0,
                "goals_per_game":      float(np.mean(pl_g))  if pl_g  else 0.0,
                "assists_per_game":    float(np.mean(pl_a))  if pl_a  else 0.0,
                "clean_sheet_rate":    float(np.mean(pl_cs)) if pl_cs else 0.0,
                "saves_per_game":      float(np.mean(pl_sv)) if pl_sv else 0.0,
                "value":               latest_price,
                "was_home":            float(home_lookup.get((team, target_gw), 0)),
                "fdr":                 float(fdr_lookup.get((team, target_gw), 3.0)),
                "team_goals_last3":    tf.get("team_goals_last3",   1.5),
                "team_cs_rate_last3":  tf.get("team_cs_rate_last3", 0.3),
                "opp_goals_last3":     opp.get("opp_goals_last3",   1.5),
                "opp_cs_rate_last3":   opp.get("opp_cs_rate_last3", 0.3),
                "total_points":        target_pts,
            })
    return rows_by_pos


# ── Model Training ────────────────────────────────────────────────────────────

def train_models(rows_by_pos, weights_by_pos=None):
    """Train XGBoost or LightGBM per position depending on MODEL_TYPE."""
    models = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        rows = rows_by_pos.get(pos, [])
        if len(rows) < 5:
            models[pos] = None
            continue
        df = pd.DataFrame(rows)
        X  = df[FEAT_COLS].fillna(0.0).values
        y  = df["total_points"].values
        w  = weights_by_pos.get(pos) if weights_by_pos else None
        if MODEL_TYPE == "lgbm":
            m = lgb.LGBMRegressor(**LGBM_PARAMS)
        else:
            m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X, y, sample_weight=w)
        models[pos] = m
        print(f"    {pos}: {len(rows)} rows")
    return models


def _safe_float(val):
    """Convert a value that may be a Series (duplicate columns) to float."""
    if hasattr(val, "iloc"):
        val = val.iloc[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def build_hist_rows(train_dfs, hist_team_form=None):
    """Extract all 6-season historical rows in FEAT_COLS format for retraining."""
    rows_by_pos = defaultdict(list)
    for pos, df in train_dfs.items():
        if df.empty:
            continue
        df = df.loc[:, ~df.columns.duplicated()]
        avail = [c for c in FEAT_COLS if c in df.columns]
        if "total_points" not in df.columns or not avail:
            continue
        for _, row in df.iterrows():
            d = {c: _safe_float(row[c]) if c in row.index else 0.0 for c in FEAT_COLS}
            # Attach historical team + opponent form (keyed by team_name_lower, season, gw)
            if hist_team_form and TEAM_FEATURES:
                team   = str(row.get("team",          "")).lower().strip()
                opp    = str(row.get("opponent_team", "")).lower().strip()
                season = str(row.get("season", ""))
                gw     = int(_safe_float(row.get("GW", 0)))
                tf     = hist_team_form.get((team, season, gw), {})
                opp_tf = hist_team_form.get((opp,  season, gw), {})
                d["team_goals_last3"]   = tf.get("team_goals_last3",   1.5)
                d["team_cs_rate_last3"] = tf.get("team_cs_rate_last3", 0.3)
                d["opp_goals_last3"]    = opp_tf.get("team_goals_last3",   1.5)
                d["opp_cs_rate_last3"]  = opp_tf.get("team_cs_rate_last3", 0.3)
            else:
                d["team_goals_last3"]   = 1.5
                d["team_cs_rate_last3"] = 0.3
                d["opp_goals_last3"]    = 1.5
                d["opp_cs_rate_last3"]  = 0.3
            d["total_points"] = _safe_float(row["total_points"])
            rows_by_pos[pos].append(d)
    return rows_by_pos


def train_gw1_models(train_dfs, hist_team_form=None):
    rows_by_pos = build_hist_rows(train_dfs, hist_team_form)
    print("  Training GW1 models on all 6-season historical data:")
    return train_models(rows_by_pos)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_pool(pool, models, gw, current_squad_ids,
                 gw1_preds=None, loyalty_override=None, sold_last_gw=None):
    """
    Predict points for all players.
    gw1_preds: stored after GW1 — blended into GW2-8 predictions for stability.
    loyalty_override: overrides loyalty_bonus(gw) (used for forced WC at GW17).
    sold_last_gw: reserved for future sell-buyback penalty (not active).
    """
    bonus = loyalty_override if loyalty_override is not None else loyalty_bonus(gw)
    completed_gws = gw - 1
    season_weight = min(1.0, completed_gws / BLEND_GWS) if gw1_preds else 1.0

    for p in pool:
        pos   = p["pos"]
        pid   = p["player_id"]
        model = models.get(pos)
        feats = np.array([[p.get(f, 0.0) for f in FEAT_COLS]], dtype=float)

        if model is not None:
            pred = float(model.predict(feats)[0])
        else:
            pred = p.get("avg_points_per_game", 2.0)

        # FDR adjustment — GK/DEF use position-specific multiplier
        fdr      = p.get("fdr", 3.0)
        pos      = p.get("pos", "MID")
        fdr_mult = FDR_MULT_DEF if pos in ("GK", "DEF") else FDR_MULT
        pred *= max(0.5, 1.0 - fdr_mult * (fdr - 3.0))

        # Zero-minutes filter
        if p.get("zero_minutes", False):
            pred = 0.0

        # GW1: add ownership signal on top of model prediction
        if gw == 1:
            pred += p.get("sbp", 0.0) * OWN_BOOST_GW1

        # GW2-8: blend retrained pred with GW1 pred for early-season stability
        if gw1_preds and 2 <= gw <= 8 and pid in gw1_preds:
            pred = season_weight * pred + (1.0 - season_weight) * gw1_preds[pid]

        # Per-player prediction ceiling
        pred = min(max(0.0, pred), PRED_CAP)

        # Loyalty bonus for existing squad members
        if pid in current_squad_ids:
            pred += bonus

        p["pred"] = pred

    # Keep predictions calibrated — if top-11 sum exceeds cap, scale all down
    top11_sum = sum(sorted([p["pred"] for p in pool], reverse=True)[:11])
    if top11_sum > XI_PRED_CAP:
        scale = XI_PRED_CAP / top11_sum
        for p in pool:
            p["pred"] *= scale

    return pool


# ── Captain Selection ─────────────────────────────────────────────────────────

def select_captain(xi_pids, pool_by_pid, cap_streak=0, last_cap_pid=None,
                   captain_history=None, gw=1, current_squad_ids=None,
                   last_cap_actual=None):
    """
    Select captain by position-adjusted prediction with form gate, streak
    breaker, home advantage tiebreak, and minutes-reliability filter.
    GW1: no form data — use raw adj score only.
    """
    if captain_history is None:
        captain_history = []
    if current_squad_ids is None:
        current_squad_ids = set()
    use_form = (gw >= CAP_FORM_GW_MIN)

    candidates = []
    for pid in xi_pids:
        p = pool_by_pid.get(pid, {})
        if p.get("pos") == "GK":
            continue
        if p.get("minutes_reliability", 1.0) < CAP_MIN_RELIABILITY:
            continue

        raw_pred = p.get("pred", 0.0)
        adj      = raw_pred * CAP_MULT.get(p.get("element_type", 3), 1.0)
        form3    = p.get("form_last3", 0.0)
        fdr      = p.get("fdr", 3.0)
        was_home = p.get("was_home", 0)

        # Extra FDR penalty for captaincy — hard fixtures count double here
        adj *= max(0.5, 1.0 - CAP_FDR_MULT * (fdr - 3.0))

        if use_form:
            # Form gate — penalise out-of-form players
            if form3 < CAP_FORM_GATE:
                adj *= CAP_FORM_PENALTY
            # Streak breaker — penalise repeated captain if form has dropped
            if (cap_streak >= CAP_STREAK_LIMIT and
                    pid == last_cap_pid and
                    form3 < CAP_STREAK_FORM):
                adj *= CAP_STREAK_MULT
            # Blank rotation — penalise if this player blanked as captain last GW
            if (pid == last_cap_pid and
                    last_cap_actual is not None and
                    last_cap_actual <= CAP_BLANK_THRESH):
                adj *= CAP_BLANK_PENALTY
                print(f"  [CAP] GW{gw} blank penalty on {p.get('web_name',str(pid))} "
                      f"(capped last GW, scored {last_cap_actual}pts)")

        candidates.append({
            "adj": adj, "pid": pid, "raw": raw_pred,
            "form3": form3, "was_home": was_home,
            "name": p.get("web_name", str(pid)),
            "pos":  p.get("pos", "?"),
        })

    if not candidates:
        return last_cap_pid

    candidates.sort(key=lambda x: x["adj"], reverse=True)

    # Change 3: home advantage tiebreak — if top is away and 2nd is home
    # and within CAP_HOME_MARGIN pts, swap
    if (len(candidates) >= 2 and
            candidates[0]["was_home"] == 0 and
            candidates[1]["was_home"] == 1 and
            (candidates[0]["adj"] - candidates[1]["adj"]) <= CAP_HOME_MARGIN):
        candidates[0], candidates[1] = candidates[1], candidates[0]

    chosen = candidates[0]

    # Change 5: print top 3 candidates
    print(f"  [CAP] GW{gw} candidates:")
    for i, c in enumerate(candidates[:3]):
        tag = "  <- SELECTED" if i == 0 else ""
        print(f"    {i+1}. {c['name']} ({c['pos']}) "
              f"raw:{c['raw']:.1f} adj:{c['adj']:.1f} "
              f"form3:{c['form3']:.1f} home:{c['was_home']}{tag}")

    streak_now = cap_streak + 1 if chosen["pid"] == last_cap_pid else 1
    print(f"  [CAP] GW{gw} -> {chosen['name']} selected "
          f"(streak: {streak_now} consecutive)")

    return chosen["pid"]


# ── ILP Optimizer ─────────────────────────────────────────────────────────────

# Bonus consistency constants
BONUS_RATE_WEIGHT = 0.15   # how much bonus_rate affects prediction
BONUS_AVG_WEIGHT  = 0.10   # how much bonus_avg affects prediction
BONUS_MIN_GW      = 5      # only apply from GW5+ (need enough data)


def apply_bonus_adjustment(pool, bonus_lookup, gw):
    """
    Post-prediction adjustment based on bonus consistency.
    pred *= (1.0 + BONUS_RATE_WEIGHT * bonus_rate_last5)
    pred += BONUS_AVG_WEIGHT * bonus_avg_last5
    Only applied from GW5+. Only outfield players.
    """
    if gw < BONUS_MIN_GW:
        return pool, 0
    n_adjusted = 0
    for p in pool:
        pid = p["player_id"]
        if p.get("pos") == "GK":
            continue
        if p.get("zero_minutes", False):
            continue
        bl = bonus_lookup.get(pid, {})
        bonus_rate = bl.get("bonus_rate_last5", 0.0)
        bonus_avg  = bl.get("bonus_avg_last5",  0.0)
        if bonus_rate == 0.0 and bonus_avg == 0.0:
            continue
        old_pred = p["pred"]
        p["pred"] = old_pred * (1.0 + BONUS_RATE_WEIGHT * bonus_rate)                     + BONUS_AVG_WEIGHT * bonus_avg
        if p["pred"] != old_pred:
            n_adjusted += 1
    return pool, n_adjusted


def run_ilp(pool, current_squad_ids, available_budget, free_transfers,
            is_wildcard=False, is_freehit=False, gw=1, allow_fail=False):
    players = [p for p in pool if p.get("price", 0) > 0]
    n   = len(players)
    idx = list(range(n))

    pred  = [p["pred"]         for p in players]
    # corrected mode: owned players carry ilp_price = sell value, so the
    # budget identity  Σ_new mp ≤ bank + Σ_sold sv  holds exactly (§5)
    price = [p.get("ilp_price", p["price"]) for p in players]
    pos   = [p["element_type"] for p in players]
    team  = [p["team"]         for p in players]
    pid   = [p["player_id"]    for p in players]
    in_sq = [1 if p["player_id"] in current_squad_ids else 0 for p in players]

    no_transfers = (is_wildcard or is_freehit or gw == 1)

    # corrected mode: exactly one attempt — never spend phantom money
    n_attempts = 1 if RULES_MODE == "corrected" else 5
    for attempt in range(n_attempts):
        budget_limit = available_budget + 0.5 * attempt
        prob = LpProblem(f"fpl_gw{gw}_a{attempt}", LpMaximize)

        x = [LpVariable(f"x{i}", cat=LpBinary) for i in idx]  # squad
        s = [LpVariable(f"s{i}", cat=LpBinary) for i in idx]  # XI

        if not no_transfers:
            ti   = [LpVariable(f"ti{i}", cat=LpBinary) for i in idx]
            to   = [LpVariable(f"to{i}", cat=LpBinary) for i in idx]
            hits = LpVariable("hits", lowBound=0, cat=LpInteger)
        else:
            ti = to = hits = None

        # Objective: maximise XI score minus hit penalties
        obj = lpSum(pred[i] * s[i] for i in idx)
        if hits is not None:
            obj -= 4.0 * hits
        prob += obj

        # Squad structure
        prob += lpSum(x) == 15
        prob += lpSum(x[i] for i in idx if pos[i] == 1) == 2
        prob += lpSum(x[i] for i in idx if pos[i] == 2) == 5
        prob += lpSum(x[i] for i in idx if pos[i] == 3) == 5
        prob += lpSum(x[i] for i in idx if pos[i] == 4) == 3

        # Budget
        prob += lpSum(price[i] * x[i] for i in idx) <= budget_limit

        # Club limit
        for cl in set(team):
            prob += lpSum(x[i] for i in idx if team[i] == cl) <= MAX_CLUB

        # XI structure
        prob += lpSum(s) == 11
        prob += lpSum(s[i] for i in idx if pos[i] == 1) == 1
        prob += lpSum(s[i] for i in idx if pos[i] == 2) >= 3
        prob += lpSum(s[i] for i in idx if pos[i] == 2) <= 5
        prob += lpSum(s[i] for i in idx if pos[i] == 3) >= 2
        prob += lpSum(s[i] for i in idx if pos[i] == 3) <= 5
        prob += lpSum(s[i] for i in idx if pos[i] == 4) >= 1
        prob += lpSum(s[i] for i in idx if pos[i] == 4) <= 3

        for i in idx:
            prob += s[i] <= x[i]

        # Transfer constraints
        if ti is not None:
            for i in idx:
                prob += x[i] == in_sq[i] + ti[i] - to[i]
            prob += lpSum(ti) == lpSum(to)
            prob += hits >= lpSum(ti) - free_transfers
            prob += lpSum(ti) - free_transfers <= MAX_HITS  # max 1 hit

        prob.solve(PULP_CBC_CMD(msg=0))

        if LpStatus[prob.status] == "Optimal":
            sq_idx  = [i for i in idx if x[i].value() and x[i].value() > 0.5]
            xi_idx  = [i for i in sq_idx if s[i].value() and s[i].value() > 0.5]
            sq_pids = [pid[i] for i in sq_idx]
            xi_pids = [pid[i] for i in xi_idx]
            bench_pids = [p for p in sq_pids if p not in set(xi_pids)]

            if ti is not None:
                in_pids  = [pid[i] for i in idx if ti[i].value() and ti[i].value() > 0.5]
                out_pids = [pid[i] for i in idx if to[i].value() and to[i].value() > 0.5]
                n_hits   = max(0, int(round(hits.value() or 0)))
            else:
                in_pids  = [p for p in sq_pids if p not in current_squad_ids] if gw > 1 else sq_pids
                out_pids = [p for p in current_squad_ids if p not in set(sq_pids)] if gw > 1 else []
                n_hits   = 0

            return {
                "squad": sq_pids, "xi": xi_pids, "bench": bench_pids,
                "transfers_in": in_pids, "transfers_out": out_pids, "hits": n_hits,
            }

    if RULES_MODE == "corrected" and not allow_fail:
        raise RuntimeError(
            f"ILP infeasible at GW{gw} with budget {available_budget:.1f}m — "
            "corrected mode never relaxes the budget; investigate inputs")
    print(f"  [WARN] ILP infeasible at GW{gw} after {n_attempts} attempts")
    return None


# ── Auto-subs ─────────────────────────────────────────────────────────────────

def apply_auto_subs(xi_pids, bench_pids, pool_by_pid, gw_actuals):
    xi    = list(xi_pids)
    bench = list(bench_pids)
    subs  = []

    def pos(pid):  return pool_by_pid.get(pid, {}).get("pos", "MID")
    def mins(pid): return gw_actuals.get(pid, {}).get("minutes", 0)
    def valid(lst):
        defs = sum(1 for p in lst if pos(p) == "DEF")
        fwds = sum(1 for p in lst if pos(p) == "FWD")
        return defs >= 3 and fwds >= 1

    # Order bench: GK last, outfield by pred descending
    bench_sorted = sorted(bench,
        key=lambda p: (1 if pos(p) == "GK" else 0,
                       -pool_by_pid.get(p, {}).get("pred", 0)))

    for i, starter in enumerate(xi):
        if mins(starter) > 0:
            continue
        starter_pos = pos(starter)
        subbed = False

        # Same position first
        for sub in bench_sorted[:]:
            if sub not in bench or pos(sub) != starter_pos or mins(sub) == 0:
                continue
            new_xi = list(xi); new_xi[i] = sub
            if valid(new_xi) or starter_pos == "GK":
                xi[i] = sub
                bench.remove(sub); bench_sorted.remove(sub)
                subs.append([
                    pool_by_pid.get(starter, {}).get("web_name", str(starter)),
                    pool_by_pid.get(sub, {}).get("web_name", str(sub))
                ])
                subbed = True; break

        if subbed:
            continue

        # Any outfield
        for sub in bench_sorted[:]:
            if sub not in bench or pos(sub) == "GK" or mins(sub) == 0:
                continue
            new_xi = list(xi); new_xi[i] = sub
            if valid(new_xi):
                xi[i] = sub
                bench.remove(sub); bench_sorted.remove(sub)
                subs.append([
                    pool_by_pid.get(starter, {}).get("web_name", str(starter)),
                    pool_by_pid.get(sub, {}).get("web_name", str(sub))
                ])
                break

    return xi, bench, subs


# ── Chip Logic ────────────────────────────────────────────────────────────────

def _best_cap_stats(squad_pool, home_lookup, gw, loyalty_strip=0.0):
    """
    Return (best_adj, best_form3, best_home) for the best captain candidate.
    loyalty_strip: subtract this from pred before computing adj, so loyalty
    inflation does not cause premature TC/force-chip triggers.
    """
    best_adj, best_form3, best_home = 0.0, 0.0, False
    for p in squad_pool:
        if p.get("pos") == "GK":
            continue
        raw = max(0.0, p.get("pred", 0.0) - loyalty_strip)
        adj = raw * CAP_MULT.get(p.get("element_type", 3), 1.0)
        if adj > best_adj:
            best_adj  = adj
            best_form3 = p.get("form_last3", 0.0)
            best_home  = bool(home_lookup.get((p["team"], gw), 0))
    return best_adj, best_form3, best_home


def decide_chip(gw, chips_used, pool, current_squad_ids,
                dgw_gws, gw_teams, bench_pred_history, home_lookup,
                bb_target_gw=None, loyalty_strip=None):
    if gw <= CHIP_LOCKOUT:
        return None

    use_set1 = (gw <= 19)
    s1 = lambda c: c + "1"
    chip_name = lambda c: s1(c) if use_set1 else c + "2"

    squad_pool = [p for p in pool if p["player_id"] in current_squad_ids]

    # loyalty inflation to strip for chip-quality checks (0 in mp mode —
    # matrix predictions carry no loyalty)
    lb = loyalty_bonus(gw) if loyalty_strip is None else loyalty_strip

    # Diagnostic: log chip state each GW
    print(f"  [CHIP-CHECK] GW{gw} | chips_used: {sorted(chips_used)} | dgw: {gw in dgw_gws}")

    # ── Force-use safety net (Fix 3 — with quality gates) ────────────────────
    if gw == 17 and s1("wc") not in chips_used and use_set1:
        # WC17 force: loyalty override applied in main loop (Fix 3)
        return s1("wc")

    if gw == 18 and s1("tc") not in chips_used and use_set1:
        # Only use TC if captain's form_last3 >= TC_FORM_FORCE, else waste
        best_adj, best_form3, best_home = _best_cap_stats(squad_pool, home_lookup, gw, lb)
        if best_form3 >= TC_FORM_FORCE:
            return s1("tc")
        print(f"  [CHIP] GW18 TC force skipped — captain form3={best_form3:.1f} < {TC_FORM_FORCE}, wasting TC1")
        return None

    if gw == 19 and use_set1:
        for c in ["fh", "tc", "wc"]:   # BB excluded — let it be wasted naturally
            if s1(c) not in chips_used:
                if c == "fh":
                    # Only force FH if GW19 is actually a DGW — never waste it on a normal week
                    if gw in dgw_gws:
                        print(f"  [CHIP] GW19 FH1 force-used (DGW detected)")
                        return s1("fh")
                    else:
                        print(f"  [CHIP] GW19 FH1 NOT force-used — GW19 is not a DGW, wasting FH1")
                        continue
                elif c == "tc":
                    best_adj, best_form3, _ = _best_cap_stats(squad_pool, home_lookup, gw, lb)
                    if best_form3 >= TC_FORM_FORCE:
                        return s1("tc")
                    print(f"  [CHIP] GW19 TC force skipped — form3={best_form3:.1f} < {TC_FORM_FORCE}, wasting TC1")
                    continue
                else:
                    return s1(c)  # WC: use without additional checks

    # ── Natural triggers ──────────────────────────────────────────────────────

    # Free Hit: DGW with <6 squad players having DGW fixtures (lockout respected by outer check)
    fh2_blocked = (not use_set1 and gw < FH2_EARLIEST_GW)
    if gw in dgw_gws and gw > CHIP_LOCKOUT and not fh2_blocked:
        fh = chip_name("fh")
        if fh not in chips_used:
            dgw_count = sum(1 for p in squad_pool
                            if gw_teams.get(gw, {}).get(p["team"], 0) >= 2)
            if dgw_count < 6:
                print(f"  [CHIP] GW{gw} FH triggered naturally — DGW with only {dgw_count} DGW squad players")
                return fh

    # Wildcard: 5+ squad members below position average
    wc = chip_name("wc")
    if wc not in chips_used:
        pos_preds = defaultdict(list)
        for p in pool:
            pos_preds[p["pos"]].append(p["pred"])
        pos_mean = {ps: np.mean(vs) for ps, vs in pos_preds.items()}
        below = sum(1 for p in squad_pool if p["pred"] < pos_mean.get(p["pos"], 0))
        if below >= WC_THRESH:
            return wc

    # Triple Captain: loyalty-stripped adj >= TC_THRESH and form gate
    # Strip loyalty bonus so the threshold compares raw model quality,
    # matching intel_06's adj_pred (which also excludes loyalty).
    # No home requirement — a dominant away favourite still warrants TC.
    tc = chip_name("tc")
    if tc not in chips_used:
        if not use_set1 and FORCE_TC2_GW is not None:
            # Force tc2 on the exact GW specified — ignore threshold/form
            if gw == FORCE_TC2_GW:
                return tc
        elif not use_set1 and gw < TC2_MIN_GW:
            pass  # delay tc2 until TC2_MIN_GW
        else:
            best_adj, best_form3, best_home = _best_cap_stats(squad_pool, home_lookup, gw, lb)
            if best_adj >= TC_THRESH and best_form3 >= TC_FORM_MIN:
                return tc

    # Bench Boost: only trigger on Intel 07 target GW
    bb = chip_name("bb")
    if bb not in chips_used and bb_target_gw is not None:
        if gw == bb_target_gw:
            print(f"  [CHIP] BB triggered on Intel 07 target GW{gw}")
            return bb

    return None


# ── Chip Logic v2 — rolling-horizon, calendar-agnostic ───────────────────────
# docs/chip_strategy_redesign.md — decides WHEN from fixtures + squad state,
# never from hardcoded gameweek numbers.

def compute_blank_gws(gw_teams, max_gw=None):
    """GWs where at least one team has no fixture (a blank for that team)."""
    if max_gw is None:
        max_gw = SIM_END_GW
    all_teams = set()
    for teams in gw_teams.values():
        all_teams.update(teams.keys())
    return {g for g in range(1, max_gw + 1)
            if any(gw_teams.get(g, {}).get(t, 0) == 0 for t in all_teams)}


def build_lookahead_pool(pool, g, cur_gw, squad_ids,
                         fdr_lookup, home_lookup, gw_teams, lb,
                         bonus_strip=None):
    """
    Approximate the player pool under GW `g`'s fixture context (same trick as
    get_bench_intel's lookahead): copy current preds, swap fdr/home, zero
    blank teams, boost doubles. Current preds are de-inflated first — bench
    intel bonuses stripped, the CURRENT GW's DGW boost undone (so it isn't
    applied twice), and squad loyalty removed so chip-quality checks compare
    raw model signal (matches _best_cap_stats' loyalty_strip convention).
    Model preds are NOT re-run — fixture-count effects (x2 / x0) dominate
    chip value and are exact; form drift is absorbed by weekly re-planning.
    """
    cur_teams = gw_teams.get(cur_gw, {})
    fut_teams = gw_teams.get(g, {})
    out = []
    for p in pool:
        q    = dict(p)
        team = q.get("team")
        pred = q.get("pred", 0.0)
        if bonus_strip:
            pred -= bonus_strip.get(q["player_id"], 0.0)
        if cur_teams.get(team, 0) >= 2:
            pred /= DGW_PRED_MULT              # undo current-GW DGW boost (approx)
        if q["player_id"] in squad_ids:
            pred = max(0.0, pred - lb)         # strip loyalty inflation
        n_fix = fut_teams.get(team, 0)
        q["fdr"]      = float(fdr_lookup.get((team, g), 3.0))
        q["was_home"] = float(home_lookup.get((team, g), 0))
        if n_fix == 0:
            pred = 0.0                         # blank — no fixture, no points
        elif n_fix >= 2:
            pred = min(PRED_CAP, pred * DGW_PRED_MULT)
        q["pred"]  = max(0.0, pred)
        q["n_fix"] = n_fix
        out.append(q)
    return out


def _xi_greedy(players):
    """
    Best legal XI (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 total) by pred,
    greedy: fill position minima with the best of each, then top-up with the
    best remaining outfielders within position maxima.
    Returns (xi_players, xi_pred_total).
    """
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in players:
        pos = p.get("pos", "MID")
        if pos in by_pos:
            by_pos[pos].append(p)
    for lst in by_pos.values():
        lst.sort(key=lambda q: -q.get("pred", 0.0))
    xi   = by_pos["GK"][:1] + by_pos["DEF"][:3] + by_pos["MID"][:2] + by_pos["FWD"][:1]
    fill = by_pos["DEF"][3:5] + by_pos["MID"][2:5] + by_pos["FWD"][1:3]
    fill.sort(key=lambda q: -q.get("pred", 0.0))
    xi  += fill[:11 - len(xi)]
    return xi, sum(p.get("pred", 0.0) for p in xi)


def _best_feasible_assignment(V, reset_used_gws):
    """
    Assign chips to distinct GWs maximizing total value.
    Constraints: one chip per GW; an assigned FH/WC must sit >= SPACING_GAP
    GWs from every already-used or co-assigned reset chip. A chip may go
    unplaced (value 0) when placement is infeasible or worthless.
    Search space is tiny (<=3 chips x top-8 GWs) — brute force.
    """
    import itertools
    chips = [c for c in V if V[c]]
    if not chips:
        return {}
    options = {}
    for c in chips:
        top = sorted(V[c].items(), key=lambda kv: -kv[1])[:8]
        options[c] = top + [(None, 0.0)]
    best_plan, best_val = {}, -1.0
    for combo in itertools.product(*(options[c] for c in chips)):
        gws = [g for g, _ in combo if g is not None]
        if len(gws) != len(set(gws)):
            continue
        ok, resets = True, list(reset_used_gws)
        for c, (g, _) in zip(chips, combo):
            if g is not None and c in ("wc", "fh"):
                if any(abs(g - r) < SPACING_GAP for r in resets):
                    ok = False
                    break
                resets.append(g)
        if not ok:
            continue
        total = sum(v for _, v in combo)
        if total > best_val:
            best_val  = total
            best_plan = {c: g for c, (g, _) in zip(chips, combo) if g is not None}
    return best_plan


def decide_chip_v2(gw, chips_used, chip_gw_map, pool, current_squad_ids,
                   free_transfers, available_budget,
                   dgw_gws, gw_teams, blank_gws,
                   fdr_lookup, home_lookup, bench_bonus_map=None,
                   loyalty_strip=None):
    """
    Rolling-horizon chip scheduler. Each GW:
      1. Candidate weeks = near lookahead ∪ ALL known double/blank GWs left in
         the set (fixture calendar is announced ahead — schedule knowledge,
         not result leakage). This lets the planner RESERVE BB/FH for a known
         far event instead of burning them on a mediocre near week.
      2. Value BB/TC/FH per candidate week (FH only on event weeks — never
         burn it on a normal week; its value is budget-true via the same FH
         ILP the sim would actually run).
      3. Assign chips to weeks maximizing total value under constraints
         (one chip/GW, WC<->FH spacing); fire only THIS GW's assignment and
         only if it clears its bar — or unconditionally under deadline
         pressure (use-it-or-lose-it picks best remaining week by value).
      4. WC is squad-state-driven, not event-driven: evaluated rolling at the
         current GW ("how much do the next WC_HORIZON GWs gain if I rebuild
         NOW?", budget-true via the WC ILP) and fires when the gain clears
         the bar and reset spacing allows.
    """
    if gw <= CHIP_LOCKOUT:
        return None

    set_id  = 1 if gw <= 19 else 2
    set_end = 19 if set_id == 1 else SIM_END_GW
    remaining = [c for c in ("wc", "fh", "bb", "tc")
                 if f"{c}{set_id}" not in chips_used]
    if not remaining:
        return None

    lb = loyalty_bonus(gw) if loyalty_strip is None else loyalty_strip

    # GWs already holding a reset chip (spacing applies across set boundary too)
    reset_used_gws = [g for c, g in chip_gw_map.items() if c[:2] in ("wc", "fh")]

    # ── 1. Candidate weeks: near lookahead ∪ known event weeks in the set ──
    horizon_end = min(set_end, gw + CHIP_LOOKAHEAD)
    if set_end - gw <= CHIP_LOOKAHEAD:
        horizon_end = set_end
    event_gws     = {g for g in range(gw, set_end + 1)
                     if g in dgw_gws or g in blank_gws}
    candidate_gws = sorted(set(range(gw, horizon_end + 1)) | event_gws)

    lp_memo = {}
    def lp(g):
        if g not in lp_memo:
            lp_memo[g] = build_lookahead_pool(
                pool, g, gw, current_squad_ids,
                fdr_lookup, home_lookup, gw_teams, lb,
                bonus_strip=bench_bonus_map)
        return lp_memo[g]

    def squad_of(pool_g):
        return [p for p in pool_g if p["player_id"] in current_squad_ids]

    # ── 2. Value matrix for BB / TC / FH ────────────────────────────────────
    V = {c: {} for c in remaining if c != "wc"}
    for g in candidate_gws:
        pool_g  = lp(g)
        squad_g = squad_of(pool_g)

        if "bb" in V:
            _, xi_pred = _xi_greedy(squad_g)
            V["bb"][g] = sum(p["pred"] for p in squad_g) - xi_pred

        if "tc" in V:
            best_adj, best_form3, _ = _best_cap_stats(squad_g, home_lookup, g)
            if best_form3 >= TC_FORM_MIN:
                V["tc"][g] = best_adj

        if "fh" in V and (g in dgw_gws or g in blank_gws):
            res = run_ilp(pool_g, current_squad_ids, available_budget,
                          15, is_wildcard=False, is_freehit=True, gw=g,
                          allow_fail=True)
            if res is not None:
                by_pid = {p["player_id"]: p for p in pool_g}
                temp_xi_pred   = sum(by_pid.get(p, {}).get("pred", 0.0)
                                     for p in res["xi"])
                _, cur_xi_pred = _xi_greedy(squad_g)
                V["fh"][g] = temp_xi_pred - cur_xi_pred

    # ── 3. Assignment + commit rule ─────────────────────────────────────────
    plan   = _best_feasible_assignment(V, reset_used_gws)
    # Deadline pressure: no more spare weeks than unused chips — every
    # remaining week must fire per plan regardless of the value bar.
    forced = (set_end - gw + 1) <= len(remaining)
    bars   = {"bb": CHIP_BAR_BB, "tc": TC_THRESH, "fh": CHIP_BAR_FH}

    plan_str = ", ".join(f"{c}->GW{g}" for c, g in sorted(plan.items())) or "none"
    print(f"  [CHIP-V2] GW{gw} | remaining: {remaining} | plan: {plan_str}"
          f"{' | DEADLINE' if forced else ''}")

    chip_here = next((c for c, g in plan.items() if g == gw), None)
    if chip_here is not None:
        val = V[chip_here].get(gw, 0.0)
        if forced or val >= bars[chip_here]:
            print(f"  [CHIP-V2] firing {chip_here}{set_id} — value {val:.1f} "
                  f"(bar {bars[chip_here]:.1f}"
                  f"{', forced' if val < bars[chip_here] else ''})")
            return f"{chip_here}{set_id}"
        print(f"  [CHIP-V2] holding {chip_here} — value {val:.1f} < bar "
              f"{bars[chip_here]:.1f}")

    # ── 4. Wildcard: rolling evaluation at the current GW ───────────────────
    if "wc" in remaining:
        planned_fh  = plan.get("fh")
        near_resets = [g for g in reset_used_gws +
                       ([planned_fh] if planned_fh is not None else [])
                       if abs(gw - g) < SPACING_GAP]
        if near_resets:
            print(f"  [CHIP-V2] WC blocked — reset chip within "
                  f"{SPACING_GAP} GWs (at {near_resets})")
            return None

        pool_now  = lp(gw)
        squad_now = squad_of(pool_now)
        res = run_ilp(pool_now, current_squad_ids, available_budget,
                      15, is_wildcard=True, is_freehit=False, gw=gw,
                      allow_fail=True)
        if res is not None:
            reset_ids = set(res["squad"])
            gain = 0.0
            for k in range(gw, min(set_end, gw + WC_HORIZON - 1) + 1):
                pool_k = lp(k)
                _, reset_pred = _xi_greedy(
                    [p for p in pool_k if p["player_id"] in reset_ids])
                _, cur_pred = _xi_greedy(squad_of(pool_k))
                gain += reset_pred - cur_pred
            # Hits we'd otherwise pay fixing the squad with normal transfers
            pos_preds = defaultdict(list)
            for p in pool_now:
                pos_preds[p["pos"]].append(p["pred"])
            pos_mean = {ps: float(np.mean(v)) for ps, v in pos_preds.items()}
            below = sum(1 for p in squad_now
                        if p["pred"] < pos_mean.get(p["pos"], 0.0))
            gain += 4.0 * max(0, below - free_transfers)

            print(f"  [CHIP-V2] WC rolling value: {gain:.1f} "
                  f"(bar {CHIP_BAR_WC:.1f})")
            if forced or gain >= CHIP_BAR_WC:
                print(f"  [CHIP-V2] firing wc{set_id}"
                      f"{' (forced)' if gain < CHIP_BAR_WC else ''}")
                return f"wc{set_id}"

    return None


# ── Free Transfer Updates ─────────────────────────────────────────────────────

def next_ft(gw, ft_start, used, is_wc, is_fh):
    """Compute free transfers for next GW after completing gw."""
    if RULES_MODE == "corrected":
        # Real 2025-26 rules: +1/week, bank 1..5 from GW1, chips consume
        # nothing. One-off grants live in RULE_EVENTS_FT, not in code.
        return next_free_transfers(gw, ft_start, used, is_wc, is_fh,
                                   ft_cap=5, ft_events=RULE_EVENTS_FT)
    actual_used = 0 if (is_wc or is_fh or gw == 1) else used
    remaining   = max(0, ft_start - actual_used)
    next_gw     = gw + 1
    if next_gw == 15:
        return 5              # GW15 always gets 5 FTs flat (banked lost)
    cap = 5 if next_gw > 15 else 2
    return min(cap, remaining + 1)


# ── Score Computation ─────────────────────────────────────────────────────────

def compute_score(xi_pids, bench_pids, captain_pid, chip, gw_actuals,
                  penalty_pts, vice_pid=None):
    cap_mult = 3 if chip in ("tc1", "tc2") else 2
    # Real FPL rule: if the captain doesn't play, the vice gets the armband
    # (legacy calls pass no vice — behaviour unchanged there).
    eff_cap = captain_pid
    if (vice_pid is not None and
            gw_actuals.get(captain_pid, {}).get("minutes", 0) == 0):
        eff_cap = vice_pid
    total = 0
    for pid in xi_pids:
        pts = gw_actuals.get(pid, {}).get("total_points", 0)
        total += pts * cap_mult if pid == eff_cap else pts
    if chip in ("bb1", "bb2"):
        for pid in bench_pids:
            total += gw_actuals.get(pid, {}).get("total_points", 0)
    return int(total) - penalty_pts


def mp_chip_proxy_values(result, rows, current_squad, replacement, theta):
    """Current-week proxy values used by the unanchored-chip percentile bar."""
    captain = result.get("captain")
    tc_value = (milp_kappa(rows[captain], theta)
                if captain in rows else 0.0)
    bb_value = sum(
        r["mu"] * (1.0 - MP_W_BENCH * r["pi"])
        for pid in result.get("bench", [])
        if (r := rows.get(pid)) is not None
    )
    wc_value = sum(
        max(0.0, replacement.get(rows[pid]["element_type"], 0.0)
            - rows[pid]["mu"])
        for pid in current_squad if pid in rows
    )
    return {"tc": tc_value, "bb": bb_value, "wc": wc_value}


# ── Main Simulation ───────────────────────────────────────────────────────────

def run_simulation():
    print("=" * 70)
    print(f"  FPL Season Simulator — GW1 to GW{SIM_END_GW}")
    print(f"  SEASON={SIM_SEASON} | RULES_MODE={RULES_MODE} | "
          f"CHIP_STRATEGY={CHIP_STRATEGY} | OPTIMIZER={OPTIMIZER} | "
          f"out={os.path.basename(OUTPUT_JSON)}")
    print("=" * 70)

    print("\n[LOAD] Loading data...")
    hist_lookup                          = load_player_history()
    players_df                           = load_players_raw()
    fdr_lookup, home_lookup, dgw_gws, gw_teams = load_fixtures()
    blank_gws                            = compute_blank_gws(gw_teams)
    fixture_list = pmx.load_fixture_list(FIXTURES_CSV) if OPTIMIZER == "mp" else None
    train_dfs                            = load_training_data()
    avail_gws                            = load_availability()
    recs_gws                             = load_recommendations()
    pi_intel = (load_pi_intel()
                if (OPTIMIZER == "mp" and MP_PI_MODEL and MP_PI_INTEL)
                else {})
    if pi_intel:
        print(f"  [PI-MODEL] intel features for {len(pi_intel)} GWs")
    print(f"  Players: {len(players_df)} | DGW GWs in range: {sorted(g for g in dgw_gws if g <= SIM_END_GW)}")
    print(f"  Blank GWs in range: {sorted(g for g in blank_gws if g <= SIM_END_GW)} | Chip strategy: {CHIP_STRATEGY}")
    print(f"  Availability data: {len(avail_gws)} GWs loaded")

    print("\n[LOAD] Building historical team form lookup (Intel 08)...")
    hist_team_form = build_hist_team_form_lookup()

    print("\n[TRAIN] Initial models from all 6-season historical data...")
    models = train_gw1_models(train_dfs, hist_team_form)

    # Pre-build historical rows once — used as weighted prior in GW2+ retraining
    hist_rows_for_retrain = build_hist_rows(train_dfs, hist_team_form)

    # Weight for 2025-26 rows relative to historical rows.
    # Increases each GW so current-season signal dominates progressively.
    # GW2: 3x, GW10: 11x, GW38: 39x  (formula: 1 + completed_gws)
    CURRENT_SEASON_BASE_WEIGHT = 1  # historical weight = 1.0 always

    # State
    current_squad    = set()
    purchase_price   = {}    # pid -> £m paid (corrected-mode sell ledger)
    sold_at          = {}    # pid -> gw of last REAL sale (rebuy-lock memory)
    hits_paid        = 0     # season running total (MP_HIT_BUDGET enforcement)
    form_hold        = {}    # pid -> sell-penalty for last GW's haulers
    minutes_model    = MinutesModel() if OPTIMIZER == "mp" else None
    bank             = 0.0
    free_transfers   = 1
    chips_used       = set()
    chip_percentile = ChipPercentileLedger(
        MP_CHIP_PERCENTILE_STATE,
        q=MP_CHIP_PERCENTILE_Q,
        min_observations=MP_CHIP_PERCENTILE_WARMUP,
        load=MP_CHIP_PERCENTILE_RESUME,
    )
    pre_fh_squad     = None
    bank_pre_fh      = 0.0
    bench_pred_hist  = []
    # Captain tracking
    cap_streak       = 0
    last_cap_pid     = None
    last_cap_actual  = None   # actual pts scored by last GW's captain
    captain_history  = []
    gw1_preds        = None
    # Intel 08: team form (populated each GW from GW2 onward)
    team_form_lookup = {}
    bonus_lookup     = {}   # Intel 09: bonus consistency
    opp_lookup       = {}

    log = {
        "generated_at":     datetime.now().isoformat(),
        "rules_mode":       RULES_MODE,
        "chip_strategy":    CHIP_STRATEGY,
        "optimizer":        OPTIMIZER,
        "mp_horizon":       MP_HORIZON,
        "mp_theta":         MP_THETA,
        "mp_chips":         MP_CHIPS,
        "mp_hit_cost":      MP_HIT_COST,
        "mp_hit_cap":       MP_HIT_CAP,
        "mp_hit_budget":    MP_HIT_BUDGET,
        "mp_rebuy_gap":     MP_REBUY_GAP,
        "mp_ft_value":      MP_FT_VALUE,
        "mp_form_hold":     MP_FORM_HOLD,
        "mp_form_hold_min": MP_FORM_HOLD_MIN,
        "mp_pi_model":      MP_PI_MODEL,
        "mp_pi_intel":      MP_PI_INTEL,
        "mp_bench_slots":   list(MP_BENCH_SLOTS) if MP_BENCH_SLOTS else None,
        "mp_w_bench_gk":    MP_W_BENCH_GK,
        "mp_delta":         MP_DELTA,
        "mp_delta_chip":    MP_DELTA_CHIP,
        "mp_gamma":         MP_GAMMA,
        "mp_w_bench":       MP_W_BENCH,
        "mp_chip_bar":      MP_CHIP_BAR,
        "mp_chip_percentile_q": MP_CHIP_PERCENTILE_Q,
        "mp_chip_percentile_warmup": MP_CHIP_PERCENTILE_WARMUP,
        "mp_chip_percentile_state": MP_CHIP_PERCENTILE_STATE,
        "season":           SIM_SEASON,
        "total_actual_pts": 0,
        "total_predicted_pts": 0,
        "total_penalties":  0,
        "chips_used":       [],
        "learning_curve":   [],
        "gameweeks":        [],
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    for gw in range(1, SIM_END_GW + 1):
        print(f"\n{'-'*70}")
        print(f"  GW {gw}  |  FT: {free_transfers}  |  Bank: {bank:.1f}m")

        # ── Build pool ──────────────────────────────────────────────────────
        if gw == 1:
            pool = build_gw1_pool(players_df, train_dfs, fdr_lookup, home_lookup)
        else:
            # Intel 08: rebuild team form from accumulated actuals
            team_form_lookup = build_team_form_lookup(hist_lookup, players_df, gw - 1)
            opp_lookup       = build_opponent_lookup(FIXTURES_CSV, team_form_lookup)
            attacking = sorted(
                [(t, v["team_goals_last3"]) for (t, g), v in team_form_lookup.items() if g == gw],
                key=lambda x: -x[1]
            )[:3]
            defensive = sorted(
                [(t, v["team_cs_rate_last3"]) for (t, g), v in team_form_lookup.items() if g == gw],
                key=lambda x: -x[1]
            )[:3]
            print(f"  [TEAM FORM] GW{gw} top attack: {attacking} | top defence: {defensive}")
            pool = build_rolling_pool(players_df, hist_lookup,
                                      fdr_lookup, home_lookup, gw - 1,
                                      team_form_lookup=team_form_lookup,
                                      opp_lookup=opp_lookup)
        pool_by_pid = {p["player_id"]: p for p in pool}

        # ── Update available budget ─────────────────────────────────────────
        if gw == 1:
            available_budget = BUDGET
        elif RULES_MODE == "corrected":
            # Real FPL budget: bank + squad SELL value (50% profit rule).
            # Owned players also get ilp_price = sell value so the ILP's
            # budget constraint matches the transfer cash identity exactly.
            sv_total = 0.0
            for pid in current_squad:
                p  = pool_by_pid.get(pid)
                mp = p.get("price", 0.0) if p else 0.0
                sv = sell_value(purchase_price.get(pid, mp), mp)
                if p is not None:
                    p["ilp_price"] = sv
                sv_total += sv
            available_budget = bank + sv_total
        else:
            squad_val = sum(pool_by_pid.get(pid, {}).get("price", 0.0)
                            for pid in current_squad)
            available_budget = bank + squad_val

        # ── Fix 3 (legacy only): pre-detect WC17 force for loyalty override ─
        wc17_force    = (CHIP_STRATEGY == "legacy"
                         and gw == 17 and "wc1" not in chips_used)
        loyalty_ovr   = WC17_LOYALTY if wc17_force else None

        # ── Predict ─────────────────────────────────────────────────────────
        mp_rows = None
        if OPTIMIZER == "mp":
            # Phase 2: matrix predictions — per-fixture DGW sums, hard blank
            # zeros, availability tiers baked into mu. No loyalty/ownership/
            # caps/blending; pool preds mirror mu for chip logic + auto-subs.
            pi_overrides = None
            if MP_PI_MODEL:
                minutes_model.fit(hist_lookup, gw, intel=pi_intel)
                if minutes_model.ready:
                    pi_overrides = minutes_model.predict(
                        hist_lookup, [p["player_id"] for p in pool], gw,
                        intel=pi_intel)
                    print(f"  [PI-MODEL] learned pi for "
                          f"{len(pi_overrides)} players")
            mp_matrix = pmx.build_matrix(
                pool, models, fixture_list, gw, MP_HORIZON,
                hist_lookup=hist_lookup, avail_gws=avail_gws,
                purchase_price=purchase_price, pi_overrides=pi_overrides)
            mp_rows = mp_matrix[gw]
            for p in pool:
                p["pred"] = mp_rows.get(p["player_id"], {}).get("mu", 0.0)
        else:
            pool = predict_pool(pool, models, gw, current_squad,
                                gw1_preds=gw1_preds, loyalty_override=loyalty_ovr)

            # ── DGW boost: players with 2 fixtures this GW play twice ────────
            if gw in dgw_gws:
                n_dgw = 0
                for p in pool:
                    if gw_teams.get(gw, {}).get(p["team"], 0) >= 2:
                        p["pred"] = min(PRED_CAP, p["pred"] * DGW_PRED_MULT)
                        n_dgw += 1
                if n_dgw > 0:
                    print(f"  [DGW] GW{gw}: {n_dgw} players boosted x{DGW_PRED_MULT}")

        # ── Availability penalties (intel_03 tier data) ─────────────────────
        if OPTIMIZER != "mp":       # matrix already applies tier multipliers
            pool, _ = apply_availability_penalties(pool, avail_gws, gw)
        pool_by_pid = {p["player_id"]: p for p in pool}

        # Save GW1 predictions for early-season blending
        if gw == 1:
            gw1_preds = {p["player_id"]: p["pred"] for p in pool}

        # ── Intel 07: bench intelligence and BB targeting ────────────────────
        bb1_used    = "bb1" in chips_used
        bb2_used    = "bb2" in chips_used
        if OPTIMIZER == "mp" and MP_CHIPS == "model":
            # BB targeting is inside the MILP — intel_07 not needed
            bench_intel = {"bench_candidates": {}, "bb_target_gw": None,
                           "recommended_bench": [], "is_bb_target_gw": False}
        else:
            bench_intel = get_bench_intel(
                pool, gw, chips_used, bb1_used, bb2_used,
                fdr_lookup=fdr_lookup,
                home_lookup=home_lookup
            )

        # Apply bench candidate bonus to predictions (legacy only — the mp
        # objective prices the bench directly via its EV term)
        if OPTIMIZER != "mp":
            for p in pool:
                bpid = p["player_id"]
                if bpid in bench_intel["bench_candidates"]:
                    p["pred"] += bench_intel["bench_candidates"][bpid]

        pool_by_pid = {p["player_id"]: p for p in pool}  # refresh after bonuses

        # ── Chip decision ───────────────────────────────────────────────────
        if OPTIMIZER == "mp" and MP_CHIPS == "model":
            chip = None          # decided inside the horizon MILP below
        elif CHIP_STRATEGY == "v2":
            chip_gw_map = {c["chip"]: c["gw"] for c in log["chips_used"]}
            chip = decide_chip_v2(gw, chips_used, chip_gw_map, pool,
                                  current_squad, free_transfers,
                                  available_budget, dgw_gws, gw_teams,
                                  blank_gws, fdr_lookup, home_lookup,
                                  bench_bonus_map=({} if OPTIMIZER == "mp"
                                                   else bench_intel["bench_candidates"]),
                                  loyalty_strip=(0.0 if OPTIMIZER == "mp" else None))
        else:
            chip = decide_chip(gw, chips_used, pool, current_squad,
                               dgw_gws, gw_teams, bench_pred_hist, home_lookup,
                               bb_target_gw=bench_intel.get("bb_target_gw"),
                               loyalty_strip=(0.0 if OPTIMIZER == "mp" else None))
        is_wc = chip in ("wc1", "wc2")
        is_fh = chip in ("fh1", "fh2")
        is_tc = chip in ("tc1", "tc2")
        is_bb = chip in ("bb1", "bb2")

        # GW19 end-of-window diagnostic (legacy): confirm FH1 fate
        if (CHIP_STRATEGY == "legacy" and gw == 19
                and "fh1" not in chips_used and chip != "fh1"):
            print(f"  [CHIP] GW19 complete — FH1 wasted (not a DGW, better to waste than misuse)")

        if is_fh:
            pre_fh_squad = set(current_squad)
            bank_pre_fh  = bank

        ft_ilp = 15 if (gw == 1 or is_wc or is_fh) else free_transfers

        # ── ILP ─────────────────────────────────────────────────────────────
        vice_id = None
        if OPTIMIZER == "mp":
            # rebuy lock: pids sold recently enough that their lock window
            # (sold_gw + GAP) still covers a horizon week
            no_rebuy = {p: g0 + MP_REBUY_GAP for p, g0 in sold_at.items()
                        if MP_REBUY_GAP > 0 and g0 + MP_REBUY_GAP >= gw
                        and p not in current_squad}
            if no_rebuy:
                print(f"  [REBUY-LOCK] {len(no_rebuy)} recently sold "
                      f"players locked out")
            # season hit budget: shrink this week's cap to what remains
            if MP_HIT_BUDGET >= 0:
                hit_cap_now = max(0, min(MP_HIT_CAP,
                                         MP_HIT_BUDGET - hits_paid))
                if hit_cap_now < MP_HIT_CAP:
                    print(f"  [HIT-BUDGET] {hits_paid}/{MP_HIT_BUDGET} paid "
                          f"— cap now {hit_cap_now}/GW")
            else:
                hit_cap_now = MP_HIT_CAP

            def _plan_print(plan):
                nxt = [f"GW{g}:{len(w['transfers_in'])}t"
                       f"{'/-' + str(w['hits'] * 4) if w['hits'] else ''}"
                       for g, w in sorted(plan["weeks"].items())
                       if g > gw and w["transfers_in"]]
                chips_str = ", ".join(f"{k}@GW{g}" for g, k in
                                      sorted(plan.get("chips", {}).items()))
                print(f"  [MILP-H] plan ahead: {', '.join(nxt) or 'hold'}"
                      f" | chips: {chips_str or 'none'}"
                      f" | ft_plan {plan['ft_plan']}")

            if MP_CHIPS == "model" and MP_HORIZON > 1:
                # Phase 4: chips decided inside the MILP.
                # Scarcity guards (phase4_report.md fix): cold-start lockout,
                # BB/TC held for known far doubles, WC gated on squad state —
                # >= MP_WC_BELOW owned players under their position's
                # replacement level (25th pct of top-40 mu, from the matrix),
                # or the set deadline inside the horizon.
                horizon_end = min(gw + MP_HORIZON - 1, SIM_END_GW)
                mrows_t = mp_matrix[gw]
                repl = {}
                for et in (1, 2, 3, 4):
                    mus = sorted((r["mu"] for r in mrows_t.values()
                                  if r["element_type"] == et),
                                 reverse=True)[:40]
                    repl[et] = float(np.percentile(mus, 25)) if mus else 0.0
                best_mu = {}
                for rows2 in mp_matrix.values():
                    for pid2, r2 in rows2.items():
                        if pid2 in current_squad:
                            best_mu[pid2] = max(best_mu.get(pid2, 0.0),
                                                r2["mu"])
                below = sum(
                    1 for pid2 in current_squad
                    if best_mu.get(pid2, 0.0) <
                    repl.get(mrows_t.get(pid2, {}).get("element_type", 3),
                             0.0))
                set_end = 19 if gw <= 19 else SIM_END_GW
                wc_ok = below >= MP_WC_BELOW or set_end <= horizon_end
                if not wc_ok:
                    print(f"  [CHIP-GUARD] WC held — only {below} squad "
                          f"players below replacement (< {MP_WC_BELOW})")
                chip_state = {
                    "used": set(chips_used),
                    "reset_gws": [cc["gw"] for cc in log["chips_used"]
                                  if cc["chip"][:2] in ("wc", "fh")],
                    "far_dgw": {
                        1: any(horizon_end < d <= 19 for d in dgw_gws),
                        2: any(horizon_end < d <= SIM_END_GW and d >= 20
                               for d in dgw_gws),
                    },
                    "lockout_until": CHIP_LOCKOUT,
                    "wc_ok": wc_ok,
                }
                try:
                    plan = milp_solve_horizon(
                        mp_matrix, current_squad,
                        BUDGET if gw == 1 else bank,
                        free_transfers, gw, ft_events=RULE_EVENTS_FT, delta=MP_DELTA, delta_chip=MP_DELTA_CHIP,
                        chip_state=chip_state, theta=MP_THETA,
                        hit_cost=MP_HIT_COST, hit_cap=hit_cap_now, ft_value=MP_FT_VALUE, sell_hold=form_hold, gamma=MP_GAMMA, w_bench=MP_W_BENCH, w_bench_slots=MP_BENCH_SLOTS, w_bench_gk=MP_W_BENCH_GK,
                        no_rebuy=no_rebuy)
                except RuntimeError as e:
                    print(f"  [MILP-H] chipped solve failed ({e}) — "
                          f"retrying without chips")
                    # Preserve the theta ablation even if the chipped solve
                    # falls back to the no-chip horizon model.
                    plan = milp_solve_horizon(
                        mp_matrix, current_squad,
                        BUDGET if gw == 1 else bank,
                        free_transfers, gw, ft_events=RULE_EVENTS_FT, delta=MP_DELTA, delta_chip=MP_DELTA_CHIP,
                        theta=MP_THETA,
                        hit_cost=MP_HIT_COST, hit_cap=hit_cap_now, ft_value=MP_FT_VALUE, sell_hold=form_hold, gamma=MP_GAMMA, w_bench=MP_W_BENCH, w_bench_slots=MP_BENCH_SLOTS, w_bench_gk=MP_W_BENCH_GK,
                        no_rebuy=no_rebuy)
                chip  = plan["chips"].get(gw)
                # A plain-week chip must clear the KIND-level historical bar
                # (wc/tc/bb — one season-wide series per kind, so set-2 chips
                # inherit set-1 history instead of a fresh warm-up free pass).
                # NO deadline disarm: tried 2026-07-15, cost 54 pts on
                # 2025-26 — it released wc1 at GW15 on a weak week the bar
                # was rightly holding; the MILP's within-horizon placement
                # is noisier than the bar near set ends.  Future planned
                # chips are deliberately left free (every GW is re-solved).
                plain_week = all(r["n_fix"] == 1 for r in mrows_t.values())
                bar_active = plain_week and MP_CHIP_BAR
                blocked_now = set()
                while chip and bar_active and chip[:2] in ("wc", "tc", "bb"):
                    result = plan["weeks"][gw]
                    proxies = mp_chip_proxy_values(
                        result, mrows_t, current_squad, repl, MP_THETA)
                    kind = chip[:2]
                    if chip_percentile.allows(kind, proxies[kind]):
                        break
                    blocked_now.add(kind)
                    cutoff = chip_percentile.threshold(kind)
                    print(f"  [CHIP-BAR] holding {chip}: {proxies[kind]:.2f} "
                          f"< q{MP_CHIP_PERCENTILE_Q:.2f} {cutoff:.2f}")
                    retry_state = dict(chip_state)
                    retry_state["blocked_now"] = blocked_now
                    plan = milp_solve_horizon(
                        mp_matrix, current_squad,
                        BUDGET if gw == 1 else bank,
                        free_transfers, gw, ft_events=RULE_EVENTS_FT, delta=MP_DELTA, delta_chip=MP_DELTA_CHIP,
                        chip_state=retry_state, theta=MP_THETA,
                        hit_cost=MP_HIT_COST, hit_cap=hit_cap_now, ft_value=MP_FT_VALUE, sell_hold=form_hold, gamma=MP_GAMMA, w_bench=MP_W_BENCH, w_bench_slots=MP_BENCH_SLOTS, w_bench_gk=MP_W_BENCH_GK,
                        no_rebuy=no_rebuy)
                    chip = plan["chips"].get(gw)

                result = plan["weeks"][gw]
                # The bar's population is PLAIN weeks only: DGW-doubled (or
                # blank-deflated) proxies distort the threshold plain weeks
                # must clear (2023-24 has Set-1 events; excluding them was
                # worth +32).  Keyed by KIND so the series spans the season.
                if plain_week:
                    final_proxies = mp_chip_proxy_values(
                        result, mrows_t, current_squad, repl, MP_THETA)
                    chip_percentile.record({
                        kind: value
                        for kind, value in final_proxies.items()
                        if (f"{kind}1" not in chips_used
                            or f"{kind}2" not in chips_used)
                    })
                is_wc = chip in ("wc1", "wc2")
                is_fh = chip in ("fh1", "fh2")
                is_tc = chip in ("tc1", "tc2")
                is_bb = chip in ("bb1", "bb2")
                if is_fh:
                    pre_fh_squad = set(current_squad)
                    bank_pre_fh  = bank
                _plan_print(plan)
            elif is_fh or MP_HORIZON <= 1:
                # FH squad reverts — horizon planning is moot for it
                result = milp_solve_gw(mp_rows, current_squad,
                                       available_budget, ft_ilp, gw,
                                       is_wildcard=is_wc, is_freehit=is_fh,
                                       theta=MP_THETA,
                                       hit_cost=MP_HIT_COST, hit_cap=hit_cap_now, ft_value=MP_FT_VALUE, sell_hold=form_hold, gamma=MP_GAMMA, w_bench=MP_W_BENCH, w_bench_slots=MP_BENCH_SLOTS, w_bench_gk=MP_W_BENCH_GK,
                                       no_rebuy=set(no_rebuy))
            else:
                try:
                    plan = milp_solve_horizon(
                        mp_matrix, current_squad,
                        BUDGET if gw == 1 else bank,
                        free_transfers, gw,
                        is_wildcard_now=is_wc, ft_events=RULE_EVENTS_FT, delta=MP_DELTA, delta_chip=MP_DELTA_CHIP,
                        theta=MP_THETA,
                        hit_cost=MP_HIT_COST, hit_cap=hit_cap_now, ft_value=MP_FT_VALUE, sell_hold=form_hold, gamma=MP_GAMMA, w_bench=MP_W_BENCH, w_bench_slots=MP_BENCH_SLOTS, w_bench_gk=MP_W_BENCH_GK,
                        no_rebuy=no_rebuy)
                    result = plan["weeks"][gw]
                    _plan_print(plan)
                except RuntimeError as e:
                    # fallback ladder (blueprint §7.3): horizon -> H=1
                    print(f"  [MILP-H] horizon failed ({e}) — "
                          f"falling back to H=1")
                    result = milp_solve_gw(mp_rows, current_squad,
                                           available_budget, ft_ilp, gw,
                                           is_wildcard=is_wc, is_freehit=False,
                                           theta=MP_THETA,
                                           hit_cost=MP_HIT_COST, hit_cap=hit_cap_now, ft_value=MP_FT_VALUE, sell_hold=form_hold, gamma=MP_GAMMA, w_bench=MP_W_BENCH, w_bench_slots=MP_BENCH_SLOTS, w_bench_gk=MP_W_BENCH_GK,
                                           no_rebuy=set(no_rebuy))
        else:
            result = run_ilp(pool, current_squad, available_budget,
                             ft_ilp, is_wc, is_fh, gw)
        if result is None:
            print(f"  [ERROR] ILP failed GW{gw}, skipping.")
            continue

        xi_pids    = result["xi"]
        bench_pids = result["bench"]
        t_in_pids  = result["transfers_in"]
        t_out_pids = result["transfers_out"]
        n_hits     = result["hits"]

        # ── Captain ─────────────────────────────────────────────────────────
        if OPTIMIZER == "mp":
            # in-MILP captain/vice (kappa = pi*[(1-theta)mu + theta*q90]) —
            # no streak/blank/form heuristics, no intel_05 override
            captain_id, vice_id, cap_source = (result["captain"],
                                               result["vice"], "milp")
        else:
            captain_id = select_captain(xi_pids, pool_by_pid, cap_streak, last_cap_pid,
                                        captain_history, gw, last_cap_actual=last_cap_actual)
            # Intel_05 override (0.5 adj-pt threshold, post-ILP, no squad effect)
            captain_id, cap_source = apply_intel_captain_override(
                captain_id, xi_pids, pool_by_pid, recs_gws, gw)

        # ── Update bank from transfers ───────────────────────────────────────
        if gw == 1:
            squad_cost = sum(pool_by_pid.get(p, {}).get("price", 0.0)
                             for p in result["squad"])
            bank = BUDGET - squad_cost
            if RULES_MODE == "corrected":
                # ledger init: GW1 squad bought at market price
                purchase_price.clear()
                purchase_price.update({
                    p: pool_by_pid.get(p, {}).get("price", 0.0)
                    for p in result["squad"]
                })
        elif RULES_MODE == "corrected":
            # Sells credit SELL value (50% profit rule); buys debit market.
            # Same formula covers normal weeks and WC/FH: run_ilp returns
            # t_in/t_out as the squad diff on chip weeks too.
            _mp  = lambda p: pool_by_pid.get(p, {}).get("price", 0.0)
            sell = sum(sell_value(purchase_price.get(p, _mp(p)), _mp(p))
                       for p in t_out_pids)
            buy  = sum(_mp(p) for p in t_in_pids)
            bank += sell - buy
            if bank < -1e-6:
                raise RuntimeError(f"GW{gw}: bank negative ({bank:.2f}m) — "
                                   "sell-value accounting violated")
            if not is_fh:
                # FH squad is temporary: ledger untouched, bank reverts below
                for p in t_out_pids:
                    purchase_price.pop(p, None)
                for p in t_in_pids:
                    purchase_price[p] = _mp(p)
                # rebuy-lock memory: buys clear their lock, sales start one.
                # WC sales DO seed locks — exempting them was tried
                # 2026-07-15 and lost 88/66 pts on the neutral seasons:
                # committing to a wildcard's dropped players for GAP weeks
                # is empirically worth more than post-WC flexibility.
                for p in t_in_pids:
                    sold_at.pop(p, None)
                for p in t_out_pids:
                    sold_at[p] = gw
        elif not (is_wc or is_fh):
            sell = sum(pool_by_pid.get(p, {}).get("price", 0.0) for p in t_out_pids)
            buy  = sum(pool_by_pid.get(p, {}).get("price", 0.0) for p in t_in_pids)
            bank += sell - buy
        else:
            # WC/FH: new squad cost
            new_cost = sum(pool_by_pid.get(p, {}).get("price", 0.0)
                           for p in result["squad"])
            bank = available_budget - new_cost

        penalty_pts = n_hits * 4
        hits_paid  += n_hits
        if chip:
            chips_used.add(chip)
            log["chips_used"].append({"chip": chip, "gw": gw})

        current_squad = set(result["squad"])

        # ── Actuals ─────────────────────────────────────────────────────────
        gw_actuals = {}
        for pid in current_squad:
            if pid in hist_lookup and gw in hist_lookup[pid]:
                gw_actuals[pid] = hist_lookup[pid][gw]
            else:
                gw_actuals[pid] = {"total_points": 0, "minutes": 0}

        # form hold for NEXT GW's solve: this week's double-digit haulers
        # cost extra to sell (skip FH weeks — shadow squad reverts)
        if MP_FORM_HOLD > 0 and not is_fh:
            form_hold = {
                pid: MP_FORM_HOLD for pid, a in gw_actuals.items()
                if a.get("total_points", 0) >= MP_FORM_HOLD_MIN
            }
        elif not is_fh:
            form_hold = {}

        # ── Auto-subs ───────────────────────────────────────────────────────
        xi_final, bench_final, auto_subs = apply_auto_subs(
            xi_pids, bench_pids, pool_by_pid, gw_actuals)

        # ── Score ────────────────────────────────────────────────────────────
        actual_total    = compute_score(xi_final, bench_final, captain_id,
                                        chip, gw_actuals, penalty_pts,
                                        vice_pid=vice_id)
        pred_total      = sum(pool_by_pid.get(p, {}).get("pred", 0) for p in xi_pids)
        if is_tc:
            pred_total += pool_by_pid.get(captain_id, {}).get("pred", 0)

        # ── MAE (GW2+) ───────────────────────────────────────────────────────
        if gw >= 2:
            xi_preds   = [pool_by_pid.get(p, {}).get("pred", 0.0) for p in xi_pids]
            xi_acts    = [gw_actuals.get(p, {}).get("total_points", 0.0) for p in xi_pids]
            mae = float(np.mean(np.abs(np.array(xi_preds) - np.array(xi_acts))))
            log["learning_curve"].append({"gw": gw, "mae": round(mae, 3)})

        # Update captain streak + history
        if captain_id == last_cap_pid:
            cap_streak += 1
        else:
            cap_streak = 1
        last_cap_actual = gw_actuals.get(captain_id, {}).get("total_points", 0)
        last_cap_pid    = captain_id
        captain_history.append(captain_id)

        # ── Bench pred history (retained for reference) ──────────────────────
        bench_of = sum(pool_by_pid.get(p, {}).get("pred", 0.0)
                       for p in bench_pids
                       if pool_by_pid.get(p, {}).get("pos") != "GK")
        bench_pred_hist.append(bench_of)

        # ── Update squad value & bank post-GW ───────────────────────────────
        squad_end_val = sum(
            hist_lookup.get(pid, {}).get(gw, {}).get("value",
                pool_by_pid.get(pid, {}).get("price", 0.0))
            for pid in current_squad
        )

        if is_fh and pre_fh_squad is not None:
            # Revert squad for next GW
            current_squad = set(pre_fh_squad)
            bank = bank_pre_fh
            squad_end_val = sum(
                hist_lookup.get(pid, {}).get(gw, {}).get("value",
                    pool_by_pid.get(pid, {}).get("price", 0.0))
                for pid in current_squad
            )
            pre_fh_squad = None

        # Bank after price movements: total = bank + squad_val (invariant)
        # Bank itself doesn't change from price movements — only squad_val does

        # ── Free transfer rollover ───────────────────────────────────────────
        used_fts       = 0 if (is_wc or is_fh or gw == 1) else len(t_in_pids)
        free_transfers = next_ft(gw, free_transfers, used_fts, is_wc, is_fh)

        # ── Retrain ─────────────────────────────────────────────────────────
        if gw < SIM_END_GW:
            # current-season weight grows each GW: GW1->2x, GW10->11x, GW38->39x
            cs_weight = CURRENT_SEASON_BASE_WEIGHT + gw
            print(f"  [RETRAIN] History (w=1.0) + GW1-{gw} actuals (w={cs_weight})...")
            retrain_rows = build_retrain_rows(
                players_df, hist_lookup, fdr_lookup, home_lookup, gw,
                team_form_lookup=team_form_lookup, opp_lookup=opp_lookup)
            combined_rows = {}
            combined_weights = {}
            for pos in ["GK", "DEF", "MID", "FWD"]:
                h_rows = hist_rows_for_retrain.get(pos, [])
                r_rows = retrain_rows.get(pos, [])
                combined_rows[pos] = h_rows + r_rows
                combined_weights[pos] = (
                    [1.0] * len(h_rows) + [float(cs_weight)] * len(r_rows)
                )
            if any(combined_rows.values()):
                models = train_models(combined_rows, combined_weights)

        # ── Log entry ───────────────────────────────────────────────────────
        cap_mult_log = 3 if is_tc else 2

        def player_entry(pid, is_cap=False):
            p   = pool_by_pid.get(pid, {})
            act = gw_actuals.get(pid, {})
            apt = float(act.get("total_points", 0))
            cm  = cap_mult_log if is_cap else 1
            return {
                "player_id":         pid,
                "web_name":          p.get("web_name", str(pid)),
                "pos":               p.get("pos", "?"),
                "team":              p.get("team", 0),
                "price":             round(p.get("price", 0), 1),
                "predicted_pts":     round(p.get("pred", 0), 2),
                "actual_pts":        int(apt),
                "is_captain":        is_cap,
                "captain_multiplier": cm,
                "pts_counted":       int(apt * cm),
            }

        xi_entries    = [player_entry(p, p == captain_id) for p in xi_final]
        bench_entries = [player_entry(p) for p in bench_final]

        # Players subbed OUT (were in original XI, got 0 mins) go to bench display
        xi_final_set = set(xi_final)
        for pid in xi_pids:
            if pid not in xi_final_set:
                entry = player_entry(pid)
                entry["auto_subbed_out"] = True
                bench_entries.append(entry)

        cap_pool_entry = pool_by_pid.get(captain_id, {})
        cap_actual_pts = gw_actuals.get(captain_id, {}).get("total_points", 0)
        gw_entry = {
            "gw":             gw,
            "chip":           chip,
            "free_transfers": ft_ilp,
            "bank":           round(bank, 1),
            "squad_value":    round(squad_end_val, 1),
            "transfers_in":   [pool_by_pid.get(p, {}).get("web_name", str(p)) for p in t_in_pids],
            "transfers_out":  [pool_by_pid.get(p, {}).get("web_name", str(p)) for p in t_out_pids],
            "penalty_pts":    -penalty_pts,
            "xi":             xi_entries,
            "bench":          bench_entries,
            "captain_id":     captain_id,
            "vice_id":        vice_id,
            "vice_name":      (pool_by_pid.get(vice_id, {}).get("web_name")
                               if vice_id else None),
            "captain_source": cap_source,
            "captain": {
                "player_id":  captain_id,
                "web_name":   cap_pool_entry.get("web_name", str(captain_id)),
                "pos":        cap_pool_entry.get("pos", "?"),
                "actual_pts": cap_actual_pts,
                "predicted_pts": round(cap_pool_entry.get("pred", 0), 1),
                "source":     cap_source,
            },
            "predicted_total": round(pred_total, 1),
            "actual_total":   actual_total,
            "auto_subs":               auto_subs,
            "bench_intel_target":      bench_intel.get("bb_target_gw"),
            "bench_intel_recommended": [p["web_name"] for p in bench_intel.get("recommended_bench", [])],
            "is_bb_target_gw":         bench_intel.get("is_bb_target_gw", False),
        }
        log["gameweeks"].append(gw_entry)
        log["total_actual_pts"]     += actual_total
        log["total_predicted_pts"]  += pred_total
        log["total_penalties"]      -= penalty_pts

        cap_name = pool_by_pid.get(captain_id, {}).get("web_name", str(captain_id))
        chip_str = f" [{chip.upper()}]" if chip else ""
        print(f"  Predicted: {pred_total:.1f} | Actual: {actual_total} | "
              f"Cap: {cap_name}{chip_str} | Hits: {n_hits}")
        print(f"  FT next GW: {free_transfers} | Bank: {bank:.1f}m | "
              f"Squad val: {squad_end_val:.1f}m")

        with open(OUTPUT_JSON, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  [SAVED]")

    print(f"\n{'='*70}")
    print(f"  SIMULATION COMPLETE")
    print(f"  Total actual pts:    {log['total_actual_pts']}")
    print(f"  Total predicted:     {log['total_predicted_pts']:.1f}")
    print(f"  Total penalties:     {log['total_penalties']}")
    print(f"  Chips used:          {[c['chip'] for c in log['chips_used']]}")
    print("=" * 70)
    return log


if __name__ == "__main__":
    run_simulation()
