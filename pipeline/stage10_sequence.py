"""
pipeline/stage10_sequence.py
Stage 10 Phase 1 — Step 2: leakage-safe per-player gameweek sequence builder.

One sample = (player `name`, target season `S`, target GW `t`). The sequence is
that player's timeline of GWs 1..t-1 WITHIN season S only (season reset — rule #3).
Cross-season continuity is carried by the static career_* features already baked
into each base row by Stage 6, not by the recurrence.

OFFLINE (this module): built from
  - data/processed/base_gw_table.csv   pre-kickoff snapshot + realized outcomes
  - models/stage10/oof_preds.csv       walk-forward GBM prediction + residual
RUNTIME (stage10_refine.py, step 4/5): the same tensor layout, built from
  season_simulator's hist_lookup + pool — SAMPLE_SPEC is the shared contract.

Leakage contract (asserted in tests/test_stage10_sequence.py):
  * every timestep's source GW g satisfies g < t
  * no field references a season != S
  * mu_gbm_oof[g] was itself trained only on {season < S} ∪ {season S, GW < g}
    (guaranteed by stage10_oof.py's protocol) — safe as a timestep input
  * mutating actual[t] changes ONLY y_resid, never ts_feats / query

Note on folds: the earliest season (2019-20) has no OOF residuals (no prior
season to walk-forward from) so it is GBM-history-only and never a sequence
target/train season. LSTM pretrain folds therefore start at 2021-22:
  train {2020-21} -> val 2021-22 ; ... ; train {2020-21..2024-25} -> val 2025-26
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# canonical 27-feature order, already kept in sync with season_simulator.FEAT_COLS
from prediction_matrix import DEFAULT_FEAT_COLS as FEAT_COLS   # noqa: E402

BASE_CSV = os.path.join(_ROOT, "data", "processed", "base_gw_table.csv")
OOF_CSV = os.path.join(_ROOT, "models", "stage10", "oof_preds.csv")

POSITIONS = ["GK", "DEF", "MID", "FWD"]
POS_ID = {p: i for i, p in enumerate(POSITIONS)}

GAP_CAP = 10
STREAK_CAP = 10
ROLL_N = 5
AVAIL_TIER_DEFAULT = 4          # "available" on a 0..4 ordinal scale
DEFAULT_MIN_HISTORY = 1         # need >= 1 prior timestep to be a training sample

OUTCOME_COLS = ["total_points", "minutes", "goals_scored", "assists",
                "clean_sheets", "goals_conceded", "bonus", "bps", "saves"]

# --- timestep feature layout (F columns, fixed order) ------------------------
_TS_TAIL = [
    "mu_gbm_oof", "residual",
    "o_points", "o_minutes", "o_goals", "o_assists", "o_cs", "o_gc", "o_bonus",
    "o_bps", "o_saves",
    "ts_was_home", "ts_fdr", "ts_is_dgw", "ts_is_blank",
    "played", "gap_since_played", "did_not_feature", "is_first_appearance",
    "streak_len", "roll_std_points5", "roll_std_minutes5",
    # Stage 10 fix 4 (ceiling features max_points_last5 / hauls_last10 /
    # p75_points_last8) was tried and RULED OUT — added per-fold gate-2
    # instability with no mean gain, and worsened captain regret on all folds
    # (same over-promotes-boom-bust failure as fix 3). See report §12.
    "avail_tier_ord", "rotation_score", "intel_present",
    "pos_gk", "pos_def", "pos_mid", "pos_fwd",
]
TS_COLS = list(FEAT_COLS) + _TS_TAIL
F = len(TS_COLS)

# --- query feature layout (Q columns, fixed order) --------------------------
_Q_TAIL = [
    "mu_gbm_t",
    "q_was_home", "q_fdr", "q_is_dgw", "q_is_blank",
    "n_played", "n_timesteps", "gws_into_season",
    "pos_gk", "pos_def", "pos_mid", "pos_fwd",
]
QUERY_COLS = list(FEAT_COLS) + _Q_TAIL
Q = len(QUERY_COLS)

SAMPLE_SPEC = {
    "feat_cols": list(FEAT_COLS),
    "ts_cols": TS_COLS, "F": F,
    "query_cols": QUERY_COLS, "Q": Q,
    "positions": POSITIONS,
    "gap_cap": GAP_CAP, "streak_cap": STREAK_CAP, "roll_n": ROLL_N,
}


@dataclass
class Sample:
    name: str
    season: str
    target_gw: int
    position: str
    pos_id: int
    ts_feats: np.ndarray            # [L, F]
    query: np.ndarray              # [Q]
    length: int
    n_played: int
    y_resid: float
    mu_gbm_t: float
    actual_t: float
    ts_gws: list = field(default_factory=list)   # source GW of each timestep (test hook)


# ---------------------------------------------------------------------------
# Load / join
# ---------------------------------------------------------------------------
def load_joined(base_csv: str = BASE_CSV, oof_csv: str = OOF_CSV) -> pd.DataFrame:
    """base_gw_table x oof_preds on (name, season, GW). Left join keeps every
    base row; mu_gbm_oof is NaN for rows with no OOF (2019-20, and any GW1
    handled separately)."""
    b = pd.read_csv(base_csv)
    b = b.loc[:, ~b.columns.duplicated()]
    b["season"] = b["season"].astype(str)
    b["GW"] = b["GW"].astype(int)
    if "was_home" in b.columns:
        b["was_home"] = b["was_home"].astype(float)
    for c in FEAT_COLS:
        if c not in b.columns:
            b[c] = 0.0
    b[FEAT_COLS] = b[FEAT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for c in OUTCOME_COLS:
        if c not in b.columns:
            b[c] = 0.0
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0.0)

    o = pd.read_csv(oof_csv)
    o["season"] = o["season"].astype(str)
    o["GW"] = o["GW"].astype(int)
    o = o[["name", "season", "GW", "mu_gbm_oof"]]

    df = b.merge(o, on=["name", "season", "GW"], how="left")
    if "current_gw_fdr" in df.columns:
        df["_fdr"] = pd.to_numeric(df["current_gw_fdr"], errors="coerce").fillna(3.0)
    else:
        df["_fdr"] = 3.0
    df["_home"] = df["was_home"].fillna(0.0).astype(float) if "was_home" in df.columns else 0.0
    df = df.sort_values(["name", "season", "GW"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-(player, season) timestep table  — built ONCE, sliced per target
# ---------------------------------------------------------------------------
def _player_season_timesteps(g: pd.DataFrame, intel_lookup):
    """g = one (name, season) group, sorted by GW. Returns a DataFrame with one
    row per GW the player has a base row for, columns = TS_COLS, plus 'GW'."""
    gw = g["GW"].to_numpy()
    mins = g["minutes"].to_numpy(dtype=float)
    played = (mins > 0).astype(float)

    # gap_since_played / streak_len computed over GW *numbers* (blanks widen the gap)
    gap = np.zeros(len(gw))
    streak = np.zeros(len(gw))
    last_played_gw = None
    run = 0
    for i in range(len(gw)):
        if last_played_gw is None:
            gap[i] = 0.0
        else:
            gap[i] = min(GAP_CAP, gw[i] - last_played_gw)
        streak[i] = min(STREAK_CAP, run)
        if played[i] > 0:
            run += 1
            last_played_gw = gw[i]
        else:
            run = 0

    pts = g["total_points"].to_numpy(dtype=float)
    roll_sp = np.zeros(len(gw))
    roll_sm = np.zeros(len(gw))
    for i in range(len(gw)):
        w_p = pts[max(0, i - ROLL_N + 1): i + 1]
        w_m = mins[max(0, i - ROLL_N + 1): i + 1]
        roll_sp[i] = float(np.std(w_p)) if len(w_p) > 1 else 0.0
        roll_sm[i] = float(np.std(w_m)) if len(w_m) > 1 else 0.0

    name = g["name"].iloc[0]
    season = g["season"].iloc[0]
    pos = str(g["position"].iloc[0])
    pid_oh = [1.0 if pos == p else 0.0 for p in POSITIONS]

    mu_oof = g["mu_gbm_oof"].to_numpy(dtype=float)
    mu_oof_filled = np.where(np.isnan(mu_oof), pts * 0.0 + np.nan, mu_oof)
    resid = pts - mu_oof            # NaN where no OOF (won't be used as a timestep < t
                                    #   unless t is in a val season, where all g<t have OOF)

    out = pd.DataFrame(index=range(len(gw)))
    out["GW"] = gw
    for j, c in enumerate(FEAT_COLS):
        out[c] = g[c].to_numpy(dtype=float)
    out["mu_gbm_oof"] = np.nan_to_num(mu_oof_filled, nan=0.0)
    out["residual"] = np.nan_to_num(resid, nan=0.0)
    out["o_points"] = pts
    out["o_minutes"] = mins
    out["o_goals"] = g["goals_scored"].to_numpy(dtype=float)
    out["o_assists"] = g["assists"].to_numpy(dtype=float)
    out["o_cs"] = g["clean_sheets"].to_numpy(dtype=float)
    out["o_gc"] = g["goals_conceded"].to_numpy(dtype=float)
    out["o_bonus"] = g["bonus"].to_numpy(dtype=float)
    out["o_bps"] = g["bps"].to_numpy(dtype=float)
    out["o_saves"] = g["saves"].to_numpy(dtype=float)
    out["ts_was_home"] = g["_home"].to_numpy(dtype=float)
    out["ts_fdr"] = g["_fdr"].to_numpy(dtype=float)
    out["ts_is_dgw"] = 0.0
    out["ts_is_blank"] = 0.0
    out["played"] = played
    out["gap_since_played"] = gap
    out["did_not_feature"] = 1.0 - played
    out["is_first_appearance"] = 0.0
    if len(out):
        out.loc[0, "is_first_appearance"] = 1.0
    out["streak_len"] = streak
    out["roll_std_points5"] = roll_sp
    out["roll_std_minutes5"] = roll_sm

    tier = np.full(len(gw), float(AVAIL_TIER_DEFAULT))
    rot = np.zeros(len(gw))
    present = np.zeros(len(gw))
    if intel_lookup is not None:
        for i, gnum in enumerate(gw):
            v = intel_lookup.get((name, season, int(gnum)))
            if v is not None:
                tier[i] = float(v.get("avail_tier_ord", AVAIL_TIER_DEFAULT))
                rot[i] = float(v.get("rotation_score", 0.0))
                present[i] = 1.0
    out["avail_tier_ord"] = tier
    out["rotation_score"] = rot
    out["intel_present"] = present

    out["pos_gk"], out["pos_def"], out["pos_mid"], out["pos_fwd"] = pid_oh
    return out, pos, season, name


# ---------------------------------------------------------------------------
# Build samples
# ---------------------------------------------------------------------------
def build_samples(target_seasons, *, base_csv: str = BASE_CSV,
                  oof_csv: str = OOF_CSV, min_history: int = DEFAULT_MIN_HISTORY,
                  max_len: int = 38, intel_lookup=None, joined: pd.DataFrame = None,
                  require_oof_target: bool = True) -> list[Sample]:
    """
    target_seasons : iterable of season strings a sample's TARGET GW may fall in.
    require_oof_target : if True, only build a sample when (name, season, t) has an
                         OOF prediction (i.e. a defined residual target). For
                         inference-shaped builds set False.
    """
    target_seasons = set(map(str, target_seasons))
    df = joined if joined is not None else load_joined(base_csv, oof_csv)

    samples: list[Sample] = []
    for (name, season), g in df.groupby(["name", "season"], sort=False):
        if season not in target_seasons:
            continue
        g = g.sort_values("GW").reset_index(drop=True)
        ts_table, pos, _, _ = _player_season_timesteps(g, intel_lookup)
        if pos not in POS_ID:
            continue
        gw_arr = ts_table["GW"].to_numpy()
        ts_mat_full = ts_table[TS_COLS].to_numpy(dtype=float)
        mu_oof_arr = g["mu_gbm_oof"].to_numpy(dtype=float)
        pts_arr = g["total_points"].to_numpy(dtype=float)
        mins_arr = g["minutes"].to_numpy(dtype=float)

        for i, t in enumerate(gw_arr):
            t = int(t)
            if t < 2:
                continue                       # GW1 is a Stage-10 no-op (blind)
            prior = np.where(gw_arr < t)[0]     # timestep rows strictly before t
            if len(prior) < min_history:
                continue
            mu_t = mu_oof_arr[i]
            if require_oof_target and np.isnan(mu_t):
                continue
            prior = prior[-max_len:]
            ts_feats = ts_mat_full[prior].copy()
            n_played = int((mins_arr[prior] > 0).sum())

            # query row for target GW t (pre-kickoff snapshot from base row i)
            q = np.zeros(Q)
            q[:len(FEAT_COLS)] = g[FEAT_COLS].to_numpy(dtype=float)[i]
            tail = dict(
                mu_gbm_t=float(0.0 if np.isnan(mu_t) else mu_t),
                q_was_home=float(g["_home"].to_numpy(dtype=float)[i]),
                q_fdr=float(g["_fdr"].to_numpy(dtype=float)[i]),
                q_is_dgw=0.0, q_is_blank=0.0,
                n_played=float(n_played), n_timesteps=float(len(prior)),
                gws_into_season=float(t),
                pos_gk=float(pos == "GK"), pos_def=float(pos == "DEF"),
                pos_mid=float(pos == "MID"), pos_fwd=float(pos == "FWD"),
            )
            for k, name_c in enumerate(_Q_TAIL):
                q[len(FEAT_COLS) + k] = tail[name_c]

            y = float(pts_arr[i] - mu_t) if not np.isnan(mu_t) else float("nan")
            samples.append(Sample(
                name=name, season=season, target_gw=t, position=pos,
                pos_id=POS_ID[pos], ts_feats=ts_feats, query=q,
                length=len(prior), n_played=n_played, y_resid=y,
                mu_gbm_t=float(0.0 if np.isnan(mu_t) else mu_t),
                actual_t=float(pts_arr[i]),
                ts_gws=[int(x) for x in gw_arr[prior]],
            ))
    return samples


def to_padded_arrays(samples: list[Sample], max_len: int = 38):
    """(ts [N,L,F], query [N,Q], lengths [N], pos_ids [N], y [N], meta DataFrame).
    Sequences are RIGHT-padded with zeros (real steps at indices 0..length-1);
    `lengths` holds the true length. Both the torch model (gather at length-1)
    and numpy_forward (slice [:length]) assume front-aligned sequences."""
    n = len(samples)
    L = min(max_len, max((s.length for s in samples), default=1))
    ts = np.zeros((n, L, F), dtype=np.float32)
    query = np.zeros((n, Q), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    pos_ids = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.float32)
    meta = []
    for k, s in enumerate(samples):
        seq = s.ts_feats[-L:]               # keep the most recent L steps
        lengths[k] = len(seq)
        ts[k, :len(seq)] = seq             # right-pad
        query[k] = s.query
        pos_ids[k] = s.pos_id
        y[k] = s.y_resid
        meta.append((s.name, s.season, s.target_gw, s.position,
                     s.n_played, s.mu_gbm_t, s.actual_t))
    meta_df = pd.DataFrame(meta, columns=["name", "season", "target_gw",
                                          "position", "n_played", "mu_gbm_t",
                                          "actual_t"])
    return ts, query, lengths, pos_ids, y, meta_df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+",
                    default=["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])
    args = ap.parse_args()
    print(f"SAMPLE_SPEC: F={F} Q={Q}")
    s = build_samples(args.seasons)
    print(f"built {len(s):,} samples over {args.seasons}")
    ts, q, ln, pid, y, meta = to_padded_arrays(s)
    print(f"ts {ts.shape}  query {q.shape}  lengths[min/mean/max]="
          f"{ln.min()}/{ln.mean():.1f}/{ln.max()}")
    print(f"y_resid mean={np.nanmean(y):+.3f} std={np.nanstd(y):.3f}  "
          f"NaN={int(np.isnan(y).sum())}")
    print(meta.groupby(['season', 'position']).size().to_string())
