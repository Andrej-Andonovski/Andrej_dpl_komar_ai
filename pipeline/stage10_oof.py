"""
pipeline/stage10_oof.py
Stage 10 Phase 1 — Step 1: out-of-fold GBM predictions = the residual training target.

For every historical (name, season, GW) row in data/processed/train_*.csv, produce
the point prediction a walk-forward LightGBM (production LGBM_PARAMS + FEAT_COLS)
would have made WITHOUT seeing that row's season at or after that gameweek.

Protocol (matches season_simulator's online-retrain weighting — docs/stage10_phase1_plan.md §4.1):

    predict (season S, GW t) with a model trained on
        {rows: season <  S}                 weight 1.0
      U {rows: season == S and GW < t}       weight t   (= 1 + completed_gws)
    t == 1  ->  historical only, no in-season rows   (blind; == train_gw1_models)

Row construction is Stage 6's (the train_*.csv rows) for BOTH training and
prediction — one consistent method, a cleaner walk-forward than the production
dual-method mix (build_hist_rows + build_retrain_rows). The runtime residual in
stage10_refine is computed against the real production mu regardless.

Config (FEAT_COLS / LGBM_PARAMS / _COL_ALIAS / TRAIN_FILES) is imported from
season_simulator — never reimplemented (drift = biased residuals).

Usage:
    python pipeline/stage10_oof.py            # full: per-(season, GW) models  (~20-40 min)
    python pipeline/stage10_oof.py --quick    # per-season only (24 fits, fast dev)
    python pipeline/stage10_oof.py --report   # re-print distributions from an existing csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# --- production config: single source of truth -------------------------------
from pipeline.season_simulator import (          # noqa: E402
    FEAT_COLS, LGBM_PARAMS, _COL_ALIAS, TRAIN_FILES, TRAIN_DIR,
)
import lightgbm as lgb                            # noqa: E402

OUT_DIR = os.path.join(_ROOT, "models", "stage10")
OUT_CSV = os.path.join(OUT_DIR, "oof_preds.csv")

ID_COLS = ["name", "season", "GW", "position"]
TARGET = "total_points"
POSITIONS = ["GK", "DEF", "MID", "FWD"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_position_frame(pos: str) -> pd.DataFrame:
    """train_<pos>.csv with production column aliasing, all FEAT_COLS present."""
    path = os.path.join(TRAIN_DIR, TRAIN_FILES[pos])
    df = pd.read_csv(path)
    df = df.rename(columns=_COL_ALIAS)
    df = df.loc[:, ~df.columns.duplicated()]
    missing = [c for c in ID_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{TRAIN_FILES[pos]} missing required columns: {missing}")
    # Position-specific prev cols (prev_int_per_90 etc.) are absent for some
    # positions — same handling as build_hist_rows: reindex + fill 0.0.
    feat = df.reindex(columns=FEAT_COLS).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out = pd.concat([df[ID_COLS + [TARGET]].reset_index(drop=True),
                     feat.reset_index(drop=True)], axis=1)
    out["GW"] = out["GW"].astype(int)
    out["season"] = out["season"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Walk-forward OOF
# ---------------------------------------------------------------------------
def _fit_predict(X_tr, y_tr, w_tr, X_pred):
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    return model.predict(X_pred)


def oof_for_position(df: pd.DataFrame, pos: str, quick: bool) -> pd.DataFrame:
    seasons = sorted(df["season"].unique())
    val_seasons = seasons[1:]                       # earliest season is train-only
    Xall = df[FEAT_COLS].values
    yall = df[TARGET].values
    recs = []
    t0 = time.time()

    for S in val_seasons:
        hist_mask = (df["season"] < S).values
        cur_mask = (df["season"] == S).values
        Xh, yh = Xall[hist_mask], yall[hist_mask]
        cur = df[cur_mask]
        if len(Xh) == 0 or len(cur) == 0:
            continue

        if quick:
            preds = _fit_predict(Xh, yh, np.ones(len(Xh)), cur[FEAT_COLS].values)
            regime = "season_wf"
            for (idx, row), pr in zip(cur.iterrows(), preds):
                recs.append((row["name"], S, int(row["GW"]), pos,
                             float(pr), float(row[TARGET]), len(Xh), regime))
        else:
            for t in sorted(cur["GW"].unique()):
                prior = cur[cur["GW"] < t]
                tgt = cur[cur["GW"] == t]
                Xp = np.vstack([Xh, prior[FEAT_COLS].values]) if len(prior) else Xh
                yp = np.concatenate([yh, prior[TARGET].values]) if len(prior) else yh
                wp = np.concatenate([np.ones(len(Xh)),
                                     np.full(len(prior), float(t))])
                preds = _fit_predict(Xp, yp, wp, tgt[FEAT_COLS].values)
                regime = "blind_gw1" if t == 1 else "online"
                for (idx, row), pr in zip(tgt.iterrows(), preds):
                    recs.append((row["name"], S, int(t), pos,
                                 float(pr), float(row[TARGET]), len(Xp), regime))

        print(f"    {pos}  val={S}  cur_rows={len(cur):>6}  "
              f"cum_recs={len(recs):>7}  ({time.time()-t0:5.1f}s)")

    return pd.DataFrame(recs, columns=[
        "name", "season", "GW", "position",
        "mu_gbm_oof", "actual", "n_train_rows", "regime"])


# ---------------------------------------------------------------------------
# Distribution report
# ---------------------------------------------------------------------------
def _ascii_hist(vals, lo=-8, hi=12, bins=20, width=54):
    vals = np.asarray(vals, dtype=float)
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(np.clip(vals, lo, hi), bins=edges)
    peak = counts.max() or 1
    lines = []
    for i in range(bins):
        bar = "#" * int(round(width * counts[i] / peak))
        lines.append(f"  [{edges[i]:6.1f},{edges[i+1]:6.1f})  {counts[i]:7d} |{bar}")
    return "\n".join(lines)


def report(df: pd.DataFrame):
    df = df.copy()
    df["residual"] = df["actual"] - df["mu_gbm_oof"]
    r = df["residual"].values
    ae_mu = np.abs(r)                                  # MAE of the raw GBM
    sep = "=" * 78

    print(f"\n{sep}\n  STAGE 10 — OOF RESIDUAL DISTRIBUTIONS\n{sep}")
    print(f"  rows: {len(df):,}   seasons: {sorted(df['season'].unique())}")
    print(f"  regimes: {df['regime'].value_counts().to_dict()}")

    def block(name, sub):
        rr = (sub["actual"] - sub["mu_gbm_oof"]).values
        ae = np.abs(rr)
        q = np.percentile(rr, [1, 5, 25, 50, 75, 95, 99])
        print(f"\n  {name}  (n={len(sub):,})")
        print(f"    residual  mean(bias)={rr.mean():+.3f}  std={rr.std():.3f}  "
              f"MAE={ae.mean():.3f}  median={np.median(rr):+.3f}")
        print(f"    quantiles 1/5/25/50/75/95/99: "
              + " ".join(f"{v:+.2f}" for v in q))
        print(f"    within +/-1: {np.mean(ae <= 1)*100:5.1f}%   "
              f"within +/-2: {np.mean(ae <= 2)*100:5.1f}%   "
              f"within +/-3: {np.mean(ae <= 3)*100:5.1f}%")
        print(f"    actual mean={sub['actual'].mean():.3f}  "
              f"pred mean={sub['mu_gbm_oof'].mean():.3f}  "
              f"pred range=[{sub['mu_gbm_oof'].min():.2f}, {sub['mu_gbm_oof'].max():.2f}]")

    block("OVERALL", df)

    print(f"\n{sep}\n  BY POSITION\n{sep}")
    for pos in POSITIONS:
        sub = df[df["position"] == pos]
        if len(sub):
            block(pos, sub)

    print(f"\n{sep}\n  BY VALIDATION SEASON\n{sep}")
    for S in sorted(df["season"].unique()):
        block(S, df[df["season"] == S])

    print(f"\n{sep}\n  BY GAMEWEEK BUCKET\n{sep}")
    buckets = [("GW1 (blind)", df["GW"] == 1),
               ("GW2-5", df["GW"].between(2, 5)),
               ("GW6-19", df["GW"].between(6, 19)),
               ("GW20-38", df["GW"].between(20, 38))]
    for name, mask in buckets:
        sub = df[mask]
        if len(sub):
            block(name, sub)

    print(f"\n{sep}\n  RESIDUAL HISTOGRAM (clipped to [-8, 12])\n{sep}")
    print(_ascii_hist(r))

    # sanity vs stage7_results.json expected MAE band
    print(f"\n{sep}\n  SANITY\n{sep}")
    print(f"  overall GBM MAE = {ae_mu.mean():.3f}  "
          f"(stage7 walk-forward weighted MAE band ~2.0-2.9 per position)")
    print(f"  overall bias    = {r.mean():+.4f}  (should be small; large => "
          f"systematic GBM miscalibration the LSTM can exploit)")
    hi_var = df.assign(ar=np.abs(r)).groupby("position")["ar"].mean()
    print(f"  MAE by position:\n{hi_var.to_string()}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="per-season models only (24 fits) instead of per-(season, GW)")
    ap.add_argument("--report", action="store_true",
                    help="re-print distributions from an existing oof_preds.csv")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.report:
        if not os.path.exists(OUT_CSV):
            sys.exit(f"no file at {OUT_CSV} — run without --report first")
        report(pd.read_csv(OUT_CSV))
        return

    print("=" * 78)
    print(f"  STAGE 10 OOF  |  mode={'quick' if args.quick else 'full'}  "
          f"|  FEAT_COLS={len(FEAT_COLS)}  |  LGBM={LGBM_PARAMS['n_estimators']}t "
          f"d{LGBM_PARAMS['max_depth']}")
    print("=" * 78)

    parts = []
    for pos in POSITIONS:
        print(f"\n  [{pos}] loading {TRAIN_FILES[pos]} ...")
        df = load_position_frame(pos)
        print(f"    {len(df):,} rows  seasons={sorted(df['season'].unique())}")
        parts.append(oof_for_position(df, pos, args.quick))

    oof = pd.concat(parts, ignore_index=True)
    oof["residual"] = oof["actual"] - oof["mu_gbm_oof"]
    oof = oof[["name", "season", "GW", "position", "mu_gbm_oof", "actual",
               "residual", "n_train_rows", "regime"]]
    oof.to_csv(OUT_CSV, index=False)
    print(f"\n  saved {len(oof):,} rows -> {OUT_CSV}")

    report(oof)


if __name__ == "__main__":
    main()
