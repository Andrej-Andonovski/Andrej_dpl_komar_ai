"""
Stage 7 — XGBoost Model Training with Optuna Hyperparameter Tuning
===================================================================
Trains 4 separate XGBoost regression models (GK, DEF, MID, FWD) to predict
total_points per gameweek. Uses walk-forward temporal validation and Optuna
for 50-trial hyperparameter search.

Usage:
    python pipeline/train_xgboost_stage7.py
"""

import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Columns to exclude from features
# ---------------------------------------------------------------------------
EXCLUDE_COLS = {
    "name", "season", "GW", "team", "opponent_team", "position",
    "was_home", "fdr_is_proxy", "trajectory_is_full",
    "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "bonus", "bps", "yellow_cards", "red_cards",
    "cumulative_points_season", "saves",
    "total_points",  # target
}

TARGET = "total_points"

# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------
ALL_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

FOLDS = [
    (["2019-20"],                                                    "2020-21"),
    (["2019-20", "2020-21"],                                         "2021-22"),
    (["2019-20", "2020-21", "2021-22"],                              "2022-23"),
    (["2019-20", "2020-21", "2021-22", "2022-23"],                   "2023-24"),
    (["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"],        "2024-25"),
]
FOLD_WEIGHTS = [1, 1.5, 2, 2.5, 3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns: all numeric cols not in EXCLUDE_COLS."""
    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            feature_cols.append(col)
    return feature_cols


def build_split(df: pd.DataFrame, train_seasons: list[str], val_season: str,
                feature_cols: list[str]):
    """Return (X_train, y_train, X_val, y_val) for a single fold."""
    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"] == val_season
    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, TARGET].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, TARGET].values
    return X_train, y_train, X_val, y_val


def top_k_precision(df_val: pd.DataFrame, y_pred: np.ndarray, k: int) -> float:
    """
    For each GW in the validation fold:
      - rank players by predicted score (descending)
      - take top-k
      - fraction of those top-k who scored above the GW median actual score
    Return mean across GWs.
    """
    df_val = df_val.copy()
    df_val["_pred"] = y_pred
    df_val["_actual"] = df_val[TARGET].values

    gw_precisions = []
    for gw, grp in df_val.groupby("GW"):
        if len(grp) < k:
            continue
        median_score = grp["_actual"].median()
        top_k_idx = grp.nlargest(k, "_pred").index
        above_median = (grp.loc[top_k_idx, "_actual"] > median_score).sum()
        gw_precisions.append(above_median / k)

    return float(np.mean(gw_precisions)) if gw_precisions else 0.0


def make_objective(df: pd.DataFrame, feature_cols: list[str]):
    """Return an Optuna objective function for a given position dataframe."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        }

        fold_maes = []
        for train_seasons, val_season in FOLDS:
            X_tr, y_tr, X_val, y_val = build_split(df, train_seasons, val_season, feature_cols)
            if len(X_val) == 0:
                fold_maes.append(None)
                continue
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            mae = float(np.mean(np.abs(preds - y_val)))
            fold_maes.append(mae)

        valid_maes = [m for m in fold_maes if m is not None]
        valid_weights = [FOLD_WEIGHTS[i] for i, m in enumerate(fold_maes) if m is not None]
        total_w = sum(valid_weights)
        weighted_mae = sum(m * w for m, w in zip(valid_maes, valid_weights)) / total_w
        return weighted_mae

    return objective


def train_position(pos_name: str, csv_path: str) -> dict:
    """Full train+tune pipeline for one position. Returns summary dict."""
    banner = f"  {pos_name}  "
    border = "=" * (len(banner) + 4)
    print(f"\n{border}")
    print(f"| {banner} |")
    print(f"{border}")

    t0 = time.time()

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} cols from {os.path.basename(csv_path)}")

    # Cast was_home to int if present
    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(int)

    feature_cols = get_feature_cols(df)
    print(f"Feature columns: {len(feature_cols)}")

    # ---- Optuna tuning ----
    print(f"Running Optuna (50 trials) ...")
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(df, feature_cols), n_trials=50, show_progress_bar=False)

    best_params = study.best_params
    print("\nBest hyperparameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # ---- Per-fold evaluation with best params ----
    final_params = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        **best_params,
    }

    print("\nWalk-forward validation with best params:")
    fold_maes = []
    fold_top3 = []
    fold_top5 = []

    for i, (train_seasons, val_season) in enumerate(FOLDS):
        X_tr, y_tr, X_val, y_val = build_split(df, train_seasons, val_season, feature_cols)
        model = xgb.XGBRegressor(**final_params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        mae = float(np.mean(np.abs(preds - y_val)))
        fold_maes.append(mae)

        # For top-k we need GW column from val set
        val_df = df[df["season"] == val_season].copy()
        p3 = top_k_precision(val_df, preds, k=3)
        p5 = top_k_precision(val_df, preds, k=5)
        fold_top3.append(p3)
        fold_top5.append(p5)

        print(f"  Fold {i+1} (train={train_seasons[0]}->{train_seasons[-1]}, "
              f"val={val_season}): MAE={mae:.4f} | top3={p3:.3f} | top5={p5:.3f}")

    total_w = sum(FOLD_WEIGHTS)
    weighted_mae = sum(m * w for m, w in zip(fold_maes, FOLD_WEIGHTS)) / total_w
    mean_top3 = float(np.mean(fold_top3))
    mean_top5 = float(np.mean(fold_top5))
    print(f"\nWeighted MAE: {weighted_mae:.4f}")
    print(f"Mean top3 precision: {mean_top3:.3f}")
    print(f"Mean top5 precision: {mean_top5:.3f}")

    # ---- Feature importances ----
    # Fit on all seasons for importance (same as final model below)
    X_all = df[feature_cols].values
    y_all = df[TARGET].values
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_all, y_all)

    importances = final_model.get_booster().get_score(importance_type="gain")
    # Map back from f0, f1... to actual feature names
    feat_imp = {feature_cols[int(k[1:])]: v for k, v in importances.items()}
    top10 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 features (gain):")
    for rank, (feat, score) in enumerate(top10, 1):
        print(f"  {rank:2d}. {feat:<45s} {score:.2f}")

    # ---- Save model ----
    model_path = os.path.join(MODELS_DIR, f"xgb_{pos_name.lower()}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": feature_cols}, f)
    print(f"\nModel saved to {model_path}")

    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s")

    return {
        "best_params": best_params,
        "mae_per_fold": fold_maes,
        "weighted_mae": weighted_mae,
        "mean_top3_precision": mean_top3,
        "mean_top5_precision": mean_top5,
        "feature_count": len(feature_cols),
        "training_rows": len(df),
        "training_time_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    positions = [
        ("GK",  os.path.join(DATA_DIR, "train_gk.csv")),
        ("DEF", os.path.join(DATA_DIR, "train_def.csv")),
        ("MID", os.path.join(DATA_DIR, "train_mid.csv")),
        ("FWD", os.path.join(DATA_DIR, "train_fwd.csv")),
    ]

    results = {}
    overall_t0 = time.time()

    for pos_name, csv_path in positions:
        results[pos_name] = train_position(pos_name, csv_path)

    # ---- Save JSON summary ----
    summary_path = os.path.join(MODELS_DIR, "stage7_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nStage 7 complete. Summary saved to {summary_path}")
    print(f"Total elapsed: {time.time() - overall_t0:.1f}s")

    print("\n=== STAGE 7 SUMMARY ===")
    for pos, r in results.items():
        print(f"  {pos}: weighted_MAE={r['weighted_mae']:.4f} | "
              f"top3={r['mean_top3_precision']:.3f} | "
              f"top5={r['mean_top5_precision']:.3f} | "
              f"features={r['feature_count']} | "
              f"rows={r['training_rows']:,}")
