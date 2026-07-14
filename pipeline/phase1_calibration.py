"""
pipeline/phase1_calibration.py
Phase 1 calibration report (docs/optimizer_redesign.md §3, §9 Phase 1).

Walk-forward over 2025-26: at each GW t, train models exactly as the season
simulator would (same data, same weights), build the prediction matrix for
offsets 0..H-1, and score every (player, future-GW) prediction against the
actual outcome. Produces THE number the MILP design depends on: MAE by
horizon distance (how fast do predictions rot with lookahead?), plus the
phi-bucket gate (§3.5) and q90 coverage (§3.6).

Also scores a "sim-style" offset-0 baseline (model pred + legacy FDR
post-multipliers) — exit criterion: the feature-swap matrix must not be
worse than that at offset 0.

Run (background, ~25-40 min):
  docker run --rm -v "<repo>:/app" -w /app fpl-sim python -u \
      pipeline/phase1_calibration.py --out data/intel/phase1_calibration.json
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/app")
try:
    from pipeline import season_simulator as sim
    from pipeline import prediction_matrix as pm
except ImportError:
    import season_simulator as sim
    import prediction_matrix as pm

TOPN = 60          # decision-relevant stratum: top-N by mu per (t, offset)
Q90_MAX_OFFSET = 2 # q90 coverage measured on near offsets only
PHI_BUCKETS = [(0.0, 0.70), (0.70, 0.85), (0.85, 1.01)]


def spearman(x, y):
    """Spearman rho with average ranks (no scipy)."""
    def ranks(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a))
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else None


def simstyle_baseline(pool, models):
    """Legacy-style next-GW prediction: model + FDR post-multiplier."""
    out = {}
    per_pos = defaultdict(lambda: {"X": [], "who": []})
    for p in pool:
        per_pos[p["pos"]]["X"].append(
            [float(p.get(f, 0.0)) for f in sim.FEAT_COLS])
        per_pos[p["pos"]]["who"].append(p)
    for pos, b in per_pos.items():
        model = models.get(pos)
        X = np.array(b["X"], dtype=float)
        preds = (np.maximum(np.asarray(model.predict(X), dtype=float), 0.0)
                 if model is not None else
                 np.array([p.get("avg_points_per_game", 2.0) for p in b["who"]]))
        for p, pr in zip(b["who"], preds):
            fdr = p.get("fdr", 3.0)
            mult = sim.FDR_MULT_DEF if pos in ("GK", "DEF") else sim.FDR_MULT
            pr *= max(0.5, 1.0 - mult * (fdr - 3.0))
            out[p["player_id"]] = 0.0 if p.get("zero_minutes", False) else float(pr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--end", type=int, default=38)
    ap.add_argument("--offsets", type=int, default=6)
    ap.add_argument("--out", default="data/intel/phase1_calibration.json")
    args = ap.parse_args()

    assert pm.DEFAULT_FEAT_COLS == sim.FEAT_COLS, \
        "prediction_matrix feature order diverged from season_simulator"

    print("[LOAD] data...")
    hist_lookup = sim.load_player_history()
    players_df  = sim.load_players_raw()
    fdr_lookup, home_lookup, dgw_gws, gw_teams = sim.load_fixtures()
    train_dfs   = sim.load_training_data()
    avail_gws   = sim.load_availability()
    fixture_list = pm.load_fixture_list(sim.FIXTURES_CSV)
    hist_team_form = sim.build_hist_team_form_lookup()
    hist_rows   = sim.build_hist_rows(train_dfs, hist_team_form)
    print(f"[LOAD] DGWs: {sorted(g for g in dgw_gws if g <= 38)}")

    # accumulators keyed by offset h
    acc = defaultdict(lambda: {"err_played": [], "err_top": [],
                               "mu": [], "act": [], "blank_viol": 0,
                               "n_blank": 0, "n_dgw": 0,
                               "q90_cov": [], "phi_err": defaultdict(list)})
    base_err_played, base_err_top = [], []

    for t in range(args.start, args.end + 1):
        team_form = sim.build_team_form_lookup(hist_lookup, players_df, t - 1)
        opp = sim.build_opponent_lookup(sim.FIXTURES_CSV, team_form)
        pool = sim.build_rolling_pool(players_df, hist_lookup, fdr_lookup,
                                      home_lookup, t - 1,
                                      team_form_lookup=team_form,
                                      opp_lookup=opp)
        cs_weight = 1 + (t - 1)              # sim: CURRENT_SEASON_BASE_WEIGHT+gw
        retrain = sim.build_retrain_rows(players_df, hist_lookup, fdr_lookup,
                                         home_lookup, t - 1,
                                         team_form_lookup=team_form,
                                         opp_lookup=opp)
        rows, weights = {}, {}
        for pos in ("GK", "DEF", "MID", "FWD"):
            h, r = hist_rows.get(pos, []), retrain.get(pos, [])
            rows[pos] = h + r
            weights[pos] = [1.0] * len(h) + [float(cs_weight)] * len(r)
        models = sim.train_models(rows, weights)

        matrix = pm.build_matrix(pool, models, fixture_list, t, args.offsets,
                                 hist_lookup=hist_lookup, avail_gws=avail_gws)
        base = simstyle_baseline(pool, models)

        for g, rows_g in matrix.items():
            h = g - t
            a = acc[h]
            scored = []
            for pid, r in rows_g.items():
                act_row = hist_lookup.get(pid, {}).get(g)
                actual = float(act_row["total_points"]) if act_row else 0.0
                played = bool(act_row and act_row.get("minutes", 0) > 0)
                if r["n_fix"] == 0:
                    a["n_blank"] += 1
                    if r["mu"] != 0.0:
                        a["blank_viol"] += 1
                if r["n_fix"] >= 2:
                    a["n_dgw"] += 1
                if r["n_fix"] >= 1:
                    scored.append((pid, r, actual, played))
            a["mu"].extend(x[1]["mu"] for x in scored)
            a["act"].extend(x[2] for x in scored)
            for pid, r, actual, played in scored:
                if played:
                    a["err_played"].append(abs(r["mu"] - actual))
                    if h <= Q90_MAX_OFFSET:
                        a["q90_cov"].append(1.0 if actual <= r["q90"] else 0.0)
                    if h <= 1:
                        for lo, hi in PHI_BUCKETS:
                            if lo <= r["phi"] < hi:
                                a["phi_err"][f"{lo:.2f}-{hi:.2f}"].append(
                                    abs(r["mu"] - actual))
            for pid, r, actual, played in sorted(
                    scored, key=lambda x: -x[1]["mu"])[:TOPN]:
                a["err_top"].append(abs(r["mu"] - actual))
                if h == 0:
                    base_err_top.append(abs(base.get(pid, 0.0) - actual))
            if h == 0:
                for pid, r, actual, played in scored:
                    if played:
                        base_err_played.append(abs(base.get(pid, 0.0) - actual))
        print(f"  GW{t}: done ({len(matrix)} offsets)")

    # ---- report -------------------------------------------------------------
    result = {"start": args.start, "end": args.end, "topn": TOPN, "offsets": {}}
    print("\n" + "=" * 76)
    print("  PHASE 1 CALIBRATION — MAE by horizon distance")
    print("=" * 76)
    print(f"  {'h':>2} {'n_played':>9} {'MAE_played':>11} {'MAE_top60':>10} "
          f"{'spearman':>9} {'blank_viol':>10} {'n_dgw':>6}")
    mae0 = None
    for h in sorted(acc):
        a = acc[h]
        mae_p = statistics.mean(a["err_played"]) if a["err_played"] else None
        mae_t = statistics.mean(a["err_top"]) if a["err_top"] else None
        rho = spearman(a["mu"], a["act"])
        if h == 0:
            mae0 = mae_p
        result["offsets"][h] = {
            "n_played": len(a["err_played"]),
            "mae_played": round(mae_p, 4) if mae_p else None,
            "mae_top60": round(mae_t, 4) if mae_t else None,
            "spearman": round(rho, 4) if rho is not None else None,
            "mae_ratio_vs_h0": round(mae_p / mae0, 4) if mae_p and mae0 else None,
            "blank_violations": a["blank_viol"],
            "n_blank_cells": a["n_blank"], "n_dgw_cells": a["n_dgw"],
            "q90_coverage": round(statistics.mean(a["q90_cov"]), 4)
                            if a["q90_cov"] else None,
            "phi_bucket_mae": {k: round(statistics.mean(v), 4)
                               for k, v in sorted(a["phi_err"].items()) if v},
        }
        print(f"  {h:>2} {len(a['err_played']):>9} "
              f"{mae_p:>11.3f} {mae_t:>10.3f} "
              f"{(rho if rho is not None else float('nan')):>9.3f} "
              f"{a['blank_viol']:>10} {a['n_dgw']:>6}")

    base_mae_p = statistics.mean(base_err_played) if base_err_played else None
    base_mae_t = statistics.mean(base_err_top) if base_err_top else None
    result["simstyle_baseline_h0"] = {
        "mae_played": round(base_mae_p, 4) if base_mae_p else None,
        "mae_top60": round(base_mae_t, 4) if base_mae_t else None,
    }
    print("-" * 76)
    print(f"  sim-style baseline (h=0): MAE_played {base_mae_p:.3f} | "
          f"MAE_top60 {base_mae_t:.3f}")
    print(f"  matrix h=0 vs baseline:   "
          f"{'PASS' if result['offsets'][0]['mae_played'] <= round(base_mae_p, 4) + 0.05 else 'CHECK'}"
          f" (exit gate: matrix must not be materially worse)")
    q0 = result["offsets"][0].get("q90_coverage")
    print(f"  q90 coverage (h<=2 pooled per offset): h0={q0} (target ~0.90)")
    print(f"  phi buckets (h<=1): "
          f"{result['offsets'][0]['phi_bucket_mae']} — gate: low-phi MAE > high-phi MAE")
    print("=" * 76)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
