"""
pipeline/stage10_refine.py
Stage 10 Phase 1 — step 4: the runtime hook.

season_simulator calls refine() ONLY when STAGE10=="on". This module (and its
whole import chain: stage10_infer -> stage10_model, stage10_sequence ->
prediction_matrix) is numpy-only — torch is never imported at runtime.

refine() reconstructs each player's GW-sequence from the live hist_lookup,
runs the numpy LSTM, and returns an additive correction to the RAW GBM mu
plus the calibrated q90. FDR / intel / cap / loyalty stay downstream, exactly
as today (docs/stage10_phase1_plan.md §5).

Runtime approximations vs the offline training sequences (tightened in step 6):
  * per-timestep FEAT_COLS snapshot: the 8 rolling features are recomputed
    as-of GW g from hist_lookup; the slow features (value / prev_* / career_*)
    use the current-GW pool value (they barely move within a season)
  * mu_gbm_oof[g] / residual[g]: from past_mu_history (the model-as-of-g's own
    raw prediction, accumulated by the sim each GW — walk-forward correct);
    falls back to 0 for GWs before the sim's start
  * o_gc / o_bps timesteps: hist_lookup lacks them -> 0 (2 of 56 features)
  * is_dgw / is_blank / intel timesteps: 0 (as in offline training)
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stage10_sequence as sq          # noqa: E402  (numpy/pandas only)
import stage10_infer as si             # noqa: E402  (numpy only)
import stage10_model as sm             # noqa: E402  (numpy only)

FEAT_COLS = list(sq.FEAT_COLS)
ROLL_COLS = FEAT_COLS[:8]              # form_last3..saves_per_game — recomputed as-of g
MAX_LEN = 38


def _rolling_asof(ph, g):
    """The 8 rolling FEAT_COLS as they would have been known before GW g,
    from live actuals ph = {gw: {...}} — mirrors build_rolling_pool()."""
    pts, mins, gl, ast, cs, sv = [], [], [], [], [], []
    for k in range(1, g):
        h = ph.get(k)
        if h is None:
            continue
        pts.append(h.get("total_points", 0.0)); mins.append(h.get("minutes", 0))
        gl.append(h.get("goals_scored", 0)); ast.append(h.get("assists", 0))
        cs.append(h.get("clean_sheets", 0)); sv.append(h.get("saves", 0))
    played = [i for i in range(len(pts)) if mins[i] > 0]
    pj = lambda a: [a[i] for i in played]                       # noqa: E731
    tot_mins = float(sum(mins))
    denom = max(1, g - 1) * 90.0
    return {
        "form_last3": float(np.mean(pts[-3:])) if pts else 0.0,
        "form_last5": float(np.mean(pts[-5:])) if pts else 0.0,
        "avg_points_per_game": float(np.mean(pj(pts))) if played else 0.0,
        "minutes_reliability": tot_mins / denom,
        "goals_per_game": float(np.mean(pj(gl))) if played else 0.0,
        "assists_per_game": float(np.mean(pj(ast))) if played else 0.0,
        "clean_sheet_rate": float(np.mean(pj(cs))) if played else 0.0,
        "saves_per_game": float(np.mean(pj(sv))) if played else 0.0,
    }


def _timestep(p, g, ph, mu_g, fdr_g, home_g):
    """One [F] timestep row for player p at GW g (TS_COLS order)."""
    base = {c: float(p.get(c, 0.0)) for c in FEAT_COLS}
    base.update(_rolling_asof(ph, g))
    h = ph.get(g, {})
    mins = h.get("minutes", 0)
    played = 1.0 if mins > 0 else 0.0
    resid = (h.get("total_points", 0.0) - mu_g) if mu_g is not None else 0.0
    row = [base[c] for c in FEAT_COLS] + [
        float(mu_g or 0.0), float(resid),
        float(h.get("total_points", 0.0)), float(mins),
        float(h.get("goals_scored", 0)), float(h.get("assists", 0)),
        float(h.get("clean_sheets", 0)), 0.0,                   # o_gc unavailable
        float(h.get("bonus", 0)), 0.0,                          # o_bps unavailable
        float(h.get("saves", 0)),
        float(home_g), float(fdr_g), 0.0, 0.0,                  # is_dgw/is_blank
        played, 0.0, 1.0 - played, 0.0, 0.0,                    # gap/first/streak filled below
        0.0, 0.0,                                               # roll_std filled below
        float(sq.AVAIL_TIER_DEFAULT), 0.0, 0.0,                 # intel
        1.0 if p["pos"] == "GK" else 0.0, 1.0 if p["pos"] == "DEF" else 0.0,
        1.0 if p["pos"] == "MID" else 0.0, 1.0 if p["pos"] == "FWD" else 0.0,
    ]
    return row


def build_runtime_arrays(pool, gw, hist_lookup, past_mu_history,
                         mu_raw_by_pid, fdr_lookup, home_lookup):
    """Returns (ts, query, lengths, pos_ids, n_played, positions, mu_gbm, pids)."""
    idx = {c: i for i, c in enumerate(sq.TS_COLS)}
    gap_i, first_i, streak_i = idx["gap_since_played"], idx["is_first_appearance"], idx["streak_len"]
    rp_i, rm_i = idx["roll_std_points5"], idx["roll_std_minutes5"]
    o_pts_i, o_min_i = idx["o_points"], idx["o_minutes"]

    seqs, queries, lengths, pos_ids, n_played, positions, mu_gbm, pids = \
        [], [], [], [], [], [], [], []
    for p in pool:
        pid = p["player_id"]
        ph = hist_lookup.get(pid, {})
        gws = sorted(k for k in ph.keys() if 1 <= k < gw)
        gws = gws[-MAX_LEN:]
        rows = []
        for g in gws:
            team = p.get("team")
            mu_g = past_mu_history.get(g, {}).get(pid)
            fdr_g = float(fdr_lookup.get((team, g), 3.0))
            home_g = float(home_lookup.get((team, g), 0))
            rows.append(_timestep(p, g, ph, mu_g, fdr_g, home_g))
        L = len(rows)
        if L:
            arr = np.array(rows, dtype=np.float32)
            # continuity + volatility over the built rows
            last_played = None
            run = 0
            for i, g in enumerate(gws):
                arr[i, gap_i] = 0.0 if last_played is None else min(sq.GAP_CAP, g - last_played)
                arr[i, streak_i] = min(sq.STREAK_CAP, run)
                if arr[i, idx["played"]] > 0:
                    run += 1; last_played = g
                else:
                    run = 0
                w0 = max(0, i - sq.ROLL_N + 1)
                wp, wm = arr[w0:i + 1, o_pts_i], arr[w0:i + 1, o_min_i]
                arr[i, rp_i] = float(np.std(wp)) if len(wp) > 1 else 0.0
                arr[i, rm_i] = float(np.std(wm)) if len(wm) > 1 else 0.0
            arr[0, first_i] = 1.0
        else:
            arr = np.zeros((0, sq.F), dtype=np.float32)

        # query row for GW gw
        q = np.zeros(sq.Q, dtype=np.float32)
        for j, c in enumerate(FEAT_COLS):
            q[j] = float(p.get(c, 0.0))
        team = p.get("team")
        npl = sum(1 for g in gws if ph.get(g, {}).get("minutes", 0) > 0)
        tail = [
            float(mu_raw_by_pid.get(pid, 0.0)),
            float(home_lookup.get((team, gw), 0)), float(fdr_lookup.get((team, gw), 3.0)),
            0.0, 0.0, float(npl), float(L), float(gw),
            1.0 if p["pos"] == "GK" else 0.0, 1.0 if p["pos"] == "DEF" else 0.0,
            1.0 if p["pos"] == "MID" else 0.0, 1.0 if p["pos"] == "FWD" else 0.0,
        ]
        q[len(FEAT_COLS):] = tail

        seqs.append(arr); queries.append(q); lengths.append(L)
        pos_ids.append(sq.POS_ID.get(p["pos"], 2)); n_played.append(npl)
        positions.append(p["pos"]); mu_gbm.append(float(mu_raw_by_pid.get(pid, 0.0)))
        pids.append(pid)

    W = max((a.shape[0] for a in seqs), default=1)
    ts = np.zeros((len(seqs), W, sq.F), dtype=np.float32)
    for k, a in enumerate(seqs):
        if a.shape[0]:
            ts[k, :a.shape[0]] = a
    return (ts, np.array(queries, dtype=np.float32), np.array(lengths),
            np.array(pos_ids), np.array(n_played), positions,
            np.array(mu_gbm, dtype=float), pids)


def refine(pool, gw, season, hist_lookup, past_mu_history, mu_raw_by_pid,
           fdr_lookup, home_lookup, live=True):
    """
    Returns {pid: {"r": r_applied, "sigma": sigma_eff, "q90": q90}}.
    Empty dict if the checkpoint for `season` isn't present (STAGE10=on then
    degrades to exactly STAGE10=off).
    """
    if gw <= 1:
        return {}                          # GW1 is the blind test — no sequence
    refiner = si.load_refiner(season, live=live, gw=gw)
    if not refiner.ready:
        print(f"  [STAGE10] no checkpoint for season {season} — refine is a no-op")
        return {}
    ts, query, lengths, pos_ids, n_played, positions, mu_gbm, pids = \
        build_runtime_arrays(pool, gw, hist_lookup, past_mu_history,
                             mu_raw_by_pid, fdr_lookup, home_lookup)
    out = refiner.predict_arrays(ts, query, lengths, pos_ids, n_played,
                                 positions, mu_gbm)
    res = {}
    for i, pid in enumerate(pids):
        res[pid] = {"r": float(out["r_applied"][i]),
                    "sigma": float(out["sigma_eff"][i]),
                    "q90": float(out["q90"][i])}
    npos = int((out["r_applied"] > 0.05).sum())
    nneg = int((out["r_applied"] < -0.05).sum())
    print(f"  [STAGE10] gw{gw}: refined {len(pids)} players  "
          f"(+{npos} up / -{nneg} down, |r| mean {np.abs(out['r_applied']).mean():.2f})")
    return res
