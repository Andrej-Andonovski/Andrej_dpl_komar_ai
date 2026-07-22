"""
pipeline/prediction_matrix.py
Phase 1 of the optimizer redesign (docs/optimizer_redesign.md §3).

Builds the per-player, per-future-GW prediction matrix the multi-period MILP
consumes:  {gw: {player_id: {mu, n_fix, pi, phi, q90, price, sell_value}}}

Key properties (vs the legacy predict_pool + post-multipliers):
  - feature-swap re-prediction: each future GW's fdr/was_home go INTO the
    model (deletes the FDR post-multipliers)
  - per-fixture DGW sums: one model call per fixture, summed
    (deletes DGW_PRED_MULT = 2.0)
  - hard blank zeros: no fixture in GW g -> mu = 0 for GW g exactly
  - pi   = play probability (intel tiers x rotation, decaying to base rate)
  - phi  = player-specific confidence in [PHI_FLOOR, 1] (NO time decay —
    time decay belongs to the objective discount delta, §3.5)
  - q90  = captaincy ceiling: mu + z90 * sigma_p * sqrt(n_fix)

The module is deliberately importable without lightgbm/pulp — models are
passed in as {pos: obj with .predict(ndarray)} so tests can use stubs.
"""
import math
import os
from collections import defaultdict

import numpy as np

# Canonical feature order for the redesign. Must match the trainer's order —
# phase1_calibration.py asserts this equals season_simulator.FEAT_COLS.
DEFAULT_FEAT_COLS = [
    "form_last3", "form_last5", "avg_points_per_game",
    "minutes_reliability", "goals_per_game", "assists_per_game",
    "clean_sheet_rate", "saves_per_game",
    "value", "was_home", "fdr",
]

# ── Availability tiers (intel_03) ─────────────────────────────────────────────
# mu multiplier (expected-value semantics, same spirit as legacy AVAIL_MULT)
MU_TIER = {
    "out": 0.0, "suspended": 0.0, "unlikely": 0.3, "doubtful": 0.5,
    "unknown": 0.85, "probable": 0.95, "available": 1.0,
}
# play-probability by tier (blueprint §3.3)
PI_TIER = {
    "out": 0.0, "suspended": 0.0, "unlikely": 0.25, "doubtful": 0.50,
    "unknown": 0.80, "probable": 0.90, "available": 0.98,
}
INTEL_DECAY = 0.25      # per-GW decay of intel weight: w = max(0, 1 - 0.25*(g-t))

# ── Confidence phi (§3.5) ─────────────────────────────────────────────────────
PHI_FLOOR    = 0.55
SAMPLE_RAMP  = 6        # played GWs to reach full sample confidence
RETURN_GAP   = 3        # absence >= this many GWs triggers the return factor
RETURN_GWS   = 2        # ... for this many GWs after coming back
R_RETURN     = 0.6

# ── Ceiling q90 (§3.6, revised after Phase 2 diagnosis) ──────────────────────
# v1 (mu + z*sigma) assumed a symmetric spread; that hands defenders captain-
# grade ceilings their real upside doesn't support (their sigma is clean-sheet
# swing, not attacking tail) — 6 DEF armbands in the Phase 2 run. v2 uses the
# EMPIRICAL headroom: (p90 - mean) of the player's own last-N played scores,
# shrunk toward the position's average headroom. q90 = mu + headroom*sqrt(n_fix).
SIGMA_LAST_N = 10       # played GWs used for the per-player headroom
SIGMA_PRIOR_M = 5       # shrinkage prior strength
HEADROOM_PRIOR_DEFAULT = {"GK": 2.5, "DEF": 3.0, "MID": 4.5, "FWD": 5.0}

# ── GW1 community prior (blueprint §1.2 interim) ─────────────────────────────
# At t=1 the model is blind (2024-25 snapshot features). Pre-season ownership
# is the strongest available quality signal; legacy carried it as the Optuna-
# tuned OWN_BOOST_GW1. Kept as ONE documented constant until ownership becomes
# a model feature (§11). Applied at t == 1 only. Phase 2 evidence for keeping
# it: deleting it cost −98 pts across GW1-2 (docs/phase2_report.md).
OWN_PRIOR_GW1 = 0.213

MU_SANITY_MAX = 40.0    # assert-only ceiling — never clamp (§1.2)

PI_MAX = 0.98


# ── Fixture list (per-fixture, unlike load_fixtures' last-write-wins lookup) ──

def load_fixture_list(fixtures_csv):
    """
    fixtures_raw.csv -> {(team_id, gw): [{"fdr": int, "is_home": 0/1}, ...]}
    Keeps EVERY fixture of a DGW (the legacy fdr_lookup overwrites doubles).
    """
    import pandas as pd
    df = pd.read_csv(fixtures_csv)
    df = df.dropna(subset=["gameweek"])
    df["gameweek"] = df["gameweek"].astype(int)
    fl = defaultdict(list)
    for r in df.itertuples(index=False):
        gw = int(r.gameweek)
        fl[(int(r.team_h), gw)].append(
            {"fdr": int(r.team_h_difficulty), "is_home": 1})
        fl[(int(r.team_a), gw)].append(
            {"fdr": int(r.team_a_difficulty), "is_home": 0})
    return dict(fl)


# ── Per-player history stats (pi_base, rotation, phi, sigma) ─────────────────

def player_stats(ph, t, minutes_reliability_fallback=0.5):
    """
    Stats from actuals in GW 1..t-1. ph = {gw: {"total_points", "minutes",...}}
    A missing GW row means the player was not in the game data -> not played.
    """
    prior = list(range(1, t))
    played = [g for g in prior if g in ph and ph[g].get("minutes", 0) > 0]
    window5 = prior[-5:]

    if not prior:
        # GW1: no season data — fall back to snapshot minutes reliability
        base = max(0.0, min(PI_MAX, minutes_reliability_fallback))
        return {"pi_base": base, "rot": min(1.0, base + 0.2),
                "phi": PHI_FLOOR, "sigma_hat": None, "n_sigma": 0}

    if window5:
        apps   = sum(1 for g in window5 if g in ph and ph[g].get("minutes", 0) > 0)
        starts = sum(1 for g in window5 if g in ph and ph[g].get("minutes", 0) >= 60)
        mins5  = [ph[g]["minutes"] if g in ph else 0 for g in window5]
        pi_base = apps / len(window5)
        rot     = min(1.0, starts / len(window5) + 0.2)
        min_std = float(np.std(mins5))
    else:
        pi_base, rot, min_std = 0.5, 0.7, 45.0

    # phi = floor + span * (r_sample * r_minutes * r_return)
    r_sample  = min(1.0, len(played) / SAMPLE_RAMP)
    r_minutes = 1.0 - 0.5 * min(1.0, min_std / 45.0)
    r_return  = 1.0
    if played:
        # returning from an absence: short current played-streak preceded by
        # a gap of >= RETURN_GAP unplayed GWs
        streak = 0
        g = t - 1
        while g >= 1 and g in ph and ph[g].get("minutes", 0) > 0:
            streak += 1
            g -= 1
        if 0 < streak <= RETURN_GWS:
            gap = 0
            while g >= 1 and not (g in ph and ph[g].get("minutes", 0) > 0):
                gap += 1
                g -= 1
            if gap >= RETURN_GAP:
                r_return = R_RETURN
    phi = PHI_FLOOR + (1.0 - PHI_FLOOR) * (r_sample * r_minutes * r_return)

    pts = [ph[g]["total_points"] for g in played[-SIGMA_LAST_N:]]
    if len(pts) >= 3:
        # empirical captaincy headroom: 90th percentile minus mean
        sigma_hat = max(0.0, float(np.percentile(pts, 90) - np.mean(pts)))
        n_sigma = len(pts)
    else:
        sigma_hat, n_sigma = None, 0

    return {"pi_base": max(0.0, min(PI_MAX, pi_base)), "rot": rot,
            "phi": phi, "sigma_hat": sigma_hat, "n_sigma": n_sigma}


# ── Matrix builder ────────────────────────────────────────────────────────────

def build_matrix(pool, models, fixture_list, t, horizon,
                 feat_cols=None, hist_lookup=None, avail_gws=None,
                 purchase_price=None, max_gw=38, pi_overrides=None):
    """
    pool         : list of player dicts (build_rolling_pool / build_gw1_pool
                   shape: player_id, pos, team, price, zero_minutes, features)
    models       : {pos: model with .predict(ndarray) or None}
    fixture_list : load_fixture_list output
    t            : current GW (decisions being made for GW t)
    horizon      : number of GWs covered -> g in t .. min(t+horizon-1, max_gw)
    hist_lookup  : {pid: {gw: {...}}} actuals for stats (may be None at GW1)
    avail_gws    : intel_03 availability {gw_str: {"players": {pid_str: ...}}}
    purchase_price: {pid: paid} for sell values (optional)
    pi_overrides : {pid: (p_play, p_start)} from the learned minutes model
                   (pipeline/minutes_model.py) — replaces the heuristic
                   pi_base/rot for those pids; the intel blend, phi and q90
                   are untouched. Absent pids keep the heuristic.

    Returns {g: {pid: {"mu","n_fix","pi","phi","q90","price","sell_value"}}}
    """
    feat_cols = feat_cols or DEFAULT_FEAT_COLS
    hist_lookup = hist_lookup or {}
    avail_gws = avail_gws or {}
    fdr_idx  = feat_cols.index("fdr")
    home_idx = feat_cols.index("was_home")

    if purchase_price is not None:
        try:
            from pipeline.fpl_rules import sell_value as _sv
        except ImportError:
            from fpl_rules import sell_value as _sv
    else:
        _sv = None

    # per-player stats + position sigma priors
    stats = {}
    sig_by_pos = defaultdict(list)
    for p in pool:
        s = player_stats(hist_lookup.get(p["player_id"], {}), t,
                         p.get("minutes_reliability", 0.5))
        ov = (pi_overrides or {}).get(p["player_id"])
        if ov is not None:
            p_play, p_start = ov
            s = dict(s, pi_base=max(0.0, min(PI_MAX, float(p_play))),
                     rot=min(1.0, float(p_start) + 0.2))
        stats[p["player_id"]] = s
        if s["sigma_hat"] is not None:
            sig_by_pos[p["pos"]].append(s["sigma_hat"])
    sigma_prior = {pos: (float(np.mean(v)) if v
                         else HEADROOM_PRIOR_DEFAULT.get(pos, 4.0))
                   for pos in ("GK", "DEF", "MID", "FWD")
                   for v in [sig_by_pos.get(pos, [])]}

    base_feats = {p["player_id"]:
                  np.array([float(p.get(f, 0.0)) for f in feat_cols])
                  for p in pool}

    gws = list(range(t, min(t + horizon - 1, max_gw) + 1))
    matrix = {}
    for g in gws:
        w = max(0.0, 1.0 - INTEL_DECAY * (g - t))     # intel weight
        g_avail = avail_gws.get(str(g), {}).get("players", {})
        rows = {}

        # batch per position: one predict() per (pos, g)
        per_pos = defaultdict(lambda: {"X": [], "who": []})
        fix_by_pid = {}
        for p in pool:
            pid = p["player_id"]
            fixtures = fixture_list.get((p["team"], g), [])
            fix_by_pid[pid] = fixtures
            if p.get("zero_minutes", False) or not fixtures:
                continue
            for fx in fixtures:
                v = base_feats[pid].copy()
                v[fdr_idx]  = float(fx["fdr"])
                v[home_idx] = float(fx["is_home"])
                per_pos[p["pos"]]["X"].append(v)
                per_pos[p["pos"]]["who"].append(pid)

        mu_raw = defaultdict(float)
        for pos, batch in per_pos.items():
            X = np.vstack(batch["X"])
            model = models.get(pos)
            if model is not None:
                preds = np.asarray(model.predict(X), dtype=float)
            else:
                # no model (too few rows): fall back to per-game average
                preds = np.array([
                    next((q.get("avg_points_per_game", 2.0) for q in pool
                          if q["player_id"] == pid), 2.0)
                    for pid in batch["who"]], dtype=float)
            preds = np.maximum(preds, 0.0)            # per-fixture floor
            for pid, pr in zip(batch["who"], preds):
                mu_raw[pid] += float(pr)              # DGW: sum fixtures

        for p in pool:
            pid, pos = p["player_id"], p["pos"]
            s = stats[pid]
            n_fix = len(fix_by_pid[pid])
            mu = mu_raw.get(pid, 0.0)

            # availability: blend tier effect with distance-decayed weight
            # GW1 community prior (see OWN_PRIOR_GW1 note)
            if t == 1 and n_fix > 0 and mu > 0.0:
                mu += p.get("sbp", 0.0) * OWN_PRIOR_GW1

            tier = g_avail.get(str(pid), {}).get("availability_tier")
            if tier is not None and w > 0.0:
                mu *= (w * MU_TIER.get(tier, 1.0) + (1.0 - w))
                pi_intel = PI_TIER.get(tier, 0.90) * s["rot"]
            else:
                pi_intel = s["pi_base"]
            pi = w * pi_intel + (1.0 - w) * s["pi_base"]

            if n_fix == 0 or p.get("zero_minutes", False):
                mu, pi = 0.0, 0.0
            if mu > MU_SANITY_MAX:                    # assert-only, no clamp
                print(f"  [MATRIX] WARNING mu={mu:.1f} > {MU_SANITY_MAX} "
                      f"for pid={pid} g={g}")

            if s["sigma_hat"] is not None:
                headroom = ((s["n_sigma"] * s["sigma_hat"] +
                             SIGMA_PRIOR_M * sigma_prior[pos]) /
                            (s["n_sigma"] + SIGMA_PRIOR_M))
            else:
                headroom = sigma_prior[pos]
            q90 = mu + headroom * math.sqrt(n_fix) if n_fix > 0 else 0.0

            price = p.get("price", 0.0)
            rows[pid] = {
                "mu": mu, "n_fix": n_fix,
                "pi": max(0.0, min(PI_MAX, pi)),
                "phi": s["phi"], "q90": q90,
                "price": price,
                "sell_value": (_sv(purchase_price[pid], price)
                               if _sv and pid in (purchase_price or {})
                               else price),
                # identity fields for MILP constraints (Phase 2+)
                "pos": pos, "element_type": p.get("element_type", 3),
                "team": p.get("team", 0),
            }
        matrix[g] = rows
    return matrix
