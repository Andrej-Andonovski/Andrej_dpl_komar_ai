"""
pipeline/minutes_model.py — learned play-probability (π) for the mp matrix.

Replaces the prediction-matrix heuristic (pi_base = appearances/5,
rot = starts/5 + 0.2) with a LightGBM classifier over three outcomes:

    0 = no minutes    1 = cameo (1-59')    2 = start (60'+)

Trained ONLINE from the season's own actuals (hist_lookup), exactly like
the points models: at gameweek t it fits on every (player, gw) row with
gw < t, features built strictly from that row's prior gameweeks — no
leakage, no external data. Early season (< MIN_TRAIN_GW) there is too
little signal, `ready` stays False and the caller keeps the heuristic.

Outputs per player:  p_play = P(minutes > 0),  p_start = P(60'+),
consumed by prediction_matrix as pi_base / rot replacements (the intel
availability blend on top is unchanged).
"""

import numpy as np

MIN_TRAIN_GW = 7        # first GW with enough rows (~6 * pool) to fit
MIN_ROWS     = 1500

FEATURES = [
    "played_last", "start_last", "mins_last",
    "played_rate3", "start_rate3", "mins_avg3",
    "played_rate5", "start_rate5", "mins_avg5",
    "mins_trend", "gap_since_played", "start_streak",
    "season_start_rate", "n_prior",
    "intel_avail", "intel_rot",     # intel_03 / intel_04 (2025-26 only;
]                                   # neutral defaults elsewhere)

INTEL_AVAIL_DEFAULT = 95.0
INTEL_ROT_DEFAULT = 30.0


def _row_features(ph, g, intel_row=None):
    """Features for target gameweek g from GWs strictly before g.

    ph = {gw: {"minutes": int, ...}}; a missing GW = did not play.
    intel_row = (availability_pct, rotation_risk) scraped BEFORE g's
    deadline, or None -> neutral defaults.
    Returns None when there is no prior season data at all.
    """
    prior = list(range(1, g))
    if not prior:
        return None
    mins = [ph.get(w, {}).get("minutes", 0) for w in prior]
    last3, last5 = mins[-3:], mins[-5:]

    played = [m > 0 for m in mins]
    starts = [m >= 60 for m in mins]

    gap = 0
    for m in reversed(mins):
        if m > 0:
            break
        gap += 1
    streak = 0
    for m in reversed(mins):
        if m < 60:
            break
        streak += 1

    avail, rot = intel_row if intel_row else (INTEL_AVAIL_DEFAULT,
                                              INTEL_ROT_DEFAULT)
    return [
        1.0 if mins[-1] > 0 else 0.0,
        1.0 if mins[-1] >= 60 else 0.0,
        float(mins[-1]),
        float(np.mean([m > 0 for m in last3])),
        float(np.mean([m >= 60 for m in last3])),
        float(np.mean(last3)),
        float(np.mean([m > 0 for m in last5])),
        float(np.mean([m >= 60 for m in last5])),
        float(np.mean(last5)),
        float(np.mean(last3) - np.mean(last5)),
        float(gap),
        float(streak),
        float(np.mean(starts)),
        float(len(prior)),
        float(avail),
        float(rot),
    ]


def _label(minutes):
    return 2 if minutes >= 60 else (1 if minutes > 0 else 0)


class MinutesModel:
    """Online-retrained 3-class minutes classifier."""

    def __init__(self):
        self.model = None
        self.ready = False

    def fit(self, hist_lookup, t, intel=None):
        """Train on all (pid, gw) rows with 2 <= gw < t.

        intel = {gw: {pid: (availability_pct, rotation_risk)}} scraped
        pre-deadline per GW (intel_03/intel_04), or None.
        """
        self.ready = False
        if t < MIN_TRAIN_GW:
            return self
        intel = intel or {}
        X, y = [], []
        for pid, ph in hist_lookup.items():
            for g in range(2, t):
                feats = _row_features(ph, g, intel.get(g, {}).get(pid))
                if feats is None:
                    continue
                X.append(feats)
                y.append(_label(ph.get(g, {}).get("minutes", 0)))
        if len(X) < MIN_ROWS or len(set(y)) < 3:
            return self
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3,
            n_estimators=120, max_depth=4, num_leaves=15,
            learning_rate=0.08, min_child_samples=40,
            subsample=0.9, subsample_freq=1,
            verbosity=-1, random_state=42)
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y))
        self.ready = True
        return self

    def predict(self, hist_lookup, pids, t, intel=None):
        """{pid: (p_play, p_start)} for gameweek t; {} when not ready."""
        if not self.ready:
            return {}
        intel_t = (intel or {}).get(t, {})
        feats, keep = [], []
        for pid in pids:
            f = _row_features(hist_lookup.get(pid, {}), t, intel_t.get(pid))
            if f is None:
                continue
            feats.append(f)
            keep.append(pid)
        if not feats:
            return {}
        proba = self.model.predict_proba(np.asarray(feats, dtype=float))
        out = {}
        for pid, pr in zip(keep, proba):
            p_none, p_cameo, p_start = float(pr[0]), float(pr[1]), float(pr[2])
            out[pid] = (1.0 - p_none, p_start)
        return out
