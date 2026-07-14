"""
tests/test_prediction_matrix.py — Phase 1 matrix unit tests (stub models,
no lightgbm needed; numpy only).

Run:  docker run --rm -v "<repo>:/app" -w /app fpl-sim \
        python tests/test_prediction_matrix.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
import prediction_matrix as pm

FC = pm.DEFAULT_FEAT_COLS
FDR = FC.index("fdr")
HOME = FC.index("was_home")


class StubModel:
    """pred = 10 - fdr + 0.5*was_home — fixture features fully determine it."""
    def predict(self, X):
        return 10.0 - X[:, FDR] + 0.5 * X[:, HOME]


class NegativeModel:
    def predict(self, X):
        return np.full(len(X), -3.0)


def make_player(pid=1, pos="MID", team=1, price=8.0, zero_minutes=False):
    p = {"player_id": pid, "pos": pos, "element_type": 3, "team": team,
         "price": price, "zero_minutes": zero_minutes}
    for f in FC:
        p[f] = 1.0
    p["minutes_reliability"] = 0.9
    return p


MODELS = {"GK": StubModel(), "DEF": StubModel(),
          "MID": StubModel(), "FWD": StubModel()}


def hist_played(t, minutes=90, pts=5):
    """Full history GW1..t-1, always played."""
    return {g: {"total_points": pts, "minutes": minutes} for g in range(1, t)}


def test_single_fixture_feature_swap():
    fl = {(1, 10): [{"fdr": 2, "is_home": 1}],
          (1, 11): [{"fdr": 5, "is_home": 0}]}
    m = pm.build_matrix([make_player()], MODELS, fl, t=10, horizon=2,
                        hist_lookup={1: hist_played(10)})
    # gw10: 10-2+0.5 = 8.5 ; gw11: 10-5+0 = 5.0
    assert abs(m[10][1]["mu"] - 8.5) < 1e-9, m[10][1]["mu"]
    assert abs(m[11][1]["mu"] - 5.0) < 1e-9, m[11][1]["mu"]
    assert m[10][1]["n_fix"] == 1 and m[11][1]["n_fix"] == 1


def test_blank_is_hard_zero():
    fl = {(1, 10): [{"fdr": 3, "is_home": 1}]}     # nothing for gw11
    m = pm.build_matrix([make_player()], MODELS, fl, t=10, horizon=2,
                        hist_lookup={1: hist_played(10)})
    r = m[11][1]
    assert r["mu"] == 0.0 and r["n_fix"] == 0 and r["pi"] == 0.0 and r["q90"] == 0.0


def test_dgw_sums_per_fixture():
    fl = {(1, 10): [{"fdr": 2, "is_home": 1}, {"fdr": 4, "is_home": 0}]}
    m = pm.build_matrix([make_player()], MODELS, fl, t=10, horizon=1,
                        hist_lookup={1: hist_played(10)})
    # (10-2+0.5) + (10-4+0) = 8.5 + 6.0 = 14.5 — NOT flat x2 of either
    assert abs(m[10][1]["mu"] - 14.5) < 1e-9, m[10][1]["mu"]
    assert m[10][1]["n_fix"] == 2


def test_negative_pred_floored_per_fixture():
    fl = {(1, 10): [{"fdr": 3, "is_home": 0}]}
    models = {k: NegativeModel() for k in MODELS}
    m = pm.build_matrix([make_player()], models, fl, t=10, horizon=1,
                        hist_lookup={1: hist_played(10)})
    assert m[10][1]["mu"] == 0.0


def test_zero_minutes_player_zeroed():
    fl = {(1, 10): [{"fdr": 3, "is_home": 1}]}
    m = pm.build_matrix([make_player(zero_minutes=True)], MODELS, fl,
                        t=10, horizon=1, hist_lookup={})
    assert m[10][1]["mu"] == 0.0 and m[10][1]["pi"] == 0.0


def test_availability_out_zeroes_near_week_only():
    fl = {(1, 10): [{"fdr": 3, "is_home": 1}],
          (1, 14): [{"fdr": 3, "is_home": 1}]}
    avail = {"10": {"players": {"1": {"availability_tier": "out"}}},
             "14": {"players": {"1": {"availability_tier": "out"}}}}
    m = pm.build_matrix([make_player()], MODELS, fl, t=10, horizon=5,
                        hist_lookup={1: hist_played(10)}, avail_gws=avail)
    assert m[10][1]["mu"] == 0.0            # w=1 at g=t: hard zero
    # g=t+4: w=0 — intel fully decayed, prediction untouched
    assert m[14][1]["mu"] > 0.0


def test_pi_bounds_and_base_rate():
    fl = {(1, 10): [{"fdr": 3, "is_home": 1}]}
    # played 3 of last 5 GWs
    ph = {g: {"total_points": 4, "minutes": (90 if g % 2 else 0)}
          for g in range(1, 10)}
    m = pm.build_matrix([make_player()], MODELS, fl, t=10, horizon=1,
                        hist_lookup={1: ph})
    assert 0.0 <= m[10][1]["pi"] <= pm.PI_MAX


def test_phi_bounds_and_ramp():
    fl = {(1, 3): [{"fdr": 3, "is_home": 1}], (1, 20): [{"fdr": 3, "is_home": 1}]}
    # early season: 2 played GWs -> low phi; late: many played -> high phi
    m_early = pm.build_matrix([make_player()], MODELS, fl, t=3, horizon=1,
                              hist_lookup={1: hist_played(3)})
    m_late = pm.build_matrix([make_player()], MODELS, fl, t=20, horizon=1,
                             hist_lookup={1: hist_played(20)})
    phi_e, phi_l = m_early[3][1]["phi"], m_late[20][1]["phi"]
    assert pm.PHI_FLOOR <= phi_e < phi_l <= 1.0, (phi_e, phi_l)


def test_phi_return_from_absence():
    fl = {(1, 20): [{"fdr": 3, "is_home": 1}]}
    # played 1..14, absent 15..18 (4 gws), returned 19
    ph = {g: {"total_points": 5, "minutes": 90} for g in range(1, 15)}
    ph[19] = {"total_points": 5, "minutes": 90}
    m = pm.build_matrix([make_player()], MODELS, fl, t=20, horizon=1,
                        hist_lookup={1: ph})
    # same player without the absence
    m2 = pm.build_matrix([make_player()], MODELS, fl, t=20, horizon=1,
                         hist_lookup={1: hist_played(20)})
    assert m[20][1]["phi"] < m2[20][1]["phi"]


def test_q90_above_mu_and_dgw_scaling():
    fl_s = {(1, 10): [{"fdr": 3, "is_home": 1}]}
    fl_d = {(1, 10): [{"fdr": 3, "is_home": 1}, {"fdr": 3, "is_home": 1}]}
    ph = {g: {"total_points": 4 + (g % 5), "minutes": 90} for g in range(1, 10)}
    ms = pm.build_matrix([make_player()], MODELS, fl_s, t=10, horizon=1,
                         hist_lookup={1: ph})
    md = pm.build_matrix([make_player()], MODELS, fl_d, t=10, horizon=1,
                         hist_lookup={1: ph})
    s, d = ms[10][1], md[10][1]
    assert s["q90"] > s["mu"]
    # DGW headroom scales with sqrt(2)
    assert abs((d["q90"] - d["mu"]) - (s["q90"] - s["mu"]) * 2 ** 0.5) < 1e-6


def test_horizon_truncates_at_max_gw():
    fl = {(1, g): [{"fdr": 3, "is_home": 1}] for g in range(36, 39)}
    m = pm.build_matrix([make_player()], MODELS, fl, t=36, horizon=6,
                        hist_lookup={1: hist_played(36)})
    assert sorted(m.keys()) == [36, 37, 38]


def test_sell_value_column():
    fl = {(1, 10): [{"fdr": 3, "is_home": 1}]}
    m = pm.build_matrix([make_player(price=8.4)], MODELS, fl, t=10, horizon=1,
                        hist_lookup={1: hist_played(10)},
                        purchase_price={1: 8.0})
    assert abs(m[10][1]["sell_value"] - 8.2) < 1e-9   # 8.0 + floor(0.4/2)


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
