"""
tests/test_minutes_model.py — learned play-probability model + matrix hook.

Run:  docker run --rm -v "<repo>:/app" -w /app fpl-sim \
        python tests/test_minutes_model.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minutes_model as mm
import prediction_matrix as pm
from test_prediction_matrix import MODELS, make_player, hist_played


def ph(minutes_seq):
    """{gw: {"minutes": m, "total_points": 2}} from a GW1.. sequence."""
    return {g + 1: {"minutes": m, "total_points": 2}
            for g, m in enumerate(minutes_seq)}


def synth_history(n_each=60, gws=12):
    """Population of always-starters, rotation players and never-players."""
    hist = {}
    pid = 1
    for _ in range(n_each):
        hist[pid] = ph([90] * gws); pid += 1
        hist[pid] = ph([90, 0, 60, 0, 20, 90, 0, 60, 0, 20, 90, 0][:gws]); pid += 1
        hist[pid] = ph([0] * gws); pid += 1
    return hist


def test_features_capture_recency():
    f = dict(zip(mm.FEATURES, mm._row_features(ph([90, 90, 0, 0, 0]), 6)))
    assert f["played_last"] == 0 and f["gap_since_played"] == 3
    assert f["start_rate5"] == 0.4 and f["mins_last"] == 0
    f2 = dict(zip(mm.FEATURES, mm._row_features(ph([0, 0, 90, 90, 90]), 6)))
    assert f2["start_streak"] == 3 and f2["played_last"] == 1


def test_not_ready_early_or_thin():
    m = mm.MinutesModel().fit({1: ph([90] * 3)}, 4)     # before MIN_TRAIN_GW
    assert not m.ready and m.predict({}, [1], 4) == {}
    m = mm.MinutesModel().fit({1: ph([90] * 10)}, 10)   # one player: too thin
    assert not m.ready


def test_learns_starter_vs_ghost():
    hist = synth_history()
    m = mm.MinutesModel().fit(hist, 12)
    assert m.ready
    preds = m.predict(hist, [1, 3], 12)     # pid 1 starter, pid 3 ghost
    p_play_starter, p_start_starter = preds[1]
    p_play_ghost, _ = preds[3]
    assert p_play_starter > 0.9 and p_start_starter > 0.85
    assert p_play_ghost < 0.1


def test_matrix_pi_override_applies():
    fl = {(1, 10): [{"fdr": 2, "is_home": 1}]}
    hist = {1: hist_played(10)}
    base = pm.build_matrix([make_player()], MODELS, fl, t=10, horizon=1,
                           hist_lookup=hist)
    over = pm.build_matrix([make_player()], MODELS, fl, t=10, horizon=1,
                           hist_lookup=hist, pi_overrides={1: (0.11, 0.05)})
    assert abs(base[10][1]["pi"] - 0.11) > 0.2      # heuristic was high
    assert abs(over[10][1]["pi"] - 0.11) < 1e-6     # override took
    # mu and q90 untouched by the override
    assert over[10][1]["mu"] == base[10][1]["mu"]
    assert over[10][1]["q90"] == base[10][1]["q90"]


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
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
