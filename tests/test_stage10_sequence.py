"""
tests/test_stage10_sequence.py — Stage 10 Phase 1, step 2.

Leakage + cross-season-bleed + provenance asserts for the sequence builder.
Plain python, no pytest.

Run:  python tests/test_stage10_sequence.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
import stage10_sequence as sq   # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OOF_CSV = os.path.join(_ROOT, "models", "stage10", "oof_preds.csv")

# Two seasons is enough coverage and keeps the run fast.
TEST_SEASONS = ["2023-24", "2024-25"]

_JOINED = sq.load_joined()
_SAMPLES = sq.build_samples(TEST_SEASONS, joined=_JOINED)
_OOF = pd.read_csv(OOF_CSV)
_OOF["season"] = _OOF["season"].astype(str)


def test_samples_exist():
    assert len(_SAMPLES) > 3000, f"only {len(_SAMPLES)} samples built"


def test_gw1_and_min_history_excluded():
    for s in _SAMPLES:
        assert s.target_gw >= 2, f"{s.name} {s.season} target GW {s.target_gw} < 2"
        assert s.length >= sq.DEFAULT_MIN_HISTORY, \
            f"{s.name} {s.season} GW{s.target_gw}: length {s.length}"


def test_no_future_timestep():
    """Every timestep source GW is strictly before the target GW."""
    for s in _SAMPLES:
        assert len(s.ts_gws) == s.length
        assert all(g < s.target_gw for g in s.ts_gws), \
            f"{s.name} {s.season} GW{s.target_gw}: ts_gws={s.ts_gws}"
        assert s.ts_gws == sorted(s.ts_gws), "timesteps not time-ordered"


def test_no_cross_season_bleed():
    """A player who appears in both test seasons: the 2024-25 sample's timeline
    length can never exceed the count of that player's 2024-25 base rows before t."""
    by_name = {}
    for s in _SAMPLES:
        by_name.setdefault(s.name, []).append(s)
    multi = [n for n, ss in by_name.items()
             if {x.season for x in ss} >= {"2023-24", "2024-25"}]
    assert multi, "no player spans both test seasons — widen TEST_SEASONS"
    checked = 0
    for name in multi:
        for s in [x for x in by_name[name] if x.season == "2024-25"]:
            n_rows = len(_JOINED[(_JOINED.name == name)
                                 & (_JOINED.season == "2024-25")
                                 & (_JOINED.GW < s.target_gw)])
            assert s.length == min(n_rows, 38), \
                f"{name} 2024-25 GW{s.target_gw}: length {s.length} vs {n_rows} in-season rows"
            checked += 1
    assert checked > 5


def test_mutating_target_actual_only_changes_y():
    """Changing actual[t] must change ONLY y_resid — never ts_feats or query."""
    s0 = next(x for x in _SAMPLES if x.season == "2024-25"
             and x.length >= 5 and x.target_gw >= 10)
    j2 = _JOINED.copy()
    mask = ((j2.name == s0.name) & (j2.season == s0.season)
            & (j2.GW == s0.target_gw))
    assert mask.sum() == 1
    for col in ("total_points",):
        j2.loc[mask, col] = j2.loc[mask, col] + 50.0
    s1 = next(x for x in sq.build_samples([s0.season], joined=j2)
              if x.name == s0.name and x.target_gw == s0.target_gw)
    assert np.array_equal(s0.ts_feats, s1.ts_feats), "ts_feats moved when actual[t] changed"
    assert np.array_equal(s0.query, s1.query), "query moved when actual[t] changed"
    assert abs((s1.actual_t - s0.actual_t) - 50.0) < 1e-6
    assert abs((s0.y_resid - s1.y_resid) + 50.0) < 1e-6, "y_resid did not track actual[t]"


def test_mu_oof_provenance():
    """Each timestep's mu_gbm_oof / residual match models/stage10/oof_preds.csv
    exactly — the walk-forward target, trained only on data before that GW."""
    oidx = {(r.name, r.season, int(r.GW)): (r.mu_gbm_oof, r.actual)
            for r in _OOF.itertuples(index=False)}
    mu_col = sq.TS_COLS.index("mu_gbm_oof")
    res_col = sq.TS_COLS.index("residual")
    o_pts_col = sq.TS_COLS.index("o_points")
    checked = 0
    for s in _SAMPLES[::7]:                     # every 7th sample, plenty
        for k, g in enumerate(s.ts_gws):
            key = (s.name, s.season, g)
            if key not in oidx:
                continue                        # only 2019-20 lacks OOF (never a target/timestep season)
            mu_ref, act_ref = oidx[key]
            assert abs(s.ts_feats[k, mu_col] - mu_ref) < 1e-6, \
                f"{key}: ts mu {s.ts_feats[k, mu_col]} vs oof {mu_ref}"
            assert abs(s.ts_feats[k, res_col] - (act_ref - mu_ref)) < 1e-6
            assert abs(s.ts_feats[k, o_pts_col] - act_ref) < 1e-6
            checked += 1
    assert checked > 500, f"only {checked} timesteps cross-checked"


def test_query_is_prekickoff_snapshot():
    """query[:27] == the base_gw_table FEAT_COLS row for (name, season, t):
    the pre-kickoff snapshot Stage 6 already leakage-checked. No outcome of t."""
    checked = 0
    for s in _SAMPLES[::11]:
        row = _JOINED[(_JOINED.name == s.name) & (_JOINED.season == s.season)
                      & (_JOINED.GW == s.target_gw)]
        assert len(row) == 1
        ref = row[sq.FEAT_COLS].to_numpy(dtype=float)[0]
        assert np.allclose(s.query[:len(sq.FEAT_COLS)], ref, atol=1e-6), \
            f"{s.name} {s.season} GW{s.target_gw}: query snapshot != base row"
        checked += 1
    assert checked > 200


def test_mu_gbm_t_matches_oof():
    oidx = {(r.name, r.season, int(r.GW)): r.mu_gbm_oof
            for r in _OOF.itertuples(index=False)}
    mu_t_col = sq.QUERY_COLS.index("mu_gbm_t")
    for s in _SAMPLES[::13]:
        ref = oidx.get((s.name, s.season, s.target_gw))
        assert ref is not None, f"{s.name} {s.season} GW{s.target_gw} has no OOF target"
        assert abs(s.mu_gbm_t - ref) < 1e-6
        assert abs(s.query[mu_t_col] - ref) < 1e-6
        assert abs((s.actual_t - s.mu_gbm_t) - s.y_resid) < 1e-6


def test_shapes_and_no_nan():
    assert sq.F == len(sq.TS_COLS) == len(sq.SAMPLE_SPEC["ts_cols"])
    assert sq.Q == len(sq.QUERY_COLS) == len(sq.SAMPLE_SPEC["query_cols"])
    for s in _SAMPLES:
        assert s.ts_feats.shape == (s.length, sq.F)
        assert s.query.shape == (sq.Q,)
        assert np.isfinite(s.ts_feats).all(), f"non-finite ts_feats {s.name} {s.season} GW{s.target_gw}"
        assert np.isfinite(s.query).all(), f"non-finite query {s.name} {s.season} GW{s.target_gw}"
        assert np.isfinite(s.y_resid), "y_resid NaN with require_oof_target=True"


def test_padded_arrays_roundtrip():
    ts, q, ln, pid, y, meta = sq.to_padded_arrays(_SAMPLES[:500])
    assert ts.shape[0] == 500 and ts.shape[2] == sq.F
    assert q.shape == (500, sq.Q)
    assert (ln >= 1).all() and (ln <= ts.shape[1]).all()
    # right-pad: row k, the first ln[k] rows are the real sequence, rest zero
    for k in range(0, 500, 50):
        assert np.array_equal(ts[k, :ln[k]], _SAMPLES[k].ts_feats[-ln[k]:].astype(np.float32))
        if ln[k] < ts.shape[1]:
            assert np.all(ts[k, ln[k]:] == 0.0)
    assert np.isfinite(ts).all() and np.isfinite(q).all()


def test_determinism():
    s2 = sq.build_samples(["2024-25"], joined=_JOINED)
    s1 = [x for x in _SAMPLES if x.season == "2024-25"]
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        assert a.name == b.name and a.target_gw == b.target_gw
        assert np.array_equal(a.ts_feats, b.ts_feats)
        assert np.array_equal(a.query, b.query)
        assert a.y_resid == b.y_resid


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
        except Exception as e:                  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed  "
          f"({len(_SAMPLES):,} samples over {TEST_SEASONS})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
