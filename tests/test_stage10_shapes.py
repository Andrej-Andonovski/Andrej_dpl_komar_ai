"""
tests/test_stage10_shapes.py — Stage 10 Phase 1, numpy runtime path.

Shapes, NaN-freedom, empty-sequence handling, gate behaviour, and torch<->numpy
agreement on the shipped checkpoints (< 1e-4).

Run:  python tests/test_stage10_shapes.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
import stage10_model as sm          # noqa: E402
import stage10_sequence as sq       # noqa: E402
import stage10_infer as si          # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT = os.path.join(_ROOT, "models", "stage10", "pretrain_2024-25.npz")
HAVE_CKPT = os.path.exists(CKPT)


def _rand_npz(path):
    rng = np.random.default_rng(0)
    F, Q, H, P = sq.F, sq.Q, sm.HIDDEN, sm.POS_EMB_DIM
    r = lambda *s: (rng.standard_normal(s) * 0.1).astype(np.float32)   # noqa: E731
    w = dict(ts_mean=r(F), ts_std=np.abs(r(F)) + 1, q_mean=r(Q), q_std=np.abs(r(Q)) + 1,
             pos_emb=r(4, P), qmlp_w=r(sm.MLP_HIDDEN, Q + P), qmlp_b=r(sm.MLP_HIDDEN),
             fuse_w=r(sm.MLP_HIDDEN, H + sm.MLP_HIDDEN), fuse_b=r(sm.MLP_HIDDEN),
             head_r_w=r(1, sm.MLP_HIDDEN), head_r_b=r(1),
             head_lv_w=r(1, sm.MLP_HIDDEN), head_lv_b=r(1),
             meta=np.array({"F": F, "Q": Q, "hidden": H, "layers": sm.LSTM_LAYERS,
                            "pos_emb_dim": P, "z_cal": {"GK": 1.4, "DEF": 1.4,
                                                        "MID": 1.6, "FWD": 1.5}},
                           dtype=object))
    for k in range(sm.LSTM_LAYERS):
        ink = (F + P) if k == 0 else H
        w[f"lstm_w_ih_l{k}"] = r(4 * H, ink); w[f"lstm_w_hh_l{k}"] = r(4 * H, H)
        w[f"lstm_b_ih_l{k}"] = r(4 * H); w[f"lstm_b_hh_l{k}"] = r(4 * H)
    np.savez(path, **w)


_TMP = os.path.join(_ROOT, "models", "stage10", "_test_rand.npz")
_rand_npz(_TMP)
NET = sm.NumpyResidualLSTM(CKPT if HAVE_CKPT else _TMP)


def _batch(n=40, maxlen=12):
    rng = np.random.default_rng(1)
    lengths = rng.integers(0, maxlen, n)          # includes 0
    L = max(1, lengths.max())
    ts = (rng.standard_normal((n, L, sq.F)) * 2).astype(np.float32)
    for i, ln in enumerate(lengths):
        ts[i, ln:] = 0.0
    query = (rng.standard_normal((n, sq.Q)) * 2).astype(np.float32)
    pos_ids = rng.integers(0, 4, n)
    return ts, query, lengths, pos_ids


def test_shapes_and_finite():
    ts, q, ln, pid = _batch()
    r, s = NET.predict(ts, q, ln, pid)
    assert r.shape == (len(ln),) and s.shape == (len(ln),)
    assert np.isfinite(r).all() and np.isfinite(s).all()
    assert (s >= sm.SIGMA_FLOOR - 1e-6).all()


def test_empty_sequence_ok():
    ts, q, ln, pid = _batch()
    ln = np.zeros_like(ln)
    r, s = NET.predict(ts, q, ln, pid)
    assert np.isfinite(r).all() and np.isfinite(s).all()


def test_gate_shrinks_and_q90_orders():
    ts, q, ln, pid = _batch()
    r, s = NET.predict(ts, q, ln, pid)
    positions = [sm.POSITIONS[p] for p in pid]
    n_played = np.minimum(ln, 8)
    r_app, s_eff = sm.apply_gate(r, s, n_played, positions)
    # confidence gate never grows |r|
    assert np.all(np.abs(r_app) <= np.abs(r) + 1e-9)
    # thin history -> r fully gated to 0
    assert np.all(np.abs(r_app[n_played == 0]) < 1e-9)
    z = sm.z_array(NET.z_cal, positions)
    mu = np.full(len(r), 4.0)
    q90 = sm.q90_from(mu, r_app, s_eff, z=z)
    assert np.all(q90 >= mu + r_app - 1e-6)
    assert np.isfinite(q90).all()


def test_determinism():
    ts, q, ln, pid = _batch()
    r1, s1 = NET.predict(ts, q, ln, pid)
    r2, s2 = NET.predict(ts, q, ln, pid)
    assert np.array_equal(r1, r2) and np.array_equal(s1, s2)


def test_infer_refiner_missing_checkpoint_is_noop():
    ref = si.Stage10Refiner("1999-00", live=False)
    assert ref.ready is False
    ts, q, ln, pid = _batch()
    out = ref.predict_arrays(ts, q, ln, pid, np.minimum(ln, 8),
                             [sm.POSITIONS[p] for p in pid], np.full(len(ln), 4.0))
    assert np.all(out["r_applied"] == 0.0)


def test_torch_numpy_parity():
    """Only meaningful with the real checkpoints + torch present (dev machine)."""
    if not HAVE_CKPT:
        print("     (skip — no checkpoint)"); return
    try:
        import torch  # noqa: F401
    except Exception:                              # noqa: BLE001
        print("     (skip — torch not importable here)"); return
    import stage10_train as st
    rc = st.check_numpy("2024-25")
    assert rc == 0, "torch vs numpy diverged > 1e-4"


def _cleanup():
    try:
        os.remove(_TMP)
    except OSError:
        pass


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
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    _cleanup()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
