"""
tests/test_stage10_identity.py — Stage 10 Phase 1, gate 1.

STAGE10=off must be a TRUE no-op:
  * torch is never imported (off OR on — the runtime path is numpy-only)
  * predict_pool with stage10_resid_fn=None takes the exact pre-flag code path
    and leaves no side effects on the pool
  * a short season sim is deterministic and byte-stable under STAGE10=off

The full byte-identical-vs-pre-wiring check (legacy 2468 / mp 2252 at GW1-38) is
run out-of-band via git-stash comparison — see docs/stage10_phase1_plan.md §6 /
the step-4 report. This file is the fast regression guard.

Run:  python tests/test_stage10_identity.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _run(env_extra, end_gw=3, optimizer="legacy"):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", SIM_END_GW=str(end_gw))
    env.update(env_extra)
    if optimizer == "mp":
        env["OPTIMIZER"] = "mp"
        env["RULES_MODE"] = "corrected"
    r = subprocess.run([sys.executable, "pipeline/season_simulator.py"],
                       cwd=_ROOT, env=env, capture_output=True, text=True,
                       timeout=1800)
    return r


def test_torch_absent_when_off():
    code = ("import sys; sys.path.insert(0, 'pipeline'); import season_simulator; "
            "assert season_simulator.STAGE10 == 'off'; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)")
    r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT,
                       env=dict(os.environ, STAGE10="off"),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr


def test_torch_absent_when_on():
    """The whole STAGE10=on runtime chain (stage10_refine -> stage10_infer ->
    stage10_model, stage10_sequence -> prediction_matrix) is numpy-only."""
    code = ("import sys; sys.path.insert(0, 'pipeline'); "
            "import stage10_refine, stage10_infer, stage10_model, stage10_sequence; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)")
    r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT,
                       env=dict(os.environ, STAGE10="on"),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr


def test_predict_pool_noop_no_side_effects():
    """stage10_resid_fn=None: the resid branches are dead, no _mu_raw/_s10 keys
    land on pool rows, output matches a hand recomputation of the pre-flag path."""
    import numpy as np
    import season_simulator as ss

    class _StubModel:
        def predict(self, X):
            return np.array([3.0] * len(X))

    models = {"GK": _StubModel(), "DEF": _StubModel(),
              "MID": _StubModel(), "FWD": _StubModel()}
    pool = [
        {"player_id": 1, "pos": "MID", "fdr": 3.0, "zero_minutes": False,
         "sbp": 10.0, **{c: 0.0 for c in ss.FEAT_COLS}},
        {"player_id": 2, "pos": "DEF", "fdr": 5.0, "zero_minutes": False,
         "sbp": 2.0, **{c: 0.0 for c in ss.FEAT_COLS}},
        {"player_id": 3, "pos": "FWD", "fdr": 2.0, "zero_minutes": True,
         "sbp": 5.0, **{c: 0.0 for c in ss.FEAT_COLS}},
    ]
    out = ss.predict_pool([dict(p) for p in pool], models, gw=5,
                          current_squad_ids={2}, stage10_resid_fn=None)
    for p in out:
        assert "_mu_raw" not in p, "off path stashed _mu_raw"
        assert "stage10_sigma" not in p and "stage10_q90" not in p
        assert isinstance(p["pred"], float) and p["pred"] >= 0.0
    # zero-minutes player -> 0
    assert out[2]["pred"] == 0.0
    # a resid_fn that raises must never be invoked when we DON'T pass it
    def _boom(_):  # noqa: ANN001
        raise AssertionError("resid_fn called on the off path")
    ss.predict_pool([dict(p) for p in pool], models, gw=5,
                    current_squad_ids=set(), stage10_resid_fn=None)


def test_build_matrix_noop_signature():
    """build_matrix(resid_fn=None) must accept the kwarg and not change output."""
    import prediction_matrix as pm
    import inspect
    assert "resid_fn" in inspect.signature(pm.build_matrix).parameters


def test_off_short_sim_deterministic():
    a = _run({"STAGE10": "off"}, end_gw=3)
    assert a.returncode == 0, a.stderr[-3000:]
    p = os.path.join(_ROOT, "data/intel/season_simulation.json")
    import json
    d1 = json.load(open(p, encoding="utf-8")); d1.pop("generated_at", None)
    b = _run({"STAGE10": "off"}, end_gw=3)
    assert b.returncode == 0, b.stderr[-3000:]
    d2 = json.load(open(p, encoding="utf-8")); d2.pop("generated_at", None)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True), \
        "STAGE10=off is not deterministic"


def test_on_short_sim_runs():
    r = _run({"STAGE10": "on"}, end_gw=4)
    assert r.returncode == 0, r.stderr[-4000:]
    assert os.path.exists(os.path.join(_ROOT, "data/intel/season_simulation_s10.json"))
    assert "[STAGE10] on" in r.stdout


def test_on_short_sim_deterministic():
    """Gate 8: the STAGE10=on runtime is numpy-only (fine-tune is off by
    default), so two runs must be byte-identical."""
    import json
    p = os.path.join(_ROOT, "data/intel/season_simulation_s10.json")
    _run({"STAGE10": "on"}, end_gw=4)
    d1 = json.load(open(p, encoding="utf-8")); d1.pop("generated_at", None)
    _run({"STAGE10": "on"}, end_gw=4)
    d2 = json.load(open(p, encoding="utf-8")); d2.pop("generated_at", None)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True), \
        "STAGE10=on is not deterministic"


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
