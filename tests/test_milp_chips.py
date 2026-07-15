"""
tests/test_milp_chips.py — Phase 4: chips as MILP variables.

Run:  docker run --rm -v "<repo>:/app" -w /app fpl-sim \
        python tests/test_milp_chips.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import milp_core as mc
from test_milp_core import OWNED
from test_milp_horizon import make_matrix, T

FRESH = {"used": set(), "reset_gws": [], "far_dgw": {}}


def solve(matrix, chip_state=FRESH, owned=OWNED, bank=25.0, ft=1, **kw):
    return mc.solve_horizon(matrix, owned, bank, ft, T,
                            chip_state=dict(chip_state), **kw)


def dgw_over(pids, g, mult=2.0):
    """Make players double at GW g (mu doubled, n_fix=2)."""
    return {(p, g): {"mu": 10.0, "n_fix": 2} for p in pids}


def test_no_chip_state_means_no_chips():
    plan = mc.solve_horizon(make_matrix(), OWNED, 25.0, 1, T)
    assert plan["chips"] == {}


def test_bb_fires_on_the_double_week():
    # WHOLE-POOL double gameweek at t+2: every bench doubles too, so BB's
    # bonus there strictly dominates any other week
    ov = {}
    rows0 = make_matrix()[T]
    for pid, r in rows0.items():
        ov[(pid, T + 2)] = {"mu": r["mu"] * 2.0, "n_fix": 2}
    plan = solve(make_matrix(week_over=ov))
    bb = [g for g, k in plan["chips"].items() if k.startswith("bb")]
    assert bb == [T + 2], plan["chips"]


def test_bb_reservation_guard_holds_for_far_dgw():
    # decent bench week in horizon, but a known DGW beyond it in the set:
    # the guard must forbid in-horizon BB on non-double weeks
    plan = solve(make_matrix(),
                 chip_state={"used": set(), "reset_gws": [],
                             "far_dgw": {1: True, 2: True}})
    assert not any(k.startswith("bb") for k in plan["chips"].values())


def test_used_chips_unavailable():
    st = {"used": {"bb1", "tc1", "wc1", "fh1"}, "reset_gws": [],
          "far_dgw": {}}
    ov = dgw_over((201, 202, 306, 307), T + 2)     # tempting BB week
    plan = solve(make_matrix(week_over=ov), chip_state=st)   # T=10: set 1
    assert plan["chips"] == {}, plan["chips"]


def test_tc_fires_on_captain_peak_week():
    ov = {(403, T + 3): {"mu": 15.0, "q90": 24.0, "n_fix": 2}}
    plan = solve(make_matrix(week_over=ov))
    tc = [g for g, k in plan["chips"].items() if k.startswith("tc")]
    assert tc == [T + 3], plan["chips"]
    assert plan["weeks"][T + 3]["captain"] == 403


def test_fh_only_on_event_weeks():
    # no doubles/blanks anywhere -> FH may never fire
    plan = solve(make_matrix())
    assert not any(k.startswith("fh") for k in plan["chips"].values())


def _blank_owned_at(g):
    ov = {}
    for p in OWNED:
        if p not in (101, 201, 202, 203):        # keep a rump playing
            ov[(p, g)] = {"mu": 0.0, "n_fix": 0, "pi": 0.0}
    return ov


def test_blank_week_rescued_by_a_reset_chip():
    # most of the owned squad blanks at t+2 -> a reset chip (FH or WC — an
    # in-model value tie; out-of-horizon chip scarcity is a known gap,
    # see phase4 report) must rescue the week with a playing XI, no hits
    ov = _blank_owned_at(T + 2)
    plan = solve(make_matrix(week_over=ov))
    chip = plan["chips"].get(T + 2, "")
    assert chip[:2] in ("fh", "wc"), plan["chips"]
    wk = plan["weeks"][T + 2]
    blanked = {p for (p, g) in ov if g == T + 2}
    assert not (set(wk["xi"]) & blanked), "blanked players in rescue XI"
    assert wk["hits"] == 0


def test_fh_rescues_blank_when_wc_unavailable():
    ov = _blank_owned_at(T + 2)
    st = {"used": {"wc1"}, "reset_gws": [], "far_dgw": {}}
    plan = solve(make_matrix(week_over=ov), chip_state=st)
    assert plan["chips"].get(T + 2, "").startswith("fh"), plan["chips"]
    wk = plan["weeks"][T + 2]
    blanked = {p for (p, g) in ov if g == T + 2}
    assert not (set(wk["xi"]) & blanked)
    assert wk["hits"] == 0


def test_wc_rebuilds_trash_squad_free():
    # whole owned squad worthless for every week; alternatives strong ->
    # WC fires and rebuilds without hits
    ov = {}
    for g in range(T, T + 5):
        for p in OWNED:
            ov[(p, g)] = {"mu": 1.0}
        for p in (103, 206, 207, 208, 306, 307, 308, 404, 405):
            ov[(p, g)] = {"mu": 9.0}
    plan = solve(make_matrix(week_over=ov), ft=1)
    wc = [g for g, k in plan["chips"].items() if k.startswith("wc")]
    assert wc, plan["chips"]
    g = wc[0]
    assert plan["weeks"][g]["hits"] == 0
    assert len(plan["weeks"][g]["transfers_in"]) >= 5


def test_reset_spacing_vs_played():
    # a reset chip was played at T-2 -> WC/FH blocked until T+2
    ov = {}
    for g in range(T, T + 5):
        for p in OWNED:
            ov[(p, g)] = {"mu": 1.0}
        for p in (103, 206, 207, 208, 306, 307, 308, 404, 405):
            ov[(p, g)] = {"mu": 9.0}
    st = {"used": {"fh1"}, "reset_gws": [T - 2], "far_dgw": {}}
    plan = solve(make_matrix(week_over=ov), chip_state=st, ft=1)
    wc = [g for g, k in plan["chips"].items() if k.startswith("wc")]
    assert all(g >= T - 2 + mc.SPACING_GAP for g in wc), plan["chips"]


def test_lockout_blocks_all_early_chips():
    # everything screams "chip now" but the lockout covers the horizon start
    ov = dgw_over((201, 202, 306, 307, 403), T + 1)
    st = {"used": set(), "reset_gws": [], "far_dgw": {},
          "lockout_until": T + 1}
    plan = solve(make_matrix(week_over=ov), chip_state=st)
    assert all(g > T + 1 for g in plan["chips"]), plan["chips"]


def test_percentile_rejection_blocks_plain_week_chip_only_now():
    # The simulator re-solves after the percentile ledger rejects a weak
    # plain-week WC/TC/BB.  FH is independently event-only, so no chip can
    # remain at t when all three unanchored kinds are blocked.
    st = {"used": set(), "reset_gws": [], "far_dgw": {},
          "blocked_now": {"wc", "tc", "bb"}}
    plan = solve(make_matrix(), chip_state=st)
    assert T not in plan["chips"], plan["chips"]


def test_tc_held_for_far_dgw():
    # decent captain weeks in horizon but no double; a far DGW exists in
    # the set -> TC must be reserved (same guard as BB)
    ov = {(403, T + 1): {"mu": 11.0, "q90": 18.0}}
    st = {"used": set(), "reset_gws": [], "far_dgw": {1: True, 2: True}}
    plan = solve(make_matrix(week_over=ov), chip_state=st)
    assert not any(k.startswith("tc") for k in plan["chips"].values()), \
        plan["chips"]


def test_tc_allowed_on_inhorizon_double_despite_far_flag():
    ov = {(403, T + 2): {"mu": 14.0, "q90": 22.0, "n_fix": 2}}
    st = {"used": set(), "reset_gws": [], "far_dgw": {1: True, 2: True}}
    plan = solve(make_matrix(week_over=ov), chip_state=st)
    tc = [g for g, k in plan["chips"].items() if k.startswith("tc")]
    assert tc == [T + 2], plan["chips"]


def test_wc_gated_on_squad_state():
    # trash squad + strong alternatives, but wc_ok=False -> WC must not fire
    ov = {}
    for g in range(T, T + 5):
        for p in OWNED:
            ov[(p, g)] = {"mu": 1.0}
        for p in (103, 206, 207, 208, 306, 307, 308, 404, 405):
            ov[(p, g)] = {"mu": 9.0}
    st = {"used": set(), "reset_gws": [], "far_dgw": {}, "wc_ok": False}
    plan = solve(make_matrix(week_over=ov), chip_state=st, ft=1)
    assert not any(k.startswith("wc") for k in plan["chips"].values()), \
        plan["chips"]


def test_one_chip_per_gw():
    ov = dgw_over((201, 202, 306, 307, 403), T + 2)
    plan = solve(make_matrix(week_over=ov))
    seen = list(plan["chips"].keys())
    assert len(seen) == len(set(seen))


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
