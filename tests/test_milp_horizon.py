"""
tests/test_milp_horizon.py — Phase 3 multi-period MILP tests.

The behaviours the horizon exists to create — verified on synthetic pools:
FT banking emerges, hits priced on horizon-summed gains, blank-week
benching, FT trajectory, WC hit waiver.

Run:  docker run --rm -v "<repo>:/app" -w /app fpl-sim \
        python tests/test_milp_horizon.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import milp_core as mc
from test_milp_core import base_rows, OWNED

T = 10


def make_matrix(t=T, H=5, week_over=None):
    """{g: {pid: row}} from the Phase 2 pool, with per-(pid, g) overrides."""
    rows0 = base_rows()
    m = {}
    for g in range(t, t + H):
        wk = {}
        for pid, r in rows0.items():
            rr = dict(r)
            ov = (week_over or {}).get((pid, g))
            if ov:
                rr.update(ov)
                if "mu" in ov and "q90" not in ov:
                    rr["q90"] = ov["mu"] + 3.0
            wk[pid] = rr
        m[g] = wk
    return m


def solve(matrix, owned=OWNED, bank=25.0, ft=1, **kw):
    return mc.solve_horizon(matrix, owned, bank, ft, T, **kw)


def test_week_t_result_shape():
    plan = solve(make_matrix())
    wk = plan["weeks"][T]
    for k in ("squad", "xi", "bench", "captain", "vice",
              "transfers_in", "transfers_out", "hits"):
        assert k in wk, k
    assert len(wk["squad"]) == 15 and len(wk["xi"]) == 11
    assert set(plan["weeks"]) == set(range(T, T + 5))


def test_banking_emerges():
    # 306/307 are worthless THIS week but elite from t+1; selling starters
    # 304/305 now costs week-t XI points and a hit. Optimal: hold the FT,
    # make both moves at t+1 with the banked pair. This is the behaviour
    # no single-GW optimizer can express.
    ov = {}
    for g in range(T, T + 5):
        ov[(306, g)] = {"mu": 0.0 if g == T else 10.0}
        ov[(307, g)] = {"mu": 0.0 if g == T else 10.0}
    plan = solve(make_matrix(week_over=ov), ft=1)
    assert len(plan["weeks"][T]["transfers_in"]) == 0, \
        plan["weeks"][T]["transfers_in"]
    assert set(plan["weeks"][T + 1]["transfers_in"]) == {306, 307}
    assert all(w["hits"] == 0 for w in plan["weeks"].values())
    assert plan["ft_plan"][T + 1] == 2          # the banked pair


def test_hit_priced_on_horizon_gains():
    # BOTH upgrades explosive NOW (mu 12 vs owned 5) then +2/week —
    # deferring either loses 7 pts this week > the 4-pt hit, so taking
    # the second transfer on a hit at t is genuinely optimal.
    ov = {}
    for g in range(T, T + 5):
        ov[(306, g)] = {"mu": 12.0 if g == T else 7.0}
        ov[(307, g)] = {"mu": 12.0 if g == T else 7.0}
    plan = solve(make_matrix(week_over=ov), ft=1)
    wk = plan["weeks"][T]
    assert set(wk["transfers_in"]) == {306, 307}, wk["transfers_in"]
    assert wk["hits"] == 1


def test_steady_upgrade_deferred_not_hit():
    # one explosive (307), one steady +2/week (306): the steady one should
    # be DEFERRED to next week's free transfer, not bought on a -4 hit —
    # the exact reasoning a single-GW optimizer cannot express
    ov = {}
    for g in range(T, T + 5):
        ov[(306, g)] = {"mu": 7.0}
        ov[(307, g)] = {"mu": 12.0 if g == T else 7.0}
    plan = solve(make_matrix(week_over=ov), ft=1)
    assert plan["weeks"][T]["transfers_in"] == [307]
    assert 306 in plan["weeks"][T + 1]["transfers_in"]
    assert all(w["hits"] == 0 for w in plan["weeks"].values())


def test_small_gain_hit_declined():
    # both upgrades only +0.8/week — one uses the FT (free, worth taking),
    # the second never justifies -4 within the horizon
    ov = {}
    for g in range(T, T + 5):
        ov[(306, g)] = {"mu": 5.8}
        ov[(307, g)] = {"mu": 5.8}
    plan = solve(make_matrix(week_over=ov), ft=1)
    assert all(w["hits"] == 0 for w in plan["weeks"].values())


def test_hit_cost_flips_horizon_hit():
    # the test_hit_priced_on_horizon_gains scenario: the marginal hit buys
    # ~7 pts (the deferred upgrade's week-t burst). Worth it at -4; at a
    # -10 decision price the second move waits for next week's FT instead.
    ov = {}
    for g in range(T, T + 5):
        ov[(306, g)] = {"mu": 12.0 if g == T else 7.0}
        ov[(307, g)] = {"mu": 12.0 if g == T else 7.0}
    plan = solve(make_matrix(week_over=ov), ft=1, hit_cost=10.0)
    wk = plan["weeks"][T]
    assert wk["hits"] == 0 and len(wk["transfers_in"]) == 1
    later = {p for g in range(T + 1, T + 5)
             for p in plan["weeks"][g]["transfers_in"]}
    assert ({306, 307} - set(wk["transfers_in"])) <= later
    assert all(w["hits"] == 0 for w in plan["weeks"].values())


def test_ft_friction_holds_over_horizon():
    # 306 edges an owned starter by +0.3/wk all horizon: 5 discounted weeks
    # of paper gain (~1.35) beat one week of friction, so friction must be
    # charged per week-of-horizon... it is charged ONCE per executed
    # transfer at its week's discount — the swap clears 1.0 friction over
    # a full horizon but NOT 2.5 (sub-noise edges stay blocked).
    ov = {}
    for g in range(T, T + 5):
        ov[(306, g)] = {"mu": 5.3}   # owned MIDs are 5.0
    plan = solve(make_matrix(week_over=ov), ft=1, ft_value=2.5)
    assert all(w["transfers_in"] == [] for w in plan["weeks"].values())
    # a real upgrade (+7/wk) sails through the same friction
    ov = {(306, g): {"mu": 12.0} for g in range(T, T + 5)}
    plan = solve(make_matrix(week_over=ov), ft=1, ft_value=2.5)
    assert 306 in plan["weeks"][T]["transfers_in"]


def test_bench_slots_solve_across_horizon():
    plan = solve(make_matrix(), ft=1,
                 w_bench_slots=(0.35, 0.10, 0.02), w_bench_gk=0.04)
    for wk in plan["weeks"].values():
        assert len(wk["squad"]) == 15 and len(wk["xi"]) == 11
        assert len(wk["bench"]) == 4


def test_form_hold_protects_hauler_across_horizon():
    # 305 is weakest all horizon but just hauled: with the hold he must
    # not be the sale in ANY planned week
    ov = {}
    for g in range(T, T + 5):
        ov[(305, g)] = {"mu": 3.0}
        ov[(306, g)] = {"mu": 6.5}
    plan = solve(make_matrix(week_over=ov), ft=1, sell_hold={305: 5.0})
    for wk in plan["weeks"].values():
        assert 305 not in wk["transfers_out"]


def test_rebuy_lock_defers_to_expiry():
    # 306 elite all horizon; owned 305 is dead weight. Lock runs through
    # T+1, so the buy may happen at T+2 at the earliest — and does.
    ov = {}
    for g in range(T, T + 5):
        ov[(306, g)] = {"mu": 12.0}
        ov[(305, g)] = {"mu": 0.0}
    plan = solve(make_matrix(week_over=ov), ft=1, no_rebuy={306: T + 1})
    assert 306 not in plan["weeks"][T]["transfers_in"]
    assert 306 not in plan["weeks"][T + 1]["transfers_in"]
    assert any(306 in plan["weeks"][g]["transfers_in"]
               for g in range(T + 2, T + 5))


def test_churn_guard_structural():
    # busy scenario: nobody may be bought twice or sold twice across the plan
    ov = {}
    for g in range(T, T + 5):
        ov[(306, g)] = {"mu": 12.0 if g % 2 == 0 else 0.0}   # oscillating
        ov[(307, g)] = {"mu": 0.0 if g % 2 == 0 else 12.0}
    plan = solve(make_matrix(week_over=ov), ft=2)
    ins, outs = {}, {}
    for g, wk in plan["weeks"].items():
        for p in wk["transfers_in"]:
            ins[p] = ins.get(p, 0) + 1
        for p in wk["transfers_out"]:
            outs[p] = outs.get(p, 0) + 1
    assert all(n <= 1 for n in ins.values()), ins
    assert all(n <= 1 for n in outs.values()), outs


def test_ft_trajectory_banks_to_cap():
    # no upgrade anywhere -> no transfers; FTs accrue 1,2,3,4,5
    plan = solve(make_matrix(), ft=1)
    assert all(len(w["transfers_in"]) == 0 for w in plan["weeks"].values())
    assert [plan["ft_plan"][g] for g in range(T, T + 5)] == [1, 2, 3, 4, 5]


def test_ft_event_grant_inside_horizon():
    plan = solve(make_matrix(), ft=1, ft_events={T + 2: 5})
    assert plan["ft_plan"][T + 2] == 5


def test_blank_week_player_benched():
    # owned starter 305 blanks at t+2 -> must not start that week
    ov = {(305, T + 2): {"mu": 0.0, "n_fix": 0, "pi": 0.0}}
    plan = solve(make_matrix(week_over=ov))
    assert 305 not in plan["weeks"][T + 2]["xi"]


def test_wildcard_now_waives_hits():
    ov = {}
    for g in range(T, T + 5):
        for pid in (306, 307, 308, 404, 405):
            ov[(pid, g)] = {"mu": 12.0}
    plan = solve(make_matrix(week_over=ov), ft=1, is_wildcard_now=True)
    wk = plan["weeks"][T]
    assert len(wk["transfers_in"]) >= 4       # wholesale, free
    assert wk["hits"] == 0


def test_gw1_initial_build():
    m = {g: base_rows() for g in range(1, 6)}
    plan = mc.solve_horizon(m, set(), 100.0, 1, 1)
    wk = plan["weeks"][1]
    assert len(wk["squad"]) == 15 and wk["hits"] == 0
    assert plan["bank_plan"][1] >= 0


def test_bank_recursion_consistent():
    # upgrade at t+1 to a pricier player must be funded by the sale + bank
    ov = {}
    for g in range(T, T + 5):
        ov[(306, g)] = {"mu": 9.0, "price": 7.0}
    plan = solve(make_matrix(week_over=ov), bank=1.0, ft=1)
    for g in range(T, T + 5):
        assert plan["bank_plan"][g] >= -1e-6


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
