"""
tests/test_milp_core.py — Phase 2 MILP core tests on synthetic pools.

Run:  docker run --rm -v "<repo>:/app" -w /app fpl-sim \
        python tests/test_milp_core.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
import milp_core as mc

_team_seq = {}


def mk(pid, et, price=5.0, mu=5.0, pi=0.9, q90=None, sell=None, team=None):
    return {"mu": mu, "pi": pi, "q90": q90 if q90 is not None else mu + 3.0,
            "n_fix": 1, "phi": 1.0, "price": price,
            "sell_value": sell if sell is not None else price,
            "pos": {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}[et],
            "element_type": et, "team": team if team is not None else pid}


def base_rows(overrides=None):
    """24-player pool; owned = a legal 2/5/5/3 squad of the x01..x05 ids."""
    rows = {}
    for pid in (101, 102, 103):
        rows[pid] = mk(pid, 1, mu=4.0)
    for pid in (201, 202, 203, 204, 205, 206, 207, 208):
        rows[pid] = mk(pid, 2, mu=4.5)
    for pid in (301, 302, 303, 304, 305, 306, 307, 308):
        rows[pid] = mk(pid, 3, mu=5.0)
    for pid in (401, 402, 403, 404, 405):
        rows[pid] = mk(pid, 4, mu=5.5)
    for pid, kw in (overrides or {}).items():
        rows[pid].update(kw)
        if "sell" in kw:
            rows[pid]["sell_value"] = kw["sell"]
        if "mu" in kw and "q90" not in kw:
            rows[pid]["q90"] = kw["mu"] + 3.0
    return rows


OWNED = {101, 102, 201, 202, 203, 204, 205,
         301, 302, 303, 304, 305, 401, 402, 403}


def solve(rows, owned=OWNED, budget=100.0, ft=1, **kw):
    return mc.solve_gw(rows, owned, budget, ft, gw=10, **kw)


def test_structure():
    res = solve(base_rows())
    ets = [base_rows()[p]["element_type"] for p in res["squad"]]
    assert len(res["squad"]) == 15 and len(res["xi"]) == 11
    assert ets.count(1) == 2 and ets.count(2) == 5
    assert ets.count(3) == 5 and ets.count(4) == 3
    xi_ets = [base_rows()[p]["element_type"] for p in res["xi"]]
    assert xi_ets.count(1) == 1 and 3 <= xi_ets.count(2) <= 5
    assert 2 <= xi_ets.count(3) <= 5 and 1 <= xi_ets.count(4) <= 3
    assert len(res["bench"]) == 4


def test_captain_vice_basics():
    rows = base_rows({305: {"mu": 12.0, "q90": 20.0},
                        403: {"mu": 10.0, "q90": 14.0}})
    res = solve(rows)
    assert res["captain"] == 305
    assert res["vice"] is not None and res["vice"] != res["captain"]
    assert res["captain"] in res["xi"] and res["vice"] in res["xi"]


def test_captain_prefers_ceiling_at_equal_mu():
    rows = base_rows({305: {"mu": 9.0, "q90": 18.0},
                        403: {"mu": 9.0, "q90": 12.0}})
    res = solve(rows)
    assert res["captain"] == 305   # theta=0.5 blend favors the ceiling


def test_captain_penalized_by_low_play_prob():
    rows = base_rows({305: {"mu": 9.0, "q90": 14.0, "pi": 0.45},
                        403: {"mu": 8.0, "q90": 12.5, "pi": 0.95}})
    res = solve(rows)
    # kappa(305) = .45*11.5 = 5.18 < kappa(403) = .95*10.25 = 9.7
    assert res["captain"] == 403


def test_budget_and_sell_value_identity():
    # Owned player price rose to 8.0 but sells at 6.5; budget is exactly
    # squad sell value -> keeping the squad must stay feasible.
    rows = base_rows({301: {"price": 8.0, "sell": 6.5}})
    budget = sum(rows[p]["sell_value"] if p in OWNED else 0 for p in rows)
    res = solve(rows, budget=budget)
    spend = sum(rows[p]["sell_value"] if p in OWNED else rows[p]["price"]
                for p in res["squad"])
    assert spend <= budget + 1e-6


def test_free_transfer_used_hit_declined():
    # one huge upgrade (worth it), one tiny upgrade (not worth a -4 hit)
    rows = base_rows({305: {"mu": 0.0}, 306: {"mu": 10.0},
                        304: {"mu": 4.8}, 307: {"mu": 5.3}})
    res = solve(rows, ft=1)
    assert 306 in res["transfers_in"] and 305 in res["transfers_out"]
    assert res["hits"] == 0 and len(res["transfers_in"]) == 1


def test_hit_taken_when_horizon_gain_exceeds_4():
    rows = base_rows({305: {"mu": 0.0}, 306: {"mu": 10.0},
                        304: {"mu": 0.0}, 307: {"mu": 9.0}})
    res = solve(rows, ft=1)
    assert set(res["transfers_in"]) >= {306, 307}
    assert res["hits"] == 1


def test_hit_cap_enforced():
    ov = {p: {"mu": 0.0} for p in (301, 302, 303, 304, 305)}
    ov.update({p: {"mu": 12.0} for p in (306, 307, 308)})
    rows = base_rows(ov)
    res = solve(rows, ft=1)
    assert res["hits"] <= mc.HIT_CAP
    assert len(res["transfers_in"]) <= 1 + mc.HIT_CAP


def test_hit_cost_raises_bar():
    # 307 in (mu 10) displaces the marginal XI player (mu 4.5): +5.5 gain
    # clears the default -4 price but not a raised -8.
    rows = base_rows({305: {"mu": 0.0}, 306: {"mu": 10.0},
                        304: {"mu": 0.0}, 307: {"mu": 10.0}})
    assert solve(rows, ft=1)["hits"] == 1
    res = solve(rows, ft=1, hit_cost=8.0)
    assert res["hits"] == 0
    # the FT still buys one of the (equal) upgrades — no second on a hit
    assert len(res["transfers_in"]) == 1
    assert set(res["transfers_in"]) <= {306, 307}


def test_hit_cap_zero_blocks_all_hits():
    rows = base_rows({305: {"mu": 0.0}, 306: {"mu": 10.0},
                        304: {"mu": 0.0}, 307: {"mu": 9.0}})
    res = solve(rows, ft=1, hit_cap=0)
    assert res["hits"] == 0 and len(res["transfers_in"]) == 1
    assert 306 in res["transfers_in"]        # FT goes to the bigger upgrade


def test_ft_friction_blocks_sideways_swap():
    # 307 edges owned 305 by +0.4 mu — a sub-noise sideways move. With
    # friction 1.0 the FT is held; a genuinely big upgrade still happens.
    rows = base_rows({305: {"mu": 5.0}, 307: {"mu": 5.4}})
    assert 307 in solve(rows, ft=1)["transfers_in"]      # free => churn
    res = solve(rows, ft=1, ft_value=1.0)
    assert res["transfers_in"] == []                     # friction => hold
    rows = base_rows({305: {"mu": 0.0}, 307: {"mu": 9.0}})
    res = solve(rows, ft=1, ft_value=1.0)
    assert 307 in res["transfers_in"]                    # real upgrade goes


def test_form_hold_protects_hauler():
    # owned 305 is the weakest (mu 3) and would normally be the one sold
    # for incoming 307 — but he just hauled, so the hold makes the solver
    # sell someone else (or hold) rather than the in-form player
    rows = base_rows({305: {"mu": 3.0}, 307: {"mu": 6.5}})
    res = solve(rows, ft=1)
    assert res["transfers_out"] == [305]                 # baseline: sold
    res = solve(rows, ft=1, sell_hold={305: 5.0})
    assert 305 not in res["transfers_out"]               # hauler protected


def test_rebuy_lock_blocks_transfer_in():
    # 306 is the obvious FT upgrade — but recently sold, so locked out
    rows = base_rows({305: {"mu": 0.0}, 306: {"mu": 10.0}})
    assert 306 in solve(rows, ft=1)["transfers_in"]
    res = solve(rows, ft=1, no_rebuy={306})
    assert 306 not in res["transfers_in"]


def test_wildcard_free_rebuild():
    ov = {p: {"mu": 0.0} for p in OWNED}
    rows = base_rows(ov)
    res = solve(rows, is_wildcard=True)
    assert res["hits"] == 0
    # pool has exactly 9 non-owned alternatives — all must come in, free
    assert len(res["transfers_in"]) == 9


def test_bench_ev_prefers_playing_bench():
    # two equal-price fodder MIDs: one plays (pi .9), one never (pi 0).
    # Budget 74.0 = 14x5.0 + 4.0 — exactly one cheap MID must be squadded;
    # the bench EV term must pick the one who actually plays.
    rows = base_rows({307: {"mu": 3.0, "pi": 0.9, "price": 4.0},
                      308: {"mu": 3.0, "pi": 0.0, "price": 4.0}})
    res = mc.solve_gw(rows, set(), 74.0, 1, gw=1)
    assert 307 in res["squad"] and 308 not in res["squad"]


def test_gw1_fresh_pick_no_transfers():
    res = mc.solve_gw(base_rows(), set(), 100.0, 1, gw=1)
    assert len(res["squad"]) == 15 and res["hits"] == 0
    assert len(res["transfers_in"]) == 15 and res["transfers_out"] == []


def test_infeasible_raises():
    try:
        solve(base_rows(), budget=10.0)      # cannot afford any 15
        assert False, "expected RuntimeError"
    except RuntimeError:
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
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
