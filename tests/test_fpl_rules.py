"""
tests/test_fpl_rules.py — golden tests for pure FPL rule accounting (Phase 0).

No pytest dependency. Run directly:
  python tests/test_fpl_rules.py
Docker (this machine has no local Python):
  docker run --rm -v "<repo>:/app" -w /app fpl-sim python tests/test_fpl_rules.py
Exit code 0 = all pass.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
from fpl_rules import (sell_value, squad_sell_value,
                       next_free_transfers, hit_points)


# ── sell_value ────────────────────────────────────────────────────────────────

def test_sell_value_golden():
    # (purchase, market, expected) — hand-checked against the FPL rule:
    # rise -> purchase + floor(rise/2, 0.1);  fall/flat -> market
    cases = [
        (5.0,  5.0,  5.0),   # unchanged
        (5.0,  5.1,  5.0),   # +0.1 -> +0.05 -> floors to +0.0
        (5.0,  5.2,  5.1),   # +0.2 -> +0.1
        (5.0,  5.3,  5.1),   # +0.3 -> +0.15 -> floors to +0.1
        (5.0,  5.4,  5.2),
        (5.0,  6.0,  5.5),
        (6.3,  6.4,  6.3),   # odd tenths: +0.1 floors to 0
        (7.5,  8.0,  7.7),   # +0.5 -> +0.25 -> +0.2
        (10.0, 11.1, 10.5),  # +1.1 -> +0.55 -> +0.5
        (5.0,  4.6,  4.6),   # price fell: sell at market
        (5.0,  4.9,  4.9),
        (14.0, 14.5, 14.2),
        (4.0,  4.0,  4.0),
    ]
    for pp, mp, want in cases:
        got = sell_value(pp, mp)
        assert abs(got - want) < 1e-9, f"sell_value({pp},{mp}) = {got}, want {want}"


def test_sell_value_properties():
    # Rising market: purchase <= sell <= market, sell on 0.1 grid.
    # Falling market: sell == market exactly.
    for pp10 in range(38, 155, 3):          # £3.8m .. £15.4m purchase
        for mp10 in range(38, 160, 7):
            pp, mp = pp10 / 10.0, mp10 / 10.0
            sv = sell_value(pp, mp)
            assert abs(sv * 10 - round(sv * 10)) < 1e-9, "not on 0.1 grid"
            if mp <= pp:
                assert abs(sv - mp) < 1e-9
            else:
                assert pp - 1e-9 <= sv <= mp + 1e-9
    # Monotone in market price for fixed purchase price
    prev = 0.0
    for mp10 in range(50, 90):
        sv = sell_value(5.0, mp10 / 10.0)
        assert sv >= prev - 1e-9
        prev = sv


def test_squad_sell_value():
    pp = {1: 5.0, 2: 10.0}
    mp = {1: 5.3, 2: 9.5, 3: 4.5}          # pid 3 not in ledger -> market
    # 5.1 + 9.5 + 4.5 = 19.1
    assert abs(squad_sell_value(pp, mp) - 19.1) < 1e-9


# ── next_free_transfers ───────────────────────────────────────────────────────

def test_ft_accrual_and_banking():
    assert next_free_transfers(2, 1, 0, False, False) == 2   # bank one
    assert next_free_transfers(2, 1, 1, False, False) == 1   # spent it
    assert next_free_transfers(9, 4, 0, False, False) == 5   # bank to cap
    assert next_free_transfers(9, 5, 0, False, False) == 5   # cap holds
    assert next_free_transfers(9, 5, 5, False, False) == 1   # spend all five
    assert next_free_transfers(9, 3, 2, False, False) == 2
    # No cap-2 before GW15 (the legacy quirk must NOT exist here)
    assert next_free_transfers(5, 2, 0, False, False) == 3
    assert next_free_transfers(10, 4, 0, False, False) == 5


def test_ft_hits_do_not_go_negative():
    # 3 transfers with 1 FT = 2 hits; only 1 FT was consumed
    assert next_free_transfers(6, 1, 3, False, False) == 1
    assert next_free_transfers(6, 2, 5, False, False) == 1


def test_ft_chips_preserve_bank():
    # WC/FH consume nothing; accrual continues
    assert next_free_transfers(8, 3, 12, True,  False) == 4   # wildcard
    assert next_free_transfers(8, 3, 11, False, True)  == 4   # free hit
    assert next_free_transfers(8, 5, 15, True,  False) == 5   # capped


def test_ft_gw1_free():
    assert next_free_transfers(1, 1, 15, False, False) == 2


def test_ft_events():
    ev = {15: 5}
    assert next_free_transfers(14, 2, 1, False, False, ft_events=ev) == 5
    assert next_free_transfers(14, 2, 1, False, False, ft_events={}) == 2
    assert next_free_transfers(15, 5, 0, False, False, ft_events=ev) == 5


def test_ft_season_walk_invariant():
    # 38-GW walk with a mix of usage — FT must stay in [1, 5] throughout
    ft = 1
    usage = [0, 1, 2, 0, 0, 1, 3, 0, 1, 0] * 4
    for gw in range(1, 39):
        used = usage[gw % len(usage)]
        is_wc = (gw == 17)
        is_fh = (gw == 26)
        ft = next_free_transfers(gw, ft, used, is_wc, is_fh, ft_events={15: 5})
        assert 1 <= ft <= 5, f"GW{gw}: ft={ft} out of [1,5]"


# ── hit_points ────────────────────────────────────────────────────────────────

def test_hit_points():
    assert hit_points(1, 1, False, False) == 0
    assert hit_points(2, 1, False, False) == 4
    assert hit_points(3, 1, False, False) == 8
    assert hit_points(5, 5, False, False) == 0
    assert hit_points(0, 1, False, False) == 0
    assert hit_points(15, 1, True,  False) == 0   # wildcard
    assert hit_points(15, 1, False, True)  == 0   # free hit


# ── runner ────────────────────────────────────────────────────────────────────

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
