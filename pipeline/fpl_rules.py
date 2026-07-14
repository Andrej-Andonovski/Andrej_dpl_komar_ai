"""
pipeline/fpl_rules.py
Pure FPL rule accounting — stdlib only, no pandas/model deps.

Source of truth for the Phase 0 "corrected rules" mode
(docs/optimizer_redesign.md §5 sell price, §4.6 free transfers, §9 Phase 0).
Golden tests: tests/test_fpl_rules.py (run with plain python, no pytest).

All prices are in £m. Arithmetic is done in integer tenths so that
0.1-step rounding is exact and float drift cannot change a sell price.
"""


def _tenths(price):
    """£m float -> integer tenths (5.3 -> 53)."""
    return int(round(price * 10))


def sell_value(purchase_price, market_price):
    """
    FPL sell rule:
      - price fell or unchanged: sell at current market price
      - price rose: sell = purchase + 50% of the rise, rounded DOWN to £0.1m
    Examples: bought 5.0 now 5.3 -> 5.1;  bought 5.0 now 5.1 -> 5.0;
              bought 6.3 now 6.4 -> 6.3;  bought 5.0 now 4.6 -> 4.6.
    """
    pp = _tenths(purchase_price)
    mp = _tenths(market_price)
    if mp <= pp:
        return mp / 10.0
    return (pp + (mp - pp) // 2) / 10.0


def squad_sell_value(purchase_prices, market_prices):
    """
    Total sell value of a squad.
    purchase_prices: {player_id: price_paid}
    market_prices:   {player_id: current_price} — must cover every ledger key.
    Players missing from purchase_prices are valued at market (no profit),
    which is the safe fallback for an incomplete ledger.
    """
    total = 0.0
    for pid, mp in market_prices.items():
        pp = purchase_prices.get(pid, mp)
        total += sell_value(pp, mp)
    return round(total, 1)


def next_free_transfers(gw, ft_start, transfers_made,
                        is_wildcard, is_freehit,
                        ft_cap=5, ft_events=None):
    """
    Free transfers available at GW gw+1 under real 2025-26 rules:
      - +1 accrues every week, bank capped at ft_cap (5), never below 1
      - Wildcard / Free Hit weeks consume NO free transfers (chips preserve
        the bank; accrual still happens)
      - GW1 squad selection consumes nothing
      - hits (transfers beyond ft_start) cannot push next week's FTs below 1
      - ft_events: {gw: granted_ft} one-off rule events (e.g. the AFCON
        grant) — configuration, never hardcoded in optimizer logic
    """
    ft_events = ft_events or {}
    if is_wildcard or is_freehit or gw == 1:
        consumed = 0
    else:
        consumed = min(transfers_made, ft_start)
    nxt = gw + 1
    if nxt in ft_events:
        return ft_events[nxt]
    return max(1, min(ft_cap, ft_start - consumed + 1))


def hit_points(transfers_made, ft_available, is_wildcard, is_freehit):
    """Points deducted for transfers beyond the free allowance (-4 each)."""
    if is_wildcard or is_freehit:
        return 0
    return 4 * max(0, transfers_made - ft_available)
