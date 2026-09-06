"""
pipeline/free_hit_preview.py
What-if: what would a Free Hit squad look like for the target GW, built
from your REAL bank + real squad sell value, if you played the chip?

Single-GW MILP (milp_core.solve_gw, is_freehit=True) — full budget rebuild,
no hit cost, squad reverts next week (this script doesn't touch your real
squad, it's a preview only). Reuses the same live-fetch + retrain machinery
as gw_recommendation.py.

Usage: python pipeline/free_hit_preview.py
"""
import os
import sys

os.environ.setdefault("RULES_MODE", "corrected")
os.environ.setdefault("OPTIMIZER", "mp")
os.environ.setdefault("MP_HORIZON", "5")
os.environ.setdefault("OWN_PRIOR_GW1", "0.213")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pipeline import season_simulator as ss
from pipeline import prediction_matrix as pmx
from pipeline.milp_core import solve_gw as milp_solve_gw
from pipeline.fpl_rules import sell_value
from pipeline.gw_recommendation import (fetch_team_state, build_name_lookup,
                                        team_short_names, FPL_TEAM_ID)


def main():
    print("=" * 70)
    print(f"  FREE HIT PREVIEW — team {FPL_TEAM_ID}")
    print("=" * 70)

    print("\n[FPL] Fetching live team state...")
    state = fetch_team_state(FPL_TEAM_ID)
    target_gw = state["target_gw"]
    completed_gw = state["completed_gw"]
    print(f"  Target GW: {target_gw}  |  Bank: £{state['bank']:.1f}m")

    print("\n[LOAD] Loading data...")
    hist_lookup = ss.load_player_history()
    players_df = ss.load_players_raw()
    fdr_lookup, home_lookup, dgw_gws, gw_teams = ss.load_fixtures()
    fixture_list = pmx.load_fixture_list(ss.FIXTURES_CSV)
    train_dfs = ss.load_training_data()
    avail_gws = ss.load_availability()

    name_lookup = build_name_lookup(players_df)
    team_names = team_short_names()

    print("\n[LOAD] Building historical team form lookup...")
    hist_team_form = ss.build_hist_team_form_lookup()
    hist_rows_for_retrain = ss.build_hist_rows(train_dfs, hist_team_form)

    print(f"\n[TRAIN] Retraining on actuals through GW{completed_gw}...")
    team_form_lookup = ss.build_team_form_lookup(hist_lookup, players_df,
                                                 completed_gw)
    opp_lookup = ss.build_opponent_lookup(ss.FIXTURES_CSV, team_form_lookup)
    retrain_rows = ss.build_retrain_rows(
        players_df, hist_lookup, fdr_lookup, home_lookup, completed_gw,
        team_form_lookup=team_form_lookup, opp_lookup=opp_lookup)
    cs_weight = 1 + completed_gw
    combined_rows, combined_weights = {}, {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        h_rows = hist_rows_for_retrain.get(pos, [])
        r_rows = retrain_rows.get(pos, [])
        combined_rows[pos] = h_rows + r_rows
        combined_weights[pos] = [1.0] * len(h_rows) + [float(cs_weight)] * len(r_rows)
    models = ss.train_models(combined_rows, combined_weights)

    print(f"\n[POOL] Building GW{target_gw} pool...")
    pool = ss.build_rolling_pool(players_df, hist_lookup, fdr_lookup,
                                 home_lookup, completed_gw,
                                 team_form_lookup=team_form_lookup,
                                 opp_lookup=opp_lookup)

    print(f"\n[MATRIX] Building prediction matrix for GW{target_gw}...")
    mp_matrix = pmx.build_matrix(pool, models, fixture_list, target_gw, 1,
                                 hist_lookup=hist_lookup, avail_gws=avail_gws,
                                 purchase_price=None)
    rows = mp_matrix[target_gw]

    # Real total budget: bank + sell value of your CURRENT real squad
    # (market-price approximation, same caveat as gw_recommendation.py —
    # no purchase-price ledger without login).
    owned = set(state["squad_ids"])
    sv_total = 0.0
    for pid in owned:
        r = rows.get(pid)
        mp = r["price"] if r else name_lookup.get(pid, {}).get("price", 0.0)
        sv_total += mp  # no ledger -> sell_value falls back to market price
    available_budget = state["bank"] + sv_total
    print(f"  Approx. Free Hit budget: £{available_budget:.1f}m "
         f"(bank £{state['bank']:.1f}m + squad market value £{sv_total:.1f}m)")

    print(f"\n[MILP] Solving Free Hit squad for GW{target_gw}...")
    result = milp_solve_gw(rows, owned, available_budget,
                           free_transfers=0, gw=target_gw,
                           is_freehit=True, theta=ss.MP_THETA,
                           gamma=ss.MP_GAMMA, w_bench=ss.MP_W_BENCH,
                           w_bench_slots=ss.MP_BENCH_SLOTS,
                           w_bench_gk=ss.MP_W_BENCH_GK)

    def name(pid):
        p = name_lookup.get(pid, {})
        team = team_names.get(p.get("team"), "?")
        return f"{p.get('web_name', pid)} ({team})"

    kept   = [p for p in result["squad"] if p in owned]
    fresh  = [p for p in result["squad"] if p not in owned]

    print("\n" + "=" * 70)
    print(f"  FREE HIT SQUAD — GW{target_gw}  (preview only, not executed)")
    print("=" * 70)
    print(f"\nXI:")
    for pid in result["xi"]:
        tag = " [C]" if pid == result["captain"] else \
              (" [VC]" if pid == result["vice"] else "")
        row = rows.get(pid, {})
        owned_tag = "" if pid in owned else "  (NEW)"
        print(f"  {name(pid):32s} mu={row.get('mu', 0):.2f}{tag}{owned_tag}")
    print(f"\nBench:")
    for pid in result["bench"]:
        row = rows.get(pid, {})
        owned_tag = "" if pid in owned else "  (NEW)"
        print(f"  {name(pid):32s} mu={row.get('mu', 0):.2f}{owned_tag}")

    print(f"\nKept from your real squad: {len(kept)}/15 — {[name(p) for p in kept]}")
    print(f"Brought in only for this GW: {len(fresh)} — {[name(p) for p in fresh]}")
    print(f"Predicted XI total (sum mu): "
         f"{sum(rows.get(p, {}).get('mu', 0) for p in result['xi']):.1f}")
    print(f"Objective: {result['objective']:.2f}")
    print(f"\nBudget used: £{sum((rows[p]['sell_value'] if p in owned else rows[p]['price']) for p in result['squad'] if p in rows):.1f}m "
         f"of £{available_budget:.1f}m available")


if __name__ == "__main__":
    main()
