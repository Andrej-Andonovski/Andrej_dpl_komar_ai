"""
pipeline/gw_recommendation.py
Standalone weekly transfer/captain recommendation for a REAL owned squad.

Does NOT hack season_simulator.run_simulation()'s full season loop (too much
season-long internal state — sold_at rebuy-lock, hits_paid budget, chip
ledgers — to safely fake mid-season). Instead calls the same building-block
functions run_simulation() calls, for a single target GW:

  load_player_history()      -- real actuals via FPL API
  build_hist_rows()          -- 7-season historical training rows
  build_retrain_rows()       -- current-season actuals as retrain rows
  train_models()             -- LightGBM per position (RULES_MODE=corrected
                                 requires OPTIMIZER=mp -> season_simulator
                                 enforces this)
  build_rolling_pool()       -- feature pool for the target GW
  prediction_matrix.build_matrix() + milp_core.solve_horizon()
                              -- MP_HORIZON-week MILP transfer/captain plan

Squad / bank / free-transfers are pulled live from the public FPL API for
FPL_TEAM_ID (no login needed for these fields) rather than assumed, per the
project's free-transfer-accrual bug history (fpl_rules.next_free_transfers).

Known approximation: no purchase-price ledger (would need an authenticated
my-team call). Owned players are priced at CURRENT MARKET price for the
sell-value/budget calc, which is the safe fallback `fpl_rules.sell_value`
itself falls back to. This slightly UNDERSTATES available budget for anyone
who has risen in price (real sell = purchase + 50% of any rise), so a
transfer that looks marginally unaffordable here may be feasible in-game.

Usage: python pipeline/gw_recommendation.py [--horizon N]
"""
import os
import sys
import json
import argparse

os.environ.setdefault("RULES_MODE", "corrected")
os.environ.setdefault("OPTIMIZER", "mp")
os.environ.setdefault("MP_HORIZON", "5")
os.environ.setdefault("OWN_PRIOR_GW1", "0.213")

import requests
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # so "pipeline.xxx" imports resolve

from pipeline import season_simulator as ss
from pipeline import prediction_matrix as pmx
from pipeline.milp_core import solve_horizon as milp_solve_horizon
from pipeline.milp_core import MAX_FIXTURE_EXPOSURE as _DEFAULT_FIXTURE_CAP
from pipeline.fpl_rules import next_free_transfers

FPL_TEAM_ID = int(os.environ.get("FPL_TEAM_ID", "7426041"))
POS_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# ── Live squad / bank / free-transfers from the public FPL API ────────────────

def fetch_team_state(team_id):
    boot = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        timeout=30).json()
    events = boot["events"]
    next_ev = next((e for e in events if e.get("is_next")), None)
    if next_ev is None:
        # season over / between seasons: fall back to current
        cur_ev = next(e for e in events if e.get("is_current"))
        target_gw = cur_ev["id"]
    else:
        target_gw = next_ev["id"]
    last_completed = target_gw - 1

    hist = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/history/",
        timeout=30).json()
    chip_gws = {c["event"]: c["name"] for c in hist.get("chips", [])}

    # Walk FT accrual from GW1 -> entering target_gw, using the SAME rule
    # function the project's corrected-mode simulator uses (no re-derivation).
    ft = 1
    for row in sorted(hist["current"], key=lambda r: r["event"]):
        gw = row["event"]
        if gw >= target_gw:
            break
        chip = chip_gws.get(gw)
        is_wc = chip in ("wildcard",)
        is_fh = chip in ("freehit", "fh2")
        ft = next_free_transfers(gw, ft, row["event_transfers"],
                                 is_wc, is_fh, ft_cap=5,
                                 ft_events=ss.RULE_EVENTS_FT)

    picks = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}"
        f"/event/{last_completed}/picks/", timeout=30).json()
    squad_ids = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10.0
    prev_captain = next((p["element"] for p in picks["picks"]
                         if p["is_captain"]), None)

    return {
        "target_gw": target_gw,
        "completed_gw": last_completed,
        "squad_ids": squad_ids,
        "bank": bank,
        "free_transfers": ft,
        "prev_captain": prev_captain,
    }


def build_fixture_pairs(fixtures_csv, from_gw, horizon):
    """{gw: [(team_h, team_a), ...]} for gw in [from_gw, from_gw+horizon-1] --
    feeds milp_core's same-fixture exposure cap (see milp_core.MAX_FIXTURE_EXPOSURE)."""
    df = pd.read_csv(fixtures_csv)
    df = df.dropna(subset=["gameweek"])
    df["gameweek"] = df["gameweek"].astype(int)
    df = df[(df.gameweek >= from_gw) & (df.gameweek < from_gw + horizon)]
    pairs = {}
    for r in df.itertuples(index=False):
        pairs.setdefault(int(r.gameweek), []).append((int(r.team_h), int(r.team_a)))
    return pairs


def build_name_lookup(players_df):
    return {int(r.id): {
        "web_name": r.web_name,
        "team": int(r.team),
        "element_type": int(r.element_type),
        "price": float(r.price),
    } for r in players_df.itertuples(index=False)}


def team_short_names():
    boot = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        timeout=30).json()
    return {t["id"]: t["short_name"] for t in boot["teams"]}


# ── Fixture sanity check for any recommended transfer ──────────────────────

def upcoming_fixtures(fixtures_csv, team_id, from_gw, n=5):
    df = pd.read_csv(fixtures_csv)
    df = df.dropna(subset=["gameweek"])
    df["gameweek"] = df["gameweek"].astype(int)
    rows = []
    for r in df[df["gameweek"] >= from_gw].sort_values("gameweek").itertuples():
        if r.team_h == team_id:
            rows.append((r.gameweek, r.team_a, r.team_h_difficulty, True))
        elif r.team_a == team_id:
            rows.append((r.gameweek, r.team_h, r.team_a_difficulty, False))
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=None,
                    help="override MP_HORIZON")
    ap.add_argument("--max-fixture-exposure", type=int,
                    default=int(os.environ.get("MAX_FIXTURE_EXPOSURE",
                                               _DEFAULT_FIXTURE_CAP)),
                    help="max combined XI players from both clubs of a "
                         "single fixture (caps same-match concentration "
                         "risk); 0 disables the cap")
    args = ap.parse_args()
    if args.horizon:
        ss.MP_HORIZON = args.horizon

    print("=" * 70)
    print(f"  GW RECOMMENDATION — team {FPL_TEAM_ID}")
    print(f"  RULES_MODE={ss.RULES_MODE} OPTIMIZER={ss.OPTIMIZER} "
          f"MP_HORIZON={ss.MP_HORIZON}")
    print("=" * 70)

    print("\n[FPL] Fetching live team state...")
    state = fetch_team_state(FPL_TEAM_ID)
    target_gw = state["target_gw"]
    completed_gw = state["completed_gw"]
    print(f"  Target GW: {target_gw}  (actuals through GW{completed_gw})")
    print(f"  Bank: £{state['bank']:.1f}m  |  Free transfers: "
          f"{state['free_transfers']}")
    print(f"  Squad ({len(state['squad_ids'])}): {state['squad_ids']}")

    print("\n[LOAD] Loading data...")
    hist_lookup = ss.load_player_history()
    players_df = ss.load_players_raw()
    fdr_lookup, home_lookup, dgw_gws, gw_teams = ss.load_fixtures()
    fixture_list = pmx.load_fixture_list(ss.FIXTURES_CSV)
    train_dfs = ss.load_training_data()
    avail_gws = ss.load_availability()

    name_lookup = build_name_lookup(players_df)
    team_names = team_short_names()

    missing = [pid for pid in state["squad_ids"] if pid not in name_lookup]
    if missing:
        raise RuntimeError(f"Owned player ids missing from players_raw.csv "
                           f"(stale data?): {missing}")

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
    cs_weight = 1 + completed_gw   # matches season_simulator's live weighting
    combined_rows, combined_weights = {}, {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        h_rows = hist_rows_for_retrain.get(pos, [])
        r_rows = retrain_rows.get(pos, [])
        combined_rows[pos] = h_rows + r_rows
        combined_weights[pos] = [1.0] * len(h_rows) + [float(cs_weight)] * len(r_rows)
    models = ss.train_models(combined_rows, combined_weights)

    print(f"\n[POOL] Building GW{target_gw} pool from GW1-{completed_gw} actuals...")
    pool = ss.build_rolling_pool(players_df, hist_lookup, fdr_lookup,
                                 home_lookup, completed_gw,
                                 team_form_lookup=team_form_lookup,
                                 opp_lookup=opp_lookup)

    print(f"\n[MATRIX] Building {ss.MP_HORIZON}-week prediction matrix "
          f"from GW{target_gw}...")
    mp_matrix = pmx.build_matrix(pool, models, fixture_list, target_gw,
                                 ss.MP_HORIZON, hist_lookup=hist_lookup,
                                 avail_gws=avail_gws, purchase_price=None)

    fixture_pairs = (build_fixture_pairs(ss.FIXTURES_CSV, target_gw, ss.MP_HORIZON)
                     if args.max_fixture_exposure > 0 else None)
    if fixture_pairs:
        print(f"  [FIXTURE-CAP] max {args.max_fixture_exposure} XI players "
             f"from either club of a single fixture")

    print(f"\n[MILP] Solving {ss.MP_HORIZON}-week horizon for GW{target_gw}...")
    owned = set(state["squad_ids"])
    plan = milp_solve_horizon(
        mp_matrix, owned, state["bank"], state["free_transfers"], target_gw,
        ft_events=ss.RULE_EVENTS_FT, chip_state=None,
        theta=ss.MP_THETA, delta=ss.MP_DELTA, delta_chip=ss.MP_DELTA_CHIP,
        gamma=ss.MP_GAMMA, w_bench=ss.MP_W_BENCH,
        hit_cap=ss.MP_HIT_CAP, hit_cost=ss.MP_HIT_COST,
        ft_value=ss.MP_FT_VALUE, w_bench_slots=ss.MP_BENCH_SLOTS,
        w_bench_gk=ss.MP_W_BENCH_GK,
        fixture_pairs=fixture_pairs,
        max_fixture_exposure=args.max_fixture_exposure)

    result = plan["weeks"][target_gw]

    def name(pid):
        p = name_lookup.get(pid, {})
        team = team_names.get(p.get("team"), "?")
        return f"{p.get('web_name', pid)} ({team})"

    print("\n" + "=" * 70)
    print(f"  RESULT — GW{target_gw}")
    print("=" * 70)
    print(f"\nXI:")
    for pid in result["xi"]:
        tag = ""
        if pid == result["captain"]:
            tag = " [C]"
        elif pid == result["vice"]:
            tag = " [VC]"
        row = mp_matrix[target_gw].get(pid, {})
        print(f"  {name(pid):32s} mu={row.get('mu', 0):.2f}{tag}")
    print(f"\nBench:")
    for pid in result["bench"]:
        row = mp_matrix[target_gw].get(pid, {})
        print(f"  {name(pid):32s} mu={row.get('mu', 0):.2f}")

    print(f"\nTransfers IN:  {[name(p) for p in result['transfers_in']]}")
    print(f"Transfers OUT: {[name(p) for p in result['transfers_out']]}")
    print(f"Hits: {result['hits']}  (-{4 * result['hits']} pts)")
    print(f"Objective: {plan['objective']:.2f}")

    # ── Fixture sanity check on any recommended transfer ────────────────────
    if result["transfers_out"]:
        print("\n" + "-" * 70)
        print("  FIXTURE SANITY CHECK on transfers OUT")
        print("-" * 70)
        for pid in result["transfers_out"]:
            p = name_lookup.get(pid, {})
            recent_pts = [hist_lookup.get(pid, {}).get(g, {}).get("total_points")
                         for g in range(max(1, completed_gw - 2), completed_gw + 1)]
            recent_pts = [x for x in recent_pts if x is not None]
            fx = upcoming_fixtures(ss.FIXTURES_CSV, p.get("team"), target_gw, n=3)
            fx_str = ", ".join(
                f"GW{g} vs {team_names.get(opp,'?')}"
                f"{'(H)' if home else '(A)'} FDR{fdr}"
                for g, opp, fdr, home in fx)
            avg_fdr = (sum(f[2] for f in fx) / len(fx)) if fx else None
            flag = ""
            if recent_pts and sum(recent_pts) / len(recent_pts) >= 6.0 \
                    and avg_fdr is not None and avg_fdr <= 2.5:
                flag = "  <-- FLAG: good recent form + easy fixtures, scrutinize this sell"
            print(f"  {name(pid)}: recent pts {recent_pts}, next fixtures: "
                  f"{fx_str}{flag}")

    out = {
        "target_gw": target_gw,
        "xi": result["xi"], "bench": result["bench"],
        "captain": result["captain"], "vice": result["vice"],
        "transfers_in": result["transfers_in"],
        "transfers_out": result["transfers_out"],
        "hits": result["hits"],
        "bank": state["bank"], "free_transfers": state["free_transfers"],
    }
    out_path = os.path.join(ss.DATA_DIR, "intel", "gw_recommendation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
