"""
pipeline/backtest_metrics.py
Phase 0 metric suite (docs/optimizer_redesign.md §10.3).
Post-hoc analyzer for season_simulation*.json logs. Stdlib only — runs
without pandas/models, so it works on this machine and on any log copy.

Usage:
  python pipeline/backtest_metrics.py data/intel/season_simulation.json
  python pipeline/backtest_metrics.py <log.json> \
      --history data/raw/fpl_api/player_history.csv --out metrics.json

Without --history: transfer-out counterfactuals and hit ROI are skipped
(the log only contains actuals for owned players); everything else works.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

WINDOW = 4          # GWs over which a transfer's payoff is measured
BUYBACK_GAP = 6     # re-buying within this many GWs counts as churn
SHORT_HOLD = 2      # holds of <= this many GWs count as churn

RESET_CHIPS = {"wc1", "wc2", "fh1", "fh2"}
FH_CHIPS    = {"fh1", "fh2"}
BB_CHIPS    = {"bb1", "bb2"}
TC_CHIPS    = {"tc1", "tc2"}


# ── loading ───────────────────────────────────────────────────────────────────

def load_log(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_history(path):
    """player_history.csv -> {pid: {gw: total_points}} (DGW rows summed)."""
    pts = defaultdict(lambda: defaultdict(float))
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = int(float(row["player_id"]))
            gw  = int(float(row["gameweek"]))
            pts[pid][gw] += float(row["total_points"])
    return pts


def index_log(log):
    """Build name->pid, owned-actuals, and per-GW squad sets from entries."""
    name2pid = {}
    owned = defaultdict(dict)          # pid -> {gw: actual_pts (base)}
    for e in log["gameweeks"]:
        gw = e["gw"]
        for entry in list(e.get("xi", [])) + list(e.get("bench", [])):
            pid = entry["player_id"]
            name2pid[entry["web_name"]] = pid
            owned[pid][gw] = float(entry.get("actual_pts", 0))
        cap = e.get("captain") or {}
        if cap.get("web_name") is not None and cap.get("player_id") is not None:
            name2pid[cap["web_name"]] = cap["player_id"]
    return name2pid, owned


def window_pts(pid, start_gw, source, max_gw):
    """Sum pts over [start_gw, start_gw+WINDOW-1]. Returns (pts, complete)."""
    total, missing = 0.0, False
    for g in range(start_gw, min(start_gw + WINDOW, max_gw + 1)):
        if pid in source and g in source[pid]:
            total += source[pid][g]
        else:
            missing = True
    return total, not missing


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(log, history=None):
    gws = log["gameweeks"]
    max_gw = max(e["gw"] for e in gws)
    name2pid, owned = index_log(log)
    m = {"source": {"rules_mode": log.get("rules_mode", "legacy(pre-flag)"),
                    "chip_strategy": log.get("chip_strategy", "unknown"),
                    "has_history": history is not None}}

    # ---- totals -------------------------------------------------------------
    actuals = [e["actual_total"] for e in gws]
    m["totals"] = {
        "actual": log.get("total_actual_pts", sum(actuals)),
        "predicted": round(log.get("total_predicted_pts", 0.0), 1),
        "penalties": log.get("total_penalties", 0),
        "gw_mean": round(statistics.mean(actuals), 2),
        "gw_std": round(statistics.stdev(actuals), 2) if len(actuals) > 1 else 0.0,
        "n_gws": len(gws),
    }

    # ---- transfers / churn ----------------------------------------------------
    sold_at, bought_at = {}, {}
    buybacks, short_holds = [], []
    n_transfers, hit_pts_total, hit_gws = 0, 0, []
    ft_hist = defaultdict(int)
    payoffs, payoff_partial = [], 0
    hit_roi = []

    for e in gws:
        gw, chip = e["gw"], e.get("chip")
        pen = abs(e.get("penalty_pts", 0))
        ins  = [name2pid.get(n) for n in e.get("transfers_in", [])]
        outs = [name2pid.get(n) for n in e.get("transfers_out", [])]
        ins  = [p for p in ins if p is not None]
        outs = [p for p in outs if p is not None]

        if gw == 1 or chip in FH_CHIPS:
            continue                     # GW1 squad build / FH reverts

        # churn tracking (normal + WC weeks)
        for p in ins:
            if p in sold_at and gw - sold_at[p] <= BUYBACK_GAP:
                buybacks.append((p, sold_at[p], gw))
            bought_at[p] = gw
        for p in outs:
            if p in bought_at and gw - bought_at[p] <= SHORT_HOLD:
                short_holds.append((p, bought_at[p], gw))
            sold_at[p] = gw

        if chip in RESET_CHIPS:
            continue                     # WC moves are chip value, not transfers

        n_transfers += len(ins)
        hit_pts_total += pen
        ft_hist[e.get("free_transfers", 0)] += 1

        # payoff per transfer: in-player window minus out-player window
        gw_delta, gw_complete = 0.0, True
        for p in ins:
            src = history if history else owned
            pts, complete = window_pts(p, gw, src, max_gw)
            gw_delta += pts
            gw_complete &= complete
        if history:
            for p in outs:
                pts, complete = window_pts(p, gw, history, max_gw)
                gw_delta -= pts
                gw_complete &= complete
        if ins:
            if not gw_complete:
                payoff_partial += 1
            payoffs.append({"gw": gw, "n": len(ins), "delta": round(gw_delta, 1),
                            "hit_pts": pen, "complete": gw_complete})
            if pen > 0:
                hit_gws.append(gw)
                if history:
                    hit_roi.append({"gw": gw, "roi": round(gw_delta - pen, 1)})

    deltas = [p["delta"] - p["hit_pts"] for p in payoffs]
    m["transfers"] = {
        "total": n_transfers,
        "hit_pts": hit_pts_total,
        "hit_gws": hit_gws,
        "buybacks_within_6": len(buybacks),
        "short_holds_within_2": len(short_holds),
        "ft_at_deadline_hist": dict(sorted(ft_hist.items())),
        "payoff_window_gws": WINDOW,
        "payoff_mode": "in_minus_out" if history else "in_only (need --history)",
        "payoff_per_transfer_gw": payoffs,
        "net_delta_mean": round(statistics.mean(deltas), 2) if deltas else None,
        "net_delta_positive_pct":
            round(100 * sum(1 for d in deltas if d > 0) / len(deltas), 1)
            if deltas else None,
        "partial_windows": payoff_partial,
        "hit_roi": hit_roi if history else "need --history",
    }

    # ---- captain ---------------------------------------------------------------
    regrets, cap_actuals, zero_caps = [], [], []
    for e in gws:
        cap = e.get("captain") or {}
        cap_act = float(cap.get("actual_pts", 0))
        cap_actuals.append(cap_act)
        if cap_act == 0:
            zero_caps.append(e["gw"])
        xi_best = max((float(x.get("actual_pts", 0)) for x in e.get("xi", [])),
                      default=0.0)
        regrets.append(xi_best - cap_act)
    m["captain"] = {
        "avg_actual": round(statistics.mean(cap_actuals), 2),
        "zero_gws": zero_caps,
        "zero_rate_pct": round(100 * len(zero_caps) / len(gws), 1),
        "regret_vs_best_xi_mean": round(statistics.mean(regrets), 2),
        "note": "regret uses post-auto-sub XI; vice not modelled in legacy",
    }

    # ---- chips -----------------------------------------------------------------
    chips = []
    for c in log.get("chips_used", []):
        gw, chip = c["gw"], c["chip"]
        e = next((x for x in gws if x["gw"] == gw), None)
        info = {"chip": chip, "gw": gw,
                "gw_actual": e["actual_total"] if e else None}
        if e and chip in BB_CHIPS:
            info["bench_pts_gained"] = round(sum(
                float(b.get("actual_pts", 0)) for b in e.get("bench", [])
                if not b.get("auto_subbed_out")), 1)
        if e and chip in TC_CHIPS:
            info["tc_extra_pts"] = float((e.get("captain") or {}).get("actual_pts", 0))
        chips.append(info)
    m["chips"] = chips

    # ---- bench / auto-subs -------------------------------------------------------
    bench_waste, subs_count, subs_pts = [], 0, 0.0
    for e in gws:
        ents = {x["web_name"]: float(x.get("actual_pts", 0))
                for x in list(e.get("xi", [])) + list(e.get("bench", []))}
        if e.get("chip") not in BB_CHIPS:
            bench_waste.append(sum(
                float(b.get("actual_pts", 0)) for b in e.get("bench", [])
                if not b.get("auto_subbed_out")))
        for pair in e.get("auto_subs", []):
            subs_count += 1
            if len(pair) == 2:
                subs_pts += ents.get(pair[1], 0.0)
    m["bench"] = {
        "avg_pts_wasted_per_gw": round(statistics.mean(bench_waste), 2)
                                 if bench_waste else 0.0,
        "auto_subs": subs_count,
        "auto_sub_pts_rescued": round(subs_pts, 1),
    }

    # ---- prediction quality ---------------------------------------------------
    maes = []
    for e in gws:
        errs = [abs(float(x.get("predicted_pts", 0)) - float(x.get("actual_pts", 0)))
                for x in e.get("xi", [])]
        if errs:
            maes.append(statistics.mean(errs))
    m["prediction"] = {
        "xi_mae_mean": round(statistics.mean(maes), 3) if maes else None,
        "caveat": "legacy predicted_pts include loyalty/bench-bonus inflation",
    }
    return m


# ── report ────────────────────────────────────────────────────────────────────

def print_report(m, path):
    src, t, tr, cap = m["source"], m["totals"], m["transfers"], m["captain"]
    print("=" * 72)
    print(f"  Phase 0 metrics — {os.path.basename(path)}")
    print(f"  rules_mode={src['rules_mode']} | chips={src['chip_strategy']}"
          f" | history={'yes' if src['has_history'] else 'NO (partial metrics)'}")
    print("=" * 72)
    print(f"  Points   : {t['actual']} actual | {t['predicted']} predicted "
          f"| {t['penalties']} penalties | {t['gw_mean']}±{t['gw_std']}/GW "
          f"({t['n_gws']} GWs)")
    print(f"  Transfers: {tr['total']} total, {tr['hit_pts']} hit pts "
          f"(GWs {tr['hit_gws'] or '—'})")
    print(f"             buybacks<= {BUYBACK_GAP}gw: {tr['buybacks_within_6']} | "
          f"holds<={SHORT_HOLD}gw: {tr['short_holds_within_2']}")
    print(f"             FT-at-deadline histogram: {tr['ft_at_deadline_hist']}")
    print(f"             {WINDOW}GW payoff [{tr['payoff_mode']}]: "
          f"mean net {tr['net_delta_mean']} | "
          f"{tr['net_delta_positive_pct']}% positive | "
          f"{tr['partial_windows']} partial windows")
    if isinstance(tr["hit_roi"], list) and tr["hit_roi"]:
        rois = [h["roi"] for h in tr["hit_roi"]]
        print(f"             hit ROI (gain−4/hit): {rois} "
              f"(mean {round(statistics.mean(rois), 1)})")
    print(f"  Captain  : avg {cap['avg_actual']} | zero-return GWs "
          f"{cap['zero_gws'] or '—'} ({cap['zero_rate_pct']}%) | "
          f"regret vs best-XI {cap['regret_vs_best_xi_mean']}/GW")
    print("  Chips    :")
    for c in m["chips"]:
        extra = {k: v for k, v in c.items() if k not in ("chip", "gw", "gw_actual")}
        print(f"             {c['chip']} @ GW{c['gw']} (GW total {c['gw_actual']})"
              f"{'  ' + str(extra) if extra else ''}")
    b = m["bench"]
    print(f"  Bench    : {b['avg_pts_wasted_per_gw']} pts/GW wasted | "
          f"{b['auto_subs']} auto-subs rescued {b['auto_sub_pts_rescued']} pts")
    print(f"  Predict  : XI MAE {m['prediction']['xi_mae_mean']} "
          f"({m['prediction']['caveat']})")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description="Phase 0 backtest metrics")
    ap.add_argument("log", help="season_simulation*.json path")
    ap.add_argument("--history", help="player_history.csv for full metrics")
    ap.add_argument("--out", help="write metrics JSON here")
    args = ap.parse_args()

    log = load_log(args.log)
    history = load_history(args.history) if args.history else None
    m = compute_metrics(log, history)
    print_report(m, args.log)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        print(f"  metrics saved -> {args.out}")


if __name__ == "__main__":
    main()
