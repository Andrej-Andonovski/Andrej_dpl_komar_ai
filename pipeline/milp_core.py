"""
pipeline/milp_core.py
Phases 2-4 of the optimizer redesign (docs/optimizer_redesign.md §4).

solve_gw      — single-GW MILP (Phase 2; also the FH-week and fallback solve)
solve_horizon — multi-period MILP (Phase 3) with chips as variables (Phase 4)

Objective per week g (discounted delta^(g-t)):
  Σ μ·s + Σ κ·c + γ·Σ μ·v + Σ β·(x−s) − 4·hits
  κ = π·[(1−θ)μ + θ·q90]   (captain: play-prob-weighted ceiling blend)
  β = w̄·π·μ                (bench auto-sub EV)
Chip bonuses (BB/TC aux) are discounted at delta_chip > delta: chip value on
event weeks is calendar-driven (known), so the usual forecast discount would
create an early-burn bias (§4.8).

Correct rules: owned players priced at SELL value (§5 identity), FT banking
1..5 with the no-phantom-hit ≤-recursion (§4.6 proof), infeasible ⇒ raise.
Solver: HiGHS if available, CBC fallback.
"""
from collections import defaultdict

from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                  LpBinary, LpInteger, LpStatus, PULP_CBC_CMD)

# ── The honest constants (blueprint §12) ─────────────────────────────────────
THETA   = 0.5    # captain mean<->ceiling blend
GAMMA   = 0.07   # captain-miss probability for the vice term
W_BENCH = 0.15   # bench-slot activation prob (measured: Phase 0 metrics
                 # showed 18-24 auto-subs/season over 4 slots ≈ 0.12-0.16)
HIT_CAP = 2      # max paid hits per GW (rollout guardrail, blueprint §12)
HIT_COST = 4.0   # objective price per hit — the DECISION threshold only.
                 # Scoring always subtracts the real -4; raising this above 4
                 # demands a margin over the paper gain (predicted upgrade mu
                 # is upward-biased by selection — Phase 4 measured realized
                 # hit ROI at -6.8). Env: MP_HIT_COST via season_simulator.
FT_VALUE = 0.0   # objective price per EXECUTED transfer (even FT-funded) —
                 # transfer friction. A free transfer is not free: banking
                 # has option value and sub-noise mu edges trigger sideways
                 # churn (sell a hauler for a +0.2/wk paper edge). The move
                 # must beat holding by this margin over the horizon. The
                 # honest version of legacy's tuned loyalty bonus. WC/FH/GW1
                 # rebuilds exempt. Env: MP_FT_VALUE via season_simulator.
N_CAP_CANDIDATES = 60   # captain/vice variables restricted to top-N by kappa

# ── Phase 3: horizon constants ────────────────────────────────────────────────
DELTA = 0.94     # per-week discount. Phase 1 measured decay supports
                 # 0.90-0.97 (MAE +4.1%/5wk, Spearman -2.5%/wk); midpoint
                 # pre-Phase-6 default, final value from the H/delta ablation.
PRUNE_POS_N = {1: 12, 2: 40, 3: 45, 4: 25}   # top-N per position by max_g mu
PRUNE_VALUE_N = 15                            # plus top-N by mu/price
FT_MAX = 5

# ── Phase 4: chip constants ───────────────────────────────────────────────────
DELTA_CHIP  = 0.97   # chip-term discount (§4.8: calendar-known value)
SPACING_GAP = 4      # min GWs between any two reset chips (WC/FH)
M_BB, M_TC, M_Z = 60.0, 30.0, 200.0   # big-M for aux linearizations

POS_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}          # squad by element_type
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}             # formation minima
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}             # formation maxima
MAX_CLUB = 3

_solver_logged = False


def _get_solver(time_limit=120):
    """HiGHS when available (Phase 3+ scale needs it), else CBC."""
    global _solver_logged
    try:
        from pulp import HiGHS
        candidate = HiGHS(msg=False, timeLimit=time_limit)
        if candidate.available():
            s = candidate
            name = "HiGHS"
        else:
            raise RuntimeError("HiGHS executable is unavailable")
    except Exception:
        s = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        name = "CBC"
    if not _solver_logged:
        print(f"  [MILP] solver: {name}")
        _solver_logged = True
    return s


def kappa(row, theta=THETA):
    """Captain coefficient: play-prob-weighted mean/ceiling blend."""
    return row["pi"] * ((1.0 - theta) * row["mu"] + theta * row["q90"])


def solve_gw(rows, owned, available_budget, free_transfers, gw,
             is_wildcard=False, is_freehit=False,
             theta=THETA, gamma=GAMMA, w_bench=W_BENCH, hit_cap=HIT_CAP,
             hit_cost=HIT_COST, ft_value=FT_VALUE, no_rebuy=None,
             sell_hold=None, time_limit=120):
    """
    rows  : {pid: matrix row} for ONE gameweek — needs mu, pi, q90, price,
            sell_value, element_type, team (prediction_matrix output).
    owned : set of pids currently in the squad (empty at GW1).
    available_budget : bank + Σ sell_value(owned)  (corrected accounting).

    Returns {squad, xi, bench, captain, vice, transfers_in, transfers_out,
             hits, objective} — pids everywhere. Raises on infeasibility.
    """
    pids = [pid for pid, r in rows.items() if r.get("price", 0) > 0]
    r_of = {pid: rows[pid] for pid in pids}
    missing = set(owned) - set(pids)
    if missing and not (is_wildcard or is_freehit or gw == 1):
        raise RuntimeError(f"GW{gw}: owned players missing from matrix: "
                           f"{sorted(missing)}")

    no_transfers = is_wildcard or is_freehit or gw == 1 or not owned

    # cost: owned at sell value, everyone else at market (§5 identity)
    cost = {pid: (r_of[pid]["sell_value"] if pid in owned else
                  r_of[pid]["price"]) for pid in pids}
    kap = {pid: kappa(r_of[pid], theta) for pid in pids}
    cap_cands = set(sorted(pids, key=lambda p: -kap[p])[:N_CAP_CANDIDATES])

    prob = LpProblem(f"mp_gw{gw}", LpMaximize)
    x = {p: LpVariable(f"x{p}", cat=LpBinary) for p in pids}
    s = {p: LpVariable(f"s{p}", cat=LpBinary) for p in pids}
    c = {p: LpVariable(f"c{p}", cat=LpBinary) for p in cap_cands}
    v = {p: LpVariable(f"v{p}", cat=LpBinary) for p in cap_cands}

    if not no_transfers:
        ti = {p: LpVariable(f"i{p}", cat=LpBinary) for p in pids}
        to = {p: LpVariable(f"o{p}", cat=LpBinary) for p in pids}
        hits = LpVariable("hits", lowBound=0, upBound=hit_cap, cat=LpInteger)
        # cross-solve rebuy lock: recently-sold players stay out (WC/FH
        # weeks never reach here — no_transfers rebuilds are exempt)
        for p in (no_rebuy or ()):
            if p in pids and p not in owned:
                prob += ti[p] == 0
    else:
        ti = to = hits = None

    mu = {p: r_of[p]["mu"] for p in pids}
    beta = {p: w_bench * r_of[p]["pi"] * r_of[p]["mu"] for p in pids}

    obj = (lpSum(mu[p] * s[p] for p in pids)
           + lpSum(kap[p] * c[p] for p in cap_cands)
           + gamma * lpSum(mu[p] * v[p] for p in cap_cands)
           + lpSum(beta[p] * (x[p] - s[p]) for p in pids))
    if hits is not None:
        obj -= hit_cost * hits
        if ft_value > 0:
            obj -= ft_value * lpSum(ti.values())
        # form hold: selling last week's haulers costs extra (soft) —
        # the solver prefers selling someone else or holding the FT
        for p, pen in (sell_hold or {}).items():
            if p in pids and p in owned:
                obj -= pen * to[p]
    prob += obj

    prob += lpSum(x.values()) == 15
    for et, quota in POS_QUOTA.items():
        prob += lpSum(x[p] for p in pids
                      if r_of[p]["element_type"] == et) == quota
    clubs = {r_of[p]["team"] for p in pids}
    for cl in clubs:
        prob += lpSum(x[p] for p in pids if r_of[p]["team"] == cl) <= MAX_CLUB

    prob += lpSum(cost[p] * x[p] for p in pids) <= available_budget + 1e-6

    prob += lpSum(s.values()) == 11
    for et in POS_QUOTA:
        n_et = lpSum(s[p] for p in pids if r_of[p]["element_type"] == et)
        prob += n_et >= XI_MIN[et]
        prob += n_et <= XI_MAX[et]
    for p in pids:
        prob += s[p] <= x[p]

    prob += lpSum(c.values()) == 1
    prob += lpSum(v.values()) == 1
    for p in cap_cands:
        prob += c[p] <= s[p]
        prob += v[p] <= s[p]
        prob += c[p] + v[p] <= 1

    if not no_transfers:
        for p in pids:
            prob += x[p] == (1 if p in owned else 0) + ti[p] - to[p]
            prob += ti[p] + to[p] <= 1
        prob += lpSum(ti.values()) == lpSum(to.values())
        prob += hits >= lpSum(ti.values()) - free_transfers

    prob.solve(_get_solver(time_limit))
    if LpStatus[prob.status] != "Optimal":
        raise RuntimeError(
            f"MILP {LpStatus[prob.status]} at GW{gw} "
            f"(budget {available_budget:.1f}m, owned {len(owned)}) — "
            "corrected mode never relaxes; investigate inputs")

    def on(var):
        return var.value() is not None and var.value() > 0.5

    squad = [p for p in pids if on(x[p])]
    xi = [p for p in squad if on(s[p])]
    captain = next((p for p in cap_cands if on(c[p])), None)
    vice = next((p for p in cap_cands if on(v[p])), None)
    if not no_transfers:
        t_in = [p for p in pids if on(ti[p])]
        t_out = [p for p in pids if on(to[p])]
        n_hits = int(round(hits.value() or 0))
    else:
        t_in = [p for p in squad if p not in owned] if gw > 1 else squad
        t_out = [p for p in owned if p not in squad] if gw > 1 else []
        n_hits = 0

    return {
        "squad": squad, "xi": xi,
        "bench": [p for p in squad if p not in set(xi)],
        "captain": captain, "vice": vice,
        "transfers_in": t_in, "transfers_out": t_out, "hits": n_hits,
        "objective": float(prob.objective.value() or 0.0),
    }


# ── Phase 3/4: multi-period program ───────────────────────────────────────────

def prune_pool(matrix, owned):
    """
    Blueprint §7.2: owned squad + top-N per position by best horizon mu +
    top-N by value (mu/price). Returns the kept pid set. A monthly unpruned
    sanity solve is the correctness valve (§7.2) — not done per-GW.
    """
    gws = sorted(matrix)
    t = gws[0]
    pids = [p for p, r in matrix[t].items() if r.get("price", 0) > 0]
    best_mu = {p: max(matrix[g][p]["mu"] for g in gws if p in matrix[g])
               for p in pids}
    keep = set(owned)
    by_pos = defaultdict(list)
    for p in pids:
        by_pos[matrix[t][p]["element_type"]].append(p)
    for et, lst in by_pos.items():
        lst.sort(key=lambda p: -best_mu[p])
        keep.update(lst[:PRUNE_POS_N.get(et, 30)])
    by_value = sorted(pids, key=lambda p: -best_mu[p] / matrix[t][p]["price"])
    keep.update(by_value[:PRUNE_VALUE_N])
    return keep


def _set_of(g):
    return 1 if g <= 19 else 2


def solve_horizon(matrix, owned, bank, free_transfers, t,
                  is_wildcard_now=False, ft_events=None, chip_state=None,
                  delta=DELTA, delta_chip=DELTA_CHIP, theta=THETA,
                  gamma=GAMMA, w_bench=W_BENCH, hit_cap=HIT_CAP,
                  hit_cost=HIT_COST, ft_value=FT_VALUE, no_rebuy=None,
                  sell_hold=None, time_limit=240):
    """
    Multi-period MILP over the matrix weeks {t..t+H-1}: per-week squad/XI/
    captain/vice + per-week transfers with FT banking, churn guard (max one
    in and one out per player per horizon), bank recursion with exact sell
    values — and, when chip_state is given (Phase 4), chips as variables:

      WC_g  — waives week-g hit costs (unlimited rebuild); K3 spacing
      FH_g  — event weeks only; a full shadow squad scores that week while
              the real squad is frozen (F1-F6); one-week gain is emergent
      BB_g  — aux bonus = bench mu that week (K4); reservation guard holds
              it when a known DGW lies beyond the horizon in the same set
      TC_g  — aux bonus = one extra captain kappa copy (K5)
      one chip per GW; per-set availability from the ledger; WC/FH spacing
      vs each other and vs already-played reset chips.

    chip_state = {"used": {"wc1", ...}, "reset_gws": [gw, ...],
                  "far_dgw": {1: bool, 2: bool}}  (None = Phase 3, no chips)

    Only week t is executed; later weeks are the plan (rolling re-solve).
    Returns {"weeks": {g: solve_gw-shaped dict}, "chips": {g: "bb2"...},
             "ft_plan", "bank_plan", "objective"}. Raises on infeasibility.
    """
    ft_events = ft_events or {}
    gws = sorted(g for g in matrix if g >= t)
    if not gws or gws[0] != t:
        raise RuntimeError(f"horizon matrix must start at t={t}, got {gws[:1]}")

    missing = set(owned) - set(matrix[t])
    if missing:
        raise RuntimeError(f"GW{t}: owned players missing from matrix: "
                           f"{sorted(missing)}")

    keep = prune_pool(matrix, owned)
    pids = [p for p, r in matrix[t].items()
            if p in keep and r.get("price", 0) > 0]
    initial_build = not owned

    # prices assumed constant within the horizon (blueprint §8 R6)
    mp_ = {p: matrix[t][p]["price"] for p in pids}
    sv_ = {p: (matrix[t][p]["sell_value"] if p in owned else mp_[p])
           for p in pids}
    et_ = {p: matrix[t][p]["element_type"] for p in pids}
    cl_ = {p: matrix[t][p]["team"] for p in pids}

    def row(p, g):
        return matrix[g][p]

    prob = LpProblem(f"mp_horizon_gw{t}", LpMaximize)
    x, s, ti, to = {}, {}, {}, {}
    for g in gws:
        for p in pids:
            x[p, g] = LpVariable(f"x{p}_{g}", cat=LpBinary)
            s[p, g] = LpVariable(f"s{p}_{g}", cat=LpBinary)
            ti[p, g] = LpVariable(f"i{p}_{g}", cat=LpBinary)
            to[p, g] = LpVariable(f"o{p}_{g}", cat=LpBinary)

    cap_cands = {g: set(sorted(pids, key=lambda p: -kappa(row(p, g), theta)
                               )[:N_CAP_CANDIDATES]) for g in gws}
    c = {(p, g): LpVariable(f"c{p}_{g}", cat=LpBinary)
         for g in gws for p in cap_cands[g]}
    v = {(p, g): LpVariable(f"v{p}_{g}", cat=LpBinary)
         for g in gws for p in cap_cands[g]}

    h = {g: LpVariable(f"h{g}", lowBound=0, upBound=hit_cap) for g in gws}
    # transfer-friction counter: executed transfers net of WC/GW1 waivers
    fr = {g: LpVariable(f"fr{g}", lowBound=0) for g in gws}
    bankv = {g: LpVariable(f"bank{g}", lowBound=0) for g in gws}
    fvar = {g: LpVariable(f"f{g}", lowBound=1, upBound=FT_MAX, cat=LpInteger)
            for g in gws[1:]}

    def ft_expr(g):
        return free_transfers if g == t else fvar[g]

    # ── Phase 4: chip variables + eligibility (K1-K3 + reservation guard) ────
    chips_on = chip_state is not None
    used = (chip_state or {}).get("used", set())
    reset_played = (chip_state or {}).get("reset_gws", [])
    far_dgw = (chip_state or {}).get("far_dgw", {})
    # Chip scarcity guards (phase4_report.md fix — measurable, not tuned):
    lockout = (chip_state or {}).get("lockout_until", 0)   # no chips <= this GW
    wc_ok = (chip_state or {}).get("wc_ok", True)          # squad-state gate
    blocked_now = set((chip_state or {}).get("blocked_now", set()))

    def is_event(g):
        return any(r["n_fix"] != 1 for r in matrix[g].values())

    def has_double(g):
        return any(r["n_fix"] >= 2 for r in matrix[g].values())

    CH = {"wc": {}, "fh": {}, "bb": {}, "tc": {}}
    if chips_on:
        for g in gws:
            if g <= lockout:
                continue            # cold-start lockout: never chip early
            sid = _set_of(g)
            for k in ("wc", "fh", "bb", "tc"):
                if g == t and k in blocked_now:
                    continue        # rejected by the percentile bar; re-solve
                if f"{k}{sid}" in used:
                    continue
                if k == "fh" and not is_event(g):
                    continue        # never burn FH on a plain week
                if k in ("bb", "tc") and far_dgw.get(sid) and not has_double(g):
                    continue        # reservation guard: hold BB/TC for the
                                    # known far double (disarms automatically
                                    # when the set deadline enters the horizon)
                if k == "wc" and not wc_ok:
                    continue        # WC only when the squad measurably needs
                                    # a rebuild, or near the set deadline
                if k in ("wc", "fh") and any(abs(g - r) < SPACING_GAP
                                             for r in reset_played):
                    continue        # spacing vs already-played resets
                CH[k][g] = LpVariable(f"{k}_{g}", cat=LpBinary)
        # K1: one per chip per set
        for k, d in CH.items():
            for sid in (1, 2):
                in_set = [var for g, var in d.items() if _set_of(g) == sid]
                if in_set:
                    prob += lpSum(in_set) <= 1
        # K2: one chip per GW
        for g in gws:
            here = [d[g] for d in CH.values() if g in d]
            if len(here) > 1:
                prob += lpSum(here) <= 1
        # K3: in-horizon reset spacing (wc-wc, fh-fh, wc-fh pairs)
        resets = [(g, var) for k in ("wc", "fh") for g, var in CH[k].items()]
        for i in range(len(resets)):
            for j in range(i + 1, len(resets)):
                ga, va = resets[i]
                gb, vb = resets[j]
                if ga != gb and abs(ga - gb) < SPACING_GAP:
                    prob += va + vb <= 1

    fh_cands = sorted(CH["fh"]) if chips_on else []

    # ── FH shadow squads (F1-F6) ──────────────────────────────────────────────
    xs, ss, cs, vs, ks = {}, {}, {}, {}, {}
    for g in fh_cands:
        FH = CH["fh"][g]
        for p in pids:
            xs[p, g] = LpVariable(f"xs{p}_{g}", cat=LpBinary)
            ss[p, g] = LpVariable(f"ss{p}_{g}", cat=LpBinary)
            ks[p, g] = LpVariable(f"ks{p}_{g}", lowBound=0, upBound=1)
        for p in cap_cands[g]:
            cs[p, g] = LpVariable(f"cs{p}_{g}", cat=LpBinary)
            vs[p, g] = LpVariable(f"vs{p}_{g}", cat=LpBinary)

        prob += lpSum(xs[p, g] for p in pids) == 15 * FH
        for et, quota in POS_QUOTA.items():
            prob += lpSum(xs[p, g] for p in pids
                          if et_[p] == et) == quota * FH
        for cl in set(cl_.values()):
            prob += lpSum(xs[p, g] for p in pids if cl_[p] == cl) <= MAX_CLUB

        prob += lpSum(ss[p, g] for p in pids) == 11 * FH
        for et in POS_QUOTA:
            n_et = lpSum(ss[p, g] for p in pids if et_[p] == et)
            prob += n_et >= XI_MIN[et] * FH
            prob += n_et <= XI_MAX[et]
        for p in pids:
            prob += ss[p, g] <= xs[p, g]

        prob += lpSum(cs[p, g] for p in cap_cands[g]) == FH
        prob += lpSum(vs[p, g] for p in cap_cands[g]) == FH
        for p in cap_cands[g]:
            prob += cs[p, g] <= ss[p, g]
            prob += vs[p, g] <= ss[p, g]
            prob += cs[p, g] + vs[p, g] <= 1

        # F5 budget with keep-linearization (sv<=mp ⇒ solver maxes keeps)
        gi = gws.index(g)
        prev_bank = bank if g == t else bankv[gws[gi - 1]]
        for p in pids:
            prob += ks[p, g] <= xs[p, g]
            prev_x = ((1 if p in owned else 0) if g == t
                      else x[p, gws[gi - 1]])
            prob += ks[p, g] <= prev_x
        prev_val = (lpSum(sv_[p] * ((1 if p in owned else 0) if g == t
                                    else x[p, gws[gi - 1]]) for p in pids))
        prob += (lpSum(mp_[p] * xs[p, g] for p in pids)
                 - lpSum((mp_[p] - sv_[p]) * ks[p, g] for p in pids)
                 <= prev_bank + prev_val)

        # F4: no permanent transfers on an FH week
        for p in pids:
            prob += ti[p, g] + FH <= 1
            prob += to[p, g] + FH <= 1

    # ── objective ─────────────────────────────────────────────────────────────
    terms = []
    for g in gws:
        w = delta ** (g - t)
        W = (lpSum(row(p, g)["mu"] * s[p, g] for p in pids)
             + lpSum(kappa(row(p, g), theta) * c[p, g] for p in cap_cands[g])
             + gamma * lpSum(row(p, g)["mu"] * v[p, g] for p in cap_cands[g])
             + lpSum(w_bench * row(p, g)["pi"] * row(p, g)["mu"]
                     * (x[p, g] - s[p, g]) for p in pids))
        if g in fh_cands:
            FH = CH["fh"][g]
            z = LpVariable(f"z_{g}", lowBound=0)
            zt = LpVariable(f"zt_{g}", lowBound=0)
            prob += z <= W
            prob += z <= M_Z * (1 - FH)
            Wt = (lpSum(row(p, g)["mu"] * ss[p, g] for p in pids)
                  + lpSum(kappa(row(p, g), theta) * cs[p, g]
                          for p in cap_cands[g])
                  + gamma * lpSum(row(p, g)["mu"] * vs[p, g]
                                  for p in cap_cands[g])
                  + lpSum(w_bench * row(p, g)["pi"] * row(p, g)["mu"]
                          * (xs[p, g] - ss[p, g]) for p in pids))
            prob += zt <= Wt
            prob += zt <= M_Z * FH
            terms.append(w * (z + zt))
        else:
            terms.append(w * W)
        terms.append(-hit_cost * w * h[g])
        if ft_value > 0:
            terms.append(-ft_value * w * fr[g])
        # form hold (soft): selling last week's haulers costs extra
        for p, pen in (sell_hold or {}).items():
            if p in pids and p in owned:
                terms.append(-pen * w * to[p, g])

        wc_ = delta_chip ** (g - t)
        if g in CH["bb"]:
            ybb = LpVariable(f"ybb_{g}", lowBound=0)
            prob += ybb <= lpSum(
                (row(p, g)["mu"]
                 - w_bench * row(p, g)["pi"] * row(p, g)["mu"])
                * (x[p, g] - s[p, g]) for p in pids)
            prob += ybb <= M_BB * CH["bb"][g]
            terms.append(wc_ * ybb)
        if g in CH["tc"]:
            ytc = LpVariable(f"ytc_{g}", lowBound=0)
            prob += ytc <= lpSum(kappa(row(p, g), theta) * c[p, g]
                                 for p in cap_cands[g])
            prob += ytc <= M_TC * CH["tc"][g]
            terms.append(wc_ * ytc)

    # epsilon FT reward: makes ft_plan reporting meaningful (f vars otherwise
    # float freely below their bound) — far too small to buy a phantom hit
    prob += lpSum(terms) + 1e-4 * lpSum(fvar.values())

    # ── per-week structure ────────────────────────────────────────────────────
    for g in gws:
        prob += lpSum(x[p, g] for p in pids) == 15
        for et, quota in POS_QUOTA.items():
            prob += lpSum(x[p, g] for p in pids if et_[p] == et) == quota
        for cl in set(cl_.values()):
            prob += lpSum(x[p, g] for p in pids if cl_[p] == cl) <= MAX_CLUB

        prob += lpSum(s[p, g] for p in pids) == 11
        for et in POS_QUOTA:
            n_et = lpSum(s[p, g] for p in pids if et_[p] == et)
            prob += n_et >= XI_MIN[et]
            prob += n_et <= XI_MAX[et]
        for p in pids:
            prob += s[p, g] <= x[p, g]

        prob += lpSum(c[p, g] for p in cap_cands[g]) == 1
        prob += lpSum(v[p, g] for p in cap_cands[g]) == 1
        for p in cap_cands[g]:
            prob += c[p, g] <= s[p, g]
            prob += v[p, g] <= s[p, g]
            prob += c[p, g] + v[p, g] <= 1

    # ── transfers, FT banking, bank recursion ─────────────────────────────────
    for gi, g in enumerate(gws):
        n_g = lpSum(ti[p, g] for p in pids)
        for p in pids:
            prev = ((1 if p in owned else 0) if g == t
                    else x[p, gws[gi - 1]])
            prob += x[p, g] == prev + ti[p, g] - to[p, g]
            prob += ti[p, g] + to[p, g] <= 1
        if not (initial_build and g == t):
            prob += n_g == lpSum(to[p, g] for p in pids)

        # hits: waived at GW1 build, on an external WC at t, or in-model WC_g
        waived = 15 if (g == t and (is_wildcard_now or initial_build)) else 0
        wc_var = CH["wc"].get(g)
        prob += h[g] >= (n_g - ft_expr(g) - waived
                         - (15 * wc_var if wc_var is not None else 0))
        # friction counts every non-waived transfer, FT-funded included —
        # spending a transfer must beat holding it (option value)
        prob += fr[g] >= (n_g - waived
                          - (15 * wc_var if wc_var is not None else 0))

        # bank recursion (exact sell values; bank >= 0 via lowBound)
        prev_bank = bank if g == t else bankv[gws[gi - 1]]
        prob += bankv[g] == (prev_bank
                             + lpSum(sv_[p] * to[p, g] for p in pids)
                             - lpSum(mp_[p] * ti[p, g] for p in pids))

        # FT recursion — LE-only: more FTs never hurt; phantom hits never
        # profitable since 4*delta^w > 4*delta^(w+1) (blueprint §4.6 proof)
        if gi + 1 < len(gws):
            g2 = gws[gi + 1]
            if g2 in ft_events:
                prob += fvar[g2] == max(1, min(FT_MAX, ft_events[g2]))
            else:
                relax = waived + (15 * wc_var if wc_var is not None else 0)
                prob += fvar[g2] <= ft_expr(g) - n_g + h[g] + 1 + relax

    # churn guard (T4): also keeps the sell-value ledger linear (§5)
    for p in pids:
        prob += lpSum(ti[p, g] for g in gws) <= 1
        prob += lpSum(to[p, g] for g in gws) <= 1

    # cross-solve rebuy lock (no_rebuy = {pid: last locked gw}): a player
    # sold in a PREVIOUS executed week cannot come back while his lock
    # runs — unless a wildcard rebuild fires that week (external WC at t,
    # or the in-model WC_g variable, which relaxes the bound only if set)
    for p, until in (no_rebuy or {}).items():
        if p not in pids or p in owned:
            continue
        for g in gws:
            if g > until:
                break
            if g == t and (is_wildcard_now or initial_build):
                continue
            wc_var = CH["wc"].get(g)
            if wc_var is not None:
                prob += ti[p, g] <= wc_var
            else:
                prob += ti[p, g] == 0

    prob.solve(_get_solver(time_limit))
    if LpStatus[prob.status] != "Optimal":
        raise RuntimeError(
            f"horizon MILP {LpStatus[prob.status]} at GW{t} "
            f"(H={len(gws)}, pool={len(pids)}, chips={chips_on}, "
            f"bank={bank:.1f})")

    def on(var):
        return var.value() is not None and var.value() > 0.5

    chips_plan = {}
    for k, d in CH.items():
        for g, var in d.items():
            if on(var):
                chips_plan[g] = f"{k}{_set_of(g)}"

    weeks, ft_plan, bank_plan = {}, {}, {}
    for gi, g in enumerate(gws):
        if chips_plan.get(g, "").startswith("fh"):
            squad = [p for p in pids if on(xs[p, g])]
            xi = [p for p in squad if on(ss[p, g])]
            prev_owned = (set(owned) if g == t else
                          {p for p in pids if on(x[p, gws[gi - 1]])})
            weeks[g] = {
                "squad": squad, "xi": xi,
                "bench": [p for p in squad if p not in set(xi)],
                "captain": next((p for p in cap_cands[g]
                                 if on(cs[p, g])), None),
                "vice": next((p for p in cap_cands[g]
                              if on(vs[p, g])), None),
                "transfers_in": [p for p in squad if p not in prev_owned],
                "transfers_out": [p for p in prev_owned if p not in squad],
                "hits": 0,
            }
        else:
            squad = [p for p in pids if on(x[p, g])]
            xi = [p for p in squad if on(s[p, g])]
            weeks[g] = {
                "squad": squad, "xi": xi,
                "bench": [p for p in squad if p not in set(xi)],
                "captain": next((p for p in cap_cands[g]
                                 if on(c[p, g])), None),
                "vice": next((p for p in cap_cands[g]
                              if on(v[p, g])), None),
                "transfers_in": [p for p in pids if on(ti[p, g])],
                "transfers_out": [p for p in pids if on(to[p, g])],
                "hits": int(round(h[g].value() or 0)),
            }
        ft_plan[g] = (free_transfers if g == t
                      else int(round(fvar[g].value() or 1)))
        bank_plan[g] = round(bankv[g].value() or 0.0, 1)

    return {"weeks": weeks, "chips": chips_plan, "ft_plan": ft_plan,
            "bank_plan": bank_plan,
            "objective": float(prob.objective.value() or 0.0)}
