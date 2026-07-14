# Chip Strategy Redesign — Design Doc

Status: **IMPLEMENTED — backtest pending** (2026-07-02)
Author: pairing session, 2026-07-02
Scope: `pipeline/season_simulator.py` chip logic (`decide_chip`, force blocks, BB targeting)

> Implementation notes (deltas from this design):
> - `CHIP_STRATEGY = "v2" | "legacy"` flag added — legacy policy kept intact for
>   the thesis ablation (§10.4). All new code: `compute_blank_gws`,
>   `build_lookahead_pool`, `_xi_greedy`, `_best_feasible_assignment`,
>   `decide_chip_v2`.
> - FH/WC values are **budget-true via `run_ilp`** (not greedy): an unbudgeted
>   greedy "optimal XI" overestimates the gap vs your squad every week and
>   would misfire chips. FH is valued only on event weeks (blank/double);
>   WC is evaluated rolling at the current GW (fires when the WC_HORIZON gain
>   clears the bar) rather than being placed by the assignment step.
> - Lookahead pools de-inflate current preds (bench-intel bonuses stripped,
>   current-GW DGW boost undone, squad loyalty removed) before applying the
>   target GW's fixture context.
> - **Backtest not yet run**: this machine is a clone without `data/raw/`
>   (gitignored) and the FPL API has rolled over to 2026-27, so the 2025-26
>   raw files can't be re-fetched. Run §10 on the machine that has
>   `data/raw/fpl_api/{player_history,players_raw,fixtures_raw}.csv`.
>   A Docker image (`fpl-sim`: python3.11 + numpy/pandas/xgboost/lightgbm/pulp)
>   is available here for execution once the data is copied over.
> - The pre-v2 production result is backed up at
>   `data/intel/archive/season_simulation_legacy_2468.json`.

---

## 1. Why we're redoing this

The current chip logic works for the 2025-26 backtest (2468 pts) but is **overfit to
this season's fixture calendar**. It decides *when* to play chips using hardcoded
gameweek numbers:

- `if gw == 17: force Wildcard` ([season_simulator.py:1425](../pipeline/season_simulator.py#L1425))
- `if gw == 18: force Triple Captain` ([:1429](../pipeline/season_simulator.py#L1429))
- `if gw == 19: force FH/TC/WC` ([:1437](../pipeline/season_simulator.py#L1437))
- `TC2_MIN_GW = 20`, `FH2_EARLIEST_GW = 20`, `WC17_LOYALTY` tied to GW17

Two concrete problems:

1. **Doesn't generalize.** Next season's double gameweeks, blanks, and fixture swings
   land on *different* GWs. GW17 won't be a sensible wildcard week; GW19 may or may not
   be a double. The magic numbers were fit to this calendar and won't transfer.
2. **No structural guard against wasteful overlap.** Nothing stops the policy from
   playing two "reset" chips back-to-back — the exact mistake made manually this season
   (Free Hit GW23 → Wildcard GW24), where the Free Hit's one-week team is thrown away a
   week later by the Wildcard, wasting one of the two chips.
3. **Chips left on the table.** The 2468 run used only 6 of 8 chips — `wc2` (never
   triggered) and `fh1` (no first-half double) were wasted.

**Goal:** a chip policy that decides *when* from **conditions** (fixtures + squad state),
not fixed GWs, so the same code plays sensibly on any season's calendar, never stacks
two reset chips, and uses every chip it can.

---

## 2. Design principles

1. **No hardcoded gameweeks.** The only fixed numbers allowed are the FPL-mandated set
   boundaries (below). Every trigger is a function of fixtures + squad state, recomputed
   each GW.
2. **Each chip fires on what it is *for*** — a value function specific to the chip.
3. **Rolling horizon, re-planned every GW.** We only have near-term predictions (the
   walk-forward loop knows form up to `gw-1`), so we plan over a short lookahead and
   re-decide each week — the correct online approach, and how a real manager plans chips.
4. **Constraints prevent waste** — one chip/GW, reset-chip spacing, chip exclusivity.
5. **Use-it-or-lose-it by value, not by GW.** Near a set deadline, force the *best
   remaining* week, chosen by the value function — never a hardcoded GW.
6. **Balanced aggressiveness** (per your call): fire when this GW is the best in the
   lookahead *and* clears a moderate value bar; otherwise hold and re-plan.

---

## 3. FPL 2025-26 chip rules (the only fixed structure)

Two of each chip. Each "set" has a usage window:

| Set | Window | Chips |
|-----|--------|-------|
| Set 1 | GW1–GW19 | `wc1`, `fh1`, `bb1`, `tc1` |
| Set 2 | GW20–GW38 | `wc2`, `fh2`, `bb2`, `tc2` |

A chip not used within its window is lost. Only one chip may be played per GW.
These four numbers (1, 19, 20, 38) are the *only* constants the policy is allowed to hardcode.

---

## 4. Signals available in the codebase

Everything the value functions need already exists in `run_simulation`:

| Signal | Source | Meaning |
|--------|--------|---------|
| `gw_teams[g][team]` | `load_fixtures()` | fixture count for a team in GW `g` — **2 = double, 0 = blank** |
| `dgw_gws` | `load_fixtures()` | set of double gameweeks |
| `fdr_lookup[(team,g)]`, `home_lookup[(team,g)]` | `load_fixtures()` | future fixture difficulty / venue |
| `pool[*]["pred"]`, `form_last3`, `minutes_reliability`, `pos`, `price` | `predict_pool` | per-player projection + form/reliability |
| `current_squad`, `free_transfers`, `bank` | main loop state | squad + transfer budget |
| `optimize_squad(...)` | existing ILP | build an optimal 15/XI under budget |
| `_best_cap_stats(squad_pool, home_lookup, g, strip)` | existing | best captain adj-pred + form |
| `CAP_MULT`, availability tiers (`avail_gws`) | existing | captain multipliers, injury tiers |

**Lookahead approximation (reuse as-is).** `get_bench_intel` already builds a future-GW
pool by copying the current pool and swapping in `fdr`/`was_home` for the target GW
([:473-485](../pipeline/season_simulator.py#L473)). It does **not** re-run the model — it
reuses current `pred` under future fixture context. We reuse this exact trick for all chip
scoring, and additionally apply the DGW multiplier (`DGW_PRED_MULT`) to players whose
`gw_teams[g][team] >= 2`, and zero out players with `gw_teams[g][team] == 0` (blank).
Extract it into a shared helper:

```python
def build_lookahead_pool(pool, g, fdr_lookup, home_lookup, gw_teams):
    fut = []
    for p in pool:
        q = dict(p)
        q["fdr"]      = float(fdr_lookup.get((p["team"], g), 3.0))
        q["was_home"] = float(home_lookup.get((p["team"], g), 0))
        n_fix = gw_teams.get(g, {}).get(p["team"], 1)
        if   n_fix == 0: q["pred"] = 0.0                              # blank
        elif n_fix >= 2: q["pred"] = min(PRED_CAP, q["pred"] * DGW_PRED_MULT)  # double
        q["n_fix"] = n_fix
        fut.append(q)
    return fut
```

---

## 5. Per-chip value functions

For a candidate GW `g`, build `lp = build_lookahead_pool(...)`, restrict to squad where
noted, and compute a **value = extra points this chip would earn on GW `g`**.

### Bench Boost — `bb_value(g)`
Points your bench (GK + 3 outfield) would add if they counted. Best on a double where
most of your 15 play twice.
```
squad_lp = [p in lp if p.player_id in current_squad]
xi, bench = split_by_formation(squad_lp)      # optimal legal XI + 4 bench
bb_value(g) = sum(p.pred for p in bench)      # DGW already baked into pred
```

### Triple Captain — `tc_value(g)`
Marginal points from the 3rd captain multiple (TC = ×3 vs normal ×2, so the extra is one
more copy of the captain's score). DGW captains already have doubled `pred`.
```
best_adj, best_form3, _ = _best_cap_stats(squad_lp, home_lookup, g, loyalty_strip)
tc_value(g)  = best_adj                        # the extra ×1
tc_eligible  = best_form3 >= TC_FORM_BAR and best captain reliability ok
```

### Free Hit — `fh_value(g)`
One-week gain from fielding an *optimal temporary* team vs your real squad. Large when
your squad blanks (your players' pred → 0) or a double lets an optimal team stack DGW
players you don't own.
```
current_xi_pred = sum(best legal XI from squad_lp)
temp_xi_pred    = optimize_squad(lp, budget=∞-ish, no_transfer_cost).xi_pred   # fresh optimal XI
fh_value(g)     = temp_xi_pred - current_xi_pred
```
(If a full ILP per candidate GW is too slow, approximate `temp_xi_pred` with a greedy
top-by-position pick — the ranking is what matters, not the exact squad.)

### How WC vs FH disambiguate (worked example)

The two reset chips answer different questions, and their value functions encode it:
**FH = "how bad is this ONE week?"** (one-week gain, squad reverts after);
**WC = "how bad is my squad FROM HERE ON?"** (gain summed over a horizon, rebuild persists).

*Blank-week case:* 6-7 squad players blank in GW `g`, squad otherwise good →
blanking players get `pred = 0` in the lookahead pool → your XI collapses (~35 pts),
optimal temp XI ≈ 60 → `fh_value ≈ 25` clears the bar and FH fires. Meanwhile
`wc_value` stays low: after the blank the squad is back to strength, so the summed
reset gain is small — no permanent rebuild for a one-week problem.

*Stale-squad case:* several players injured/out-of-form entering a bad fixture block →
the deficit vs an optimal squad persists across every horizon GW → `wc_value` large,
`fh_value` (one week only) modest → WC fires, timed before the good fixture run.

Spacing (§6.2) then guarantees the two can never land adjacent even when both look
attractive — the structural fix for the manual FH GW23 → WC GW24 mistake.

### Wildcard — `wc_value(g)`
Unlike the others, a Wildcard's benefit accrues over **many** future GWs (the new squad
persists). Value = summed gain of an optimal reset squad vs your current squad over a
horizon `W`, plus hits you'd otherwise pay to fix the squad normally.
```
wc_value(g) = Σ_{k=g..min(set_end, g+W)} ( optimal_reset_squad_pred(k) - current_squad_pred(k) )
            + saved_hit_penalty(current_squad, free_transfers)
```
`saved_hit_penalty` ≈ 4 × max(0, transfers_needed_to_reach_good_squad − free_transfers).
High when the squad is stale (many players below their positional pred mean, injured, or
out of form) — this replaces the old "5+ below average" heuristic with a forward-looking one.

---

## 6. Constraints

1. **One chip per GW** (FPL rule).
2. **Reset-chip spacing:** any two of {Wildcard, Free Hit} must be `≥ SPACING_GAP` GWs
   apart. *This is the structural fix for the FH→WC-next-week mistake.*
3. **Exclusivity:** Free Hit and Bench Boost never the same GW (FH replaces the very
   bench a BB would score).
4. **Set eligibility:** set-1 chips only for `g ≤ 19`; set-2 chips only for `20 ≤ g ≤ 38`.
5. **Deadline:** each unused chip must be placed on some GW `≤ set_end`.

---

## 7. Decision algorithm (rolling horizon)

Runs every GW, replaces `decide_chip` + all force blocks:

```
def decide_chip_v2(gw, chips_used, pool, current_squad, free_transfers,
                   fdr_lookup, home_lookup, gw_teams, dgw_gws):
    if gw <= CHIP_LOCKOUT: return None

    set_id  = 1 if gw <= 19 else 2
    set_end = 19 if set_id == 1 else 38
    remaining = [c for c in ("wc","fh","bb","tc")
                 if f"{c}{set_id}" not in chips_used]
    if not remaining: return None

    # Candidate GWs — EVENT-AWARE: near lookahead PLUS all known double/blank
    # GWs remaining in this set. The fixture calendar (dgw_gws / gw_teams) is
    # known ahead of time — in the sim it's loaded season-wide, in real FPL
    # doubles/blanks are announced weeks ahead when cup rounds reschedule.
    # Schedule knowledge, not result leakage. This is what lets the planner
    # RESERVE BB for a known GW34 double while sitting at GW22, and hold FH
    # for a known blank — "saving" emerges from assignment, no hoarding rule.
    horizon_end   = min(set_end, gw + LOOKAHEAD)
    if set_end - gw <= LOOKAHEAD: horizon_end = set_end
    event_gws     = {g for g in range(gw, set_end + 1)
                     if g in dgw_gws or has_blank_teams(g, gw_teams)}
    candidate_gws = sorted(set(range(gw, horizon_end + 1)) | event_gws)

    # Value matrix V[chip][g] over candidate GWs (+ eligibility gates)
    V = { c: { g: value_of(c, g) for g in candidate_gws
               if eligible(c, g) } for c in remaining }

    # Assign chips -> GWs maximizing total value under the constraints in §6.
    # Small problem (<=4 chips, <=~10 GWs): brute-force feasible assignments,
    # or greedy-by-descending-value with constraint checks. Also fold in chips
    # ALREADY used this set (their GW is fixed) when checking spacing.
    plan = best_feasible_assignment(V, chips_used, SPACING_GAP)

    # Commit only THIS GW's assignment, and only if it clears the balanced bar.
    chip_here = plan.get(gw)                    # e.g. "wc" or None
    if chip_here is None: return None
    forced = (set_end - gw == 0) or runway_exhausted(chip_here, gw, plan)
    if forced or V[chip_here][gw] >= BAR[chip_here]:
        return f"{chip_here}{set_id}"
    return None
```

`best_feasible_assignment` enumerates assignments of `remaining` chips to distinct
candidate GWs, discards any violating §6 (including spacing vs already-played chips this
set), and returns the max-total-value one. With ≤4 chips this is tiny.

---

## 8. "Balanced" aggressiveness — starting bars

Fire when this GW is the plan's assigned week for the chip **and** value ≥ bar. Bars are
starting guesses to be Optuna-tuned later (§10); "balanced" = moderate, not eager, not
hoarding. Deadline pressure drops the bar to 0 (use-it-or-lose-it).

| Param | Start | Meaning |
|-------|-------|---------|
| `LOOKAHEAD` | 4 | GWs of forward planning |
| `SPACING_GAP` | 4 | min GWs between Wildcard and Free Hit |
| `WC_HORIZON W` | 5 | GWs over which Wildcard gain is summed |
| `BAR["bb"]` | ~14 | bench must project ≥ ~14 pts |
| `BAR["tc"]` | 6.17 | keep current `TC_THRESH`/`TC_FORM_MIN` — the natural TC trigger fired both chips (GW6, GW23) in the 2468 run and delivered; it is already condition-based, not hardcoded |
| `BAR["fh"]` | ~16 | optimal temp XI beats squad by ≥ ~16 pts |
| `BAR["wc"]` | ~20 | windowed reset gain ≥ ~20 pts |

---

## 9. Integration plan (what changes in the file)

- **Add** `build_lookahead_pool(...)` helper; refactor `get_bench_intel` to use it (BB
  value becomes one caller among four).
- **Add** `value_of`, `eligible`, `best_feasible_assignment`, `decide_chip_v2`.
- **Replace** the `decide_chip` force blocks ([:1424-1455](../pipeline/season_simulator.py#L1424))
  and natural triggers ([:1457-1505](../pipeline/season_simulator.py#L1457)) with
  `decide_chip_v2`.
- **Remove** hardcoded `GW17/18/19` logic and `WC17_LOYALTY`; apply the loyalty override
  whenever WC is the committed chip this GW (not just at GW17).
- **Keep** the two-set boundaries and `CHIP_LOCKOUT`.
- **Fix** the cosmetic `"GW1 to GW28"` print ([:1539](../pipeline/season_simulator.py#L1539))
  and stale comment ([:1563](../pipeline/season_simulator.py#L1563)).

Nothing outside the chip logic changes — models, ILP, intel penalties, transfer logic stay put.

---

## 10. Validation plan

1. **Backtest GW1-38** on 2025-26. Success = all in-window chips used, **no FH/WC within
   `SPACING_GAP`**, total **≥ 2468** (expect a gain from recovering `wc2`/`fh1`).
2. **Generalization check** — run the same code against 2023-24 and 2024-25 fixture
   calendars (historical data already in repo). Success = no crashes, chips land on
   plausible weeks (doubles/blanks), no adjacency violations. This is the real proof it's
   not overfit.
3. **Optuna re-tune** the bars/params in §8 over GW1-38 once logic is verified — same
   harness as `optuna_search_gw38`.
4. **Ablation for thesis:** old hardcoded policy vs new policy, same models → report the
   delta and the generalization result.

---

## 11. Resolved decisions (2026-07-02 session)

1. **FH temp-team value: greedy approximation** (top-by-position under budget), full ILP
   only as an optional verify step. The *ranking* of candidate weeks is what matters, and
   with event-aware candidates we may score many GWs per week — greedy keeps it fast.
2. **Blanks are first-class FH targets** — not just doubles. `fh_value` naturally spikes
   on blanks because blanking squad players drop to `pred = 0` (see the worked example in
   §5). Second-half seasons cluster both doubles *and* blanks, so FH2 will usually gravitate
   to a blank while BB2 takes the biggest double — matching how the chips are meant to be used.
3. **WC horizon `W = 5`** to start, Optuna-tunable. Not rest-of-set: far-future preds
   reflect current form and get noisy; 5 GWs captures a fixture block without overweighting
   noise. (Event-GW values for BB/FH are exempt from this concern — the ×2/×0 fixture-count
   effect dominates and is exact.)
4. **Spacing: WC↔FH only** (`SPACING_GAP = 4`). TC/BB on or near reset weeks is fine —
   e.g. WC into a double then BB on it is a *good* combo, not a wasteful one. FH+BB same-GW
   exclusion (§6.3) already covers the one genuinely bad overlap.
5. **TC logic unchanged in substance** — the existing natural trigger (adj ≥ `TC_THRESH`,
   form ≥ `TC_FORM_MIN`) fired tc1 GW6 and tc2 GW23 in the 2468 run and both delivered.
   Only the GW18 force-net is replaced (by the generic deadline net); `TC2_MIN_GW = 20` is
   dropped as redundant (Set 2 starts GW20 anyway).

---

## 12. Thesis framing

Positions chip usage as a **constrained rolling-horizon assignment problem** solved online
under prediction uncertainty — a clean contrast to the hardcoded-heuristic baseline. The
generalization test (§10.2) and old-vs-new ablation (§10.4) give a concrete results
section: the redesign trades a season-specific 2468 for a policy that is calendar-agnostic
and provably avoids the reset-chip-overlap failure mode.
