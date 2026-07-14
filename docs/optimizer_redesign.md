# Multi-Period Optimizer Redesign — Complete Blueprint

Status: **DESIGN — not yet implemented** (2026-07-03)
Scope: replaces the single-GW ILP core of `pipeline/season_simulator.py`
(`run_ilp`, `predict_pool` adjustment chain, `select_captain`, `decide_chip`/
`decide_chip_v2`, `next_ft`, bank bookkeeping) with one multi-period MILP.
Baseline to beat: 2468 pts (GW1-38, 2025-26, legacy chip policy).

Related docs: `chip_strategy_redesign.md` (v2 chip scheduler — partially
superseded by this design; its lookahead machinery is reused).

---

## 0. Design goal in one paragraph

Every gameweek, solve **one** mixed-integer linear program over a rolling
horizon of H=5 gameweeks (tunable 4–6) that jointly decides: transfers per
week, squad per week, starting XI per week, captain and vice per week, bench,
and chip placement — under exact FPL rules (50% sell-on profit, 5-FT banking,
one chip per GW, chip set windows). Execute only the first week's decisions,
advance the world, re-solve. The intelligence moves out of ~25 jointly-tuned
constants and into (a) an honest per-player-per-GW prediction matrix and
(b) the structure of the program. Target: ≤ 8 tunable constants, each with a
meaning that survives a season change.

---

## 1. Architecture

Five modules with hard interfaces (each independently testable):

```
┌────────────────┐   ┌──────────────────────┐   ┌───────────────────┐
│ State ledger    │→ │ Prediction matrix     │→ │ MILP core          │
│ squad, purchase │   │ μ, π, φ, q90 per     │   │ variables, obj,    │
│ prices, bank,   │   │ (player, horizon GW)  │   │ constraints, solve │
│ FTs, chips used │   └──────────────────────┘   └───────────────────┘
└────────────────┘              ↑                        ↓
        ↑              ┌──────────────────────┐   ┌───────────────────┐
        └───────────── │ Reservation guard     │   │ Executor           │
                       │ far-event chip gating │   │ apply week-t only, │
                       └──────────────────────┘   │ auto-subs, scoring │
                                                  └───────────────────┘
```

- **State ledger** — persistent, replaces scattered loop state. Holds squad
  with per-player purchase price, bank, FT count, chips used with their GWs.
- **Prediction matrix** — extends `build_lookahead_pool`
  ([season_simulator.py:1536](../pipeline/season_simulator.py#L1536)) from
  multiplier-swaps to model-consistent per-GW predictions (§3).
- **MILP core** — the formulation in §4. Pure function:
  `(state, matrix, calendar, rules) → plan`.
- **Reservation guard** — the one surviving heuristic: prevents burning a
  chip inside the horizon when a known far event (announced DGW/BGW beyond
  t+H−1) values it higher (§6.5).
- **Executor** — applies week-t decisions only; unchanged auto-subs, FH
  revert, scoring. Updates the ledger (incl. purchase prices on buys).

### 1.1 What is preserved from the current system (verbatim or near)

| Component | Disposition |
|---|---|
| Joint squad/XI/transfer exact ILP | kept, extended to H weeks |
| Constraint layer (15-man 2/5/5/3, ≤3/club, XI formations incl. 5-DEF) | kept, indexed by week |
| Walk-forward data hygiene, per-GW retraining | untouched |
| Intel availability multiplicative penalties | kept — now feeding π and μ (§3.4) |
| Auto-subs (`apply_auto_subs`), FH revert, bank invariants | kept in Executor |
| FT accrual +1/wk, preserved through WC/FH | kept, correct 5-cap (§4.6) |
| `build_lookahead_pool` fixture-swap machinery | kept, upgraded (§3.1) |
| chip-v2 value functions | demoted to Reservation guard estimates only |

### 1.2 What is deleted, and what structurally replaces it

| Removed constant/hack | Was compensating for | Structural replacement |
|---|---|---|
| Loyalty bonus (+10/+2/+1) | no planning horizon → churn | horizon objective prices holding vs switching |
| Bench-intel bonus (2.71/2.25) | bench absent from objective | bench EV term (§4.3) |
| `XI_PRED_CAP` global rescale | miscalibrated predictions | calibration in Phase 1 (§9); no clamps |
| `PRED_CAP=20` | tail blowups | isotonic/quantile calibration; keep only as sanity assert |
| GW2–8 blend to GW1 preds | early-season model noise | confidence shrinkage φ (§3.5) |
| Ownership boost GW1 (0.213) | cold start | φ ramp + shrinkage to replacement; ownership becomes a model *feature* (deferred, §11) |
| FDR post-multipliers (0.028/0.084) | FDR under-used by model | per-GW feature-swap predictions (§3.1) — model consumes FDR directly |
| DGW_PRED_MULT = 2.0 | no per-fixture predictions | per-fixture prediction sum (§3.2) |
| `CAP_MULT` position multipliers | variance-seeking proxy | explicit q90 ceiling in captain coefficient (§3.6) |
| CAP streak/blank/home/form-gate constants | outcome-chasing noise | deleted, no replacement — captaincy is priced by μ/q90/π only |
| TC/BB/WC/FH thresholds + force GWs | no chip planning | chip variables in objective (§4.7) + reservation guard |
| `MAX_HITS=1` | untrusted hit valuation | horizon-priced hits; soft guardrail h_g ≤ 2 during rollout (§7) |
| FT cap 2 pre-GW15, GW15:=5 gift | hardcoded calendar | rules config: `RULE_EVENTS` list of dated one-off grants (§4.6) |
| Budget relaxation +0.5m×4 | infeasibility panic | infeasible ⇒ raise, log, investigate. Never phantom money |

---

## 2. Notation and inputs

- t — current GW; horizon 𝒢 = {t, …, min(t+H−1, 38)}; w = g−t week offset.
- P — pruned player pool (§7.2); e(p) ∈ {GK,DEF,MID,FWD}; club(p).
- State: initial squad S₀ (|S₀|=15) with purchase prices pp_p; bank B₀;
  free transfers F₀ ∈ {1..5}; chips used with GWs; set windows
  Set1 = GW1–19, Set2 = GW20–38.
- Calendar: n_fix(τ, g) fixtures for club τ in GW g (0 = blank, ≥2 = double);
  per-fixture opponent, FDR, venue.
- Prices: mp_p current market price. **Assumed constant over the horizon**
  (§8 R6). Sell value sv_p precomputed (§5).
- Prediction matrix (§3): μ_{p,g}, π_{p,g}, φ_{p,g}, q90_{p,g}.

Derived objective coefficients:

- ŝ_{p,g} = φ_{p,g}·μ_{p,g} + (1−φ_{p,g})·ρ_{e(p),g} — confidence-shrunk score
  (§3.5). ρ is the position replacement level, NOT zero — multiplying by
  φ alone would systematically punish uncertain players even when their
  uncertainty is symmetric.
- κ_{p,g} = π_{p,g}·[(1−θ)·μ_{p,g} + θ·q90_{p,g}] — captain coefficient
  (the extra ×1 armband copy; ceiling-blended because captaincy is convex).
- β_{p,g} = w̄·π_{p,g}·μ_{p,g} — bench expected auto-sub contribution (§4.3).
- δ ∈ (0,1) — per-week discount (start 0.84 ≈ the old 1/0.7/0.5 spirit).
- δ_c — chip-term discount, **δ_c > δ, start 0.97** (§4.8 explains why).

---

## 3. Prediction matrix

The single most important upgrade. For every (p, g ∈ 𝒢) produce
(μ, π, φ, q90) **before** the solve. Everything downstream is linear in these.

### 3.1 Per-GW model-consistent baseline

Current `build_lookahead_pool` copies this week's `pred` and applies
multipliers. Replace with **feature-swap re-prediction**: for each future GW
g, rebuild the feature vector with that GW's `fdr` and `was_home` (from
`fdr_lookup`/`home_lookup`, exactly as the lookahead pool already swaps them)
and **re-run the LGBM model**. Batched, this is 4 model calls of ~(|P|·H)
rows — milliseconds. This deletes the FDR post-multipliers: the model already
has `fdr` and `was_home` as features; let it use them.

### 3.2 Doubles and blanks — exact, per fixture

For GW g and club τ with n_fix(τ,g) fixtures:

- n_fix = 0 → μ_{p,g} = 0. Hard zero, applied in the matrix so *every*
  consumer (XI, captain, chips, transfers) sees it. Fixes the current
  blank-blindness where `fdr` silently defaults to 3.0
  ([season_simulator.py:901](../pipeline/season_simulator.py#L901)).
- n_fix ≥ 1 → μ_{p,g} = Σ over fixtures of the model prediction with that
  fixture's (fdr, venue). Replaces the flat ×2.0 with per-opponent sums; a
  double vs MCI(a)+ARS(h) is no longer worth the same as LUT(h)+SHU(a).
- Cap check only as an assert (μ ≤ 40), never a silent clamp.

### 3.3 Minutes / play probability π

Two-part target design (Phase 1 model change): train on played rows only so
the model estimates E[pts | plays] =: μ^play, and estimate P(plays) =: π
separately. Then μ = π·μ^play. Until that lands, fallback: keep the current
all-rows model (which bakes in historical benching) and use π only as a
*deviation* factor: μ = model_pred · clip(π_now/π_hist, 0, 1.25).

π_{p,g} = w(g)·π_intel + (1−w(g))·π_base, where:
- π_intel from availability tiers (available .98, probable .90, doubtful .50,
  unlikely .25, out/susp 0) × rotation factor from start share
  (min(1, starts_last5/5 + 0.2)),
- π_base = player's season start rate,
- w(g) = max(0, 1 − 0.25·(g−t)) — intel is near-week information; it decays
  to the base rate by t+4. This is where intel_04 rotation data (currently
  unused in production) re-enters.

### 3.4 Availability

Unchanged principle: intel_03 tiers multiply into π (and hence μ). "out" ⇒
π = 0 ⇒ μ = 0 ⇒ avoidance is emergent — preserved by design.

### 3.5 Confidence φ (player-specific ONLY)

φ_{p,g} ∈ [0.55, 1.0]. **All time-decay lives in δ; φ must not decay with
g** or uncertainty is double-counted. φ = 0.55 + 0.45·r, with
r = r_sample · r_minutes · r_return · r_new:

- r_sample = min(1, played_GWs_this_season / 6) — early-season ramp
  (replaces the GW1-pred blending hack),
- r_minutes = 1 − 0.5·min(1, std(minutes_last5)/45) — minutes volatility,
- r_return = 0.6 for the first 2 GWs after an absence ≥ 3 GWs,
- r_new = 0.8 for the first 4 GWs after a club change.

Used only via shrinkage: ŝ = φμ + (1−φ)ρ. Replacement level ρ_{pos,g} :=
the 25th-percentile ŝ among players of that position predicted to start in
GW g (computed from the matrix itself, one fixed-point pass with μ as seed —
not a hardcoded constant).

**Calibration gate:** before φ ships, verify on 2023-24/2024-25 that low-φ
buckets have materially higher MAE than high-φ buckets. If they don't, set
φ ≡ 1 and delete the machinery. Uncertainty features must earn their place.

### 3.6 Ceiling q90 (captain only)

q90_{p,g} = μ_{p,g} + 1.2816·σ_{p}·√n_fix(club(p),g), where σ_p is a shrunk
per-player std: σ_p = (n·σ̂_p + m·σ̄_pos)/(n+m) with σ̂_p = std of last-10
played-GW scores, σ̄_pos the position average, m = 5 prior strength. Enters
only κ (captain/TC coefficient) with blend θ (start 0.5). Never enters ŝ —
we want ceiling-chasing armbands, not ceiling-chasing squads.

### 3.7 Vice / captain-miss probability

Captain-miss enters the objective through a fixed γ (start 0.07 ≈ 1 − mean π
of top captain candidates). Exact pairwise modelling (1−π_captain)·μ_vice is
bilinear in c,v; §4.4 justifies the constant-γ linearization (error is
second-order: it can only misrank *vice* candidates, by at most
γ·Δμ ≈ 0.07·2 ≈ 0.1 pts).

---

## 4. The MILP — complete formulation

### 4.1 Decision variables

| Var | Domain | Meaning |
|---|---|---|
| x_{p,g} | {0,1} | p in 15-man squad during g (post that week's transfers) |
| s_{p,g} | {0,1} | p starts in g |
| c_{p,g}, v_{p,g} | {0,1} | captain / vice in g (restricted to candidate set C_g, §7.2) |
| in_{p,g}, out_{p,g} | {0,1} | p bought / sold at deadline g |
| f_g | int [1,5] | free transfers available at g (f_t = F₀ constant) |
| h_g | cont ≥ 0 | paid hits at g (integral at optimum since n,f integer) |
| bank_g | cont ≥ 0 | bank after g's trades |
| WC_g, FH_g, BB_g, TC_g | {0,1} | chip played at g |
| z_g, z̃_g | cont ≥ 0 | week-g main / Free-Hit-shadow score (aux, §4.5) |
| y^{BB}_g, y^{TC}_g | cont ≥ 0 | chip bonus score (aux, §4.7) |
| x̃_p, s̃_p, c̃_p, ṽ_p, k_p | {0,1}/[0,1] | FH shadow squad block, only if an FH-candidate week is in 𝒢 (§4.5) |

Squad/XI/transfer vars exist for p ∈ P (pruned pool), g ∈ 𝒢. With |P| ≈ 180,
H = 5: ≈ 3,600 core binaries + ≤ 800 shadow/captain binaries. See §7.

### 4.2 Objective

maximize

```
  Σ_{g∈𝒢} δ^{g−t} · ( z_g + z̃_g − 4·h_g )
+ Σ_{g∈𝒢} δ_c^{g−t} · ( y^{BB}_g + y^{TC}_g )
+ ε · Σ_p V_p · x_{p, t+H−1}                     [terminal value, default ε=0]
```

where the week-g **main score expression** is

```
W_g =  Σ_p ŝ_{p,g}·s_{p,g}                       (XI, confidence-shrunk mean)
     + Σ_{p∈C_g} κ_{p,g}·c_{p,g}                 (captain extra ×1, ceiling-blended)
     + γ · Σ_{p∈C_g} μ_{p,g}·v_{p,g}             (vice, miss-prob weighted)
     + Σ_p β_{p,g}·(x_{p,g} − s_{p,g})           (bench auto-sub EV)
```

and z_g / z̃_g link to W_g and the shadow week score W̃_g via §4.5. On weeks
with no FH candidate, skip the aux and put W_g directly in the objective.

Rationale notes:
- Hits are discounted at δ like gains. A conservative variant (hits at δ_c —
  future hits cost more relative to their discounted gains) is an ablation
  flag, not the default.
- Terminal value V_p = fixture-agnostic model score averaged over the 3 GWs
  after the horizon; default OFF (ε=0). Rolling re-solve makes end-effects
  mostly harmless (§8 R4); enable only if the H-sweep ablation shows
  end-of-horizon transfer starvation.

### 4.3 Bench EV term — why β and not exact simulation

Exact auto-sub probability depends on bench order and on which starters miss,
which is combinatorial. We linearize with per-player coefficients:
β_{p,g} = w̄·π_{p,g}·μ_{p,g}, w̄ = average probability a bench slot is
activated. Calibrate w̄ from history (league-wide ≈ 0.3–0.5 auto-subs per team
per GW across 3 outfield slots + GK ⇒ w̄ ≈ 0.10–0.15; measure it, don't guess).
Bench *order* stays a post-solve sort by π·μ (Executor), as today.
This deletes the bench-bonus hack while making the ILP pay real (small) money
for reliable bench players — and much more on BB weeks via y^{BB}.

### 4.4 Roster, XI, captain constraints (per g ∈ 𝒢)

```
(R1)  Σ_p x_{p,g} = 15
(R2)  Σ_{e(p)=GK} x = 2;  DEF = 5;  MID = 5;  FWD = 3
(R3)  Σ_{club(p)=τ} x_{p,g} ≤ 3            ∀ clubs τ
(X1)  Σ_p s_{p,g} = 11·(1 − FH_g)          [main XI stands down on FH week]
(X2)  Σ_{GK} s = 1·(1−FH_g);  3(1−FH_g) ≤ Σ_{DEF} s ≤ 5;
      2(1−FH_g) ≤ Σ_{MID} s ≤ 5;  1(1−FH_g) ≤ Σ_{FWD} s ≤ 3
(X3)  s_{p,g} ≤ x_{p,g}
(C1)  Σ_p c_{p,g} = 1 − FH_g;   Σ_p v_{p,g} = 1 − FH_g
(C2)  c_{p,g} ≤ s_{p,g};  v_{p,g} ≤ s_{p,g};  c_{p,g} + v_{p,g} ≤ 1
```

Justifications: R1–R3, X2–X3, C1–C2 carry over from production `run_ilp`
(which is correct, incl. 5-DEF formations — stage8's DEF≤4 bug must NOT be
copied). GK captaincy is legal in FPL; the old hard ban is removed — κ for a
GK is naturally tiny. The (1−FH_g) gating replaces the old separate FH re-run.

### 4.5 Free Hit shadow block (only when an FH candidate week is in 𝒢)

FH candidates: weeks g ∈ 𝒢 that are event weeks (some club has n_fix ≠ 1) —
same restriction as chip-v2, keeps at most 1–2 candidates per solve. If none:
FH_g = 0 fixed, no shadow vars. For the (single) candidate week g*:

```
(F1)  Σ_p x̃_p = 15·FH_{g*};  positional quotas ·FH_{g*};  club ≤ 3
(F2)  Σ_p s̃_p = 11·FH_{g*};  formation bounds gated by FH_{g*};  s̃ ≤ x̃
(F3)  Σ c̃ = FH_{g*};  Σ ṽ = FH_{g*};  c̃ ≤ s̃, ṽ ≤ s̃, c̃+ṽ ≤ 1
(F4)  in_{p,g*} ≤ 1 − FH_{g*};  out_{p,g*} ≤ 1 − FH_{g*}     [no permanent
      transfers on an FH week; squad continuity x_{p,g*} = x_{p,g*−1} follows]
(F5)  budget with keep-linearization (k_p = kept player, sv ≤ mp so the
      solver maxes k automatically):
      Σ_p mp_p·(x̃_p − k_p) ≤ bank_{g*−1} + Σ_p sv_p·(x_{p,g*−1} − k_p)
      k_p ≤ x̃_p;  k_p ≤ x_{p,g*−1}
(F6)  score switch:  z_{g*} ≤ W_{g*};          z_{g*} ≤ M_z·(1 − FH_{g*})
                     z̃_{g*} ≤ W̃_{g*};         z̃_{g*} ≤ M_z·FH_{g*}
      (W̃ = shadow analogue of W;  M_z = 120 safe upper bound on any W)
```

The ≤-only linearization is valid because z, z̃ carry positive objective
weight — the solver lifts them to the binding bound; no lower-bound
constraints needed. FH squad revert needs no modelling: x_{p,g} simply never
changed. Executor behaviour on a fired FH week is today's revert logic.

### 4.6 Transfers, free-transfer banking, hits

```
(T1)  x_{p,g} = x_{p,g−1} + in_{p,g} − out_{p,g}    (x_{p,t−1} := [p ∈ S₀])
(T2)  in_{p,g} + out_{p,g} ≤ 1
(T3)  Σ_p in_{p,g} = Σ_p out_{p,g}   =: n_g          (squad always 15)
(T4)  Σ_{g∈𝒢} in_{p,g} ≤ 1  and  Σ_{g∈𝒢} out_{p,g} ≤ 1     [churn guard]
(T5)  h_g ≥ n_g − f_g − 15·(WC_g + FH_g)             (hits; free on WC, none
      possible on FH by F4)
(T6)  f_{g+1} ≤ f_g − n_g + h_g + 1 + 15·(WC_g + FH_g)
(T7)  f_{g+1} ≤ 5;  f_{g+1} ≥ 1
(T8)  h_g ≤ H_cap  (rollout guardrail, start 2; relax after validation)
```

Justifications and proofs:
- T4 is doing three jobs: kills sell-buyback churn inside a horizon, keeps
  the sell-price ledger linear (§5), and prunes the search space. It forbids
  only "sell-and-rebuy within ≤5 weeks" — behaviour we explicitly want gone.
- T6/T7 are ≤-only: since larger f_{g+1} only relaxes future hit constraints
  and never hurts, the solver sets f maximal — equality is unnecessary.
  Correctness of the recursion: FTs consumed on a normal week =
  n_g − h_g (transfers beyond FT are paid), so
  f_{g+1} = min(5, f_g − (n_g − h_g) + 1); chip weeks consume 0 and still
  accrue +1 (real FPL rule: WC/FH preserve saved FTs).
- **No phantom-hit exploit:** raising h_g by 1 (cost 4δ^w) lifts the f_{g+1}
  bound by 1, which can save at most one future hit (gain 4δ^{w+1}).
  Since δ^{w+1} < δ^w, it is never profitable. ∎
- FT state is otherwise exactly FPL 2025-26: bank to 5 from GW1 (the current
  code's pre-GW15 cap of 2 and the hardcoded `GW15 → 5` are deleted). One-off
  rule events (e.g. an AFCON FT grant) live in a `RULE_EVENTS` config read by
  the ledger, never in code.

### 4.7 Chips as variables

Let used(k, set) ∈ {0,1} be ledger state; Set(g) ∈ {1,2}.

```
(K1)  Σ_{g∈𝒢 ∩ Set_s} k_g ≤ 1 − used(k, s)     ∀ chip k, set s      [availability]
(K2)  WC_g + FH_g + BB_g + TC_g ≤ 1             ∀ g                  [one chip/GW]
(K3)  spacing (Δ = 4) between reset chips, incl. already-played ones:
        WC_g + FH_{g'} ≤ 1        ∀ g,g' ∈ 𝒢, |g−g'| < Δ
        WC_g + WC_{g'} ≤ 1,  FH_g + FH_{g'} ≤ 1   ∀ |g−g'| < Δ  (cross-set)
        k_g = 0                   ∀ reset chip k, g with an already-played
                                   reset chip within Δ (ledger constants)
(K4)  BB bonus:  y^{BB}_g ≤ Σ_p (ŝ_{p,g} − β_{p,g})·(x_{p,g} − s_{p,g})
                 y^{BB}_g ≤ M_BB·BB_g                     (M_BB = 60)
(K5)  TC bonus:  y^{TC}_g ≤ Σ_{p∈C_g} κ_{p,g}·c_{p,g}
                 y^{TC}_g ≤ M_TC·TC_g                     (M_TC = 30)
(K6)  FH: shadow block §4.5.  WC: no score term — its value is emergent
      (T5 waives hits; unlimited n_g that week rebuilds the persistent squad).
```

Chip value is now fully emergent: BB is worth its actual bench that week
(so the solver will *build* a bench in g−1, g−2 to serve a planned BB — the
coordination the old system could never express); TC is worth one extra
captain copy on the best ceiling week; FH is worth the shadow-vs-main score
gap on an event week; WC is worth the hit-free rebuild summed over the
remaining horizon. No thresholds, no force weeks. Deadline pressure is also
emergent: if a set deadline is inside the horizon, an unused chip with any
positive value gets placed (K1 permits, objective wants it); a truly
worthless chip dies — correct (better wasted than misused).

Aux-≤ correctness: y carries positive weight ⇒ lifted to min of its bounds ⇒
equals (chip fired ? bonus : 0) at optimum. BB−β subtraction avoids
double-counting the bench EV already in W_g. TC on FH weeks is impossible
via K2; y^{BB} on FH weeks is zero because BB_g = 0 by K2.

### 4.8 Why chips get a slower discount (δ_c > δ)

Under the ordinary δ, a chip played at t+3 looks ~40% less valuable than
today, so the solver would systematically burn chips early — the mirror image
of the old threshold-crossing bug. But chip value on event weeks is dominated
by the fixture calendar (×2 / ×0), which is *known*, not forecast — the usual
uncertainty argument for δ barely applies. δ_c ≈ 0.97 keeps a mild
tie-breaking preference for earlier certainty without early-burn bias. This
is one of the 8 honest constants and gets a dedicated ablation.

### 4.9 Budget and sell price

```
(B1)  bank_g = bank_{g−1} + Σ_p sv_p·out_{p,g} − Σ_p mp_p·in_{p,g}
(B2)  bank_g ≥ 0
```
bank_{t−1} := B₀. No relaxation loop: infeasible ⇒ exception + diagnostic
dump (which constraint set is binding). Infeasibility with a legal state is
a bug, not a budget problem.

---

## 5. Purchase-price ledger and the 50% sell-on rule

Real rule: sell = pp + ⌊(mp − pp)/2⌋ rounded down to £0.1m, floor mp if
mp < pp (you sell at market when in loss... precisely: sell = mp − ⌈loss⌉?
FPL rule: if price dropped, sell price = current price; if risen, purchase +
half the rise rounded down). Implementation:

- **Ledger**: `{player_id: purchase_price}` persisted per GW; updated by the
  Executor on executed buys (at mp that week), removed on sells. For the
  2025-26 backtest, reconstruct from GW1 by replaying the transfer log.
- **Pre-solve constants**: sv_p = sell_value(pp_p, mp_p) for p ∈ S₀;
  sv_p = mp_p for all other p (a player bought inside the horizon at mp and
  later sold has zero profit under the constant-price assumption).
- This makes B1 exactly linear. Two facts make it sound:
  1. Prices are assumed constant within the horizon (§8 R6), so
     horizon-bought players never accrue profit inside the plan.
  2. T4 (≤1 in, ≤1 out per player per horizon) means no player needs two
     different sell values within one solve.
- The old `bank += sell − buy` at full market value
  ([season_simulator.py:2003](../pipeline/season_simulator.py#L2003)) is
  deleted. Note for validation: this *lowers* achievable squad value vs the
  2468 baseline — the fair baseline must be re-run under corrected rules
  (§10.2).

---

## 6. Chip reservation guard (the one heuristic that must remain)

A rolling H=5 window cannot see a GW34 double from GW22. Fully extending the
horizon to the set end is intractable and mostly noise. The guard:

1. Each solve, list known far events: announced DGW/BGW weeks g_e in the
   current chip set with g_e > t+H−1.
2. For each unused chip k, estimate far value Ṽ_k(g_e) with the *existing*
   chip-v2 machinery (`build_lookahead_pool` + `_xi_greedy` + FH/WC ILP
   one-shots) — it is already budget-true and loyalty-free.
3. If max_e Ṽ_k(g_e) > (1+λ)·V̂_k(in-horizon best, same estimator), fix
   Σ_{g∈𝒢} k_g = 0 for this solve (reserve the chip). λ = 0.2 margin.
4. Guard disarms automatically when the set deadline enters the horizon.

In live play the far-event list contains only *announced* rearrangements —
the guard degrades gracefully to "no information, no reservation".

---

## 7. Computational tractability

### 7.1 Size estimate (H=5, |P|=180)

- binaries: x,s,in,out = 4·180·5 = 3,600; c,v on |C_g|≈30 candidates:
  300; chips ≤ 20; FH shadow ≤ 2·180+60 = 420 (at most one candidate week).
  ≈ **4,300 binaries**, ~250 integers/continuous, ~15–25k constraints.
- This is a small-to-medium MIP. Expected solve: seconds on HiGHS, tens of
  seconds worst case; CBC may take minutes.

### 7.2 Pruning rules (pre-solve, deterministic, logged)

- Always keep S₀ (15) and any player intel marks as newly relevant.
- Per position, keep top-N by max_g ŝ_{p,g} (N: GK 12, DEF 40, MID 45,
  FWD 25) **plus** top-15 by ŝ/price (value picks), **plus** all players with
  n_fix ≥ 2 in any horizon week priced ≤ 6.0 (BB/bench fodder for doubles).
  Union ≈ 150–200.
- Captain candidates C_g: top-30 by κ_{p,g} among plausible starters.
- Sanity valve: monthly (and in CI) run one unpruned solve and assert the
  pruned solution's objective is within 0.1%. If not, widen N.

### 7.3 Solver, warm start, fallback ladder

- **Solver: HiGHS** (open-source, PuLP-compatible `HiGHS_CMD`, typically
  5–20× CBC on this class). Add to the `fpl-sim` Docker image. CBC remains
  the fallback.
- **Warm start**: feed last week's plan shifted by one week as a MIP start
  (HiGHS accepts solution files). Cheap, big node-count savings, and gives a
  plan-stability diagnostic for free.
- **Time limit ladder** (always produce a decision):
  1. full model, 120 s → 2. gap 1%, 180 s → 3. freeze transfers for weeks
  t+2.. (variables fixed to 0), re-solve → 4. H := 3 → 5. legacy single-GW
  optimizer (`OPTIMIZER="legacy"` flag — the old system stays runnable
  end-to-end for exactly this reason and for the thesis ablation).

---

## 8. Risks and guardrails

| # | Risk | Why real | Guardrail |
|---|---|---|---|
| R1 | **Horizon hurts**: far-week μ noise pollutes week-t decisions (known MPC failure mode) | per-GW-ahead MAE may be flat-to-random beyond +2 | Measure MAE by horizon distance first (Phase 1 exit gate). δ tuned on it. H-sweep {1,2,3,4,5,6} ablation — if H=2 ≈ H=5, ship the smaller |
| R2 | **Early chip burn** via discounting | §4.8 | δ_c separate; chip-timing metric in every backtest report |
| R3 | **Chip–transfer degeneracy**: solver stuffs the GW g bench with DGW fodder purely to inflate y^{BB} | it's partially *desired* (bench building); pathological only if it sacrifices XI | bench build cost is priced (transfers spent, β small); monitor "XI points sacrificed in chip build-up weeks" metric; H_cap on hits bounds the damage |
| R4 | **End-of-horizon distortion**: transfers at t+H−1 undervalued (payoff truncated) | structural | rolling re-solve executes week t only; optional terminal value ε·V_p (ablation); H-sweep detects it |
| R5 | **Plan thrash**: plan flips weekly, churning week-t decisions | re-solve each week with new noise | plan-stability metric (Jaccard of planned week-t+1 transfers vs realized); if < 0.5, raise δ or add tiny plan-change penalty (documented tunable, last resort) |
| R6 | **Constant-price assumption**: real prices move ±0.1–0.3 within 5 GWs | sell-price ledger and budget drift | executor re-syncs prices weekly; error is bounded by ±0.3·n_transfers; do NOT attempt price prediction in this phase (§11) |
| R7 | **β/γ/w̄ miscalibration** silently distorts bench/vice pricing | new coefficients, no history | each is measured from historical data, not tuned to points; calibration notebooks are deliverables (Phase 5) |
| R8 | **q90 lottery captains**: ceiling term picks 3-game-sample variance monsters | σ̂ noisy for small n | σ shrinkage (m=5), θ blend, π factor in κ; captain-regret metric |
| R9 | **Intractability spikes** (FH shadow + DGW weeks + big pool) | occasional | fallback ladder §7.3; alert if ladder step ≥ 3 fires |
| R10 | **Losing to the 2468 baseline on 2025-26** | the 25 old constants were *fit to this exact season*; a general system can lose to an overfit one in-sample | success criteria are defined against the corrected-rules baseline AND cross-season (§10); thesis frames this explicitly |
| R11 | **State ledger corruption** (purchase prices) | new persistent state | ledger is append-only events + derived snapshot; invariant checks each GW (bank ≥ 0, Σ prices consistent, 15 players); golden tests |
| R12 | **Infeasibility crashes** where old code silently relaxed | stricter rules | pre-solve feasibility lint (budget sanity, club counts); exception carries the IIS (irreducible infeasible subsystem) if available |

---

## 9. Build plan — incremental, always-working

Flag-gated: `OPTIMIZER = "legacy" | "mp"` (mirrors `CHIP_STRATEGY`).
Each phase merges only with its exit criterion green. **Blocker inherited
from the repo state: `data/raw/` lives only on the original machine; all
backtests run there or after copying (Docker `fpl-sim`, add HiGHS).**

- **Phase 0 — Harness + fair baseline.**
  Extract a season-runner that takes any optimizer behind one interface and
  emits the §10.3 metrics. Re-run legacy under corrected rules (real sell
  prices, no budget relaxation, no FT-cap-2) → this number, not 2468, is the
  fair baseline. Golden unit tests for ledger math (sell price table from
  FPL docs, FT recursion cases, bank invariants).
  *Exit: legacy reproduces 2468 under old rules; corrected-rules baseline
  recorded.*
- **Phase 1 — Prediction matrix.**
  Feature-swap per-GW predictions, per-fixture DGW sums, blank zeros, π, φ,
  q90, sv. Standalone module + calibration report (MAE by horizon distance;
  φ-bucket MAE; q90 coverage ≈ 90%).
  *Exit: MAE(+1) ≤ current model's next-GW MAE; blank/DGW spot-checks exact;
  φ buckets separate or φ dropped (gate §3.5).*
- **Phase 2 — New objective at H=1.**
  Single-week MILP with bench EV, in-ILP captain+vice, correct rules, all
  fudge constants removed. Isolates "structure vs hacks" before any horizon
  effect. Chips still via v2 scheduler.
  *Exit: GW1-38 within −3% of fair baseline (expected: roughly par — the
  hacks were doing real work; parity with 8 constants instead of 25 is a win).*
- **Phase 3 — Horizon H>1, transfers only.**
  Full §4 minus chip variables (chips stay v2). The core deliverable.
  *Exit: beats Phase 2 on points AND transfer metrics (§10.3) on 2025-26;
  H-sweep documented.*
- **Phase 4 — Chips in-model + reservation guard.**
  §4.5–§4.8, §6. Delete v2 scheduler from the loop (keep for guard estimates).
  *Exit: all 8 chips placed on ≥2 seasons' calendars, zero spacing
  violations, chip-quality metric ≥ v2's.*
- **Phase 5 — Uncertainty refinements + calibration notebooks** (φ shrinkage
  live, θ/γ/w̄ measured, q90 captaincy). Each behind an ablation flag.
- **Phase 6 — Tuning done right.**
  Optuna over the 8 constants (H, δ, δ_c, θ, γ, w̄, λ, H_cap) with a
  **cross-season objective**: maximize Σ points over 2023-24 + 2024-25,
  test untouched on 2025-26 (and vice-versa folds). Never single-season
  tuning again. Report in-fold vs out-of-fold gap as the overfit measure.

---

## 10. Validation plan

### 10.1 Correctness (before any points comparison)
- Rules audit on every backtest: bank ≥ 0 every GW, FT ∈ [1,5] with correct
  recursion vs a straight-Python simulator of the rules, sell prices match a
  hand-computed table, ≤1 chip/GW, chips within windows, no player in/out
  same week, XI always legal, no blank-week starters unless unavoidable.
- MILP self-checks: aux vars at their bounds (z = W etc.), no phantom hits
  (h_g == max(0, n−f) on non-chip weeks), pruning-valve solve within 0.1%.

### 10.2 Same-season performance (2025-26)
- Compare vs **fair baseline** (Phase 0). Report: total points, and the
  decomposition below. Also run new optimizer under *old* lenient rules once,
  purely to quote against 2468 in the thesis with both numbers footnoted.

### 10.3 Metric suite (each per season, each phase)
- **Points**: total, per-GW mean/std, predicted-vs-actual gap.
- **Transfer efficiency**: Σ over executed transfers of (in − out actual
  points over following 4 GWs) / transfer count; hold-counterfactual: points
  of the "no transfers after GW1" team as a floor reference.
- **Hit ROI**: per hit, 4-GW realized gain − 4. Target: mean > 0.
- **Churn**: buybacks ≤ 6 GWs (baseline: 4), holds ≤ 2 GWs (baseline: 11),
  FT utilization histogram (does banking actually happen pre-events?).
- **Captain**: mean captain-regret = best-XI-player actual − captain actual
  (baseline measurable from `season_simulation.json`); captain-miss rate
  (captain 0-pointers — baseline GW38 Haaland case would be caught by vice).
- **Chips**: realized chip delta vs best-alternative-week-in-set
  (post-hoc oracle); spacing violations = 0; chips used = 8.
- **Bench**: auto-sub points gained; bench points wasted on non-BB weeks.
- **Plan stability**: Jaccard(planned t+1 transfers at solve t, executed at
  t+1); chip plan flip count.
- **Prediction**: MAE by horizon distance; φ-bucket calibration; q90
  coverage.
- **Rank proxy**: percentile vs the public per-GW score distribution
  (FPL analytics dumps), reported per season — the thesis-facing number.

### 10.4 Generalization (the real test)
- Full runs on 2023-24 and 2024-25 calendars (raw data in repo per
  vaastav). Success: (a) no crashes/violations; (b) chips land on
  event-plausible weeks on *both* calendars; (c) out-of-fold points drop
  ≤ 5% vs in-fold after Phase 6 tuning; (d) beats the legacy optimizer
  *with its constants frozen as-is* on both held-out seasons — the legacy
  system's cross-season decay **is** the thesis argument.
- Ablation table for the writeup: H ∈ {1..6} × {chips in-model vs v2} ×
  {φ on/off} × {q90 on/off} × {δ_c = δ vs 0.97}.

---

## 11. Explicitly out of scope (documented future work)
- Price-change prediction / team-value farming (R6 accepts bounded error).
- Effective-ownership / rank-aware objective (shield-vs-differential play).
- Per-fixture minutes model beyond π (substitution-pattern modelling).
- Ownership as a model feature (touches Stage 6/7 training pipeline).
- Stochastic programming / scenario trees — the φ/δ/q90 machinery is the
  deliberate cheap approximation; revisit only if R1 measurements say the
  deterministic horizon is the binding error source.

## 12. The 8 constants that remain (and their meaning)

| Constant | Start | Meaning | Set by |
|---|---|---|---|
| H | 5 | horizon length | ablation sweep |
| δ | 0.84 | forecast-uncertainty discount | MAE-by-distance, then Optuna (§9 P6) |
| δ_c | 0.97 | chip-term discount | ablation |
| θ | 0.5 | captain mean↔ceiling blend | Optuna cross-season |
| γ | 0.07 | captain-miss prob for vice term | measured from history |
| w̄ | ~0.12 | bench-slot activation prob | measured from history |
| λ | 0.2 | reservation-guard margin | ablation |
| H_cap | 2 | max hits/week guardrail | relax when Hit-ROI > 0 proven |

Everything else that used to be a constant is either a rule (in config), a
measured calibration (notebook artifact), or gone.
