# Recommendation Layer — Design Blueprint

Status: **DESIGN — not yet implemented** (2026-07-06)
Scope: a presentation-and-interrogation layer over the decision core. Turns
the system from a fully autonomous manager into a hybrid: **autonomous by
default, human veto through a structured gate, everything logged.**
Depends on: `optimizer_redesign.md` (the multi-period MILP is the primary
content generator for this layer — build order is Phase 0 → MILP → this).
Related docs: `chip_strategy_redesign.md` (v2 value functions are surfaced
directly in the chip advisory, §3.3).

---

## 0. Design goal in one paragraph

Each gameweek, produce a report and a conversational drill-in that let the
manager make every deadline decision in ~10 minutes with full confidence:
the MILP's plan presented as a plan (not a fait accompli), every
recommendation ranked by confidence (perturbation-tested, §2.3), every
number traceable to a model output, every override logged and later scored.
The system is NOT a chatbot with opinions and NOT a dashboard of raw model
output — it is the optimizer's own reasoning, surfaced instead of discarded,
plus the two things the optimizer cannot do: admit out-of-data information
(broken-leg overrides, §6) and reason about rank strategy (§3.2, §8).

---

## 1. Why this exists (and the honest premises)

1. The autonomous system makes decisions a human would never make. Some of
   those are its edge (unsentimental sells, off-template captains like the
   GW21 Garner call). Some are failures of data coverage (the intel_02
   Newcastle blind spot). A recommendation layer must separate the two
   instead of letting the human veto both.
2. The clinical-vs-actuarial literature (Meehl 1954; Grove & Meehl
   meta-analyses) says mechanical prediction beats or ties expert judgment
   in most domains, and that free-form human overrides usually make the
   combination WORSE than the model alone. The exception is the
   **broken-leg case**: discrete, verifiable, outside-the-model information.
   FPL has real broken-leg cases weekly (late injury news, manager quotes,
   uncovered clubs). The hybrid is designed around admitting exactly those
   and resisting everything else.
3. The 2468-pt backtest was tuned (341 Optuna trials) on the season being
   scored; the live out-of-sample edge is unproven. This layer does not fix
   prediction quality and must not be sold as if it does. What it CAN fix:
   information gaps, rank-awareness, and trust/verifiability.
4. **Timeline fact that shapes everything:** deadline is 90 min before
   first kickoff; official lineups drop 60 min before kickoff — i.e. 30 min
   AFTER the deadline. There is no post-lineup decision window. All
   decisions are made under lineup uncertainty; minutes probability (π) is
   a modeling problem, not a waiting problem.

---

## 2. Architecture

```
┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│ Prediction matrix    │→ │ Multi-period MILP    │→ │ Decision artifacts    │
│ μ, π, φ, q90         │   │ (optimizer_redesign) │   │ horizon plan, chip    │
│ (LGBM → +GRU/GNN)    │   │ runs on REAL squad   │   │ values, sensitivity   │
└─────────────────────┘   └─────────────────────┘   │ re-solves, alternates │
        ↑                                            └──────────────────────┘
┌─────────────────────┐                                       ↓
│ Intel 01–04 + NEW:   │                             ┌──────────────────────┐
│ intel_08 top-10k EO  │                             │ Report generator      │
│ intel_09 price watch │                             │ gw_report_<GW>.json   │
└─────────────────────┘                             └──────────────────────┘
        ↑                                                     ↓
┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│ Override / calib     │← │ Human decisions      │← │ Gemini narrative +    │
│ ledger + shadow      │   │ via structured gate  │   │ rendered report +     │
│ season (§7)          │   │ (§6)                 │   │ conversational layer  │
└─────────────────────┘   └─────────────────────┘   └──────────────────────┘
```

### 2.1 Non-negotiable data-first rule

The report is generated as a single structured JSON per GW
(`data/intel/gw_report_<GW>.json`) containing every number, ranking, and
MILP artifact. Human-readable rendering (HTML/markdown), Gemini narration,
and the conversational layer are all derived from it. Reasons: (a) the
conversational layer needs a grounding document, (b) the ledger (§7) scores
recommendations after the fact and needs them in machine form, (c) Gemini
is constrained to it (§5).

### 2.2 What the MILP already computes and currently throws away

| Report need | Existing internal artifact |
|---|---|
| "How long to hold this buy" | horizon plan weeks t..t+H−1 |
| "Chip now vs. later opportunity cost" | reservation guard value comparison |
| "Is this transfer forced or marginal" | objective delta between plan A and constrained re-solve |
| "What the system is doing about the BGW" | horizon plan + reservation guard state |
| "Risk if captain blanks" | trivial recompute from μ/π matrix |

The layer's job is to persist these instead of discarding them after
executing week t.

### 2.3 The one new computation: confidence via perturbation

The MILP returns one optimal plan. A recommendation needs alternatives and
fragility. Per GW, run a small batch (~5–10) of constrained re-solves:

- captain forced to each of the top-k alternatives
- each recommended transfer individually forbidden
- recommended chip forced OFF (and best alternative week forced ON)
- availability of any player with intel score < 70 set to 100 (i.e. "what
  if he's passed fit Friday")

Confidence tiers from the diffs:
- **HIGH** — plan unchanged under all perturbations, or objective delta of
  the best alternative > TAU_HIGH pts
- **MEDIUM** — plan stable but best alternative within TAU_HIGH
- **LOW** — any single-input perturbation flips the plan

LOW-confidence decisions and all high-stakes decisions (§6) escalate to
mandatory human review. Confidence tiers must be visually load-bearing in
the rendered report — fluent narration makes weak recommendations feel as
solid as strong ones otherwise (§9, failure mode 3).

---

## 3. The weekly report — seven sections

Every section lists its feeds and its gaps. Gaps are real; two require new
pipeline components (§4).

### 3.1 Transfer plan (not transfer picks)

The horizon plan as a narrative plan: sell/buy this week, WHY (fixture run
shown explicitly for t..t+3), how long the plan holds the incoming player,
what the plan already intends to do in t+1..t+H−1 (e.g. "sells Z in GW+2
before his blank"). Per move:
- expected points gained over the horizon, FT/hit cost
- **fragility**: which single input drives the move, from the perturbation
  batch ("80% of this move's value is X's availability score of 40; passed
  fit → move loses most of its value")
- confidence tier
Feeds: MILP plan, prediction matrix, intel_03/04. Gaps: none structural.

### 3.2 Captain matrix

Top 4–5 candidates × dimensions that actually differ:

| dimension | source |
|---|---|
| μ (expected pts) | prediction matrix |
| q90 (ceiling) | prediction matrix (redesign §3) |
| π (plays 60'+ / 90') | intel_03/04 + start-rate history |
| H2H record this fixture | vaastav historical |
| top-10k effective ownership | **intel_08 (NEW, §4.1)** |

One recommended pick with the tradeoff STATED, not resolved silently:
"Haaland at 55% EO is rank-neutral protection; Semenyo at 8% EO, lower μ,
similar q90, is the pick if chasing rank." Vice = highest-π among remaining
premiums (vice only matters when captain doesn't play — mechanical rule).
Gap: top-10k EO does not exist in current data (FPL API gives overall
ownership only). Load-bearing for §3.2, §3.5, §3.7 and any future
rank-aware objective — first new component to build.

### 3.3 Chip advisory

Surface chip-v2 value functions (currently consumed silently by triggers)
as a table per chip: value THIS week, best future window in the set and its
projected value, verdict (play/hold) with the reservation-guard reasoning.
BB additionally shows the actual bench's projection for each candidate week
under the current plan. SPACING_GAP is explained, not just enforced (the
guard that would have caught the manual FH GW23 → WC GW24 mistake now says
WHY). Feeds: chip v2 value functions, reservation guard, fixture calendar.
Gaps: none.

### 3.4 Squad health

Per squad player: form trajectory (rolling stats now; GRU later makes this
richer), next-4 FDR run, availability score WITH SOURCE ("Tue presser:
'we'll assess him'"), rotation risk with the signal behind it, price-change
pressure (**intel_09, §4.2**).

**Coverage disclosure (turns a known bug into a hybrid feature):** intel_02
only detects clubs appearing as FFS section headers (documented Newcastle
blind spot). The report must state per-GW press coverage explicitly —
"press data covers 16/20 clubs; NO signal for: Newcastle, Brentford, …" —
telling the human exactly where their own news-reading adds value. This is
the primary designed channel for legitimate information overrides (§6).

### 3.5 Differentials

Players < 5–10% top-10k EO whose μ or q90 ranks top-N at position over the
next 3 GWs. Framed by CEILING (q90), not mean, with the variance framing
explicit: "you take these to climb, not to protect." Feeds: prediction
matrix + intel_08. Gap: intel_08.

### 3.6 Blank/double horizon

6-GW calendar strip: announced blanks/doubles, which OWNED players are
affected, what the horizon plan is already doing about it, what the
reservation guard is holding chips for. Mostly narrates decisions the
optimizer already makes silently — the point is to stop those surfacing as
"weird transfers" with no explanation. Gaps: none.

### 3.7 Risk dashboard

Four traffic-light numbers + one aggregate:
1. players with π < ~0.8
2. players with FDR ≥ 4 this GW
3. max single-team concentration
4. captain-blank delta (score with captain at 2 pts vs expected)
5. squad expected pts vs projected top-10k template squad (needs intel_08)
   — "are you positioned ahead of or behind the pack this week"

### 3.8 The closing box

Mandatory final element: **"Do these N things. Confidence: X."** One box,
≤ 3 actions. Guards against failure mode 4 (§9): seven sections of
tradeoffs recreating the hour of uncertainty with better production values.

---

## 4. New pipeline components

### 4.1 intel_08 — top-10k effective ownership

Scrape LiveFPL (or equivalent) for top-10k EO per player per GW. Store
alongside intel_01 output. Consumers: captain matrix, differentials,
template comparison, and eventually an EO-aware term in the MILP objective
(§8). Buildable NOW, independent of the optimizer redesign. Archive per-GW
snapshots — EO history is needed for the shadow season (§7.2) and cannot be
backfilled.

**✅ IMPLEMENTED 2026-07-14** — `pipeline/intel_08_effective_ownership.py`
+ `tests/test_intel_08.py` (12/12). Source resolved by live investigation:
LiveFPL's public data host serves `https://livefpl.us/top10k.json` (and a
tighter `elite.json`) — a JSON map of **FPL element id → EO fraction**,
already keyed by element id so no name resolution is needed. Values exceed
1.0 for captained players (B.Fernandes 1.37 at GW38), confirming these are
effective ownership (ownership + captaincy), not plain ownership. The
snapshot's canonical `eo` field is the top-10k figure (design requirement);
`eo_elite` and `eo_overall` (from bootstrap `selected_by_percent`) ride
alongside for the differential framing. GW tag comes from bootstrap
current/next event. Outputs `data/intel/effective_ownership.json` (latest) +
`data/intel/eo_history/gw{N}.json` (per-GW archive, latest-wins). Query
helpers `eo_of` / `differentials` / `template` are the §3.2/§3.5/§3.7 feeds.
First live snapshot captured (GW38 2025-26, 840 players): e.g. Bowen 79%
top-10k EO vs 17.5% overall — the template signal plain ownership hides.
Degrades gracefully: elite/bootstrap failures still save the raw top-10k EO.
NOTE the season-start urgency is now live — the moment livefpl.us flips to
2026-27 GW1, only snapshots taken from that point exist; schedule this to run
each GW (belongs with the press-scraper scheduler, press redesign step 5).

### 4.2 intel_09 — price-change watch

Pragmatic v1: scrape LiveFPL's price predictor (rise/fall probability
tonight / within 3 days). Own model from intel_01 transfer-pressure data is
a nice-to-have later. Consumer: squad health §3.4 ("target before the
rise") and the T−3h delta watch (§10).

### 4.3 Override & calibration ledger (see §7)

Trivial to start logging, impossible to backfill — start with the first
live GW even if nothing else is built.

---

## 5. Gemini narrative layer — division of labor

Gemini (intel_05's `gemini-2.5-flash` lineage) writes the connective
tissue, never the facts.

**Hard rules:**
1. Input = the report JSON. Every number, player, and claim in the prose
   must exist in the JSON. No new facts. (An LLM fluently rationalizing a
   number it didn't compute is failure mode 3, §9.)
2. Output sections mirror JSON sections; rendered report interleaves
   narration with the actual tables so prose is verifiable at a glance.
3. Temperature 0; narration regenerated whenever the JSON changes (delta
   watch, §10).

**Gemini's one genuinely additive job — soft intel:** read raw
press-conference text (intel_02's scraped articles, pre-keyword-matching)
and flag qualitative signals the scrapers miss ("manager's phrasing on X
sounded like rotation"). Output goes into the report as clearly-labeled
SOFT INTEL, visually separate from model output, never fed into μ/π
automatically. It is input for the HUMAN, and a candidate source of
legitimate overrides.

**Conversational layer:** chat grounded on the report JSON + underlying
data, with one power tool: a constrained MILP re-solve. "What if I keep
Saka?" = `forbid transfer(Saka out) → re-solve → diff plans → Gemini
narrates the diff`. Counterfactuals answered with the same rigor as the
main recommendation, not speculation. Rate-limit: re-solves are cheap
(seconds) but cap at ~10/GW to keep the 10-minute promise honest.

---

## 6. The decision protocol (the hybrid, precisely)

**Autonomous by default, human veto through a structured gate.**

1. **Default-execute tier:** HIGH-confidence, low-stakes decisions (routine
   transfer, obvious captain) are presented as decisions, not questions:
   "Doing X unless you object by T−2h."
2. **Mandatory-review tier:** any chip, any hit ≥ 8 pts, anything the
   perturbation test marks LOW, any decision touching a player from an
   uncovered club (§3.4 disclosure). These require explicit human
   confirm/override before the deadline.
3. **The override gate** asks exactly one question: **"What do you know
   that the model doesn't?"** Answer is required and stored. Two labels:
   - `information` — discrete, verifiable, outside-the-data (true
     broken-leg: "Isak ruled out on club site; intel missed it"). Also
     auto-files a coverage gap against the intel pipeline.
   - `judgment` — everything else ("don't trust Garner as captain").
     Allowed — it's the user's team — but labeled, because the literature
     says these overrides are net-negative at base rates and the ledger
     will test that on OUR data.

---

## 7. Learning loop

### 7.1 Override & calibration ledger

Per GW, append-only log of: every recommendation (with confidence tier),
the human decision, the override label + stated reason (if any), and — once
actuals land — the point outcome of recommendation-as-given vs
decision-as-made. Weekly scoring; season-end decomposition of every
divergence into information vs judgment overrides and their point values.
Calibration tracking on confidence tiers: HIGH should flip less often than
MEDIUM ex post, and stated probabilities (π) get a reliability curve from
the T+30min lineup snapshots (§10.4).

### 7.2 Shadow season

The fully autonomous system plays the entire season in parallel on the same
data (same prediction matrix, same intel, no human). Season end yields:
actual rank, shadow rank, and the ledger's divergence decomposition. Nobody
has real evidence on human+model in FPL; after one season we will. If
judgment overrides are net-negative (base-rate expectation), the hybrid
degrades gracefully toward pure-autonomous by tightening the gate — the
architecture self-corrects toward whichever pure form is actually better.
(Also a genuinely novel thesis-adjacent artifact.)

---

## 8. Rank-awareness (flagged, not designed here)

Top 100 is a rank target, not a points target, and rank optimization is
game-theoretic: behind → positive-variance low-EO positions; ahead →
template shielding. No part of the current system treats ownership as a
strategic variable (OWN_BOOST_GW1 is a prediction nudge). The report
surfaces the raw material (EO everywhere, q90 framing, template
comparison §3.7.5) and leaves strategy to the human FOR NOW. The proper fix
is an EO-aware term in the MILP objective — deliberately out of scope for
both this doc and optimizer_redesign v1; revisit after the fair baseline
and intel_08 history exist. This gap is a modeling gap wearing a
human-judgment costume; do not let it become a permanent argument for
manual control.

---

## 9. Failure modes and their designed mitigations

| # | Failure mode | Mitigation |
|---|---|---|
| 1 | Algorithm aversion (Dietvorst): two visible model errors in Sep → discounting it in Dec | Ledger makes the running score explicit; shadow season settles it with data, not memory |
| 2 | Override without learning — same argument relitigated weekly | Gate requires stated reason; ledger scores every override |
| 3 | Narrative-induced overconfidence — Gemini makes weak recs feel solid | No-new-facts rule (§5); confidence tiers visually load-bearing (§2.3) |
| 4 | Decision fatigue reintroduced — 7 sections recreate the hour of uncertainty | Closing box (§3.8): ≤3 actions, one confidence statement |
| 5 | Alert noise → alerts ignored | Alert only on RECOMMENDATION CHANGE, never on news (§10.3) |
| 6 | Silent intel blind spots corrupt decisions | Per-GW coverage disclosure (§3.4) routes the gap to the human |
| 7 | Hindsight bias ("sold before a haul") drives distrust | Ledger scores decisions on information available at the time; report archives the JSON so the ex-ante case is reviewable |

---

## 10. Timeline / cadence per GW

| When | Artifact | Content |
|---|---|---|
| **T−48h** (≈Thu eve) | Draft report | Full pipeline on available data; marked PROVISIONAL; purpose: sleep on big calls (chips, hits) |
| **T−12..−6h** | **Final report** | All pressers in, predicted lineups in, overnight prices settled. The decision document. Default-execute clock starts. |
| **T−3h → deadline** | Delta watch | No new report — a diff stream. Re-run MILP only on material input change; **alert only if a recommendation changes.** "Palmer 75% flag → plan unchanged" = silence. |
| **T+30min** (lineups) | Accountability snapshot | Nothing actionable (deadline passed 30 min ago — §1.4). Records: π hits/misses, predicted-lineup accuracy, would-vice-have-fired. Feeds ledger + next week's minutes estimates. This is the weekly ground-truth event for the minutes stack, not a "damage report". |

---

## 11. Build order & scope

Prerequisite chain: **Phase 0 fair baseline → multi-period MILP →
this layer.** Building the report over the current 25-constant simulator
would inherit exactly the opaque decisions that motivated it.

Independent of that chain — start anytime:
1. ✅ **intel_08** top-10k EO scraper (+ per-GW archiving; cannot backfill) —
   DONE 2026-07-14 (§4.1); needs scheduling each GW at season start
2. **Ledger** schema + logging (cannot backfill)
3. **intel_09** price watch (small)
4. **intel_10** preseason/launch snapshot (§13.2) — August-window only,
   cannot backfill; same urgency class as intel_08

After the MILP lands, the layer itself is a ~3–4 week presentation job:
1. Persist MILP artifacts instead of discarding (small change to executor)
2. Perturbation batch + confidence tiers (§2.3)
3. Report JSON schema + generator (§3)
4. Gemini narration with the no-new-facts contract (§5)
5. Rendering + delta watch + alerting (§10)
6. Conversational layer with constrained re-solve tool (§5) — last; the
   report alone already delivers most of the value

New tunable constants introduced: TAU_HIGH (confidence threshold),
alert-materiality threshold, re-solve cap. Keep ≤ 4, in the spirit of the
redesign's ≤ 8.

---

## 12. Verdict recorded (from the 2026-07-06 design discussion)

Ranked by expected chance of top-100 across seasons:
1. **Hybrid (this doc):** autonomous core + structured veto + ledger.
   Keeps the model's consistency, admits human information exactly where
   the pipeline is provably weakest, and is the only architecture that
   MEASURES whether the human helps — so it self-corrects.
2. **Pure autonomous:** close second; ceiling set by prediction quality
   and the missing rank-aware objective, not by absence of a human. The
   hybrid degrades into this gracefully if the ledger says so.
3. **Pure recommendation (human decides all):** weakest for rank —
   reintroduces noise, tilt, and plan abandonment across 38 deadlines
   while keeping the model's prediction errors anyway. Best for enjoying
   FPL; worst of the three for climbing.

---

## 13. GW1 cold-start mode (added 2026-07-06)

Second entry point: a human with NO existing team builds the entire GW1
squad from scratch (100.0m), then flows into ongoing mode (§§3–10).
Grounding fact: the system already solves this — every simulation run's
GW1 step selects 15 from the full pool with an empty initial squad, and
`OWN_BOOST_GW1` exists precisely because cold-start predictions alone were
found (empirically, by Optuna) not trustworthy enough without a crowd
prior. The optimization machinery exists; what is new is the cold-start
prediction treatment (§13.2), preseason intel (§13.2), and presentation
under maximum uncertainty (§13.1).

### 13.1 Formulation: same MILP, degenerate initial state

- **Not a separate formulation.** Multi-period MILP with S0 = ∅: week-1
  degenerates to "buy 15 within 100.0" under the unchanged constraint set;
  transfer-cost/sell variables zeroed (no hits possible). Horizon
  objective (GW1–5) applies unchanged and is ESSENTIAL: the opening
  fixture run is the most reliable data available at GW1 (fixtures are
  certain when nothing else is).
- **Combinatorics are a non-issue** — 15-from-~600 is the textbook FPL ILP
  (stage 8 solves it in seconds). REJECTED: two-stage
  budget-per-position-then-pick decomposition — a heuristic the exact
  solve strictly dominates.
- **The real GW1 problem is a flat objective landscape**: with every μ at
  maximum uncertainty, dozens of squads sit within the noise band.
  Response:
  1. **Enumerate near-optima** (no-good cuts / solution pool, ~10–15
     solves), cluster into 2–3 **squad archetypes** (premium-heavy /
     balanced / differential-tilted) presented with tradeoffs. GW1
     analogue of §2.3 confidence — same philosophy, enumeration instead of
     perturbation because at GW1 EVERYTHING is perturbed.
  2. **Robustness selection**: re-solve under resampled prediction
     matrices (bootstrapped priors, jittered continuity weights); report
     each archetype's stability count. Squads appearing across resamples
     are structural; one-draw squads are noise artifacts.
- **GW1 captaincy inverts the usual weighting**: μ/q90 orderings are noise
  at GW1, so rank by multi-season consistency + continuity (same club/
  role/set pieces, proven premium) and treat template EO as a feature —
  with no information edge over the field, differential captaincy is
  maximally unjustified. Deviate least when you know least. Vice rule
  unchanged (highest-π), π from continuity + preseason starts.
- **State the WC1 escape hatch in the build report**: GW1 mistakes are the
  season's cheapest — "if 3+ of the 15 bust by GW4–6, WC1 resets at zero
  cost; do not agonize past the archetype choice." Protects the 10-minute
  promise at the deadline where over-deliberation temptation peaks.

### 13.2 Cold-start prediction: the prior IS the prediction

Shrinkage φ (optimizer_redesign §3.5) at its limiting case: φ minimal for
ALL players simultaneously → GW1 quality is entirely determined by prior
quality. Design work is all in the prior:

**Continuity-weighted prior-season carry** — explicit tiers as
weight/multiplier on prior-season features AND on φ itself (Tier A gets
real GW1 confidence; "everyone is cold" must not mean "everyone is
equally cold"):

| Tier | Situation | Carry | Status |
|---|---|---|---|
| A | same club + manager + role | ~0.85+ | NEW: continuity score |
| B | same club, new manager | discounted | NEW: manager-change table (~20 rows/summer, manual) |
| C | new club, same league | moderate | stage 4a machinery exists |
| D | new-league arrival | lowest | stage 4b `prev_league_multiplier` exists |
| E | promoted-team player | league-adjusted, widest σ | partially stage 4b |

**Two crowd priors, GW1-strength only, decaying to zero by ~GW5:**
- **Launch price** — FPL's pricing team is an expert forecaster; at GW1
  price enters the PRIOR (weighted up), not just the feature set.
- **Launch EO/template** — the crowd aggregates preseason info no scraper
  fully captures. `OWN_BOOST_GW1=0.213` is the crude Optuna-validated
  ancestor; principled version: shrink μ toward a crowd-implied prior
  (price- + EO-implied points blend), weight → 0 as φ ramps on real data.
  This is the same mechanism that replaced the deleted "GW2–8 blend" hack,
  pointed at a better prior target.

**intel_10 — preseason/GW0 intel (NEW), graded honestly by data quality:**
1. Transfer window tracker (→ continuity tiers, positional competition) —
   structured, reliable, partially exists (transfermarkt data).
2. Preseason friendly lineups/minutes — best available π signal (final two
   friendlies ≈ GW1 XI) but SCRAPPY: no structured API, popularity-bias
   risk like intel_02 → soft intel with mandatory coverage disclosure
   (§3.4 pattern).
3. Manager preseason quotes — unstructured → Gemini SOFT INTEL per §5
   contract, never into μ/π automatically.
4. **Launch price + launch EO snapshot — August window only, cannot be
   backfilled.** Same urgency class as intel_08.

**Objective under uniform low confidence:** shrinkage arithmetic naturally
shifts decision weight toward what is actually known (fixtures, team
strength, price structure, continuity) — no new objective term. But VERIFY
it: required test — GW1 predictions under full shrinkage must correlate
strongly with the crowd/price prior and deviate mainly on Tier-A players
(where multi-season model signal is real). A miscalibrated φ floor at GW1
lets noise masquerade as signal across the entire pool at once.

### 13.3 Transition into ongoing mode (deliberately boring)

The build's output IS a state ledger: purchase prices = GW1 buy prices
(sell = buy initially), bank = 100.0 − spend (leaving ITB is allowed —
MILP decides), FT init per FPL rules (none at GW1, 1 FT granted for GW2,
banking from there — `fpl_rules.py` domain), chips all unused, reservation
guard at full inventory. GW2 horizon plan starts from the built squad with
zero special-casing.

Two explicit transition features:
- **Cold-start watch panel (GW2–6 reports):** which of the 15 are
  confirming their prior (starts/minutes/role) vs diverging, framed
  against WC1 — "2/15 busts; WC threshold ~4." Makes the classic early-WC
  correction a quantitative trigger, not a panic response; plugs into the
  chip advisory's value-function framing.
- **Mid-season entry = FPL team ID.** The entry endpoints expose full
  picks, transfer history, and chip usage → reconstruct the exact
  purchase-price ledger, FT count, chip state. Canonical onboarding;
  manual squad entry is fallback for drafts/hypotheticals only.

### 13.4 Both entry points, one infrastructure — confirmed

Same report JSON schema with ONE section swapped: transfer plan (§3.1) →
**squad build section** (15 buys with per-player rationale: continuity
tier, prior source, fixture run, archetype membership) + archetype
comparison. Chip advisory renders as a season-opening roadmap instead of a
this-week verdict; all other sections unchanged. Conversational re-solve
works identically ("swap X for Y" = force/forbid → re-solve → diff) and is
MOST valuable here: on a flat landscape the honest answer is often "costs
0.8 pts over 5 GWs — defensible preference," which builds calibrated
trust. Ledger logs GW1 build overrides like any other → shadow season
starts at GW1, not GW2.

### 13.5 What is genuinely harder at GW1 — stated plainly

1. Prediction quality is the season's worst, irreducibly (GW1-blind rule
   already acknowledges this). Posture: "2–3 defensible squads and why,"
   never "the optimal squad."
2. Flat landscape → presentation IS the product; hiding near-ties means
   wrong-with-confidence in week one — the worst trust start (§9 #1).
3. Preseason intel is the scrappiest data in the whole design:
   unstructured, popularity-biased, partially paywalled. Coverage
   disclosure mandatory from day zero.
4. Cannot backfill: launch price/EO snapshot (intel_10) joins intel_08 EO
   history in the "start before August or lose it forever" bucket.
5. New data needed: manager-change table (tiny, manual), continuity score
   (derivable from existing transfer data), friendlies scrape, launch
   snapshot. Everything else reuses stages 4a/4b + existing intel
   patterns.
