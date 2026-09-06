# Player-Identity & Context Features — Design

Status: **§2 (career-level player quality) IMPLEMENTED 2026-09-04.**
§3 (new-manager flag) and §4 (transfer-sentiment) remain **DESIGN — not yet
implemented**, per the sequencing in §5 below.
Scope: adds features to the LightGBM prediction layer (`feature_engineering_stage6.py`
training rows + `season_simulator.py` live pool-building + both `FEAT_COLS` /
`DEFAULT_FEAT_COLS` lists) so the model can distinguish an established,
proven player from a generic player with the same 1-2-gameweek current-season
stat line. Does not touch the MILP core (`milp_core.py`) or the optimizer
objective — this is purely a prediction-input change.

**§2 implementation notes (2026-09-04):** shipped as designed — 4 fields
(`career_price_tier_rank` cut, per §7's own recommendation), `build_career_lookup`
+ `build_career_snapshot` in `feature_engineering_stage6.py` (§2.1), wired into
`FEAT_COLS`/`DEFAULT_FEAT_COLS` and `build_rolling_pool`/`build_retrain_rows`
via a cached `_lookup_career()` in `season_simulator.py` (`build_gw1_pool`
needed no change — it already picks career_* up via `train_dfs`). Stage 6's
Check 11 needed a mid-implementation fix: `career_seasons_established == 0`
does NOT imply "no prior data" (a player can have real sub-900-minute
prior-season output), so the real leakage check is against each player's
GLOBAL first season, not a proxy field. Validated: all 11 Stage 6 checks
pass; 6-fold walk-forward MAE 2.403 (no regression vs. the prior documented
2.34-2.44 range); and the motivating case is fixed — re-running the live
GW3→GW4 recommendation with the retrained models, the optimizer now holds
Haaland instead of proposing to sell him.

Related docs: [[optimizer-redesign]] (the MILP consumer of these predictions),
[[feature-engineering]] (Stage 6, the component this modifies),
[[walkforward-no-leakage]] (the rule every feature below must satisfy).

---

## 0. Why — the case that exposed the gap

2026-27 GW3 live run (this session, team 7426041): the `mp` optimizer
recommended selling Haaland (15 pts across GW1-2, FDR2 home fixture next)
for Isak (10 pts across GW1-2, £6.5m cheaper). Diagnosis (full trace in this
session's transcript, reproducible via a debug script against
`prediction_matrix.build_matrix`):

1. FDR was correct and favourable (2.0) — not the cause.
2. `value` (price) is the FWD model's single most important feature (gain
   226, top of the list) and is **positively** correlated with points
   (r=0.204 across all historical FWD rows) — the model does not penalise
   expensive players, so "price bias" was ruled out.
3. The actual cause: **`load_player_history()` (season_simulator.py:409)
   reads only the live 2026-27 season's `player_history.csv`.** Every
   rolling feature (`form_last3`, `form_last5`, `avg_points_per_game`,
   `minutes_reliability`, `goals_per_game`, `assists_per_game`,
   `clean_sheet_rate`) is computed exclusively from *this season's* played
   gameweeks. At GW3 that is 2 data points — for every player, rookie or
   4-season incumbent alike. Haaland's three prior Man City seasons exist
   only smeared anonymously inside the 7-season historical *training* rows
   (92 of 6,636 FWD rows have `value >= £13m`, mean 6.68 pts, **std 5.07** —
   a small, high-variance bucket mixing him with every other expensive
   forward who ever bust or got injured). There is no feature anywhere in
   `FEAT_COLS` / `DEFAULT_FEAT_COLS` that says "this specific player has
   been elite for 3 seasons" — only price and a 2-game window.
4. Consequence confirmed directly: holding Haaland's other features fixed
   and only lowering `value` from 15.5→9.0 dropped his own prediction from
   6.05→5.06 — *below* Isak's actual prediction (5.89, built from Isak's own,
   lower, form). A tree-model artifact from the sparse £13m+ region, not a
   deliberate signal.

This is a **structural gap**, not a bug to hotfix: the model has no
persistent notion of player identity/quality that survives across seasons.

---

## 1. Design goal in one paragraph

Give the live prediction pool (`build_rolling_pool`, `build_retrain_rows`,
and the historical training rows built by `feature_engineering_stage6.py`)
access to **multi-season, player-specific priors** that exist independently
of how many gameweeks have been played in the current live season — so an
established player's track record anchors his prediction from GW1 onward,
the same way `prev_league_*` already anchors debutants arriving from outside
the PL. Two secondary context signals (new-manager, transfer-sentiment) are
scoped alongside it since they were raised together, but are lower priority
and higher data-sourcing risk (§4-5).

---

## 2. Feature group A — career-level player quality (priority 1)

**What it fixes directly:** the Haaland case in §0. Recommended first per
this session's discussion — highest leverage, and the only one of the three
that reuses data the project already has (`data/raw/vaastav/`, 7 seasons).

### 2.1 Proposed features

| Feature | Definition | Leakage guard |
|---|---|---|
| `career_ppg_last_season` | Total PL points ÷ games played, prior completed PL season only | Prior season only — never the in-progress one |
| `career_ppg_last3_seasons` | Same, rolling 3-season window | Same |
| `career_minutes_reliability_last_season` | Minutes played ÷ available minutes, prior season | Same |
| `career_seasons_established` | Count of prior PL seasons with ≥900 minutes | Static per player per season-start |
| `career_price_tier_rank` | Player's `now_cost` percentile within his position at the time of prediction, cross-referenced against his own `career_ppg_last_season` percentile — i.e. is he expensive *because* he's proven, or expensive on reputation/hype alone | Uses only already-known price + prior-season output |

A player with zero prior PL seasons (rookie, promoted-club player who never
featured, new-to-PL signing) gets `career_seasons_established = 0` and the
PPG fields default to the position's replacement-level average — this is
exactly the same shape of fallback `prev_league_*` already uses for
non-PL debutants, so the model already has precedent for "no prior data"
handling.

### 2.2 Where it plugs in

- **Historical training rows** (`feature_engineering_stage6.py`): for each
  row at `(player, season, GW)`, compute the career fields from **seasons
  strictly before `season`** — this is a straightforward extension of the
  walk-forward pattern the file already implements for `prev_league_*`.
- **Live pool** (`season_simulator.py: build_rolling_pool`,
  `build_retrain_rows`, `build_gw1_pool`): compute from vaastav's completed
  2019-20…2025-26 seasons (already loaded as `train_dfs`) plus, once the
  live season has completed seasons of its own in future years, those too.
  No live-season data leaks in — GW1-2 of 2026-27 are current-season form,
  already covered by the existing rolling features; this is additive, not a
  replacement.
- **Both `FEAT_COLS` lists** (`season_simulator.py` and
  `prediction_matrix.py: DEFAULT_FEAT_COLS`) must be extended identically —
  these two lists are hand-duplicated today with no shared source of truth;
  worth fixing as part of this change (single constant imported by both,
  not two copies to keep in sync by hand).

### 2.3 Validation

- Stage 6's existing 10 leakage checks, re-run — new fields must pass
  the "knowable before kickoff" and "no cross-season bleed" checks
  ([[walkforward-no-leakage]]).
- Repeat the 6-fold walk-forward MAE sanity check from the 2026-27 refresh
  (CLAUDE.md) — expect MAE to hold or improve, not regress.
- **Targeted regression test**: re-run this session's Haaland diagnostic
  script (feature vector + price-perturbation trace) before/after — the
  new features should visibly separate Haaland's prediction from a
  same-price/same-current-form player with zero prior seasons.

---

## 3. Feature group B — new-manager flag (priority 2, blocked on data)

**Signal:** a managerial change often resets a team's tactics, rotation
patterns, and even individual players' roles (e.g. a new manager benching a
previously-nailed player, or unlocking a previously-fringe one). Two
features: `games_since_manager_change` (team-level) and a decay-weighted
"pre-change form is less predictive" multiplier, most naturally implemented
as a *sample-weight* adjustment on historical rows (down-weight a team's
rows from before a manager change when training on data spanning the
change) rather than a raw feature.

**Blocker:** the project has no managerial-change dataset. Neither the FPL
API nor vaastav's repo carries this. Would need a new source — options to
evaluate before scoping further: Transfermarkt (already integrated for
Stage 4a/4c signings scraping, so the scrape/match machinery exists and
could plausibly be extended), or a manually-curated table (small enough
domain — ~20 clubs × a handful of changes per season — to hand-maintain,
similar in spirit to `RULE_EVENTS_FT`'s one-off config approach).

**Recommendation:** defer until §2 is shipped and validated; revisit with a
short data-sourcing spike (confirm Transfermarkt coverage) before committing
to a build.

---

## 4. Feature group C — transfer-sentiment / ownership momentum (priority 3)

**Signal:** `intel_01_fpl_live.py` already collects most-bought/most-sold
and price-change data every run (`data/intel/fpl_live.json`) — currently
display-only, never reaching the predictor. Proposal: a
`net_transfer_momentum` feature (net transfers in/out as a fraction of total
owners, decayed over the last N gameweeks) as a wisdom-of-crowds signal.

**Risk, flagged honestly:** this is the weakest of the three signals.
Ownership momentum is reactive (mirrors recent form/fixtures the model
already sees via `form_last3`/`fdr`) more often than it's predictive, and it
is INTEL_SEASON-only data (2026-27), so it cannot be back-tested against 7
seasons of history the way §2 can — there's no historical
`net_transfer_momentum` series to train on, only to apply live, which risks
train/serve skew (a feature the model never saw in training suddenly
present at inference). Would need either (a) backfilling a historical proxy
from vaastav's `transfers_in`/`transfers_out` columns (already present in
`player_history.csv` per season_simulator.py:431-432, unused today), which
makes this tractable, or (b) treating it as a post-hoc scoring adjustment
outside the model entirely (closer to how `intel_03`/`intel_04` availability
and rotation penalties already work — multiplicative, not model-trained).

**Recommendation:** lowest priority; if pursued, prefer the intel-penalty
pattern (b) over a trained feature (a) — smaller blast radius, no
retraining or leakage risk, consistent with how the project already handles
other soft signals.

---

## 5. Sequencing

1. **§2 career-level player quality** — build, validate, ship. Directly
   closes the gap this session found.
2. **§3 new-manager flag** — data-sourcing spike first (confirm a usable
   source exists), then scope a follow-up design note.
3. **§4 transfer-sentiment** — optional, lowest confidence; implement as an
   intel-style multiplicative penalty (not a trained feature) if pursued at
   all.

## 6. Non-goals for this pass

- No changes to `milp_core.py` objective/constraints.
- No changes to chip logic.
- No player-identity *categorical* feature (raw `player_id` as a
  high-cardinality category) — rejected in favour of the numeric
  career-quality summary features in §2, which generalise to players with
  few historical rows instead of requiring the model to have seen that
  exact `player_id` before.

## 7. Open questions for sign-off

- ~~§2.1: are the five proposed fields the right set, or should
  `career_price_tier_rank` be cut for v1~~ **Resolved 2026-09-04: cut.**
  Shipped with the other 4.
- §3: worth a short spike to check Transfermarkt managerial-change coverage
  before deciding to build, or shelve indefinitely? **Still open.**
- §4: agree to defer, and if revisited, agree the intel-penalty pattern
  (b) over a trained feature? **Still open** (deferred, not yet revisited).
- ~~Retraining cost: §2 requires a full 7-season Stage 6 re-run + the 6-fold
  walk-forward re-validation — acceptable to schedule as its own session?~~
  **Resolved 2026-09-04:** done in the same session as implementation — all
  11 Stage 6 checks pass, 6-fold walk-forward MAE 2.403 (no regression).
