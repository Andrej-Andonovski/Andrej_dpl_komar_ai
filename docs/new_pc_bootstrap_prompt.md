# New-PC bootstrap prompt

Paste this into a fresh Claude Code session, opened in this repo's root, after
`pip install -r requirements.txt` and creating a `.env` with `ANTHROPIC_API_KEY`
and `GEMINI_API_KEY`.

---

This is the FPL AI project (see CLAUDE.md for full context). I need you to
refresh all data for the current live gameweek and give me the transfer/
captain recommendation for my real squad. Read CLAUDE.md first for project
context, then do the following in order:

**1. Full data refresh (real actuals for completed gameweeks):**
- Clear the stale per-player cache so real results actually get pulled:
  delete every file in `data/raw/fpl_api/player_summaries/*.json`
  (they're a resumable cache — if left in place, `data_fetcher_stage1.py`
  silently reuses last week's pre-match data instead of fetching real
  results).
- Run `python pipeline/data_fetcher_stage1.py` (takes several minutes —
  ~650 individual player API calls). This refreshes players/fixtures/teams
  and rebuilds `player_history.csv` with real results for every completed
  gameweek.

**2. Intel refresh, in order:**
- `python pipeline/intel_01_fpl_live.py` — live injuries/prices/ownership,
  always safe to just run.
- `python pipeline/intel_02_press_conferences.py` — press-conference
  scrape. **Check first**: it has a resume-cache
  (`data/intel/press_conferences.json`) keyed by GW number with a
  `GW_START`/`GW_END` range hardcoded near the bottom of the file. If the
  current gameweek isn't already marked `"status": "success"` in that
  JSON, temporarily set `GW_START = GW_END = <current_gw>` before running
  (avoids wasting ~12s/request searching for 30+ future gameweeks that
  don't exist yet), then set it back to `(1, 38)` afterward. It's normal
  for this to say "URL not found" if the article isn't published yet
  (usually posts a day or two before deadline).
- `python pipeline/intel_03_availability.py` and
  `python pipeline/intel_04_rotation_risk.py` — no caching, always safe
  to just run.
- `python pipeline/intel_05_recommendations.py` — LLM recommendations
  (costs a real API call per GW). Same resume-cache pattern as intel_02
  but in `data/intel/recommendations.json`; temporarily set
  `GW_START = 1, GW_END = <current_gw>` if needed, then restore to 38.

**3. Compute the actual transfer/captain decision for MY real squad**
(not a fresh rebuild — I already own a squad and update it week to week):

My current real squad is:
- XI: Raya (GK), Calafiori/Shaw/Hume (DEF), Tzolis/Rogers/B.Fernandes/Mbeumo
  (MID), Calvert-Lewin/Haaland/João Pedro (FWD)
- Bench: Dubravka (GK), van Ewijk/Diop (DEF), Hughes (MID — Will Hughes,
  Crystal Palace, NOT Charlie Hughes of Hull City, they share a surname)

(Tell Claude here if any transfers have actually happened since this was
written — this list may be stale by the time you run it.)

Write a script (don't try to hack the full `run_simulation()` loop — it
has too much season-long internal state like chip tracking and purchase-
price ledgers to safely override mid-season). Instead, directly call:
`load_player_history()` for real actuals, `build_retrain_rows()` +
`build_hist_rows()` + `train_models()` to retrain on real results,
`build_rolling_pool()` for the current GW's feature pool, then
`prediction_matrix.build_matrix()` + `milp_core.solve_horizon()` with
`owned=<my real 15 player IDs>` and the correct `free_transfers` count
(check what I actually have banked in-game, don't assume — the free-
transfer accrual was a real bug fixed this project, see CLAUDE.md /
git log, so trust the in-game number over any prior assumption).

Use env vars `RULES_MODE=corrected OPTIMIZER=mp MP_HORIZON=5
OWN_PRIOR_GW1=0.213` (OWN_PRIOR_GW1 only matters for GW1 builds, harmless
otherwise).

**4. Before recommending any transfer, sanity-check it against the real
fixture list** (`data/raw/fpl_api/fixtures_raw.csv`) — don't just trust
the optimizer's output number. If it recommends selling a player who just
had a good week AND has an easy upcoming fixture, flag that clearly as
worth scrutiny rather than presenting it as obviously correct — the model
has known blind spots (small sample size early season, no real rotation-
risk signal until several GWs of history exist) that are worth a gut-
check against what you can independently verify.

**5. Player names**: use `player_summaries`/`players_raw.csv`'s `web_name`
field for identification, not raw surname matching — there are real
surname collisions in the player pool (e.g. two different "Richards",
two different "Hughes") that will silently misattribute stats if you
match on surname alone.

Give me the final XI, bench, captain/vice, and any transfers with a plain-
English reason for each, plus your honest read on how confident you are
in each recommendation.
