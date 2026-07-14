# Press Conference Scraper Redesign — Progress

Last updated: 2026-07-07. Companion to `docs/press_scraper_redesign.md`
(the full design) and `data/intel/validation_2526/` (validation evidence).

---

## 1. Where we are

Build steps 1, 2, 3 and 4 of the six-step plan (design doc §9) are
**implemented and tested**; steps 1–3 are also **validated against archived
2025-26 season content**. The Newcastle class of miss — 7 clubs invisible to
the old scraper for an entire season — is fixed and proven fixed, and the FFS
liveblog (the richest presser source) is now discovered via the WordPress
REST API with no hardcoded URLs, no member cookie, and no day-name
assumptions. The GW18/19 Christmas-midweek coverage holes from the step-2
validation are plugged. Step 4 adds the Tier 3 Sky Sports sitemap adapter,
the Tier 4 Reach-local RSS adapter, and the T-24h coverage escalation — all
unit-tested (20/20) and live-smoke-verified against the real Sky/Reach
endpoints (in-season injury signal can't be validated off-season, but every
parse/attribution/escalation path runs live without error). Steps 5–6
(scheduler + GitHub Actions, re-scrape diff vs the old output) are not
started.

Nothing in production has been touched: the old `intel_02` scraper, its
`press_conferences.json`, and everything downstream (intel_03…06, season
simulator, the 2468-pt run) are exactly as they were.

## 2. What we did

### Investigation (design phase)
- **Audited the old scraper** and proved the real failure mode: club headers
  were exact-matched against short names, so "NEWCASTLE UNITED" ≠ "Newcastle"
  — 7 clubs (Newcastle, Spurs, West Ham, Wolves, Brighton, Leeds, Burnley)
  produced **zero claims across all 38 GWs**, and their players were silently
  misattributed to the preceding club's section. Reported `"success"` anyway.
- **Researched sources and discovery live** (design doc §4): FFS's WordPress
  REST API is public and un-paywalled; Guardian's Content API has a stable
  series tag for its weekly all-20-club team-news article; three structured
  injury tables (FFS injuries, SportsGambler, KnocksAndBans) have stable URLs
  — no URL discovery problem at all. Rejected with evidence: Premier Injuries
  (Cloudflare), PhysioRoom (degraded), RotoWire (paywall), WhoScored, BBC.

### Build (steps 1–2, all run via Docker image `fpl-scrape`)
- **`pipeline/intel_identity.py`** — shared identity module: club registry
  built from FPL data at runtime (never a hardcoded season list), a
  season-independent alias table, never-guess matching (ambiguous tokens like
  bare "UNITED" are rejected, unmatched headers become section boundaries),
  and club-constrained player resolution to FPL element ids.
  Tests: `tests/test_intel_identity.py`, **13/13**, including all 7 real
  failing headers.
- **`pipeline/intel_02_ledger.py`** — claim schema, append-only per-GW JSONL
  ledger with claim-id dedup, and the reconciler (suspension override,
  per-source last-write-wins, tier × recency weighted scoring, conflict flag
  with pessimistic midpoint cap, N-source agreement bonus, staleness rules —
  see §3 below). Tests: `tests/test_intel_02_ledger.py`, **23/23**.
- **`pipeline/intel_02_sources.py`** — four Tier 1 adapters (FFS injuries
  table, SportsGambler, KnocksAndBans, Guardian Content API) with a shared
  fetch/extract split so archived content flows through the identical
  extraction path used live.
- **`pipeline/intel_02_scrape.py`** — orchestrator: adapters → ledger →
  reconcile → `press_conferences_v2.json` (legacy-schema superset intel_03
  can read unchanged, plus additive `player_id`/`score`/`n_sources`/
  `conflict` fields), per-GW coverage block, per-source health file with
  structure-drift alarms.

### Validation (off-season, against archived 2025-26 content)
- **`scripts/validate_scraper_v2.py`**: Guardian validated via its own
  historical API (full-season archive, deadline-aligned); the three table
  sources via Wayback Machine snapshots near each deadline. All 38 GW
  deadline dates fetched from the FPL API before the 2026-27 rollover and
  cached (`deadlines_2526.json`).
- **Result: all 7 blind clubs recovered season-wide; 30/38 GWs at full
  20/20 club coverage.** The weak weeks are archive artifacts, documented in
  `validation_report.json` (GW18/19: Guardian's series skipped the Christmas
  midweek rounds and no usable snapshots exist; a few Guardian-only weeks
  where sub-20 partly means "healthy club, nothing to report").

### Hardening (two real bugs caught by review of the output)
1. **Stale-snapshot poisoning (the GW38 Xhaka case).** A February snapshot
   claim scored "out" at full strength for GW38 — relative recency decay
   cancels out when a stale claim is a player's only claim, and the claim's
   own `return_date` (Feb 28) was ignored. 49 of 173 GW38 players rested on
   that snapshot. Fixed: hard 10-day staleness window anchored to the GW
   deadline (all-stale player → dropped = healthy), return-date-passed →
   "unknown" (also defuses expired suspensions), harness skips snapshots
   >7 days from a deadline.
2. **Long-term-injury dropping (the GW38 Odobert case).** The staleness fix
   initially keyed on `published_at` (row-edit time), wrongly discarding
   long-term injury rows last edited in February but still asserted by a
   fresh May snapshot. Fixed by the semantic that now defines the schema:
   **`observed_at` = when the source asserted the state** (tables: fetch/
   snapshot time; articles: publication time), and staleness keys on it;
   `published_at` affects recency weighting only. Odobert now reconciles
   "out" with all 3 sources counted.
- **Ledger hygiene:** 2,581 stale-attached claims (26%) purged from the
  validation ledgers — pre-cap harness runs had attached e.g. Feb-1 snapshot
  claims to GW37/38. The reconciler was already filtering them; the purge
  makes the ledger files themselves a truthful audit trail. Known gotcha
  recorded: `claim_id` excludes timestamps, so fixing timestamp semantics
  requires stripping affected claims before re-appending (dedup keeps old
  versions otherwise).
- Conflict machinery verified on real data: 2 genuine GW16 conflicts
  (Bissouma, Diarra), pessimistic cap applied.

### Build (step 3 — FFS liveblog adapter, 2026-07-07)
- **`FfsTeamNewsAdapter`** (Tier 2, in `pipeline/intel_02_sources.py`):
  - **Discovery = one WP REST request** (`categories=3` + deadline window
    `after=D-6d&before=D`), matched against `^FPL Gameweek {N}\b team news`
    (the `\b` kills the "gameweek 2" → "gameweek 22" LIKE hazard);
    secondary title-only search (`search_columns=post_title`) re-filtered
    by the same regex. `content.rendered` carries the full body, so
    discovery and fetch collapse into a single JSON request. KNOWN_URLS,
    the member cookie, and the HTML search page are all gone.
  - **Extraction through the identity registry**: club headers match
    alias-based (all 7 blind-club header forms), unmatched headers
    ("FRIDAY'S PRESS CONFERENCE TIMES") are section boundaries — never the
    previous club's content (§1.2 fix applied to the liveblog itself).
    Bold-name player mentions, per-player parenthetical injuries
    ("Krafth (knee) and Livramento (hamstring)" each get their own), and
    the old scraper's proven regex classifier ported verbatim (stage A).
  - **Timestamp semantics**: `observed_at` = post `modified_gmt` (a
    liveblog's content is asserted as of its last edit, not our fetch);
    `published_at` = `date_gmt` drives recency + rule-2 ordering, so a
    Friday edition supersedes Thursday's per player.
  - Also accepts raw archived HTML pages (narrows to the article container)
    — the input path step 6's re-scrape diff will use.
- **`pipeline/intel_02_llm_extract.py`** (stage B, §5.3): Gemini Flash
  gap-filler for club sections where regex found < 2 players but ≥200 chars
  of prose. Roster-grounded strict-JSON prompt, intel_05's retry pattern,
  ≤40 calls/GW cost bound, claims tagged `extractor: "llm"`. Fully
  optional: no `GEMINI_API_KEY`/`google-genai` → regex-only, never fails.
- **Orchestrator wiring**: `ffs_teamnews` registered in `intel_02_scrape.py`;
  `ffs_window()` derives the discovery window from the live FPL deadline
  (or `--deadline`); `--llm` flag enables stage B.
- Tests: `tests/test_intel_02_ffs_liveblog.py`, **15/15** (discovery regex,
  blind-club extraction, boundary regression, per-player injuries, LLM gap
  detection with a stub, multi-edition supersession through the reconciler,
  archived-page narrowing). Identity 13/13 + ledger 23/23 still green.

### Validation (step 3 — against the WP post archive, no Wayback needed)
- The WP REST API serves FFS's full post history, so historical discovery
  is the **exact live code path**. Run via `validate_scraper_v2.py`
  (now includes an `ffs_teamnews` section per GW):
  - **GW18** (Christmas): 3 editions found — Tue, **Xmas Eve** ("+ Bruno
    latest", the ad-hoc slug class that broke KNOWN_URLS), Friday —
    120 claims, **20/20 clubs from the liveblog alone** (Guardian had no
    article that week and archive.org was 429-rate-limited for tables).
  - **GW19**: 2 editions (Mon/Tue), 112 claims, 15/20 — the 5 missing
    clubs simply held no presser that midweek round.
  - **GW22** (spot-check): 2 editions, 146 claims, 20/20; full ledger now
    reconciles 520 claims → 211 players.
  - Blind-club check: all 7 recovered on every validated GW.
- Known wrinkle (validation only): editions edited after the deadline carry
  a post-deadline `modified_gmt` → `observed_at` slightly ahead of
  `ref_time`; staleness tolerates it and live ticks always run pre-deadline.

### Build (step 4 — Tier 3/4 adapters + T-24h escalation, 2026-07-07)
- **Shared story extractor** (`_extract_stories` in
  `pipeline/intel_02_sources.py`): Tier 3/4 sources are story-per-article
  (one article ≈ one club/player event), not section-per-club like Tier 1/2.
  Club attribution is headline-first via the identity registry
  (`_attribute_club`) — a title naming two clubs ("Arsenal transfers: Aston
  Villa demand fee") is ambiguous and **skipped, never guessed** (§3.1 applied
  to a new content shape). Then a **roster-grounded regex scan**
  (`_roster_scan`, stage A): each roster player named in an availability
  sentence gets a classified claim, club-constrained so surname matches are
  safe; accent-tolerant (`_fold`) so "Schar"/"Odegaard" match FPL
  "Schär"/"Ødegaard". Stage-B LLM gap-fill (same `GeminiExtractor`) fires for
  a club left with <1 regex claim but ≥300 chars of prose.
- **`SkySportsAdapter`** (Tier 3, §4.5): per-tick discovery = the Google news
  sitemap (`sitemap-news.xml`, ~50 URLs all <48h old), filtered to
  `/football/` + availability keywords; escalation = a club's `{slug}-news`
  index page scanned for that club's fresh team-news items. `SKY_SLUGS` maps
  registry slugs → Sky URL slugs (verified the Newcastle page resolves with
  15 news links live).
- **`ReachLocalAdapter`** (Tier 4, escalation-only, §4.7): per-club
  `?service=rss` feed → article `application/ld+json` `articleBody` → story
  extractor. `REACH_OUTLETS` maps 17 clubs to their Reach local paper
  (ChronicleLive/Newcastle is the live-verified anchor; football.london,
  Liverpool Echo, MEN, BirminghamLive, etc. live-probed 2026-07-07).
  Brighton/Bournemouth/Southampton intentionally unmapped — no confident
  Reach outlet, so escalation reports them uncovered rather than guessing a
  dead feed. LLM extraction is the intended precision path for this noisy
  tier; degrades to regex-only without a key.
- **Orchestrator wiring** (`pipeline/intel_02_scrape.py`): `sky` added to the
  default tick sources; `reach` is escalation-only (never bulk-fetched). New
  `--escalate` flag runs the T-24h escalation (§7) — after reconciliation,
  `missing_clubs()` finds clubs with zero Tier 1-3 claims and runs Sky
  per-club + Reach per-club for exactly those, then re-reconciles. (The
  *window gate* that decides when to set `--escalate` is step 5's scheduler;
  step 4 is the mechanism.)
- Tests: `tests/test_intel_02_tier34.py`, **20/20** (sitemap filter+date
  parse, RSS parse, JSON-LD `@graph` unwrap, headline attribution incl.
  ambiguous-skip and no-club-skip, roster scan + accent tolerance, dotted
  web_name handling, stage-B gap detection with a stub, unmapped-club no-op,
  `missing_clubs` selection, cross-tier Sky+Reach reconciliation). Identity
  13/13 + ledger 23/23 + FFS liveblog 15/15 still green.
- **Live smoke** (2026-07-07, off-season): Sky sitemap discovery→fetch→extract
  and Reach Newcastle RSS→article→extract both run end-to-end with no errors;
  Sky correctly rejects World Cup/international items as unattributed
  (never-guess on real noise), Reach correctly attributes a Newcastle article
  to NEW. Zero claims is expected — July news is transfers, not injuries to
  current FPL squad players; the in-season signal is proven by the fixtures.

## 3. Current state

| Piece | State |
|---|---|
| Identity module + tests | done, 13/13 |
| Ledger + reconciler + tests | done, 23/23 (incl. Xhaka + Odobert regression tests) |
| Tier 1 adapters (FFS injuries, SportsGambler, KnocksAndBans, Guardian) | done, smoke-tested live + validated on archives |
| Tier 2 FFS liveblog adapter (WP REST discovery, regex→LLM extraction) + tests | done, 15/15; validated on GW18/19/22 via the WP post archive — GW18 hole plugged at 20/20 |
| Tier 3 Sky sitemap adapter + Tier 4 Reach RSS adapter + shared story extractor + tests | done, 20/20; live-smoke-verified (parse/attribution/escalation paths run; off-season yields no injury signal) |
| T-24h escalation (`missing_clubs` → Sky per-club + Reach per-club, `--escalate`) | done; escalation loop live-smoke-verified end-to-end (fetch_club→extract→ledger→reconcile) |
| Stage-B LLM extractor (Gemini, optional) | done (code + gating tested with a stub for both liveblog and story sources; not yet exercised against the live Gemini API) |
| Orchestrator + coverage accounting + health alarms | done (liveblog + Sky wired in, `--deadline`/`--llm`/`--escalate` flags) |
| Full-season archived validation | done — 7 blind clubs recovered, 30/38 GWs at 20/20 |
| Ledger audit trail | purged clean (2,581 stale-attached claims removed) and fully rebuilt — same headline numbers reproduced from clean ledgers (30/38 at 20/20; Odobert 3-source "out"; stale-snapshot players absent). archive.org rate-limited later lookups (429s) but the accumulated ledger + Guardian carried the rebuild; individual GWs can be topped up later with `--gws N` |
| Production pipeline | untouched |

Known limitations of the validation (not of the pipeline):
- Identity reference is the GW29 `fpl_live.json` snapshot → players who
  changed clubs mid-season resolve as `player_id: null` (kept, flagged).
  Live operation fetches bootstrap-static fresh per run.
- Wayback coverage of table sources is patchy and its availability API is
  flaky/rate-limited; Guardian is the only deadline-aligned historical source.
- Legacy intel_03's own club list lacks Burnley — its name-based join would
  drop Burnley claims until it consumes the `player_id` fields the v2 output
  already carries. intel_03 has NOT been executed against v2 output (it
  overwrites production `availability.json`; belongs to step 6 in a copied
  environment).

## 4. Next steps (design doc §9)

1. ✅ **Step 4 — Tier 3/4 (DONE 2026-07-07)**: Sky Sports news-sitemap adapter
   (Tier 3, per-tick) + Reach-local `?service=rss` adapter (Tier 4) + shared
   story extractor + T-24h `--escalate` for uncovered clubs. 20/20 tests,
   live-smoke-verified. Season-start check needed: Reach outlet slugs and
   `SKY_SLUGS` are off-season-probed — the §7 parse alarms + step-5 checklist
   re-verify them on GW1 of 2026-27 (LOWER-CONFIDENCE per design §4.7).
2. **Step 5 — scheduler**: `--tick` gate (deadline-aware cadence windows) +
   GitHub Actions cron; secrets (Guardian API key, Gemini) to Actions
   secrets; one dress-rehearsal GW against a live 2026-27 deadline.
   **Season-start re-verification checklist** is in the design doc §9.5.
   Include one `--llm` run against the live Gemini API (stage B is
   stub-tested only so far).
3. **Step 6 — re-scrape diff vs old output**: run v2 against the same
   2025-26 GWs, diff player sets vs the old `press_conferences.json`,
   quantify what the 2468 run never saw (thesis material); dry-run intel_03
   on v2 output in a copied environment. The liveblog adapter's raw-HTML
   input path (archived article pages) exists for exactly this.
4. **Unrelated but blocking other work**: copy `data/raw/` from the original
   machine (chip-v2 backtest and Phase 0 gates still pending on it).
