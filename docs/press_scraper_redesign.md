# Press Conference Scraper Redesign — Autonomous Multi-Source Intel (intel_02 v2)

Status: DESIGN ONLY (2026-07-06). No implementation yet.
Replaces: `pipeline/intel_02_press_conferences.py` (single-source FFS scraper).
Downstream consumers: intel_03 (availability merge), intel_04 (rotation press
keywords), intel_05 (LLM recommendations), intel_06/season_simulator (penalties).

---

## 0. Design goal in one paragraph

Turn intel_02 from a manually-fed, single-source, run-once scraper into an
autonomous multi-source aggregation service that (a) discovers press-conference
and team-news content for **all 20 PL clubs** before every gameweek deadline
without any hardcoded URLs, (b) runs on a deadline-aware schedule with zero
manual intervention across a 38-GW season, (c) reconciles conflicting claims
from multiple sources into one per-player availability record with full
provenance, and (d) fails loudly (per-club coverage accounting, parse alarms)
instead of silently — the current design reported "success" for 28 gameweeks
while seven clubs had zero coverage.

---

## 1. Autopsy of the current scraper — why Newcastle was missed

The Newcastle miss is not one bug; it is five compounding structural defects.
All verified against the code and `data/intel/press_conferences.json`.

### 1.1 Club-header matching is exact-match against a hardcoded short-name list

`normalize_club()` (intel_02_press_conferences.py:177) requires the article's
section header to equal a `PL_CLUBS` entry exactly (case-insensitive, with only
a trailing-"s" tolerance). FFS writes full club names in headers. Result,
measured over all 28 scraped GWs — clubs extracted vs. never extracted:

| Extracted (14) | Never extracted once (7) | Why the header failed |
|---|---|---|
| Aston Villa (297 mentions), Chelsea (291), Liverpool (288), Arsenal (248), Bournemouth (225), Man Utd (225), Man City (213), Crystal Palace (210), Everton (187), Brentford (181), Fulham (160), Nott'm Forest (150), Sunderland (138), Southampton (8) | **Newcastle** | header "NEWCASTLE UNITED" ≠ "Newcastle" |
| | **Tottenham** | "TOTTENHAM HOTSPUR" ≠ "Tottenham" |
| | **West Ham** | "WEST HAM UNITED" ≠ "West Ham" |
| | **Wolves** | "WOLVERHAMPTON WANDERERS" ≠ "Wolves" |
| | **Brighton** | "BRIGHTON AND HOVE ALBION" ≠ "Brighton" |
| | **Leeds** | "LEEDS UNITED" ≠ "Leeds" |
| | **Burnley** | not in `PL_CLUBS` at all |

So this was never "Newcastle's page wasn't picked up" — the page was fetched
and parsed every week. The Newcastle **section** was invisible, for 7 of 20
clubs, every single gameweek. 35% structural blindness reported as `"status":
"success"`.

### 1.2 Unrecognized headers cause misattribution, not just omission

In `parse_article()`, an element that is not a recognized club header is
appended to the *current* club's element list. An unrecognized "NEWCASTLE
UNITED" h2 is therefore treated as body text of the preceding club's section
— Bruno Guimarães's injury sentence gets attributed to whichever club came
before Newcastle in the article. intel_03 then drops it because player+club
don't match. This is why the misses were silent: player counts looked healthy.

### 1.3 URL resolution is hardcoded, and the fallback contradicts the data

23 URLs are hardcoded in `KNOWN_URLS`. The slugs are demonstrably not
constructible: day names vary (`fridays`, `tuesdays`, `weds`, `thursday`),
WordPress dedup suffixes vary (`-2`, `-3`), and ad-hoc keywords appear
(`ekitike-latest`, `bruno-latest`). The `discover_url()` fallback searches for
`gameweek-{gw}...team-news...friday` — but 5 of the 23 known URLs don't contain
"friday" (midweek GWs), so the fallback structurally cannot find exactly the
articles most likely to be missing from `KNOWN_URLS`. Discovery also depends on
FFS's HTML search page, an unstable interface.

### 1.4 Auth and season facts are frozen in the source code

A member session cookie is hardcoded (expires ~2026-03-27); `PL_CLUBS` still
contains Ipswich/Leicester/Southampton (relegated 2024-25) and is missing
Burnley (promoted 2025-26). Every season rollover requires code edits — the
opposite of autonomous.

### 1.5 Run-once semantics can't track a live news window

Press conferences trickle in Thursday–Friday (and match-day-1 for midweek GWs).
The FFS article is a **liveblog updated continuously**. The current scraper
marks a GW `success` on first parse and never revisits it (`already_scraped`
skip), so a Friday-morning run permanently misses every presser held after it.

Design consequences: (1) club identification must be alias-based and derived
from the FPL API each season, never a hardcoded list; (2) parse results must be
validated against expected coverage (20 clubs), never trusted because HTTP said
200; (3) URLs must be discovered from machine-readable indexes, not
constructed; (4) a GW is an open collection window, not a one-shot scrape;
(5) no single source may be a single point of failure.

---

## 2. Architecture — tiered multi-source aggregation

```
                 ┌────────────────────────────────────────────┐
                 │ scheduler (deadline-aware, §6)             │
                 └──────────────────┬─────────────────────────┘
                                    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ intel_02 v2 — per-source adapters                           │
  │                                                             │
  │ Tier 0  FPL API player flags (news/status/chance) — already │
  │         intel_01; baseline + arbiter for suspensions        │
  │ Tier 1  Structured injury aggregators (stable URL, tables,  │
  │         all 20 clubs) — §4                                  │
  │ Tier 2  FFS team-news liveblog (rich presser text) — §4     │
  │ Tier 3  Media roundups (BBC/Sky) — §4                       │
  │ Tier 4  Official club sites — fallback for clubs still      │
  │         uncovered at T-24h — §4                             │
  └───────────────┬─────────────────────────────────────────────┘
                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ normalization (§3): canonical club ids + FPL player ids     │
  │ claim store  (§5.1): append-only per-GW claim ledger        │
  │ reconciliation (§5.2): claims → one record per player/GW    │
  │ coverage accounting (§7): 20-club checkboard, alarms        │
  └───────────────┬─────────────────────────────────────────────┘
                  ▼
        press_conferences.json (schema-compatible superset, §5.4)
                  ▼
        intel_03 (unchanged interface, optional upgrades)
```

Principles:

- **Adapters are isolated.** Each source is one module with a common interface
  `discover() -> [candidate_urls]`, `fetch(url) -> raw`, `extract(raw) ->
  [claims]`. A source failing (paywall, redesign, Cloudflare) degrades coverage
  by one tier; it never aborts the run.
- **Claims, not conclusions.** Adapters emit raw *claims* ("source S said at
  time T that player P is doubtful with a hamstring issue"). Only the
  reconciler produces per-player conclusions. All claims are kept for audit.
- **Coverage is the success metric.** A run is successful when ≥18/20 clubs
  have at least one Tier 1–3 claim or an explicit "no fresh news" verdict, not
  when HTTP requests returned 200.

---

## 3. Normalization layer (season-proof identity)

### 3.1 Club registry — built from the FPL API, never hardcoded

At the start of every run, build the club registry from `bootstrap-static`
`teams[]` (id, name, short_name). Attach a static **alias table** keyed by
canonical FPL id — the only hand-maintained artifact, and it is
season-independent because it covers all ~30 clubs that cycle through the PL:

```
newcastle:  ["Newcastle", "Newcastle United", "NEWCASTLE UNITED", "NUFC", "Magpies"]
spurs:      ["Tottenham", "Tottenham Hotspur", "Spurs", "THFC"]
wolves:     ["Wolves", "Wolverhampton", "Wolverhampton Wanderers"]
brighton:   ["Brighton", "Brighton & Hove Albion", "Brighton and Hove Albion", "BHAFC"]
...
```

Matching is: exact alias → normalized alias (strip FC/AFC, punctuation,
casefold) → token-subset match ("NEWCASTLE UNITED" ⊇ "NEWCASTLE"). A header
that matches **no** club is still treated as a section boundary (fixes §1.2
misattribution) and logged as `unmatched_header` — structure drift becomes
visible instead of silently corrupting the previous club.

### 3.2 Player resolution — FPL element id is the join key

Claims carry the source's player string; the normalizer resolves it to an FPL
`element` id using the reference in `fpl_live.json`: accent-folded exact match
on web_name / full name → last-name match constrained to the claimed club →
unresolved (kept, flagged). Club-constrained last-name matching is safe
precisely because claims are already partitioned by club. Reuse intel_03's
`normalize_name` + alias table; move both into a shared module so intel_02 and
intel_03 use one implementation.

---

## 4. Source portfolio and per-source discovery

> Populated from live investigation of each source (2026-07-06). Verdicts and
> mechanisms per source below; summary table in §4.9.

### 4.1 Tier 2 — FFS team-news liveblog (verified live 2026-07-06)

**Verdict: keep as the richest press-conference source; replace all URL
handling with the WordPress REST API.** Investigation results:

- The WP REST API is fully public: `GET /wp-json/wp/v2/posts` returns standard
  post JSON — no auth, no API key, no Cloudflare (origin Apache responds
  directly to plain requests). The hardcoded member cookie is unnecessary:
  the liveblog articles themselves are **not paywalled** (full manager quotes
  visible logged-out; FFS members-only covers tools, not news). §1.4's cookie
  can be deleted outright.
- `content.rendered` in the API response contains the **full article body** —
  discovery and fetch collapse into a single JSON request; no HTML page fetch
  needed at all.
- The "Team News" category is id **3** (slug `team-news`, ~2,840 posts).
- **Primary discovery — deadline-window query (no search ambiguity):**
  ```
  /wp-json/wp/v2/posts?categories=3&after={deadline-6d}&before={deadline}
      &per_page=50&_fields=title,slug,link,date,content
  ```
  then match `title.rendered` against `^FPL Gameweek {N} team news`. Verified
  against GW22: returns both the Thursday and Friday editions with their
  `-2`/`-3` suffixed slugs — the suffix chaos (§1.3) becomes irrelevant
  because we never construct slugs. Multiple weekday editions per GW are all
  taken (each is a separate fetch/claim batch; recency handles supersession).
- **Secondary discovery:** title-only search
  `?search=gameweek {N} team news&search_columns=post_title&categories=3` —
  works, but plain `search=` without `search_columns` matches body text and is
  unreliable; and `LIKE` matching means "gameweek 2" hits "gameweek 22", so
  always re-filter with the title regex + an `after=` season bound.
- **Fallbacks, in order:** on-site `?s=` search (HTML), Yoast
  `sitemap_index.xml` → highest `post-sitemap{n}.xml` slug match, `/feed`
  (RSS exists but only 15 site-wide items — too shallow as primary; category
  feeds are disabled).
- Format continuity: liveblog editions published every GW through GW38
  (2026-05-22), including Mon/Tue/Wed editions for midweek and double GWs —
  confirming that day-name assumptions (§1.3) must never appear in matching.
- Client must follow redirects (trailing-slash URLs 301 to slashless).

### 4.2 Tier 1 — FFS injuries & bans table (bonus finding)

`https://www.fantasyfootballscout.co.uk/fantasy-football-injuries/` — public,
stable URL (zero discovery problem), server-rendered plain HTML `<table>`
(verified: no SPA framework, no paywall on this page). Confirmed columns:
**Player | Club | Status | Return Date | Latest News | Last Updated**, with
status values `Injured`, `Doubt N%` (aligns 1:1 with FPL flag semantics),
`Suspended`, and a dated free-text news column — the only free source found
with narrative + per-entry timestamp. **Verdict: adopt as a Tier 1 adapter**
— deterministic parse, all-20-club coverage by construction, and it
complements the liveblog (table = current state; liveblog = presser-fresh
deltas).

### 4.3 Tier 1 — The Guardian Content API (verified live 2026-07-06)

**Verdict: adopt as the anchor Tier 1 source.** An official, documented,
free-keyed JSON API (open-platform.theguardian.com; free tier ~500 calls/day —
we need ~5/GW).

- The weekly **"Premier League team news: predicted lineups"** article
  published essentially every Friday of the 2025-26 season (verified 15 hits
  Jan–May 2026; gaps only on blank/cup weekends) carries the stable series tag
  **`football/series/match-previews`**.
- Discovery is one query — no URL logic at all:
  ```
  https://content.guardianapis.com/search?tag=football/series/match-previews
      &order-by=newest&show-fields=body&api-key={key}
  ```
- Content is near-structured availability data for **all 20 clubs**: per club
  "Doubtful: Wieffer (ankle); Injured: Mitoma (hamstring, Jun), …; Suspended:
  …" — parseable with a small deterministic grammar, LLM fallback rarely
  needed. Publishes Friday ~15:00–17:00 UTC: after Friday pressers, before the
  final pre-deadline ticks.
- No manager quotes (Tier 2/3 cover those). URL section segment flips between
  `/football/` and `/sport/` — irrelevant since the API returns the link.

### 4.4 Tier 1 — premierleague.com Pulselive content API (verified live 2026-07-06)

**Verdict: adopt, wrapped in drift detection.** The PL site's backing content
API `footballapi.pulselive.com` is publicly accessible (no auth, no Origin
header needed):

- List endpoint with tag filter returns the weekly **"Predicted line-ups for
  Matchweek N"** series (one per Friday ~19:00Z, all clubs with a fixture,
  including manager presser quotes like Howe confirming Livramento's injury):
  ```
  https://footballapi.pulselive.com/content/PremierLeague/text/EN/
      ?pageSize=8&page=0&tagNames=franchise:predicted-line-ups
  ```
- Single-item endpoint returns the **full article body as JSON** with
  structured `references` (club/match entities) — club attribution is
  machine-readable, no header parsing at all.
- The site's evergreen **`/en/latest-player-injuries`** page (stable URL, all
  20 clubs, "Last updated" timestamp) is JS-rendered — player entries don't
  appear in a plain fetch. Do NOT scrape the page; get the same content
  through the API (harvest its content id once, then use the single-item
  endpoint).
- Caveats: undocumented API (could change or lock down without notice — this
  is exactly what per-source health tracking in §7 exists for); pagination
  metadata unreliable without a tag filter; tag names must be harvested from
  known items, not guessed.

### 4.5 Tier 3 — Sky Sports (verified live 2026-07-06)

**Verdict: adopt as the quote-rich breaking-news layer.** No paywall, no bot
protection observed.

- **Discovery: Google news sitemap** `skysports.com/sitemap/sitemap-news.xml`
  — ~50 URLs, all <48h old, each with `news:publication_date` + title. Filter
  titles/keywords for press conference & team news items each tick.
- Fallback: stable per-club index pages `skysports.com/{club-slug}-news`
  (verified, full headline lists) — also the T-24h targeted-escalation path
  for a specific missing club. RSS feeds exist (`/rss/12040`, `/rss/11095`)
  but are 20-item mixed-sport — polling supplement only.
- In-season Sky also runs a rolling "Premier League team news, injury latest"
  live blog (dormant off-season). Presser write-ups land 1–3h after Thu/Fri
  pressers. Extraction is consistent-template HTML → the two-stage extractor
  (§5.3) applies.

### 4.6 Tier 3 — BBC Sport (evaluated, deferred)

Discovery exists (`feeds.bbci.co.uk/sport/football/rss.xml`, 80 fresh items;
per-team feeds at `/sport/football/teams/{slug}/rss.xml`) but there is **no
systematic all-club weekly team-news product** — presser coverage is
selective, per-team feeds were sparse and partly audio links. **Deferred**:
not worth an adapter while Guardian + PL + Sky + FFS overlap already gives
4-source coverage of every club. Re-evaluate in-season if coverage accounting
shows persistent gaps. (Operational note: bbc.co.uk serves plain HTTP clients
fine but is blocked to some agent-side fetch tools — direct `requests` is
unaffected.)

### 4.7 Tier 4 — Official club sites & local outlets (fallback tier, thin verification)

> The dedicated club-sites investigation was cut short; this tier's design is
> from the media research plus general platform knowledge and is flagged
> LOWER-CONFIDENCE — verify during implementation step 4.

- Role: **targeted escalation only** (§7) — invoked at T-24h for specific
  clubs with zero Tier 1–3 claims, never a bulk-scraped tier. Maintaining 20
  bespoke club scrapers is exactly the maintenance burden this redesign
  removes; official sites also bury availability news in fluff.
- One verified pattern that generalizes: **Reach plc local outlets** expose
  per-club RSS via `?service=rss` on topic pages (verified:
  `chroniclelive.co.uk/all-about/newcastle-united-fc?service=rss`, 25 fresh
  items, articles carry `application/ld+json` with `articleBody` for clean
  extraction). One Reach adapter covers most clubs via their local paper
  (football.london, Liverpool Echo, MEN, BirminghamLive, ChronicleLive…).
  High noise/clickbait ratio → LLM extraction (§5.3) mandatory for this tier,
  and claims enter at a modest weight (§5.2).
- Official club sites proper (nufc.co.uk etc.): candidate mechanism is their
  news sitemaps / Pulselive-family content APIs where present; investigate
  only if the Reach pattern proves insufficient for escalation.

### 4.8 Tier 1 — additional structured injury aggregators (verified live 2026-07-06)

**SportsGambler — adopt (top structured pick).**
`sportsgambler.com/injuries/football/england-premier-league/` — single stable
URL, all 20 clubs (already showing the 2026-27 promoted clubs), page-level
"Last updated" timestamp showing same-day updates even in the off-season,
plain HTTP 200 (no Cloudflare), 282KB server-rendered HTML. Extraction is
near-deterministic thanks to semantic classes: per-club
`h3.injuries-title#[club-slug]`, then `div.inj-row` blocks with
`span.inj-player`, `span.inj-position`, `span.inj-info` (injury type),
`span.inj-return` (ISO-style expected return date), and **severity encoded in
the class attribute** — `injury-questionmark` (doubtful) vs `injury-plus`
(out). Supplies injury type + return dates that the FPL API lacks.

**KnocksAndBans — adopt as corroboration feed.** `knocksandbans.com` — all 20
clubs as cards *including explicit "No injuries and suspensions" empty states*
(valuable for coverage accounting: absence becomes a positive signal), status
in FPL-native buckets (OUT / 75% / 50% / 25%), estimated return date, and a
**per-entry last-update date** (feeds recency_decay directly). Server-rendered,
no bot protection. Caveat: Tailwind utility-class markup — key extraction on
club headings and DOM position, not class names; slightly more brittle.

**Transfermarkt — optional fourth.** Clean HTML injury table across all 20
clubs with authoritative long-term return dates (e.g. cruciate-ligament
month-precision). Needs a browser UA and polite request rates. Adopt only if
return-date reconciliation proves valuable to intel_03.

**Rejected after live testing:** Premier Injuries (Cloudflare interactive
challenge — 403 `cf-mitigated`, would need headless browser), PhysioRoom
(degraded to an aggregate page: no status, no return dates, no timestamps,
only clubs with current injuries), RotoWire (JS-injected AND paywalled),
WhoScored (no stable injuries page anymore), Fantasy Football Hub (no free
structured feed), fpl.page/injuries (dead feature).

**Tier 0 note.** FPL API fields confirmed intact (`news`, `news_added`,
`status`, `chance_of_playing_next_round`). Flags are set manually by the FPL
game team, usually only after official confirmation — hours behind pressers,
sometimes never for soft doubts. Since `news_added` is a timestamp, the
pipeline should **log the delta between each Tier 1–3 claim and the matching
FPL flag change** — an empirical measurement of exactly the latency gap
intel_02 exists to close (thesis-grade evidence, free).

### 4.9 Source portfolio summary

| # | Source | Tier | Discovery | Structure | All 20 clubs | Freshness | Risk notes |
|---|--------|------|-----------|-----------|--------------|-----------|------------|
| 1 | FPL API flags | 0 | fixed API | JSON | yes | laggy (manual flags) | none; already intel_01 |
| 2 | Guardian team-news article | 1 | Content API, series tag `football/series/match-previews` | JSON body, semi-structured per-club lists | yes (weekly) | Fri ~15–17:00 UTC | official API; needs free key |
| 3 | SportsGambler injuries | 1 | stable URL | semantic HTML classes | yes (daily) | same-day | none observed |
| 4 | FFS injuries table | 1 | stable URL | server-rendered table | yes (daily) | daily | none observed |
| 5 | KnocksAndBans | 1 | stable URL | HTML cards + empty states | yes (daily) | per-entry stamps | brittle markup |
| 6 | premierleague.com Pulselive | 1–2 | content API, tag `franchise:predicted-line-ups` | JSON body + club refs | yes (weekly) | Fri ~19:00 UTC | undocumented API |
| 7 | FFS team-news liveblog | 2 | WP REST API, category 3 + deadline window | HTML in JSON, presser quotes | ~all, multi-edition | live Thu–Fri | header parsing (fixed by §3) |
| 8 | Sky Sports | 3 | news sitemap + per-club index | consistent HTML | yes (rolling) | 1–3h post-presser | scraping, no API |
| 9 | Reach locals (ChronicleLive etc.) | 4 | `?service=rss` per club topic page | JSON-LD `articleBody` | per-club | fastest (~1h) | noisy; escalation only |
| — | BBC Sport | — | deferred | — | no all-club product | — | revisit if gaps persist |

Every club is covered by at least four independent sources (rows 3–5 by
construction, row 2 weekly, rows 7–8 when pressers happen), so a single-source
miss can no longer blind the pipeline to a club — the failure mode that
produced the Newcastle gap requires four simultaneous independent failures,
and §7 alarms fire long before that.

---

## 5. Claim model and reconciliation

### 5.1 Claim schema (append-only ledger, `data/intel/press_claims/gw{N}.jsonl`)

```json
{
  "claim_id":     "sha1(source,url,player_id,status,quote)",
  "gw":           23,
  "source":       "ffs_teamnews",
  "tier":         2,
  "url":          "https://...",
  "observed_at":  "2026-01-22T14:03:11Z",   // when the source ASSERTED this state:
                                            //   tables -> fetch/snapshot time
                                            //   articles -> publication time
  "published_at": "2026-01-22T13:40:00Z",   // row-edit/article timestamp if available
                                            // (recency weighting only — staleness
                                            //  keys on observed_at: a long-term
                                            //  injury row edited months ago is
                                            //  still asserted by a fresh table)
  "club_id":      14,                        // canonical FPL team id
  "player_id":    432,                       // FPL element id (null if unresolved)
  "player_raw":   "Bruno Guimarães",
  "status_claim": "doubtful",                // out|doubtful|available|suspended|unknown
  "injury":       "hamstring",
  "text":         "Bruno Guimarães (hamstring) faces a late fitness test...",
  "extractor":    "structured|regex|llm"
}
```

Append-only with `claim_id` dedup: re-scraping a liveblog re-emits unchanged
claims (deduped) and appends genuinely new/changed ones with fresh timestamps.
The ledger is the audit trail the thesis can cite.

### 5.2 Reconciliation — one record per (player, GW)

Inputs: all claims for (player_id, gw) + the Tier-0 FPL API flag. Rules, in
order:

1. **Suspension override.** Any `suspended` claim from Tier 0–2, or an FPL
   status of `s`, wins outright. Suspensions are administrative facts, not
   opinions.
2. **Freshness within a source: last write wins.** A source's newer claim for
   the same player supersedes its older one (liveblogs update: "doubtful" at
   13:00 → "ruled out" at 15:30). Only each source's latest claim enters
   scoring.
3. **Cross-source scoring.** Map each surviving claim to the existing intel_03
   scale (`available` 95, `doubtful` 40, `out` 5, `suspended` 0, `unknown` 50)
   and combine as a weighted mean:

   ```
   weight(claim) = W_tier[tier] * recency_decay(observed_at)
   W_tier   = {0: 0.8, 1: 1.0, 2: 1.2, 3: 0.9, 4: 1.1}
   recency_decay = 0.5 ** (hours_since / 48)
   press_score   = Σ w·score / Σ w
   ```

   Tier 2 (direct presser quotes) weighs highest among scraped sources; Tier 4
   (official club statement) nearly as high; Tier 0 lowest because FPL flags
   lag press conferences by hours (that lag is the entire reason intel_02
   exists).
4. **Conflict flagging with a pessimistic floor.** If the latest claims from
   two sources ≤12h apart disagree by more than one severity step (e.g.
   `available` vs `out`), set `conflict: true`, keep both in the record, and
   cap the reconciled score at the **midpoint between the two claims' scores**
   — never above it. Rationale: for a captaincy/transfer engine, wrongly
   trusting "fit" costs far more than wrongly benching a fit player; the
   asymmetry is the same one behind the existing availability multiplier.
   `doubtful` vs `out` (adjacent severities) is NOT a conflict — it is the
   normal presser-to-presser progression; recency weighting handles it.
5. **Agreement bonus preserved.** Independent sources agreeing within one
   severity step add +5 (bounded at 100) — the exact analogue of intel_03's
   existing both-sources-agree bonus, extended to N sources.

The reconciled record keeps `sources: [claim_id, ...]`, `conflict`,
`n_sources`, and `latest_observed_at` so intel_03/05 can reason about
confidence, not just the score.

### 5.3 Extraction strategy per content type

- **Structured sources (Tier 1 tables):** deterministic parsers. Tables have
  named columns; regex classification is unnecessary.
- **Free-text press articles (Tiers 2–4):** two-stage. Stage A: the existing
  regex `classify()` as a zero-cost first pass. Stage B: an LLM extraction call
  (same Gemini Flash usage pattern as intel_05) with a strict JSON schema,
  grounded with the club's actual FPL player list in the prompt, for (a) any
  article where regex finds < 2 players for a club that played a presser, and
  (b) a verification sample. LLM output is claims like any other — it goes
  through the same reconciler, tagged `extractor: "llm"`. Cost bound: ≤ ~40
  short calls per GW.

### 5.4 Output compatibility

`press_conferences.json` keeps its shape (`gameweeks.{gw}.clubs.{club}.players[]`
+ `all_player_news[]`) so intel_03/04 run unchanged on day one. New fields are
additive: per-player `score`, `n_sources`, `conflict`, `sources`; per-GW
`coverage` (§7). intel_03's 65/35 merge then optionally upgrades to consume
the reconciled score directly (a later, separately-validated change — the
cross-club fallback lesson from the 2468 run says: never change scraper and
merge semantics in the same step).

---

## 6. Scheduling — deadline-aware, zero manual input

### 6.1 The window model

Everything keys off `bootstrap-static` `events[].deadline_time` (UTC) — fetched
live, so blank/double/rescheduled GWs and midweek deadlines are handled
automatically with no season calendar in code.

For the next unfinished GW with deadline D:

| Window | Cadence | Rationale |
|---|---|---|
| D-96h → D-48h | every 12h | early pressers for midweek-adjacent GWs; cheap |
| D-48h → D-24h | every 6h | main presser wave (Thu/Fri for a Sat deadline) |
| D-24h → D-4h | every 2h | liveblog updates, late fitness tests |
| D-4h → D | one final sweep at D-2h | last-minute team news; freeze output |
| outside windows | daily health ping | source-health + FPL flag drift only |

Each tick is **idempotent**: discover → fetch (conditional GET / content-hash;
skip unchanged) → extract → append claims → reconcile → rewrite output. Ticks
are cheap when nothing changed; the ledger makes re-runs safe.

### 6.2 Trigger mechanism on this project's infrastructure

The repo has no always-on server, no local Python (Docker `fpl-sim` only), and
the machine may be off. Two options, recommendation first:

- **Recommended: GitHub Actions cron.** A workflow on `cron: "17 */2 * * *"`
  runs a gate step (fetch deadline, compute window, exit 0 fast if outside
  cadence), then the scrape inside the existing container image, then commits
  `data/intel/press_claims/` + `press_conferences.json`. Free-tier minutes are
  ample (≤ ~25 short runs per GW); the git history doubles as the snapshot
  archive; runs happen even with the laptop closed. Secrets (FFS credentials,
  Gemini key) live in Actions secrets, not source (fixes §1.4).
- **Fallback: Windows Task Scheduler** firing the same gate+scrape via
  `docker run --rm -v repo:/app fpl-sim python -u pipeline/intel_02_scrape.py
  --tick`. Same code path, machine must be on.

The gate logic lives in the script (`--tick` decides window/cadence from
state in the repo: last tick time per GW in a small state file), so the
external trigger stays a dumb fixed-interval cron — no dynamic cron rewriting.

### 6.3 Pages that don't exist yet

The window model dissolves Problem 1's "page doesn't exist yet" issue: we never
construct-and-hope. Each tick runs *discovery* (feeds/sitemaps/APIs/index
pages, §4) and scrapes whatever exists *now*. An article published Friday 13:00
is found by the Friday 14:xx tick. Absence is a normal tick outcome, recorded
as such — not an error.

---

## 7. Coverage accounting and failure visibility

Per GW, the output carries a `coverage` block updated every tick:

```json
"coverage": {
  "clubs_covered": 18,
  "missing": ["Burnley", "Wolves"],
  "per_club": {"Newcastle": {"tiers": [1,2], "claims": 7, "latest": "..."}}
}
```

- **T-24h escalation:** clubs with zero Tier 1–3 claims trigger the Tier 4
  (official site) adapter for exactly those clubs.
- **Parse alarms:** an adapter returning 0 claims from an HTTP-200 fetch that
  previously yielded claims raises `structure_drift` in a per-source health
  file (consecutive-failure counters). This is the alarm §1.1 never had.
- **Freeze report:** the D-2h sweep writes a one-screen summary (clubs
  covered, conflicts, unresolved player names, sources degraded) — the
  human-auditable artifact per GW.
- **Hard floor:** if `clubs_covered < 15` at freeze, the GW record is marked
  `"degraded": true` so intel_03 can widen uncertainty instead of trusting a
  thin scrape.

---

## 8. Explicitly out of scope

- Changing intel_03's 65/35 merge weights or the intel_06 penalty formula —
  separate change, separate backtest (lesson: the tested cross-club fallback
  cost ~113 pts by cascading into squad decisions).
- Any Twitter/X or paid-API source (auth instability, cost).
- Historical backfill of the 7 missed clubs for 2025-26 (possible via the same
  adapters against archived URLs, but a thesis-analysis task, not pipeline).

## 9. Build plan (after design approval)

1. ✅ DONE (2026-07-07) — `pipeline/intel_identity.py` + `tests/test_intel_identity.py`
   (13/13, incl. all 7 §1.1 failing headers; never-guess invariants).
2. ✅ DONE (2026-07-07) — `pipeline/intel_02_sources.py` (4 Tier 1 adapters),
   `pipeline/intel_02_ledger.py` (ledger + reconciler, 14/14 tests),
   `pipeline/intel_02_scrape.py` (orchestrator + health file + compat output
   to `press_conferences_v2.json`). Validated against archived 2025-26
   content (`scripts/validate_scraper_v2.py`, GW12/22/38): 20/20 club
   coverage each GW, **all 7 blind clubs recovered** (see
   `data/intel/validation_2526/validation_report.json`). Docker image:
   `fpl-scrape` (= fpl-sim + requests/bs4). **The Newcastle class of miss is
   fixed** without touching the FFS liveblog.
3. ✅ DONE (2026-07-07) — `FfsTeamNewsAdapter` (Tier 2) in
   `pipeline/intel_02_sources.py`: WP REST discovery (category 3 + deadline
   window, title-regex re-filter; secondary title-search fallback) replaces
   KNOWN_URLS and the member cookie outright. Extraction via the identity
   registry (unmatched headers = boundaries), per-player parenthetical
   injuries, stage-A regex classifier ported verbatim; stage-B LLM
   gap-filling in `pipeline/intel_02_llm_extract.py` (Gemini Flash, roster-
   grounded strict-JSON, ≤40 calls/GW, degrades to regex-only without key).
   Also accepts raw archived HTML pages (step 6 path). Tests:
   `tests/test_intel_02_ffs_liveblog.py`, **15/15**. Validated against the
   WP post archive (full history, no Wayback needed): GW18 3 editions
   (incl. the Xmas Eve ad-hoc slug) → **20/20 clubs from the liveblog
   alone**; GW19 2 editions → 15/20 (all presser-holding clubs); GW22
   2 editions, 146 claims, 20/20 — the GW18/19 midweek/holiday holes from
   the step-2 validation are plugged.
4. ✅ DONE (2026-07-07) — Tier 3/4 adapters + T-24h escalation.
   `SkySportsAdapter` (Tier 3): Google news sitemap discovery per tick +
   `{slug}-news` per-club index for escalation. `ReachLocalAdapter` (Tier 4,
   escalation-only): per-club `?service=rss` → JSON-LD `articleBody`. Both
   feed a shared **story extractor** (`_extract_stories`) — the story-per-
   article analogue of the section-per-club Tier 1/2 parsers: headline club
   attribution via the identity registry (ambiguous/no-club → skipped, §3.1),
   a roster-grounded accent-tolerant regex scan (stage A), and LLM gap-fill
   (stage B). `intel_02_scrape.py --escalate` runs the §7 escalation:
   `missing_clubs()` (zero Tier 1-3 claims after reconcile) → Sky per-club +
   Reach per-club for exactly those clubs → re-reconcile. Tests:
   `tests/test_intel_02_tier34.py`, **20/20**; all live paths smoke-verified
   against real Sky/Reach endpoints (in-season injury signal not validatable
   off-season). `REACH_OUTLETS`/`SKY_SLUGS` are off-season-probed and
   LOWER-CONFIDENCE (§4.7) — re-verify on GW1 2026-27 via step-5's checklist
   and the §7 parse alarms. The window gate that *decides when* to set
   `--escalate` belongs to step 5.
5. Scheduler gate + GitHub Actions workflow; one full GW dress rehearsal
   against a live deadline; freeze-report review.
   **Season-start re-verification (all discovery was verified during the
   2026 off-season):** on the first GW of 2026-27, confirm the Guardian
   series tag and PL `franchise:predicted-line-ups` tag resume, the FFS
   liveblog title format is unchanged, and SportsGambler/KnocksAndBans
   markup survived any summer redesign. The §7 parse alarms catch this
   automatically, but week 1 deserves a manual look at the freeze report.
6. Re-scrape validation: run v2 against 2025-26 archived FFS URLs, diff player
   sets vs old output — expect the 7 missing clubs to appear; quantify what the
   2468 run never saw.
