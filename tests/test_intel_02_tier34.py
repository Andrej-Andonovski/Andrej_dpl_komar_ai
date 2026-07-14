"""
tests/test_intel_02_tier34.py — Tier 3/4 adapter tests
(scraper redesign step 4, docs/press_scraper_redesign.md §4.5/§4.7/§7).

Covers: Sky news-sitemap discovery (football + keyword filter, date parse),
Reach per-club RSS parse and JSON-LD articleBody extraction, the shared
story extractor (headline club attribution, ambiguous/no-club skip, roster-
grounded regex scan, accent-tolerant name matching, stage-B LLM gap-fill),
the escalation adapters' graceful no-op on unmapped clubs, and run_gw's
missing_clubs escalation target selection.

Uses the real 2025-26 identity reference (data/intel/fpl_live.json).
No pytest dependency. Run directly:
  docker run --rm -v "<repo>:/app" -w /app fpl-scrape python tests/test_intel_02_tier34.py
Exit code 0 = all pass.
"""
import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
from intel_identity import ClubRegistry, PlayerResolver
from intel_02_ledger import reconcile_player
from intel_02_sources import (SkySportsAdapter, ReachLocalAdapter,
                              REACH_OUTLETS, SKY_SLUGS,
                              _attribute_club, _fold, _player_name_tokens)
from intel_02_scrape import missing_clubs

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FPL_LIVE = os.path.join(ROOT, "data", "intel", "fpl_live.json")

REGISTRY = ClubRegistry.from_fpl_live(FPL_LIVE)
RESOLVER = PlayerResolver.from_fpl_live(FPL_LIVE)
SKY = SkySportsAdapter()
REACH = ReachLocalAdapter()

FakeClub = namedtuple("FakeClub", ["team_id", "name", "short_name", "slug"])


# ── Sky news-sitemap discovery ────────────────────────────────────────────────

SITEMAP_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
  <url>
    <loc>https://www.skysports.com/football/news/11095/1/newcastle-team-news-howe-confirms-livramento-out</loc>
    <news:news><news:publication><news:name>Sky Sports</news:name></news:publication>
      <news:publication_date>2026-01-16</news:publication_date>
      <news:title>Newcastle team news: Howe confirms Livramento ruled out for the trip</news:title>
    </news:news>
  </url>
  <url>
    <loc>https://www.skysports.com/tennis/news/32498/1/wimbledon-preview</loc>
    <news:news><news:publication><news:name>Sky Sports</news:name></news:publication>
      <news:publication_date>2026-01-16</news:publication_date>
      <news:title>Wimbledon: team news for the final</news:title>
    </news:news>
  </url>
  <url>
    <loc>https://www.skysports.com/football/news/12/1/arsenal-transfers-paper-talk</loc>
    <news:news><news:publication><news:name>Sky Sports</news:name></news:publication>
      <news:publication_date>2026-01-16</news:publication_date>
      <news:title>Arsenal transfers: paper talk round-up on deadline day</news:title>
    </news:news>
  </url>
  <url>
    <loc>https://www.skysports.com/football/news/13/1/wolves-injury-latest</loc>
    <news:news><news:publication><news:name>Sky Sports</news:name></news:publication>
      <news:publication_date>2026-01-15</news:publication_date>
      <news:title>Wolves injury latest ahead of the weekend</news:title>
    </news:news>
  </url>
</urlset>"""


def test_sky_sitemap_filters_football_and_keywords():
    cands = SkySportsAdapter._parse_sitemap(SITEMAP_XML)
    urls = [c["url"] for c in cands]
    # football + availability keyword only: Newcastle team-news, Wolves injury.
    # tennis (non-football) and Arsenal "transfers paper talk" (no keyword)
    # are dropped.
    assert len(cands) == 2, urls
    assert any("newcastle-team-news" in u for u in urls)
    assert any("wolves-injury-latest" in u for u in urls)
    assert not any("tennis" in u for u in urls)
    assert not any("paper-talk" in u for u in urls)


def test_sky_sitemap_parses_date_and_title():
    c = SkySportsAdapter._parse_sitemap(SITEMAP_XML)[0]
    assert c["title"].startswith("Newcastle team news")
    assert c["published"] == "2026-01-16T12:00:00+00:00"


# ── story extractor: club attribution (headline, never-guess) ─────────────────

def test_attribute_club_from_headline():
    c = _attribute_club("Newcastle team news: Howe update", "", REGISTRY)
    assert c is not None and c.short_name == "NEW"


def test_attribute_club_ambiguous_two_clubs_skipped():
    # a headline naming two clubs must not be guessed (§3.1)
    c = _attribute_club("Arsenal transfers: Aston Villa demand record fee",
                        "", REGISTRY)
    assert c is None


def test_attribute_club_none_when_no_club():
    c = _attribute_club("World Cup 2026: England out after freak injury",
                        "The tournament continues without them.", REGISTRY)
    assert c is None


# ── story extractor: roster-grounded regex scan (Sky) ─────────────────────────

SKY_STORY = {
    "title": "Newcastle team news: Howe injury update before the weekend",
    "url": "https://www.skysports.com/football/news/1/2/newcastle-team-news",
    "published": "2026-01-16T10:30:00+00:00",
    "html": """<html><body><article>
      <p>Eddie Howe delivered his pre-match press conference on Friday.</p>
      <p>Tino Livramento has been ruled out of the trip with a hamstring
         problem picked up in training.</p>
      <p>There was better news elsewhere as Anthony Gordon is fit and
         available again after his own knock.</p>
      <p>The manager also spoke about the busy schedule ahead.</p>
    </article></body></html>""",
}


def test_sky_extract_roster_scan_classifies():
    claims, stats = SKY.extract([SKY_STORY], 22, REGISTRY, RESOLVER)
    assert stats["clubs_seen"] == ["NEW"], stats["clubs_seen"]
    by_name = {c["player_raw"]: c for c in claims}
    assert "Livramento" in by_name, [c["player_raw"] for c in claims]
    liv = by_name["Livramento"]
    assert liv["status_claim"] == "out"
    assert liv["tier"] == 3 and liv["source"] == "sky"
    assert liv["extractor"] == "regex"
    assert liv["injury"] == "hamstring"
    assert liv["player_id"] is not None
    # publication time is the assertion time for an article (§5.1)
    assert liv["observed_at"] == "2026-01-16T10:30:00+00:00"
    gordon = next(c for c in claims if "Gordon" in c["player_raw"])
    assert gordon["status_claim"] == "available"


def test_sky_extract_unattributed_story_logged_not_claimed():
    story = {"title": "Paper talk: the weekend's biggest rumours",
             "url": "https://www.skysports.com/football/news/1/2/paper",
             "published": "2026-01-16T10:30:00+00:00",
             "html": "<article><p>A quiet round-up of gossip.</p></article>"}
    claims, stats = SKY.extract([story], 22, REGISTRY, RESOLVER)
    assert claims == []
    assert stats["unmatched_clubs"], stats


# ── accent-tolerant matching (English spelling vs FPL diacritics) ─────────────

def test_roster_scan_accent_tolerant():
    # body writes "Schar"/"Odegaard"; roster web_names are "Schär"/"Ødegaard"
    story = {"title": "Newcastle team news: defenders latest",
             "url": "https://x/y", "published": "2026-01-16T10:30:00+00:00",
             "html": "<article><p>Fabian Schar has been ruled out with an "
                     "ankle injury.</p></article>"}
    claims, _ = SKY.extract([story], 22, REGISTRY, RESOLVER)
    schar = next((c for c in claims if c["player_id"] is not None), None)
    assert schar is not None, "accented roster name not matched from 'Schar'"
    assert schar["status_claim"] == "out"


def test_fold_handles_diacritics_and_special_letters():
    assert _fold("Schär") == "schar"
    assert _fold("Ødegaard") == "odegaard"
    assert _fold("Guimarães") == "guimaraes"
    assert _fold("Bruno  G.  ") == "bruno g."


def test_player_name_tokens_skips_dotted_webnames():
    # "J.Ramsey" -> web_name skipped (initial+dot), last name used
    toks = _player_name_tokens({"web_name": "J.Ramsey",
                                "full_name": "Jacob Ramsey"})
    assert "J.Ramsey" not in toks
    assert "Ramsey" in toks


# ── Reach RSS + JSON-LD extraction ────────────────────────────────────────────

RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>ChronicleLive - Newcastle United FC</title>
  <item>
    <title>Newcastle injury blow as key defender ruled out</title>
    <link>https://www.chroniclelive.co.uk/sport/football/news/1</link>
    <pubDate>Fri, 16 Jan 2026 09:59:00 +0000</pubDate>
  </item>
  <item>
    <title>Newcastle transfer news LIVE: latest on summer window</title>
    <link>https://www.chroniclelive.co.uk/sport/football/transfer-news/2</link>
    <pubDate>Fri, 16 Jan 2026 08:00:00 +0000</pubDate>
  </item>
</channel></rss>"""


def test_reach_rss_parse():
    items = ReachLocalAdapter._parse_rss(RSS_XML)
    assert len(items) == 2
    assert items[0]["url"].endswith("/news/1")
    assert items[0]["published"] == "2026-01-16T09:59:00+00:00"


def test_reach_jsonld_body_extraction():
    html = ('<html><head>'
            '<script type="application/ld+json">'
            '{"@type":"NewsArticle","headline":"h",'
            '"articleBody":"Sven Botman is a doubt for the weekend with a '
            'knee problem, the manager confirmed."}'
            '</script></head><body><p>ignored</p></body></html>')
    body = ReachLocalAdapter._body_from_html(html)
    assert body.startswith("Sven Botman is a doubt")


def test_reach_jsonld_graph_wrapped():
    html = ('<script type="application/ld+json">'
            '{"@context":"x","@graph":[{"@type":"WebPage"},'
            '{"@type":"NewsArticle","articleBody":"Body text here."}]}'
            '</script>')
    assert ReachLocalAdapter._body_from_html(html) == "Body text here."


def test_reach_extract_from_body_story():
    story = {"title": "Newcastle injury blow as defender ruled out",
             "url": "https://www.chroniclelive.co.uk/sport/football/news/1",
             "published": "2026-01-16T09:59:00+00:00",
             "body": "Sven Botman has been ruled out for the weekend fixture "
                     "with a knee injury, Eddie Howe confirmed on Friday."}
    claims, stats = REACH.extract([story], 22, REGISTRY, RESOLVER)
    assert stats["clubs_seen"] == ["NEW"]
    botman = next(c for c in claims if "Botman" in c["player_raw"])
    assert botman["tier"] == 4 and botman["source"] == "reach"
    assert botman["status_claim"] == "out"
    assert botman["player_id"] is not None


# ── stage-B LLM gap-fill for thin story sections ──────────────────────────────

class StubLLM:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def extract_club(self, club_name, section_text, roster):
        self.calls.append((club_name, section_text, tuple(roster)))
        return self.rows


# prose long enough, but the availability facts name no roster player in a
# way the regex scan catches -> stage B fires
GAP_STORY = {
    "title": "Leeds United team news: head coach gives fitness update",
    "url": "https://x/leeds", "published": "2026-01-16T10:00:00+00:00",
    "body": ("The head coach spoke at length about the demands of the "
             "congested festive schedule and the way he intends to rotate "
             "his squad across the coming weeks, before turning to the "
             "fitness of his players and confirming that the club captain "
             "had suffered a fresh setback in training and would not be "
             "involved at the weekend, an unwelcome blow before a difficult "
             "run of fixtures for the newly promoted side."),
}


def test_llm_gap_detected_and_claims_tagged():
    leeds = REGISTRY.match("LEEDS UNITED")
    roster = [p["web_name"] for p in RESOLVER.team_players(leeds.team_id)]
    stub = StubLLM([
        {"player": roster[0], "status": "out", "injury": "knock",
         "quote": "suffered a fresh setback in training"},
        {"player": "Nobody Real", "status": "bogus", "injury": "",
         "quote": ""},                       # invalid status -> dropped
    ])
    claims, stats = REACH.extract([GAP_STORY], 22, REGISTRY, RESOLVER,
                                  llm=stub)
    assert stats["llm_gap_clubs"] == ["LEE"], stats["llm_gap_clubs"]
    assert len(stub.calls) == 1
    _, prose, sent_roster = stub.calls[0]
    assert "setback in training" in prose
    assert sent_roster == tuple(roster), "roster grounding not passed"
    llm_claims = [c for c in claims if c["extractor"] == "llm"]
    assert len(llm_claims) == 1, llm_claims
    assert llm_claims[0]["status_claim"] == "out"
    assert llm_claims[0]["tier"] == 4
    assert stats["llm_claims"] == 1


def test_llm_not_called_when_regex_covered():
    stub = StubLLM([])
    _, stats = SKY.extract([SKY_STORY], 22, REGISTRY, RESOLVER, llm=stub)
    assert stub.calls == [], "stage B fired for a section regex already covered"
    assert stats["llm_gap_clubs"] == []


# ── escalation plumbing (no network) ──────────────────────────────────────────

def test_fetch_club_noop_on_unmapped_slug():
    # a club with no Sky/Reach mapping returns [] before any network call
    fake = FakeClub(999, "Nowhere FC", "NOW", "nowhere-fc")
    assert SKY.fetch_club(None, fake) == []
    assert REACH.fetch_club(None, fake) == []


def test_missing_clubs_selects_uncovered():
    all_clubs = REGISTRY.all_clubs()
    covered = all_clubs[:18]
    records = [{"club_id": c.team_id} for c in covered]
    missing = missing_clubs(records, REGISTRY)
    missing_ids = {c.team_id for c in missing}
    assert missing_ids == {all_clubs[18].team_id, all_clubs[19].team_id}


def test_reach_outlets_and_sky_slugs_anchor():
    # the one live-verified Reach anchor and Sky slug must be present
    assert REACH_OUTLETS["newcastle"] == (
        "https://www.chroniclelive.co.uk", "all-about/newcastle-united-fc")
    assert SKY_SLUGS["newcastle"] == "newcastle-united"
    # every Reach outlet slug maps to a real registry club
    for slug in REACH_OUTLETS:
        assert REGISTRY.match_any(slug.replace("-", " ")) is not None or \
            slug in ("nottm-forest", "man-utd", "man-city", "spurs",
                     "west-ham", "crystal-palace", "aston-villa"), slug


# ── multi-source reconciliation across tiers ──────────────────────────────────

def test_sky_and_reach_claims_reconcile_together():
    # Sky (tier 3) says out, Reach (tier 4) agrees out — same player, one GW
    sky_claims, _ = SKY.extract([SKY_STORY], 22, REGISTRY, RESOLVER)
    reach_story = {"title": "Newcastle injury news: Livramento latest",
                   "url": "https://www.chroniclelive.co.uk/x",
                   "published": "2026-01-16T11:00:00+00:00",
                   "body": "Tino Livramento remains ruled out with a "
                           "hamstring injury and will miss the weekend."}
    reach_claims, _ = REACH.extract([reach_story], 22, REGISTRY, RESOLVER)
    liv_sky = [c for c in sky_claims if "Livramento" in c["player_raw"]]
    liv_reach = [c for c in reach_claims if "Livramento" in c["player_raw"]]
    assert liv_sky and liv_reach
    rec = reconcile_player(liv_sky + liv_reach,
                           ref_time="2026-01-16T18:30:00+00:00")
    assert rec["status"] == "out"
    assert rec["n_sources"] == 2, rec        # sky + reach counted distinctly
    assert not rec["conflict"]


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
