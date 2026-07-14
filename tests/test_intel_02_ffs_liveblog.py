"""
tests/test_intel_02_ffs_liveblog.py — FFS liveblog adapter tests
(scraper redesign step 3, docs/press_scraper_redesign.md §4.1/§5.3).

Covers: discovery title matching (incl. the "gameweek 2" vs "gameweek 22"
LIKE hazard), registry-based club headers (the 7 blind-club fix applied to
the liveblog itself), unmatched-header boundaries (§1.2 misattribution
regression), per-player parenthetical injuries, stage-A classification,
stage-B LLM gap detection with a stub extractor, multi-edition supersession
through the reconciler, and the archived-full-page input path.

Uses the real 2025-26 identity reference (data/intel/fpl_live.json).
No pytest dependency. Run directly:
  docker run --rm -v "<repo>:/app" -w /app fpl-scrape python tests/test_intel_02_ffs_liveblog.py
Exit code 0 = all pass.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
from intel_identity import ClubRegistry, PlayerResolver
from intel_02_ledger import reconcile_player
from intel_02_sources import FfsTeamNewsAdapter, classify_news, _wp_iso

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FPL_LIVE = os.path.join(ROOT, "data", "intel", "fpl_live.json")

REGISTRY = ClubRegistry.from_fpl_live(FPL_LIVE)
RESOLVER = PlayerResolver.from_fpl_live(FPL_LIVE)
ADAPTER = FfsTeamNewsAdapter()


def _post(body, *, link="https://ffs.example/gw22-friday",
          date="2026-01-16T13:00:00", modified="2026-01-16T19:09:45"):
    return {"content": {"rendered": body}, "link": link,
            "date_gmt": date, "modified_gmt": modified,
            "title": {"rendered": "FPL Gameweek 22 team news: Friday's "
                                  "live injury updates"}}


# Mirrors the real GW22 Friday edition structure fetched live 2026-07-07:
# full-caps club h2s, non-club h2s around them, strongs + blockquote quotes.
GW22_BODY = """
<h2 class="wp-block-heading">KEY GAMEWEEK 22 INJURY NEWS FROM FRIDAY</h2>
<p>The round-up paragraph mentioning <strong>Erling Haaland</strong> stays
unattributed because no club section is open yet.</p>
<h2 class="wp-block-heading" id="nufc">NEWCASTLE UNITED</h2>
<p><strong>Fabian Schar</strong> underwent ankle surgery this week, with the
defender set to be out until the spring.</p>
<blockquote class="wp-block-quote"><p>&#8220;He had successful surgery
yesterday.&#8221; &#8211; Eddie Howe</p></blockquote>
<p><strong>Emil Krafth</strong> (knee) and <strong>Tino Livramento</strong>
(hamstring) also remain out but <strong>Dan Burn</strong> (rib) is closing
in on a return.</p>
<h2 class="wp-block-heading">FRIDAY&#8217;S PRESS CONFERENCE TIMES</h2>
<p><strong>Mikel Arteta</strong> speaks to the media at 1pm ahead of the
trip, and <strong>Pep Guardiola</strong> follows at 1:30pm.</p>
<h2 class="wp-block-heading">WOLVERHAMPTON WANDERERS</h2>
<p><strong>Sam Johnstone</strong> faces a late fitness test after a
knock picked up in training.</p>
<h2 class="wp-block-heading">TOTTENHAM HOTSPUR</h2>
<p>No fresh news.</p>
"""


def _claims_for(claims, short_name):
    club = REGISTRY.match(short_name)
    return [c for c in claims if c["club_id"] == club.team_id]


# ── discovery: title regex ────────────────────────────────────────────────────

def test_title_regex_matches_real_editions():
    # exact titles returned by the WP API for the GW22 window (live probe)
    titles_22 = ["FPL Gameweek 22 team news: Friday’s live injury updates",
                 "FPL Gameweek 22 team news: Thursday’s live injury updates"]
    rex = FfsTeamNewsAdapter._title_re(22)
    for t in titles_22:
        assert rex.search(t), f"GW22 edition title not matched: {t!r}"


def test_title_regex_rejects_other_category_posts():
    rex = FfsTeamNewsAdapter._title_re(22)
    for t in ["Hincapie, Saliba, Calafiori: The latest FPL team news",
              "Wolves v Newcastle predicted line-ups + FPL team news",
              "Brighton v Bournemouth predicted line-ups + FPL team news"]:
        assert not rex.search(t), f"non-edition title matched: {t!r}"


def test_title_regex_gw_number_boundary():
    # LIKE-search hazard (§4.1): "gameweek 2" must not hit "gameweek 22"
    t22 = "FPL Gameweek 22 team news: Friday's live injury updates"
    assert not FfsTeamNewsAdapter._title_re(2).search(t22)
    assert FfsTeamNewsAdapter._title_re(22).search(t22)
    t2 = "FPL Gameweek 2 team news: Friday's live injury updates"
    assert FfsTeamNewsAdapter._title_re(2).search(t2)
    assert not FfsTeamNewsAdapter._title_re(22).search(t2)


# ── extraction: club sections via the registry ───────────────────────────────

def test_blind_club_headers_extracted():
    claims, stats = ADAPTER.extract([_post(GW22_BODY)], 22, REGISTRY, RESOLVER)
    assert set(stats["clubs_seen"]) == {"NEW", "WOL", "TOT"}, stats["clubs_seen"]
    new = _claims_for(claims, "NEWCASTLE UNITED")
    names = sorted(c["player_raw"] for c in new)
    assert names == ["Dan Burn", "Emil Krafth", "Fabian Schar",
                     "Tino Livramento"], names


def test_claim_schema_fields():
    claims, _ = ADAPTER.extract([_post(GW22_BODY)], 22, REGISTRY, RESOLVER)
    schar = next(c for c in claims if c["player_raw"] == "Fabian Schar")
    assert schar["source"] == "ffs_teamnews"
    assert schar["tier"] == 2
    assert schar["gw"] == 22
    assert schar["extractor"] == "regex"
    assert schar["status_claim"] == "out"           # "set to be out"
    assert schar["player_id"] is not None, "Schär did not resolve"
    assert schar["observed_at"] == "2026-01-16T19:09:45+00:00"   # modified_gmt
    assert schar["published_at"] == "2026-01-16T13:00:00+00:00"  # date_gmt


def test_unmatched_header_is_boundary_not_content():
    # §1.2 regression: content after "FRIDAY'S PRESS CONFERENCE TIMES" must
    # not attach to Newcastle (the previous club section)
    claims, stats = ADAPTER.extract([_post(GW22_BODY)], 22, REGISTRY, RESOLVER)
    raws = [c["player_raw"] for c in claims]
    assert "Mikel Arteta" not in raws, "manager under non-club header claimed"
    assert "Pep Guardiola" not in raws
    assert "Erling Haaland" not in raws, "pre-section round-up text claimed"
    assert any("PRESS CONFERENCE" in u for u in stats["unmatched_clubs"]), \
        stats["unmatched_clubs"]


def test_per_player_parenthetical_injuries():
    claims, _ = ADAPTER.extract([_post(GW22_BODY)], 22, REGISTRY, RESOLVER)
    by_name = {c["player_raw"]: c for c in claims}
    assert by_name["Emil Krafth"]["injury"] == "knee"
    assert by_name["Tino Livramento"]["injury"] == "hamstring"
    assert by_name["Dan Burn"]["injury"] == "rib"


def test_wolves_fitness_test_doubtful():
    claims, _ = ADAPTER.extract([_post(GW22_BODY)], 22, REGISTRY, RESOLVER)
    keeper = next(c for c in claims if "Johnstone" in c["player_raw"])
    assert keeper["status_claim"] == "doubtful", keeper
    assert keeper["player_id"] is not None, "Johnstone did not resolve"


# ── stage A classifier port ───────────────────────────────────────────────────

def test_classify_news_golden():
    cases = [("He has been ruled out for three weeks", "out"),
             ("faces a late fitness test", "doubtful"),
             ("is suspended after his red card", "suspended"),
             ("is fit and in the squad", "available"),
             ("with Saka missing out", "available"),   # old-scraper pre-check
             ("started every session this week", "unknown")]
    for text, want in cases:
        got = classify_news(text)
        assert got == want, f"classify_news({text!r}) = {got}, want {want}"


# ── stage B: LLM gap detection ────────────────────────────────────────────────

class StubLLM:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def extract_club(self, club_name, section_text, roster):
        self.calls.append((club_name, section_text, tuple(roster)))
        return self.rows


# Long prose, availability facts NOT in <strong> -> regex finds nothing
GAP_BODY = """
<h2 class="wp-block-heading">LEEDS UNITED</h2>
<p>The head coach spoke for a long while about the busy schedule and rotation
across the squad before confirming that the club captain had suffered a
setback in training on Thursday and would play no part this weekend, while
the goalkeeper who missed the last three matches has now returned to full
training and should be involved again, barring any late reaction.</p>
"""


def test_llm_gap_detected_and_stub_claims_tagged():
    leeds = REGISTRY.match("LEEDS UNITED")
    roster = [p["web_name"] for p in RESOLVER.team_players(leeds.team_id)]
    assert roster, "no Leeds roster in reference — fixture broken"
    stub = StubLLM([
        {"player": roster[0], "status": "out", "injury": "knock",
         "quote": "suffered a setback in training"},
        {"player": "Nobody Real", "status": "bogus", "injury": "",
         "quote": ""},                    # invalid status -> dropped
    ])
    claims, stats = ADAPTER.extract([_post(GAP_BODY)], 22, REGISTRY, RESOLVER,
                                    llm=stub)
    assert stats["llm_gap_clubs"] == ["LEE"], stats["llm_gap_clubs"]
    assert len(stub.calls) == 1
    club_name, prose, sent_roster = stub.calls[0]
    assert "setback in training" in prose
    assert sent_roster == tuple(roster), "roster grounding not passed"
    llm_claims = [c for c in claims if c["extractor"] == "llm"]
    assert len(llm_claims) == 1, llm_claims
    assert llm_claims[0]["status_claim"] == "out"
    assert llm_claims[0]["player_id"] is not None
    assert stats["llm_claims"] == 1


def test_llm_not_called_for_covered_or_trivial_sections():
    stub = StubLLM([])
    _, stats = ADAPTER.extract([_post(GW22_BODY)], 22, REGISTRY, RESOLVER,
                               llm=stub)
    called = [c[0] for c in stub.calls]
    # Newcastle: 4 regex claims -> covered. Spurs: "No fresh news." is under
    # the prose threshold. Wolves: 1 claim + short section -> gap candidate.
    assert "Newcastle" not in " ".join(called)
    assert stats["llm_gap_clubs"] == [], (
        "short sections must not trigger stage B: " + str(stats))


# ── multi-edition supersession through the reconciler ───────────────────────

def test_thursday_then_friday_edition_last_write_wins():
    thu = _post("""<h2>WOLVERHAMPTON WANDERERS</h2>
                   <p><strong>Sam Johnstone</strong> is a doubt and
                   faces a late fitness test.</p>""",
                link="https://ffs.example/gw22-thursday",
                date="2026-01-15T13:55:00", modified="2026-01-15T18:00:00")
    fri = _post("""<h2>WOLVERHAMPTON WANDERERS</h2>
                   <p><strong>Sam Johnstone</strong> has been ruled
                   out of the trip.</p>""",
                date="2026-01-16T13:00:00", modified="2026-01-16T19:00:00")
    claims, _ = ADAPTER.extract([thu, fri], 22, REGISTRY, RESOLVER)
    assert len(claims) == 2, claims
    rec = reconcile_player(claims)
    assert rec["n_sources"] == 1          # both editions are ffs_teamnews
    assert rec["status"] == "out", rec    # Friday supersedes Thursday


# ── archived full-page input (step 6 re-scrape path) ─────────────────────────

def test_raw_html_full_page_narrows_to_article():
    page = """<html><body>
    <div class="sidebar"><p><strong>Scout Picks</strong> our team for the
    week, and <strong>Harry Kane</strong> tops the watchlist.</p></div>
    <div class="entry-content">
      <h2 class="wp-block-heading">NEWCASTLE UNITED</h2>
      <p><strong>Fabian Schar</strong> has been ruled out.</p>
    </div>
    </body></html>"""
    claims, stats = ADAPTER.extract(page, 22, REGISTRY, RESOLVER,
                                    url="https://web.archive.org/x",
                                    observed_at="2026-01-16T12:00:00+00:00",
                                    published_at="2026-01-16T10:00:00+00:00")
    assert [c["player_raw"] for c in claims] == ["Fabian Schar"], claims
    assert claims[0]["observed_at"] == "2026-01-16T12:00:00+00:00"
    assert claims[0]["url"] == "https://web.archive.org/x"


# ── misc helpers ──────────────────────────────────────────────────────────────

def test_wp_iso():
    assert _wp_iso("2026-01-16T13:00:00") == "2026-01-16T13:00:00+00:00"
    assert _wp_iso("2026-01-16T13:00:00+00:00") == "2026-01-16T13:00:00+00:00"
    assert _wp_iso(None) is None


def test_strong_only_club_header_variant():
    body = """<p><strong>Arsenal:</strong></p>
              <p><strong>Bukayo Saka</strong> is fit again and available.</p>"""
    claims, stats = ADAPTER.extract([_post(body)], 22, REGISTRY, RESOLVER)
    assert stats["clubs_seen"] == ["ARS"]
    assert claims and claims[0]["player_raw"] == "Bukayo Saka"
    assert claims[0]["status_claim"] == "available"


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
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
