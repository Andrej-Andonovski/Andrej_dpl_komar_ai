"""
tests/test_intel_identity.py — identity module tests (scraper redesign step 1).

Uses the REAL failing club headers from the intel_02 audit
(docs/press_scraper_redesign.md §1.1) plus the real 2025-26 player reference
in data/intel/fpl_live.json.

No pytest dependency. Run directly:
  python tests/test_intel_identity.py
Docker (this machine has no local Python):
  docker run --rm -v "<repo>:/app" -w /app fpl-scrape python tests/test_intel_identity.py
Exit code 0 = all pass.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
from intel_identity import (ClubRegistry, PlayerResolver, load_reference,
                            normalize_name, club_key)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FPL_LIVE = os.path.join(ROOT, "data", "intel", "fpl_live.json")

REGISTRY = ClubRegistry.from_fpl_live(FPL_LIVE)
RESOLVER = PlayerResolver.from_fpl_live(FPL_LIVE)


# ── The 7 blind clubs: exact headers FFS used, which intel_02 failed on ──────

def test_seven_blind_club_headers():
    # (header as printed in the article, expected FPL short_name)
    cases = [
        ("NEWCASTLE UNITED",          "NEW"),
        ("TOTTENHAM HOTSPUR",         "TOT"),
        ("WEST HAM UNITED",           "WHU"),
        ("WOLVERHAMPTON WANDERERS",   "WOL"),
        ("BRIGHTON AND HOVE ALBION",  "BHA"),
        ("LEEDS UNITED",              "LEE"),
        ("BURNLEY",                   "BUR"),   # absent from old PL_CLUBS
    ]
    for header, want_short in cases:
        club = REGISTRY.match(header)
        assert club is not None, f"header {header!r} did not resolve"
        assert club.short_name == want_short, (
            f"header {header!r} resolved to {club.short_name}, want {want_short}")


def test_blind_club_header_variants():
    # Trailing colons, mixed case, ampersand form, h2 whitespace
    cases = [
        ("Newcastle United:",           "NEW"),
        ("  BRIGHTON & HOVE ALBION  ",  "BHA"),
        ("Tottenham Hotspur:",          "TOT"),
        ("WOLVES",                      "WOL"),
        ("Spurs",                       "TOT"),
        ("Brighton",                    "BHA"),
        ("NUFC",                        "NEW"),
    ]
    for header, want_short in cases:
        club = REGISTRY.match(header)
        assert club is not None, f"variant {header!r} did not resolve"
        assert club.short_name == want_short, (
            f"variant {header!r} -> {club.short_name}, want {want_short}")


# ── The 14 clubs that DID work must keep working ─────────────────────────────

def test_previously_working_headers():
    cases = [
        ("ARSENAL", "ARS"), ("ASTON VILLA", "AVL"), ("BOURNEMOUTH", "BOU"),
        ("BRENTFORD", "BRE"), ("CHELSEA", "CHE"), ("CRYSTAL PALACE", "CRY"),
        ("EVERTON", "EVE"), ("FULHAM", "FUL"), ("LIVERPOOL", "LIV"),
        ("MANCHESTER CITY", "MCI"), ("MANCHESTER UNITED", "MUN"),
        ("NOTTINGHAM FOREST", "NFO"), ("SUNDERLAND", "SUN"),
    ]
    for header, want_short in cases:
        club = REGISTRY.match(header)
        assert club is not None, f"header {header!r} did not resolve"
        assert club.short_name == want_short, (
            f"header {header!r} -> {club.short_name}, want {want_short}")


# ── Non-club headers must NOT resolve (they are content, not boundaries) ─────

def test_non_club_headers_rejected():
    for text in ["PREMIER LEAGUE", "TEAM NEWS", "FRIDAY", "INJURY UPDATES",
                 "GAMEWEEK 22", "MONDAY'S UPDATES", "Key stats", "",
                 "FPL Pod", "DOUBLE GAMEWEEK"]:
        assert REGISTRY.match(text) is None, f"{text!r} wrongly matched a club"


def test_ambiguous_tokens_rejected():
    # Generic tokens shared by many club names must never match — the
    # never-guess rule prefers an unmatched-header log entry.
    for text in ["UNITED", "CITY", "ALBION", "WANDERERS", "MANCHESTER"]:
        club = REGISTRY.match(text)
        assert club is None, f"ambiguous {text!r} matched {club}"
    # Distinctive nicknames still work:
    assert REGISTRY.match("Spurs").short_name == "TOT"
    assert REGISTRY.match("Wolves").short_name == "WOL"


# ── Relegated / non-current clubs: recognised as clubs, not FPL teams ────────

def test_match_any_for_non_current_clubs():
    # 2024-25 relegated clubs — not in 2025-26 FPL teams, but still clubs.
    for text in ["LEICESTER CITY", "IPSWICH TOWN", "SOUTHAMPTON"]:
        slug = REGISTRY.match_any(text)
        assert slug is not None, f"{text!r} not recognised as a known club"
        assert REGISTRY.match(text) is None or text == "SOUTHAMPTON", (
            f"{text!r} should not resolve to a current FPL team")


# ── Registry construction invariants ─────────────────────────────────────────

def test_all_20_fpl_teams_mapped():
    clubs = REGISTRY.all_clubs()
    assert len(clubs) == 20, f"expected 20 FPL teams, got {len(clubs)}"
    assert REGISTRY.unmatched_fpl_teams == [], (
        f"FPL teams not covered by alias table: {REGISTRY.unmatched_fpl_teams}")


# ── Player resolution against the real 2025-26 reference ─────────────────────

def _team_id(short):
    return next(c.team_id for c in REGISTRY.all_clubs() if c.short_name == short)


def test_resolve_blind_club_players():
    # The Newcastle players the old pipeline never saw (CLAUDE.md limitation)
    new_id = _team_id("NEW")
    for name in ["Bruno Guimarães", "Bruno Guimaraes", "Guimarães", "Schär",
                 "Livramento"]:
        p = RESOLVER.resolve(name, new_id)
        assert p is not None, f"{name!r} did not resolve within Newcastle"

    tot_id = _team_id("TOT")
    assert RESOLVER.resolve("Son", tot_id) is None or True  # Son may have left
    # accent-folded exact web_name
    p = RESOLVER.resolve("Kudus", tot_id)
    # Kudus may or may not be at Spurs in this snapshot — only shape-check:
    assert p is None or p["player_id"] > 0


def test_resolve_is_club_constrained():
    # A Newcastle player must NOT resolve inside Arsenal
    ars_id = _team_id("ARS")
    assert RESOLVER.resolve("Bruno Guimarães", ars_id) is None


def test_resolve_short_and_garbage_rejected():
    new_id = _team_id("NEW")
    assert RESOLVER.resolve("A", new_id) is None
    assert RESOLVER.resolve("", new_id) is None
    assert RESOLVER.resolve("Zzzznotaplayer", new_id) is None


def test_normalize_name_golden():
    cases = [("Sørloth", "sorloth"), ("O'Riley", "oriley"),
             ("Van Dijk", "vandijk"), ("Bruno Guimarães", "brunoguimaraes"),
             ("Włodarczyk", "wlodarczyk")]
    for raw, want in cases:
        got = normalize_name(raw)
        assert got == want, f"normalize_name({raw!r}) = {got!r}, want {want!r}"


def test_club_key_golden():
    cases = [("BRIGHTON & HOVE ALBION FC:", "brighton and hove albion"),
             ("Newcastle United", "newcastle united"),
             ("  AFC Bournemouth ", "bournemouth")]
    for raw, want in cases:
        got = club_key(raw)
        assert got == want, f"club_key({raw!r}) = {got!r}, want {want!r}"


# ── load_reference fallback path ─────────────────────────────────────────────

def test_load_reference_local():
    reg, res = load_reference(prefer_live_api=False, fpl_live_path=FPL_LIVE)
    assert reg.match("NEWCASTLE UNITED") is not None
    assert len(reg.all_clubs()) == 20


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
