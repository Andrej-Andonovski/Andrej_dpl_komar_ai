"""
intel_02_press_conferences.py
Scrape Fantasy Football Scout Friday team news articles for GW1-10 (2025/26).
Reads:  data/intel/fpl_live.json   (player name reference)
Writes: data/intel/press_conferences.json

FFS article structure (confirmed via debug):
  <article class="post-...">
    <div class="article-holder">
      ...
      <div>  ← parent of wp-block-heading h2s
        <h2 class="wp-block-heading">ARSENAL</h2>   ← club headers (ALL CAPS)
        <p><strong>Player Name</strong> is fit...</p>
        ...
      </div>
    </div>
  </article>
"""

import sys
import json
import time
import re
import os
from datetime import datetime, timezone
from collections import Counter

import requests
from bs4 import BeautifulSoup

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COOKIE_NAME  = "wordpress_logged_in_19ae9e06500d60c2ba447787db8e3a35"
COOKIE_VALUE = (
    "somalianboi%7C1774611704%7CO0d1UqPDPbvRu7oL0c35669JyZmmQibPVzhLcwAscHo"
    "%7Cd18ba6e19f361b929bfb60aa0014a3f86568ef03f15c5f3f34159a948ef20b83"
)
COOKIE_DOMAIN = ".fantasyfootballscout.co.uk"
BASE_HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.fantasyfootballscout.co.uk/",
}

SLEEP_BETWEEN = 12  # seconds between every HTTP request

ROOT_DIR  = os.path.join(os.path.dirname(__file__), "..")
INTEL_DIR = os.path.join(ROOT_DIR, "data", "intel")
OUT_PATH  = os.path.join(INTEL_DIR, "press_conferences.json")
LIVE_PATH = os.path.join(INTEL_DIR, "fpl_live.json")

# Known URLs (confirmed). GWs not listed here are discovered via search.
KNOWN_URLS = {
    1:  "https://www.fantasyfootballscout.co.uk/2025/08/15/fpl-gameweek-1-team-news-fridays-live-injury-updates",
    5:  "https://www.fantasyfootballscout.co.uk/2025/09/19/fpl-gameweek-5-team-news-fridays-live-injury-updates",
    8:  "https://www.fantasyfootballscout.co.uk/2025/10/17/fpl-gameweek-8-team-news-fridays-live-injury-updates-2",
    9:  "https://www.fantasyfootballscout.co.uk/2025/10/24/fpl-gameweek-9-team-news-fridays-live-injury-updates-2",
    10: "https://www.fantasyfootballscout.co.uk/2025/10/31/fpl-gameweek-10-team-news-fridays-live-injury-updates-2",
    11: "https://www.fantasyfootballscout.co.uk/2025/11/07/fpl-gameweek-11-team-news-fridays-live-injury-updates-3",
    12: "https://www.fantasyfootballscout.co.uk/2025/11/21/fpl-gameweek-12-team-news-fridays-live-injury-updates-3",
    13: "https://www.fantasyfootballscout.co.uk/2025/11/28/fpl-gameweek-13-team-news-fridays-live-injury-updates-2",
    14: "https://www.fantasyfootballscout.co.uk/2025/12/02/fpl-gameweek-14-team-news-tuesdays-live-injury-updates-2",
    15: "https://www.fantasyfootballscout.co.uk/2025/12/05/fpl-gameweek-15-team-news-fridays-live-injury-updates",
    16: "https://www.fantasyfootballscout.co.uk/2025/12/12/fpl-gameweek-16-team-news-fridays-live-injury-updates-2",
    17: "https://www.fantasyfootballscout.co.uk/2025/12/19/fpl-gameweek-17-team-news-fridays-live-injury-updates",
    18: "https://www.fantasyfootballscout.co.uk/2025/12/26/fpl-gameweek-18-team-news-thursday-updates-bruno-latest",
    19: "https://www.fantasyfootballscout.co.uk/2025/12/30/fpl-gameweek-19-team-news-tuesdays-live-injury-updates",
    20: "https://www.fantasyfootballscout.co.uk/2026/01/02/fpl-gameweek-20-team-news-fridays-live-injury-update",
    21: "https://www.fantasyfootballscout.co.uk/2026/01/07/fpl-gameweek-21-team-news-weds-live-injury-updates-ekitike-latest",
    22: "https://www.fantasyfootballscout.co.uk/2026/01/16/fpl-gameweek-22-team-news-fridays-live-injury-updates-2",
    23: "https://www.fantasyfootballscout.co.uk/2026/01/23/fpl-gameweek-23-team-news-fridays-live-injury-updates-3",
    24: "https://www.fantasyfootballscout.co.uk/2026/01/30/fpl-gameweek-24-team-news-fridays-live-injury-updates-3",
    25: "https://www.fantasyfootballscout.co.uk/2026/02/06/fpl-gameweek-25-team-news-fridays-live-injury-updates-3",
    26: "https://www.fantasyfootballscout.co.uk/2026/02/10/fpl-gameweek-26-team-news-tuesdays-live-injury-updates",
    27: "https://www.fantasyfootballscout.co.uk/2026/02/20/fpl-gameweek-27-team-news-fridays-live-injury-updates-2",
    28: "https://www.fantasyfootballscout.co.uk/2026/02/27/fpl-gameweek-28-team-news-fridays-live-injury-updates-3",
}

PL_CLUBS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leeds", "Leicester", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle", "Nottingham Forest", "Southampton", "Sunderland",
    "Tottenham", "West Ham", "Wolves",
]

# ---------------------------------------------------------------------------
# Availability classification
# ---------------------------------------------------------------------------

OUT_PHRASES       = [r"will miss", r"ruled out", r"definitely out", r"won.t be available",
                     r"out for", r"\bout\b"]
DOUBTFUL_PHRASES  = [r"doubt", r"50/50", r"to be assessed", r"late call", r"concern", r"niggle",
                     r"assessed", r"check", r"fitness test"]
AVAILABLE_PHRASES = [r"\bfit\b", r"available", r"\bback\b", r"in the squad", r"returns",
                     r"return to", r"ready", r"could feature", r"expected to play",
                     r"missing out"]
SUSPENDED_PHRASES = [r"suspended", r"\bban\b", r"banned", r"serving a"]


def classify(text: str) -> str:
    t = text.lower()
    # Pre-check: phrases where "out" appears but the player is actually available
    # Must run before OUT_PHRASES to prevent \bout\b false-positive matches
    if re.search(r"missing out", t):
        return "available"
    for p in SUSPENDED_PHRASES:
        if re.search(p, t):
            return "suspended"
    for p in OUT_PHRASES:
        if re.search(p, t):
            return "out"
    for p in DOUBTFUL_PHRASES:
        if re.search(p, t):
            return "doubtful"
    for p in AVAILABLE_PHRASES:
        if re.search(p, t):
            return "available"
    return "unknown"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.cookies.set(COOKIE_NAME, COOKIE_VALUE, domain=COOKIE_DOMAIN)
    s.headers.update(BASE_HEADERS)
    return s


def safe_get(session: requests.Session, url: str, label: str = "") -> requests.Response | None:
    try:
        r = session.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        print(f"  [ERROR] {label}: {e}")
        return None


# ---------------------------------------------------------------------------
# URL discovery via search
# ---------------------------------------------------------------------------

def discover_url(session: requests.Session, gw: int) -> str | None:
    search_url = f"https://www.fantasyfootballscout.co.uk/?s=gameweek+{gw}+team+news+friday"
    print(f"  Discovering URL via search...", end=" ", flush=True)
    time.sleep(SLEEP_BETWEEN)
    r = safe_get(session, search_url, f"search GW{gw}")
    if r is None:
        print("FAILED")
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    pattern = re.compile(rf"gameweek-{gw}[^\"']*team-news[^\"']*friday", re.IGNORECASE)
    for a in soup.find_all("a", href=True):
        if pattern.search(a["href"]):
            print("found")
            return a["href"].split("?")[0].rstrip("/")
    print("not found")
    return None


# ---------------------------------------------------------------------------
# Article parsing
# ---------------------------------------------------------------------------

def normalize_club(text: str) -> str | None:
    """Match text to a PL club name, tolerating ALL CAPS, trailing colon, plurals."""
    text = text.strip().rstrip(":").strip()
    for club in PL_CLUBS:
        if text.upper() == club.upper():
            return club
        # e.g. "Wolves" vs "Wolf" — strip trailing S
        if text.upper().rstrip("S") == club.upper().rstrip("S"):
            return club
    return None


def extract_injury_type(text: str) -> str:
    """Pull injury label from parentheses: 'Player (hamstring) will...' -> 'hamstring'"""
    m = re.search(r"\(([^)]{4,40})\)", text)
    if m:
        return m.group(1).strip().lower()
    return ""


def find_article_body(soup: BeautifulSoup):
    """
    Locate the prose container holding club sections.

    Confirmed FFS structure (no div.entry-content):
      <article> → <div.article-holder> → ... → parent div of h2.wp-block-heading

    Strategy: find the first h2 with class "wp-block-heading" whose text matches
    a PL club name, then return its parent element (the content container).
    Fall back to broader searches if needed.
    """
    # Primary: find a wp-block-heading h2 that IS a club header, return its parent
    for h2 in soup.find_all("h2", class_="wp-block-heading"):
        if normalize_club(h2.get_text(" ", strip=True)):
            parent = h2.parent
            if parent and parent.name in ("div", "section", "article"):
                return parent

    # Secondary: standard WP entry-content div
    body = soup.find("div", class_="entry-content")
    if body:
        return body

    # Tertiary: full article tag
    return soup.find("article")


def _is_club_header(elem) -> str | None:
    """
    Return the normalised club name if this element is a club section header,
    else None.

    Handles:
      <h2 class="wp-block-heading">ARSENAL</h2>
      <p><strong>Arsenal:</strong></p>   (strong-only paragraph with colon)
    """
    if not hasattr(elem, "name"):
        return None

    # h2/h3/h4 with club name text
    if elem.name in ("h2", "h3", "h4"):
        return normalize_club(elem.get_text(" ", strip=True))

    # <p> whose only non-empty child is a <strong> tag matching a club name
    if elem.name == "p":
        non_empty = [c for c in elem.children
                     if not (isinstance(c, str) and c.strip() == "")]
        if (len(non_empty) == 1
                and hasattr(non_empty[0], "name")
                and non_empty[0].name == "strong"):
            return normalize_club(non_empty[0].get_text(" ", strip=True))

    return None


def _looks_like_player(name: str) -> bool:
    """Heuristic: does this <strong> text look like a player name?"""
    name = name.strip()
    if len(name) < 3:
        return False
    words = name.split()
    if not (1 <= len(words) <= 5):
        return False
    # Every word must start with uppercase
    if not all(w[0].isupper() for w in words):
        return False
    # All-caps single token = club/section header
    if len(words) == 1 and name.isupper() and len(name) > 3:
        return False
    # Reject known non-player strings
    reject = {
        "Manager", "Head", "Coach", "Premier", "League", "Friday",
        "Saturday", "Sunday", "Team", "News", "Match", "Game", "Week",
        "Full", "Half", "First", "Second", "GW",
        # Club names (single-word ones most likely to appear in bold)
        "Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham",
        "Brentford", "Bournemouth", "Brighton", "Newcastle", "Southampton",
        "Tottenham", "Wolves", "Leicester", "Ipswich", "Leeds", "Sunderland",
        "Crystal", "Palace", "Manchester", "Nottingham", "Forest",
        "Villa", "West", "Ham", "City", "United",
    }
    if name in reject or words[0] in reject:
        return False
    return True


def extract_players_from_elements(elements: list) -> list[dict]:
    """Extract player entries from a list of soup elements (one club section)."""
    players = []
    seen: set[str] = set()

    for elem in elements:
        if not hasattr(elem, "find_all"):
            continue
        for strong in elem.find_all("strong"):
            name = strong.get_text(" ", strip=True)
            if not _looks_like_player(name) or name in seen:
                continue
            seen.add(name)

            # Use containing <p> for sentence context
            parent_p = strong.find_parent("p")
            sentence = (parent_p or elem).get_text(" ", strip=True)

            players.append({
                "player": name,
                "injury": extract_injury_type(sentence),
                "news": sentence,
                "availability": classify(sentence),
            })

    return players


def parse_article(html: str) -> dict:
    """
    Parse an FFS team news article into per-club player availability data.

    Returns:
      {
        "clubs": {club_name: {"raw_text": str, "players": [...]}},
        "all_player_news": [{"player", "club", "injury", "news", "availability"}]
      }
    """
    soup = BeautifulSoup(html, "html.parser")
    body = find_article_body(soup)
    if body is None:
        return {"clubs": {}, "all_player_news": []}

    # Walk direct children of the content container
    blocks = [elem for elem in body.children
              if hasattr(elem, "name")
              and elem.name in ("h2", "h3", "h4", "p", "ul", "ol", "div")]

    clubs: dict[str, dict] = {}
    current_club: str | None = None
    club_elems: list = []

    def flush():
        nonlocal current_club, club_elems
        if current_club is None:
            return
        raw = "\n".join(e.get_text(" ", strip=True) for e in club_elems if e.get_text(strip=True))
        players = extract_players_from_elements(club_elems)
        clubs[current_club] = {"raw_text": raw, "players": players}
        current_club = None
        club_elems = []

    for elem in blocks:
        club = _is_club_header(elem)
        if club:
            flush()
            current_club = club
        elif current_club is not None:
            club_elems.append(elem)

    flush()

    all_news = [
        {"player": p["player"], "club": club,
         "injury": p["injury"], "news": p["news"],
         "availability": p["availability"]}
        for club, data in clubs.items()
        for p in data["players"]
    ]

    return {"clubs": clubs, "all_player_news": all_news}


# ---------------------------------------------------------------------------
# GW scraper
# ---------------------------------------------------------------------------

def scrape_gw(session: requests.Session, gw: int, url: str) -> dict:
    """Fetch and parse one GW article. Returns the GW result dict."""
    print(f"  Waiting {SLEEP_BETWEEN}s...", end=" ", flush=True)
    time.sleep(SLEEP_BETWEEN)
    print("fetching...", end=" ", flush=True)

    t0 = time.time()
    r = safe_get(session, url, f"GW{gw}")
    elapsed = time.time() - t0

    if r is None:
        print(f"FAILED ({elapsed:.1f}s)")
        return {
            "gw": gw, "url": url, "status": "fetch_failed",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "clubs": {}, "all_player_news": [],
        }

    plain = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
    print(f"fetched in {elapsed:.1f}s | {len(plain):,} chars")

    if len(plain) < 300:
        print("  [WARN] Content < 300 chars — possibly paywalled")
        return {
            "gw": gw, "url": url, "status": "fetch_failed",
            "note": "possibly_paywalled",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "clubs": {}, "all_player_news": [],
        }

    return _build_result(gw, url, r.text)


def _build_result(gw: int, url: str, html: str) -> dict:
    """Parse HTML and build the result dict, printing a summary line."""
    parsed   = parse_article(html)
    clubs    = parsed["clubs"]
    all_news = parsed["all_player_news"]

    out_p = [e["player"] for e in all_news if e["availability"] == "out"]
    dbt_p = [e["player"] for e in all_news if e["availability"] == "doubtful"]

    def fmt_list(names: list[str]) -> str:
        return ", ".join(names[:5]) + (f" +{len(names)-5} more" if len(names) > 5 else "")

    print(f"  [+] {len(clubs)} clubs | {len(all_news)} player mentions")
    if out_p:
        print(f"    OUT:      {fmt_list(out_p)}")
    if dbt_p:
        print(f"    DOUBTFUL: {fmt_list(dbt_p)}")

    return {
        "gw": gw,
        "url": url,
        "status": "success",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "clubs": {c: {"raw_text": d["raw_text"], "players": d["players"]}
                  for c, d in clubs.items()},
        "all_player_news": all_news,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GW_START = 1
GW_END   = 28


def main():
    os.makedirs(INTEL_DIR, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   FPL PRESS CONFERENCE INTEL -- GW1 to GW28         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Load existing data (merge, don't overwrite) ──────────────────────────
    existing: dict[str, dict] = {}
    already_scraped: set[int] = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        existing = saved.get("gameweeks", {})
        for k, v in existing.items():
            if v.get("status") == "success":
                already_scraped.add(int(k))
        print(f"Found existing data: GWs already scraped = {sorted(already_scraped)}")
        print()

    gws_to_scrape = [g for g in range(GW_START, GW_END + 1) if g not in already_scraped]
    if not gws_to_scrape:
        print("All GWs already scraped. Nothing to do.")
        return

    print(f"GWs to scrape: {gws_to_scrape}")
    print()

    session = make_session()

    # ── Verify login ─────────────────────────────────────────────────────────
    # Use any known URL for the login check (pick first needed GW or GW10 as fallback)
    verify_gw = gws_to_scrape[0] if gws_to_scrape[0] in KNOWN_URLS else 10
    verify_url = KNOWN_URLS.get(verify_gw, KNOWN_URLS[10])

    print(f"Verifying FFS login (GW{verify_gw})...", end=" ", flush=True)
    r_verify = safe_get(session, verify_url, "login-verify")
    if r_verify is None:
        print("FAILED — could not reach FFS. Check internet.")
        sys.exit(1)
    verify_plain = BeautifulSoup(r_verify.text, "html.parser").get_text(" ", strip=True)
    if len(verify_plain) < 500:
        print(f"FAILED ({len(verify_plain)} chars — check cookie)")
        sys.exit(1)
    print("OK")
    print()

    # Start with existing data; we'll fill in newly scraped GWs
    gameweeks_out: dict[str, dict] = dict(existing)
    gws_scraped: list[int] = list(already_scraped)
    gws_failed:  list[int] = []

    for gw in gws_to_scrape:
        print(f"── GW{gw} {'─' * 47}")

        # Resolve URL
        if gw in KNOWN_URLS:
            url = KNOWN_URLS[gw]
        else:
            url = discover_url(session, gw)
            if url is None:
                print(f"  [SKIP] GW{gw}: URL not found")
                gameweeks_out[str(gw)] = {
                    "gw": gw, "url": None, "status": "url_not_found",
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "clubs": {}, "all_player_news": [],
                }
                gws_failed.append(gw)
                print()
                continue

        # Reuse verification fetch if the URL matches
        if gw == verify_gw and url == verify_url:
            print(f"  (reusing verification fetch | {len(verify_plain):,} chars)")
            result = _build_result(gw, url, r_verify.text)
        else:
            result = scrape_gw(session, gw, url)

        gameweeks_out[str(gw)] = result
        if result["status"] == "success":
            gws_scraped.append(gw)
        else:
            gws_failed.append(gw)
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    total_mentions = sum(len(v.get("all_player_news", [])) for v in gameweeks_out.values())
    injury_counter: Counter = Counter()
    for gw_data in gameweeks_out.values():
        for entry in gw_data.get("all_player_news", []):
            inj = entry.get("injury", "").strip().lower()
            if inj:
                injury_counter[inj] += 1

    print()
    print("SUMMARY")
    print("=" * 55)
    print(f"GWs in output:           {sorted(int(k) for k in gameweeks_out)}")
    print(f"GWs scraped this run:    {[g for g in gws_to_scrape if g not in gws_failed]}")
    print(f"GWs failed this run:     {gws_failed if gws_failed else 'none'}")
    print(f"Total player mentions:   {total_mentions}")
    if injury_counter:
        print()
        print("Most common injuries across all GWs:")
        print("  " + " | ".join(f"{k}: {v}" for k, v in injury_counter.most_common(8)))
    print()

    output = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "gws_scraped": sorted(gws_scraped),
        "gws_failed":  gws_failed,
        "total_player_mentions": total_mentions,
        "gameweeks": gameweeks_out,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
