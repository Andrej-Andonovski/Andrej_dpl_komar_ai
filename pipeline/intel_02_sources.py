"""
pipeline/intel_02_sources.py
Source adapters for the multi-source scraper (redesign steps 2-3).

Tier 1 (stable URL or official API — no discovery risk, step 2):
  FfsInjuriesAdapter      fantasyfootballscout.co.uk/fantasy-football-injuries/
  SportsGamblerAdapter    sportsgambler.com/injuries/football/england-premier-league/
  KnocksAndBansAdapter    knocksandbans.com
  GuardianAdapter         content.guardianapis.com (series: match-previews)
Tier 2 (step 3, §4.1):
  FfsTeamNewsAdapter      FFS "FPL Gameweek N team news" liveblog, discovered
                          via the public WordPress REST API (category 3 +
                          deadline window) — replaces KNOWN_URLS outright
Tier 3/4 (step 4, §4.5/§4.7):
  SkySportsAdapter        skysports.com Google news sitemap (per-tick breaking
                          news) + per-club index pages (T-24h escalation)
  ReachLocalAdapter       Reach plc local outlets' per-club `?service=rss`
                          feeds → JSON-LD articleBody (T-24h escalation only)

The Tier 3/4 sources are story-per-article (one article ≈ one club/player
event), unlike the section-per-club Tier 1/2 sources. They share a story
extractor (§5.3 two-stage): club attribution from the headline via the
identity registry (never-guess — a title naming two clubs is ambiguous and
skipped), a roster-grounded regex scan (stage A), then optional LLM
gap-filling (stage B) for club sections the regex left thin.

Common shape (§2 "adapters are isolated"):
  adapter.fetch(session) -> raw            (str HTML or dict JSON)
  adapter.extract(raw, gw, registry, resolver,
                  url=..., observed_at=..., published_at=...)
      -> (claims, stats)
fetch and extract are separate so archived content (Wayback snapshots,
Guardian date-window queries) flows through the exact same extraction path
used live. Adapters never raise on content problems — they return what they
could parse plus stats for coverage accounting (§7); a 0-claim result from a
200 response is the caller's structure-drift alarm.
"""

import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape

from bs4 import BeautifulSoup

from intel_02_ledger import STATUSES, make_claim, utcnow_iso

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/126.0.0.0 Safari/537.36")}
TIMEOUT = 30


# ---------------------------------------------------------------------------
# date helpers
# ---------------------------------------------------------------------------

def _iso_from_ddmmyyyy(s: str) -> str | None:
    """'24/05/2026' or '24/05/26' -> ISO date (midnight UTC)."""
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s or "")
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    try:
        return datetime(y, mo, d, tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def _iso_from_yyyymmdd(s: str) -> str | None:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s or "")
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}


def _iso_from_daymonth(s: str, ref_iso: str) -> str | None:
    """
    '12 May 18:04' (no year) -> ISO, inferring the year from ref_iso:
    same year as ref, unless that lands in the future of ref, then year-1.
    """
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{1,2}):(\d{2})", s or "")
    if not m:
        return None
    mo = next((v for k, v in _MONTHS.items()
               if k.startswith(m.group(2).lower()[:3])), None)
    if not mo:
        return None
    ref = datetime.fromisoformat(ref_iso.replace("Z", "+00:00"))
    try:
        dt = datetime(ref.year, mo, int(m.group(1)), int(m.group(3)),
                      int(m.group(4)), tzinfo=timezone.utc)
    except ValueError:
        return None
    if dt > ref:
        dt = dt.replace(year=ref.year - 1)
    return dt.isoformat()


def _iso_from_long_date(s: str) -> str | None:
    """'7 July 2026, 03:18' -> ISO."""
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})(?:,\s*(\d{1,2}):(\d{2}))?",
                  s or "")
    if not m:
        return None
    mo = _MONTHS.get(m.group(2).lower())
    if not mo:
        return None
    hh = int(m.group(4) or 0)
    mm = int(m.group(5) or 0)
    try:
        return datetime(int(m.group(3)), mo, int(m.group(1)), hh, mm,
                        tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def _new_stats() -> dict:
    return {"rows_seen": 0, "clubs_seen": set(), "unmatched_clubs": set(),
            "unresolved_players": []}


def _finalize(claims: list, stats: dict) -> tuple:
    stats["clubs_seen"] = sorted(stats["clubs_seen"])
    stats["unmatched_clubs"] = sorted(stats["unmatched_clubs"])
    stats["n_claims"] = len(claims)
    return claims, stats


# ---------------------------------------------------------------------------
# 1. FFS injuries & bans table (Tier 1, §4.2)
# ---------------------------------------------------------------------------

class FfsInjuriesAdapter:
    NAME = "ffs_injuries"
    TIER = 1
    URL = "https://www.fantasyfootballscout.co.uk/fantasy-football-injuries/"

    def fetch(self, session):
        r = session.get(self.URL, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text

    @staticmethod
    def _status(text: str) -> str:
        t = (text or "").strip().lower()
        if "doubt" in t or "%" in t:
            return "doubtful"
        if "suspen" in t or "ban" in t:
            return "suspended"
        if "injur" in t or t == "out":
            return "out"
        if "fit" in t or "available" in t:
            return "available"
        return "unknown"

    def extract(self, raw, gw, registry, resolver, *, url=None,
                observed_at=None, published_at=None):
        url = url or self.URL
        observed_at = observed_at or utcnow_iso()
        soup = BeautifulSoup(raw, "html.parser")
        claims, stats = [], _new_stats()

        for row in soup.select("tr.injuries-bans-item"):
            stats["rows_seen"] += 1
            code = (row.get("data-team-code") or "").strip()
            club = registry.match(code) if code else None
            if club is None:
                # fall back to the club cell text if the code is missing/new
                tds = row.find_all("td")
                if len(tds) > 1:
                    club = registry.match(tds[1].get_text(" ", strip=True))
            if club is None:
                stats["unmatched_clubs"].add(code or "?")
                continue
            stats["clubs_seen"].add(club.short_name)

            tds = row.find_all("td")
            if len(tds) < 5:
                continue
            player_raw = tds[0].get_text(" ", strip=True)
            # FFS mixes "Jurriën Timber" with "White (Ben)" (surname-first):
            # reorder the parenthesised given name for resolution.
            m = re.match(r"^(?P<sur>[^()]+?)\s*\((?P<first>[^()]+)\)$",
                         player_raw)
            if m:
                player_raw = f"{m.group('first').strip()} {m.group('sur').strip()}"
            status_txt = tds[2].get_text(" ", strip=True)
            return_dt = _iso_from_ddmmyyyy(tds[3].get_text(" ", strip=True))
            news = tds[4].get_text(" ", strip=True)
            upd = (tds[5].get_text(" ", strip=True) if len(tds) > 5 else "")
            row_pub = (_iso_from_ddmmyyyy(upd) or published_at)

            p = resolver.resolve(player_raw, club.team_id)
            if p is None:
                stats["unresolved_players"].append(
                    f"{player_raw} ({club.short_name})")

            claims.append(make_claim(
                gw, self.NAME, self.TIER, url, club.team_id, player_raw,
                self._status(status_txt),
                player_id=p["player_id"] if p else None,
                injury="", text=f"[{status_txt}] {news}".strip(),
                observed_at=observed_at, published_at=row_pub,
                return_date=return_dt))
        return _finalize(claims, stats)


# ---------------------------------------------------------------------------
# 2. SportsGambler injuries page (Tier 1, §4.8)
# ---------------------------------------------------------------------------

class SportsGamblerAdapter:
    NAME = "sportsgambler"
    TIER = 1
    URL = ("https://www.sportsgambler.com/injuries/football/"
           "england-premier-league/")

    def fetch(self, session):
        r = session.get(self.URL, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text

    @staticmethod
    def _status(row) -> str:
        """Severity is encoded in the inj-type span's class attribute."""
        span = row.select_one("[class*=injury-]")
        classes = " ".join(span.get("class", [])) if span else ""
        if "questionmark" in classes:
            return "doubtful"
        if "suspend" in classes or "card" in classes or "ban" in classes:
            return "suspended"
        if "plus" in classes or "cross" in classes:
            return "out"
        return "unknown"

    def extract(self, raw, gw, registry, resolver, *, url=None,
                observed_at=None, published_at=None):
        url = url or self.URL
        observed_at = observed_at or utcnow_iso()
        soup = BeautifulSoup(raw, "html.parser")
        claims, stats = [], _new_stats()

        if published_at is None:
            m = re.search(r"Last Updated:\s*([^<]{4,40})", raw)
            published_at = _iso_from_long_date(m.group(1)) if m else None

        current_club = None
        for el in soup.find_all(["h3", "div"]):
            if el.name == "h3" and "injuries-title" in (el.get("class") or []):
                slug = (el.get("id") or "").replace("-", " ")
                current_club = (registry.match(slug)
                                or registry.match(el.get_text(" ", strip=True)))
                if current_club is None:
                    stats["unmatched_clubs"].add(el.get("id") or "?")
                else:
                    stats["clubs_seen"].add(current_club.short_name)
                continue
            if current_club is None or "inj-row" not in (el.get("class") or []):
                continue
            player_el = el.select_one(".inj-player")
            if player_el is None:
                continue
            stats["rows_seen"] += 1
            player_raw = player_el.get_text(" ", strip=True)
            info = el.select_one(".inj-info")
            ret = el.select_one(".inj-return")
            injury = info.get_text(" ", strip=True) if info else ""
            ret_txt = ret.get_text(" ", strip=True) if ret else ""
            return_dt = _iso_from_yyyymmdd(ret_txt)

            p = resolver.resolve(player_raw, current_club.team_id)
            if p is None:
                stats["unresolved_players"].append(
                    f"{player_raw} ({current_club.short_name})")

            claims.append(make_claim(
                gw, self.NAME, self.TIER, url, current_club.team_id,
                player_raw, self._status(el),
                player_id=p["player_id"] if p else None,
                injury=injury.lower(), text=f"{injury} (return: {ret_txt})",
                observed_at=observed_at, published_at=published_at,
                return_date=return_dt))
        return _finalize(claims, stats)


# ---------------------------------------------------------------------------
# 3. KnocksAndBans (Tier 1, §4.8)
# ---------------------------------------------------------------------------

class KnocksAndBansAdapter:
    NAME = "knocksandbans"
    TIER = 1
    URL = "https://www.knocksandbans.com/"

    # Observed formats for the trailing update stamp:
    #   "... Est. Return 29/05/26 Last update: 12/05/26"   (older layout)
    #   "... Est. Return 29/05/26 12 May 18:04"            (current layout,
    #                                                       no year)
    _ENTRY = re.compile(
        r"(?P<name>[A-ZÀ-Þ][\w'’.\-]+(?:\s+[A-ZÀ-Þa-zà-þ][\w'’.\-]*){0,4})\s*"
        r"[-–]\s*(?P<injury>[^:]{2,40}?)\s*Status:\s*(?P<status>OUT|\d{1,3}%)\s*"
        r"(?:Est\.?\s*Return\s*(?P<ret>[\d/]+|Unknown|TBC))?\s*"
        r"(?:Last update:\s*(?P<upd>[\d/]+)"
        r"|(?P<upd2>\d{1,2}\s+[A-Za-z]{3,9}\s+\d{1,2}:\d{2}))?",
        re.UNICODE)

    def fetch(self, session):
        r = session.get(self.URL, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text

    @staticmethod
    def _status(s: str) -> str:
        s = s.strip().upper()
        if s == "OUT":
            return "out"
        if s.endswith("%"):
            return "doubtful"     # FPL-style 25/50/75 -> doubt (pct kept in text)
        return "unknown"

    def extract(self, raw, gw, registry, resolver, *, url=None,
                observed_at=None, published_at=None):
        url = url or self.URL
        observed_at = observed_at or utcnow_iso()
        soup = BeautifulSoup(raw, "html.parser")
        claims, stats = [], _new_stats()

        # h2 club headings partition the page; entries parsed from the text
        # between consecutive h2s (markup is Tailwind soup — the research
        # note says: key on headings and text, not classes).
        headings = soup.find_all("h2")
        for h in headings:
            club = registry.match(h.get_text(" ", strip=True))
            if club is None:
                continue
            stats["clubs_seen"].add(club.short_name)
            # collect text until the next CLUB h2 — some layouts put
            # non-club h2s ("7 Unavailable") between a club heading and
            # its entries, so only a recognised club ends the section
            chunks = []
            for sib in h.find_all_next():
                if sib.name == "h2":
                    if registry.match(sib.get_text(" ", strip=True)):
                        break
                    continue
                if sib.name in ("p", "div", "li", "span"):
                    chunks.append(sib.get_text(" ", strip=True))
            section = " ".join(chunks)

            seen_here = set()
            for m in self._ENTRY.finditer(section):
                player_raw = m.group("name").strip()
                if player_raw in seen_here:
                    continue
                seen_here.add(player_raw)
                stats["rows_seen"] += 1
                status_raw = m.group("status")
                p = resolver.resolve(player_raw, club.team_id)
                if p is None:
                    stats["unresolved_players"].append(
                        f"{player_raw} ({club.short_name})")
                row_pub = (_iso_from_ddmmyyyy(m.group("upd") or "")
                           or _iso_from_daymonth(m.group("upd2") or "",
                                                 observed_at)
                           or published_at)
                claims.append(make_claim(
                    gw, self.NAME, self.TIER, url, club.team_id, player_raw,
                    self._status(status_raw),
                    player_id=p["player_id"] if p else None,
                    injury=(m.group("injury") or "").strip().lower(),
                    text=f"Status: {status_raw}"
                         + (f" Est. return {m.group('ret')}" if m.group("ret") else ""),
                    observed_at=observed_at,
                    published_at=row_pub,
                    return_date=_iso_from_ddmmyyyy(m.group("ret") or "")))
        return _finalize(claims, stats)


# ---------------------------------------------------------------------------
# 4. Guardian Content API — weekly team news (Tier 1, §4.3)
# ---------------------------------------------------------------------------

class GuardianAdapter:
    NAME = "guardian"
    TIER = 1
    API = "https://content.guardianapis.com/search"
    SERIES_TAG = "football/series/match-previews"

    # "Injured Mitoma (hamstring, Jun), Webster (knee, Jun)" line prefixes
    _LINE_STATUS = {"doubtful": "doubtful", "doubt": "doubtful",
                    "injured": "out", "suspended": "suspended",
                    "unavailable": "out", "banned": "suspended",
                    "ineligible": "out"}
    _ITEM = re.compile(r"(?P<name>[^,()]+?)\s*(?:\((?P<detail>[^)]*)\))?\s*(?:,|$)")

    def __init__(self, api_key: str = "test"):
        self.api_key = api_key

    def fetch(self, session, *, from_date=None, to_date=None, page_size=5):
        """Return API JSON with article bodies for the date window."""
        params = {"tag": self.SERIES_TAG, "order-by": "newest",
                  "show-fields": "body", "page-size": page_size,
                  "api-key": self.api_key}
        if from_date:
            params["from-date"] = from_date
        if to_date:
            params["to-date"] = to_date
        r = session.get(self.API, params=params, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def extract(self, raw, gw, registry, resolver, *, url=None,
                observed_at=None, published_at=None):
        """raw: API JSON (search response) OR a single result item dict."""
        observed_at = observed_at or utcnow_iso()
        claims, stats = [], _new_stats()

        if isinstance(raw, dict) and "response" in raw:
            items = raw["response"].get("results", [])
        elif isinstance(raw, dict):
            items = [raw]
        else:
            items = []

        for item in items:
            body = (item.get("fields") or {}).get("body", "")
            art_url = item.get("webUrl") or url or ""
            art_pub = item.get("webPublicationDate") or published_at
            # An article's assertion time is its PUBLICATION time — fetching
            # it later (live tick or archive query) neither renews nor
            # expires it. Tables assert at fetch time; articles at pub time.
            c, s = self._extract_article(body, gw, registry, resolver,
                                         art_url, art_pub or observed_at,
                                         art_pub)
            claims.extend(c)
            stats["rows_seen"] += s["rows_seen"]
            stats["clubs_seen"].update(s["clubs_seen"])
            stats["unmatched_clubs"].update(s["unmatched_clubs"])
            stats["unresolved_players"].extend(s["unresolved_players"])
        return _finalize(claims, stats)

    def _extract_article(self, body_html, gw, registry, resolver,
                         url, observed_at, published_at):
        soup = BeautifulSoup(body_html, "html.parser")
        claims, stats = [], _new_stats()
        current_club = None

        for el in soup.find_all(["h2", "p"]):
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            if el.name == "h2":
                # club headers ("Brighton") match; fixture headers
                # ("BRIGHTON v MANCHESTER UNITED") don't and reset the club —
                # unmatched headers are boundaries, never content (§3.1)
                club = registry.match(text)
                current_club = club
                if club is not None:
                    stats["clubs_seen"].add(club.short_name)
                elif " v " in text.casefold():
                    pass                       # fixture header — expected
                else:
                    stats["unmatched_clubs"].add(text[:40])
                continue
            if current_club is None:
                continue

            first_word = text.split(" ", 1)[0].casefold()
            status = self._LINE_STATUS.get(first_word)
            if status is None:
                continue
            rest = text.split(" ", 1)[1] if " " in text else ""
            if rest.strip().casefold() in ("none", "none.", ""):
                continue                       # explicit empty state

            for m in self._ITEM.finditer(rest):
                name = m.group("name").strip(" .;")
                if not name or name.casefold() in ("none", "n/a"):
                    continue
                detail = (m.group("detail") or "").strip()
                injury, ret = detail, None
                if "," in detail:              # "(hamstring, Jun)"
                    injury, ret_txt = [x.strip() for x in detail.split(",", 1)]
                    ret = ret_txt or None
                stats["rows_seen"] += 1
                p = resolver.resolve(name, current_club.team_id)
                if p is None:
                    stats["unresolved_players"].append(
                        f"{name} ({current_club.short_name})")
                claims.append(make_claim(
                    gw, self.NAME, self.TIER, url, current_club.team_id,
                    name, status,
                    player_id=p["player_id"] if p else None,
                    injury=injury.lower(), text=text[:200],
                    observed_at=observed_at, published_at=published_at,
                    return_date=None if ret is None else ret))
        return claims, stats


# ---------------------------------------------------------------------------
# 5. FFS team-news liveblog (Tier 2, §4.1 — redesign step 3)
# ---------------------------------------------------------------------------

# Stage A regex classifier (§5.3) — ported unchanged from the old intel_02
# scraper: these phrase lists are proven over 28 GWs of liveblog copy.
_OUT_PHRASES       = [r"will miss", r"ruled out", r"definitely out",
                      r"won.t be available", r"out for", r"\bout\b"]
_DOUBTFUL_PHRASES  = [r"doubt", r"50/50", r"to be assessed", r"late call",
                      r"concern", r"niggle", r"assessed", r"check",
                      r"fitness test"]
_AVAILABLE_PHRASES = [r"\bfit\b", r"available", r"\bback\b", r"in the squad",
                      r"returns", r"return to", r"ready", r"could feature",
                      r"expected to play", r"missing out"]
_SUSPENDED_PHRASES = [r"suspended", r"\bban\b", r"banned", r"serving a"]


def classify_news(text: str) -> str:
    """Stage A availability classification of one liveblog sentence."""
    t = text.lower()
    # "missing out" contains \bout\b but the old scraper classifies it
    # available — pre-check kept so v2 matches proven behaviour exactly
    if re.search(r"missing out", t):
        return "available"
    for phrases, status in ((_SUSPENDED_PHRASES, "suspended"),
                            (_OUT_PHRASES, "out"),
                            (_DOUBTFUL_PHRASES, "doubtful"),
                            (_AVAILABLE_PHRASES, "available")):
        for p in phrases:
            if re.search(p, t):
                return status
    return "unknown"


def _wp_iso(s: str | None) -> str | None:
    """WP REST date_gmt/modified_gmt ('2026-01-16T13:00:00') -> tz-aware ISO."""
    if not s:
        return None
    return s if ("+" in s[10:] or s.endswith("Z")) else s + "+00:00"


class FfsTeamNewsAdapter:
    """
    FFS "FPL Gameweek N team news" liveblog (multi-edition per GW: Thu/Fri,
    plus Mon-Wed for midweek rounds — each edition is its own claim batch;
    reconciler recency handles supersession).

    Discovery via the public WP REST API (§4.1): one JSON request returns
    title + full body (content.rendered) — no slug construction, no member
    cookie, no HTML search page. Primary = category + deadline window;
    secondary = title-only search re-filtered by the title regex.
    """
    NAME = "ffs_teamnews"
    TIER = 2
    API = "https://www.fantasyfootballscout.co.uk/wp-json/wp/v2/posts"
    CATEGORY_TEAM_NEWS = 3          # slug "team-news"
    WINDOW_DAYS = 6                 # primary window: deadline-6d -> deadline
    SEARCH_BOUND_DAYS = 21          # secondary search season bound (§4.1:
                                    # LIKE-match needs an after= guard)
    _FIELDS = "id,title,slug,link,date_gmt,modified_gmt,content"

    # LLM stage B fires for a club section with fewer regex claims than this
    # and at least this much prose (short "no fresh news" stubs are skipped)
    LLM_MIN_CLAIMS = 2
    LLM_MIN_SECTION_CHARS = 200

    # Bold strings that pass the shape check but are never player names.
    # Club names are NOT listed — the registry rejects those alias-based.
    _REJECT_WORDS = {"Manager", "Head", "Coach", "Premier", "League",
                     "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday", "Team", "News", "Match", "Game",
                     "Week", "Gameweek", "Full", "Half", "First", "Second",
                     "GW", "FPL", "Deadline", "Update", "Updates", "Key"}

    @staticmethod
    def _title_re(gw: int) -> re.Pattern:
        # "FPL Gameweek 22 team news: Friday's live injury updates".
        # \b after the number stops "gameweek 2" matching "gameweek 22".
        return re.compile(rf"^\s*FPL\s+Gameweek\s+{gw}\b\s+team\s+news",
                          re.IGNORECASE)

    # -- discovery + fetch (one request, §4.1) --------------------------------

    def fetch(self, session, gw: int, *, after: str | None = None,
              before: str | None = None, per_page: int = 50) -> list:
        """
        Return this GW's liveblog edition post dicts (full body included).
        after/before: ISO datetimes bounding the deadline window; defaults
        to the trailing WINDOW_DAYS ending now (live tick before a deadline).
        """
        now = datetime.now(timezone.utc)
        if before is None:
            before = now.strftime("%Y-%m-%dT%H:%M:%S")
        if after is None:
            after = ((_parse_iso(before) - timedelta(days=self.WINDOW_DAYS))
                     .strftime("%Y-%m-%dT%H:%M:%S"))
        rex = self._title_re(gw)

        def matched(posts):
            return [p for p in posts
                    if rex.search(unescape((p.get("title") or {})
                                           .get("rendered", "")))]

        r = session.get(self.API,
                        params={"categories": self.CATEGORY_TEAM_NEWS,
                                "after": after[:19], "before": before[:19],
                                "per_page": per_page, "_fields": self._FIELDS},
                        headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        posts = matched(r.json())
        if posts:
            return posts

        # secondary: title search, wider window, same strict re-filter
        season_bound = ((_parse_iso(before)
                         - timedelta(days=self.SEARCH_BOUND_DAYS))
                        .strftime("%Y-%m-%dT%H:%M:%S"))
        r = session.get(self.API,
                        params={"categories": self.CATEGORY_TEAM_NEWS,
                                "search": f"gameweek {gw} team news",
                                "search_columns": "post_title",
                                "after": season_bound, "before": before[:19],
                                "per_page": per_page, "_fields": self._FIELDS},
                        headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        return matched(r.json())

    # -- extraction (stage A regex + optional stage B LLM, §5.3) -------------

    def extract(self, raw, gw, registry, resolver, *, url=None,
                observed_at=None, published_at=None, llm=None):
        """
        raw: fetch() output (list of WP post dicts), a single post dict, or
        a raw HTML string (archived article page — the step 6 re-scrape path).
        llm: optional stage-B extractor (intel_02_llm_extract.GeminiExtractor)
        for club sections where regex found < LLM_MIN_CLAIMS players.
        """
        claims, stats = [], _new_stats()
        stats["llm_gap_clubs"] = []
        stats["llm_claims"] = 0

        if isinstance(raw, str):
            posts = [{"content": {"rendered": raw}, "link": url or ""}]
        elif isinstance(raw, dict):
            posts = [raw]
        else:
            posts = list(raw or [])

        for post in posts:
            body = ((post.get("content") or {}).get("rendered")) or ""
            art_url = post.get("link") or url or ""
            pub = _wp_iso(post.get("date_gmt")) or published_at
            # A liveblog is an article: its content is asserted as of its
            # last EDIT, not our fetch (observed_at semantics, §5.1); the
            # original publication time drives recency and rule 2 ordering.
            mod = _wp_iso(post.get("modified_gmt")) or observed_at or pub
            self._extract_post(body, gw, registry, resolver, art_url,
                               mod or utcnow_iso(), pub, claims, stats, llm)
        return _finalize(claims, stats)

    def _extract_post(self, body_html, gw, registry, resolver, url,
                      observed_at, published_at, claims, stats, llm):
        soup = BeautifulSoup(body_html, "html.parser")
        root = self._article_root(soup, registry)
        current = None
        sections: dict = {}     # team_id -> {"club", "text": [], "n": int}
        seen: set = set()       # (team_id, player_raw) within this post

        def open_section(club):
            stats["clubs_seen"].add(club.short_name)
            sections.setdefault(club.team_id,
                                {"club": club, "text": [], "n": 0})

        for el in root.find_all(["h2", "h3", "h4", "p", "li"]):
            text = el.get_text(" ", strip=True)
            if el.name in ("h2", "h3", "h4"):
                # Any header is a section boundary; only a recognised club
                # header opens a section (§1.2 fix — unmatched headers must
                # never become the previous club's content)
                current = registry.match(text)
                if current is not None:
                    open_section(current)
                elif text:
                    stats["unmatched_clubs"].add(text[:40])
                continue
            club_hdr = self._strong_only_club(el, registry)
            if club_hdr is not None:
                current = club_hdr
                open_section(current)
                continue
            if current is None or not text:
                continue
            sections[current.team_id]["text"].append(text)

            for strong in el.find_all("strong"):
                name = strong.get_text(" ", strip=True).strip(" :")
                if not self._looks_like_player(name, registry):
                    continue
                key = (current.team_id, name)
                if key in seen:
                    continue        # first mention per club per edition
                seen.add(key)
                stats["rows_seen"] += 1
                p = resolver.resolve(name, current.team_id)
                if p is None:
                    stats["unresolved_players"].append(
                        f"{name} ({current.short_name})")
                claims.append(make_claim(
                    gw, self.NAME, self.TIER, url, current.team_id, name,
                    classify_news(text),
                    player_id=p["player_id"] if p else None,
                    injury=self._injury_for(strong, text), text=text[:300],
                    observed_at=observed_at, published_at=published_at,
                    extractor="regex"))
                sections[current.team_id]["n"] += 1

        # -- stage B: LLM gap-filling for thin club sections (§5.3) ----------
        for sec in sections.values():
            club, prose = sec["club"], " ".join(sec["text"])
            if (sec["n"] >= self.LLM_MIN_CLAIMS
                    or len(prose) < self.LLM_MIN_SECTION_CHARS):
                continue
            stats["llm_gap_clubs"].append(club.short_name)
            if llm is None:
                continue
            roster = [p["web_name"]
                      for p in resolver.team_players(club.team_id)]
            for row in llm.extract_club(club.name, prose, roster):
                name = (row.get("player") or "").strip()
                status = (row.get("status") or "").strip().lower()
                if not name or status not in STATUSES:
                    continue
                key = (club.team_id, name)
                if key in seen:
                    continue
                seen.add(key)
                p = resolver.resolve(name, club.team_id)
                if p is None:
                    stats["unresolved_players"].append(
                        f"{name} ({club.short_name})")
                claims.append(make_claim(
                    gw, self.NAME, self.TIER, url, club.team_id, name,
                    status, player_id=p["player_id"] if p else None,
                    injury=(row.get("injury") or "").strip().lower(),
                    text=(row.get("quote") or "")[:300],
                    observed_at=observed_at, published_at=published_at,
                    extractor="llm"))
                stats["llm_claims"] += 1

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _article_root(soup, registry):
        """
        content.rendered is a body fragment — parse it whole. An archived
        full page (step 6) is narrowed to the article container so sidebar
        and nav bolds never become claims (old find_article_body strategy).
        """
        if soup.find("body") is None:
            return soup
        for h in soup.find_all(["h2", "h3"]):
            if registry.match(h.get_text(" ", strip=True)):
                parent = h.parent
                if parent is not None and parent.name in ("div", "section",
                                                          "article"):
                    return parent
        return (soup.find("div", class_="entry-content")
                or soup.find("article") or soup)

    @staticmethod
    def _strong_only_club(el, registry):
        """<p><strong>Arsenal:</strong></p> — bold-paragraph club header."""
        if el.name != "p":
            return None
        kids = [c for c in el.children
                if not (isinstance(c, str) and not c.strip())]
        if (len(kids) == 1 and getattr(kids[0], "name", None) == "strong"):
            return registry.match(kids[0].get_text(" ", strip=True))
        return None

    def _looks_like_player(self, name: str, registry) -> bool:
        """Shape heuristic for a bold string being a player name."""
        name = name.strip()
        if len(name) < 3 or name.isupper():      # ALL-CAPS = inline header
            return False
        words = name.split()
        if not 1 <= len(words) <= 5:
            return False
        if not all(w[0].isupper() for w in words if w):
            return False
        if registry.match_any(name) is not None:  # club name in bold
            return False
        if name in self._REJECT_WORDS or words[0] in self._REJECT_WORDS:
            return False
        return True

    @staticmethod
    def _injury_for(strong, sentence: str) -> str:
        """
        "(hamstring)" straight after the bold name binds to that player —
        multi-player sentences carry one parenthetical each. Sentence-wide
        fallback matches the old scraper's behaviour.
        """
        nxt = strong.next_sibling
        if isinstance(nxt, str):
            m = re.match(r"\s*\(([^)]{2,40})\)", nxt)
            if m:
                return m.group(1).strip().lower()
        m = re.search(r"\(([^)]{4,40})\)", sentence)
        return m.group(1).strip().lower() if m else ""


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Story-per-article extraction (shared by Tier 3/4, §4.5/§4.7/§5.3)
# ---------------------------------------------------------------------------
# Tier 1/2 documents carry all 20 clubs in per-club sections; Tier 3/4 sources
# are individual news stories, each about one club/player. Attribution is:
# headline -> registry.match (a title naming two clubs is ambiguous -> skipped,
# never-guessed, §3.1), then a roster-grounded regex scan, then LLM gap-fill.

# Title filter for discovery — availability-relevant items only (the sitemap/
# RSS feeds are majority transfer/opinion noise).
_STORY_KEYWORDS = re.compile(
    r"team news|injur|press conference|ruled out|fitness|doubt|return|"
    r"boost|blow|available|suspend|comeback|sideline|setback|"
    r"line[- ]?ups?|predicted|starting xi|latest", re.IGNORECASE)

# a club section left thinner than this by the regex scan, with at least this
# much prose, is handed to the LLM (stage B). Story sources are noisy, so one
# regex claim already counts as "covered".
_STORY_LLM_MIN_CLAIMS = 1
_STORY_LLM_MIN_CHARS = 300

_INJURY_WORDS = ("hamstring", "knee", "ankle", "calf", "groin", "thigh",
                 "muscle", "shoulder", "achilles", "concussion", "illness",
                 "knock", "virus", "fracture", "broken", "surgery", "back",
                 "hip", "foot", "toe", "rib")

# Letters NFD cannot decompose (mirror of intel_identity.normalize_name so an
# English source's "Schar"/"Odegaard" matches roster "Schär"/"Ødegaard").
_FOLD_MAP = {"Ł": "l", "ł": "l", "Ø": "o", "ø": "o", "Đ": "d", "đ": "d",
             "Æ": "ae", "æ": "ae", "Œ": "oe", "œ": "oe", "ı": "i", "ß": "ss"}


def _fold(s: str) -> str:
    """Accent-fold + lowercase but KEEP word boundaries (spaces)."""
    s = s or ""
    for a, b in _FOLD_MAP.items():
        s = s.replace(a, b)
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return " ".join(stripped.split()).lower()


def _sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text or "").strip()
    return [p for p in re.split(r"(?<=[.!?])\s+", text) if p]


def _story_text(story: dict) -> str:
    """Plain body text of a story dict — pre-extracted `body`, else `html`."""
    if story.get("body"):
        return story["body"]
    html = story.get("html") or ""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    node = (soup.find("article")
            or soup.find(attrs={"class": re.compile(
                r"article-body|story-body|main-content|entry-content", re.I)})
            or soup)
    return " ".join(p.get_text(" ", strip=True) for p in node.find_all("p"))


def _iso_from_date_only(s: str) -> str | None:
    """'2026-07-07' -> ISO at 12:00 UTC (sitemap news dates carry no time)."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s or "")
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        12, 0, tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def _iso_from_rfc822(s: str) -> str | None:
    """RSS pubDate 'Tue, 7 Jul 2026 11:01:37 +0000' -> ISO UTC."""
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _player_name_tokens(player: dict) -> list:
    """Distinctive name forms to scan a story body for (>=4 alpha chars)."""
    toks = []
    wn = (player.get("web_name") or "").strip()
    if len(wn) >= 4 and wn.replace(" ", "").replace(".", "").replace("-", "").isalpha() \
            and "." not in wn:
        toks.append(wn)
    parts = (player.get("full_name") or "").split()
    if parts:
        last = parts[-1]
        if len(last) >= 4 and last.replace("-", "").isalpha() and last not in toks:
            toks.append(last)
    return toks


def _injury_in(sentence: str) -> str:
    m = re.search(r"\(([^)]{3,40})\)", sentence)
    if m:
        return m.group(1).strip().lower()
    low = sentence.lower()
    for w in _INJURY_WORDS:
        if w in low:
            return w
    return ""


def _attribute_club(title: str, body_text: str, registry):
    """Headline first (most reliable); else the article's opening sentence.
    Ambiguous (two clubs) or no-club -> None (never-guess, §3.1)."""
    club = registry.match(title or "")
    if club is not None:
        return club
    head = (_sentences(body_text)[:1] or [""])[0][:200]
    return registry.match(head)


def _roster_scan(body_text, club, gw, source_name, tier, url, resolver,
                 observed_at, published_at, claims, stats) -> int:
    """Stage A: for each roster player named in an availability sentence,
    emit a classified claim. Club-constrained, so surname matches are safe."""
    roster = resolver.team_players(club.team_id)
    seen = set()          # player_id, one claim per player per article
    n = 0
    for sent in _sentences(body_text):
        status = classify_news(sent)
        if status == "unknown":
            continue
        folded = _fold(sent)
        for p in roster:
            pid = p.get("player_id")
            if pid in seen:
                continue
            for cand in _player_name_tokens(p):
                if re.search(rf"\b{re.escape(_fold(cand))}\b", folded):
                    seen.add(pid)
                    n += 1
                    stats["rows_seen"] += 1
                    claims.append(make_claim(
                        gw, source_name, tier, url, club.team_id,
                        p.get("web_name", cand), status, player_id=pid,
                        injury=_injury_in(sent), text=sent[:300],
                        observed_at=observed_at, published_at=published_at,
                        extractor="regex"))
                    break
    return n


def _extract_stories(stories, gw, source_name, tier, registry, resolver,
                     claims, stats, llm, fallback_ts):
    """Attribute → roster-scan → LLM gap-fill for a batch of story dicts."""
    per_club: dict = {}       # team_id -> {"club", "prose": [], "n": int}
    for story in stories:
        title = (story.get("title") or "").strip()
        art_url = story.get("url") or ""
        # an article asserts its facts as of publication (§5.1 observed_at)
        ts = story.get("published") or fallback_ts
        body_text = _story_text(story)
        club = _attribute_club(title, body_text, registry)
        if club is None:
            stats["unmatched_clubs"].add((title or art_url or "?")[:50])
            continue
        stats["clubs_seen"].add(club.short_name)
        sec = per_club.setdefault(club.team_id,
                                  {"club": club, "prose": [], "n": 0})
        sec["prose"].append(body_text)
        sec["n"] += _roster_scan(body_text, club, gw, source_name, tier,
                                 art_url, resolver, ts, ts, claims, stats)

    # -- stage B: LLM gap-filling for clubs the regex left thin (§5.3) -------
    for sec in per_club.values():
        club, prose = sec["club"], " ".join(sec["prose"])
        if sec["n"] >= _STORY_LLM_MIN_CLAIMS or len(prose) < _STORY_LLM_MIN_CHARS:
            continue
        stats["llm_gap_clubs"].append(club.short_name)
        if llm is None:
            continue
        roster = [p["web_name"] for p in resolver.team_players(club.team_id)]
        for row in llm.extract_club(club.name, prose, roster):
            name = (row.get("player") or "").strip()
            status = (row.get("status") or "").strip().lower()
            if not name or status not in STATUSES:
                continue
            p = resolver.resolve(name, club.team_id)
            if p is None:
                stats["unresolved_players"].append(f"{name} ({club.short_name})")
            claims.append(make_claim(
                gw, source_name, tier, "", club.team_id, name, status,
                player_id=p["player_id"] if p else None,
                injury=(row.get("injury") or "").strip().lower(),
                text=(row.get("quote") or "")[:300],
                observed_at=fallback_ts, published_at=fallback_ts,
                extractor="llm"))
            stats["llm_claims"] += 1


def _new_story_stats() -> dict:
    stats = _new_stats()
    stats["llm_gap_clubs"] = []
    stats["llm_claims"] = 0
    return stats


# ---------------------------------------------------------------------------
# 6. Sky Sports — Google news sitemap + per-club index (Tier 3, §4.5)
# ---------------------------------------------------------------------------

class SkySportsAdapter:
    """
    Per-tick: the Google news sitemap (~50 URLs, all <48h old) filtered to
    football availability stories. Escalation (§7): a club's `{slug}-news`
    index page, scanned for exactly that club's fresh team-news items.
    Both flows produce story dicts consumed by the shared story extractor.
    """
    NAME = "sky"
    TIER = 3
    SITEMAP = "https://www.skysports.com/sitemap/sitemap-news.xml"
    CLUB_INDEX = "https://www.skysports.com/{slug}-news"
    BASE = "https://www.skysports.com"
    MAX_ARTICLES = 25
    ESCALATION_ARTICLES = 4
    _SM_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9",
              "news": "http://www.google.com/schemas/sitemap-news/0.9"}

    @classmethod
    def _parse_sitemap(cls, xml_bytes, max_articles=MAX_ARTICLES) -> list:
        root = ET.fromstring(xml_bytes)
        out = []
        for u in root.findall("s:url", cls._SM_NS):
            loc = u.findtext("s:loc", default="", namespaces=cls._SM_NS)
            title = u.findtext(".//news:title", default="",
                               namespaces=cls._SM_NS)
            date = u.findtext(".//news:publication_date", default="",
                              namespaces=cls._SM_NS)
            if "/football/" not in loc or not _STORY_KEYWORDS.search(title):
                continue
            out.append({"url": loc, "title": unescape(title),
                        "published": (_iso_from_date_only(date)
                                      or _iso_from_rfc822(date))})
            if len(out) >= max_articles:
                break
        return out

    def discover(self, session, max_articles=MAX_ARTICLES) -> list:
        r = session.get(self.SITEMAP, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        return self._parse_sitemap(r.content, max_articles)

    def fetch(self, session, max_articles=MAX_ARTICLES) -> list:
        stories = []
        for cand in self.discover(session, max_articles):
            try:
                r = session.get(cand["url"], headers=UA, timeout=TIMEOUT)
                r.raise_for_status()
            except Exception:                                 # noqa: BLE001
                continue
            stories.append({**cand, "html": r.text})
        return stories

    def fetch_club(self, session, club, max_articles=ESCALATION_ARTICLES):
        """T-24h escalation: fresh team-news items from a club's index page."""
        slug = SKY_SLUGS.get(club.slug)
        if not slug:
            return []
        try:
            r = session.get(self.CLUB_INDEX.format(slug=slug),
                            headers=UA, timeout=TIMEOUT)
            r.raise_for_status()
        except Exception:                                     # noqa: BLE001
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        seen, picked = set(), []
        for a in soup.find_all("a", href=True):
            href, txt = a["href"], a.get_text(" ", strip=True)
            if "/football/news/" not in href or not _STORY_KEYWORDS.search(txt):
                continue
            full = href if href.startswith("http") else self.BASE + href
            if full in seen:
                continue
            seen.add(full)
            picked.append((full, txt))
            if len(picked) >= max_articles:
                break
        stories = []
        for full, txt in picked:
            try:
                a = session.get(full, headers=UA, timeout=TIMEOUT)
                a.raise_for_status()
            except Exception:                                 # noqa: BLE001
                continue
            stories.append({"url": full, "title": txt,
                            "published": None, "html": a.text})
        return stories

    def extract(self, raw, gw, registry, resolver, *, url=None,
                observed_at=None, published_at=None, llm=None):
        stories = raw if isinstance(raw, list) else ([raw] if raw else [])
        claims, stats = [], _new_story_stats()
        _extract_stories(stories, gw, self.NAME, self.TIER, registry, resolver,
                         claims, stats, llm, observed_at or utcnow_iso())
        return _finalize(claims, stats)


# ---------------------------------------------------------------------------
# 7. Reach plc local outlets — per-club RSS (Tier 4 escalation, §4.7)
# ---------------------------------------------------------------------------
# One pattern across the Reach network: `{topic-page}?service=rss` gives a
# per-club feed whose articles carry application/ld+json `articleBody`.
# (base_url, topic_path) keyed by registry slug. LOWER-CONFIDENCE per §4.7 —
# these outlet slugs need season-start re-verification (step 5 checklist / §7
# health alarms catch drift). ChronicleLive/Newcastle is the verified anchor;
# the rest were live-probed 2026-07-07. Clubs without a confident Reach outlet
# (Brighton/Bournemouth/Southampton) are intentionally absent — escalation
# simply reports them uncovered rather than guessing a dead feed.

REACH_OUTLETS = {
    "arsenal":        ("https://www.football.london", "arsenal-fc"),
    "chelsea":        ("https://www.football.london", "chelsea-fc"),
    "spurs":          ("https://www.football.london", "tottenham-hotspur-fc"),
    "crystal-palace": ("https://www.football.london", "crystal-palace-fc"),
    "west-ham":       ("https://www.football.london", "west-ham-united-fc"),
    "liverpool":      ("https://www.liverpoolecho.co.uk",
                       "all-about/liverpool-fc"),
    "everton":        ("https://www.liverpoolecho.co.uk",
                       "all-about/everton-fc"),
    "man-utd":        ("https://www.manchestereveningnews.co.uk",
                       "all-about/manchester-united-fc"),
    "man-city":       ("https://www.manchestereveningnews.co.uk",
                       "all-about/manchester-city-fc"),
    "newcastle":      ("https://www.chroniclelive.co.uk",
                       "all-about/newcastle-united-fc"),      # verified anchor
    "sunderland":     ("https://www.chroniclelive.co.uk",
                       "all-about/sunderland-afc"),
    "aston-villa":    ("https://www.birminghammail.co.uk",
                       "all-about/aston-villa-fc"),
    "wolves":         ("https://www.birminghammail.co.uk",
                       "all-about/wolverhampton-wanderers-fc"),
    "nottm-forest":   ("https://www.nottinghampost.com",
                       "all-about/nottingham-forest-fc"),
    "leeds":          ("https://www.leeds-live.co.uk",
                       "all-about/leeds-united-fc"),
    "burnley":        ("https://www.lancs.live", "all-about/burnley-fc"),
    "leicester":      ("https://www.leicestermercury.co.uk",
                       "all-about/leicester-city-fc"),
}


class ReachLocalAdapter:
    """Escalation-only (Tier 4): per-club Reach RSS → JSON-LD articleBody →
    the shared story extractor. Never in the default tick source list — driven
    by run_gw's T-24h escalation for clubs with zero Tier 1-3 claims."""
    NAME = "reach"
    TIER = 4
    MAX_ARTICLES = 6

    @staticmethod
    def _parse_rss(xml_bytes) -> list:
        root = ET.fromstring(xml_bytes)
        items = []
        for it in root.findall(".//item"):
            items.append({
                "title": (it.findtext("title") or "").strip(),
                "url":   (it.findtext("link") or "").strip(),
                "published": _iso_from_rfc822(it.findtext("pubDate") or ""),
            })
        return items

    @staticmethod
    def _body_from_html(html: str) -> str:
        """First application/ld+json node carrying an articleBody."""
        for block in re.findall(
                r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>'
                r'(.*?)</script>', html or "", re.S):
            try:
                data = json.loads(block)
            except (json.JSONDecodeError, ValueError):
                continue
            nodes = data if isinstance(data, list) else [data]
            # some Reach pages wrap nodes under @graph
            graph = []
            for n in nodes:
                if isinstance(n, dict) and isinstance(n.get("@graph"), list):
                    graph.extend(n["@graph"])
            for n in nodes + graph:
                if isinstance(n, dict) and n.get("articleBody"):
                    return unescape(str(n["articleBody"]))
        return ""

    def _article_body(self, session, url: str) -> str:
        try:
            r = session.get(url, headers=UA, timeout=TIMEOUT)
            r.raise_for_status()
        except Exception:                                     # noqa: BLE001
            return ""
        return self._body_from_html(r.text)

    def fetch_club(self, session, club, max_articles=MAX_ARTICLES):
        outlet = REACH_OUTLETS.get(club.slug)
        if not outlet:
            return []
        base, topic = outlet
        try:
            r = session.get(f"{base}/{topic}?service=rss",
                            headers=UA, timeout=TIMEOUT)
            r.raise_for_status()
            items = self._parse_rss(r.content)
        except Exception:                                     # noqa: BLE001
            return []
        stories = []
        for it in items:
            if not it["url"] or not _STORY_KEYWORDS.search(it["title"]):
                continue
            body = self._article_body(session, it["url"])
            if not body:
                continue
            stories.append({**it, "body": body})
            if len(stories) >= max_articles:
                break
        return stories

    def extract(self, raw, gw, registry, resolver, *, url=None,
                observed_at=None, published_at=None, llm=None):
        stories = raw if isinstance(raw, list) else ([raw] if raw else [])
        claims, stats = [], _new_story_stats()
        _extract_stories(stories, gw, self.NAME, self.TIER, registry, resolver,
                         claims, stats, llm, observed_at or utcnow_iso())
        return _finalize(claims, stats)


# Sky per-club index slugs (escalation). Verified/known forms; §7 alarms and
# the step-5 season-start checklist re-verify these against a live redesign.
SKY_SLUGS = {
    "arsenal": "arsenal", "aston-villa": "aston-villa",
    "bournemouth": "bournemouth", "brentford": "brentford",
    "brighton": "brighton-and-hove-albion", "burnley": "burnley",
    "chelsea": "chelsea", "crystal-palace": "crystal-palace",
    "everton": "everton", "fulham": "fulham", "leeds": "leeds-united",
    "leicester": "leicester-city", "liverpool": "liverpool",
    "man-city": "manchester-city", "man-utd": "manchester-united",
    "newcastle": "newcastle-united", "nottm-forest": "nottingham-forest",
    "southampton": "southampton", "sunderland": "sunderland",
    "spurs": "tottenham-hotspur", "west-ham": "west-ham-united",
    "wolves": "wolves",
}


ALL_TIER1_ADAPTERS = [FfsInjuriesAdapter, SportsGamblerAdapter,
                      KnocksAndBansAdapter, GuardianAdapter]
# Tier 3 runs every tick; Tier 4 is escalation-only (run_gw drives it).
TICK_ADAPTERS = ALL_TIER1_ADAPTERS + [FfsTeamNewsAdapter, SkySportsAdapter]
ESCALATION_ADAPTERS = [SkySportsAdapter, ReachLocalAdapter]
