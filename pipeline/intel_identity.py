"""
pipeline/intel_identity.py
Shared identity module for the intel pipeline (scraper redesign step 1).

Single source of truth for mapping free-text club names and player names
(from press articles, injury tables, media feeds) onto canonical FPL ids.
Replaces the three drifting copies of this logic in intel_02/03/04, and the
exact-match club headers that made 7 clubs invisible for a full season
(docs/press_scraper_redesign.md §1.1).

Design (docs/press_scraper_redesign.md §3):
  - Club registry is built from FPL teams data at runtime (bootstrap-static
    teams[] or the local fpl_live.json snapshot) — never a hardcoded season
    club list.
  - A static season-independent ALIAS table covers every club that cycles
    through the PL. An FPL team that matches no alias entry gets an
    auto-generated entry from its own name (promoted-club safety net).
  - Matching ladder: exact normalized alias -> token-subset match.
    Ambiguous token matches (two clubs) return None, never a guess.
  - match_any() recognises known clubs even when they are not in the current
    FPL season (relegated) so article parsers can still treat their headers
    as section boundaries instead of misattributing content (§1.2).
  - Player resolution joins on FPL element id, club-constrained, using the
    intel_03 ladder (exact web_name -> exact full name -> substring ->
    last name -> last-name substring).

No third-party dependencies — stdlib only, so it runs in any image.
"""

import json
import os
import re
import unicodedata
from collections import namedtuple

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_CLUB_DROP_TOKENS = {"fc", "afc", "the"}


def normalize_name(name: str) -> str:
    """
    Player-name normalisation (identical to intel_03/04 behaviour):
    lowercase, strip diacritics, remove all non-alphanumerics.
    "Sørloth" -> "sorloth", "O'Riley" -> "oriley".
    """
    nfd = unicodedata.normalize("NFD", name)
    cleaned = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Letters NFD cannot decompose (the old intel_03 copy silently dropped
    # these, e.g. "Sørloth" -> "srloth"):
    for src, dst in (("Ł", "l"), ("ł", "l"), ("Ø", "o"), ("ø", "o"),
                     ("Đ", "d"), ("đ", "d"), ("Æ", "ae"), ("æ", "ae"),
                     ("Œ", "oe"), ("œ", "oe"), ("ı", "i"), ("ß", "ss")):
        cleaned = cleaned.replace(src, dst)
    return re.sub(r"[^a-z0-9]", "", cleaned.lower())


def club_tokens(text: str) -> tuple:
    """
    Club-name normalisation to a token tuple:
    casefold, fold accents, "&" -> "and", strip punctuation, drop FC/AFC/the.
    "BRIGHTON & HOVE ALBION FC:" -> ("brighton", "and", "hove", "albion")
    """
    nfd = unicodedata.normalize("NFD", text)
    text = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    text = text.replace("&", " and ")
    text = re.sub(r"[^A-Za-z0-9 ]", " ", text).casefold()
    return tuple(t for t in text.split() if t and t not in _CLUB_DROP_TOKENS)


def club_key(text: str) -> str:
    """Whole-string normalized form of a club name: joined tokens."""
    return " ".join(club_tokens(text))


# ---------------------------------------------------------------------------
# Season-independent club alias table
# ---------------------------------------------------------------------------
# slug -> aliases. Covers every club in the PL from 2016-17 onward plus the
# 2026-27 promotions. Aliases are matched exact-normalized first; multi-word
# aliases also participate in token-subset matching. 3-letter FPL short codes
# are exact-match only (added automatically from FPL data at registry build).

CLUB_ALIASES = {
    "arsenal":        ["Arsenal"],
    "aston-villa":    ["Aston Villa", "Villa"],
    "bournemouth":    ["Bournemouth", "AFC Bournemouth", "Cherries"],
    "brentford":      ["Brentford", "Bees"],
    "brighton":       ["Brighton", "Brighton and Hove Albion",
                       "Brighton & Hove Albion", "Brighton Hove Albion"],
    "burnley":        ["Burnley", "Clarets"],
    "cardiff":        ["Cardiff", "Cardiff City"],
    "chelsea":        ["Chelsea"],
    "coventry":       ["Coventry", "Coventry City", "Sky Blues"],
    "crystal-palace": ["Crystal Palace", "Palace", "Eagles"],
    "everton":        ["Everton", "Toffees"],
    "fulham":         ["Fulham", "Cottagers"],
    "huddersfield":   ["Huddersfield", "Huddersfield Town"],
    "hull":           ["Hull", "Hull City", "Tigers"],
    "ipswich":        ["Ipswich", "Ipswich Town", "Tractor Boys"],
    "leeds":          ["Leeds", "Leeds United"],
    "leicester":      ["Leicester", "Leicester City", "Foxes"],
    "liverpool":      ["Liverpool", "Reds"],
    "luton":          ["Luton", "Luton Town", "Hatters"],
    # NOTE: bare "United"/"City" are deliberately NOT aliases — a lone
    # "UNITED" header is ambiguous (Man Utd / Newcastle / West Ham / Leeds)
    # and the never-guess rule prefers an unmatched-header log entry.
    "man-city":       ["Manchester City", "Man City", "MCFC"],
    "man-utd":        ["Manchester United", "Man United", "Man Utd", "MUFC"],
    "middlesbrough":  ["Middlesbrough", "Boro"],
    "newcastle":      ["Newcastle", "Newcastle United", "NUFC", "Magpies",
                       "Newcastle Utd"],
    "norwich":        ["Norwich", "Norwich City", "Canaries"],
    "nottm-forest":   ["Nottingham Forest", "Nottm Forest", "Notts Forest",
                       "Forest", "Nott'm Forest"],
    "sheffield-utd":  ["Sheffield United", "Sheffield Utd", "Sheff Utd",
                       "Blades"],
    "southampton":    ["Southampton", "Saints"],
    "spurs":          ["Tottenham", "Tottenham Hotspur", "Spurs", "THFC"],
    "stoke":          ["Stoke", "Stoke City"],
    "sunderland":     ["Sunderland", "Black Cats"],
    "swansea":        ["Swansea", "Swansea City"],
    "watford":        ["Watford", "Hornets"],
    "west-brom":      ["West Brom", "West Bromwich Albion", "WBA"],
    "west-ham":       ["West Ham", "West Ham United", "Hammers", "Irons"],
    "wolves":         ["Wolves", "Wolverhampton", "Wolverhampton Wanderers"],
}

# Aliases too generic for token-subset matching: exact whole-string only.
# ("United" alone must not token-match "NEWCASTLE UNITED"; "City" must not
#  token-match "LEICESTER CITY".)
_EXACT_ONLY_ALIASES = {"united", "city", "villa", "palace", "forest", "reds",
                       "eagles", "bees", "blades", "boro", "saints", "irons",
                       "hammers", "toffees", "magpies", "cherries", "hatters",
                       "foxes", "tigers", "clarets", "hornets", "canaries",
                       "cottagers"}


FplClub = namedtuple("FplClub", ["team_id", "name", "short_name", "slug"])


class ClubRegistry:
    """
    Maps free-text club names -> canonical clubs.

    match(text)     -> FplClub | None   (clubs in the current FPL season)
    match_any(text) -> slug | None      (any known club, incl. relegated —
                                         for section-boundary detection)
    """

    def __init__(self, teams: list):
        """
        teams: list of {"id": int, "name": str, "short_name": str}
        (bootstrap-static teams[] shape).
        """
        # exact-normalized alias -> slug ; token tuple -> slug (subset pool)
        self._exact = {}
        self._token_aliases = []          # (frozenset(tokens), slug)
        for slug, aliases in CLUB_ALIASES.items():
            for alias in aliases:
                self._register_alias(alias, slug)

        # Map slugs -> FPL teams by matching each FPL team name through the
        # alias table itself. Unmatched FPL teams (never-seen promoted club)
        # get an auto-generated slug from their own name.
        self._by_slug = {}
        self.unmatched_fpl_teams = []
        for t in teams:
            tid = int(t["id"])
            name = str(t.get("name", ""))
            short = str(t.get("short_name", ""))
            slug = self._lookup_slug(name)
            if slug is None:
                slug = club_key(name).replace(" ", "-") or f"team-{tid}"
                self._register_alias(name, slug)
                self.unmatched_fpl_teams.append(name)
            club = FplClub(tid, name, short, slug)
            self._by_slug[slug] = club
            # FPL's own name and 3-letter code are always exact aliases
            self._exact.setdefault(club_key(name), slug)
            if short:
                self._exact.setdefault(club_key(short), slug)

    # -- construction helpers ------------------------------------------------

    def _register_alias(self, alias: str, slug: str):
        key = club_key(alias)
        if not key:
            return
        self._exact.setdefault(key, slug)
        tokens = club_tokens(alias)
        if key not in _EXACT_ONLY_ALIASES and len("".join(tokens)) >= 4:
            self._token_aliases.append((frozenset(tokens), slug))

    @classmethod
    def from_bootstrap(cls, bootstrap: dict) -> "ClubRegistry":
        return cls(bootstrap.get("teams", []))

    @classmethod
    def from_fpl_live(cls, path: str) -> "ClubRegistry":
        """Build from the local intel_01 snapshot (teams keyed by id str)."""
        with open(path, encoding="utf-8") as f:
            live = json.load(f)
        teams = [{"id": int(tid), "name": t.get("name", ""),
                  "short_name": t.get("short_name", "")}
                 for tid, t in live.get("teams", {}).items()]
        return cls(teams)

    # -- matching ------------------------------------------------------------

    def _lookup_slug(self, text: str) -> str | None:
        """Matching ladder: exact normalized -> unique token-subset."""
        key = club_key(text)
        if not key:
            return None
        if key in self._exact:
            return self._exact[key]

        text_tok = frozenset(club_tokens(text))
        if not text_tok:
            return None
        hits = set()
        for alias_tok, slug in self._token_aliases:
            # alias contained in text ("Newcastle" in "NEWCASTLE UNITED"),
            # or multi-token text contained in alias. Single-token text gets
            # exact matching only — a lone generic token ("WANDERERS",
            # "UNITED") must never subset-match a club.
            if alias_tok <= text_tok or (len(text_tok) >= 2
                                         and text_tok <= alias_tok):
                hits.add(slug)
        if len(hits) == 1:
            return hits.pop()
        return None  # no match, or ambiguous — never guess

    def match_any(self, text: str) -> str | None:
        """Slug of any known club (even if not in the current FPL season)."""
        return self._lookup_slug(text)

    def match(self, text: str) -> FplClub | None:
        """FplClub for clubs in the current FPL season; else None."""
        slug = self._lookup_slug(text)
        if slug is None:
            return None
        return self._by_slug.get(slug)

    def all_clubs(self) -> list:
        return sorted(self._by_slug.values(), key=lambda c: c.team_id)


# ---------------------------------------------------------------------------
# Player resolution
# ---------------------------------------------------------------------------

# Display-name aliases seen in press copy (kept from intel_03/04)
PLAYER_ALIASES = {
    "Son":         "Son Heung-min",
    "B.Fernandes": "Bruno Fernandes",
    "De Gea":      "David de Gea",
}


def _last_name(name: str) -> str:
    parts = name.strip().split()
    return parts[-1] if parts else name


def _translit_variants(norm: str) -> list:
    """
    ASCII-transliteration fallbacks for a normalized name. English sources
    write German/Nordic umlauts as digraphs ("Schaer" for Schär) while
    normalize_name folds the accent instead ("schar"). Try the literal form
    first, then digraphs collapsed.
    """
    variants = [norm]
    collapsed = (norm.replace("ae", "a").replace("oe", "o")
                     .replace("ue", "u"))
    if collapsed != norm:
        variants.append(collapsed)
    return variants


class PlayerResolver:
    """
    Resolves a source's player string to an FPL element id, constrained to
    the claimed club. Ladder (same as intel_03, which worked well):
      1. exact normalized web_name          within club
      2. exact normalized full name         within club
      3. source name substring of full name (>=4 chars)
      4. last-name exact match              (>=4 chars)
      5. last-name substring of full name   (>=4 chars)
    Returns the player dict (with player_id) or None. Never guesses across
    clubs — claims are club-partitioned before resolution.
    """

    def __init__(self, players: dict):
        """
        players: {pid_str: {web_name, full_name, team_id, ...}}
        (fpl_live.json players shape; bootstrap elements can be adapted).
        """
        self._by_team = {}
        for pid_str, p in players.items():
            entry = dict(p)
            entry["player_id"] = int(pid_str)
            self._by_team.setdefault(int(p.get("team_id", 0)), []).append(entry)

    @classmethod
    def from_fpl_live(cls, path: str) -> "PlayerResolver":
        with open(path, encoding="utf-8") as f:
            live = json.load(f)
        return cls(live.get("players", {}))

    @classmethod
    def from_bootstrap(cls, bootstrap: dict) -> "PlayerResolver":
        players = {}
        for e in bootstrap.get("elements", []):
            players[str(int(e["id"]))] = {
                "web_name":  str(e.get("web_name", "")),
                "full_name": f"{e.get('first_name', '')} {e.get('second_name', '')}".strip(),
                "team_id":   int(e.get("team", 0)),
            }
        return cls(players)

    def team_players(self, team_id: int) -> list:
        """All players registered to a club (e.g. LLM roster grounding)."""
        return list(self._by_team.get(int(team_id), []))

    def resolve(self, source_name: str, team_id: int) -> dict | None:
        display = PLAYER_ALIASES.get(source_name, source_name)
        candidates = self._by_team.get(int(team_id), [])
        if not candidates:
            return None
        norm = normalize_name(display)
        if len(norm) < 3:
            return None

        for variant in _translit_variants(norm):
            p = self._resolve_norm(variant, display, candidates)
            if p is not None:
                return p
        return None

    def _resolve_norm(self, norm: str, display: str,
                      candidates: list) -> dict | None:
        for p in candidates:
            if normalize_name(p["web_name"]) == norm:
                return p
        for p in candidates:
            if normalize_name(p["full_name"]) == norm:
                return p
        if len(norm) >= 4:
            for p in candidates:
                if norm in normalize_name(p["full_name"]):
                    return p
        last_src = normalize_name(_last_name(display))
        for last in _translit_variants(last_src):
            if len(last) < 4:
                continue
            for p in candidates:
                if normalize_name(_last_name(p["full_name"])) == last:
                    return p
            for p in candidates:
                if last in normalize_name(p["full_name"]):
                    return p
        return None


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPL_LIVE_PATH = os.path.join(_ROOT, "data", "intel", "fpl_live.json")
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def load_reference(prefer_live_api: bool = False,
                   fpl_live_path: str = FPL_LIVE_PATH):
    """
    Return (ClubRegistry, PlayerResolver).

    prefer_live_api=True fetches bootstrap-static (needs `requests` and
    network); otherwise — and as fallback — uses the local intel_01 snapshot.
    """
    if prefer_live_api:
        try:
            import requests
            r = requests.get(BOOTSTRAP_URL,
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            r.raise_for_status()
            boot = r.json()
            return (ClubRegistry.from_bootstrap(boot),
                    PlayerResolver.from_bootstrap(boot))
        except Exception as e:                       # noqa: BLE001
            print(f"  [IDENTITY] bootstrap fetch failed ({e}); "
                  f"falling back to {fpl_live_path}")
    return (ClubRegistry.from_fpl_live(fpl_live_path),
            PlayerResolver.from_fpl_live(fpl_live_path))
