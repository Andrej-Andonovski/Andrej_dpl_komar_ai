"""
pipeline/intel_02_scrape.py
Multi-source scraper orchestrator (redesign steps 2-4).

Runs the Tier 1 adapters, the Tier 2 FFS liveblog, and the Tier 3 Sky Sports
sitemap for one GW, appends claims to the append-only ledger, reconciles, and
writes a schema-compatible superset of the old press_conferences.json so
intel_03 can consume it unchanged (§5.4). With --escalate, clubs left with
zero Tier 1-3 claims trigger Tier 3/4 escalation (Sky per-club index + Reach
local RSS) for exactly those clubs (§7).

Outputs (never touches the production press_conferences.json):
  data/intel/press_claims/gw{N}.jsonl       append-only claim ledger
  data/intel/press_claims/source_health.json per-source health counters
  data/intel/press_conferences_v2.json      reconciled compat output

Usage (Docker — this machine has no local Python):
  docker run --rm -v "<repo>:/app" -w /app fpl-scrape \
      python pipeline/intel_02_scrape.py --gw 22 [--sources guardian,ffs_teamnews]
      [--deadline 2026-01-16T18:30] [--llm] [--escalate]

--deadline anchors the FFS liveblog discovery window (deadline-6d ->
deadline); omitted, it is fetched from the live FPL API, falling back to a
trailing window ending now. --llm enables stage-B Gemini gap-filling (§5.3);
it needs GEMINI_API_KEY and google-genai, and degrades to regex-only without.

The scheduler (--tick mode, §6) is a later build step.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intel_identity import load_reference, BOOTSTRAP_URL
from intel_02_ledger import ClaimLedger, reconcile_gw, utcnow_iso
from intel_02_sources import (FfsInjuriesAdapter, SportsGamblerAdapter,
                              KnocksAndBansAdapter, GuardianAdapter,
                              FfsTeamNewsAdapter, SkySportsAdapter,
                              ReachLocalAdapter, UA)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEL_DIR = os.path.join(ROOT, "data", "intel")
LEDGER_DIR = os.path.join(INTEL_DIR, "press_claims")
OUT_PATH = os.path.join(INTEL_DIR, "press_conferences_v2.json")
HEALTH_PATH = os.path.join(LEDGER_DIR, "source_health.json")

# slug -> club name as legacy intel_03 expects it (its press_clubs list).
# Clubs outside that list (e.g. Burnley, 2026-27 promotions) keep their FPL
# name — legacy intel_03 will not map them until it consumes player_id
# directly; the ids are already in the output (additive fields).
COMPAT_CLUB_NAMES = {
    "arsenal": "Arsenal", "aston-villa": "Aston Villa",
    "bournemouth": "Bournemouth", "brentford": "Brentford",
    "brighton": "Brighton", "burnley": "Burnley", "chelsea": "Chelsea",
    "crystal-palace": "Crystal Palace", "everton": "Everton",
    "fulham": "Fulham", "ipswich": "Ipswich", "leeds": "Leeds",
    "leicester": "Leicester", "liverpool": "Liverpool",
    "man-city": "Manchester City", "man-utd": "Manchester United",
    "newcastle": "Newcastle", "nottm-forest": "Nottingham Forest",
    "southampton": "Southampton", "sunderland": "Sunderland",
    "spurs": "Tottenham", "west-ham": "West Ham", "wolves": "Wolves",
}

# Default per-tick sources (Tier 1-3). Reach (Tier 4) is escalation-only and
# is driven by run_gw's T-24h escalation, never listed here (fetch_club needs
# a specific club, not a bulk fetch).
ADAPTERS = {
    "ffs_injuries":  FfsInjuriesAdapter,
    "sportsgambler": SportsGamblerAdapter,
    "knocksandbans": KnocksAndBansAdapter,
    "guardian":      GuardianAdapter,
    "ffs_teamnews":  FfsTeamNewsAdapter,
    "sky":           SkySportsAdapter,
}

# Sources whose extract() takes an optional stage-B llm= gap-filler.
LLM_SOURCES = {"ffs_teamnews", "sky"}


def ffs_window(gw: int, deadline_iso: str | None = None) -> dict | None:
    """
    fetch kwargs for the FFS liveblog discovery window (deadline-6d ->
    deadline, §4.1). deadline_iso omitted -> ask the live FPL API for this
    GW's deadline; None on any failure (adapter then uses its trailing
    now-window, correct for live pre-deadline ticks).
    """
    if deadline_iso is None:
        try:
            r = requests.get(BOOTSTRAP_URL, headers=UA, timeout=20)
            r.raise_for_status()
            deadline_iso = next(
                (e["deadline_time"] for e in r.json().get("events", [])
                 if int(e.get("id", 0)) == gw), None)
        except Exception as e:                                # noqa: BLE001
            print(f"  [ffs_teamnews] deadline lookup failed ({e}); "
                  f"using trailing window")
    if not deadline_iso:
        return None
    d = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
    fmt = "%Y-%m-%dT%H:%M:%S"
    return {"after": (d - timedelta(days=FfsTeamNewsAdapter.WINDOW_DAYS))
            .strftime(fmt),
            "before": d.strftime(fmt)}


# ---------------------------------------------------------------------------
# Per-source health (§7 parse alarms)
# ---------------------------------------------------------------------------

def load_health() -> dict:
    if os.path.exists(HEALTH_PATH):
        with open(HEALTH_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def update_health(health: dict, source: str, ok_fetch: bool, n_claims: int):
    h = health.setdefault(source, {"consecutive_failures": 0,
                                   "last_claims": None, "alerts": []})
    now = utcnow_iso()
    h["last_run"] = now
    if not ok_fetch:
        h["consecutive_failures"] += 1
        h["alerts"].append({"at": now, "type": "fetch_failed"})
    elif n_claims == 0 and (h.get("last_claims") or 0) > 0:
        # HTTP OK but a previously-productive parser found nothing
        h["consecutive_failures"] += 1
        h["alerts"].append({"at": now, "type": "structure_drift"})
    else:
        h["consecutive_failures"] = 0
        h["last_claims"] = n_claims
    h["alerts"] = h["alerts"][-20:]


def save_health(health: dict):
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(HEALTH_PATH, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Compat output (§5.4)
# ---------------------------------------------------------------------------

def build_compat_gw(gw: int, records: list, registry, sources_run: list,
                    scraped_at: str | None = None) -> dict:
    """Reconciled records -> one GW block in press_conferences.json shape."""
    club_by_id = {c.team_id: c for c in registry.all_clubs()}

    def compat_name(club_id: int) -> str:
        c = club_by_id.get(club_id)
        if c is None:
            return f"team-{club_id}"
        return COMPAT_CLUB_NAMES.get(c.slug, c.name)

    clubs: dict = {}
    all_news: list = []
    for r in sorted(records, key=lambda x: (x["club_id"],
                                            x["player_raw"].casefold())):
        cname = compat_name(r["club_id"])
        entry = {
            # legacy fields — exactly what intel_03 reads today
            "player":       r["player_raw"],
            "injury":       r["injury"],
            "news":         r["text"],
            "availability": r["status"],
            # additive fields (§5.4)
            "player_id":    r["player_id"],
            "score":        r["score"],
            "n_sources":    r["n_sources"],
            "conflict":     r["conflict"],
            "sources":      r["sources"],
            "return_date":  r["return_date"],
        }
        clubs.setdefault(cname, {"raw_text": "", "players": []})
        clubs[cname]["players"].append(entry)
        all_news.append({**entry, "club": cname})

    for cname, blk in clubs.items():
        blk["raw_text"] = "\n".join(
            f"{p['player']}: {p['news']}" for p in blk["players"])

    covered = {compat_name(r["club_id"]) for r in records}
    all_current = {compat_name(c.team_id) for c in registry.all_clubs()}
    coverage = {
        "clubs_covered": len(covered),
        "missing": sorted(all_current - covered),
        "per_club": {
            cname: {
                "claims": len(blk["players"]),
                "conflicts": sum(1 for p in blk["players"] if p["conflict"]),
            } for cname, blk in sorted(clubs.items())
        },
    }

    return {
        "gw": gw,
        "url": None,
        "status": "success" if records else "no_claims",
        "scraped_at": scraped_at or utcnow_iso(),
        "sources_run": sources_run,
        "coverage": coverage,
        "clubs": clubs,
        "all_player_news": all_news,
    }


def merge_compat_output(gw: int, gw_block: dict, out_path: str = OUT_PATH):
    """Merge one GW block into press_conferences_v2.json (never production)."""
    data = {"generated_by": "intel_02_scrape (v2 multi-source)",
            "gameweeks": {}}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
    data["scraped_at"] = utcnow_iso()
    data.setdefault("gameweeks", {})[str(gw)] = gw_block
    gws_ok = sorted(int(k) for k, v in data["gameweeks"].items()
                    if v.get("status") == "success")
    data["gws_scraped"] = gws_ok
    data["gws_failed"] = sorted(int(k) for k in data["gameweeks"]
                                if int(k) not in gws_ok)
    data["total_player_mentions"] = sum(
        len(v.get("all_player_news", [])) for v in data["gameweeks"].values())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# One scrape pass
# ---------------------------------------------------------------------------

def missing_clubs(records: list, registry) -> list:
    """Current-season clubs with zero reconciled records — the §7 escalation
    target set (clubs with no Tier 1-3 claim after reconciliation)."""
    covered = {r["club_id"] for r in records}
    return [c for c in registry.all_clubs() if c.team_id not in covered]


def run_gw(gw: int, source_names: list, *, registry=None, resolver=None,
           prefetched: dict | None = None, ref_time: str | None = None,
           ledger_dir: str = LEDGER_DIR, out_path: str = OUT_PATH,
           guardian_fetch_kw: dict | None = None,
           ffs_fetch_kw: dict | None = None, llm=None, escalate: bool = False,
           verbose: bool = True):
    """
    Run adapters for one GW and rebuild that GW's reconciled output.

    prefetched: {source_name: (raw, url, observed_at, published_at)} —
    lets the validation harness feed archived content (Wayback snapshots,
    Guardian date-window queries) through the identical pipeline.
    ffs_fetch_kw: ffs_window() output anchoring liveblog discovery.
    llm: optional GeminiExtractor for stage-B gap-filling (§5.3).
    escalate: after reconciling, run the Tier 3/4 escalation (§7) for clubs
    still uncovered — Sky per-club index then Reach local RSS. The scheduler
    (step 5) sets this only inside the T-24h window; here it is caller-driven.
    Returns the compat GW block.
    """
    if registry is None or resolver is None:
        registry, resolver = load_reference()

    session = requests.Session()
    ledger = ClaimLedger(ledger_dir)
    health = load_health()
    sources_run = []

    for name in source_names:
        cls = ADAPTERS[name]
        adapter = cls() if name != "guardian" else cls(
            api_key=os.environ.get("GUARDIAN_API_KEY", "test"))
        raw, url, observed_at, published_at = None, None, None, None
        ok_fetch = True
        try:
            if prefetched and name in prefetched:
                raw, url, observed_at, published_at = prefetched[name]
            elif name == "guardian":
                raw = adapter.fetch(session, **(guardian_fetch_kw or {}))
            elif name == "ffs_teamnews":
                raw = adapter.fetch(session, gw, **(ffs_fetch_kw or {}))
            else:
                raw = adapter.fetch(session)
        except Exception as e:                                # noqa: BLE001
            ok_fetch = False
            if verbose:
                print(f"  [{name}] FETCH FAILED: {e}")

        claims, stats = [], {}
        extract_kw = {"llm": llm} if name in LLM_SOURCES else {}
        if ok_fetch and raw is not None:
            try:
                claims, stats = adapter.extract(
                    raw, gw, registry, resolver, url=url,
                    observed_at=observed_at, published_at=published_at,
                    **extract_kw)
            except Exception as e:                            # noqa: BLE001
                ok_fetch = False
                if verbose:
                    print(f"  [{name}] EXTRACT FAILED: {e}")

        n_new = ledger.append(gw, claims) if claims else 0
        update_health(health, name, ok_fetch, len(claims))
        sources_run.append(name)
        if verbose:
            drift = (" [STRUCTURE-DRIFT?]" if ok_fetch and not claims
                     and health[name]["consecutive_failures"] else "")
            llm_note = ""
            if stats.get("llm_gap_clubs"):
                llm_note = (f" llm_gaps={','.join(stats['llm_gap_clubs'])}"
                            f" llm_claims={stats.get('llm_claims', 0)}")
            print(f"  [{name}] claims={len(claims)} new={n_new} "
                  f"clubs={len(stats.get('clubs_seen', []))} "
                  f"unresolved={len(stats.get('unresolved_players', []))}"
                  f"{llm_note}{drift}")
            for u in stats.get("unresolved_players", [])[:6]:
                print(f"      unresolved: {u}")

    # Reconcile the FULL ledger for this GW (all ticks so far), rebuild block
    all_claims = ledger.load(gw)
    records = reconcile_gw(all_claims, ref_time=ref_time)
    block = build_compat_gw(gw, records, registry, sources_run)

    # -- T-24h escalation (§7): Tier 3/4 for clubs with zero Tier 1-3 claims --
    if escalate:
        missing = missing_clubs(records, registry)
        if verbose:
            print(f"  [escalation] {len(missing)} uncovered club(s)"
                  + (f": {', '.join(c.short_name for c in missing)}"
                     if missing else ""))
        got_new = False
        for club in missing:
            for adapter in (SkySportsAdapter(), ReachLocalAdapter()):
                ok_fetch = True
                try:
                    stories = adapter.fetch_club(session, club)
                except Exception as e:                        # noqa: BLE001
                    ok_fetch, stories = False, []
                    if verbose:
                        print(f"    [{adapter.NAME}:{club.short_name}] "
                              f"fetch failed: {e}")
                if not stories:
                    continue
                claims, stats = adapter.extract(stories, gw, registry,
                                                resolver, llm=llm)
                n_new = ledger.append(gw, claims) if claims else 0
                got_new = got_new or bool(n_new)
                update_health(health, f"{adapter.NAME}_escalation",
                              ok_fetch, len(claims))
                sources_run.append(f"{adapter.NAME}_escalation")
                if verbose:
                    print(f"    [{adapter.NAME}:{club.short_name}] "
                          f"stories={len(stories)} claims={len(claims)} "
                          f"new={n_new} clubs={len(stats.get('clubs_seen', []))}")
        if got_new:
            all_claims = ledger.load(gw)
            records = reconcile_gw(all_claims, ref_time=ref_time)
            block = build_compat_gw(gw, records, registry, sources_run)

    save_health(health)
    merge_compat_output(gw, block, out_path)

    if verbose:
        cov = block["coverage"]
        print(f"  GW{gw}: {len(all_claims)} claims -> {len(records)} players "
              f"| coverage {cov['clubs_covered']}/20"
              + (f" | missing: {', '.join(cov['missing'])}"
                 if cov["missing"] else ""))
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--sources", default=",".join(ADAPTERS),
                    help="comma list: " + ",".join(ADAPTERS))
    ap.add_argument("--live-api", action="store_true",
                    help="build identity from live bootstrap-static")
    ap.add_argument("--deadline", default=None, metavar="ISO",
                    help="GW deadline anchoring FFS liveblog discovery "
                         "(default: live FPL API lookup)")
    ap.add_argument("--llm", action="store_true",
                    help="enable stage-B Gemini gap extraction (§5.3); "
                         "needs GEMINI_API_KEY")
    ap.add_argument("--escalate", action="store_true",
                    help="run Tier 3/4 escalation (§7) for clubs left with "
                         "zero Tier 1-3 claims after reconciliation")
    args = ap.parse_args()

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    bad = [n for n in names if n not in ADAPTERS]
    if bad:
        ap.error(f"unknown sources: {bad}")

    fetch_kw = (ffs_window(args.gw, args.deadline)
                if "ffs_teamnews" in names else None)
    llm = None
    if args.llm:
        from intel_02_llm_extract import GeminiExtractor
        llm = GeminiExtractor()
        if not llm.available():
            print("  [llm] GEMINI_API_KEY/google-genai missing — "
                  "regex-only extraction")
            llm = None

    print(f"intel_02 v2 — GW{args.gw} — sources: {', '.join(names)}")
    registry, resolver = load_reference(prefer_live_api=args.live_api)
    run_gw(args.gw, names, registry=registry, resolver=resolver,
           ffs_fetch_kw=fetch_kw, llm=llm, escalate=args.escalate)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
