"""
scripts/validate_scraper_v2.py
Off-season validation of the v2 multi-source scraper against ARCHIVED
2025-26 season content (redesign build-plan steps 1-3 acceptance check).

Per validation GW:
  - Guardian:      Content API date-window query (full history available)
  - FFS liveblog:  WP REST API deadline-window query (full post archive —
                   no Wayback needed), same discovery path used live (step 3)
  - FFS injuries / SportsGambler / KnocksAndBans: Wayback Machine snapshot
                   nearest the GW deadline, fed through the same extract()
                   path used live
Then: ledger -> reconcile -> compat block, and a per-GW check that the 7
clubs the old scraper never extracted (Newcastle, Spurs, West Ham, Wolves,
Brighton, Leeds, Burnley) now produce claims.

Outputs (validation namespace — production files untouched):
  data/intel/validation_2526/press_claims/gw{N}.jsonl
  data/intel/validation_2526/press_conferences_v2.json
  data/intel/validation_2526/validation_report.json

Run:
  docker run --rm -v "<repo>:/app" -w /app fpl-scrape \
      python scripts/validate_scraper_v2.py [--gws 12,22,38]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

from intel_identity import load_reference                     # noqa: E402
from intel_02_ledger import ClaimLedger, reconcile_gw         # noqa: E402
from intel_02_scrape import build_compat_gw, ADAPTERS         # noqa: E402
from intel_02_sources import (FfsInjuriesAdapter, SportsGamblerAdapter,  # noqa: E402
                              KnocksAndBansAdapter, GuardianAdapter,
                              FfsTeamNewsAdapter, UA)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAL_DIR = os.path.join(ROOT, "data", "intel", "validation_2526")
LEDGER_DIR = os.path.join(VAL_DIR, "press_claims")
OUT_PATH = os.path.join(VAL_DIR, "press_conferences_v2.json")
REPORT_PATH = os.path.join(VAL_DIR, "validation_report.json")

# Fallback GW -> deadline-week date (from the FFS liveblog publication dates
# the old scraper recorded in KNOWN_URLS). Used only if the FPL API has
# rolled over to 2026-27 AND no local cache exists.
GW_DATES = {
    1: "2025-08-15", 5: "2025-09-19", 8: "2025-10-17", 12: "2025-11-21",
    16: "2025-12-12", 22: "2026-01-16", 25: "2026-02-06", 28: "2026-02-27",
    32: "2026-04-10", 38: "2026-05-22",
}

DEADLINES_CACHE = os.path.join(VAL_DIR, "deadlines_2526.json")
WAYBACK_SLEEP_S = 2   # polite delay between archive.org requests
SNAP_MAX_DIST_DAYS = 7   # a current-state snapshot further than this from
                         # the deadline says nothing about that GW — skip it


def load_deadlines(session) -> dict:
    """
    {gw: 'YYYY-MM-DD'} for all 38 GWs of 2025-26.
    Priority: local cache -> FPL bootstrap-static (still serving 2025-26
    during the 2026 off-season) -> KNOWN_URLS fallback sample.
    """
    if os.path.exists(DEADLINES_CACHE):
        with open(DEADLINES_CACHE, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    try:
        r = session.get("https://fantasy.premierleague.com/api/bootstrap-static/",
                        headers=UA, timeout=20)
        r.raise_for_status()
        events = r.json()["events"]
        dates = {int(e["id"]): e["deadline_time"][:10] for e in events}
        if dates.get(1, "").startswith("2025-08"):      # really 2025-26?
            os.makedirs(VAL_DIR, exist_ok=True)
            with open(DEADLINES_CACHE, "w", encoding="utf-8") as f:
                json.dump(dates, f, indent=2)
            return dates
        print("  [DEADLINES] API already rolled to a new season — "
              "falling back to KNOWN_URLS sample dates")
    except Exception as e:                                # noqa: BLE001
        print(f"  [DEADLINES] API fetch failed ({e}) — using fallback dates")
    return dict(GW_DATES)

BLIND_CLUBS = ["Newcastle", "Tottenham", "West Ham", "Wolves",
               "Brighton", "Leeds", "Burnley"]

WAYBACK_AVAIL = "https://archive.org/wayback/available"


# ---------------------------------------------------------------------------
# Wayback helpers
# ---------------------------------------------------------------------------

def wayback_snapshot(session, target_url: str, date_yyyymmdd: str):
    """
    Return (snapshot_url_id, snapshot_ts_iso, distance_days) for the snapshot
    closest to the date, or (None, None, None) if none exists.
    """
    try:
        r = session.get(WAYBACK_AVAIL,
                        params={"url": target_url,
                                "timestamp": date_yyyymmdd + "120000"},
                        headers=UA, timeout=45)
        r.raise_for_status()
        snap = (r.json().get("archived_snapshots") or {}).get("closest")
        if not snap or not snap.get("available"):
            return None, None, None
        ts = snap["timestamp"]                       # YYYYMMDDhhmmss
        iso = datetime(int(ts[0:4]), int(ts[4:6]), int(ts[6:8]),
                       int(ts[8:10]), int(ts[10:12]),
                       tzinfo=timezone.utc).isoformat()
        want = datetime.strptime(date_yyyymmdd, "%Y%m%d").replace(
            tzinfo=timezone.utc)
        dist = abs((datetime.fromisoformat(iso) - want).days)
        # id_ flag -> original bytes, no Wayback banner rewriting
        url = snap["url"].replace("/http", "id_/http", 1) \
            if "id_/" not in snap["url"] else snap["url"]
        return url, iso, dist
    except Exception as e:                                    # noqa: BLE001
        print(f"    wayback lookup failed for {target_url}: {e}")
        return None, None, None


def fetch_snapshot(session, snap_url: str) -> str | None:
    try:
        r = session.get(snap_url, headers=UA, timeout=90)
        r.raise_for_status()
        return r.text
    except Exception as e:                                    # noqa: BLE001
        print(f"    snapshot fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-GW validation
# ---------------------------------------------------------------------------

def validate_gw(gw: int, session, registry, resolver, report: dict,
                gw_dates: dict):
    date = gw_dates[gw]
    ymd = date.replace("-", "")
    print(f"\n== GW{gw} (deadline week of {date}) " + "=" * 30)
    ledger = ClaimLedger(LEDGER_DIR)
    gw_report = {"date": date, "sources": {}, "blind_clubs": {}}

    # --- Guardian: real historical API query --------------------------------
    g = GuardianAdapter(api_key=os.environ.get("GUARDIAN_API_KEY", "test"))
    try:
        from datetime import timedelta
        d = datetime.strptime(date, "%Y-%m-%d")
        raw = g.fetch(session,
                      from_date=(d - timedelta(days=4)).strftime("%Y-%m-%d"),
                      to_date=(d + timedelta(days=1)).strftime("%Y-%m-%d"))
        n_articles = len(raw.get("response", {}).get("results", []))
        claims, stats = g.extract(raw, gw, registry, resolver)
        ledger.append(gw, claims)
        gw_report["sources"]["guardian"] = {
            "mode": "content-api historical query", "articles": n_articles,
            "claims": len(claims), "clubs": stats["clubs_seen"],
            "unresolved": stats["unresolved_players"]}
        print(f"  [guardian] articles={n_articles} claims={len(claims)} "
              f"clubs={len(stats['clubs_seen'])}")
    except Exception as e:                                    # noqa: BLE001
        gw_report["sources"]["guardian"] = {"error": str(e)}
        print(f"  [guardian] FAILED: {e}")

    # --- FFS liveblog: WP REST API deadline-window query (step 3) ------------
    # The WP API serves the full post archive, so historical discovery is the
    # exact live code path — no Wayback dependency. Editions edited after the
    # deadline carry a post-deadline modified_gmt (observed_at); staleness
    # tolerates it, and live ticks always run pre-deadline.
    fl = FfsTeamNewsAdapter()
    try:
        from datetime import timedelta
        from html import unescape
        d = datetime.strptime(date, "%Y-%m-%d")
        posts = fl.fetch(
            session, gw,
            after=(d - timedelta(days=6)).strftime("%Y-%m-%dT00:00:00"),
            before=d.strftime("%Y-%m-%dT23:59:59"))
        claims, stats = fl.extract(posts, gw, registry, resolver)
        ledger.append(gw, claims)
        gw_report["sources"]["ffs_teamnews"] = {
            "mode": "wp-rest deadline-window query", "editions": len(posts),
            "titles": [unescape(p["title"]["rendered"]) for p in posts],
            "claims": len(claims), "clubs": stats["clubs_seen"],
            "unmatched_headers": stats["unmatched_clubs"],
            "llm_gap_clubs": stats.get("llm_gap_clubs", []),
            "unresolved": stats["unresolved_players"][:10]}
        print(f"  [ffs_teamnews] editions={len(posts)} claims={len(claims)} "
              f"clubs={len(stats['clubs_seen'])}")
    except Exception as e:                                    # noqa: BLE001
        gw_report["sources"]["ffs_teamnews"] = {"error": str(e)}
        print(f"  [ffs_teamnews] FAILED: {e}")

    # --- Stable-URL sources: Wayback snapshots -------------------------------
    for name, adapter in [("ffs_injuries", FfsInjuriesAdapter()),
                          ("sportsgambler", SportsGamblerAdapter()),
                          ("knocksandbans", KnocksAndBansAdapter())]:
        time.sleep(WAYBACK_SLEEP_S)
        snap_url, snap_iso, dist = wayback_snapshot(session, adapter.URL, ymd)
        if snap_url is None:
            gw_report["sources"][name] = {
                "mode": "wayback", "snapshot": None,
                "note": "no snapshot available near this date"}
            print(f"  [{name}] no Wayback snapshot near {date}")
            continue
        if dist > SNAP_MAX_DIST_DAYS:
            gw_report["sources"][name] = {
                "mode": "wayback", "snapshot": snap_url,
                "distance_days": dist,
                "note": f"nearest snapshot ±{dist}d from deadline — "
                        f"too stale for this GW, skipped"}
            print(f"  [{name}] nearest snapshot ±{dist}d — skipped (>"
                  f"{SNAP_MAX_DIST_DAYS}d)")
            continue
        html = fetch_snapshot(session, snap_url)
        if html is None:
            gw_report["sources"][name] = {
                "mode": "wayback", "snapshot": snap_url,
                "note": "snapshot fetch failed"}
            continue
        claims, stats = adapter.extract(
            html, gw, registry, resolver, url=snap_url,
            observed_at=snap_iso, published_at=snap_iso)
        ledger.append(gw, claims)
        gw_report["sources"][name] = {
            "mode": "wayback", "snapshot": snap_url,
            "snapshot_time": snap_iso, "distance_days": dist,
            "claims": len(claims), "clubs": stats["clubs_seen"],
            "unmatched_clubs": stats["unmatched_clubs"],
            "unresolved": stats["unresolved_players"][:10]}
        print(f"  [{name}] snapshot {snap_iso[:10]} (±{dist}d) "
              f"claims={len(claims)} clubs={len(stats['clubs_seen'])}")

    # --- Reconcile + compat block + blind-club check --------------------------
    # ref_time = GW deadline: anchors staleness (claims >10d old drop out)
    # and the return-date-passed rule in the reconciler.
    claims = ledger.load(gw)
    records = reconcile_gw(claims, ref_time=f"{date}T23:59:59+00:00")
    block = build_compat_gw(gw, records, registry,
                            sorted(gw_report["sources"].keys()))
    cov = block["coverage"]
    print(f"  -> {len(claims)} claims, {len(records)} reconciled players, "
          f"coverage {cov['clubs_covered']}/20"
          + (f", missing: {cov['missing']}" if cov["missing"] else ""))

    for club in BLIND_CLUBS:
        players = block["clubs"].get(club, {}).get("players", [])
        gw_report["blind_clubs"][club] = {
            "claims": len(players),
            "players": [f"{p['player']} ({p['availability']}"
                        f"{', conflict' if p['conflict'] else ''})"
                        for p in players]}
        mark = "OK " if players else "MISS"
        names = ", ".join(p["player"] for p in players[:6])
        print(f"    [{mark}] {club:<12} {len(players)} players"
              + (f": {names}" + (" ..." if len(players) > 6 else "")
                 if players else ""))

    gw_report["coverage"] = cov
    gw_report["n_claims"] = len(claims)
    gw_report["n_players"] = len(records)
    gw_report["conflicts"] = [
        {"player": r["player_raw"], "club_id": r["club_id"],
         "score": r["score"], "sources": r["source_names"]}
        for r in records if r["conflict"]]
    report["gameweeks"][str(gw)] = gw_report
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gws", default="12,22,38",
                    help="comma list of GWs, or 'all' for 1-38")
    args = ap.parse_args()

    os.makedirs(VAL_DIR, exist_ok=True)
    session = requests.Session()
    gw_dates = load_deadlines(session)
    print(f"Deadline dates available for {len(gw_dates)} GWs")

    if args.gws.strip().lower() == "all":
        gws = sorted(gw_dates)
    else:
        gws = [int(x) for x in args.gws.split(",")]
        bad = [g for g in gws if g not in gw_dates]
        if bad:
            ap.error(f"no deadline date for GWs {bad}; "
                     f"available: {sorted(gw_dates)}")

    print("Loading identity reference (2025-26 fpl_live.json)...")
    registry, resolver = load_reference(prefer_live_api=False)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "purpose": ("off-season validation of scraper v2 steps 1-2 "
                          "against archived 2025-26 content"),
              "gameweeks": {}}
    # keep previously validated GWs in the compat output across runs
    out = {"generated_by": "validate_scraper_v2 (archived 2025-26)",
           "gameweeks": {}}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            out = json.load(f)
    for gw in gws:
        block = validate_gw(gw, session, registry, resolver, report, gw_dates)
        out["gameweeks"][str(gw)] = block

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # --- Summary ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BLIND-CLUB RECOVERY SUMMARY (old scraper: 0 claims ever)")
    print("=" * 60)
    for club in BLIND_CLUBS:
        per_gw = [f"GW{g}:{report['gameweeks'][str(g)]['blind_clubs'][club]['claims']}"
                  for g in gws]
        total = sum(report["gameweeks"][str(g)]["blind_clubs"][club]["claims"]
                    for g in gws)
        print(f"  {club:<12} {'RECOVERED' if total else 'STILL MISSING':<14} "
              + "  ".join(per_gw))
    print(f"\nSaved -> {OUT_PATH}")
    print(f"Saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
