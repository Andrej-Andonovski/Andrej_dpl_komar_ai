"""
pipeline/intel_08_effective_ownership.py
Top-10k effective-ownership scraper (recommendation-layer §4.1).

WHY THIS EXISTS AND WHY IT IS URGENT
The FPL API exposes only OVERALL ownership (`selected_by_percent`) and no
captaincy split. The recommendation layer's captain matrix (§3.2),
differentials (§3.5) and template comparison (§3.7) all need *rank-cohort
effective ownership* — how heavily a player is owned AND captained inside the
top 10k. That data exists only as a live snapshot on third-party sites; it is
NOT in any FPL endpoint and CANNOT be backfilled. Every GW that passes
without a stored snapshot is a permanent hole in the EO history the shadow
season (§7.2) needs. Hence: build now, snapshot every GW, archive per GW.

SOURCE (verified live 2026-07-14)
LiveFPL's public data host `livefpl.us` serves the same JSON its SPA consumes:
  - https://livefpl.us/top10k.json   element_id -> top-10k EO fraction
  - https://livefpl.us/elite.json    element_id -> tighter "elite" cohort EO
Both are keyed by the FPL element id directly (no name resolution needed) and
carry EFFECTIVE ownership: values exceed 1.0 for heavily-captained players
(e.g. B.Fernandes 1.37 = 137% at GW38 = ~48% owned, heavily captained). That
>100% ceiling is the proof these are EO, not plain ownership. The FPL
bootstrap-static supplies the current GW (to tag the snapshot) and player
enrichment (web_name/team/position/overall ownership).

OUTPUT (never overwrites intel_01's fpl_live.json)
  data/intel/effective_ownership.json   latest snapshot
  data/intel/eo_history/gw{N}.json      per-GW archive (latest-wins per GW)

`eo` per player is the canonical top-10k figure the design requires;
`eo_elite`/`eo_overall` are carried alongside for the differential framing.

Usage (Docker — this machine has no local Python):
  docker run --rm -v "<repo>:/app" -w /app fpl-scrape \
      python pipeline/intel_08_effective_ownership.py [--gw N] [--no-archive]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEL_DIR = os.path.join(ROOT, "data", "intel")
OUT_PATH = os.path.join(INTEL_DIR, "effective_ownership.json")
HISTORY_DIR = os.path.join(INTEL_DIR, "eo_history")

TOP10K_URL = "https://livefpl.us/top10k.json"
ELITE_URL = "https://livefpl.us/elite.json"
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/126.0.0.0 Safari/537.36")}
TIMEOUT = 30

_POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _fetch_json(session, url: str) -> dict | list:
    r = session.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_eo(session, *, top10k_url=TOP10K_URL, elite_url=ELITE_URL) -> dict:
    """Return {'top10k': {id: eo}, 'elite': {id: eo}}. Elite is best-effort:
    a failure there must not lose the primary top-10k snapshot."""
    top10k = _fetch_json(session, top10k_url)
    try:
        elite = _fetch_json(session, elite_url)
    except Exception as e:                                    # noqa: BLE001
        print(f"  [intel_08] elite.json unavailable ({e}); top-10k only")
        elite = {}
    return {"top10k": top10k, "elite": elite}


def fetch_bootstrap(session, *, url=BOOTSTRAP_URL) -> dict | None:
    """FPL bootstrap-static for GW tagging + enrichment; None on failure
    (the EO snapshot is still saved raw — enrichment is convenience)."""
    try:
        return _fetch_json(session, url)
    except Exception as e:                                    # noqa: BLE001
        print(f"  [intel_08] bootstrap-static unavailable ({e}); "
              f"saving raw EO without enrichment")
        return None


# ---------------------------------------------------------------------------
# Build snapshot
# ---------------------------------------------------------------------------

def resolve_gw(bootstrap: dict | None) -> tuple[int | None, str]:
    """The GW this EO snapshot describes.

    LiveFPL serves the CURRENT live snapshot only, so the tag is the GW whose
    squads that ownership reflects: the in-progress GW if one is live, else
    the next upcoming GW (pre-deadline picks), else the latest event.
    Returns (gw, basis) — basis records which rule fired, for the archive.
    """
    if not bootstrap:
        return None, "unknown"
    events = bootstrap.get("events", [])
    cur = next((e for e in events if e.get("is_current")), None)
    if cur:
        return int(cur["id"]), "current"
    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt:
        return int(nxt["id"]), "next"
    if events:
        return int(max(events, key=lambda e: int(e["id"]))["id"]), "fallback"
    return None, "unknown"


def _enrichment(bootstrap: dict | None) -> dict:
    """element_id(str) -> {web_name, team, position, price, eo_overall}."""
    if not bootstrap:
        return {}
    teams = {int(t["id"]): t.get("short_name", "")
             for t in bootstrap.get("teams", [])}
    out = {}
    for e in bootstrap.get("elements", []):
        try:
            overall = float(e.get("selected_by_percent", 0) or 0) / 100.0
        except (TypeError, ValueError):
            overall = None
        out[str(int(e["id"]))] = {
            "web_name":  e.get("web_name", ""),
            "team":      teams.get(int(e.get("team", 0)), ""),
            "position":  _POSITIONS.get(int(e.get("element_type", 0)), ""),
            "price":     (e.get("now_cost", 0) or 0) / 10.0,
            "eo_overall": overall,
        }
    return out


def _as_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_snapshot(eo: dict, bootstrap: dict | None,
                   gw_override: int | None = None) -> dict:
    """Merge cohort EO maps + enrichment into one snapshot dict.

    `eo` = fetch_eo() output. The canonical `eo` field per player is the
    top-10k figure (design requirement); elite/overall ride alongside.
    A player present in ANY cohort is kept even if bootstrap lacks it
    (element-id churn across seasons must never silently drop a row).
    """
    top10k = eo.get("top10k", {}) or {}
    elite = eo.get("elite", {}) or {}
    enrich = _enrichment(bootstrap)

    gw, basis = resolve_gw(bootstrap)
    if gw_override is not None:
        gw, basis = gw_override, "override"

    players = {}
    for pid in set(top10k) | set(elite):
        eo_t = _as_float(top10k.get(pid))
        eo_e = _as_float(elite.get(pid))
        rec = {
            "eo":        eo_t if eo_t is not None else eo_e,
            "eo_top10k": eo_t,
            "eo_elite":  eo_e,
        }
        rec.update(enrich.get(pid, {}))
        players[pid] = rec

    return {
        "generated_at":   utcnow_iso(),
        "source":         "https://livefpl.us",
        "cohort_primary": "top10k",
        "gw":             gw,
        "gw_basis":       basis,
        "enriched":       bool(enrich),
        "n_players":      len(players),
        "players":        players,
    }


# ---------------------------------------------------------------------------
# Query helpers (consumed by the recommendation layer — §3.2/§3.5/§3.7)
# ---------------------------------------------------------------------------

def eo_of(snapshot: dict, element_id) -> float | None:
    rec = snapshot.get("players", {}).get(str(element_id))
    return rec.get("eo") if rec else None


def differentials(snapshot: dict, max_eo: float = 0.10,
                  position: str | None = None) -> list:
    """Low-EO players (< max_eo top-10k), most-owned-first — the pool §3.5
    ranks by ceiling. Filter by position ('MID' etc.) if given."""
    out = []
    for pid, r in snapshot.get("players", {}).items():
        eo = r.get("eo")
        if eo is None or eo >= max_eo:
            continue
        # a position filter must exclude unknown-position rows too — we can't
        # assert a match we don't have (unenriched / cross-season id)
        if position and r.get("position") != position:
            continue
        out.append({"element_id": pid, **r})
    return sorted(out, key=lambda r: -(r.get("eo") or 0))


def template(snapshot: dict, min_eo: float = 0.50) -> list:
    """High-EO 'template' players (>= min_eo) — the pack you're measured
    against (§3.7), highest-EO first."""
    out = [{"element_id": pid, **r}
           for pid, r in snapshot.get("players", {}).items()
           if (r.get("eo") or 0) >= min_eo]
    return sorted(out, key=lambda r: -(r.get("eo") or 0))


# ---------------------------------------------------------------------------
# Save + archive
# ---------------------------------------------------------------------------

def save(snapshot: dict, *, out_path: str = OUT_PATH,
         history_dir: str = HISTORY_DIR, archive: bool = True) -> list:
    """Write the latest snapshot and (unless disabled) the per-GW archive.
    Archive is latest-wins per GW: the meaningful figure is the final
    pre-deadline snapshot, and re-running within a GW refreshes it. Returns
    the paths written."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    written = [out_path]
    if archive and snapshot.get("gw") is not None:
        os.makedirs(history_dir, exist_ok=True)
        gw_path = os.path.join(history_dir, f"gw{snapshot['gw']}.json")
        with open(gw_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        written.append(gw_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(gw_override: int | None = None, archive: bool = True,
        verbose: bool = True) -> dict:
    session = requests.Session()
    eo = fetch_eo(session)
    bootstrap = fetch_bootstrap(session)
    snap = build_snapshot(eo, bootstrap, gw_override=gw_override)
    written = save(snap, archive=archive)
    if verbose:
        gw = snap["gw"]
        print(f"intel_08 — GW{gw} ({snap['gw_basis']}) — "
              f"{snap['n_players']} players "
              f"({'enriched' if snap['enriched'] else 'raw, no bootstrap'})")
        top = template(snap, min_eo=0.30)[:8]
        for r in top:
            name = r.get("web_name") or f"#{r['element_id']}"
            print(f"    {name:<16} EO {(r['eo'] or 0)*100:5.1f}%  "
                  f"(overall {((r.get('eo_overall') or 0))*100:4.1f}%)")
        for p in written:
            print(f"  saved -> {p}")
    return snap


def main():
    ap = argparse.ArgumentParser(description="Top-10k effective-ownership "
                                             "snapshot (intel_08)")
    ap.add_argument("--gw", type=int, default=None,
                    help="override the GW tag (default: FPL API current/next)")
    ap.add_argument("--no-archive", action="store_true",
                    help="write only the latest snapshot, skip per-GW archive")
    args = ap.parse_args()
    run(gw_override=args.gw, archive=not args.no_archive)


if __name__ == "__main__":
    main()
