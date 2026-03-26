"""
Intel 01: FPL Live Data Snapshot
FPL AI Thesis -- real-time player status, injuries, transfers, price changes.

Fetches from FPL public API and writes data/intel/fpl_live.json.
No external dependencies beyond requests + standard library.
"""

import os
import sys
import json
import time
import datetime

# Force UTF-8 output (Windows safe)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEL_DIR = os.path.join(ROOT, "data", "intel")
OUT_PATH  = os.path.join(INTEL_DIR, "fpl_live.json")

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL  = "https://fantasy.premierleague.com/api/fixtures/?future=1"
LIVE_URL      = "https://fantasy.premierleague.com/api/event/{gw}/live/"

HEADERS  = {"User-Agent": "Mozilla/5.0"}
TIMEOUT  = 15
RETRY_WAIT = 5

# ---------------------------------------------------------------------------
# Historical GW range
# ---------------------------------------------------------------------------
GW_START = 1
GW_END   = 28

# ---------------------------------------------------------------------------
# Position map
# ---------------------------------------------------------------------------
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# ===========================================================================
# HTTP helpers
# ===========================================================================
def fetch_json(url, label=""):
    """
    Fetch JSON from url with one retry on failure.
    Returns (data, elapsed_seconds) or (None, elapsed_seconds).
    """
    for attempt in range(2):
        t0 = time.time()
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0
            return data, elapsed
        except Exception as e:
            elapsed = time.time() - t0
            if attempt == 0:
                print(f"  [{label}] Attempt 1 failed ({e}), retrying in {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
            else:
                print(f"  [{label}] Both attempts failed ({e}). Continuing without this data.")
                return None, elapsed
    return None, 0.0


# ===========================================================================
# Transfer / ownership signal helpers
# ===========================================================================
def transfer_pressure(net):
    if net is None:
        return "unknown"
    if net < -50_000:
        return "selling"
    if net > 50_000:
        return "buying"
    return "stable"


def price_direction(change):
    if change is None:
        return "unknown"
    if change > 0:
        return "rising"
    if change < 0:
        return "falling"
    return "stable"


def ownership_tier(pct):
    if pct is None:
        return "unknown"
    if pct > 30.0:
        return "template"
    if pct > 10.0:
        return "popular"
    if pct < 5.0:
        return "differential"
    return "unknown"


# ===========================================================================
# Historical GW live data
# ===========================================================================
def fetch_historical_gws(gw_start, gw_end):
    """
    Fetch /event/{gw}/live/ for each GW in range and return per-GW stats.
    Returns {gw_str: {player_id_str: {total_points, minutes, goals_scored,
                                      assists, clean_sheets, saves, bonus}}}.
    Skips GWs that fail (network error or future GW with no data).
    """
    results = {}
    print(f"\nFetching historical GW live data (GW{gw_start}-{gw_end})...")
    for gw in range(gw_start, gw_end + 1):
        url = LIVE_URL.format(gw=gw)
        data, elapsed = fetch_json(url, label=f"GW{gw}")
        if data is None:
            print(f"  GW{gw}: fetch failed — skipping")
            continue
        elements = data.get("elements", [])
        if not elements:
            print(f"  GW{gw}: no data yet — stopping")
            break
        gw_players = {}
        for elem in elements:
            pid   = str(int(elem["id"]))
            stats = elem.get("stats", {})
            gw_players[pid] = {
                "total_points":  int(stats.get("total_points",  0)),
                "minutes":       int(stats.get("minutes",       0)),
                "goals_scored":  int(stats.get("goals_scored",  0)),
                "assists":       int(stats.get("assists",        0)),
                "clean_sheets":  int(stats.get("clean_sheets",  0)),
                "saves":         int(stats.get("saves",          0)),
                "bonus":         int(stats.get("bonus",          0)),
            }
        results[str(gw)] = gw_players
        print(f"  GW{gw}: {len(gw_players)} players ({elapsed:.1f}s)")
        time.sleep(0.3)   # polite rate limit
    return results


# ===========================================================================
# Main fetch + build
# ===========================================================================
def build_snapshot():
    os.makedirs(INTEL_DIR, exist_ok=True)

    fetched_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    # ------------------------------------------------------------------
    # 1. Bootstrap
    # ------------------------------------------------------------------
    print("Fetching Bootstrap Static...")
    boot_data, boot_elapsed = fetch_json(BOOTSTRAP_URL, label="Bootstrap")

    if boot_data is None:
        print("  FATAL: Bootstrap fetch failed. Cannot continue.")
        return None

    elements    = boot_data.get("elements", [])
    teams_raw   = boot_data.get("teams", [])
    events_raw  = boot_data.get("events", [])

    # Identify current GW (first event with is_current=True, else highest finished)
    current_gw   = None
    next_deadline = None
    for ev in events_raw:
        if ev.get("is_current"):
            current_gw    = ev["id"]
            next_deadline = ev.get("deadline_time", "")
            break
    if current_gw is None:
        # Fall back: highest finished event
        finished = [ev for ev in events_raw if ev.get("finished")]
        if finished:
            ev           = max(finished, key=lambda e: e["id"])
            current_gw   = ev["id"]
            next_deadline = ev.get("deadline_time", "")
        else:
            current_gw   = 1
            next_deadline = ""

    # Also grab the NEXT event deadline if available
    for ev in events_raw:
        if ev.get("is_next"):
            next_deadline = ev.get("deadline_time", next_deadline)
            break

    print(f"  Bootstrap: fetched in {boot_elapsed:.1f}s "
          f"({len(elements)} players, {len(teams_raw)} teams)")
    print(f"  Current GW: {current_gw}  |  Next deadline: {next_deadline}")

    # Build team lookup: id -> dict
    team_lookup = {}
    for t in teams_raw:
        tid = int(t["id"])
        team_lookup[tid] = {
            "name":                   str(t.get("name", "")),
            "short_name":             str(t.get("short_name", "")),
            "strength_overall_home":  int(t.get("strength_overall_home", 0)),
            "strength_overall_away":  int(t.get("strength_overall_away", 0)),
            "strength_attack_home":   int(t.get("strength_attack_home", 0)),
            "strength_attack_away":   int(t.get("strength_attack_away", 0)),
            "strength_defence_home":  int(t.get("strength_defence_home", 0)),
            "strength_defence_away":  int(t.get("strength_defence_away", 0)),
            "upcoming_fixtures":      [],
        }

    # ------------------------------------------------------------------
    # 2. Fixtures
    # ------------------------------------------------------------------
    print("\nFetching upcoming fixtures...")
    fix_data, fix_elapsed = fetch_json(FIXTURES_URL, label="Fixtures")

    # Build per-team fixture list (next 5 GWs from current_gw onwards)
    max_gw_offset = 5
    fixtures_count = 0
    if fix_data:
        for fix in fix_data:
            gw = fix.get("event")
            if gw is None:
                continue
            if gw < current_gw or gw > current_gw + max_gw_offset - 1:
                continue
            home_tid = int(fix.get("team_h", 0))
            away_tid = int(fix.get("team_a", 0))
            fdr_h    = int(fix.get("team_h_difficulty", 3))
            fdr_a    = int(fix.get("team_a_difficulty", 3))

            if home_tid in team_lookup:
                opp_short = team_lookup.get(away_tid, {}).get("short_name", "?")
                team_lookup[home_tid]["upcoming_fixtures"].append({
                    "gw":           gw,
                    "opponent_id":  away_tid,
                    "opponent_short": opp_short,
                    "is_home":      True,
                    "fdr":          fdr_h,
                })
                fixtures_count += 1

            if away_tid in team_lookup:
                opp_short = team_lookup.get(home_tid, {}).get("short_name", "?")
                team_lookup[away_tid]["upcoming_fixtures"].append({
                    "gw":           gw,
                    "opponent_id":  home_tid,
                    "opponent_short": opp_short,
                    "is_home":      False,
                    "fdr":          fdr_a,
                })
                fixtures_count += 1

        # Sort each team's fixture list by GW
        for tid in team_lookup:
            team_lookup[tid]["upcoming_fixtures"].sort(key=lambda f: f["gw"])

        print(f"  Fixtures: fetched in {fix_elapsed:.1f}s ({fixtures_count} upcoming fixtures)")
    else:
        print(f"  Fixtures: fetch failed in {fix_elapsed:.1f}s (0 upcoming fixtures)")

    # ------------------------------------------------------------------
    # 3. Live GW endpoint (bonus points etc.) — optional, non-fatal
    # ------------------------------------------------------------------
    print(f"\nFetching live GW{current_gw} data...")
    live_url  = LIVE_URL.format(gw=current_gw)
    live_data, live_elapsed = fetch_json(live_url, label="Live GW")

    live_pts_map = {}   # player_id -> live total_points (may override bootstrap season total)
    if live_data:
        for elem in live_data.get("elements", []):
            pid  = int(elem["id"])
            pts  = elem.get("stats", {}).get("total_points", None)
            if pts is not None:
                live_pts_map[pid] = int(pts)
        print(f"  Live GW: fetched in {live_elapsed:.1f}s "
              f"({len(live_pts_map)} player live scores)")
    else:
        print(f"  Live GW: fetch failed in {live_elapsed:.1f}s (skipping live pts)")

    # ------------------------------------------------------------------
    # 4. Build players dict
    # ------------------------------------------------------------------
    print("\nBuilding player snapshot...")

    players_out  = {}
    alerts_injured   = []
    alerts_doubtful  = []
    alerts_suspended = []
    alerts_price_rising  = []
    alerts_price_falling = []
    alerts_mass_sell     = []
    alerts_mass_buy      = []

    for e in elements:
        pid   = int(e["id"])
        tid   = int(e.get("team", 0))
        t_info = team_lookup.get(tid, {})

        # Basic identity
        web_name   = str(e.get("web_name", ""))
        first_name = str(e.get("first_name", ""))
        last_name  = str(e.get("second_name", ""))
        full_name  = f"{first_name} {last_name}".strip()
        pos_id     = int(e.get("element_type", 0))
        position   = POS_MAP.get(pos_id, "UNK")

        # Cost
        now_cost        = e.get("now_cost", 0)
        price           = round(int(now_cost) / 10.0, 1)
        cost_chg_event  = int(e.get("cost_change_event", 0) or 0)
        cost_chg_start  = int(e.get("cost_change_start", 0) or 0)

        # Status
        status    = str(e.get("status", "a"))
        chance_nr = e.get("chance_of_playing_next_round", None)
        chance_tr = e.get("chance_of_playing_this_round", None)
        news_text = str(e.get("news", "") or "").strip()
        news_added= str(e.get("news_added", "") or "").strip()

        # Transfers
        tr_in  = int(e.get("transfers_in_event",  0) or 0)
        tr_out = int(e.get("transfers_out_event", 0) or 0)
        net_tr = tr_in - tr_out

        # Ownership
        sel_str = str(e.get("selected_by_percent", "0") or "0")
        try:
            ownership = float(sel_str)
        except ValueError:
            ownership = 0.0

        # Form / stats
        form_str = str(e.get("form", "0") or "0")
        try:
            form_val = float(form_str)
        except ValueError:
            form_val = 0.0

        ppg_str = str(e.get("points_per_game", "0") or "0")
        try:
            ppg_val = float(ppg_str)
        except ValueError:
            ppg_val = 0.0

        total_pts = int(e.get("total_points", 0) or 0)

        # Signals
        tp  = transfer_pressure(net_tr)
        pd_ = price_direction(cost_chg_event)
        ot  = ownership_tier(ownership)

        players_out[str(pid)] = {
            "web_name":           web_name,
            "full_name":          full_name,
            "team_id":            tid,
            "team_short":         t_info.get("short_name", ""),
            "position":           position,
            "price":              price,
            "status":             status,
            "chance_next":        int(chance_nr) if chance_nr is not None else None,
            "chance_this":        int(chance_tr) if chance_tr is not None else None,
            "news":               news_text,
            "news_added":         news_added,
            "transfers_in":       tr_in,
            "transfers_out":      tr_out,
            "net_transfers":      net_tr,
            "transfer_pressure":  tp,
            "ownership_pct":      round(ownership, 1),
            "ownership_tier":     ot,
            "price_direction":    pd_,
            "cost_change_event":  cost_chg_event,
            "cost_change_start":  cost_chg_start,
            "form":               round(form_val, 1),
            "points_per_game":    round(ppg_val, 1),
            "total_points":       total_pts,
        }

        # Alerts
        if status == "i":
            alerts_injured.append(web_name)
        elif status == "d":
            alerts_doubtful.append(web_name)
        elif status in ("s", "u"):
            alerts_suspended.append(web_name)

        if pd_ == "rising":
            alerts_price_rising.append(web_name)
        elif pd_ == "falling":
            alerts_price_falling.append(web_name)

        if net_tr < -100_000:
            alerts_mass_sell.append(web_name)
        if net_tr > 100_000:
            alerts_mass_buy.append(web_name)

    # Sort alert lists for deterministic output
    for lst in (alerts_injured, alerts_doubtful, alerts_suspended,
                alerts_price_rising, alerts_price_falling,
                alerts_mass_sell, alerts_mass_buy):
        lst.sort()

    # ------------------------------------------------------------------
    # 5. Historical GW live data (GW1-28)
    # ------------------------------------------------------------------
    historical_gws = fetch_historical_gws(GW_START, GW_END)

    # ------------------------------------------------------------------
    # 6. Assemble output
    # ------------------------------------------------------------------
    snapshot = {
        "fetched_at":     fetched_at,
        "current_gw":     current_gw,
        "next_deadline":  next_deadline,
        "gw_range":       {"start": GW_START, "end": GW_END},
        "total_players":  len(players_out),
        "teams":          {str(k): v for k, v in team_lookup.items()},
        "players":        players_out,
        "alerts": {
            "injured":      alerts_injured,
            "doubtful":     alerts_doubtful,
            "suspended":    alerts_suspended,
            "price_rising": alerts_price_rising,
            "price_falling": alerts_price_falling,
            "mass_sell":    alerts_mass_sell,
            "mass_buy":     alerts_mass_buy,
        },
        "historical_gws": historical_gws,
    }

    return snapshot


# ===========================================================================
# Console report (Unicode box-drawing safe — Windows terminal may need UTF-8)
# ===========================================================================
def print_report(snap):
    gw       = snap["current_gw"]
    deadline = snap.get("next_deadline", "")
    fetched  = snap.get("fetched_at", "")
    alerts   = snap["alerts"]
    players  = snap["players"]
    teams    = snap["teams"]

    # Friendly timestamp
    try:
        dt = datetime.datetime.fromisoformat(fetched.rstrip("Z"))
        ts_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        ts_str = fetched

    try:
        dl = datetime.datetime.fromisoformat(str(deadline).rstrip("Z"))
        dl_str = dl.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        dl_str = str(deadline)

    hist = snap.get("historical_gws", {})
    gw_range = snap.get("gw_range", {})

    print()
    print("  +==========================================+")
    print(f"  |   FPL LIVE INTELLIGENCE -- GW{gw:<3}          |")
    print(f"  |   Historical: GW{gw_range.get('start',1)}-GW{gw_range.get('end',28)} ({len(hist)} GWs fetched)  |")
    print("  +==========================================+")
    print()
    print(f"  Fetched:        {ts_str}")
    print(f"  Next deadline:  {dl_str}")
    print()

    # ---- Injury Alerts ----
    print("  INJURY ALERTS")
    injured   = alerts.get("injured",   [])
    doubtful  = alerts.get("doubtful",  [])
    suspended = alerts.get("suspended", [])

    if not injured and not doubtful and not suspended:
        print("    None")
    else:
        # Map name -> player info for quick lookup
        name_lookup = {v["web_name"]: v for v in players.values()}

        for name in injured:
            p = name_lookup.get(name, {})
            news = p.get("news", "")[:80]
            print(f"    [i] {name} ({p.get('team_short','?')}) -- {news}")

        for name in doubtful:
            p    = name_lookup.get(name, {})
            cop  = p.get("chance_next", None)
            cop_str = f"chance: {cop}%" if cop is not None else ""
            news = p.get("news", "")[:70]
            line = f"    [d] {name} ({p.get('team_short','?')})"
            if cop_str:
                line += f" -- {cop_str}"
            if news:
                line += f" -- {news}"
            print(line)

        for name in suspended:
            p    = name_lookup.get(name, {})
            st   = p.get("status", "")
            news = p.get("news", "")[:70]
            print(f"    [s] {name} ({p.get('team_short','?')}) [{st.upper()}] -- {news}")

    print()

    # ---- Transfer Movements ----
    print("  TRANSFER MOVEMENTS (top 10 each)")
    name_lookup = {v["web_name"]: (k, v) for k, v in players.items()}

    # Sort by net transfers
    all_net = sorted(
        [(v["web_name"], v["net_transfers"]) for v in players.values()],
        key=lambda x: x[1],
        reverse=True,
    )
    most_bought = [f"{n} +{t//1000:.0f}k" for n, t in all_net[:10] if t > 0]
    most_sold   = [f"{n} {t//1000:.0f}k"  for n, t in all_net[-10:][::-1] if t < 0]

    print("  Most bought:  " + " | ".join(most_bought) if most_bought else "  Most bought:  None")
    print("  Most sold:    " + " | ".join(most_sold)   if most_sold   else "  Most sold:    None")

    mass_sell = alerts.get("mass_sell", [])
    if mass_sell:
        print(f"  Mass sells (>100k out): {', '.join(mass_sell)}")

    mass_buy = alerts.get("mass_buy", [])
    if mass_buy:
        print(f"  Mass buys  (>100k in):  {', '.join(mass_buy)}")

    print()

    # ---- Price Changes ----
    print("  PRICE CHANGES")
    rising  = alerts.get("price_rising",  [])
    falling = alerts.get("price_falling", [])

    def price_line(names):
        parts = []
        for n in names[:10]:
            p      = players.get(
                next((k for k, v in players.items() if v["web_name"] == n), ""), {}
            )
            price  = p.get("price", 0.0)
            chg    = p.get("cost_change_event", 0)
            # cost_change_event is in tenths (e.g. +1 = +0.1m)
            old_p  = round(price - chg * 0.1, 1)
            parts.append(f"{n} (GBP{old_p:.1f}m -> GBP{price:.1f}m)")
        return " | ".join(parts) if parts else "None"

    print(f"  Rising:  {price_line(rising)}")
    print(f"  Falling: {price_line(falling)}")
    print()

    # ---- Summary ----
    print("  SUMMARY")
    print(f"  Total players tracked:  {snap['total_players']}")
    print(f"  Injured/unavailable:    {len(injured) + len(suspended)}")
    print(f"  Doubtful:               {len(doubtful)}")
    print(f"  Price risers:  {len(rising)} | Price fallers: {len(falling)}")
    print()
    print(f"  Historical GWs saved: {len(hist)} ({gw_range.get('start',1)}-{gw_range.get('end',28)})")
    print(f"  Saved -> data/intel/fpl_live.json")
    print()


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    print()
    print("  FPL Live Intel -- Stage Intel 01")
    print("  " + "-" * 42)

    snap = build_snapshot()
    if snap is None:
        print("Snapshot failed.")
        return

    # Save
    os.makedirs(INTEL_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)

    print_report(snap)


if __name__ == "__main__":
    main()
