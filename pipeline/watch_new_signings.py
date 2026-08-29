#!/usr/bin/env python3
"""
Watch for new-to-PL signings and, when one appears, run the full pipeline
automatically: re-fetch FPL API -> identify new players -> scrape their
previous-league stats (Stage 4a) -> rebuild training data (Stage 6) ->
predict their expected points -> log + notify.

Designed to run unattended on a timer (Windows Task Scheduler). Cheap on a
no-op run (just re-fetches bootstrap-static and diffs against the ledger);
only pays the scraping/rebuild cost when something actually changed.

Ledger: data/raw/transfers/watch_ledger.json — fpl_id -> {name, team, first_seen}
for every new-to-PL player already processed by a past run of this script.
Anyone in the live FPL API's "new to PL" set but NOT in the ledger is new
since last run.

Usage:
    python pipeline/watch_new_signings.py                 # run once
    python pipeline/watch_new_signings.py --dry-run        # detect only, no scrape/rebuild/write
Task Scheduler wiring: see docs printed by --setup-task-scheduler.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPL_API_DIR   = os.path.join(BASE_DIR, "data", "raw", "fpl_api")
VAASTAV_DIR   = os.path.join(BASE_DIR, "data", "raw", "vaastav")
TRANSFERS_DIR = os.path.join(BASE_DIR, "data", "raw", "transfers")
TM_CACHE_DIR  = os.path.join(TRANSFERS_DIR, "raw_html_cache")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR       = os.path.join(BASE_DIR, "data", "intel", "signing_watch")

LEDGER_PATH = os.path.join(TRANSFERS_DIR, "watch_ledger.json")

sys.path.insert(0, BASE_DIR)
# new_signings_stage4a wraps sys.stdout for UTF-8 on Windows as an import
# side effect -- must not be re-wrapped here (double-wrap closes the buffer).
import pandas as pd  # noqa: E402
from pipeline.new_signings_stage4a import (  # noqa: E402
    normalize_name, step1_identify_new_signings,
)

os.makedirs(LOG_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_ledger():
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ledger(ledger):
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)


def notify(title, message):
    """Best-effort local notification: Windows toast (no extra deps beyond
    what ships with Windows 10/11) + a plain-text alert file as a durable
    fallback that doesn't depend on toast support/focus-assist settings."""
    alert_path = os.path.join(LOG_DIR, "LATEST_ALERT.txt")
    with open(alert_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n{'=' * len(title)}\n{message}\n"
                f"\nWritten: {datetime.now().isoformat()}\n")
    if sys.platform == "win32":
        ps_script = f'''
$ErrorActionPreference = "SilentlyContinue"
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$textNodes = $template.GetElementsByTagName("text")
$textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) | Out-Null
$textNodes.Item(1).AppendChild($template.CreateTextNode("{message[:180]}")) | Out-Null
$toast = [Windows.UI.Notifications.ToastNotification]::new($template)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("FPL AI Watcher").Show($toast)
'''
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_script],
                            capture_output=True, timeout=15)
        except Exception as e:
            log(f"  [WARN] toast notification failed (non-fatal): {e}")


def run_step(cmd, cwd=BASE_DIR, input_text=None, timeout=1800):
    log(f"  $ {' '.join(cmd)}")
    # Explicit UTF-8: Windows' default subprocess text encoding (cp1252) chokes
    # on the accented player names (Gyokeres, etc.) these scripts print.
    proc = subprocess.run(cmd, cwd=cwd, input=input_text, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
                            f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
                            f"--- stderr tail ---\n{proc.stderr[-3000:]}")
    # Log a tail even on success -- otherwise diagnostic detail (TM league
    # breakdown, FBref "no data found" lists, etc.) is silently discarded
    # and any "why didn't this player get scraped" question needs a manual
    # re-run to answer.
    if proc.stdout.strip():
        log(f"  --- output tail ---\n{proc.stdout[-2000:]}")
    return proc.stdout


def detect_new_players():
    """Re-fetch FPL API, diff against vaastav + the ledger. Returns
    (new_players: list[dict], all_new_to_pl_ids: set, ledger) without
    scraping. Uses Stage 4a's own step1_identify_new_signings() (not a
    reimplementation) so this stays exactly consistent with the false-
    positive overrides and multi-strategy fuzzy matching that function
    already handles (e.g. a real signing whose name happens to fuzzy-match
    an unrelated historical player)."""
    log("Refreshing FPL API bootstrap data (Stage 1) ...")
    run_step([sys.executable, "-u", "pipeline/data_fetcher_stage1.py"], timeout=900)

    log("Identifying new-to-PL players (Stage 4a step 1) ...")
    new_to_pl = step1_identify_new_signings()

    ledger = load_ledger()
    new_players = []
    for _, p in new_to_pl.iterrows():
        pid = str(int(p["fpl_id"]))
        if pid not in ledger:
            new_players.append({
                "fpl_id": int(p["fpl_id"]), "name": p["fpl_name"],
                "team": p["fpl_team"], "position": p["fpl_position"],
                "price": float(p["fpl_price"]),
            })
    return new_players, set(new_to_pl["fpl_id"].astype(int).tolist()), ledger


def predict_new_player_points(name, fpl_position, fpl_price):
    """Debut-quality estimate for a player with ZERO real PL minutes yet.

    Genuine debutants never get a standalone row in train_{pos}.csv --
    Stage 6's build_prev_lookup only uses new_signings_{pos}.csv rows to
    enrich a player's REAL first-PL-season rows, and a brand-new signing
    has no real row to attach to yet (the season hasn't started/they
    haven't played). So this builds the feature vector directly from their
    Stage 4a scrape (new_signings_{pos}.csv, reliability-weighted across
    however many prior seasons were found -- mirrors build_prev_lookup's
    own weighting) and predicts on that, with neutral fixture context
    (was_home=0.5, fdr=3.0 -- the model has never seen a "no fixture yet"
    state, this is the closest neutral stand-in) since there's no real
    fixture to condition on before a ball is kicked.

    NOTE: if this player already has real vaastav history (e.g. Stage 4a's
    VAASTAV_FALSE_POSITIVES override stale-flagged an established player as
    "new" again), this will UNDER-estimate them -- it deliberately ignores
    any real historical rows and treats every input as a zero-PL-history
    debutant. Cross-check the caller's "already have vaastav history"
    signal before trusting this for a player who's played PL minutes before.
    """
    import lightgbm as lgb

    LGBM_PARAMS = dict(
        n_estimators=200, max_depth=3, learning_rate=0.04390698211469097,
        num_leaves=31, subsample=0.9321213778928387, colsample_bytree=0.8240721576385306,
        min_child_samples=27, random_state=42, verbosity=-1
    )
    FEAT_COLS = [
        "form_last3", "form_last5", "avg_points_per_game",
        "minutes_reliability", "goals_per_game", "assists_per_game",
        "clean_sheet_rate", "saves_per_game",
        "value", "was_home", "fdr",
    ]
    COL_ALIAS = {
        "avg_points_per_game_season": "avg_points_per_game",
        "minutes_reliability_season": "minutes_reliability",
        "goals_per_game_season":      "goals_per_game",
        "assists_per_game_season":    "assists_per_game",
        "clean_sheet_rate_season":    "clean_sheet_rate",
        "saves_per_game_season":      "saves_per_game",
        "current_gw_fdr":             "fdr",
        "home_advantage":             "was_home",
    }
    TRAIN_FILES = {"GK": "train_gk.csv", "DEF": "train_def.csv",
                   "MID": "train_mid.csv", "FWD": "train_fwd.csv"}
    SIGN_FILES = {"GK": "new_signings_gk.csv", "DEF": "new_signings_def.csv",
                  "MID": "new_signings_mid.csv", "FWD": "new_signings_fwd.csv"}

    pos = fpl_position if fpl_position in TRAIN_FILES else None
    if pos is None:
        return None

    sign_path = os.path.join(BASE_DIR, "data", "raw", "fbref", "new_signings", SIGN_FILES[pos])
    if not os.path.exists(sign_path):
        return None
    sigs = pd.read_csv(sign_path, low_memory=False)
    target_norm = normalize_name(name)
    rows = sigs[sigs["name"].apply(lambda n: normalize_name(str(n)) == target_norm)]
    if rows.empty:
        return None

    w = rows["season_reliability"].fillna(0.0) if "season_reliability" in rows.columns else pd.Series([1.0] * len(rows))
    def wavg(col):
        if col not in rows.columns or w.sum() == 0:
            return 0.0
        return float((rows[col] * w).sum() / w.sum())

    feat_row = {
        "form_last3": 0.0, "form_last5": 0.0, "avg_points_per_game": 0.0,
        "minutes_reliability": wavg("minutes_reliability_season"),
        "goals_per_game": wavg("goals_per_game_season"),
        "assists_per_game": wavg("assists_per_game_season"),
        "clean_sheet_rate": wavg("clean_sheet_rate_season"),
        "saves_per_game": wavg("saves_per_game_season"),
        "value": float(fpl_price), "was_home": 0.5, "fdr": 3.0,
    }

    train_path = os.path.join(PROCESSED_DIR, TRAIN_FILES[pos])
    if not os.path.exists(train_path):
        return None
    df = pd.read_csv(train_path, low_memory=False)
    df = df.rename(columns=COL_ALIAS)
    df = df.loc[:, ~df.columns.duplicated()]
    X_all = df.reindex(columns=FEAT_COLS, fill_value=0.0).fillna(0.0).values
    y_all = df["total_points"].values
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_all, y_all)

    X_new = pd.DataFrame([feat_row])[FEAT_COLS].values
    pred = float(model.predict(X_new)[0])
    return {"position": pos, "predicted_points": round(pred, 2),
            "seasons_used": int((w > 0).sum()),
            "note": "neutral-fixture debut estimate (prev-league-adjusted "
                    "features, reliability-weighted across scraped seasons, "
                    "no real fixture context yet -- was_home/fdr held neutral)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="Detect only -- skip scraping/rebuild/prediction/ledger update")
    args = ap.parse_args()

    log("=== New-signing watcher run start ===")
    new_players, all_new_ids, ledger = detect_new_players()

    if not new_players:
        log("No new-to-PL players since last run. Nothing to do.")
        return

    log(f"Found {len(new_players)} new player(s) since last run:")
    for p in new_players:
        log(f"  {p['name']} ({p['team']}, {p['position']}, £{p['price']})")

    if args.dry_run:
        log("[--dry-run] Stopping before scrape/rebuild/predict/ledger update.")
        return

    # ── Stage 4a: scrape previous-league stats ──────────────────────────────
    # The Transfermarkt HTML cache is NOT season-keyed -- clear it so an
    # ongoing watcher always sees the current transfer window, not a stale
    # cached page from whenever it was first populated.
    log("Clearing Transfermarkt page cache (not season-keyed; must be fresh) ...")
    for fname in os.listdir(TM_CACHE_DIR) if os.path.exists(TM_CACHE_DIR) else []:
        os.remove(os.path.join(TM_CACHE_DIR, fname))

    log("Running Stage 4a (Transfermarkt + FBref scrape) ...")
    run_step([sys.executable, "-u", "pipeline/new_signings_stage4a.py", "--full-run"],
              timeout=1800)

    # ── Stage 6: rebuild training data (auto-confirm all internal gates) ───
    log("Rebuilding training data (Stage 6) ...")
    answers = "\n".join(["y"] * 8) + "\n"
    run_step([sys.executable, "-u", "pipeline/feature_engineering_stage6.py"],
              input_text=answers, timeout=900)

    # ── Predict + report ─────────────────────────────────────────────────────
    results = []
    for p in new_players:
        pred = predict_new_player_points(p["name"], p["position"], p["price"])
        results.append({**p, "prediction": pred})
        if pred:
            log(f"  {p['name']}: predicted {pred['predicted_points']} pts "
                f"({pred['position']}, debut estimate)")
        else:
            log(f"  {p['name']}: no prediction (no scraped row found -- "
                f"check Stage 4a's 'no FBref data found' list)")

    # ── Update ledger (all new-to-PL players seen this run, not just brand new) ──
    now = datetime.now(timezone.utc).isoformat()
    for pid in all_new_ids:
        if str(pid) not in ledger:
            ledger[str(pid)] = {"first_seen": now}
    save_ledger(ledger)

    # ── Persist a dated report + notify ─────────────────────────────────────
    report_path = os.path.join(LOG_DIR, f"signing_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"detected_at": now, "players": results}, f, indent=2)

    names = ", ".join(f"{r['name']} ({r['team']})" for r in results)
    msg = f"New signing(s) processed: {names}. Training data rebuilt. See {report_path}"
    log(msg)
    notify("FPL AI: New Signing Detected", msg)
    log("=== Run complete ===")


if __name__ == "__main__":
    main()
