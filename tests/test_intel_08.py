"""
tests/test_intel_08.py — top-10k effective-ownership scraper tests
(recommendation-layer §4.1).

Covers: snapshot build from cohort maps + bootstrap enrichment, the canonical
`eo` = top-10k rule, EO>100% (captaincy) preserved, GW tagging
(current/next/fallback/override), players kept even when bootstrap lacks the
element id (season id churn), the differential/template/eo_of query helpers,
graceful degradation without bootstrap, and save + per-GW archive to disk.

No pytest dependency. Run directly:
  docker run --rm -v "<repo>:/app" -w /app fpl-scrape python tests/test_intel_08.py
Exit code 0 = all pass.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
from intel_08_effective_ownership import (build_snapshot, resolve_gw, save,
                                          eo_of, differentials, template)

# element id -> EO fraction (top10k richer than elite for id 449; 449 > 100%)
EO = {
    "top10k": {"449": 1.372, "5": 0.89, "624": 0.79, "999": 0.02, "7": 0.05},
    "elite":  {"449": 1.136, "5": 0.85, "624": 0.91, "999": 0.01},
}

BOOTSTRAP = {
    "events": [{"id": 37, "is_current": False, "is_next": False,
                "finished": True},
               {"id": 38, "is_current": True, "is_next": False,
                "finished": False}],
    "teams": [{"id": 14, "short_name": "MUN"}, {"id": 1, "short_name": "ARS"},
              {"id": 19, "short_name": "WHU"}],
    "elements": [
        {"id": 449, "web_name": "B.Fernandes", "team": 14, "element_type": 3,
         "now_cost": 90, "selected_by_percent": "48.0"},
        {"id": 5, "web_name": "Gabriel", "team": 1, "element_type": 2,
         "now_cost": 63, "selected_by_percent": "45.4"},
        {"id": 624, "web_name": "Bowen", "team": 19, "element_type": 3,
         "now_cost": 78, "selected_by_percent": "17.5"},
        {"id": 7, "web_name": "Saliba", "team": 1, "element_type": 2,
         "now_cost": 62, "selected_by_percent": "9.3"},
        # note: id 999 deliberately absent from bootstrap (id churn case)
    ],
}


def test_canonical_eo_is_top10k():
    snap = build_snapshot(EO, BOOTSTRAP)
    p = snap["players"]["624"]
    assert p["eo"] == 0.79, p           # top-10k, not elite's 0.91
    assert p["eo_top10k"] == 0.79
    assert p["eo_elite"] == 0.91


def test_eo_over_100_percent_preserved():
    snap = build_snapshot(EO, BOOTSTRAP)
    assert snap["players"]["449"]["eo"] == 1.372, "captaincy EO clipped"


def test_enrichment_attached():
    snap = build_snapshot(EO, BOOTSTRAP)
    p = snap["players"]["449"]
    assert p["web_name"] == "B.Fernandes"
    assert p["team"] == "MUN"
    assert p["position"] == "MID"
    assert p["price"] == 9.0
    assert abs(p["eo_overall"] - 0.48) < 1e-9
    assert snap["enriched"] is True


def test_player_missing_from_bootstrap_kept():
    # id 999 has EO but no bootstrap element -> kept, enrichment blank
    snap = build_snapshot(EO, BOOTSTRAP)
    assert "999" in snap["players"], "EO row dropped for unknown element id"
    assert snap["players"]["999"]["eo"] == 0.02
    assert snap["players"]["999"].get("web_name") in (None, "")


def test_falls_back_to_elite_when_top10k_missing():
    eo = {"top10k": {"5": 0.5}, "elite": {"5": 0.4, "600": 0.3}}
    snap = build_snapshot(eo, None)
    assert snap["players"]["600"]["eo"] == 0.3         # elite fills the gap
    assert snap["players"]["600"]["eo_top10k"] is None


def test_gw_tagging_current_next_fallback_override():
    assert resolve_gw(BOOTSTRAP) == (38, "current")
    nxt = {"events": [{"id": 1, "is_current": False, "is_next": True}]}
    assert resolve_gw(nxt) == (1, "next")
    fb = {"events": [{"id": 5, "is_current": False, "is_next": False},
                     {"id": 6, "is_current": False, "is_next": False}]}
    assert resolve_gw(fb) == (6, "fallback")
    assert resolve_gw(None) == (None, "unknown")
    snap = build_snapshot(EO, BOOTSTRAP, gw_override=22)
    assert snap["gw"] == 22 and snap["gw_basis"] == "override"


def test_degrades_without_bootstrap():
    snap = build_snapshot(EO, None)
    assert snap["enriched"] is False
    assert snap["gw"] is None
    assert snap["players"]["449"]["eo"] == 1.372   # raw EO still captured
    assert "web_name" not in snap["players"]["449"]


def test_eo_of_helper():
    snap = build_snapshot(EO, BOOTSTRAP)
    assert eo_of(snap, 449) == 1.372
    assert eo_of(snap, "5") == 0.89
    assert eo_of(snap, 123456) is None


def test_differentials_helper():
    snap = build_snapshot(EO, BOOTSTRAP)
    diffs = differentials(snap, max_eo=0.10)
    ids = [d["element_id"] for d in diffs]
    # id 7 (0.05) and 999 (0.02) are < 0.10; sorted most-owned first
    assert ids == ["7", "999"], ids
    # position filter
    mids = differentials(snap, max_eo=0.10, position="MID")
    assert all(d.get("position") == "MID" for d in mids)


def test_template_helper():
    snap = build_snapshot(EO, BOOTSTRAP)
    tmpl = template(snap, min_eo=0.50)
    ids = [t["element_id"] for t in tmpl]
    assert ids == ["449", "5", "624"], ids   # >=0.50, highest EO first


def test_save_writes_latest_and_archive():
    snap = build_snapshot(EO, BOOTSTRAP)          # gw 38
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "effective_ownership.json")
        hist = os.path.join(d, "eo_history")
        written = save(snap, out_path=out, history_dir=hist)
        assert os.path.exists(out)
        assert os.path.exists(os.path.join(hist, "gw38.json"))
        assert len(written) == 2
        reloaded = json.load(open(out, encoding="utf-8"))
        assert reloaded["players"]["449"]["eo"] == 1.372


def test_save_no_archive_when_gw_unknown():
    snap = build_snapshot(EO, None)               # gw None
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "effective_ownership.json")
        hist = os.path.join(d, "eo_history")
        written = save(snap, out_path=out, history_dir=hist)
        assert written == [out], "archived a snapshot with no GW tag"
        assert not os.path.exists(hist)


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
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
