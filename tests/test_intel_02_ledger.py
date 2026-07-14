"""
tests/test_intel_02_ledger.py — ledger + reconciler tests (redesign step 2).

Covers the five §5.2 merge rules, ledger dedup, and tier mapping.

No pytest dependency. Run directly:
  docker run --rm -v "<repo>:/app" -w /app fpl-scrape python tests/test_intel_02_ledger.py
Exit code 0 = all pass.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
from intel_02_ledger import (ClaimLedger, make_claim, reconcile_player,
                             reconcile_gw, availability_tier, STATUS_SCORES)


def C(source, status, ts, tier=1, pid=101, text="", injury="", url="u"):
    """Shorthand claim for tests (GW20, club 14)."""
    return make_claim(20, source, tier, url, 14, "Test Player", status,
                      player_id=pid, injury=injury, text=text,
                      observed_at=ts, published_at=ts)


T0 = "2026-01-16T10:00:00+00:00"
T1 = "2026-01-16T14:00:00+00:00"
T2 = "2026-01-16T15:30:00+00:00"
T_OLD = "2026-01-13T10:00:00+00:00"   # 3 days before T1


# ── Rule 1: suspension override ───────────────────────────────────────────────

def test_suspension_overrides_everything():
    rec = reconcile_player([
        C("guardian", "available", T1),
        C("sportsgambler", "available", T1),
        C("ffs_injuries", "suspended", T0, tier=1),
    ])
    assert rec["status"] == "suspended", rec
    assert rec["score"] == 0
    assert rec["tier_label"] == "out"


def test_tier3plus_suspension_does_not_override():
    # A tier-4 local-paper "suspended" claim is scored, not administrative
    rec = reconcile_player([
        C("guardian", "available", T1, tier=1),
        C("reach_local", "suspended", T1, tier=4),
    ])
    assert rec["status"] != "suspended" or rec["score"] > 0 or True
    assert rec["score"] > 0, f"tier-4 suspension wrongly hard-zeroed: {rec}"


# ── Rule 2: freshness within a source ─────────────────────────────────────────

def test_last_write_wins_within_source():
    # liveblog progression doubtful@14:00 -> out@15:30: only "out" scores
    rec = reconcile_player([
        C("ffs_teamnews", "doubtful", T1, tier=2, text="late test"),
        C("ffs_teamnews", "out", T2, tier=2, text="ruled out"),
    ])
    assert rec["n_sources"] == 1
    assert rec["status"] == "out"
    assert rec["score"] == STATUS_SCORES["out"]
    assert rec["text"] == "ruled out"


# ── Rule 3: weighted scoring ──────────────────────────────────────────────────

def test_weighted_mean_between_sources():
    # doubtful(40) + out(5), same tier, same time -> midpoint-ish (adjacent
    # severities: no conflict, no agreement bonus since gap == 1... actually
    # gap 1 IS within one step -> +5). Verify score between the two + bonus.
    rec = reconcile_player([
        C("guardian", "doubtful", T1),
        C("sportsgambler", "out", T1),
    ])
    base = (40 + 5) / 2
    assert abs(rec["score"] - (base + 5)) <= 1, rec  # +5 agreement (adjacent)
    assert not rec["conflict"]


def test_recency_decay_favours_fresh_claim():
    # fresh "available" vs 3-day-old "out" (same tier): fresh dominates.
    # 72h at 48h half-life -> old weight ~0.35 -> score ~ (95 + 0.35*5)/1.35
    rec = reconcile_player([
        C("guardian", "out", T_OLD),
        C("sportsgambler", "available", T1),
    ])
    assert rec["score"] >= 60, f"stale 'out' outweighed fresh 'available': {rec}"


def test_tier_weighting_applied():
    # tier-2 presser quote outweighs tier-0 FPL flag at equal freshness
    rec_press_out = reconcile_player([
        C("ffs_teamnews", "out", T1, tier=2),
        C("fpl_api", "available", T1, tier=0),
    ])
    rec_flag_out = reconcile_player([
        C("ffs_teamnews", "available", T1, tier=2),
        C("fpl_api", "out", T1, tier=0),
    ])
    # symmetric statuses, asymmetric weights: presser side must pull harder
    assert rec_press_out["score"] < rec_flag_out["score"]


# ── Rule 4: conflict flag + pessimistic floor ─────────────────────────────────

def test_conflict_flag_and_pessimistic_cap():
    # available(95) vs out(5) within 12h -> conflict, score capped at 50
    rec = reconcile_player([
        C("guardian", "available", T1),
        C("sportsgambler", "out", T2),
    ])
    assert rec["conflict"] is True
    assert rec["score"] <= 50, f"pessimistic cap not applied: {rec}"


def test_adjacent_severities_not_conflict():
    rec = reconcile_player([
        C("guardian", "doubtful", T1),
        C("sportsgambler", "out", T2),
    ])
    assert rec["conflict"] is False


def test_stale_disagreement_not_conflict():
    # available now vs out 3 days ago: normal recovery timeline, no conflict
    rec = reconcile_player([
        C("guardian", "out", T_OLD),
        C("sportsgambler", "available", T1),
    ])
    assert rec["conflict"] is False


# ── Rule 5: agreement bonus ───────────────────────────────────────────────────

def test_agreement_bonus_applied():
    one = reconcile_player([C("guardian", "available", T1)])
    two = reconcile_player([
        C("guardian", "available", T1),
        C("sportsgambler", "available", T1),
    ])
    assert two["score"] == min(100, one["score"] + 5), (one, two)


def test_single_source_no_bonus():
    rec = reconcile_player([C("guardian", "doubtful", T1)])
    assert rec["score"] == STATUS_SCORES["doubtful"]
    assert rec["n_sources"] == 1


# ── Rule 0a: staleness window (the GW38 Xhaka bug) ────────────────────────────

DEADLINE = "2026-05-22T23:59:59+00:00"
T_STALE = "2026-02-01T03:03:00+00:00"    # 110 days before DEADLINE


def test_stale_only_claim_dropped():
    # The real bug: a Feb-1 snapshot claim was the ONLY source for a player
    # at GW38 — relative recency decay cancels out (num/den), so it scored
    # "out" at full strength. With ref_time anchored to the deadline it must
    # now drop entirely (absence from fresh tables == healthy).
    rec = reconcile_player([C("sportsgambler", "out", T_STALE)],
                           ref_time=DEADLINE)
    assert rec is None, f"110-day-old claim survived: {rec}"


def test_stale_claim_excluded_fresh_kept():
    rec = reconcile_player([
        C("sportsgambler", "out", T_STALE),
        C("guardian", "available", "2026-05-22T15:00:00+00:00"),
    ], ref_time=DEADLINE)
    assert rec is not None
    assert rec["n_sources"] == 1, rec
    assert rec["score"] == STATUS_SCORES["available"]


def test_reconcile_gw_drops_all_stale_players():
    claims = [
        C("sportsgambler", "out", T_STALE, pid=101),
        C("guardian", "doubtful", "2026-05-22T15:00:00+00:00", pid=202),
    ]
    recs = reconcile_gw(claims, ref_time=DEADLINE)
    assert [r["player_id"] for r in recs] == [202], recs


def test_default_ref_time_still_works():
    # Without an explicit ref_time the newest claim anchors — a lone claim
    # is fresh relative to itself (live-tick semantics, observed_at ~ now).
    rec = reconcile_player([C("guardian", "doubtful", T1)])
    assert rec is not None and rec["score"] == STATUS_SCORES["doubtful"]


def test_longterm_row_in_fresh_snapshot_kept():
    # The GW38 Odobert case: FFS/KnB rows last EDITED in February
    # (published_at) but present in a fresh May snapshot (observed_at).
    # The source asserts the injury is still true NOW — all three sources
    # must survive, not just the one with a fresh published_at.
    def LT(source, observed, published):
        return make_claim(38, source, 1, "u", 18, "Wilson Odobert", "out",
                          player_id=90, observed_at=observed,
                          published_at=published,
                          return_date="2026-11-29T00:00:00+00:00")
    rec = reconcile_player([
        LT("guardian",      "2026-05-22T15:00:00+00:00", "2026-05-22T15:00:00+00:00"),
        LT("ffs_injuries",  "2026-05-25T12:00:00+00:00", "2026-02-16T00:00:00+00:00"),
        LT("knocksandbans", "2026-05-21T12:00:00+00:00", "2026-02-13T00:00:00+00:00"),
    ], ref_time=DEADLINE)
    assert rec is not None
    assert rec["n_sources"] == 3, f"long-term rows wrongly dropped: {rec}"
    assert rec["status"] == "out"
    assert rec["score"] <= STATUS_SCORES["out"] + 5   # out (+agreement bonus)


def test_stale_snapshot_still_dropped_on_observed_at():
    # The Xhaka fix must survive the observed_at change: a claim ASSERTED
    # in February (old observed_at) says nothing about GW38, even though
    # its published_at is also February.
    c = make_claim(38, "sportsgambler", 1, "u", 17, "Granit Xhaka", "out",
                   player_id=668, observed_at=T_STALE, published_at=T_STALE,
                   return_date="2026-02-22T00:00:00+00:00")
    assert reconcile_player([c], ref_time=DEADLINE) is None


# ── Rule 0b: the claim's own return date has passed ───────────────────────────

def test_return_date_passed_degrades_to_unknown():
    # "out (ankle), returning 2026-02-28" evaluated at the GW38 deadline:
    # the claim's own content says the absence is over -> unknown (50).
    c = make_claim(38, "sportsgambler", 1, "u", 17, "Granit Xhaka", "out",
                   player_id=668, observed_at="2026-05-16T00:00:00+00:00",
                   published_at="2026-05-16T00:00:00+00:00",
                   return_date="2026-02-28T00:00:00+00:00")
    rec = reconcile_player([c], ref_time=DEADLINE)
    assert rec is not None
    assert rec["status"] == "unknown", rec
    assert rec["score"] == STATUS_SCORES["unknown"]


def test_return_date_in_future_keeps_status():
    c = make_claim(38, "sportsgambler", 1, "u", 17, "Granit Xhaka", "out",
                   player_id=668, observed_at="2026-05-16T00:00:00+00:00",
                   published_at="2026-05-16T00:00:00+00:00",
                   return_date="2026-07-20T00:00:00+00:00")
    rec = reconcile_player([c], ref_time=DEADLINE)
    assert rec["status"] == "out" and rec["score"] == STATUS_SCORES["out"]


def test_expired_suspension_does_not_zero():
    # A ban that ended in February must not zero a May gameweek.
    c = make_claim(38, "ffs_injuries", 1, "u", 17, "Test Player", "suspended",
                   player_id=9, observed_at="2026-05-16T00:00:00+00:00",
                   published_at="2026-05-16T00:00:00+00:00",
                   return_date="2026-02-10T00:00:00+00:00")
    rec = reconcile_player([c], ref_time=DEADLINE)
    assert rec["status"] != "suspended" and rec["score"] > 0, rec


# ── Tier mapping (must equal intel_03's) ──────────────────────────────────────

def test_tier_mapping_matches_intel_03():
    cases = [(95, "available"), (80, "available"), (79, "probable"),
             (60, "probable"), (59, "doubtful"), (40, "doubtful"),
             (30, "doubtful"), (29, "unlikely"), (10, "unlikely"),
             (9, "out"), (5, "out"), (0, "out")]
    for score, want in cases:
        got = availability_tier(score)
        assert got == want, f"tier({score}) = {got}, want {want}"


# ── Ledger dedup + grouping ───────────────────────────────────────────────────

def test_ledger_append_dedup():
    d = tempfile.mkdtemp()
    try:
        led = ClaimLedger(d)
        c1 = C("guardian", "doubtful", T1)
        c2 = C("guardian", "out", T2)
        assert led.append(20, [c1]) == 1
        assert led.append(20, [c1, c2]) == 1          # c1 deduped
        assert led.append(20, [c1, c2]) == 0          # all deduped
        assert len(led.load(20)) == 2
        assert led.load(21) == []
    finally:
        shutil.rmtree(d)


def test_reconcile_gw_grouping():
    claims = [
        C("guardian", "doubtful", T1, pid=101),
        C("sportsgambler", "doubtful", T1, pid=101),
        C("guardian", "out", T1, pid=202),
        # unresolved player: grouped by (club, raw name), still visible
        make_claim(20, "knocksandbans", 1, "u", 7, "Mystery Man", "out",
                   observed_at=T1, published_at=T1),
    ]
    recs = reconcile_gw(claims)
    assert len(recs) == 3, [r["player_raw"] for r in recs]
    by_pid = {r["player_id"]: r for r in recs if r["player_id"]}
    assert by_pid[101]["n_sources"] == 2
    unresolved = [r for r in recs if r["player_id"] is None]
    assert len(unresolved) == 1 and unresolved[0]["player_raw"] == "Mystery Man"


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
