"""Unit tests for the unanchored-chip percentile ledger."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
from chip_percentile import ChipPercentileLedger, nearest_rank


def test_nearest_rank_is_deterministic():
    vals = [1, 3, 5, 7]
    assert nearest_rank(vals, 0.0) == 1
    assert nearest_rank(vals, 0.5) == 3
    assert nearest_rank(vals, 0.75) == 5
    assert nearest_rank(vals, 1.0) == 7


def test_warmup_does_not_block_chip():
    ledger = ChipPercentileLedger(q=0.75, min_observations=3)
    ledger.record({"tc1": 9.0})
    ledger.record({"tc1": 10.0})
    assert ledger.threshold("tc1") is None
    assert ledger.allows("tc1", 0.0)


def test_plain_week_must_clear_prior_percentile():
    ledger = ChipPercentileLedger(q=0.75, min_observations=3)
    ledger.record({"tc1": 4.0})
    ledger.record({"tc1": 7.0})
    ledger.record({"tc1": 10.0})
    assert ledger.threshold("tc1") == 10.0
    assert not ledger.allows("tc1", 9.9)
    assert ledger.allows("tc1", 10.0)


def test_anchored_week_bypasses_bar():
    ledger = ChipPercentileLedger(q=0.75, min_observations=3)
    ledger.record({"bb1": 5.0})
    ledger.record({"bb1": 7.0})
    ledger.record({"bb1": 9.0})
    assert not ledger.allows("bb1", 1.0)
    assert ledger.allows("bb1", 1.0, anchored=True)


def test_state_round_trips_atomically():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.json")
        first = ChipPercentileLedger(path, q=0.75)
        first.record({"wc1": 2.5, "tc1": 8.0})
        second = ChipPercentileLedger(path, q=0.75)
        assert second.prior_values("wc1") == [2.5]
        assert second.prior_values("tc1") == [8.0]
        assert json.load(open(path, encoding="utf-8"))["version"] == 1


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
