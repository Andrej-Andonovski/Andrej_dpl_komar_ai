"""Persistent percentile gates for unanchored in-model chips.

The ledger records one proxy value per real gameweek for each chip token
(``wc1``, ``tc2``, ...).  A chip can fire on a plain week only when its
current value meets the q-th percentile of earlier values in that chip's
current set.  Event weeks are intentionally exempt: their calendar signal is
already captured by the chip model's existing guards.
"""
import json
import math
import os
from pathlib import Path


DEFAULT_Q = 0.75
DEFAULT_MIN_OBSERVATIONS = 3


def _validate_q(q):
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile q must be in [0, 1], got {q!r}")


def nearest_rank(values, q):
    """Return the q-th quantile by deterministic nearest-rank convention."""
    _validate_q(q)
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


class ChipPercentileLedger:
    """JSON-backed, append-only weekly values for a single simulation season."""

    def __init__(self, path=None, q=DEFAULT_Q,
                 min_observations=DEFAULT_MIN_OBSERVATIONS, load=True):
        _validate_q(q)
        if min_observations < 0:
            raise ValueError("min_observations must be non-negative")
        self.path = Path(path) if path else None
        self.q = float(q)
        self.min_observations = int(min_observations)
        self.values = {}
        if load and self.path and self.path.exists():
            self._load()

    def _load(self):
        with self.path.open(encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != 1:
            raise ValueError(f"unsupported chip percentile state: {self.path}")
        self.values = {str(k): [float(v) for v in vs]
                       for k, vs in data.get("values", {}).items()}

    def prior_values(self, chip):
        return list(self.values.get(chip, []))

    def threshold(self, chip):
        vals = self.prior_values(chip)
        if len(vals) < self.min_observations:
            return None
        return nearest_rank(vals, self.q)

    def allows(self, chip, value, *, anchored=False):
        """Whether a chip may fire now; events bypass the seasonal bar."""
        if anchored:
            return True
        cutoff = self.threshold(chip)
        return cutoff is None or float(value) >= cutoff

    def record(self, values):
        """Append this GW's proxy values once for every still-relevant chip."""
        for chip, value in values.items():
            self.values.setdefault(str(chip), []).append(float(value))
        self.save()

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"version": 1, "q": self.q,
                   "min_observations": self.min_observations,
                   "values": self.values}
        with temp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(temp, self.path)
