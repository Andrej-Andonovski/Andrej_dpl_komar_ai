"""
pipeline/intel_02_ledger.py
Claim ledger + reconciler for the multi-source scraper (redesign step 2).

Implements docs/press_scraper_redesign.md §5:
  - §5.1 append-only per-GW claim ledger (JSONL, claim_id dedup)
  - §5.2 reconciliation: suspension override, per-source freshness,
    tier+recency weighted scoring, conflict flagging with pessimistic floor,
    N-source agreement bonus
  - §5.4 schema-compatible output pieces (intel_03 tier mapping)

Adapters emit CLAIMS ("source S said at time T that player P is doubtful");
only reconcile_player() produces conclusions. All claims are kept for audit.

Stdlib only.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Claim schema (§5.1)
# ---------------------------------------------------------------------------

STATUSES = ("available", "doubtful", "out", "suspended", "unknown")

# intel_03 press score scale — kept identical so downstream semantics and the
# eventual reconciled-score consumer agree (§5.2 rule 3).
STATUS_SCORES = {"available": 95, "doubtful": 40, "out": 5,
                 "suspended": 0, "unknown": 50}

# Severity rank for conflict detection (§5.2 rule 4). `unknown` takes no part.
_SEVERITY = {"available": 0, "doubtful": 1, "out": 2, "suspended": 2}

# Cross-source weights by tier (§5.2 rule 3)
W_TIER = {0: 0.8, 1: 1.0, 2: 1.2, 3: 0.9, 4: 1.1}

RECENCY_HALF_LIFE_H = 48.0   # weight halves every 48h
CONFLICT_WINDOW_H   = 12.0   # fresh disagreement window
AGREEMENT_BONUS     = 5      # intel_03's both-agree bonus, extended to N sources
MAX_CLAIM_AGE_H     = 240.0  # claims >10 days from ref_time carry no signal
                             # for that GW (tables are current-state sources;
                             # recency decay alone cancels out when a stale
                             # claim is a player's ONLY claim)
RETURN_PASSED_H     = 48.0   # out/doubtful/suspended whose own return_date
                             # is this far before ref_time degrade to unknown

# intel_03 tier thresholds — duplicated verbatim for the compat output
AVAILABILITY_TIERS = [(80, "available"), (60, "probable"), (30, "doubtful"),
                      (10, "unlikely"), (0, "out")]


def availability_tier(score: int) -> str:
    for threshold, label in AVAILABILITY_TIERS:
        if score >= threshold:
            return label
    return "out"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_claim(gw: int, source: str, tier: int, url: str,
               club_id: int, player_raw: str, status_claim: str,
               *, player_id: int | None = None, injury: str = "",
               text: str = "", observed_at: str | None = None,
               published_at: str | None = None, return_date: str | None = None,
               extractor: str = "structured") -> dict:
    """Build a schema-complete claim dict with a stable claim_id."""
    if status_claim not in STATUSES:
        raise ValueError(f"bad status_claim {status_claim!r}")
    observed_at = observed_at or utcnow_iso()
    key = "|".join([source, url or "", str(player_id or player_raw),
                    status_claim, text])
    claim_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return {
        "claim_id":     claim_id,
        "gw":           gw,
        "source":       source,
        "tier":         tier,
        "url":          url,
        "observed_at":  observed_at,
        "published_at": published_at,
        "club_id":      club_id,
        "player_id":    player_id,
        "player_raw":   player_raw,
        "status_claim": status_claim,
        "injury":       injury,
        "return_date":  return_date,
        "text":         text,
        "extractor":    extractor,
    }


# ---------------------------------------------------------------------------
# Append-only ledger (§5.1)
# ---------------------------------------------------------------------------

class ClaimLedger:
    """JSONL ledger, one file per GW, deduplicated on claim_id."""

    def __init__(self, dir_path: str):
        self.dir = dir_path
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, gw: int) -> str:
        return os.path.join(self.dir, f"gw{gw}.jsonl")

    def load(self, gw: int) -> list:
        path = self._path(gw)
        if not os.path.exists(path):
            return []
        claims = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    claims.append(json.loads(line))
        return claims

    def append(self, gw: int, claims: list) -> int:
        """Append claims not already in the ledger. Returns count appended."""
        existing = {c["claim_id"] for c in self.load(gw)}
        fresh = [c for c in claims if c["claim_id"] not in existing]
        if fresh:
            with open(self._path(gw), "a", encoding="utf-8") as f:
                for c in fresh:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
        return len(fresh)


# ---------------------------------------------------------------------------
# Reconciliation (§5.2)
# ---------------------------------------------------------------------------

def _effective_ts(claim: dict) -> str:
    return claim.get("published_at") or claim["observed_at"]


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _hours_between(a: str, b: str) -> float:
    return abs((_parse_ts(a) - _parse_ts(b)).total_seconds()) / 3600.0


def reconcile_player(claims: list, ref_time: str | None = None) -> dict | None:
    """
    Reconcile all claims for one (player, gw) into a single record, or None
    when no claim is fresh enough to say anything about this GW.
    `ref_time` anchors staleness and recency decay — pass the GW deadline;
    defaults to the newest claim's timestamp (deterministic for backtests).

    Rule order (§5.2 + staleness guards):
      0a. staleness: claims >MAX_CLAIM_AGE_H from ref_time are excluded;
          all-stale -> None (absence from fresh tables means healthy)
      0b. return-date passed: out/doubtful/suspended whose own return_date
          is >RETURN_PASSED_H before ref_time count as "unknown"
      1. suspension override (tier <= 2)
      2. per-source last-write-wins
      3. tier x recency weighted mean on the intel_03 score scale
      4. conflict (>1 severity step within 12h) -> flag + pessimistic
         midpoint cap
      5. agreement within one step across >=2 sources -> +5
    """
    assert claims, "reconcile_player needs at least one claim"

    # -- rule 2: keep each source's latest claim only ------------------------
    latest_by_source: dict[str, dict] = {}
    for c in claims:
        prev = latest_by_source.get(c["source"])
        if prev is None or _effective_ts(c) > _effective_ts(prev):
            latest_by_source[c["source"]] = c
    live = list(latest_by_source.values())

    ref = ref_time or max(_effective_ts(c) for c in live)

    # -- rule 0a: hard staleness window ---------------------------------------
    # Keyed on observed_at, NOT published_at: a long-term injury row in a
    # fresh table snapshot was last EDITED months ago (published_at) but the
    # source is asserting it is still true NOW (observed_at). Only claims
    # whose assertion time is stale carry no signal for this GW.
    live = [c for c in live
            if _hours_between(c["observed_at"], ref) <= MAX_CLAIM_AGE_H]
    if not live:
        return None

    # -- rule 0b: the claim's own return date has passed ----------------------
    def gw_status(c: dict) -> str:
        s = c["status_claim"]
        rd = c.get("return_date")
        if s in ("out", "doubtful", "suspended") and rd:
            try:
                hours_past = (_parse_ts(ref) - _parse_ts(rd)).total_seconds() / 3600.0
            except ValueError:
                return s
            if hours_past >= RETURN_PASSED_H:
                return "unknown"
        return s

    # -- rule 1: suspension override -----------------------------------------
    susp = [c for c in live if gw_status(c) == "suspended"
            and c["tier"] <= 2]
    conflict = False
    if susp:
        score = 0
        status = "suspended"
    else:
        # -- rule 3: weighted mean -------------------------------------------
        num = den = 0.0
        status_weight: dict[str, float] = {}
        for c in live:
            w = (W_TIER.get(c["tier"], 1.0)
                 * 0.5 ** (_hours_between(_effective_ts(c), ref)
                           / RECENCY_HALF_LIFE_H))
            st = gw_status(c)
            num += w * STATUS_SCORES[st]
            den += w
            status_weight[st] = status_weight.get(st, 0.0) + w
        score = num / den if den else 50.0

        # -- rule 4: conflict flag + pessimistic midpoint cap ------------------
        ranked = [(c, gw_status(c)) for c in live
                  if gw_status(c) in _SEVERITY]
        cap = None
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                (a, sa), (b, sb) = ranked[i], ranked[j]
                gap = abs(_SEVERITY[sa] - _SEVERITY[sb])
                if gap > 1 and _hours_between(_effective_ts(a),
                                              _effective_ts(b)) <= CONFLICT_WINDOW_H:
                    conflict = True
                    mid = (STATUS_SCORES[sa] + STATUS_SCORES[sb]) / 2.0
                    cap = mid if cap is None else min(cap, mid)
        if conflict and cap is not None:
            score = min(score, cap)

        # -- rule 5: agreement bonus -------------------------------------------
        if not conflict and len(live) >= 2:
            sev_present = {_SEVERITY[st] for _, st in ranked}
            if sev_present and max(sev_present) - min(sev_present) <= 1:
                score = min(100.0, score + AGREEMENT_BONUS)

        # Compat status label: weighted vote (drives legacy intel_03, which
        # re-derives its score from the label — §5.4)
        status = max(status_weight, key=status_weight.get) if status_weight \
            else "unknown"

    score = int(round(max(0.0, min(100.0, score))))

    # Representative injury/text: newest claim that carries them
    by_new = sorted(live, key=_effective_ts, reverse=True)
    injury = next((c["injury"] for c in by_new if c.get("injury")), "")
    text = next((c["text"] for c in by_new if c.get("text")), "")
    ret = next((c["return_date"] for c in by_new if c.get("return_date")), None)

    return {
        "player_id":          claims[0].get("player_id"),
        "player_raw":         by_new[0]["player_raw"],
        "club_id":            claims[0]["club_id"],
        "status":             status,
        "score":              score,
        "tier_label":         availability_tier(score),
        "injury":             injury,
        "return_date":        ret,
        "text":               text,
        "conflict":           conflict,
        "n_sources":          len(live),
        "sources":            sorted(c["claim_id"] for c in live),
        "source_names":       sorted(latest_by_source.keys()),
        "latest_observed_at": max(c["observed_at"] for c in live),
    }


def reconcile_gw(claims: list, ref_time: str | None = None) -> list:
    """
    Group a GW's claims by player and reconcile each group.
    Resolved players group by player_id; unresolved ones by
    (club_id, accent-folded raw name) so "Paqueta"/"Paquetá" from two
    sources merge, and unresolved names stay visible for audit.
    """
    from intel_identity import normalize_name
    groups: dict = {}
    for c in claims:
        if c.get("player_id") is not None:
            key = ("pid", c["player_id"])
        else:
            key = ("raw", c["club_id"], normalize_name(c["player_raw"]))
        groups.setdefault(key, []).append(c)
    records = (reconcile_player(grp, ref_time=ref_time)
               for grp in groups.values())
    return [r for r in records if r is not None]   # None = all claims stale
