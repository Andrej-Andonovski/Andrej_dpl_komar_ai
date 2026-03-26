"""
pipeline/intel_07_bench.py
Intel 07: Bench Intelligence

Standalone data-driven module (no LLM, no external APIs).
Recommends optimal bench players and identifies the best BB window.

Scoring formula per player:
    bench_score = form_last3 * minutes_reliability * fdr_adj
    fdr_adj = max(0.5, 1.0 - 0.03 * (fdr - 3.0))

Outputs:
    - Per-GW bench template (3 outfield + 1 GK, all <= MAX_BENCH_PRICE)
    - BB target GW (highest combined bench score over 3-GW lookahead)
    - bench_candidate_bonus per player per GW
"""

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_BENCH_PRICE  = 5.5   # max price for bench candidates (£m)
MIN_MINUTES_REL  = 0.5   # must have played 50%+ of available minutes
MIN_FORM_LAST3   = 2.0   # minimum form to be considered
BENCH_BONUS_NORMAL = 2.5  # added to pred for bench candidates (normal GWs)
BENCH_BONUS_BB_GW  = 5.0  # added to pred on the target BB GW (forces strong bench)
LOOKAHEAD_GWS    = 3     # how many GWs ahead to evaluate for BB window
BB_MIN_GW        = 8     # don't target BB before this GW (model needs calibration)
BB_MAX_GW_SET1   = 19    # Set 1 BB must fire by GW19
BB_MAX_GW_SET2   = 38    # Set 2 BB must fire by GW38


# ── Core Functions ─────────────────────────────────────────────────────────────

def score_bench_candidates(pool, gw):
    """
    Score all affordable players as bench candidates.

    Returns dict: {
        "by_position": {"GK": [...], "DEF": [...], "MID": [...], "FWD": [...]},
        "recommended": {
            "GK": best GK candidate or None,
            "outfield": [best 3 outfield across DEF/MID/FWD]
        },
        "all_candidates": top 20 candidates (for logging)
    }
    Each candidate has: player_id, web_name, pos, team, price,
                        form_last3, minutes_reliability, fdr, fdr_adj,
                        bench_score, pred (+ all original pool fields)
    """
    # GW1: form data is 2024-25 season averages — artificially low, so use looser threshold
    form3_threshold = 0.5 if gw == 1 else MIN_FORM_LAST3

    candidates = []
    for p in pool:
        if p.get("price", 0) > MAX_BENCH_PRICE:
            continue
        if p.get("zero_minutes", False):
            continue
        if p.get("minutes_reliability", 0) < MIN_MINUTES_REL:
            continue
        if p.get("form_last3", 0) < form3_threshold:
            continue

        fdr     = p.get("fdr", 3.0)
        fdr_adj = max(0.5, 1.0 - 0.03 * (fdr - 3.0))
        form3   = p.get("form_last3", 0.0)
        min_rel = p.get("minutes_reliability", 0.0)

        bench_score = form3 * min_rel * fdr_adj

        candidates.append({
            **p,
            "fdr_adj":     fdr_adj,
            "bench_score": bench_score,
        })

    # Sort by bench_score descending
    candidates.sort(key=lambda x: x["bench_score"], reverse=True)

    # Split by position
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for c in candidates:
        pos = c.get("pos", "MID")
        if pos in by_pos:
            by_pos[pos].append(c)

    # Best GK
    best_gk = by_pos["GK"][0] if by_pos["GK"] else None

    # Best 3 outfield across all positions
    outfield      = [c for c in candidates if c.get("pos") != "GK"]
    best_outfield = outfield[:3]

    return {
        "by_position":   by_pos,
        "recommended":   {"GK": best_gk, "outfield": best_outfield},
        "all_candidates": candidates[:20],
    }


def find_bb_target_gw(pools_by_gw, bb_min_gw, bb_max_gw):
    """
    Given bench candidate scores for multiple GWs, find the best BB week.

    For each GW, combined score = sum of top 3 outfield bench_scores + GK bench_score.
    The BB target is the GW with the highest combined score within the valid window.

    Returns: {
        "target_gw":   int,
        "target_score": float,
        "gw_scores":   {gw: combined_score},
        "reasoning":   str
    }
    """
    gw_scores = {}
    for gw, result in pools_by_gw.items():
        if gw < bb_min_gw or gw > bb_max_gw:
            continue
        rec      = result.get("recommended", {})
        gk       = rec.get("GK")
        outfield = rec.get("outfield", [])

        gk_score  = gk["bench_score"] if gk else 0.0
        out_score = sum(p["bench_score"] for p in outfield[:3])
        gw_scores[gw] = gk_score + out_score

    if not gw_scores:
        return {
            "target_gw":   bb_min_gw,
            "target_score": 0.0,
            "gw_scores":   {},
            "reasoning":   "No valid GWs found",
        }

    target_gw = max(gw_scores, key=gw_scores.get)
    return {
        "target_gw":   target_gw,
        "target_score": gw_scores[target_gw],
        "gw_scores":   gw_scores,
        "reasoning":   (
            f"GW{target_gw} has best combined bench score "
            f"{gw_scores[target_gw]:.2f} across outfield+GK"
        ),
    }


def get_bench_intel(pool, gw, chips_used, bb1_used, bb2_used,
                    fdr_lookup=None, home_lookup=None):
    """
    Primary function: call each GW from the simulator.

    Parameters:
        pool:      list of player dicts with rolling features already computed
        gw:        current gameweek
        chips_used: set of chip strings already used (e.g. {"wc1", "tc1"})
        bb1_used:  bool — BB Set 1 already used
        bb2_used:  bool — BB Set 2 already used

    Returns: {
        "bench_candidates": {player_id: bonus_pts},   # bonus to add to pred
        "bb_target_gw":     int or None,
        "bb_set":           1 or 2,
        "recommended_bench": [list of player dicts],
        "is_bb_target_gw":  bool
    }
    """
    use_set1 = (gw <= 19)
    bb_used  = bb1_used if use_set1 else bb2_used

    if bb_used:
        # BB already used for this set — no bonus needed
        return {
            "bench_candidates":  {},
            "bb_target_gw":      None,
            "bb_set":            1 if use_set1 else 2,
            "recommended_bench": [],
            "is_bb_target_gw":   False,
        }

    # Score current GW bench candidates
    current_result = score_bench_candidates(pool, gw)

    # Look ahead LOOKAHEAD_GWS to find best BB window.
    # Simple approximation: use current form but same fdr for future GWs.
    bb_max = BB_MAX_GW_SET1 if use_set1 else BB_MAX_GW_SET2
    bb_min = max(gw, BB_MIN_GW)

    lookahead_pools = {}
    for future_gw in range(bb_min, min(bb_max + 1, gw + LOOKAHEAD_GWS + 1)):
        if fdr_lookup and home_lookup:
            future_pool = []
            for p in pool:
                team = p.get("team")
                fp = dict(p)
                fp["fdr"]      = float(fdr_lookup.get((team, future_gw), 3.0))
                fp["was_home"] = float(home_lookup.get((team, future_gw), 0))
                future_pool.append(fp)
        else:
            future_pool = pool
        lookahead_pools[future_gw] = score_bench_candidates(future_pool, future_gw)

    bb_target = find_bb_target_gw(lookahead_pools, bb_min, bb_max)
    target_gw = bb_target["target_gw"]
    is_target = (gw == target_gw)

    # Build bonus dict
    bonus = BENCH_BONUS_BB_GW if is_target else BENCH_BONUS_NORMAL

    bench_candidates  = {}
    recommended_bench = []

    rec = current_result["recommended"]
    if rec.get("GK"):
        pid = rec["GK"]["player_id"]
        bench_candidates[pid] = bonus
        recommended_bench.append(rec["GK"])

    for p in rec.get("outfield", []):
        pid = p["player_id"]
        bench_candidates[pid] = bonus
        recommended_bench.append(p)

    # Print diagnostics
    print(f"  [BENCH-INTEL] GW{gw} | BB target: GW{target_gw} "
          f"(score: {bb_target['target_score']:.2f}) | "
          f"is_target: {is_target} | bonus: {bonus:.1f}pts")
    print(f"  [BENCH-INTEL] Lookahead scores: " +
          ", ".join(f"GW{g}:{s:.2f}"
                    for g, s in sorted(bb_target["gw_scores"].items())))
    if recommended_bench:
        print(f"  [BENCH-INTEL] Recommended bench: " +
              ", ".join(
                  f"{p['web_name']}(£{p['price']:.1f}m,"
                  f"f3={p['form_last3']:.1f})"
                  for p in recommended_bench
              ))

    return {
        "bench_candidates":  bench_candidates,
        "bb_target_gw":      target_gw,
        "bb_set":            1 if use_set1 else 2,
        "recommended_bench": recommended_bench,
        "is_bb_target_gw":   is_target,
    }
