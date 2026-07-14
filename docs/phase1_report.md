# Phase 1 — Prediction Matrix: Calibration Report

Status: **COMPLETE** (2026-07-14). Blueprint: `docs/optimizer_redesign.md` §3, §9.
Artifacts: `pipeline/prediction_matrix.py` (12/12 unit tests),
`pipeline/phase1_calibration.py`, `data/intel/phase1_calibration.json`,
log `data/intel/run_phase1_calibration.log`.

Method: walk-forward GW2–38 (Docker reference env). At each GW t, models
trained exactly as the simulator trains them (same rows, same 1+(t−1)
current-season weighting), matrix built for offsets 0–5, every
(player, future-GW) cell scored against actuals.

## Headline: the horizon is trustworthy

| h (GWs ahead) | MAE played | MAE top-60 | Spearman | blank viol | DGW cells |
|---|---|---|---|---|---|
| 0 | 2.044 | 2.843 | 0.716 | 0 | 417 |
| 1 | 2.055 | 2.862 | 0.697 | 0 | 417 |
| 2 | 2.071 | 2.941 | 0.676 | 0 | 417 |
| 3 | 2.092 | 2.990 | 0.655 | 0 | 417 |
| 4 | 2.116 | 3.104 | 0.639 | 0 | 417 |
| 5 | 2.128 | 3.079 | 0.632 | 0 | 417 |

MAE degrades only **+4.1%** from h=0 to h=5; rank signal (Spearman) holds
0.72 → 0.63. Predictions rot slowly because the form features are held
constant (per design §3.1) and only fixture context changes — and fixture
context is *known*. Consequences:

- **Blueprint risk R1 (horizon hurts) is LOW.** H = 5–6 is supported by data.
- **δ = 0.84 was too pessimistic.** The measured decay (MAE ratio 1.04 over
  5 weeks; Spearman −2.5%/week) supports δ in the 0.90–0.97 range. Final
  value still goes to the Phase 6 ablation, but the H-sweep should be run
  expecting long horizons to win.

## Exit gates

1. **Matrix vs legacy at h=0: PASS.** Sim-style baseline (model + FDR
   post-multipliers): MAE_played 2.040 / MAE_top60 2.863. Matrix: 2.044 /
   **2.843 (better on the decision-relevant stratum)**. Deleting the FDR
   post-multipliers and the flat DGW ×2.0 costs nothing.
2. **Blank/DGW exactness: PASS.** 0 blank violations across ~6k blank cells;
   417 DGW cells per offset handled as per-fixture sums.
3. **φ confidence gate: FAIL — φ set to 1 for Phase 2 (per §3.5).**
   Bucket MAE (h≤1, played): φ<0.70 → 1.77, 0.70–0.85 → 2.06, ≥0.85 → 2.10.
   *Inverted* — but the measurement is magnitude-confounded: low-φ players
   are rotation/fringe players whose scores (and thus absolute errors) are
   small; high-φ nailed starters score more and carry bigger absolute
   errors. As registered, the gate fails → the MILP objective uses ŝ = μ
   (no shrinkage). Revisit in Phase 5 with a μ-matched-bucket test before
   deleting the machinery permanently (it stays computed in the matrix).

## Findings needing follow-up

- **q90 under-covers: 0.836 vs 0.90 target.** FPL points are right-skewed;
  a normal 1.2816·σ quantile is too tight. Action: raise `Z90` to ≈1.65 and
  re-measure in the next full calibration (Phase 3 reruns this harness).
  Low urgency: q90 only feeds the captain blend, whose weight θ is tuned
  later and absorbs scale; the *ranking* by q90 is what matters.
- π is produced but not yet validated against anything (no gate registered
  for Phase 1); its calibration check (predicted play-prob vs realized
  play rate) should be added to the Phase 3 harness run.

## Notes for Phase 2 (single-GW MILP with new objective)

- Use matrix μ directly as ŝ (φ≡1). κ = π·[(1−θ)μ + θ·q90] as designed.
- The matrix already returns price + sell_value per cell (ledger-aware).
- `DEFAULT_FEAT_COLS` is asserted equal to `season_simulator.FEAT_COLS` at
  calibration time; keep that assert in any new consumer.
