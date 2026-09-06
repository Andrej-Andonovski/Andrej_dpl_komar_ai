"""
pipeline/stage10_infer.py
Stage 10 Phase 1 — the torch-free runtime inference path (option D).

season_simulator / stage10_refine call this; it loads a numpy .npz checkpoint
(exported by stage10_train.py) and returns the residual correction + uncertainty
for a batch of players. NO torch import — the production simulator and the
Optuna loop stay torch-free (and it keeps running if SAC ever re-blocks torch).

Checkpoint selection:
    live season  S  ->  models/stage10/pretrain_<S-1>.npz   (the "final" fold)
    SIM_SEASON backtest S  ->  models/stage10/pretrain_<S>.npz  (walk-forward fold)
Both cases: the checkpoint's training data ends strictly before season S — no leakage.

If the checkpoint is missing (not yet trained), the refiner reports
`ready == False` and every correction is 0.0 / sigma = prior, so STAGE10=on
degrades to exactly STAGE10=off. Callers should check `ready`.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import stage10_model as sm                          # noqa: E402  (numpy-only)
import stage10_sequence as sq                       # noqa: E402

CKPT_DIR = os.path.join(_ROOT, "models", "stage10")

# season string arithmetic: "2024-25" -> "2023-24"
def _prev_season(s: str) -> str:
    a, b = s.split("-")
    a = int(a) - 1
    return f"{a}-{str(a + 1)[-2:]}"


def checkpoint_for(target_season: str, live: bool) -> str:
    base = _prev_season(target_season) if live else target_season
    return os.path.join(CKPT_DIR, f"pretrain_{base}.npz")


class Stage10Refiner:
    def __init__(self, target_season: str, live: bool = True, ckpt_path: str = None):
        self.target_season = target_season
        self.path = ckpt_path or checkpoint_for(target_season, live)
        self._net = None
        self.ready = os.path.exists(self.path)
        if self.ready:
            try:
                self._net = sm.NumpyResidualLSTM(self.path)
            except Exception as e:                    # noqa: BLE001
                print(f"  [STAGE10] failed to load {self.path}: {e}")
                self.ready = False

    # -- array API ---------------------------------------------------------
    def predict_arrays(self, ts, query, lengths, pos_ids, n_played, positions,
                       mu_gbm):
        """Returns dict with r_applied, sigma_eff, q90, mu_corrected (all [N])."""
        n = len(lengths)
        if not self.ready:
            zeros = np.zeros(n)
            prior = np.array([sm.HEADROOM_PRIOR[p] for p in positions], float)
            mu = np.asarray(mu_gbm, float)
            return {"r_applied": zeros, "sigma_eff": prior,
                    "q90": mu + sm.Z90 * prior, "mu_corrected": mu}
        r_hat, sigma = self._net.predict(ts, query, lengths, pos_ids)
        r_app, s_eff = sm.apply_gate(r_hat, sigma, n_played, positions)
        mu = np.asarray(mu_gbm, float)
        z = sm.z_array(self._net.z_cal, positions)
        return {"r_applied": r_app, "sigma_eff": s_eff,
                "q90": sm.q90_from(mu, r_app, s_eff, z=z),
                "mu_corrected": mu + r_app, "r_hat_raw": r_hat, "sigma_raw": sigma}

    # -- sample API (offline / tests) ------------------------------------
    def predict_samples(self, samples, max_len: int = 38):
        ts, query, lengths, pos_ids, _, meta = sq.to_padded_arrays(samples, max_len)
        out = self.predict_arrays(ts, query, lengths, pos_ids,
                                  meta["n_played"].to_numpy(),
                                  meta["position"].tolist(),
                                  meta["mu_gbm_t"].to_numpy())
        keyed = {}
        for i, s in enumerate(samples):
            keyed[(s.name, s.season, s.target_gw)] = {
                "r_applied": float(out["r_applied"][i]),
                "sigma_eff": float(out["sigma_eff"][i]),
                "q90": float(out["q90"][i]),
                "mu_corrected": float(out["mu_corrected"][i]),
            }
        return keyed


_CACHE: dict = {}


def load_refiner(target_season: str, live: bool = True) -> Stage10Refiner:
    key = (target_season, live)
    if key not in _CACHE:
        _CACHE[key] = Stage10Refiner(target_season, live)
    return _CACHE[key]


if __name__ == "__main__":
    # smoke: report which checkpoints exist and run the sample API on one season
    for f in sorted(os.listdir(CKPT_DIR)) if os.path.isdir(CKPT_DIR) else []:
        if f.endswith(".npz"):
            print("  found", f)
    r = load_refiner("2024-25", live=False)
    print(f"  refiner for 2024-25 (backtest): ready={r.ready}  path={os.path.basename(r.path)}")
    if r.ready:
        s = sq.build_samples(["2024-25"])[:200]
        keyed = r.predict_samples(s)
        vals = list(keyed.values())
        mc = np.array([v["mu_corrected"] for v in vals])
        print(f"  {len(vals)} preds | mu_corrected mean={mc.mean():.2f} "
              f"q90 mean={np.mean([v['q90'] for v in vals]):.2f}")
