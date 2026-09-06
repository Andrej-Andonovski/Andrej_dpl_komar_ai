"""
pipeline/stage10_model.py
Stage 10 Phase 1 — Step 3: the LSTM residual model.

Two implementations that MUST agree numerically:
  * ResidualLSTM        — torch nn.Module, used for TRAINING (stage10_train.py).
  * NumpyResidualLSTM   — dependency-free forward pass, used at RUNTIME by
                          stage10_infer.py / stage10_refine.py so the production
                          simulator + Optuna loop never import torch
                          (option D, docs/stage10_phase1_plan.md).

Contract: torch model -> export_npz() -> a plain .npz the numpy path loads.
stage10_train.py --check-numpy asserts max|torch - numpy| < 1e-4 over the val set
(verified locally: Δr ~1e-8, Δsigma ~1e-7).

Architecture (plan §3):
  feature standardization (frozen pretrain mean/std)
  position embedding (4 -> 8), concatenated to every timestep and to the query
  LSTM: input F+8, hidden 64, 2 layers, dropout 0.2  -> hidden at true last step
  QueryMLP: Linear(Q+8 -> 64) . GELU(tanh) . Dropout
  Fuse: concat(h_last, query_repr) -> Linear(128 -> 64) . GELU(tanh) . Dropout
  Head_resid : Linear(64 -> 1)                    -> r_hat
  Head_logvar: Linear(64 -> 1) -> sigma = softplus + SIGMA_FLOOR
  (confidence gate + sigma shrink are applied OUTSIDE the net, at inference only)
"""
from __future__ import annotations

import numpy as np

# ── shared hyperparameters / constants ──────────────────────────────────────
HIDDEN = 64
LSTM_LAYERS = 2
POS_EMB_DIM = 8
MLP_HIDDEN = 64
DROPOUT = 0.2
SIGMA_FLOOR = 0.5

POSITIONS = ["GK", "DEF", "MID", "FWD"]
Z90 = 1.2815515655446004                     # standard-normal 90th percentile
CONF_RAMP = 6.0                              # n_played to reach full r_hat weight
SIGMA_SHRINK_BELOW = 3                       # n_played < this -> blend sigma to prior
HEADROOM_PRIOR = {"GK": 2.5, "DEF": 3.0, "MID": 4.5, "FWD": 5.0}


# ── activations (tanh-approx GELU so numpy can match torch exactly) ──────────
_GELU_C = 0.7978845608028654                 # sqrt(2/pi)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x ** 3)))


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -60.0, 20.0))))


# ═══════════════════════════════════════════════════════════════════════════
#  numpy forward  (runtime; no torch)
# ═══════════════════════════════════════════════════════════════════════════
class NumpyResidualLSTM:
    """Loads an .npz exported by ResidualLSTM.export_npz and runs a forward pass.

    weights dict keys:
      ts_mean[F], ts_std[F], q_mean[Q], q_std[Q]
      pos_emb[4, P]
      lstm_w_ih_l{k}[4H, in_k]  lstm_w_hh_l{k}[4H, H]  lstm_b_ih_l{k}[4H]  lstm_b_hh_l{k}[4H]
      qmlp_w[64, Q+P] qmlp_b[64]
      fuse_w[64, 128] fuse_b[64]
      head_r_w[1, 64] head_r_b[1]
      head_lv_w[1, 64] head_lv_b[1]
      meta: F, Q, hidden, layers, pos_emb_dim
    """

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=True)
        self.w = {k: z[k] for k in z.files if k != "meta"}
        self.meta = z["meta"].item() if "meta" in z.files else {}
        self.F = int(self.meta.get("F", self.w["ts_mean"].shape[0]))
        self.Q = int(self.meta.get("Q", self.w["q_mean"].shape[0]))
        self.H = int(self.meta.get("hidden", HIDDEN))
        self.L = int(self.meta.get("layers", LSTM_LAYERS))

    # -- one LSTM layer over a [T, in] sequence, returns [T, H] --------------
    def _lstm_layer(self, x, k):
        W_ih = self.w[f"lstm_w_ih_l{k}"]        # [4H, in]
        W_hh = self.w[f"lstm_w_hh_l{k}"]        # [4H, H]
        b = self.w[f"lstm_b_ih_l{k}"] + self.w[f"lstm_b_hh_l{k}"]   # [4H]
        H = self.H
        h = np.zeros(H)
        c = np.zeros(H)
        out = np.empty((x.shape[0], H))
        xW = x @ W_ih.T + b                     # [T, 4H] precompute input contrib
        for t in range(x.shape[0]):
            z = xW[t] + h @ W_hh.T
            i = _sigmoid(z[:H])
            f = _sigmoid(z[H:2 * H])
            g = np.tanh(z[2 * H:3 * H])
            o = _sigmoid(z[3 * H:])
            c = f * c + i * g
            h = o * np.tanh(c)
            out[t] = h
        return out

    def _forward_one(self, ts, query, length, pos_id):
        F, Q = self.F, self.Q
        ts = (ts[:length] - self.w["ts_mean"]) / self.w["ts_std"]
        query = (query - self.w["q_mean"]) / self.w["q_std"]
        emb = self.w["pos_emb"][pos_id]                      # [P]

        seq = np.concatenate([ts, np.tile(emb, (length, 1))], axis=1)
        for k in range(self.L):
            seq = self._lstm_layer(seq, k)
        h_last = seq[-1]                                     # [H]

        q_in = np.concatenate([query, emb])
        q_repr = _gelu_tanh(self.w["qmlp_w"] @ q_in + self.w["qmlp_b"])

        fused = np.concatenate([h_last, q_repr])
        fused = _gelu_tanh(self.w["fuse_w"] @ fused + self.w["fuse_b"])

        r_hat = float(self.w["head_r_w"] @ fused + self.w["head_r_b"])
        log_var = float(self.w["head_lv_w"] @ fused + self.w["head_lv_b"])
        sigma = float(_softplus(np.array(log_var)) + SIGMA_FLOOR)
        return r_hat, sigma

    def predict(self, ts, query, lengths, pos_ids):
        """ts [N,L,F], query [N,Q], lengths [N], pos_ids [N] -> (r_hat[N], sigma[N])."""
        n = len(lengths)
        r = np.zeros(n)
        s = np.zeros(n)
        for i in range(n):
            r[i], s[i] = self._forward_one(ts[i], query[i], int(lengths[i]),
                                           int(pos_ids[i]))
        return r, s


# ── confidence gate + q90 (shared by train eval and runtime) ────────────────
def apply_gate(r_hat, sigma, n_played, positions):
    """Explicit, NOT learned (plan §3). Shrinks r_hat when the sequence is
    short; blends sigma toward the position headroom prior below 3 played GWs."""
    r_hat = np.asarray(r_hat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n_played = np.asarray(n_played, dtype=float)
    g = np.minimum(1.0, n_played / CONF_RAMP)
    r_applied = g * r_hat
    prior = np.array([HEADROOM_PRIOR[p] for p in positions], dtype=float)
    thin = n_played < SIGMA_SHRINK_BELOW
    sigma_eff = np.where(thin, 0.5 * sigma + 0.5 * prior, sigma)
    return r_applied, sigma_eff


def q90_from(mu_gbm, r_applied, sigma_eff):
    return np.asarray(mu_gbm, float) + np.asarray(r_applied, float) + Z90 * np.asarray(sigma_eff, float)


# ═══════════════════════════════════════════════════════════════════════════
#  torch model  (training only — imported lazily so runtime never needs torch)
# ═══════════════════════════════════════════════════════════════════════════
def build_torch_model(F, Q, seed=42):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class ResidualLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.F, self.Q = F, Q
            self.register_buffer("ts_mean", torch.zeros(F))
            self.register_buffer("ts_std", torch.ones(F))
            self.register_buffer("q_mean", torch.zeros(Q))
            self.register_buffer("q_std", torch.ones(Q))
            self.pos_emb = nn.Embedding(4, POS_EMB_DIM)
            self.lstm = nn.LSTM(F + POS_EMB_DIM, HIDDEN, LSTM_LAYERS,
                                batch_first=True, dropout=DROPOUT)
            self.qmlp = nn.Sequential(
                nn.Linear(Q + POS_EMB_DIM, MLP_HIDDEN), nn.GELU(approximate="tanh"),
                nn.Dropout(DROPOUT))
            self.fuse = nn.Sequential(
                nn.Linear(HIDDEN + MLP_HIDDEN, MLP_HIDDEN), nn.GELU(approximate="tanh"),
                nn.Dropout(DROPOUT))
            self.head_r = nn.Linear(MLP_HIDDEN, 1)
            self.head_lv = nn.Linear(MLP_HIDDEN, 1)

        def set_norm(self, ts_mean, ts_std, q_mean, q_std):
            self.ts_mean.copy_(torch.as_tensor(ts_mean, dtype=torch.float32))
            self.ts_std.copy_(torch.as_tensor(ts_std, dtype=torch.float32))
            self.q_mean.copy_(torch.as_tensor(q_mean, dtype=torch.float32))
            self.q_std.copy_(torch.as_tensor(q_std, dtype=torch.float32))

        def forward(self, ts, query, lengths, pos_ids):
            ts = (ts - self.ts_mean) / self.ts_std
            query = (query - self.q_mean) / self.q_std
            emb = self.pos_emb(pos_ids)                       # [B, P]
            seq = torch.cat([ts, emb.unsqueeze(1).expand(-1, ts.size(1), -1)], dim=-1)
            out, _ = self.lstm(seq)                           # [B, L, H]
            idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out.size(-1))
            h_last = out.gather(1, idx).squeeze(1)            # [B, H]  (true last step)
            q_repr = self.qmlp(torch.cat([query, emb], dim=-1))
            fused = self.fuse(torch.cat([h_last, q_repr], dim=-1))
            r_hat = self.head_r(fused).squeeze(-1)
            log_var = self.head_lv(fused).squeeze(-1)
            sigma = torch.nn.functional.softplus(log_var) + SIGMA_FLOOR
            return r_hat, sigma

        @torch.no_grad()
        def export_npz(self, path):
            sd = self.state_dict()
            w = {
                "ts_mean": sd["ts_mean"].cpu().numpy(),
                "ts_std": sd["ts_std"].cpu().numpy(),
                "q_mean": sd["q_mean"].cpu().numpy(),
                "q_std": sd["q_std"].cpu().numpy(),
                "pos_emb": sd["pos_emb.weight"].cpu().numpy(),
                "qmlp_w": sd["qmlp.0.weight"].cpu().numpy(),
                "qmlp_b": sd["qmlp.0.bias"].cpu().numpy(),
                "fuse_w": sd["fuse.0.weight"].cpu().numpy(),
                "fuse_b": sd["fuse.0.bias"].cpu().numpy(),
                "head_r_w": sd["head_r.weight"].cpu().numpy(),
                "head_r_b": sd["head_r.bias"].cpu().numpy(),
                "head_lv_w": sd["head_lv.weight"].cpu().numpy(),
                "head_lv_b": sd["head_lv.bias"].cpu().numpy(),
                "meta": np.array({"F": self.F, "Q": self.Q, "hidden": HIDDEN,
                                  "layers": LSTM_LAYERS, "pos_emb_dim": POS_EMB_DIM},
                                 dtype=object),
            }
            for k in range(LSTM_LAYERS):
                w[f"lstm_w_ih_l{k}"] = sd[f"lstm.weight_ih_l{k}"].cpu().numpy()
                w[f"lstm_w_hh_l{k}"] = sd[f"lstm.weight_hh_l{k}"].cpu().numpy()
                w[f"lstm_b_ih_l{k}"] = sd[f"lstm.bias_ih_l{k}"].cpu().numpy()
                w[f"lstm_b_hh_l{k}"] = sd[f"lstm.bias_hh_l{k}"].cpu().numpy()
            np.savez(path, **w)
            return path

    return ResidualLSTM()


def gaussian_nll(r_hat, sigma, y):
    """0.5 * [ log sigma^2 + (y - r_hat)^2 / sigma^2 ]  (torch tensors)."""
    import torch
    var = sigma ** 2
    return 0.5 * (torch.log(var) + (y - r_hat) ** 2 / var)
