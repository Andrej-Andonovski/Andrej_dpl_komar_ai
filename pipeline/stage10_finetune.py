"""
pipeline/stage10_finetune.py
Stage 10 Phase 1 — step 6: in-season head fine-tune.

OFFLINE (needs torch). For a backtest season S, walk t = FT_MIN_GW .. 38 in
strides of FT_STRIDE and, from the frozen pretrain_<S> checkpoint, fine-tune
ONLY the head stack (qmlp + fuse + head_r + head_lv — the LSTM trunk, position
embedding and norm buffers stay frozen) on season-S residual samples with
target GW < t. Writes ft_<S>_gw<t>.npz next to the pretrain checkpoints;
stage10_infer picks the largest-gw ft checkpoint <= the decision GW, else falls
back to pretrain_<S>.

Rationale (plan §4.3): the trunk learns "how a trajectory shapes the residual"
from 5 seasons; only the head needs to adapt to the current season's scoring
environment. Cheap, and stable on the thin GW<t sample.

Determinism (gate 8): fully seeded; re-running produces byte-identical .npz.
The sim that consumes these stays numpy-only, so a STAGE10=on season run is
deterministic by construction.

Usage:
    python pipeline/stage10_finetune.py --season 2024-25
    python pipeline/stage10_finetune.py --season 2024-25 --season 2023-24 --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                    # noqa: BLE001
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import stage10_sequence as sq                        # noqa: E402
import stage10_model as sm                           # noqa: E402
import stage10_train as st                           # noqa: E402

OUT_DIR = os.path.join(_ROOT, "models", "stage10")

FT_MIN_GW = int(os.environ.get("FT_MIN_GW", "5"))    # skip t<=4 (thin history)
FT_STRIDE = int(os.environ.get("FT_STRIDE", "4"))
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "12"))
FT_LR = float(os.environ.get("FT_LR", "3e-4"))
FT_SEED = 42
FROZEN_PREFIXES = ("pos_emb", "lstm", "ts_mean", "ts_std", "q_mean", "q_std")


def _load_pretrain(F, Q, npz_path):
    """Rebuild the torch model and load the pretrain weights from its .npz."""
    import torch
    z = np.load(npz_path, allow_pickle=True)
    w = {k: z[k] for k in z.files if k != "meta"}
    m = sm.build_torch_model(F, Q, seed=FT_SEED)
    m.set_norm(w["ts_mean"], w["ts_std"], w["q_mean"], w["q_std"])
    sd = m.state_dict()
    sd["pos_emb.weight"] = torch.tensor(w["pos_emb"])
    sd["qmlp.0.weight"] = torch.tensor(w["qmlp_w"]); sd["qmlp.0.bias"] = torch.tensor(w["qmlp_b"])
    sd["fuse.0.weight"] = torch.tensor(w["fuse_w"]); sd["fuse.0.bias"] = torch.tensor(w["fuse_b"])
    sd["head_r.weight"] = torch.tensor(w["head_r_w"]); sd["head_r.bias"] = torch.tensor(w["head_r_b"])
    sd["head_lv.weight"] = torch.tensor(w["head_lv_w"]); sd["head_lv.bias"] = torch.tensor(w["head_lv_b"])
    for k in range(sm.LSTM_LAYERS):
        sd[f"lstm.weight_ih_l{k}"] = torch.tensor(w[f"lstm_w_ih_l{k}"])
        sd[f"lstm.weight_hh_l{k}"] = torch.tensor(w[f"lstm_w_hh_l{k}"])
        sd[f"lstm.bias_ih_l{k}"] = torch.tensor(w[f"lstm_b_ih_l{k}"])
        sd[f"lstm.bias_hh_l{k}"] = torch.tensor(w[f"lstm_b_hh_l{k}"])
    m.load_state_dict(sd)
    return m


def _finetune_one(model, samples, seed=FT_SEED):
    """Fine-tune the head stack on `samples`; returns (model, z_cal, n)."""
    import torch
    st.set_determinism(seed)
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith(FROZEN_PREFIXES)

    Xtr, Qtr, Ltr, Ptr, Ytr, Mtr = sq.to_padded_arrays(samples, st.MAX_LEN)
    tt = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d)         # noqa: E731
    Xt, Qt, Lt, Pt, Yt = tt(Xtr), tt(Qtr), tt(Ltr, torch.long), tt(Ptr, torch.long), tt(Ytr)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=FT_LR, weight_decay=st.WEIGHT_DECAY)
    n = len(samples)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(FT_EPOCHS):
        idx = rng.permutation(n)
        for i in range(0, n, st.BATCH):
            b = idx[i:i + st.BATCH]
            opt.zero_grad()
            r, s, lv = model(Xt[b], Qt[b], Lt[b], Pt[b])
            loss = (sm.gaussian_nll(r, s, Yt[b]).mean()
                    + st.LAMBDA_R * (r ** 2).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], st.GRAD_CLIP)
            opt.step()

    model.eval()
    with torch.no_grad():
        r, s, _ = model(Xt, Qt, Lt, Pt)
    r, s = r.cpu().numpy(), s.cpu().numpy()
    ra, se = sm.apply_gate(r, s, Mtr["n_played"].to_numpy(), Mtr["position"].tolist())
    std = (Ytr - ra) / np.maximum(se, 1e-6)
    pos = Mtr["position"].to_numpy()
    z_cal = {p: (max(sm.Z90, float(np.quantile(std[pos == p], 0.915)))
                 if (pos == p).sum() > 30 else sm.Z90) for p in sm.POSITIONS}
    return model, z_cal, n


def run_season(season, joined, report=False):
    pre = os.path.join(OUT_DIR, f"pretrain_{season}.npz")
    if not os.path.exists(pre):
        print(f"  [FT] no pretrain checkpoint for {season} — skip"); return []
    all_s = sq.build_samples([season], joined=joined)
    by_gw = {}
    for s in all_s:
        by_gw.setdefault(s.target_gw, []).append(s)
    made = []
    for t in range(FT_MIN_GW, 39, FT_STRIDE):
        train = [s for s in all_s if s.target_gw < t]
        if len(train) < 200:
            continue
        t0 = time.time()
        model = _load_pretrain(sq.F, sq.Q, pre)
        model, z_cal, n = _finetune_one(model, train)
        out = os.path.join(OUT_DIR, f"ft_{season}_gw{t}.npz")
        model.export_npz(out, z_cal=z_cal)
        line = f"  [FT] {season} gw{t:2d}: {n:5d} train rows  ({time.time()-t0:.0f}s)"
        if report:
            hold = [s for s in all_s if t <= s.target_gw < t + FT_STRIDE]
            if hold:
                line += "  " + _cmp_holdout(pre, out, hold)
        print(line)
        made.append(out)
    return made


def _cmp_holdout(pre_npz, ft_npz, samples):
    """MAE(pretrain) vs MAE(ft) on the next-stride holdout GWs."""
    ts, q, ln, pid, y, meta = sq.to_padded_arrays(samples, st.MAX_LEN)
    npl, pos = meta["n_played"].to_numpy(), meta["position"].tolist()
    res = {}
    for tag, path in (("pre", pre_npz), ("ft", ft_npz)):
        net = sm.NumpyResidualLSTM(path)
        r, s = net.predict(ts, q, ln, pid)
        ra, _ = sm.apply_gate(r, s, npl, pos)
        res[tag] = float(np.abs(y - ra).mean())
    d = res["pre"] - res["ft"]
    return f"holdout MAE pre {res['pre']:.3f} -> ft {res['ft']:.3f} ({d:+.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", required=True)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    joined = sq.load_joined()
    summary = {}
    for s in args.season:
        print(f"\n### fine-tune {s}  (min_gw={FT_MIN_GW} stride={FT_STRIDE} "
              f"epochs={FT_EPOCHS} lr={FT_LR}) ###")
        summary[s] = run_season(s, joined, report=args.report)
    with open(os.path.join(OUT_DIR, "stage10_finetune.json"), "w") as f:
        json.dump({"config": {"ft_min_gw": FT_MIN_GW, "ft_stride": FT_STRIDE,
                              "ft_epochs": FT_EPOCHS, "ft_lr": FT_LR,
                              "frozen": list(FROZEN_PREFIXES)},
                   "checkpoints": {k: [os.path.basename(p) for p in v]
                                   for k, v in summary.items()}}, f, indent=2)
    print(f"\n  wrote {OUT_DIR}/stage10_finetune.json")


if __name__ == "__main__":
    main()
