"""
pipeline/stage10_train.py
Stage 10 Phase 1 — Step 3: walk-forward pretrain of the LSTM residual model.

RUN ON COLAB (Smart App Control blocks torch on the dev machine — option C).
Produces, per fold, into models/stage10/:
    pretrain_<valseason>.pt      torch checkpoint
    pretrain_<valseason>.npz     numpy weights (runtime path, option D)
    stage10_config.json          arch + hyperparams + feature spec
    stage10_calibration.json     NLL curves + gate 2 (MAE) + gate 3 (q90 coverage)

Gates checked here (plan §6):
  Gate 2  OOF MAE(mu_gbm + r_hat) <= MAE(mu_gbm)  per (position x fold); mean
          improvement > 0 with a bootstrap CI that excludes 0.  RED LINE: any
          position MAE regression vs raw GBM.
  Gate 3  q90 coverage  P(actual <= q90)  in [0.88, 0.92]   (baseline 0.836).

Usage (Colab):
    python pipeline/stage10_train.py --pretrain
    python pipeline/stage10_train.py --pretrain --ablation        # + 4 position-specific nets
    python pipeline/stage10_train.py --check-numpy                # torch vs numpy_forward
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

import stage10_sequence as sq                       # noqa: E402
import stage10_model as sm                          # noqa: E402

OUT_DIR = os.path.join(_ROOT, "models", "stage10")

# LSTM-eligible seasons: 2019-20 has no OOF residuals (no prior season) so it is
# GBM-history-only and never a residual target season.
FOLDS = [
    (["2020-21"],                                                   "2021-22"),
    (["2020-21", "2021-22"],                                        "2022-23"),
    (["2020-21", "2021-22", "2022-23"],                             "2023-24"),
    (["2020-21", "2021-22", "2022-23", "2023-24"],                  "2024-25"),
    (["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"],       "2025-26"),
]
SEASON_RECENCY = [1.0, 1.5, 2.0, 2.5, 3.0]          # mirrors Stage 7 fold weights

# training hyperparameters (plan §4.2)
MAX_EPOCHS = 80
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 8
SEED = 42
MAX_LEN = 38


# ---------------------------------------------------------------------------
def set_determinism(seed=SEED):
    import torch
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:                              # noqa: BLE001
        pass
    torch.set_num_threads(1)


def norm_stats(ts, lengths, query):
    """mean/std over real timesteps of the TRAIN split only; binary/flag columns
    guarded (std -> 1)."""
    mask = np.zeros((ts.shape[0], ts.shape[1]), dtype=bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True
    flat = ts[mask]
    ts_mean = flat.mean(0)
    ts_std = flat.std(0)
    ts_std[ts_std < 1e-6] = 1.0
    q_mean = query.mean(0)
    q_std = query.std(0)
    q_std[q_std < 1e-6] = 1.0
    return (ts_mean.astype(np.float32), ts_std.astype(np.float32),
            q_mean.astype(np.float32), q_std.astype(np.float32))


def season_weights(meta, train_seasons):
    order = {s: SEASON_RECENCY[-(len(train_seasons)):][i]
             for i, s in enumerate(sorted(train_seasons))}
    return meta["season"].map(order).fillna(1.0).to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
def train_one_fold(train_seasons, val_season, joined, *, pos_filter=None,
                   device="cpu", verbose=True):
    import torch

    set_determinism()
    tr_s = sq.build_samples(train_seasons, joined=joined)
    va_s = sq.build_samples([val_season], joined=joined)
    if pos_filter:
        tr_s = [s for s in tr_s if s.position == pos_filter]
        va_s = [s for s in va_s if s.position == pos_filter]

    Xtr, Qtr, Ltr, Ptr, Ytr, Mtr = sq.to_padded_arrays(tr_s, MAX_LEN)
    Xva, Qva, Lva, Pva, Yva, Mva = sq.to_padded_arrays(va_s, MAX_LEN)

    # widen the narrower split to the common width (right-pad with zeros only —
    # sequences are front-aligned at 0..length-1, so widening never drops or
    # reorders a real timestep)
    def _widen(X, w):
        if X.shape[1] >= w:
            return X
        out = np.zeros((X.shape[0], w, X.shape[2]), dtype=X.dtype)
        out[:, :X.shape[1]] = X
        return out
    W = max(Xtr.shape[1], Xva.shape[1])
    Xtr, Xva = _widen(Xtr, W), _widen(Xva, W)

    ts_mean, ts_std, q_mean, q_std = norm_stats(Xtr, Ltr, Qtr)
    wtr = season_weights(Mtr, train_seasons)

    model = sm.build_torch_model(sq.F, sq.Q, seed=SEED).to(device)
    model.set_norm(ts_mean, ts_std, q_mean, q_std)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    def batches(n, bs, shuffle):
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(SEED)
            rng.shuffle(idx)
        for i in range(0, n, bs):
            yield idx[i:i + bs]

    tt = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)  # noqa: E731
    Xtr_t, Qtr_t = tt(Xtr), tt(Qtr)
    Ltr_t, Ptr_t = tt(Ltr, torch.long), tt(Ptr, torch.long)
    Ytr_t, wtr_t = tt(Ytr), tt(wtr)
    Xva_t, Qva_t = tt(Xva), tt(Qva)
    Lva_t, Pva_t, Yva_t = tt(Lva, torch.long), tt(Pva, torch.long), tt(Yva)

    curve = []
    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        tl = tn = 0.0
        for b in batches(len(Xtr), BATCH, shuffle=True):
            opt.zero_grad()
            r, s = model(Xtr_t[b], Qtr_t[b], Ltr_t[b], Ptr_t[b])
            loss = (sm.gaussian_nll(r, s, Ytr_t[b]) * wtr_t[b]).sum() / wtr_t[b].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl += float(loss) * len(b)
            tn += len(b)
        sched.step()

        model.eval()
        with torch.no_grad():
            rv, svv = model(Xva_t, Qva_t, Lva_t, Pva_t)
            vnll = float(sm.gaussian_nll(rv, svv, Yva_t).mean())
            vmae_raw = float(np.abs(Yva).mean())
            vmae_cor = float(np.abs(Yva - rv.cpu().numpy()).mean())
        curve.append({"epoch": ep, "train_nll": tl / tn, "val_nll": vnll,
                      "val_mae_gbm": vmae_raw, "val_mae_corrected": vmae_cor})
        if verbose:
            print(f"    ep{ep:02d}  train_nll={tl/tn:6.3f}  val_nll={vnll:6.3f}  "
                  f"val_MAE {vmae_raw:.3f} -> {vmae_cor:.3f}")
        if vnll < best_val - 1e-4:
            best_val, best_state, bad = vnll, {k: v.detach().clone()
                                               for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                if verbose:
                    print(f"    early stop @ ep{ep} (best val_nll={best_val:.3f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        r_va, s_va = model(Xva_t, Qva_t, Lva_t, Pva_t)
    r_va, s_va = r_va.cpu().numpy(), s_va.cpu().numpy()
    return model, curve, dict(
        meta=Mva, y=Yva, r_hat=r_va, sigma=s_va,
        norm=(ts_mean, ts_std, q_mean, q_std),
        arrays=(Xva, Qva, Lva, Pva))


# ---------------------------------------------------------------------------
def eval_gates(val_season, ev):
    """Gate 2 (MAE, gated + ungated) and Gate 3 (q90 coverage) for one fold."""
    m = ev["meta"].copy()
    m["y"] = ev["y"]
    m["r_hat"] = ev["r_hat"]
    m["sigma"] = ev["sigma"]
    r_app, s_eff = sm.apply_gate(m["r_hat"].to_numpy(), m["sigma"].to_numpy(),
                                 m["n_played"].to_numpy(), m["position"].tolist())
    m["r_applied"] = r_app
    m["mae_gbm"] = m["y"].abs()
    m["mae_ungated"] = (m["y"] - m["r_hat"]).abs()
    m["mae_gated"] = (m["y"] - m["r_applied"]).abs()
    q90 = sm.q90_from(m["mu_gbm_t"].to_numpy(), r_app, s_eff)
    m["q90_cover"] = (m["actual_t"].to_numpy() <= q90).astype(float)

    def boot_ci(delta, n=2000):
        rng = np.random.default_rng(SEED)
        d = delta.to_numpy()
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    rows = []
    for pos in ["ALL"] + sm.POSITIONS:
        sub = m if pos == "ALL" else m[m["position"] == pos]
        if not len(sub):
            continue
        d_gate = sub["mae_gbm"] - sub["mae_gated"]         # >0 = improvement
        lo, hi = boot_ci(d_gate)
        rows.append(dict(
            fold=val_season, position=pos, n=int(len(sub)),
            mae_gbm=float(sub["mae_gbm"].mean()),
            mae_ungated=float(sub["mae_ungated"].mean()),
            mae_gated=float(sub["mae_gated"].mean()),
            delta_gated=float(d_gate.mean()), ci_lo=lo, ci_hi=hi,
            regression=bool(sub["mae_gated"].mean() > sub["mae_gbm"].mean() + 1e-9),
            q90_coverage=float(sub["q90_cover"].mean()),
        ))
    return rows


def print_gate_table(all_rows):
    print("\n" + "=" * 96)
    print("  GATE 2 — OOF residual MAE   |   GATE 3 — q90 coverage        "
          "(RED LINE: any position MAE regression)")
    print("=" * 96)
    hdr = f"  {'fold':9} {'pos':4} {'n':>6} {'MAE_gbm':>8} {'MAE_cor':>8} " \
          f"{'delta':>7} {'95% CI':>16} {'q90cov':>7}  flag"
    print(hdr)
    print("  " + "-" * 92)
    red = []
    for r in all_rows:
        flag = ""
        if r["regression"]:
            flag = "  <-- REGRESSION"
            red.append(r)
        elif r["ci_lo"] > 0:
            flag = "  ok (CI>0)"
        print(f"  {r['fold']:9} {r['position']:4} {r['n']:6d} "
              f"{r['mae_gbm']:8.3f} {r['mae_gated']:8.3f} {r['delta_gated']:+7.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] {r['q90_coverage']:7.3f}{flag}")
    print("  " + "-" * 92)
    if red:
        print(f"  RED LINE HIT: {len(red)} (fold,position) with MAE regression:")
        for r in red:
            print(f"     {r['fold']} {r['position']}: "
                  f"{r['mae_gbm']:.3f} -> {r['mae_gated']:.3f} ({r['delta_gated']:+.3f})")
    else:
        print("  no MAE regression in any (fold, position). ")
    allrows = [r for r in all_rows if r["position"] == "ALL"]
    if allrows:
        mean_cov = np.mean([r["q90_coverage"] for r in allrows])
        mean_delta = np.mean([r["delta_gated"] for r in allrows])
        print(f"  overall: mean ALL-delta {mean_delta:+.3f} | "
              f"mean ALL q90 coverage {mean_cov:.3f}  (target [0.88, 0.92])")
    return red


def nll_sparkline(curve):
    v = [c["val_nll"] for c in curve]
    lo, hi = min(v), max(v)
    blocks = "▁▂▃▄▅▆▇█"
    if hi - lo < 1e-9:
        return blocks[0] * len(v)
    return "".join(blocks[min(7, int(7 * (x - lo) / (hi - lo)))] for x in v)


# ---------------------------------------------------------------------------
def run_pretrain(ablation=False, device="cpu"):
    os.makedirs(OUT_DIR, exist_ok=True)
    joined = sq.load_joined()
    calib = {"folds": [], "gate_rows": [], "ablation": []}
    all_rows = []

    for train_seasons, val_season in FOLDS:
        print("\n" + "#" * 80)
        print(f"#  FOLD  train={train_seasons}  ->  val={val_season}")
        print("#" * 80)
        t0 = time.time()
        model, curve, ev = train_one_fold(train_seasons, val_season, joined,
                                          device=device)
        pt = os.path.join(OUT_DIR, f"pretrain_{val_season}.pt")
        npz = os.path.join(OUT_DIR, f"pretrain_{val_season}.npz")
        import torch
        torch.save(model.state_dict(), pt)
        model.export_npz(npz)

        rows = eval_gates(val_season, ev)
        all_rows += rows
        calib["folds"].append({
            "val_season": val_season, "train_seasons": train_seasons,
            "epochs": len(curve), "best_val_nll": min(c["val_nll"] for c in curve),
            "val_nll_sparkline": nll_sparkline(curve), "curve": curve,
            "seconds": round(time.time() - t0, 1)})
        print(f"  val_nll curve  {nll_sparkline(curve)}  "
              f"({curve[0]['val_nll']:.3f} -> {min(c['val_nll'] for c in curve):.3f})")

        if ablation:
            print("  [ABLATION] position-specific nets:")
            for pos in sm.POSITIONS:
                _, _, evp = train_one_fold(train_seasons, val_season, joined,
                                           pos_filter=pos, device=device, verbose=False)
                mp = evp["meta"].copy()
                r_app, _ = sm.apply_gate(evp["r_hat"], evp["sigma"],
                                         mp["n_played"].to_numpy(), mp["position"].tolist())
                mae_c = float(np.abs(evp["y"] - r_app).mean())
                mae_g = float(np.abs(evp["y"]).mean())
                shared = next(r for r in rows if r["position"] == pos)
                print(f"     {pos}: shared {shared['mae_gbm']:.3f}->{shared['mae_gated']:.3f}"
                      f"   posnet {mae_g:.3f}->{mae_c:.3f}")
                calib["ablation"].append(dict(fold=val_season, position=pos,
                                              shared_mae=shared["mae_gated"],
                                              posnet_mae=mae_c, mae_gbm=mae_g))

    calib["gate_rows"] = all_rows
    red = print_gate_table(all_rows)
    calib["red_line_hits"] = red

    cfg = {
        "feat_spec": sq.SAMPLE_SPEC,
        "arch": {"hidden": sm.HIDDEN, "lstm_layers": sm.LSTM_LAYERS,
                 "pos_emb_dim": sm.POS_EMB_DIM, "mlp_hidden": sm.MLP_HIDDEN,
                 "dropout": sm.DROPOUT, "sigma_floor": sm.SIGMA_FLOOR},
        "train": {"max_epochs": MAX_EPOCHS, "batch": BATCH, "lr": LR,
                  "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                  "patience": PATIENCE, "seed": SEED, "max_len": MAX_LEN},
        "gate": {"z90": sm.Z90, "conf_ramp": sm.CONF_RAMP,
                 "sigma_shrink_below": sm.SIGMA_SHRINK_BELOW,
                 "headroom_prior": sm.HEADROOM_PRIOR},
        "folds": [{"train": t, "val": v} for t, v in FOLDS],
    }
    with open(os.path.join(OUT_DIR, "stage10_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(OUT_DIR, "stage10_calibration.json"), "w") as f:
        json.dump(calib, f, indent=2, default=float)
    print(f"\n  wrote {OUT_DIR}/stage10_config.json + stage10_calibration.json")
    return 1 if red else 0


def check_numpy(val_season="2024-25"):
    """torch model vs numpy_forward on the val set — must agree < 1e-4."""
    import torch
    joined = sq.load_joined()
    train_seasons = next(t for t, v in FOLDS if v == val_season)
    model, _, ev = train_one_fold(train_seasons, val_season, joined, verbose=False)
    npz = os.path.join(OUT_DIR, f"_check_{val_season}.npz")
    model.export_npz(npz)
    Xva, Qva, Lva, Pva = ev["arrays"]
    with torch.no_grad():
        rt, st = model(torch.as_tensor(Xva), torch.as_tensor(Qva),
                       torch.as_tensor(Lva, dtype=torch.long),
                       torch.as_tensor(Pva, dtype=torch.long))
    npmod = sm.NumpyResidualLSTM(npz)
    rn, sn = npmod.predict(Xva, Qva, Lva, Pva)
    dr = np.abs(rt.numpy() - rn).max()
    ds = np.abs(st.numpy() - sn).max()
    os.remove(npz)
    print(f"  torch vs numpy  max|dr|={dr:.2e}  max|dsigma|={ds:.2e}  "
          f"({'PASS' if max(dr, ds) < 1e-4 else 'FAIL'})")
    return 0 if max(dr, ds) < 1e-4 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--check-numpy", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    rc = 0
    if args.pretrain:
        rc = run_pretrain(ablation=args.ablation, device=args.device)
    if args.check_numpy:
        rc |= check_numpy()
    sys.exit(rc)
