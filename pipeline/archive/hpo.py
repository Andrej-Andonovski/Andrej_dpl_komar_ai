"""
Hyperparameter optimisation for intel_06_optimizer.
Two demos:
  Demo 1: HPO over GW1-10  (sim_start = auto-detected real GW30)
  Demo 2: Apply best Demo-1 params to build the strongest team for GW29
          Simulates GW1-28 (sim_start_override=1) to accumulate squad
          context and retrained models, then shows the GW29 prediction.
"""
import optuna
import copy
import pickle
from collections import defaultdict
from pipeline.intel_06_optimizer import load_all_data, run_simulation

optuna.logging.set_verbosity(optuna.logging.WARNING)


def clone_data(data):
    """Deep-copy mutable parts of data; share read-only parts."""
    cloned = {}
    # Mutable — must copy per trial
    cloned['models'] = {k: pickle.loads(pickle.dumps(v))
                        for k, v in data['models'].items()}
    cloned['model_fcols'] = copy.deepcopy(data['model_fcols'])
    cloned['best_params'] = copy.deepcopy(data['best_params'])
    cloned['train_dfs']   = {k: v.copy() for k, v in data['train_dfs'].items()}
    cloned['player_pool'] = copy.deepcopy(data['player_pool'])
    # Read-only — safe to share
    for k in ('upcoming', 'dgw_gws', 'bgw_by_gw',
              'hist_lookup', 'fdr_lookup',
              'avail_data', 'rot_data', 'recommendations',
              'sim_start_gw'):
        cloned[k] = data[k]
    return cloned


def make_objective(base_data, n_gws):
    def objective(trial):
        hp = {
            'bb_threshold':    trial.suggest_float('bb_threshold',    6.0, 15.0),
            'tc_threshold':    trial.suggest_float('tc_threshold',     7.0, 14.0),
            'fdr_mult':        trial.suggest_float('fdr_mult',         0.01, 0.10),
            'loyalty_lockout': trial.suggest_float('loyalty_lockout',  5.0, 25.0),
            'loyalty_mid':     trial.suggest_float('loyalty_mid',      1.0,  6.0),
            'loyalty_late':    trial.suggest_float('loyalty_late',     0.5,  3.0),
        }
        trial_data = clone_data(base_data)
        score = run_simulation(trial_data, hp=hp, n_gws=n_gws,
                               verbose=False, save_json=False)
        return score
    return objective


def run_hpo(data, n_gws, n_trials, label):
    print(f"\n{'='*60}")
    print(f"  HPO: {label}  ({n_trials} trials, {n_gws} GWs)")
    print(f"{'='*60}")

    # Baseline score with default hp
    baseline_data = clone_data(data)
    baseline = run_simulation(baseline_data, hp={}, n_gws=n_gws,
                              verbose=False, save_json=False)
    print(f"  Baseline (default params): {baseline} pts")

    study = optuna.create_study(direction='maximize',
                                study_name=label,
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(data, n_gws), n_trials=n_trials,
                   show_progress_bar=True)

    best = study.best_trial
    print(f"\n  Best score : {best.value} pts  (+{best.value - baseline:+.0f} vs baseline)")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k:<20} = {v:.4f}")

    # Top 5 trials
    print(f"\n  Top 5 trials:")
    top5 = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
    for i, t in enumerate(top5, 1):
        print(f"    #{i}: {t.value} pts")

    return study


if __name__ == '__main__':

    N_TRIALS = 15  # ~2.5 min/trial → ~6 min for Demo 1

    # ---------------------------------------------------------------
    # Demo 1: HPO over GW1-10  (real GW30-39 backtest)
    # ---------------------------------------------------------------
    print("Loading data for Demo 1 (GW1-10 HPO)...")
    data1 = load_all_data(verbose=False)
    study1 = run_hpo(data1, n_gws=10, n_trials=N_TRIALS, label='HPO_GW1_10')
    best_hp = study1.best_params

    # ---------------------------------------------------------------
    # Demo 2: Strongest possible team for GW29 (fresh account)
    #   - No squad history, no warmup — fresh pick for GW29 only
    #   - sim_start_override=29 → sim GW1 maps to real GW29
    #   - n_gws=1 → single GW pick, GW1 is always a free squad pick
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  DEMO 2: STRONGEST FRESH TEAM FOR GW29")
    print(f"  Using best HPO params from Demo 1 (fresh account, 1 GW)")
    print(f"{'='*60}")
    print("\nLoading data for GW29 fresh pick...")
    data2 = load_all_data(verbose=False, sim_start_override=29)

    # Unlock chips for GW29 — no lockout on a fresh account
    demo2_hp = dict(best_hp)
    demo2_hp['tc_min_gw'] = 1
    demo2_hp['bb_min_gw'] = 1

    print("\nPicking best possible squad for GW29 (unconstrained, chips enabled)...\n")
    run_simulation(data2, hp=demo2_hp, n_gws=1, verbose=True, save_json=False)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  HPO SUMMARY")
    print(f"{'='*60}")
    print(f"  Demo 1 (GW1-10)  best: {study1.best_value} pts  (+{study1.best_value - 660:+d} vs 660 default)")
    print(f"  Best params:")
    for k, v in best_hp.items():
        print(f"    {k:<20} = {v:.4f}")
    print(f"\n  Demo 2: GW29 fresh squad shown above (strongest team, best params, no prior squad)")
