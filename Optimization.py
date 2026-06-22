"""
Optuna Optimization for Theta Rhythm Model
===========================================

Maximizes the theta-rhythm quality (low phase error) of the
mean-field model over the 9-dimensional parameter space using
Optuna's TPE sampler.

The objective function returns the phase-error code from
OscillationAnalyzer.analyze_oscillation, which is 0 when the
phase difference exactly matches the 150 deg target and increases
linearly with the deviation.

Storage is a SQLite database, so the study can be resumed across
runs and the best parameters are persisted.

Author: S.N. Skorokhod, S.V. Dubrovin, I.E. Mysin (refactored 2026)
"""

import numpy as np
import optuna
from tqdm import tqdm

from InhOsc import (
    ModelParameters, ThetaRhythmNetwork, OscillationAnalyzer,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Objective
# =============================================================================

def objective(trial):
    """Suggest 9 parameters and return the phase-error code (to maximize)."""
    x1 = trial.suggest_float("g12",   0, 10000)
    x2 = trial.suggest_float("g21",   0, 10000)
    x3 = trial.suggest_float("I1",    0, 600)
    x4 = trial.suggest_float("I2",    0, 600)
    x5 = trial.suggest_float("Delta", 0, 100)
    x6 = trial.suggest_float("tau_d", 1, 21)
    x7 = trial.suggest_float("tau_r", 50, 2000)
    x8 = trial.suggest_float("tau_f", 3, 300)
    x9 = trial.suggest_float("Uinc",  0, 1)

    params = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9])

    model_params = ModelParameters.from_array(params)
    network = ThetaRhythmNetwork(model_params)
    mf_results = network.simulate_meanfield(t_max=1600, dt=0.05)

    transient = int(600 / 0.05)
    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=0.05, transient_samples=transient
    )

    # error_code == phase_error in degrees; lower is better, so maximize the
    # negative error (equivalent to minimizing the error).
    return mf_analysis.error_code


# =============================================================================
# Run the study
# =============================================================================

if __name__ == "__main__":
    # SQLite-backed storage: the study is resumed automatically across runs.
    storage = "sqlite:///gb_optuna_study.db"
    study = optuna.create_study(
        study_name="gb_jt_study",
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )

    print(study)

    n_trials = 50000
    with tqdm(total=n_trials, desc="Parallel Optuna") as pbar:
        def callback(study, trial):
            pbar.update(1)

        study.optimize(objective, n_trials=n_trials, n_jobs=-1,
                       callbacks=[callback])

    # ----- Report the best trial -----
    print("\nBest trial:")
    print(f"  Value (phase error, degrees): {study.best_value:.2f}")
    print("  Parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v:.3f}")

    # ----- Re-run the best trial to inspect all metrics -----
    best_params_np = np.array([
        study.best_params[name] for name in study.best_params.keys()
    ])

    model_params = ModelParameters.from_array(best_params_np)
    network = ThetaRhythmNetwork(model_params)
    mf_results = network.simulate_meanfield(t_max=1300, dt=0.02)
    transient = int(300 / 0.02)
    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=0.02, transient_samples=transient
    )
    print("\nFull analysis of the best trial:")
    print(mf_analysis)
