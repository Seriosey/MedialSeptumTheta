"""
Grid-Search Pipeline for Theta Rhythm Model
===========================================

End-to-end pipeline for sampling the 9-dimensional parameter space,
running mean-field simulations, and analyzing the results with
Random Forest classifiers/regressors and Permutation Importance.

Pipeline stages:
    1. Sobol low-discrepancy sampling of the parameter space.
    2. Mean-field simulation + theta-rhythm analysis for each sample.
    3. Random Forest classifier (is_theta) and regressors
       (amplitude, frequency, phase error).
    4. Permutation Importance for each target.
    5. Visualization: importance bar plots, marginal distributions,
       2D heatmap, optimal-region summary.

Author: S.N. Skorokhod, S.V. Dubrovin, I.E. Mysin (refactored 2026)
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import matplotlib.pyplot as plt

from InhOsc import (
    ModelParameters, ThetaRhythmNetwork, OscillationAnalyzer,
)


# =============================================================================
# Simulation wrapper
# =============================================================================

def simulate_theta_rhythm(params):
    """
    Run a single mean-field simulation and return theta-rhythm metrics.

    Args:
        params: dict or ModelParameters-like object with parameter values.

    Returns:
        Tuple (is_theta, amplitude, error_code, frequency).
    """
    dt = 0.05
    model_params = ModelParameters.from_dict(params)
    network = ThetaRhythmNetwork(model_params)
    mf_results = network.simulate_meanfield(t_max=1600, dt=dt)
    transient = int(600 / dt)
    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=dt, transient_samples=transient
    )
    return (mf_analysis.is_theta, mf_analysis.amplitude,
            mf_analysis.error_code, mf_analysis.frequency)


# =============================================================================
# 1. Sobol sampling
# =============================================================================

def optimal_grid_sampling(n_samples, param_bounds, seed=42):
    """
    Low-discrepancy sampling of the parameter space using a scrambled
    Sobol sequence.

    Sobol sequences provide better uniformity than Latin Hypercube
    Sampling in high dimensions and avoid the "curse of dimensionality"
    of full grid sampling (e.g. 4^9 = 262k samples for 9 parameters
    with only 4 values each).

    Args:
        n_samples: Number of samples to draw.
        param_bounds: dict {name: (low, high)}.
        seed: Random seed.

    Returns:
        DataFrame of shape (n_samples, n_params) with named columns.
    """
    from scipy.stats import qmc

    np.random.seed(seed)
    n_params = len(param_bounds)
    param_names = list(param_bounds.keys())

    sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)

    # Sobol requires powers of two; generate the next power of two and slice.
    n_power_of_2 = 2 ** int(np.ceil(np.log2(n_samples)))
    sample_matrix = sampler.random(n_power_of_2)[:n_samples]

    samples = np.zeros((n_samples, n_params))
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        samples[:, i] = low + sample_matrix[:, i] * (high - low)

    return pd.DataFrame(samples, columns=param_names)


# =============================================================================
# 2. Parallel-friendly simulation loop with checkpoints
# =============================================================================

def run_simulations_parallel(samples, n_workers=-1,
                             checkpoint_dir='checkpoints', verbose=True):
    """
    Run simulations for all samples with periodic checkpointing.

    Args:
        samples: DataFrame of parameters.
        n_workers: Number of parallel workers (kept for API compatibility;
            the loop is sequential).
        checkpoint_dir: Directory for checkpoint files.
        verbose: If True, print progress.

    Returns:
        Array of shape (n_samples, 4) with columns
        (is_theta, amplitude, error_code, frequency).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    n_samples = len(samples)
    results = np.zeros((n_samples, 4))

    checkpoint_file = checkpoint_dir / 'checkpoint_full.npz'

    start_idx = 0
    if checkpoint_file.exists():
        data = np.load(checkpoint_file)
        results = data['results']
        start_idx = int(data['last_idx']) + 1
        if verbose:
            print(f"  Resumed from checkpoint: index {start_idx}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"RUNNING SIMULATIONS: {n_samples} runs")
        print(f"{'='*60}")
        print(f"  Checkpoints: {checkpoint_dir}")
        print(f"  Starting from: {start_idx}")

    start_time = time.time()
    checkpoint_interval = 1000

    for i in range(start_idx, n_samples):
        params = samples.iloc[i].to_dict()
        results[i] = np.array(simulate_theta_rhythm(params))

        if (i + 1) % checkpoint_interval == 0:
            np.savez(checkpoint_file, results=results, last_idx=i)

            elapsed = time.time() - start_time
            rate = (i + 1 - start_idx) / elapsed
            eta = (n_samples - i - 1) / rate

            # is_theta is column 0; only count it (not the amplitude/error/freq).
            n_theta = int(results[:i + 1, 0].sum())

            if verbose:
                print(f"  [{i+1}/{n_samples}] "
                      f"Theta: {n_theta} ({100*n_theta/(i+1):.1f}%) | "
                      f"Rate: {rate:.2f} sim/s | "
                      f"ETA: {eta/3600:.1f} h")

    np.savez(checkpoint_file, results=results, last_idx=n_samples - 1)

    elapsed = time.time() - start_time
    if verbose:
        n_theta = int(results[:, 0].sum())
        print(f"\n  COMPLETED in {elapsed/3600:.2f} hours")
        print(f"  Theta found: {n_theta} of {n_samples} "
              f"({100*n_theta/n_samples:.1f}%)")

    return results


# =============================================================================
# 3. Random Forest analysis (classification + regression)
# =============================================================================

def analyze_grid_search(samples, results, config):
    """
    Full analysis of grid-search results.

    Trains:
        - Random Forest Classifier on is_theta (column 0).
        - Random Forest Regressors on amplitude (col 1), frequency (col 3),
          and phase_error (col 2) restricted to theta-positive samples.

    Computes Permutation Importance (30 repeats) for each model.

    Args:
        samples: DataFrame of parameters.
        results: (n_samples, 4) array returned by run_simulations_parallel.
        config: Configuration dict with 'param_names' and 'output_dir'.

    Returns:
        (rf_clf, perm_theta, indices, reg_results) where reg_results is a
        dict {label: (rf_reg, perm_reg)}.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance

    y_theta = results[:, 0]
    y_ampl = results[:, 1]
    y_error = results[:, 2]
    y_freq = results[:, 3]

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    param_names = config['param_names']

    print(f"\n{'='*60}")
    print("GRID-SEARCH ANALYSIS")
    print(f"{'='*60}")

    # --- Classification: is_theta ---
    n_positive = int(y_theta.sum())
    n_negative = len(y_theta) - n_positive
    print(f"\n  [Classification] is_theta")
    print(f"  Class 0: {n_negative} ({100*n_negative/len(y_theta):.1f}%)")
    print(f"  Class 1: {n_positive} ({100*n_positive/len(y_theta):.1f}%)")

    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=10,
        max_features='sqrt', n_jobs=-1, random_state=42,
        oob_score=True, class_weight='balanced'
    )
    rf_clf.fit(samples, y_theta)
    print(f"  OOB Score: {rf_clf.oob_score_:.4f}")

    perm_theta = permutation_importance(
        rf_clf, samples, y_theta,
        n_repeats=30, scoring='roc_auc',
        n_jobs=-1, random_state=42
    )
    indices = np.argsort(perm_theta.importances_mean)[::-1]

    print(f"\n  {'Rank':<6} {'Param':<10} {'Importance':>12} {'± Std':>12}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*12}")
    for rank, idx in enumerate(indices, 1):
        name = param_names[idx]
        imp = perm_theta.importances_mean[idx]
        std = perm_theta.importances_std[idx]
        print(f"  {rank:<6} {name:<10} {imp:>12.4f} {std:>12.4f}")

    # --- Regression on theta-positive samples only ---
    theta_mask = y_theta == 1
    theta_samples = samples[theta_mask]

    reg_results = {}
    for label, y_reg in [('Amplitude', y_ampl[theta_mask]),
                         ('Frequency', y_freq[theta_mask]),
                         ('Phase Error', y_error[theta_mask])]:
        n_valid = len(y_reg)
        print(f"\n  [Regression] {label} ({n_valid} theta samples)")

        if n_valid < 100:
            print(f"    Too few samples, skipping")
            reg_results[label] = (None, None)
            continue

        rf_reg = RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=10,
            max_features='sqrt', n_jobs=-1, random_state=42,
            oob_score=True
        )
        rf_reg.fit(theta_samples, y_reg)
        print(f"    OOB R^2: {rf_reg.oob_score_:.4f}")

        perm_reg = permutation_importance(
            rf_reg, theta_samples, y_reg,
            n_repeats=30, scoring='r2',
            n_jobs=-1, random_state=42
        )

        indices_reg = np.argsort(perm_reg.importances_mean)[::-1]
        print(f"    {'Rank':<6} {'Param':<10} {'Importance':>12} {'± Std':>12}")
        print(f"    {'-'*6} {'-'*10} {'-'*12} {'-'*12}")
        for rank, idx in enumerate(indices_reg, 1):
            name = param_names[idx]
            imp = perm_reg.importances_mean[idx]
            std = perm_reg.importances_std[idx]
            print(f"    {rank:<6} {name:<10} {imp:>12.4f} {std:>12.4f}")

        reg_results[label] = (rf_reg, perm_reg)

    find_optimal_regions(samples, y_theta, param_names, output_dir)

    return rf_clf, perm_theta, indices, reg_results


# =============================================================================
# 4. Plotting
# =============================================================================

LABEL_MAP = {
    'g12':    r'$g_{12}$, nS',
    'g21':    r'$g_{21}$, nS',
    'Iext1':  r'$I_{ext,1}$, pA',
    'Iext2':  r'$I_{ext,2}$, pA',
    'Delta':  r'$\Delta$, pA',
    'tau_d':  r'$\tau_d$, ms',
    'tau_r':  r'$\tau_r$, ms',
    'tau_f':  r'$\tau_f$, ms',
    'Uinc':   r'$U_{inc}$',
}


def make_label(name):
    """Render a parameter name as a LaTeX label."""
    return LABEL_MAP.get(name, name)


def plot_grid_search_results(samples, y, rf, perm, indices, config,
                             reg_results=None):
    """
    Visualize grid-search results:
        1. Permutation Importance for is_theta (top-left) and for the
           three regression targets (other three panels).
        2. Marginal distributions for the top-6 parameters (is_theta).
        3. 2D heatmap of P(theta) for the top-2 parameters.
    """
    output_dir = Path(config['output_dir'])
    param_names = config['param_names']
    y_arr = y.values if isinstance(y, pd.Series) else y

    # --- 1. Importance 2x2 panel ---
    if reg_results:
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

        sorted_names = [make_label(param_names[i]) for i in indices]
        sorted_imp = perm.importances_mean[indices]
        sorted_std = perm.importances_std[indices]
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(sorted_names)))

        ax = axes[0, 0]
        ax.barh(range(len(sorted_names)), sorted_imp,
                xerr=sorted_std, align='center', color=colors, capsize=3)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Permutation Importance (ROC-AUC)', fontsize=24)
        ax.set_title('Is theta', fontsize=28)
        ax.invert_yaxis()
        ax.tick_params(axis='both', labelsize=24)

        # Place the three regression panels: (1,0), (0,1), (1,1).
        positions = [(1, 0), (0, 1), (1, 1)]
        for (label, (_, perm_reg)), (ax_idx1, ax_idx2) in zip(
                reg_results.items(), positions):
            ax = axes[ax_idx1, ax_idx2]
            if perm_reg is None:
                ax.set_title(f'{label}\n(insufficient data)', fontsize=13)
                ax.axis('off')
                continue

            idx_sorted = np.argsort(perm_reg.importances_mean)[::-1]
            names = [make_label(param_names[i]) for i in idx_sorted]
            imps = perm_reg.importances_mean[idx_sorted]
            stds = perm_reg.importances_std[idx_sorted]
            c = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(names)))

            ax.barh(range(len(names)), imps,
                    xerr=stds, align='center', color=c, capsize=3)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel('Permutation Importance (R^2)', fontsize=24)
            ax.set_title(label, fontsize=28)
            ax.invert_yaxis()
            ax.tick_params(axis='both', labelsize=24)

        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=2.0)
        plt.savefig(output_dir / 'grid_importance_regression.png', dpi=150)
        plt.close()

    # --- 2. Marginal distributions for top-6 parameters ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, param_idx in enumerate(indices[0:6]):
        ax = axes[ax_idx]
        param_name = param_names[param_idx]
        values = samples[param_name].values

        ax.hist(values[y_arr == 0], bins=50, alpha=0.5, label='No theta',
                color='blue', density=True)
        ax.hist(values[y_arr == 1], bins=50, alpha=0.5, label='Is theta',
                color='red', density=True)

        ax.set_xlabel(make_label(param_name), fontsize=22)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'grid_marginals.png', dpi=150)
    plt.close()

    # --- 3. 2D heatmap for top-2 parameters ---
    fig, ax = plt.subplots(figsize=(10, 8))

    p1_idx, p2_idx = indices[0], indices[1]
    p1_name, p2_name = param_names[p1_idx], param_names[p2_idx]

    n_bins = 30
    p1_bins = np.linspace(samples[p1_name].min(), samples[p1_name].max(), n_bins + 1)
    p2_bins = np.linspace(samples[p2_name].min(), samples[p2_name].max(), n_bins + 1)

    heatmap = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))

    p1_vals = samples[p1_name].values
    p2_vals = samples[p2_name].values

    for i in range(len(y_arr)):
        i1 = min(int((p1_vals[i] - p1_bins[0]) / (p1_bins[-1] - p1_bins[0]) * n_bins), n_bins - 1)
        i2 = min(int((p2_vals[i] - p2_bins[0]) / (p2_bins[-1] - p2_bins[0]) * n_bins), n_bins - 1)
        i1 = max(0, i1)
        i2 = max(0, i2)

        counts[i2, i1] += 1
        heatmap[i2, i1] += y_arr[i]

    with np.errstate(divide='ignore', invalid='ignore'):
        prob = np.where(counts > 5, heatmap / counts, np.nan)

    im = ax.imshow(prob, origin='lower', aspect='auto', cmap='RdYlGn',
                   extent=[p1_bins[0], p1_bins[-1], p2_bins[0], p2_bins[-1]],
                   vmin=0, vmax=0.2)

    ax.set_xlabel(p1_name, fontsize=12)
    ax.set_ylabel(p2_name, fontsize=12)
    ax.set_title(f'P(theta): {p1_name} vs {p2_name}', fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(theta)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'grid_2d_heatmap.png', dpi=150)
    plt.close()

    print(f"\n  Plots saved to: {output_dir}")


# =============================================================================
# 5. Heatmaps around a central parameter set
# =============================================================================

def nice_ticks(vmin, vmax, num_ticks=6):
    """
    Generate "nice" round tick values for an axis.

    Args:
        vmin: Axis minimum.
        vmax: Axis maximum.
        num_ticks: Desired number of ticks (5-10).

    Returns:
        numpy array of tick values.
    """
    raw_step = (vmax - vmin) / (num_ticks - 1)

    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude

    if residual < 1.5:
        nice_step = magnitude
    elif residual < 3:
        nice_step = 2 * magnitude
    elif residual < 7:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    tick_start = np.ceil(vmin / nice_step) * nice_step
    tick_end = np.floor(vmax / nice_step) * nice_step

    ticks = np.arange(tick_start, tick_end + nice_step / 2, nice_step)
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]

    if len(ticks) < 4:
        ticks = np.linspace(vmin, vmax, num_ticks)

    return ticks


def heat_plot_from_center(param1: str, param2: str, config,
                          Nsteps: int = 100,
                          center_params: dict = None,
                          return_data: bool = False):
    """
    Build a 3-panel heatmap (frequency, amplitude, phase error) by
    varying two parameters around a central parameter set.

    Args:
        param1: Parameter name for the X axis.
        param2: Parameter name for the Y axis.
        config: Configuration dict with 'param_bounds' and 'output_dir'.
        Nsteps: Number of steps along each axis.
        center_params: Central parameter values (defaults are used if None).
        return_data: If True, also return (ampls, freqs, errors).

    Returns:
        None, or (ampls, freqs, errors) if return_data=True.
    """
    from progress.bar import IncrementalBar

    if center_params is None:
        center_params = {
            'g12':   5285,
            'g21':   5321,
            'Iext1': 389,
            'Iext2': 392,
            'Delta': 56,
            'tau_d': 4.6,
            'tau_r': 1280,
            'tau_f': 183,
            'Uinc':  0.63,
        }

    params_bounds = config['param_bounds']

    param_space1 = np.linspace(*params_bounds[param1], Nsteps)
    param_space2 = np.linspace(*params_bounds[param2], Nsteps)

    ampls = np.zeros((Nsteps, Nsteps))
    freqs = np.zeros((Nsteps, Nsteps))
    errors = np.zeros((Nsteps, Nsteps))

    bar = IncrementalBar(f'Heatmap {param1} vs {param2}', max=Nsteps ** 2)

    for i in range(Nsteps):
        for j in range(Nsteps):
            current_params = center_params.copy()
            current_params[param1] = param_space1[i]
            current_params[param2] = param_space2[j]

            _, ampl, error, freq = simulate_theta_rhythm(current_params)

            if error < 500:
                ampls[i, j] = ampl
                errors[i, j] = error
                freqs[i, j] = freq

            bar.next()

    bar.finish()

    output_dir = Path(config.get('output_dir', '.'))
    output_dir.mkdir(exist_ok=True)
    heatmap_file = output_dir / f'heatmap_{param1}_{param2}.npz'
    np.savez(heatmap_file, ampls=ampls, freqs=freqs, errors=errors)

    # Replace zeros with NaN for visualization.
    errors_plot = np.where(errors == 0, np.nan, errors)
    freqs_plot = np.where(freqs == 0, np.nan, freqs)
    ampls_plot = np.where(ampls == 0, np.nan, ampls)

    xticks = nice_ticks(params_bounds[param1][0], params_bounds[param1][1], num_ticks=6)
    yticks = nice_ticks(params_bounds[param2][0], params_bounds[param2][1], num_ticks=6)

    xtick_indices = (xticks - params_bounds[param1][0]) / \
        (params_bounds[param1][1] - params_bounds[param1][0]) * (Nsteps - 1)
    ytick_indices = (yticks - params_bounds[param2][0]) / \
        (params_bounds[param2][1] - params_bounds[param2][0]) * (Nsteps - 1)

    def format_tick(val, is_integer_param=True):
        if is_integer_param:
            return f'{int(val)}'
        if val < 1:
            return f'{val:.2f}'
        elif val < 10:
            return f'{val:.1f}'
        return f'{int(val)}'

    is_int_param1 = params_bounds[param1][1] > 10
    is_int_param2 = params_bounds[param2][1] > 10

    xtick_labels = [format_tick(v, is_int_param1) for v in xticks]
    ytick_labels = [format_tick(v, is_int_param2) for v in yticks]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cmap_obj = plt.get_cmap('coolwarm').copy()
    cmap_obj.set_bad(color='black')

    # Frequency
    ax1 = axes[0]
    im1 = ax1.imshow(
        freqs_plot.T,
        origin='lower', aspect='auto',
        extent=[0, Nsteps - 1, 0, Nsteps - 1],
        cmap=cmap_obj, vmin=4, vmax=12
    )
    ax1.set_xticks(xtick_indices)
    ax1.set_xticklabels(xtick_labels, fontsize=16)
    ax1.set_yticks(ytick_indices)
    ax1.set_yticklabels(ytick_labels, fontsize=16)
    ax1.set_xlabel(make_label(param1), fontsize=24)
    ax1.set_ylabel(make_label(param2), fontsize=24)
    ax1.set_title('Theta Frequency', fontsize=24)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Frequency (Hz)', fontsize=18)

    # Amplitude
    ax2 = axes[1]
    im2 = ax2.imshow(
        ampls_plot.T,
        origin='lower', aspect='auto',
        extent=[0, Nsteps - 1, 0, Nsteps - 1],
        cmap=cmap_obj,
    )
    ax2.set_xticks(xtick_indices)
    ax2.set_xticklabels(xtick_labels, fontsize=16)
    ax2.set_yticks(ytick_indices)
    ax2.set_yticklabels(ytick_labels, fontsize=16)
    ax2.set_xlabel(make_label(param1), fontsize=24)
    ax2.set_ylabel(make_label(param2), fontsize=24)
    ax2.set_title('Theta Amplitude', fontsize=24)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Amplitude (Hz)', fontsize=18)

    # Phase error
    ax3 = axes[2]
    im3 = ax3.imshow(
        errors_plot.T,
        origin='lower', aspect='auto',
        extent=[0, Nsteps - 1, 0, Nsteps - 1],
        cmap=cmap_obj, vmin=0, vmax=60
    )
    ax3.set_xticks(xtick_indices)
    ax3.set_xticklabels(xtick_labels, fontsize=16)
    ax3.set_yticks(ytick_indices)
    ax3.set_yticklabels(ytick_labels, fontsize=16)
    ax3.set_xlabel(make_label(param1), fontsize=24)
    ax3.set_ylabel(make_label(param2), fontsize=24)
    ax3.set_title('Phase Error', fontsize=24)
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Phase Error (deg)', fontsize=18)

    fig.suptitle(f'Heatmap: {make_label(param1)} vs {make_label(param2)}',
                 fontsize=28, y=1.02)

    plt.tight_layout()

    output_file = output_dir / f'heatmap_{param1}_{param2}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

    if return_data:
        return ampls, freqs, errors


def heat_plot_single(param1: str, param2: str, config,
                     data: np.ndarray,
                     title: str,
                     cmap: str = 'viridis',
                     vmin: float = None,
                     vmax: float = None,
                     cbar_label: str = '',
                     Nsteps: int = 10):
    """
    Generic single-panel heatmap of any 2D array.

    Args:
        param1: X-axis parameter name.
        param2: Y-axis parameter name.
        config: Configuration dict with 'param_bounds'.
        data: 2D array of shape (Nsteps, Nsteps).
        title: Plot title.
        cmap: Matplotlib colormap name.
        vmin, vmax: Color-scale bounds.
        cbar_label: Colorbar label.
        Nsteps: Number of steps along each axis.
    """
    params_bounds = config['param_bounds']

    xticks = nice_ticks(params_bounds[param1][0], params_bounds[param1][1], num_ticks=6)
    yticks = nice_ticks(params_bounds[param2][0], params_bounds[param2][1], num_ticks=6)

    xtick_indices = (xticks - params_bounds[param1][0]) / \
        (params_bounds[param1][1] - params_bounds[param1][0]) * (Nsteps - 1)
    ytick_indices = (yticks - params_bounds[param2][0]) / \
        (params_bounds[param2][1] - params_bounds[param2][0]) * (Nsteps - 1)

    is_int1 = params_bounds[param1][1] > 10
    is_int2 = params_bounds[param2][1] > 10

    xtick_labels = [f'{int(v)}' if is_int1 else f'{v:.2f}' for v in xticks]
    ytick_labels = [f'{int(v)}' if is_int2 else f'{v:.2f}' for v in yticks]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        data.T,
        origin='lower', aspect='auto',
        extent=[0, Nsteps - 1, 0, Nsteps - 1],
        cmap=cmap, vmin=vmin, vmax=vmax
    )

    ax.set_xticks(xtick_indices)
    ax.set_xticklabels(xtick_labels, fontsize=11)
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels(ytick_labels, fontsize=11)
    ax.set_xlabel(param1, fontsize=13)
    ax.set_ylabel(param2, fontsize=13)
    ax.set_title(title, fontsize=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=12)

    plt.tight_layout()

    output_dir = Path(config.get('output_dir', '.'))
    output_file = output_dir / f'{title.replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


# =============================================================================
# 6. Optimal-region summary
# =============================================================================

def find_optimal_regions(samples, y, param_names, output_dir):
    """
    Find parameter regions with the highest probability of producing
    theta rhythm, by computing the 10/50/90 percentiles of each
    parameter among theta-positive samples.
    """
    y_arr = y.values if isinstance(y, pd.Series) else y

    theta_samples = samples[y_arr == 1]

    if len(theta_samples) < 10:
        print("\n  Too few theta samples for region analysis")
        return

    report_lines = ["\n" + "=" * 60,
                    "OPTIMAL PARAMETER REGIONS",
                    "=" * 60]
    report_lines.append(f"\n  Total theta-positive samples: {len(theta_samples)}")
    report_lines.append("\n  Optimal ranges (10th-50th-90th percentiles):\n")

    for param in param_names:
        vals = theta_samples[param].values
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        report_lines.append(f"    {param:<10}: [{p10:.2f} -- {p50:.2f} -- {p90:.2f}]")

    with open(output_dir / 'optimal_regions.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print('\n'.join(report_lines))


# =============================================================================
# 7. Top-level entry point
# =============================================================================

def run_grid_search(config):
    """Run the full grid-search pipeline."""
    print("\n" + "=" * 60)
    print("GRID-SEARCH: 100,000 SIMULATIONS")
    print("=" * 60)

    print("\n[1/3] Generating samples (Sobol sequence)...")
    samples = optimal_grid_sampling(
        config['n_samples'], config['param_bounds']
    )
    print(f"  Generated: {len(samples)} points")

    print("\n[2/3] Running simulations...")
    # If a checkpoint already exists, load it; otherwise run fresh.
    y = np.load('good_borders_checkpoints_grid/checkpoint_full.npz')['results']

    print("\n[3/3] Analyzing results...")
    rf_clf, perm_theta, indices, reg_results = analyze_grid_search(
        samples, y, config
    )

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    samples['theta_rhythm'] = y[:, 0]
    samples['amplitude'] = y[:, 1]
    samples['frequency'] = y[:, 3]
    samples['phase_error'] = y[:, 2]
    samples.to_csv(output_dir / 'all_simulations.csv', index=False)

    plot_grid_search_results(samples, y[:, 0], rf_clf, perm_theta, indices,
                             config, reg_results=reg_results)

    print("\n" + "=" * 60)
    print("GRID-SEARCH COMPLETE!")
    print(f"Results: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    config = {
        'param_names': ['g12', 'g21', 'Iext1', 'Iext2',
                        'Delta', 'tau_d', 'tau_r', 'tau_f', 'Uinc'],
        'param_bounds': {
            'g12':   (0, 10000),
            'g21':   (0, 10000),
            'Iext1': (0, 600),
            'Iext2': (0, 600),
            'Delta': (1, 100),
            'tau_d': (1, 21),
            'tau_r': (50, 2000),
            'tau_f': (3, 300),
            'Uinc':  (0, 1),
        },
        'n_samples': 100000,
        'output_dir': 'good_borders_grid_search_results',
    }

    run_grid_search(config)
