"""
Heatmap Visualization for Theta Rhythm Model
=============================================

Standalone module for generating 3-panel heatmaps (frequency,
amplitude, phase error) by sweeping two parameters around a
central parameter set.

Note: A more feature-complete version of heat_plot_from_center is
also provided in GridSearchFull.py. This file is kept for users
who want only the heatmap utility without importing the full
grid-search pipeline.

Author: S.N. Skorokhod, S.V. Dubrovin, I.E. Mysin (refactored 2026)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from progress.bar import IncrementalBar

from InhOsc import (
    ModelParameters, ThetaRhythmNetwork, OscillationAnalyzer,
)


# =============================================================================
# Simulation wrapper (3-value return: amplitude, error, frequency)
# =============================================================================

def simulate_theta_rhythm(params):
    """
    Run a single mean-field simulation and return theta-rhythm metrics.

    Args:
        params: dict with parameter values.

    Returns:
        Tuple (amplitude, error_code, frequency).
    """
    dt = 0.05
    model_params = ModelParameters.from_dict(params)
    network = ThetaRhythmNetwork(model_params)
    mf_results = network.simulate_meanfield(t_max=1600, dt=dt)
    transient = int(600 / dt)
    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=dt, transient_samples=transient
    )
    return mf_analysis.amplitude, mf_analysis.error_code, mf_analysis.frequency


# =============================================================================
# Utilities
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


# =============================================================================
# 3-panel heatmap
# =============================================================================

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

            ampl, error, freq = simulate_theta_rhythm(current_params)

            if error < 500:
                ampls[i, j] = ampl
                errors[i, j] = error
                freqs[i, j] = freq

            bar.next()

    bar.finish()

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

    # --- Panel 1: Frequency ---
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

    # --- Panel 2: Amplitude ---
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

    # --- Panel 3: Phase Error ---
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

    output_dir = Path(config.get('output_dir', '.'))
    output_dir.mkdir(exist_ok=True)

    heatmap_file = output_dir / f'heatmap_{param1}_{param2}.npz'
    np.savez(heatmap_file, ampls=ampls, freqs=freqs, errors=errors)

    output_file = output_dir / f'heatmap_{param1}_{param2}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

    if return_data:
        return ampls, freqs, errors


# =============================================================================
# Entry point
# =============================================================================

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

    heat_plot_from_center('g12', 'g21', config)
