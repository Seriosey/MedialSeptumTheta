"""
Analysis Scripts for Theta Rhythm Model
=======================================

This module contains analysis scripts for:
    1. Spiking network validation (N=2000 neurons)
    2. Mean-field vs spiking comparison plots

Author: S.N. Skorokhod, S.V. Dubrovin, I.E. Mysin (refactored 2026)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from scipy.ndimage import gaussian_filter1d

from InhOsc import (
    ModelParameters, ThetaRhythmNetwork, OscillationAnalyzer,
    get_default_optimal_parameters,
)


# =============================================================================
# 1. Spiking Network Validation
# =============================================================================

def validate_spiking_network(N: int = 2000, t_max: float = 2600.0,
                             dt: float = 0.02, seed: int = 42) -> Dict:
    """
    Compare the spiking network simulation with the mean-field model.

    Args:
        N: Number of neurons per population.
        t_max: Simulation time (ms).
        dt: Time step (ms).
        seed: Random seed.

    Returns:
        Dictionary with mean-field and spiking results plus analysis.
    """
    print(f"\n{'='*60}")
    print(f"SPIKING NETWORK VALIDATION (N={N} neurons per population)")
    print(f"{'='*60}")

    params = get_default_optimal_parameters()
    network = ThetaRhythmNetwork(params)

    print("Running mean-field simulation...")
    mf_results = network.simulate_meanfield(t_max=t_max, dt=dt)

    print(f"Running spiking network simulation (N={N})...")
    spk_results = network.simulate_spiking(t_max=t_max, dt=dt, N=N, seed=seed)

    transient = int(300 / dt)

    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=dt, transient_samples=transient
    )
    spk_analysis = OscillationAnalyzer.analyze_oscillation(
        spk_results['r1'], spk_results['r2'], dt=dt, transient_samples=transient
    )

    print(f"\nResults Comparison:")
    print(f"{'Metric':<25} {'Mean-Field':<15} {'Spiking':<15}")
    print(f"{'-'*55}")
    print(f"{'Is theta':<25} {str(mf_analysis.is_theta):<15} {str(spk_analysis.is_theta):<15}")
    print(f"{'Frequency (Hz)':<25} {mf_analysis.frequency:<15.2f} {spk_analysis.frequency:<15.2f}")
    print(f"{'Amplitude (Hz)':<25} {mf_analysis.amplitude:<15.2f} {spk_analysis.amplitude:<15.2f}")
    print(f"{'Phase diff (deg)':<25} {mf_analysis.phase_diff:<15.1f} {spk_analysis.phase_diff:<15.1f}")
    print(f"{'Phase error (deg)':<25} {mf_analysis.phase_error:<15.1f} {spk_analysis.phase_error:<15.1f}")

    return {
        'meanfield': mf_results,
        'spiking': spk_results,
        'mf_analysis': mf_analysis,
        'spk_analysis': spk_analysis,
        'params': params,
    }


def plot_spiking_comparison(results: Dict, dt: float = 0.01,
                            start_time: float = 0.0,
                            save_path: Optional[str] = None):
    """
    Create a 2x2 comparison figure between mean-field and spiking models:
        - Top row: firing rates of both populations.
        - Bottom row: available neurotransmitter fraction (mean-field)
          and membrane potential of representative neurons (spiking).

    Args:
        results: Output of validate_spiking_network.
        dt: Time step used during simulation (ms).
        start_time: Time to start plotting from (ms).
        save_path: Directory to save the figure (optional).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t = results['meanfield']['t']
    start_idx = int(start_time / dt)

    smooth_rate1 = gaussian_filter1d(results['meanfield']['r1'][start_idx:] * 1000, sigma=1)
    smooth_rate2 = gaussian_filter1d(results['meanfield']['r2'][start_idx:] * 1000, sigma=1)

    # Mean-field firing rates (top-left)
    axes[0, 0].set_title('Mean-Field Model', fontsize=12, fontweight='bold')
    axes[0, 0].plot(t[start_idx:], smooth_rate1, 'b-', label='Population 1', alpha=0.8)
    axes[0, 0].plot(t[start_idx:], smooth_rate2, 'r-', label='Population 2', alpha=0.8)
    axes[0, 0].set_ylabel('Firing Rate (Hz)')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_xlim([t[start_idx], t[-1]])

    # Mean-field available neurotransmitter (bottom-left)
    axes[1, 0].plot(t[start_idx:], results['meanfield']['x1'][start_idx:] * 100, 'b-')
    axes[1, 0].plot(t[start_idx:], results['meanfield']['x2'][start_idx:] * 100, 'r-')
    axes[1, 0].set_ylabel('Available NT (%)')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_xlim([t[start_idx], t[-1]])

    # Spiking firing rates (top-right)
    r1_smooth = gaussian_filter1d(results['spiking']['r1'][start_idx:] * 1000, sigma=1)
    r2_smooth = gaussian_filter1d(results['spiking']['r2'][start_idx:] * 1000, sigma=1)

    axes[0, 1].set_title('Spiking Model', fontsize=12, fontweight='bold')
    axes[0, 1].plot(t[start_idx:], r1_smooth, 'b-', label='Population 1', alpha=0.8)
    axes[0, 1].plot(t[start_idx:], r2_smooth, 'r-', label='Population 2', alpha=0.8)
    axes[0, 1].set_ylabel('Firing Rate (Hz)\n(smoothed)')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].set_xlim([t[start_idx], t[-1]])

    # Membrane potentials of two representative neurons (bottom-right)
    v1_all = results['spiking']['v1']
    v2_all = results['spiking']['v2']

    if v1_all is None or v2_all is None:
        print("Error: Membrane potential data not found in results.")
        return

    idx1 = int(v1_all.shape[0] / 2) + 700
    idx2 = int(v2_all.shape[0] / 2) + 700

    axes[1, 1].plot(t[start_idx:], v1_all[idx1, start_idx:], 'b-',
                    label='Neuron (Pop 1)', alpha=0.8)
    axes[1, 1].plot(t[start_idx:], v2_all[idx2, start_idx:], 'r-',
                    label='Neuron (Pop 2)', alpha=0.8)
    axes[1, 1].set_ylabel('Membrane Potential (mV)')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].set_xlim([t[start_idx], t[-1]])

    plt.tight_layout()

    if save_path:
        out_file = f"{save_path}/results.png"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {out_file}")

    plt.show()


# =============================================================================
# 2. Entry point
# =============================================================================

if __name__ == "__main__":
    # Median parameters from grid-search marginals (theta-producing configuration)
    params_array = np.array([2800, 2700, 360, 340, 23, 3.83, 1250, 177, 0.63])
    model_params = ModelParameters.from_array(params_array)

    dt = 0.01
    network = ThetaRhythmNetwork(model_params)

    mf_results = network.simulate_meanfield(t_max=2600, dt=dt)
    spk_results = network.simulate_spiking(t_max=2600, dt=dt, N=2000, seed=23)

    results = {
        'meanfield': mf_results,
        'spiking': spk_results,
    }

    plot_spiking_comparison(results, dt, start_time=2100)
