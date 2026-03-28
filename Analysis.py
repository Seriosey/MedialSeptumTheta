"""
Analysis Scripts for Theta Rhythm Model
=======================================

This module contains analysis scripts for:
    1. Spiking network validation (N=2000 neurons)
    2. Parameter space exploration
    3. Grid-search validation of fANOVA results
    4. STD necessity demonstration

Author: Refactored 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json
import os

# Import from model module
from InhOsc import (
    ModelParameters, ThetaRhythmNetwork, OscillationAnalyzer,
    get_default_optimal_parameters, NeuronParameters
)


# =============================================================================
# 1. Spiking Network Validation
# =============================================================================

def validate_spiking_network(N: int = 2000, t_max: float = 2600.0, 
                             dt: float = 0.02, seed: int = 42) -> Dict:
    """
    Compare spiking network simulation with mean-field model.
    
    This validates that the mean-field model correctly captures the
    collective dynamics of a spiking neuron network.
    
    Args:
        N: Number of neurons per population
        t_max: Simulation time (ms)
        dt: Time step (ms)
        seed: Random seed
        
    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*60}")
    print(f"SPIKING NETWORK VALIDATION (N={N} neurons per population)")
    print(f"{'='*60}")
    
    params = get_default_optimal_parameters()
    network = ThetaRhythmNetwork(params)
    
    # Run mean-field simulation
    print("Running mean-field simulation...")
    mf_results = network.simulate_meanfield(t_max=t_max, dt=dt)
    
    # Run spiking simulation
    print(f"Running spiking network simulation (N={N})...")
    spk_results = network.simulate_spiking(t_max=t_max, dt=dt, N=N, seed=seed)
    
    # Analyze both
    transient = int(300 / dt)
    
    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=dt, transient_samples=transient
    )
    
    spk_analysis = OscillationAnalyzer.analyze_oscillation(
        spk_results['r1'], spk_results['r2'], dt=dt, transient_samples=transient
    )
    
    # Print comparison
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
        'params': params
    }


def plot_spiking_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Create comparison plots between mean-field and spiking models.
    
    Args:
        results: Results from validate_spiking_network
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    t = results['meanfield']['t']
    start_idx = int(300 / 0.02)  # Skip transient
    
    # Mean-field plots (left column)
    axes[0, 0].set_title('Mean-Field Model', fontsize=12, fontweight='bold')
    axes[0, 0].plot(t[start_idx:], results['meanfield']['r1'][start_idx:] * 1000, 
                    'b-', label='Population 1', alpha=0.8)
    axes[0, 0].plot(t[start_idx:], results['meanfield']['r2'][start_idx:] * 1000, 
                    'r-', label='Population 2', alpha=0.8)
    axes[0, 0].set_ylabel('Firing Rate (Hz)')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_xlim([t[start_idx], t[-1]])
    
    axes[1, 0].plot(t[start_idx:], results['meanfield']['x1'][start_idx:] * 100, 'b-')
    axes[1, 0].plot(t[start_idx:], results['meanfield']['x2'][start_idx:] * 100, 'r-')
    axes[1, 0].set_ylabel('Available NT (%)')
    axes[1, 0].set_xlim([t[start_idx], t[-1]])
    
    # Raster plot for spiking (right column)
    axes[0, 1].set_title(f'Spiking Network (N={results["spiking"]["spikes1"].shape[0]})', 
                         fontsize=12, fontweight='bold')
    
    # Subsample neurons for raster plot
    N_show = min(200, results['spiking']['spikes1'].shape[0])
    spike_times_1 = np.where(results['spiking']['spikes1'][:N_show, start_idx::10])
    spike_times_2 = np.where(results['spiking']['spikes2'][:N_show, start_idx::10])
    
    t_sub = t[start_idx::10]
    axes[0, 1].scatter(t_sub[spike_times_1[1]], spike_times_1[0], s=0.5, c='b', alpha=0.5)
    axes[0, 1].scatter(t_sub[spike_times_2[1]], spike_times_2[0] + N_show, s=0.5, c='r', alpha=0.5)
    axes[0, 1].set_ylabel('Neuron Index')
    axes[0, 1].set_xlim([t[start_idx], t[-1]])
    
    # Smoothed firing rates for spiking
    from scipy.ndimage import gaussian_filter1d
    r1_smooth = gaussian_filter1d(results['spiking']['r1'][start_idx:] * 1000, sigma=100)
    r2_smooth = gaussian_filter1d(results['spiking']['r2'][start_idx:] * 1000, sigma=100)
    
    axes[1, 1].plot(t[start_idx:], r1_smooth, 'b-', label='Population 1', alpha=0.8)
    axes[1, 1].plot(t[start_idx:], r2_smooth, 'r-', label='Population 2', alpha=0.8)
    axes[1, 1].set_ylabel('Firing Rate (Hz)\n(smoothed)')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].set_xlim([t[start_idx], t[-1]])
    
    # Phase comparison
    phase_mf = OscillationAnalyzer.compute_phase(results['meanfield']['r1'][start_idx:])
    phase_spk = OscillationAnalyzer.compute_phase(r1_smooth / 1000)
    
    axes[2, 0].plot(t[start_idx:], np.degrees(phase_mf), 'b-', label='Pop 1')
    axes[2, 0].plot(t[start_idx:], np.degrees(OscillationAnalyzer.compute_phase(
        results['meanfield']['r2'][start_idx:])), 'r-', label='Pop 2')
    axes[2, 0].set_xlabel('Time (ms)')
    axes[2, 0].set_ylabel('Phase (deg)')
    axes[2, 0].legend()
    axes[2, 0].set_xlim([t[start_idx], t[-1]])
    
    axes[2, 1].plot(t[start_idx:], np.degrees(phase_spk), 'b-', label='Pop 1')
    axes[2, 1].plot(t[start_idx:], np.degrees(OscillationAnalyzer.compute_phase(
        r2_smooth / 1000)), 'r-', label='Pop 2')
    axes[2, 1].set_xlabel('Time (ms)')
    axes[2, 1].set_ylabel('Phase (deg)')
    axes[2, 1].legend()
    axes[2, 1].set_xlim([t[start_idx], t[-1]])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()



import seaborn as sns
import matplotlib.colors as mcolors
from progress.bar import IncrementalBar

tmin = 0.0
tmax = 1600 # msec
dt = 0.02
t = np.arange(tmin, tmax, dt)

Nsteps = 50

param1 = np.linspace(0,10000, Nsteps) # Iext
param2 = np.linspace(0,10000, Nsteps) # Delta
# param3 = np.linspace(0,1000, Nsteps)
# param4 = np.linspace(0,1000, Nsteps)
# param5 = np.linspace(0,300, Nsteps)

ampls = np.zeros((Nsteps, Nsteps))
freqs = np.zeros((Nsteps, Nsteps))
errors = np.zeros((Nsteps, Nsteps))


bar = IncrementalBar('Countdown', max = Nsteps**2)


for i in range(Nsteps):
    for j in range(Nsteps):

        params = np.array([param1[i], param2[j], 800, 800, 200, 3.8314141705882356, 635.5365306862744 *0.5, 15.095432365686278, 0.2738522440196079]) # S-E: 77.5955, 130.1333, 156.7755, 145.9170
        model_params = ModelParameters.from_array(params)

        network = ThetaRhythmNetwork(model_params)
        mf_results = network.simulate_meanfield(t_max=1600, dt=0.02) 
        transient = int(300 / 0.02)
        mf_analysis = OscillationAnalyzer.analyze_oscillation(
            mf_results['r1'], mf_results['r2'], dt=0.02, transient_samples=transient
    )
        
        # isoscillating = calculate_frequency(rates)
        error, ampl, freq = mf_analysis.error_code, mf_analysis.amplitude, mf_analysis.frequency

        if error < 500:
            # print('hilbert. ', param1[i], param2[j])
            # plot_rate(rates, t)
            ampls[i, j] = ampl
            errors[i, j] = error
            freqs[i, j] = freq
            
            
        bar.next()
        
bar.finish()


