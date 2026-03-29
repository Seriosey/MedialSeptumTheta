"""
Theta Rhythm Generation Model
=============================

This module implements a computational model of theta rhythm generation
in the medial septum via short-term synaptic depression (STD) between
two mutually inhibitory populations of PV+ neurons.

Reference:
    Skorokhod et al. - "The medial septum model as a pacemaker of theta rhythm"

Author: S.N. Skorokhod, S.V. Dubrovin, I.E. Mysin
Refactored: 2026
"""

import numpy as np
from scipy.stats import cauchy
from scipy.signal import hilbert, find_peaks, windows
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Data Classes for Model Parameters and Results
# =============================================================================

@dataclass
class ModelParameters:
    """
    Container for all model parameters.
    
    Attributes:
        g12: Maximal synaptic conductance from population 1 to 2 (nS)
        g21: Maximal synaptic conductance from population 2 to 1 (nS)
        Iext1: External current to population 1 (pA)
        Iext2: External current to population 2 (pA)
        Delta: Scale parameter for Cauchy distribution of currents
        tau_d: Neurotransmitter decay time constant (ms)
        tau_r: Recovery time constant from depression (ms)
        tau_f: Facilitation time constant (ms)
        Uinc: Incremental release probability per spike
    """
    # Synaptic conductances (nS)
    g12: float = 3000.0
    g21: float = 3000.0
    
    # External currents (pA)
    Iext1: float = 800.0
    Iext2: float = 800.0
    
    # Heterogeneity parameter
    Delta: float = 200.0
    
    # Tsodyks-Markram synaptic parameters
    tau_d: float = 3.83      # Decay time constant (ms)
    tau_r: float = 317.56    # Recovery time constant (ms)
    tau_f: float = 15.09     # Facilitation time constant (ms)
    Uinc: float = 0.27       # Incremental release probability
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to array for optimization."""
        return np.array([
            self.g12, self.g21, self.Iext1, self.Iext2,
            self.Delta, self.tau_d, self.tau_r, self.tau_f, self.Uinc
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ModelParameters':
        """Create parameters from array."""
        return cls(
            g12=arr[0], g21=arr[1], Iext1=arr[2], Iext2=arr[3],
            Delta=arr[4], tau_d=arr[5], tau_r=arr[6], tau_f=arr[7], Uinc=arr[8]
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelParameters':
        """Create parameters from dictionary."""
        return cls(**d)
    
    @property
    def g_ratio(self) -> float:
        """Ratio of synaptic conductances."""
        return self.g12 / self.g21 if self.g21 != 0 else np.inf
    
    @property
    def I_ratio(self) -> float:
        """Ratio of external currents."""
        return self.Iext2 / self.Iext1 if self.Iext1 != 0 else np.inf


@dataclass
class NeuronParameters:
    """
    Container for Izhikevich neuron parameters.
    
    These parameters are set for fast-spiking neurons typical of
    medial septum PV+ interneurons.
    """
    a: float = 1.0           # Scaling coefficient
    b: float = 95.0          # Linear coefficient
    c: float = 2200.0        # Constant term
    tau: float = 40.0        # Membrane time constant (ms)
    V_rest: float = -58.0    # Resting potential (mV)
    V_peak: float = 800.0    # Peak voltage threshold (mV)
    V_reset: float = -800.0  # Reset voltage (mV)
    E_rev: float = -80.0     # Reversal potential for inhibition (mV)


@dataclass
class OscillationResults:
    """
    Container for oscillation analysis results.
    
    Attributes:
        is_theta: Whether both populations show theta oscillations
        frequency: Dominant oscillation frequency (Hz)
        amplitude: Oscillation amplitude (spikes/ms)
        phase_diff: Phase difference between populations (degrees)
        phase_error: Error from target 150 degrees
    """
    is_theta: bool
    frequency: float
    amplitude: float
    phase_diff: float
    phase_error: float
    error_code: int  # 0 = success, 500 = bad amplitudes, 1000 = no theta


# =============================================================================
# Tsodyks-Markram Synapse Model
# =============================================================================

class TsodyksMarkramSynapse:
    """
    Tsodyks-Markram model for short-term synaptic plasticity.
    
    This model tracks three state variables:
        - u: instantaneous release probability
        - x: fraction of resources available for release
        - y: fraction of active resources in synaptic cleft
    
    References:
        Tsodyks, Markram (1997) - Neural Computation
    """
    
    def __init__(self, tau_d: float, tau_r: float, tau_f: float, Uinc: float):
        """
        Initialize synapse with given time constants.
        
        Args:
            tau_d: Neurotransmitter decay time constant (ms)
            tau_r: Recovery time constant from depression (ms)
            tau_f: Facilitation time constant (ms)
            Uinc: Incremental release probability per spike
        """
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.tau_f = tau_f
        self.Uinc = Uinc
        
        # Pre-compute common values for efficiency
        self._tau_ratio = np.where(
            tau_d != tau_r,
            tau_d / (tau_d - tau_r),
            1e-13
        )
    
    def update(self, r: float, y: float, x: float, u: float, 
               dt: float) -> Tuple[float, float, float]:
        """
        Update synapse state for one time step.
        
        Uses analytical solution for continuous dynamics followed by
        discrete spike update.
        
        Args:
            r: Presynaptic firing rate (spikes/ms)
            y: Current fraction of active resources
            x: Current fraction of available resources
            u: Current release probability
            dt: Time step (ms)
            
        Returns:
            Tuple of (new_y, new_x, new_u)
        """
        # Pre-compute exponential decays
        exp_d = np.exp(-dt / self.tau_d)
        exp_r = np.exp(-dt / self.tau_r)
        exp_f = np.exp(-dt / self.tau_f)
        
        # Continuous decay phase (analytical solution)
        y_decay = y * exp_d
        x_decay = 1 + (x - 1 + self._tau_ratio * y) * exp_r - self._tau_ratio * y
        u_decay = u * exp_f
        
        # Spike-triggered update
        released = u_decay * x_decay * r * dt
        
        u_new = u_decay + self.Uinc * (1 - u_decay) * r * dt
        y_new = y_decay + released
        x_new = x_decay - released
        
        return y_new, x_new, u_new
    
    def get_conductance(self, g_max: float, y: float) -> float:
        """
        Calculate effective synaptic conductance.
        
        Args:
            g_max: Maximal synaptic conductance (nS)
            y: Fraction of active resources
            
        Returns:
            Effective conductance (nS)
        """
        return g_max * y


# =============================================================================
# Mean-Field Population Model
# =============================================================================

class MeanFieldPopulation:
    """
    Mean-field model of a population of Izhikevich neurons.
    
    This model describes the collective dynamics of a homogeneous
    population using the formalism of Chen & Campbell (2022).
    """
    
    def __init__(self, params: NeuronParameters, Iext: float, 
                 Delta: float, etabar: float = 0.0):
        """
        Initialize population.
        
        Args:
            params: Neuron parameters
            Iext: External current (pA)
            Delta: Scale parameter for Cauchy distribution
            etabar: Mean of Cauchy distribution
        """
        self.params = params
        self.Iext = Iext
        self.Delta = Delta
        self.etabar = etabar
        
        # State variables
        self.r = 0.0    # Firing rate (spikes/ms)
        self.v = params.V_rest  # Mean membrane potential (mV)
    
    def reset(self):
        """Reset state variables to initial conditions."""
        self.r = 0.0
        self.v = self.params.V_rest
    
    def update(self, g_total: float, dt: float) -> Tuple[float, float]:
        """
        Update population state for one time step.
        
        Args:
            g_total: Total inhibitory conductance (nS)
            dt: Time step (ms)
            
        Returns:
            Tuple of (firing_rate, membrane_potential)
        """
        p = self.params
        
        # Store current values for correct update order
        r_old = self.r
        v_old = self.v
        
        # Firing rate dynamics (using current v)
        dr = (
            (p.b - g_total) * r_old + 
            2 * p.a * r_old * v_old + 
            self.Delta * p.a / (np.pi * p.tau)
        ) / p.tau
        
        # Update firing rate first
        self.r = r_old + dr * dt
        
        # Membrane potential dynamics (using OLD r value, same as original code)
        # Note: In original code: cI - g_total*80, with E_rev=-80 this becomes c + g_total*E_rev
        dv = (
            -(np.pi * r_old * p.tau)**2 / p.a +
            p.a * v_old**2 +
            (p.b - g_total) * v_old +
            p.c +
            g_total * p.E_rev +  # This gives -g_total*80 when E_rev=-80
            self.Iext +
            self.etabar
        ) / p.tau
        
        # Update membrane potential
        self.v = v_old + dv * dt
        
        return self.r, self.v


# =============================================================================
# Spiking Neuron Network Model
# =============================================================================

class SpikingNeuronNetwork:
    """
    Network of spiking Izhikevich neurons for validation of mean-field model.
    
    This implements a population of N neurons with heterogeneous currents
    and all-to-all connectivity via depressing synapses.
    """
    
    def __init__(self, N: int, params: NeuronParameters, 
                 Iext: float, Delta: float, seed: Optional[int] = None):
        """
        Initialize spiking network.
        
        Args:
            N: Number of neurons in population
            params: Neuron parameters
            Iext: Mean external current (pA)
            Delta: Scale parameter for Cauchy distribution
            seed: Random seed for reproducibility
        """
        self.N = N
        self.params = params
        
        if seed is not None:
            np.random.seed(seed)
        
        # Heterogeneous currents (Cauchy distribution)
        self.I_het = cauchy.rvs(loc=0, scale=Delta, size=N)
        self.Iext = Iext
        
        # Initialize membrane potentials
        self.V = np.full(N, params.V_rest)
    
    def reset(self, seed: Optional[int] = None):
        """Reset network to initial conditions."""
        if seed is not None:
            np.random.seed(seed)
        self.V = np.full(self.N, self.params.V_rest)
    
    def update(self, Isyn: float, dt: float) -> Tuple[np.ndarray, float]:
        """
        Update all neurons for one time step.
        
        Args:
            Isyn: Total synaptic current (pA)
            dt: Time step (ms)
            
        Returns:
            Tuple of (spike_array, population_firing_rate)
        """
        p = self.params
        
        # Total input current for each neuron
        I_total = self.Iext + self.I_het + Isyn
        
        # Membrane potential dynamics (Izhikevich model, simplified)
        dV = (p.a * self.V**2 + p.b * self.V + p.c + I_total) / p.tau * dt
        self.V += dV
        
        # Detect spikes
        spiked = self.V >= p.V_peak
        self.V[spiked] = p.V_reset
        
        # Calculate population firing rate
        rate = np.sum(spiked) / (self.N * dt)
        
        return spiked, rate


# =============================================================================
# Complete Network Model
# =============================================================================

class ThetaRhythmNetwork:
    """
    Complete network model of two mutually inhibitory populations
    connected via depressing synapses.
    
    This class can run either mean-field or spiking simulations.
    """
    
    def __init__(self, params: ModelParameters, 
                 neuron_params: Optional[NeuronParameters] = None):
        """
        Initialize network.
        
        Args:
            params: Model parameters
            neuron_params: Neuron parameters (optional, uses defaults)
        """
        self.params = params
        self.neuron_params = neuron_params or NeuronParameters()
        
        # Initialize synapses (from pop1 to pop2, and pop2 to pop1)
        self.synapse_12 = TsodyksMarkramSynapse(
            params.tau_d, params.tau_r, params.tau_f, params.Uinc
        )
        self.synapse_21 = TsodyksMarkramSynapse(
            params.tau_d, params.tau_r, params.tau_f, params.Uinc
        )
    
    def simulate_meanfield(self, t_max: float, dt: float,
                           transient: float = 300.0) -> Dict[str, np.ndarray]:
        """
        Run mean-field simulation.
        
        Args:
            t_max: Total simulation time (ms)
            dt: Time step (ms)
            transient: Time to discard as transient (ms)
            
        Returns:
            Dictionary with simulation results:
                - 't': time array
                - 'r1', 'r2': firing rates of populations
                - 'I12', 'I21': synaptic currents
                - 'x1', 'x2': available neurotransmitter fractions
        """
        # Time array
        t = np.arange(0, t_max, dt)
        n_steps = len(t)
        
        # Initialize populations
        pop1 = MeanFieldPopulation(
            self.neuron_params, self.params.Iext1, self.params.Delta
        )
        pop2 = MeanFieldPopulation(
            self.neuron_params, self.params.Iext2, self.params.Delta
        )
        
        # Initialize synaptic variables
        y12, x12, u12 = 0.0, 1.0, 0.0
        y21, x21, u21 = 0.0, 1.0, 0.0
        
        # Storage arrays
        r1_arr = np.zeros(n_steps)
        r2_arr = np.zeros(n_steps)
        I12_arr = np.zeros(n_steps)
        I21_arr = np.zeros(n_steps)
        x12_arr = np.ones(n_steps)
        x21_arr = np.ones(n_steps)
        
        # Main simulation loop
        for i in range(n_steps - 1):
            # Update synapses
            y12, x12, u12 = self.synapse_12.update(pop2.r, y12, x12, u12, dt)
            y21, x21, u21 = self.synapse_21.update(pop1.r, y21, x21, u21, dt)
            
            # Calculate conductances
            g12 = self.synapse_12.get_conductance(self.params.g12, y12)
            g21 = self.synapse_21.get_conductance(self.params.g21, y21)
            
            # Calculate synaptic currents (for recording)
            I12_arr[i] = g12 * (self.neuron_params.E_rev - pop1.v)
            I21_arr[i] = g21 * (self.neuron_params.E_rev - pop2.v)
            
            # Update populations
            r1_arr[i], v1 = pop1.update(g12, dt)
            r2_arr[i], v2 = pop2.update(g21, dt)
            
            # Store synaptic variables
            x12_arr[i] = x12
            x21_arr[i] = x21
        
        return {
            't': t,
            'r1': r1_arr,
            'r2': r2_arr,
            'I12': I12_arr,
            'I21': I21_arr,
            'x1': x21_arr,  # x1 is the resource available from pop1's perspective
            'x2': x12_arr
        }
    
    def simulate_spiking(self, t_max: float, dt: float, N: int = 2000,
                         seed: Optional[int] = None,
                         transient: float = 300.0) -> Dict[str, np.ndarray]:
        """
        Run spiking neuron network simulation.
        
        Args:
            t_max: Total simulation time (ms)
            dt: Time step (ms)
            N: Number of neurons per population
            seed: Random seed for reproducibility
            transient: Time to discard as transient (ms)
            
        Returns:
            Dictionary with simulation results:
                - 't': time array
                - 'r1', 'r2': firing rates (smoothed)
                - 'r1_raw', 'r2_raw': raw firing rates
                - 'spikes1', 'spikes2': spike trains (N x time array)
        """
        # Time array
        t = np.arange(0, t_max, dt)
        n_steps = len(t)
        
        # Initialize spiking networks
        net1 = SpikingNeuronNetwork(N, self.neuron_params, 
                                    self.params.Iext1, self.params.Delta, seed)
        net2 = SpikingNeuronNetwork(N, self.neuron_params,
                                    self.params.Iext2, self.params.Delta, 
                                    seed + 1 if seed else None)
        
        # Initialize synaptic variables
        y12, x12, u12 = 0.0, 1.0, 0.0
        y21, x21, u21 = 0.0, 1.0, 0.0
        
        # Storage arrays
        r1_arr = np.zeros(n_steps)
        r2_arr = np.zeros(n_steps)
        x12_arr = np.ones(n_steps)
        x21_arr = np.ones(n_steps)
        
        # Spike trains storage (for raster plots)
        spikes1 = np.zeros((N, n_steps), dtype=bool)
        spikes2 = np.zeros((N, n_steps), dtype=bool)
        
        # Main simulation loop
        for i in range(n_steps - 1):
            # Update synapses
            y12, x12, u12 = self.synapse_12.update(r2_arr[i], y12, x12, u12, dt)
            y21, x21, u21 = self.synapse_21.update(r1_arr[i], y21, x21, u21, dt)
            
            # Calculate synaptic currents
            g12 = self.synapse_12.get_conductance(self.params.g12, y12)
            g21 = self.synapse_21.get_conductance(self.params.g21, y21)
            
            I12 = g12 * (self.neuron_params.E_rev - np.mean(net1.V))
            I21 = g21 * (self.neuron_params.E_rev - np.mean(net2.V))
            
            # Update networks
            spike1, rate1 = net1.update(I12, dt)
            spike2, rate2 = net2.update(I21, dt)
            
            # Store results
            r1_arr[i + 1] = rate1
            r2_arr[i + 1] = rate2
            x12_arr[i + 1] = x12
            x21_arr[i + 1] = x21
            spikes1[:, i + 1] = spike1
            spikes2[:, i + 1] = spike2
        
        return {
            't': t,
            'r1': r1_arr,
            'r2': r2_arr,
            'x1': x21_arr,
            'x2': x12_arr,
            'spikes1': spikes1,
            'spikes2': spikes2
        }
    
    def simulate_without_std(self, t_max: float, dt: float,
                             x_fixed: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Run simulation with STD disabled (constant neurotransmitter availability).
        
        This is used to demonstrate that STD is necessary for oscillations.
        
        Args:
            t_max: Total simulation time (ms)
            dt: Time step (ms)
            x_fixed: Fixed value of available neurotransmitter (0 to 1)
            
        Returns:
            Dictionary with simulation results
        """
        # Time array
        t = np.arange(0, t_max, dt)
        n_steps = len(t)
        
        # Initialize populations
        pop1 = MeanFieldPopulation(
            self.neuron_params, self.params.Iext1, self.params.Delta
        )
        pop2 = MeanFieldPopulation(
            self.neuron_params, self.params.Iext2, self.params.Delta
        )
        
        # Storage arrays
        r1_arr = np.zeros(n_steps)
        r2_arr = np.zeros(n_steps)
        
        # Synaptic variables are now constants
        # The effective conductance is g * x_fixed
        g12_eff = self.params.g12 * x_fixed
        g21_eff = self.params.g21 * x_fixed
        
        # Main simulation loop
        for i in range(n_steps - 1):
            # Update populations with fixed synaptic strength
            r1_arr[i], _ = pop1.update(g12_eff, dt)
            r2_arr[i], _ = pop2.update(g21_eff, dt)
        
        return {
            't': t,
            'r1': r1_arr,
            'r2': r2_arr
        }


# =============================================================================
# Analysis Functions
# =============================================================================

class OscillationAnalyzer:
    """
    Analyzer for theta oscillation detection and characterization.
    
    Provides methods for:
        - Frequency and amplitude estimation via FFT
        - Phase analysis via Hilbert transform
        - Theta rhythm detection
    """
    
    @staticmethod
    def compute_fft(signal: np.ndarray, fs: float) -> Tuple[float, float]:
        """
        Compute dominant frequency and amplitude via FFT.
        
        Args:
            signal: Input signal
            fs: Sampling frequency (samples per ms, i.e., 1/dt)
            
        Returns:
            Tuple of (dominant_frequency in Hz, amplitude)
        """
        n = len(signal)
        
        # Center the signal to remove DC component
        signal_centered = signal - np.mean(signal)
        
        window = windows.hann(n)
        signal_windowed = signal_centered * window
        
        yf = fft(signal_windowed)
        xf = fftfreq(n, 1 / fs)
        
        # Find peak in positive frequencies (skip DC by starting from index 1)
        idx = np.argmax(np.abs(yf[1:n // 2])) + 1
        freq = xf[idx] * 1000  # Convert kHz to Hz
        
        # Amplitude (normalized)
        amp = 2 * np.abs(yf[idx]) / n
        
        return freq, amp
    
    @staticmethod
    def compute_phase(signal: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous phase via Hilbert transform.
        
        Args:
            signal: Input signal
            
        Returns:
            Unwrapped phase array (radians)
        """
        signal_centered = signal - np.mean(signal)
        analytic = hilbert(signal_centered)
        phase = np.unwrap(np.angle(analytic))
        return phase
    
    @staticmethod
    def compute_phase_difference(phase1: np.ndarray, 
                                  phase2: np.ndarray) -> float:
        """
        Compute circular mean phase difference.
        
        Args:
            phase1: Phase of first signal
            phase2: Phase of second signal
            
        Returns:
            Mean phase difference (degrees)
        """
        e_ip1 = np.exp(1j * phase1)
        e_ip2 = np.exp(1j * phase2)
        mean_diff = np.angle(np.mean(e_ip2 * np.conj(e_ip1)))
        return np.degrees(mean_diff)
    
    @staticmethod
    def check_theta(signal: np.ndarray, dt: float,
                    freq_range: Tuple[float, float] = (4.0, 12.0),
                    rate_range: Tuple[float, float] = (2.0, 800.0)) -> Tuple[bool, float, float]:
        """
        Check if signal represents theta oscillation.
        
        Criteria:
            - Dominant frequency in theta range (4-12 Hz)
            - Maximum rate in valid range
            - Stable amplitude (no strong damping)
            
        Args:
            signal: Firing rate signal (spikes/ms)
            dt: Time step (ms)
            freq_range: Valid frequency range (Hz)
            rate_range: Valid maximum rate range (Hz)
            
        Returns:
            Tuple of (is_theta, amplitude, frequency)
        """
        fs = 1 / dt  # Sampling frequency (kHz)
        
        # Get dominant frequency
        freq, amp = OscillationAnalyzer.compute_fft(signal, fs)
        
        # Check maximum rate
        max_rate = np.max(signal) * 1000  # Convert to Hz
        
        # Basic checks
        if not (freq_range[0] < freq < freq_range[1]):
            return False, 0.0, 0.0
        
        if not (rate_range[0] < max_rate < rate_range[1]):
            return False, 0.0, 0.0
        
        # Check for stable oscillation (no strong damping)
        # Find peaks with minimum distance of 90% of period
        period_samples = int(0.9 * 1000 / freq / dt)
        peaks, _ = find_peaks(signal, distance=period_samples, height=max_rate * 0.8 / 1000)
        
        if len(peaks) < 4:
            return False, 0.0, 0.0
        
        # Check peak stability
        peak_heights = signal[peaks]
        peak_range = np.max(peak_heights) - np.min(peak_heights)
        amplitude = np.max(peak_heights) - np.min(signal)
        
        if amplitude > 0 and peak_range / amplitude > 0.1:
            return False, 0.0, 0.0
        
        return True, amplitude * 1000, freq  # Convert amplitude to Hz
    
    @staticmethod
    def analyze_oscillation(r1: np.ndarray, r2: np.ndarray, dt: float,
                            target_phase: float = 150.0,
                            transient_samples: int = 0) -> OscillationResults:
        """
        Complete oscillation analysis for two populations.
        
        Args:
            r1: Firing rate of population 1
            r2: Firing rate of population 2
            dt: Time step (ms)
            target_phase: Target phase difference (degrees)
            transient_samples: Number of initial samples to skip
            
        Returns:
            OscillationResults object with analysis results
        """
        # Remove transient
        r1 = r1[transient_samples:]
        r2 = r2[transient_samples:]
        
        # Check for theta oscillations
        is_theta1, amp1, freq1 = OscillationAnalyzer.check_theta(r1, dt)
        is_theta2, amp2, freq2 = OscillationAnalyzer.check_theta(r2, dt)
        
        if not (is_theta1 and is_theta2):
            return OscillationResults(
                is_theta=False, frequency=0.0, amplitude=0.0,
                phase_diff=0.0, phase_error=180.0, error_code=1000
            )
        
        # Check amplitude comparability
        if amp1 > 0 and amp2 > 0:
            amp_ratio = amp1 / amp2
            if not (0.8 < amp_ratio < 1.25):
                return OscillationResults(
                    is_theta=True, frequency=min(freq1, freq2),
                    amplitude=(amp1 + amp2) / 2, phase_diff=0.0,
                    phase_error=180.0, error_code=500
                )
        
        # Compute phase difference
        phase1 = OscillationAnalyzer.compute_phase(r1)
        phase2 = OscillationAnalyzer.compute_phase(r2)
        phase_diff = OscillationAnalyzer.compute_phase_difference(phase1, phase2)
        
        # Compute phase error (minimal angular deviation)
        phase_error = min(
            abs(phase_diff - target_phase),
            360 - abs(phase_diff - target_phase)
        )
        
        return OscillationResults(
            is_theta=True,
            frequency=min(freq1, freq2),
            amplitude=(amp1 + amp2) / 2,
            phase_diff=phase_diff,
            phase_error=phase_error,
            error_code=0
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_default_optimal_parameters() -> ModelParameters:
    """
    Get default optimized parameters from the paper.
    
    These parameters produce theta oscillations with ~150 degree phase shift.
    """
    return ModelParameters(
        g12=2249.0,
        g21=2138.0,
        Iext1=876.0,
        Iext2=853.0,
        Delta=225.0,
        tau_d=3.83,
        tau_r=317.56,
        tau_f=15.09,
        Uinc=0.27
    )


def smooth_signal(signal: np.ndarray, sigma: float = 220) -> np.ndarray:
    """
    Apply Gaussian smoothing to signal.
    
    Args:
        signal: Input signal
        sigma: Smoothing kernel width (samples)
        
    Returns:
        Smoothed signal
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(signal, sigma=sigma)


if __name__ == "__main__":
    # Quick test
    print("Testing theta rhythm model...")
    
    params = get_default_optimal_parameters()
    network = ThetaRhythmNetwork(params)
    
    # Run simulation
    results = network.simulate_meanfield(t_max=1300.0, dt=0.005)
    
    # Analyze
    analysis = OscillationAnalyzer.analyze_oscillation(
        results['r1'], results['r2'], dt=0.005, transient_samples=15000
    )
    
    print(f"Is theta: {analysis.is_theta}")
    print(f"Frequency: {analysis.frequency:.2f} Hz")
    print(f"Amplitude: {analysis.amplitude:.2f} Hz")
    print(f"Phase difference: {analysis.phase_diff:.1f} degrees")
    print(f"Phase error: {analysis.phase_error:.1f} degrees")
