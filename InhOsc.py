"""
Theta Rhythm Generation Model
=============================

Computational model of theta rhythm generation in the medial septum
via short-term synaptic depression (STD) between two mutually
inhibitory populations of PV+ GABAergic neurons.

Reference:
    Skorokhod et al. - "The medial septum model as a pacemaker of theta rhythm"

Author: S.N. Skorokhod, S.V. Dubrovin, I.E. Mysin (refactored 2026)
"""

import numpy as np
from scipy.stats import cauchy
from scipy.signal import hilbert, find_peaks, windows
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
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
        g12: Maximal synaptic conductance from population 1 to 2 (nS).
        g21: Maximal synaptic conductance from population 2 to 1 (nS).
        Iext1: External current to population 1 (pA).
        Iext2: External current to population 2 (pA).
        Delta: Scale parameter for the Cauchy distribution of currents.
        tau_d: Neurotransmitter decay time constant (ms).
        tau_r: Recovery time constant from depression (ms).
        tau_f: Facilitation time constant (ms).
        Uinc: Incremental release probability per spike.
    """
    g12: float = 3000.0
    g21: float = 3000.0
    Iext1: float = 800.0
    Iext2: float = 800.0
    Delta: float = 200.0
    tau_d: float = 3.83
    tau_r: float = 317.56
    tau_f: float = 15.09
    Uinc: float = 0.27

    def to_array(self) -> np.ndarray:
        """Convert parameters to a 9-element array for optimization."""
        return np.array([
            self.g12, self.g21, self.Iext1, self.Iext2,
            self.Delta, self.tau_d, self.tau_r, self.tau_f, self.Uinc
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ModelParameters':
        """Create parameters from a 9-element array."""
        return cls(
            g12=arr[0], g21=arr[1], Iext1=arr[2], Iext2=arr[3],
            Delta=arr[4], tau_d=arr[5], tau_r=arr[6], tau_f=arr[7], Uinc=arr[8]
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelParameters':
        """Create parameters from a dictionary."""
        return cls(**d)

    @property
    def g_ratio(self) -> float:
        """Ratio of synaptic conductances g12 / g21."""
        return self.g12 / self.g21 if self.g21 != 0 else np.inf

    @property
    def I_ratio(self) -> float:
        """Ratio of external currents Iext2 / Iext1."""
        return self.Iext2 / self.Iext1 if self.Iext1 != 0 else np.inf


@dataclass
class NeuronParameters:
    """
    Container for QIF (quadratic integrate-and-fire) neuron parameters.

    Defaults correspond to fast-spiking neurons typical of medial septum
    PV+ interneurons.
    """
    a: float = 1.0           # Scaling coefficient (mV^-1)
    b: float = 95.0          # Linear coefficient (dimensionless)
    c: float = 2200.0        # Constant term (mV^2/ms)
    tau: float = 40.0        # Membrane time constant (ms)
    V_rest: float = -58.0    # Resting potential (mV)
    V_peak: float = 20.0     # Peak voltage threshold (mV)
    V_reset: float = -70.0   # Reset voltage (mV)
    E_rev: float = -80.0     # Reversal potential for inhibition (mV)


@dataclass
class OscillationResults:
    """
    Container for oscillation analysis results.

    Attributes:
        is_theta: Whether both populations show theta oscillations.
        frequency: Dominant oscillation frequency (Hz).
        amplitude: Oscillation amplitude (Hz).
        phase_diff: Phase difference between populations (degrees).
        phase_error: Absolute deviation from the target phase (degrees).
        error_code: 0 = success, 500 = bad amplitudes, 1000 = no theta.
    """
    is_theta: bool
    frequency: float
    amplitude: float
    phase_diff: float
    phase_error: float
    error_code: int


# =============================================================================
# Tsodyks-Markram Synapse Model
# =============================================================================

class TsodyksMarkramSynapse:
    """
    Tsodyks-Markram model of short-term synaptic plasticity.

    Tracks three state variables:
        - u: instantaneous release probability.
        - x: fraction of resources available for release.
        - y: fraction of active resources in the synaptic cleft.

    The inactive fraction z is implicit through the conservation law
    x + y + z = 1.

    References:
        Tsodyks, Markram (1997), Neural Computation.
    """

    def __init__(self, tau_d: float, tau_r: float, tau_f: float, Uinc: float):
        """
        Args:
            tau_d: Neurotransmitter decay time constant (ms).
            tau_r: Recovery time constant from depression (ms).
            tau_f: Facilitation time constant (ms).
            Uinc: Incremental release probability per spike.
        """
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.tau_f = tau_f
        self.Uinc = Uinc

        self.g1 = 0.0
        self.g_s = 0.0

        # Pre-compute tau_d / (tau_d - tau_r) for the analytical solution
        # of the continuous-decay phase. Avoids division by zero when
        # tau_d == tau_r.
        self._tau_ratio = np.where(
            tau_d != tau_r,
            tau_d / (tau_d - tau_r),
            1e-13
        )

    def update(self, r: float, y: float, x: float, u: float,
               dt: float) -> Tuple[float, float, float]:
        """
        Advance synapse state by one time step.

        Uses an analytical solution for the continuous dynamics followed
        by a discrete spike-triggered update.

        Args:
            r: Presynaptic firing rate (spikes/ms).
            y: Current fraction of active resources.
            x: Current fraction of available resources.
            u: Current release probability.
            dt: Time step (ms).

        Returns:
            Tuple (new_y, new_x, new_u).
        """
        exp_d = np.exp(-dt / self.tau_d)
        exp_r = np.exp(-dt / self.tau_r)
        exp_f = np.exp(-dt / self.tau_f)

        # Continuous decay
        y_decay = y * exp_d
        x_decay = 1 + (x - 1 + self._tau_ratio * y) * exp_r - self._tau_ratio * y
        u_decay = u * exp_f

        # Spike-triggered release (rate-based, r * dt = expected #spikes in dt)
        released = u_decay * x_decay * r * dt

        u_new = u_decay + self.Uinc * (1 - u_decay) * r * dt
        y_new = y_decay + released
        x_new = x_decay - released

        return y_new, x_new, u_new

    def update_no_std(self, firing_rate: float, dt: float) -> float:
        """
        Second-order synapse without short-term plasticity.

        Implements two cascaded first-order filters equivalent to:
            tau_1 * tau_2 * d^2 g_s / dt^2
                + (tau_1 + tau_2) * d g_s / dt
                + g_s = w * nu_pre

        with tau_1 = tau_d and tau_2 = tau_f. Used for control simulations
        demonstrating that STD is necessary for theta oscillations.

        Args:
            firing_rate: Presynaptic firing rate (spikes/ms).
            dt: Time step (ms).

        Returns:
            Synaptic conductance g_s.
        """
        self.g1 += dt * (self.Uinc * firing_rate - self.g1) / self.tau_d
        self.g_s += dt * (self.g1 - self.g_s) / self.tau_f
        return self.g_s

    def get_conductance(self, g_max: float, y: float) -> float:
        """
        Effective synaptic conductance.

        Args:
            g_max: Maximal synaptic conductance (nS).
            y: Fraction of active resources.

        Returns:
            Effective conductance (nS).
        """
        return g_max * y


# =============================================================================
# Mean-Field Population Model
# =============================================================================

class MeanFieldPopulation:
    """
    Mean-field model of a QIF neuron population.

    Uses the Lorentzian ansatz for the firing-rate / mean-membrane-potential
    reduction (Montbrio, Deco, et al.). Heterogeneity in the external drive
    is captured by a Cauchy distribution with scale parameter Delta.
    """

    def __init__(self, params: NeuronParameters, Iext: float,
                 Delta: float, etabar: float = 0.0):
        """
        Args:
            params: Neuron parameters.
            Iext: Mean external current (pA).
            Delta: Cauchy distribution scale (pA).
            etabar: Mean of the Cauchy distribution.
        """
        self.params = params
        self.Iext = Iext
        self.Delta = Delta
        self.etabar = etabar

        self.r = 0.0                    # Firing rate (spikes/ms)
        self.v = params.V_rest          # Mean membrane potential (mV)

    def reset(self):
        """Reset state variables to initial conditions."""
        self.r = 0.0
        self.v = self.params.V_rest

    def update(self, g_total: float, dt: float) -> Tuple[float, float]:
        """
        Advance population state by one time step using 4th-order Runge-Kutta.

        Args:
            g_total: Total inhibitory conductance received (nS).
            dt: Time step (ms).

        Returns:
            Tuple (new_r, new_v).
        """
        p = self.params
        r, v = self.r, self.v

        def f_r(r_, v_):
            return ((p.b - g_total) * r_
                    + 2 * p.a * r_ * v_
                    + self.Delta * p.a / (np.pi * p.tau)) / p.tau

        def f_v(r_, v_):
            return (-(np.pi * r_ * p.tau) ** 2 / p.a
                    + p.a * v_ ** 2
                    + (p.b - g_total) * v_
                    + p.c
                    + g_total * p.E_rev
                    + self.Iext
                    + self.etabar) / p.tau

        k1r = dt * f_r(r, v)
        k1v = dt * f_v(r, v)

        k2r = dt * f_r(r + k1r / 2, v + k1v / 2)
        k2v = dt * f_v(r + k1r / 2, v + k1v / 2)

        k3r = dt * f_r(r + k2r / 2, v + k2v / 2)
        k3v = dt * f_v(r + k2r / 2, v + k2v / 2)

        k4r = dt * f_r(r + k3r, v + k3v)
        k4v = dt * f_v(r + k3r, v + k3v)

        self.r = r + (k1r + 2 * k2r + 2 * k3r + k4r) / 6
        self.v = v + (k1v + 2 * k2v + 2 * k3v + k4v) / 6

        return self.r, self.v


# =============================================================================
# Spiking Neuron Network Model
# =============================================================================

class SpikingNeuronNetwork:
    """
    Network of spiking QIF neurons used to validate the mean-field model.

    Implements a population of N neurons with heterogeneous currents
    (Cauchy-distributed) and conductance-based synapses.
    """

    def __init__(self, N: int, params: NeuronParameters,
                 Iext: float, Delta: float, seed: Optional[int] = None):
        """
        Args:
            N: Number of neurons in the population.
            params: Neuron parameters.
            Iext: Mean external current (pA).
            Delta: Cauchy distribution scale (pA).
            seed: Random seed for reproducibility.
        """
        self.N = N
        self.params = params

        if seed is not None:
            np.random.seed(seed)

        # Heterogeneous currents drawn from a Cauchy distribution.
        self.I_het = np.sort(cauchy.rvs(loc=0, scale=Delta, size=N))
        self.Iext = Iext

        self.V = np.full(N, params.V_rest)

    def reset(self, seed: Optional[int] = None):
        """Reset network to initial conditions."""
        if seed is not None:
            np.random.seed(seed)
        self.V = np.full(self.N, self.params.V_rest)

    def update(self, g_syn: float, dt: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Advance all neurons by one time step using 4th-order Runge-Kutta.

        Args:
            g_syn: Total inhibitory synaptic conductance (nS).
            dt: Time step (ms).

        Returns:
            Tuple (spiked_mask, population_rate, V).
        """
        p = self.params

        def f_V(V_):
            # Per-neuron conductance-based current (this matches the
            # mean-field formulation, where g enters as (b - g)).
            I_total = self.Iext + self.I_het + g_syn * (p.E_rev - V_)
            return (p.a * V_ ** 2 + p.b * V_ + p.c + I_total) / p.tau

        k1 = dt * f_V(self.V)
        k2 = dt * f_V(self.V + k1 / 2)
        k3 = dt * f_V(self.V + k2 / 2)
        k4 = dt * f_V(self.V + k3)

        self.V = self.V + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        spiked = self.V >= p.V_peak
        self.V[spiked] = p.V_reset

        rate = np.sum(spiked) / (self.N * dt)
        return spiked, rate, self.V


# =============================================================================
# Complete Network Model
# =============================================================================

class ThetaRhythmNetwork:
    """
    Complete model of two mutually inhibitory populations connected
    via Tsodyks-Markram depressing synapses.

    Can run either a mean-field or a spiking simulation.
    """

    def __init__(self, params: ModelParameters,
                 neuron_params: Optional[NeuronParameters] = None):
        """
        Args:
            params: Model parameters.
            neuron_params: Neuron parameters (optional; defaults used if None).
        """
        self.params = params
        self.neuron_params = neuron_params or NeuronParameters()

        self.synapse_12 = TsodyksMarkramSynapse(
            params.tau_d, params.tau_r, params.tau_f, params.Uinc
        )
        self.synapse_21 = TsodyksMarkramSynapse(
            params.tau_d, params.tau_r, params.tau_f, params.Uinc
        )

    def simulate_meanfield(self, t_max: float, dt: float,
                           transient: float = 300.0) -> Dict[str, np.ndarray]:
        """
        Run a mean-field simulation.

        Args:
            t_max: Total simulation time (ms).
            dt: Time step (ms).
            transient: Transient time (ms); not used internally but
                recorded for downstream analysis.

        Returns:
            Dictionary with keys:
                't'   : time array,
                'r1', 'r2' : firing rates of populations 1 and 2,
                'I12', 'I21' : synaptic currents,
                'x1', 'x2' : available neurotransmitter fractions.
        """
        t = np.arange(0, t_max, dt)
        n_steps = len(t)

        pop1 = MeanFieldPopulation(
            self.neuron_params, self.params.Iext1, self.params.Delta
        )
        pop2 = MeanFieldPopulation(
            self.neuron_params, self.params.Iext2, self.params.Delta
        )

        # Synaptic state: active (y), available (x), release probability (u).
        y12, x12, u12 = 0.0, 1.0, 0.0
        y21, x21, u21 = 0.0, 1.0, 0.0

        r1_arr = np.zeros(n_steps)
        r2_arr = np.zeros(n_steps)
        I12_arr = np.zeros(n_steps)
        I21_arr = np.zeros(n_steps)
        x12_arr = np.ones(n_steps)
        x21_arr = np.ones(n_steps)

        for i in range(n_steps - 1):
            # Synapse 1->2 is driven by population 1 firing rate;
            # synapse 2->1 is driven by population 2 firing rate.
            y12, x12, u12 = self.synapse_12.update(pop2.r, y12, x12, u12, dt)
            y21, x21, u21 = self.synapse_21.update(pop1.r, y21, x21, u21, dt)

            g12 = self.synapse_12.get_conductance(self.params.g12, y12)
            g21 = self.synapse_21.get_conductance(self.params.g21, y21)

            I12_arr[i] = g12 * (self.neuron_params.E_rev - pop1.v)
            I21_arr[i] = g21 * (self.neuron_params.E_rev - pop2.v)

            r1_arr[i], _ = pop1.update(g12, dt)
            r2_arr[i], _ = pop2.update(g21, dt)

            x12_arr[i] = x12
            x21_arr[i] = x21

        return {
            't': t,
            'r1': r1_arr,
            'r2': r2_arr,
            'I12': I12_arr,
            'I21': I21_arr,
            'x1': x21_arr,   # available resource seen by population 1
            'x2': x12_arr,   # available resource seen by population 2
        }

    def simulate_spiking(self, t_max: float, dt: float, N: int = 2000,
                         seed: Optional[int] = None,
                         transient: float = 300.0) -> Dict[str, np.ndarray]:
        """
        Run a spiking neuron network simulation.

        Args:
            t_max: Total simulation time (ms).
            dt: Time step (ms).
            N: Number of neurons per population.
            seed: Random seed for reproducibility.
            transient: Transient time (ms).

        Returns:
            Dictionary with keys:
                't', 'r1', 'r2', 'x1', 'x2',
                'spikes1', 'spikes2' : boolean spike trains (N x time),
                'v1', 'v2' : membrane potentials (N x time).
        """
        t = np.arange(0, t_max, dt)
        n_steps = len(t)

        net1 = SpikingNeuronNetwork(N, self.neuron_params,
                                    self.params.Iext1, self.params.Delta, seed)
        net2 = SpikingNeuronNetwork(N, self.neuron_params,
                                    self.params.Iext2, self.params.Delta,
                                    seed + 1 if seed is not None else None)

        y12, x12, u12 = 0.0, 1.0, 0.0
        y21, x21, u21 = 0.0, 1.0, 0.0

        r1_arr = np.zeros(n_steps)
        r2_arr = np.zeros(n_steps)
        x12_arr = np.ones(n_steps)
        x21_arr = np.ones(n_steps)

        spikes1 = np.zeros((N, n_steps), dtype=bool)
        spikes2 = np.zeros((N, n_steps), dtype=bool)

        # Per-neuron membrane potentials (uses significant memory for large N).
        v1_arr = np.zeros((N, n_steps))
        v2_arr = np.zeros((N, n_steps))
        v1_arr[:, 0] = net1.V
        v2_arr[:, 0] = net2.V

        for i in range(n_steps - 1):
            y12, x12, u12 = self.synapse_12.update(r2_arr[i], y12, x12, u12, dt)
            y21, x21, u21 = self.synapse_21.update(r1_arr[i], y21, x21, u21, dt)

            g12 = self.synapse_12.get_conductance(self.params.g12, y12)
            g21 = self.synapse_21.get_conductance(self.params.g21, y21)

            spike1, rate1, V1 = net1.update(g12, dt)
            spike2, rate2, V2 = net2.update(g21, dt)

            r1_arr[i + 1] = rate1
            r2_arr[i + 1] = rate2
            x12_arr[i + 1] = x12
            x21_arr[i + 1] = x21
            spikes1[:, i + 1] = spike1
            spikes2[:, i + 1] = spike2

            v1_arr[:, i + 1] = V1
            v2_arr[:, i + 1] = V2

        return {
            't': t,
            'r1': r1_arr,
            'r2': r2_arr,
            'x1': x21_arr,
            'x2': x12_arr,
            'spikes1': spikes1,
            'spikes2': spikes2,
            'v1': v1_arr,
            'v2': v2_arr,
        }

    def simulate_without_std(self, t_max: float, dt: float,
                             x_fixed: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Run a mean-field simulation with STD disabled (constant
        neurotransmitter availability).

        Used as a control to demonstrate that STD is necessary for
        theta oscillations.

        Args:
            t_max: Total simulation time (ms).
            dt: Time step (ms).
            x_fixed: Fixed value of available neurotransmitter (0 to 1).

        Returns:
            Dictionary with keys 't', 'r1', 'r2'.
        """
        t = np.arange(0, t_max, dt)
        n_steps = len(t)

        pop1 = MeanFieldPopulation(
            self.neuron_params, self.params.Iext1, self.params.Delta
        )
        pop2 = MeanFieldPopulation(
            self.neuron_params, self.params.Iext2, self.params.Delta
        )

        r1_arr = np.zeros(n_steps)
        r2_arr = np.zeros(n_steps)

        # Effective conductance is g * x_fixed (constant).
        g12_eff = self.params.g12 * x_fixed
        g21_eff = self.params.g21 * x_fixed

        for i in range(n_steps - 1):
            r1_arr[i], _ = pop1.update(g12_eff, dt)
            r2_arr[i], _ = pop2.update(g21_eff, dt)

        return {
            't': t,
            'r1': r1_arr,
            'r2': r2_arr,
        }


# =============================================================================
# Analysis Functions
# =============================================================================

class OscillationAnalyzer:
    """
    Analyzer for theta oscillation detection and characterization.

    Provides:
        - Frequency/amplitude estimation via FFT.
        - Phase analysis via Hilbert transform.
        - Theta-rhythm detection by spectral, amplitude, and stability criteria.
    """

    @staticmethod
    def compute_fft(signal: np.ndarray, fs: float) -> Tuple[float, float]:
        """
        Compute dominant frequency and amplitude via FFT.

        Args:
            signal: Input signal.
            fs: Sampling frequency (kHz), i.e. 1/dt with dt in ms.

        Returns:
            Tuple (dominant_frequency_Hz, amplitude).
        """
        n = len(signal)
        signal_centered = signal - np.mean(signal)

        window = windows.hann(n)
        signal_windowed = signal_centered * window

        yf = fft(signal_windowed)
        xf = fftfreq(n, 1 / fs)

        # Find the peak in the positive frequencies (skip DC).
        idx = np.argmax(np.abs(yf[1:n // 2])) + 1
        freq = xf[idx] * 1000  # kHz -> Hz

        amp = 2 * np.abs(yf[idx]) / n

        return freq, amp

    @staticmethod
    def compute_phase(signal: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous phase via Hilbert transform.

        Args:
            signal: Input signal.

        Returns:
            Unwrapped phase array (radians).
        """
        signal_centered = signal - np.mean(signal)
        analytic = hilbert(signal_centered)
        return np.unwrap(np.angle(analytic))

    @staticmethod
    def compute_phase_difference(phase1: np.ndarray,
                                 phase2: np.ndarray) -> float:
        """
        Compute the circular mean phase difference.

        Args:
            phase1: Phase of the first signal.
            phase2: Phase of the second signal.

        Returns:
            Mean phase difference (degrees).
        """
        e_ip1 = np.exp(1j * phase1)
        e_ip2 = np.exp(1j * phase2)
        mean_diff = np.angle(np.mean(e_ip2 * np.conj(e_ip1)))
        return np.degrees(mean_diff)

    @staticmethod
    def check_theta(signal: np.ndarray, dt: float,
                    freq_range: Tuple[float, float] = (4.0, 12.0),
                    rate_range: Tuple[float, float] = (2.0, 800.0)
                    ) -> Tuple[bool, float, float]:
        """
        Check whether a signal represents a theta oscillation.

        Criteria:
            - Dominant frequency in the theta range (default 4-12 Hz).
            - Maximum rate within a valid range.
            - Stable amplitude (no strong damping).

        Args:
            signal: Firing rate signal (spikes/ms).
            dt: Time step (ms).
            freq_range: Valid frequency range (Hz).
            rate_range: Valid maximum-rate range (Hz).

        Returns:
            Tuple (is_theta, amplitude_Hz, frequency_Hz).
        """
        fs = 1 / dt  # kHz

        freq, amp = OscillationAnalyzer.compute_fft(signal, fs)
        max_rate = np.max(signal) * 1000  # Hz

        if not (freq_range[0] < freq < freq_range[1]):
            return False, 0.0, 0.0

        if not (rate_range[0] < max_rate < rate_range[1]):
            return False, 0.0, 0.0

        # Check for stable oscillation: require several peaks of comparable height.
        period_samples = int(0.9 * 1000 / freq / dt)
        peaks, _ = find_peaks(signal, distance=period_samples,
                              height=max_rate * 0.8 / 1000)

        if len(peaks) < 4:
            return False, 0.0, 0.0

        peak_heights = signal[peaks]
        peak_range = np.max(peak_heights) - np.min(peak_heights)
        amplitude = np.max(peak_heights) - np.min(signal)

        if amplitude > 0 and peak_range / amplitude > 0.1:
            return False, 0.0, 0.0

        return True, amplitude * 1000, freq

    @staticmethod
    def analyze_oscillation(r1: np.ndarray, r2: np.ndarray, dt: float,
                            target_phase: float = 150.0,
                            transient_samples: int = 0) -> OscillationResults:
        """
        Complete oscillation analysis for two populations.

        Args:
            r1: Firing rate of population 1 (spikes/ms).
            r2: Firing rate of population 2 (spikes/ms).
            dt: Time step (ms).
            target_phase: Target phase difference (degrees).
            transient_samples: Number of initial samples to discard.

        Returns:
            OscillationResults with all metrics.
        """
        # Remove transient and smooth.
        r1 = gaussian_filter1d(r1[transient_samples:], sigma=100)
        r2 = gaussian_filter1d(r2[transient_samples:], sigma=100)

        is_theta1, amp1, freq1 = OscillationAnalyzer.check_theta(r1, dt)
        is_theta2, amp2, freq2 = OscillationAnalyzer.check_theta(r2, dt)

        if not (is_theta1 and is_theta2):
            return OscillationResults(
                is_theta=False, frequency=0.0, amplitude=0.0,
                phase_diff=0.0, phase_error=180.0, error_code=1000
            )

        # Require comparable amplitudes in the two populations.
        if amp1 > 0 and amp2 > 0:
            amp_ratio = amp1 / amp2
            if not (0.6 < amp_ratio < 1.5):
                return OscillationResults(
                    is_theta=True, frequency=min(freq1, freq2),
                    amplitude=(amp1 + amp2) / 2, phase_diff=0.0,
                    phase_error=180.0, error_code=500
                )

        # Phase difference (absolute value, so swapping populations is symmetric).
        phase1 = OscillationAnalyzer.compute_phase(r1)
        phase2 = OscillationAnalyzer.compute_phase(r2)
        phase_diff = OscillationAnalyzer.compute_phase_difference(phase1, phase2)
        phase_diff = abs(phase_diff)

        # Minimal angular deviation from the target phase.
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
            error_code=phase_error
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_default_optimal_parameters() -> ModelParameters:
    """
    Default parameters that produce theta oscillations with a ~150 deg
    inter-population phase difference.
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
    Apply Gaussian smoothing to a signal.

    Args:
        signal: Input signal.
        sigma: Smoothing kernel width (samples).

    Returns:
        Smoothed signal.
    """
    return gaussian_filter1d(signal, sigma=sigma)


if __name__ == "__main__":
    print("Testing theta rhythm model...")

    params = get_default_optimal_parameters()
    network = ThetaRhythmNetwork(params)

    results = network.simulate_meanfield(t_max=1300.0, dt=0.005)

    analysis = OscillationAnalyzer.analyze_oscillation(
        results['r1'], results['r2'], dt=0.005, transient_samples=15000
    )

    print(f"Is theta: {analysis.is_theta}")
    print(f"Frequency: {analysis.frequency:.2f} Hz")
    print(f"Amplitude: {analysis.amplitude:.2f} Hz")
    print(f"Phase difference: {analysis.phase_diff:.1f} degrees")
    print(f"Phase error: {analysis.phase_error:.1f} degrees")
