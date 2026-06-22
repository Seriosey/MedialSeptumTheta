# Theta Rhythm Generation in the Medial Septum

A biophysically grounded computational model of theta (4–12 Hz) rhythm
generation by two mutually inhibitory GABAergic populations of the medial
septum (MS), coupled via Tsodyks–Markram depressing synapses. The model
supports both a low-dimensional mean-field reduction and a spiking-network
simulation, plus a complete parameter-sensitivity pipeline.

---

## Repository layout

| File                 | Role                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------ |
| `InhOsc.py`          | Core model: QIF mean-field population, spiking network, Tsodyks–Markram synapse, oscillation analyzer. |
| `Analysis.py`        | Side-by-side validation of the mean-field and spiking models (firing rates, NT fraction, V_m). |
| `GridSearchFull.py`  | End-to-end pipeline: Sobol sampling → simulations → Random Forest (classifier + regressors) → Permutation Importance → plots. |
| `HeatPlots.py`       | Standalone 3-panel heatmaps (frequency / amplitude / phase error) sweeping two parameters around a center. |
| `Importance.py`      | Standalone LHS + Random Forest + Permutation Importance pipeline (binary `is_theta` target). |
| `Optimization.py`    | Optuna (TPE) optimization of all 9 parameters against the 150° phase-shift target.         |

`InhOsc.py` is imported by every other file; the remaining files are
independent entry points.

---

## Model overview

The model has **9 free parameters**:

| Symbol      | Name     | Description                                             | Unit |
| ----------- | -------- | ------------------------------------------------------- | ---- |
| `g12`       | g₁₂      | Max synaptic conductance, pop 1 → pop 2                 | nS   |
| `g21`       | g₂₁      | Max synaptic conductance, pop 2 → pop 1                 | nS   |
| `Iext1`     | I_ext,1  | External drive current to pop 1                         | pA   |
| `Iext2`     | I_ext,2  | External drive current to pop 2                         | pA   |
| `Delta`     | Δ        | Cauchy scale of current heterogeneity                   | pA   |
| `tau_d`     | τ_d      | Neurotransmitter decay (IPSC decay)                     | ms   |
| `tau_r`     | τ_r      | Recovery from depression                                | ms   |
| `tau_f`     | τ_f      | Facilitation time constant                              | ms   |
| `Uinc`      | U        | Incremental release probability per spike               | —    |

### Mean-field model

QIF neurons with Lorentzian ansatz (Montbrio et al.), per population:

```
τ · dr/dt = (b - g) · r + 2 a r v + Δ a / (π τ)
τ · dv/dt = -(π τ r)² / a + a v² + (b - g) v + c + g E_rev + I_ext + η̄
```

integrated with 4th-order Runge–Kutta. `g` is the inhibitory conductance
returned by the Tsodyks–Markram synapse (per presynaptic population).

### Spiking model

Network of N QIF neurons per population with conductance-based synapses
and Cauchy-distributed heterogeneity `I_het ~ Cauchy(0, Δ)`. Per-neuron
current is `I_total = I_ext + I_het + g_syn * (E_rev - V_i)`. Also
integrated with RK4.

### Tsodyks–Markram synapse

Three state variables (`x` available, `y` active, `u` release probability),
conservation `x + y + z = 1` (z implicit). Analytical solution between
spikes; rate-based discrete release update with expected spike count
`r · dt`.

### Oscillation analysis

`OscillationAnalyzer.analyze_oscillation()` returns an `OscillationResults`
object with:

- `is_theta` — both populations show 4–12 Hz oscillation, comparable
  amplitudes, and stable peak structure (≥ 4 peaks of comparable height).
- `frequency` — dominant FFT frequency (Hz).
- `amplitude` — peak-to-trough firing rate (Hz).
- `phase_diff` — absolute circular mean phase difference (degrees),
  symmetric under population swap.
- `phase_error` — minimal angular deviation from the 150° target.
- `error_code` — 0 (success), 500 (amplitude mismatch), 1000 (no theta).

---

## Installation

```bash
pip install numpy pandas scipy matplotlib scikit-learn tqdm optuna progress
```

Python ≥ 3.9 is recommended.

---

## Reproducing the results

### 1. Quick smoke test

```bash
python InhOsc.py
```

Runs a 1.3 s simulation with the default optimal parameters and prints
the oscillation metrics. Should report `Is theta: True` with frequency
near 6–8 Hz and phase error well below 30°.

### 2. Mean-field vs spiking comparison

```bash
python Analysis.py
```

Runs both models for 2.6 s on a representative parameter set and saves
a 2×2 comparison figure (`results.png` by default — edit `save_path`
in the `__main__` block to change the output directory).

### 3. Full grid search (Sobol + RF + Permutation Importance)

```bash
python GridSearchFull.py
```

Steps performed:

1. Sobol-samples **100 000** points from the 9-dimensional parameter
   space (bounded by the `param_bounds` dict at the bottom of the file).
2. Simulates each point and records `(is_theta, amplitude, error_code,
   frequency)`. The loop checkpoints every 1000 simulations to
   `checkpoints/checkpoint_full.npz` so it can be resumed.
3. Trains:
   - A balanced Random Forest **classifier** on `is_theta`
     (OOB score, ROC-AUC cross-validation).
   - Three Random Forest **regressors** on amplitude / frequency /
     phase_error, restricted to theta-positive samples.
4. Computes Permutation Importance (30 repeats, ROC-AUC for the
   classifier and R² for the regressors).
5. Saves CSV of all simulations, importance plots, marginal
   distributions, a 2D heatmap of P(theta), and an optimal-region
   summary to `good_borders_grid_search_results/`.

> **Note:** the `__main__` block currently loads results from a
> pre-computed checkpoint (`good_borders_checkpoints_grid/checkpoint_full.npz`).
> To run from scratch, replace that line with:
>
> ```python
> y = run_simulations_parallel(samples, checkpoint_dir='good_borders_checkpoints_grid')
> ```

Expected runtime on a single CPU core: ~30–60 hours for 100k samples.
Reduce `n_samples` to 5000–10000 for a quick test.

### 4. Two-parameter heatmaps

```bash
python HeatPlots.py
```

Sweeps a 100×100 grid over two chosen parameters (edit the call to
`heat_plot_from_center('g12', 'g21', config)` at the bottom of the file)
around a central parameter set and produces a 3-panel heatmap
(frequency, amplitude, phase error) in `good_borders_grid_search_results/`.

### 5. Standalone sensitivity analysis (LHS-based)

```bash
python Importance.py
```

Same idea as the grid-search pipeline but uses Latin Hypercube Sampling
and only the binary `is_theta` target. Saves CSVs, a text report, and
four diagnostic plots to `good_bounds_sensitivity_results/`.

### 6. Optuna parameter optimization

```bash
python Optimization.py
```

Maximizes theta-rhythm quality (minimizes phase error) over the full
9-dimensional parameter space using Optuna's TPE sampler. The study is
backed by `gb_optuna_study.db`, so it is automatically resumed on
re-runs. The default block runs 50 000 trials with `n_jobs=-1`; reduce
both for a quick test.

To inspect the best trial after the run:

```python
import optuna
study = optuna.load_study(study_name="gb_jt_study",
                          storage="sqlite:///gb_optuna_study.db")
print(study.best_params, study.best_value)
```

---

## Output artifacts

| Directory                                | Produced by         | Contents                                                            |
| ---------------------------------------- | ------------------- | ------------------------------------------------------------------- |
| `good_borders_grid_search_results/`      | `GridSearchFull.py` | CSV of all simulations, importance plots, marginals, 2D heatmap, optimal regions, per-pair `.npz` and `.png` heatmaps. |
| `good_bounds_sensitivity_results/`       | `Importance.py`     | Simulation CSV, importance CSV, text report, bar / box / distribution / correlation plots. |
| `checkpoints/checkpoint_full.npz`        | `GridSearchFull.py` | Resumable simulation checkpoint (results array + last index).       |
| `gb_optuna_study.db`                     | `Optimization.py`   | SQLite database of all Optuna trials.                               |
| `results.png` (or as set by `save_path`) | `Analysis.py`       | Mean-field vs spiking comparison figure.                            |

---

## Key findings

When the pipeline is run on the default 100k Sobol sample, the
permutation-importance analysis shows a clear **functional dissociation**:

- **Amplitude** is governed by the external drives `Iext1` / `Iext2`
  (importance ≈ 0.75–0.78) — these set the peak firing rate during the
  active phase.
- **Frequency** is governed primarily by the synaptic decay time
  `tau_d` (importance ≈ 0.78) — even though `tau_d` (~5 ms) is much
  shorter than the theta period (~150 ms), it sets the cumulative IPSC
  duration per cycle.
- **Phase error** (deviation from the experimentally observed ~150°
  shift) is governed by the asymmetry between `Iext1`/`Iext2` and
  between `g12`/`g21`. Symmetric parameters produce a ~180° (antiphase)
  relationship; the 150° shift requires systematic asymmetry in drives
  and/or coupling strengths.

---

## References

- Tsodyks, M. V., & Markram, H. (1997). The neural code between neocortical
  pyramidal cells depends on neurotransmitter release probability. *PNAS*,
  94(2), 719–723.
- Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for
  networks of spiking neurons. *Physical Review X*, 5(2), 021028.
- Wheeler, D. W., et al. (2015). Hippocampome.org: A knowledge base of
  neuron types in the hippocampal formation. *Frontiers in Neural Circuits*,
  9, 78.

---

## Authors

S. N. Skorokhod, S. V. Dubrovin, I. E. Mysin
