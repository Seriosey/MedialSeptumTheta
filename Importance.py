"""
Sensitivity Analysis: LHS + Random Forest + Permutation Importance
==================================================================

Pipeline:
    1. Latin Hypercube Sampling (LHS) of the 9-dimensional parameter space.
    2. Mean-field simulation for each sample -> binary target (is_theta).
    3. Random Forest classifier with balanced class weights and OOB scoring.
    4. Permutation Importance (ROC-AUC, 20 repeats) for parameter ranking.
    5. Save CSV / PNG / TXT reports.

Recommended runtime: 2-4 hours for the default 100k samples.

Author: S.N. Skorokhod, S.V. Dubrovin, I.E. Mysin (refactored 2026)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings

from InhOsc import (
    ModelParameters, ThetaRhythmNetwork, OscillationAnalyzer,
)

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'param_names': ['g12', 'g21', 'Iext1', 'Iext2',
                    'Delta', 'tau_d', 'tau_r', 'tau_f', 'Uinc'],
    'param_bounds': {
        'g12':   (0, 10000),    # Synaptic conductance pop1 -> pop2 (nS)
        'g21':   (0, 10000),    # Synaptic conductance pop2 -> pop1 (nS)
        'Iext1': (0, 500),      # External current to pop1 (pA)
        'Iext2': (0, 500),      # External current to pop2 (pA)
        'Delta': (1, 100),      # Heterogeneity scale (pA)
        'tau_d': (1, 21),       # Neurotransmitter decay time (ms)
        'tau_r': (50, 2000),    # Recovery time (ms)
        'tau_f': (3, 300),      # Facilitation time (ms)
        'Uinc':  (0, 1),        # Incremental release probability
    },
    'n_samples': 100000,
    'random_seed': 42,

    'rf_n_estimators': 200,
    'rf_max_depth': 15,
    'rf_min_samples_leaf': 5,

    'n_repeats': 20,           # Permutation repeats

    'output_dir': 'good_bounds_sensitivity_results',
}


# =============================================================================
# LHS sampling
# =============================================================================

def latin_hypercube_sampling(n_samples, param_bounds, seed=42):
    """
    Latin Hypercube Sampling of the parameter space.

    LHS provides:
        - Uniform coverage of each marginal dimension.
        - Better efficiency than naive random sampling.
        - Sample size independent of dimensionality.

    Args:
        n_samples: Number of samples.
        param_bounds: dict {name: (low, high)}.
        seed: Random seed.

    Returns:
        DataFrame of shape (n_samples, n_params).
    """
    np.random.seed(seed)
    n_params = len(param_bounds)
    param_names = list(param_bounds.keys())

    lhs_matrix = np.zeros((n_samples, n_params))

    for i in range(n_params):
        perm = np.random.permutation(n_samples)
        lhs_matrix[:, i] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples

    samples = np.zeros((n_samples, n_params))
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        samples[:, i] = low + lhs_matrix[:, i] * (high - low)

    return pd.DataFrame(samples, columns=param_names)


# =============================================================================
# Simulation wrapper
# =============================================================================

def simulate_theta_rhythm(params):
    """
    Run a single mean-field simulation; return True if theta is detected.
    """
    model_params = ModelParameters.from_dict(params)
    network = ThetaRhythmNetwork(model_params)
    mf_results = network.simulate_meanfield(t_max=1600, dt=0.02)
    transient = int(600 / 0.02)
    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=0.02, transient_samples=transient
    )
    return mf_analysis.is_theta


def run_simulations(samples, verbose=True):
    """Run simulations for all samples."""
    n_samples = len(samples)
    results = np.zeros(n_samples)

    if verbose:
        print(f"\n{'='*60}")
        print(f"RUNNING SIMULATIONS: {n_samples} runs")
        print(f"{'='*60}")

    start_time = time.time()

    for i, (idx, row) in enumerate(samples.iterrows()):
        params = row.to_dict()
        results[i] = simulate_theta_rhythm(params)

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"  Progress: {i+1}/{n_samples} ({100*(i+1)/n_samples:.1f}%) | "
                  f"Rate: {rate:.1f} sim/s | ETA: {eta/60:.1f} min")

    elapsed = time.time() - start_time

    if verbose:
        n_positive = results.sum()
        print(f"\n  COMPLETED in {elapsed:.1f} sec ({elapsed/60:.2f} min)")
        print(f"  Theta found: {int(n_positive)} of {n_samples} "
              f"({100*n_positive/n_samples:.1f}%)")

    return results


# =============================================================================
# Random Forest + Permutation Importance
# =============================================================================

def train_random_forest(X, y, config):
    """Train a balanced Random Forest classifier with OOB scoring."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    print(f"\n{'='*60}")
    print("TRAINING RANDOM FOREST")
    print(f"{'='*60}")

    n_positive = y.sum()
    n_negative = len(y) - n_positive
    print(f"  Class 0: {n_negative} ({100*n_negative/len(y):.1f}%)")
    print(f"  Class 1: {int(n_positive)} ({100*n_positive/len(y):.1f}%)")

    if n_positive < 10 or n_negative < 10:
        print("  WARNING: Too few samples in one of the classes!")
        print("  Results may be unreliable.")

    rf = RandomForestClassifier(
        n_estimators=config['rf_n_estimators'],
        max_depth=config['rf_max_depth'],
        min_samples_leaf=config['rf_min_samples_leaf'],
        max_features='sqrt',
        n_jobs=-1,
        random_state=config['random_seed'],
        oob_score=True,
        class_weight='balanced'
    )

    rf.fit(X, y)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc', n_jobs=-1)

    print(f"\n  OOB Score: {rf.oob_score_:.4f}")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return rf, cv_scores


def compute_permutation_importance(rf, X, y, config):
    """Compute Permutation Importance with ROC-AUC scoring."""
    from sklearn.inspection import permutation_importance

    print(f"\n{'='*60}")
    print("COMPUTING PERMUTATION IMPORTANCE")
    print(f"{'='*60}")
    print(f"  Repeats: {config['n_repeats']}")
    print(f"  Metric: ROC-AUC")

    perm = permutation_importance(
        rf, X, y,
        n_repeats=config['n_repeats'],
        scoring='roc_auc',
        n_jobs=-1,
        random_state=config['random_seed']
    )

    indices = np.argsort(perm.importances_mean)[::-1]

    print(f"\n  Results (sorted by importance):")
    print(f"  {'Param':<12} {'Importance':>12} {'± Std':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")

    for idx in indices:
        name = config['param_names'][idx]
        imp = perm.importances_mean[idx]
        std = perm.importances_std[idx]
        print(f"  {name:<12} {imp:>12.4f} {std:>12.4f}")

    return perm, indices


# =============================================================================
# Visualization
# =============================================================================

def plot_results(samples, y, rf, perm, indices, config, output_dir):
    """Create all result visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    param_names = config['param_names']

    # 1. Permutation Importance bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_names = [param_names[i] for i in indices]
    sorted_importances = perm.importances_mean[indices]
    sorted_std = perm.importances_std[indices]

    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(sorted_names)))

    bars = ax.barh(range(len(sorted_names)), sorted_importances,
                   xerr=sorted_std, align='center', color=colors,
                   ecolor='black', capsize=3)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Permutation Importance (ROC-AUC)', fontsize=12)
    ax.set_title('Parameter importance for theta-rhythm emergence', fontsize=14)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'permutation_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Box plot of importance distribution
    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_importances_all = perm.importances[indices].T

    bp = ax.boxplot(sorted_importances_all, vert=False,
                    labels=sorted_names, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Permutation Importance (ROC-AUC)', fontsize=12)
    ax.set_title(f'Importance distribution ({config["n_repeats"]} repeats)',
                 fontsize=14)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'permutation_importance_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Top-4 parameter marginal distributions
    top4_indices = indices[:4]
    top4_names = [param_names[i] for i in top4_indices]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax_idx, (param_idx, param_name) in enumerate(zip(top4_indices, top4_names)):
        ax = axes[ax_idx]

        values = samples[param_name].values
        y_arr = y.values if isinstance(y, pd.Series) else y

        class0 = values[y_arr == 0]
        class1 = values[y_arr == 1]

        ax.hist(class0, bins=30, alpha=0.5, label='No theta', color='blue')
        ax.hist(class1, bins=30, alpha=0.5, label='Theta', color='red')
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Top-{ax_idx+1}: {param_name}', fontsize=12)
        ax.legend()

    plt.suptitle('Top-4 parameter distributions by class', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Parameter correlation matrix
    corr = samples.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)

    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=11)
    ax.set_title('Parameter correlation matrix', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Plots saved to: {output_dir}")


# =============================================================================
# Save results
# =============================================================================

def save_results(samples, y, rf, perm, indices, cv_scores, config, output_dir):
    """Save CSV, importance table, and text report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    param_names = config['param_names']

    # 1. CSV with all simulation results
    df = samples.copy()
    df['theta_rhythm'] = y
    df.to_csv(output_dir / 'simulation_results.csv', index=False)

    # 2. Importance table
    importance_df = pd.DataFrame({
        'parameter': param_names,
        'importance_mean': perm.importances_mean,
        'importance_std': perm.importances_std,
        'rank': [list(indices).index(i) + 1 for i in range(len(param_names))]
    }).sort_values('rank')

    importance_df.to_csv(output_dir / 'parameter_importance.csv', index=False)

    # 3. Text report
    report = []
    report.append("=" * 70)
    report.append("SENSITIVITY ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("CONFIGURATION:")
    report.append(f"  Samples: {config['n_samples']}")
    report.append(f"  Sampling: Latin Hypercube Sampling")
    report.append(f"  Random Forest: {config['rf_n_estimators']} trees")
    report.append(f"  Permutation repeats: {config['n_repeats']}")
    report.append("")
    report.append("MODEL PERFORMANCE:")
    report.append(f"  OOB Score: {rf.oob_score_:.4f}")
    report.append(f"  CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    report.append(f"  Theta found: {int(y.sum())} of {len(y)} "
                  f"({100*y.sum()/len(y):.1f}%)")
    report.append("")
    report.append("PARAMETER RANKING (by importance):")
    report.append("-" * 50)
    report.append(f"{'Rank':<6} {'Param':<12} {'Importance':>12} {'± Std':>12}")
    report.append("-" * 50)

    for rank, idx in enumerate(indices, 1):
        name = param_names[idx]
        imp = perm.importances_mean[idx]
        std = perm.importances_std[idx]
        report.append(f"{rank:<6} {name:<12} {imp:>12.4f} {std:>12.4f}")

    report.append("")
    report.append("INTERPRETATION:")
    report.append("-" * 50)

    mean_importance = perm.importances_mean.mean()
    top_params = [param_names[i] for i in indices[:3]]

    report.append(f"  Mean importance: {mean_importance:.4f}")
    report.append(f"  Top-3 parameters: {', '.join(top_params)}")

    significant = [param_names[i] for i in indices
                   if perm.importances_mean[i] > 2 * perm.importances_std[i]]

    if significant:
        report.append(f"  Significant (importance > 2*std): {', '.join(significant)}")
    else:
        report.append("  No statistically significant parameters")

    report.append("")
    report.append("FILES:")
    report.append(f"  - simulation_results.csv: all simulation results")
    report.append(f"  - parameter_importance.csv: importance table")
    report.append(f"  - permutation_importance.png: bar chart")
    report.append(f"  - permutation_importance_boxplot.png: box plot")
    report.append(f"  - parameter_distributions.png: top-4 distributions")
    report.append(f"  - correlation_matrix.png: parameter correlations")
    report.append("")
    report.append("=" * 70)

    report_text = "\n".join(report)

    with open(output_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nFiles saved to: {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the full sensitivity-analysis pipeline."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: QIF + TSODYKS-MARKRAM")
    print("=" * 70)

    # 1. Generate samples
    print("\n[1/5] Generating samples via LHS...")
    samples = latin_hypercube_sampling(
        CONFIG['n_samples'],
        CONFIG['param_bounds'],
        CONFIG['random_seed']
    )
    print(f"  Generated {len(samples)} samples")
    print(f"  Parameters: {list(CONFIG['param_names'])}")

    # 2. Run simulations
    print("\n[2/5] Running simulations...")
    y = run_simulations(samples)

    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        print("\n  ERROR: Too few samples in one of the classes!")
        print("  Increase the number of samples or check the model.")
        return

    # 3. Train Random Forest
    print("\n[3/5] Training Random Forest...")
    rf, cv_scores = train_random_forest(samples, y, CONFIG)

    # 4. Permutation Importance
    print("\n[4/5] Computing Permutation Importance...")
    perm, indices = compute_permutation_importance(rf, samples, y, CONFIG)

    # 5. Save results
    print("\n[5/5] Saving results...")
    plot_results(samples, y, rf, perm, indices, CONFIG, CONFIG['output_dir'])
    save_results(samples, y, rf, perm, indices, cv_scores, CONFIG, CONFIG['output_dir'])

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
