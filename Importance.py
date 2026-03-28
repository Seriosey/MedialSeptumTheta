import numpy as np
from InhOsc import (
    ModelParameters, ThetaRhythmNetwork, OscillationAnalyzer,
    get_default_optimal_parameters, NeuronParameters
)




def black_box(params):
    model_params = ModelParameters.from_array(params)
    network = ThetaRhythmNetwork(model_params)
    mf_results = network.simulate_meanfield(t_max=1600, dt=0.02) 
    transient = int(300 / 0.02)
    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=0.02, transient_samples=transient
    )     # 0 или 1 (есть/нет тета-ритм)

    return mf_analysis.is_theta



from SALib.analyze import morris
from SALib.sample import morris as morris_sample
import numpy as np

problem = {
    'num_vars': 9,
    'names': ['p12', 'p21', 'Iext1', 'Iext2', 'Delta', 'tau_d', 'tau_r', 'tau_f', 'Uinc'],
    'bounds': [[0, 10000], [0, 10000], [0, 2000], [0, 2000], [1, 300], [0, 10], [100, 1000], [10, 100], [0, 1]]
}

# Генерируем выборку
X = morris_sample.sample(problem, N=50, num_levels=4)

# Запускаем симуляции (ЗАМЕНИТЕ НА ВАШУ!)
Y = np.array([black_box(x) for x in X]).astype(float)

# Анализ
Si = morris.analyze(problem, X, Y)

# Результаты
print("μ* (важность):", Si['mu_star'])
print("σ  (нелинейность):", Si['sigma'])



from scipy.stats import qmc
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d=9, seed=42)
X = sampler.random(n=2000)
X = qmc.scale(X, [b[0] for b in problem['bounds']], [b[1] for b in problem['bounds']])

# Симуляции
Y = np.array([black_box(x) for x in X])

# Random Forest
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(X, Y)

# Permutation Importance
perm = permutation_importance(rf, X, Y, n_repeats=10)

print("Важность:", perm.importances_mean)
print("Стд. отклонение:", perm.importances_std)



from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

# Генерируем выборку (N должно быть степенью 2!)
X = sobol_sample.sample(problem, N=1024, calc_second_order=True)

# Симуляции
Y = np.array([black_box(x) for x in X]).astype(float)

# Анализ
Si = sobol.analyze(problem, Y, calc_second_order=True)

# Результаты
print("S₁  (главный эффект):", Si['S1'])
print("Sᵀᵢ (полный эффект):", Si['ST'])
print("S₂  (взаимодействия):", Si['S2'])  # матрица k×k