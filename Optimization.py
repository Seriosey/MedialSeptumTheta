import numpy as np
import optuna
from tqdm import tqdm
optuna.logging.set_verbosity(optuna.logging.WARNING)
from InhOsc import ModelParameters, ThetaRhythmNetwork, OscillationAnalyzer

tmin = 0.0
tmax = 1300
dt = 0.02
t = np.arange(tmin, tmax, dt)

def objective(trial):
    x1 = trial.suggest_float("g12", 0, 10000)
    x2 = trial.suggest_float("g21", 0, 10000)
    x3 = trial.suggest_float("I1", 0, 1000)
    x4 = trial.suggest_float("I2", 0, 1000)
    x5 = trial.suggest_float("Delta", 0, 300)
    x6 = trial.suggest_float("tau_release", 0, 10)
    x7 = trial.suggest_float("tau_recovery", 100, 1000)
    x8 = trial.suggest_float("tau_inactive", 10, 100)
    x9 = trial.suggest_float("Uinc", 0, 1)


    params = np.array([x1,x2,x3,x4,x5, x6, x7, x8, x9]) # S-E: 77.5955, 130.1333, 156.7755, 145.9170

    model_params = ModelParameters.from_array(params)
    network = ThetaRhythmNetwork(model_params)
    mf_results = network.simulate_meanfield(t_max=1600, dt=0.02) 
    transient = int(300 / 0.02)
    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=0.02, transient_samples=transient
    )     # 0 или 1 (есть/нет тета-ритм)

    return mf_analysis.is_theta


# storage = "sqlite:///final_optuna_study.db"   # создаст файл в текущей папке
storage = "sqlite:///no_u_optuna_study.db"
study = optuna.create_study(
    study_name="no_u_study",
    storage=storage,
    direction="minimize",
    load_if_exists=True     # ← важно! продолжит старое исследование
)


with tqdm(total=20000, desc="Parallel Optuna") as pbar:
    def callback(study, trial):
        pbar.update(1)

    study.optimize(objective, n_trials=20000, n_jobs=1, callbacks=[callback])


print("Лучшие параметры где было 1:")
print(study.best_params)
print(len(study.trials))


import optuna.visualization.matplotlib as vis
import plotly.graph_objects as go   # важно для кастомизации
from optuna.distributions import UniformDistribution



trials = study.trials
# recent_trials = trials[-N:]
# random_trials = random.sample(trials, N)


new_study = optuna.create_study(direction=study.direction)

# Adding extra-parameters p_ratio, I_ratio, Total_ratio
for trial in study.trials: 
    if trial.state != optuna.trial.TrialState.COMPLETE:
        continue
        
    new_params = trial.params.copy()
    new_distributions = trial.distributions.copy()
    new_params["g_ratio"] = new_params['p12']/new_params['p21']
    new_distributions["g_ratio"] = UniformDistribution(low=0.0, high=200000000.0)
    new_params["I_ratio"] = new_params['I2']/new_params['I1']
    new_distributions["I_ratio"] = UniformDistribution(low=0.0, high=200000000.0)
    new_params["total_ratio"] = new_params["g_ratio"]*new_params["I_ratio"]
    new_distributions["total_ratio"] = UniformDistribution(low=0.0, high=200000000.0)
    
    new_trial = optuna.trial.create_trial(
        state=trial.state,
        value=trial.value,
        params=new_params,
        distributions=new_distributions,  # важно!
        user_attrs=trial.user_attrs,
        system_attrs=trial.system_attrs
    )
    new_study.add_trial(new_trial)

print(len(new_study.trials))
# good_trials = [t for t in study.trials if t.value is not None and t.value < 10]
not_bad_trials = [t for t in new_study.trials if t.value is not None and t.value < 499] # Only trials with theta and good amplitudes

temp_study = optuna.create_study(direction=new_study.direction)
for t in not_bad_trials:
    temp_study.add_trial(t)

print(len(not_bad_trials))


best_params_np = np.array([study.best_params[name] for name in study.best_params.keys()])
print(best_params_np)


import optuna.visualization.matplotlib as vis
import matplotlib.pyplot as plt

# 2. Важность каждого параметра (какой параметр больше всего влияет)
fig = vis.plot_param_importances(new_study) # temp

# 3.Correlation with single parameters
optuna.visualization.plot_slice(temp_study, params=["g_ratio", "I_ratio", "total_ratio"]).show()

# Pair-wise Scatter 
fig = vis.plot_contour(temp_study, params=["p12", "p21"])#.show() 

fig.set_xlabel('g12')
fig.set_ylabel('g21')
plt.tight_layout()
plt.show()


import optuna.visualization as vis

fig = vis.plot_parallel_coordinate(temp_study)
fig.data[0].dimensions[0].label = 'Phase difference'

# Устанавливаем свои границы для каждой оси
param_ranges = {'Phase difference':(0,80),
                'p12':(0,10000),
                'p21':(0,10000),
                'I1':(0,1000),
                'I2':(0,1000),
                'Delta':(0,300),
                'tau_release':(0,10),
                'tau_recovery':(0,1000),
                'tau_inactive':(0,100),
                'Uinc':(0,1),
                'g_ratio':(0,2),
                'I_ratio':(0,3),
                'total_ratio':(0,2)}

   
for i, dim in enumerate(fig.data[0].dimensions):
    param_name = dim.label# if not param_rename else param_rename.get(dim.label, dim.label)
    if param_name in param_ranges:
        min_val, max_val = param_ranges[param_name]
        fig.data[0].dimensions[i].range = [min_val, max_val]

for dim in fig.data[0].dimensions:
    dim.tickvals = []
    dim.ticktext = []

fig.show()