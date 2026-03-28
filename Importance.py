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



#!/usr/bin/env python3
"""
================================================================================
АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ДЛЯ СИСТЕМЫ IZHIVEKEVICH + TSODYKS-MARKRAM
================================================================================

Метод: LHS + Random Forest + Permutation Importance
Вход: 9 параметров (p12, p21, Iext1, Iext2, Delta, tau_d, tau_r, tau_f, Uinc)
Выход: бинарный (0 = нет тета-ритма, 1 = есть тета-ритм)

Рекомендуемое время выполнения: 2-4 часа
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

CONFIG = {
    # Параметры модели
    'param_names': ['p12', 'p21', 'Iext1', 'Iext2', 'Delta', 'tau_d', 'tau_r', 'tau_f', 'Uinc'],
    
    # Границы параметров (настройте под вашу модель!)
    'param_bounds': {
        'p12':   (0, 10000),    # Вес связи популяция 1 -> 2
        'p21':   (0, 10000),    # Вес связи популяция 2 -> 1
        'Iext1': (0, 2000),     # Время депрессии синапса 1 (мс)
        'Iext2': (0, 2000),     # Время депрессии синапса 2 (мс)
        'Delta': (1, 300)
        'tau_d': (0, 10),   # Время восстановления синапса 1 (мс)
        'tau_r': (100, 1000),   # Время восстановления синапса 2 (мс)
        'tau_f': (10, 100),     # Время фасилитации синапса 1 (мс)
        'Uinc': (0, 1),     # Время фасилитации синапса 2 (мс)
    },

    # Параметры сэмплирования
    'n_samples': 5000,          # Количество сэмплов (5000-10000 рекомендуется)
    'random_seed': 42,
    
    # Параметры Random Forest
    'rf_n_estimators': 200,
    'rf_max_depth': 15,
    'rf_min_samples_leaf': 5,
    
    # Параметры Permutation Importance
    'n_repeats': 20,            # Количество перестановок
    
    # Выходные файлы
    'output_dir': 'sensitivity_results',
}


# ============================================================================
# LHS САМПЛИРОВАНИЕ
# ============================================================================

def latin_hypercube_sampling(n_samples, param_bounds, seed=42):
    """
    Latin Hypercube Sampling для генерации параметров.
    
    Преимущества LHS:
    - Равномерное покрытие пространства параметров
    - Эффективнее простого случайного сэмплирования
    - Количество сэмплов не зависит от размерности
    """
    np.random.seed(seed)
    n_params = len(param_bounds)
    param_names = list(param_bounds.keys())
    
    # Создаем LHS матрицу [0, 1]
    lhs_matrix = np.zeros((n_samples, n_params))
    
    for i in range(n_params):
        # Разбиение на n_samples интервалов
        perm = np.random.permutation(n_samples)
        # Случайная точка внутри каждого интервала
        lhs_matrix[:, i] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples
    
    # Масштабирование к реальным границам
    samples = np.zeros((n_samples, n_params))
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        samples[:, i] = low + lhs_matrix[:, i] * (high - low)
    
    return pd.DataFrame(samples, columns=param_names)


# ============================================================================
# МОДЕЛЬ IZHIVEKEVICH + TSODYKS-MARKRAM (ЗАГЛУШКА - ЗАМЕНИТЕ НА ВАШУ!)
# ============================================================================

def simulate_theta_rhythm(params):
    model_params = ModelParameters.from_array(params)
    network = ThetaRhythmNetwork(model_params)
    mf_results = network.simulate_meanfield(t_max=1600, dt=0.02) 
    transient = int(300 / 0.02)
    mf_analysis = OscillationAnalyzer.analyze_oscillation(
        mf_results['r1'], mf_results['r2'], dt=0.02, transient_samples=transient
    )     # 0 или 1 (есть/нет тета-ритм)

    return mf_analysis.is_theta



def run_simulations(samples, verbose=True):
    """
    Запуск симуляций для всех сэмплов.
    """
    n_samples = len(samples)
    results = np.zeros(n_samples)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ЗАПУСК СИМУЛЯЦИЙ: {n_samples} прогонов")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        params = row.to_dict()
        results[i] = simulate_theta_rhythm(params)
        
        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"  Прогресс: {i+1}/{n_samples} ({100*(i+1)/n_samples:.1f}%) | "
                  f"Скорость: {rate:.1f} сим/сек | ETA: {eta/60:.1f} мин")
    
    elapsed = time.time() - start_time
    
    if verbose:
        n_positive = results.sum()
        print(f"\n  ЗАВЕРШЕНО за {elapsed:.1f} сек ({elapsed/60:.2f} мин)")
        print(f"  Тета-ритм найден: {int(n_positive)} из {n_samples} ({100*n_positive/n_samples:.1f}%)")
    
    return results


# ============================================================================
# RANDOM FOREST + PERMUTATION IMPORTANCE
# ============================================================================

def train_random_forest(X, y, config):
    """
    Обучение Random Forest для классификации.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ RANDOM FOREST")
    print(f"{'='*60}")
    
    # Проверка баланса классов
    n_positive = y.sum()
    n_negative = len(y) - n_positive
    print(f"  Класс 0: {n_negative} ({100*n_negative/len(y):.1f}%)")
    print(f"  Класс 1: {int(n_positive)} ({100*n_positive/len(y):.1f}%)")
    
    if n_positive < 10 or n_negative < 10:
        print("  ВНИМАНИЕ: Слишком мало сэмплов одного из классов!")
        print("  Результаты могут быть ненадежными.")
    
    # Создание и обучение модели
    rf = RandomForestClassifier(
        n_estimators=config['rf_n_estimators'],
        max_depth=config['rf_max_depth'],
        min_samples_leaf=config['rf_min_samples_leaf'],
        max_features='sqrt',
        n_jobs=-1,
        random_state=config['random_seed'],
        oob_score=True,
        class_weight='balanced'  # Важно для несбалансированных данных!
    )
    
    rf.fit(X, y)
    
    # Кросс-валидация
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    
    print(f"\n  OOB Score: {rf.oob_score_:.4f}")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return rf, cv_scores


def compute_permutation_importance(rf, X, y, config):
    """
    Вычисление Permutation Importance с ROC-AUC scoring.
    """
    from sklearn.inspection import permutation_importance
    
    print(f"\n{'='*60}")
    print("ВЫЧИСЛЕНИЕ PERMUTATION IMPORTANCE")
    print(f"{'='*60}")
    print(f"  Количество повторений: {config['n_repeats']}")
    print(f"  Метрика: ROC-AUC")
    
    perm = permutation_importance(
        rf, X, y,
        n_repeats=config['n_repeats'],
        scoring='roc_auc',
        n_jobs=-1,
        random_state=config['random_seed']
    )
    
    # Сортировка по важности
    indices = np.argsort(perm.importances_mean)[::-1]
    
    print(f"\n  Результаты (отсортированы по важности):")
    print(f"  {'Параметр':<12} {'Важность':>12} {'± Std':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    
    for idx in indices:
        name = config['param_names'][idx]
        imp = perm.importances_mean[idx]
        std = perm.importances_std[idx]
        print(f"  {name:<12} {imp:>12.4f} {std:>12.4f}")
    
    return perm, indices


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_results(samples, y, rf, perm, indices, config, output_dir):
    """
    Создание визуализаций результатов.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    param_names = config['param_names']
    
    # 1. Permutation Importance Bar Plot
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
    ax.set_title('Важность параметров для наличия тета-ритма', fontsize=14)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()
    
    # Добавляем значения на бары
    for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'permutation_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Box Plot для распределения важности
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_importances_all = perm.importances[indices].T
    
    bp = ax.boxplot(sorted_importances_all, vert=False, 
                    labels=sorted_names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Permutation Importance (ROC-AUC)', fontsize=12)
    ax.set_title('Распределение важности параметров (20 перестановок)', fontsize=14)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'permutation_importance_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 2D проекции для топ-4 параметров
    top4_indices = indices[:4]
    top4_names = [param_names[i] for i in top4_indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax_idx, (param_idx, param_name) in enumerate(zip(top4_indices, top4_names)):
        ax = axes[ax_idx]
        
        values = samples[param_name].values
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        # Разделение на классы
        class0 = values[y_arr == 0]
        class1 = values[y_arr == 1]
        
        ax.hist(class0, bins=30, alpha=0.5, label='Нет тета-ритма', color='blue')
        ax.hist(class1, bins=30, alpha=0.5, label='Есть тета-ритм', color='red')
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel('Частота', fontsize=11)
        ax.set_title(f'Топ-{ax_idx+1}: {param_name}', fontsize=12)
        ax.legend()
    
    plt.suptitle('Распределение топ-4 параметров по классам', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation Matrix
    corr = samples.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Добавляем значения
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha='center', va='center', fontsize=8,
                          color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Корреляция', fontsize=11)
    ax.set_title('Корреляционная матрица параметров', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Графики сохранены в: {output_dir}")


# ============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def save_results(samples, y, rf, perm, indices, cv_scores, config, output_dir):
    """
    Сохранение результатов в файлы.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    param_names = config['param_names']
    
    # 1. CSV с результатами
    df = samples.copy()
    df['theta_rhythm'] = y
    df.to_csv(output_dir / 'simulation_results.csv', index=False)
    
    # 2. Важность параметров
    importance_df = pd.DataFrame({
        'parameter': param_names,
        'importance_mean': perm.importances_mean,
        'importance_std': perm.importances_std,
        'rank': [list(indices).index(i) + 1 for i in range(len(param_names))]
    }).sort_values('rank')
    
    importance_df.to_csv(output_dir / 'parameter_importance.csv', index=False)
    
    # 3. Текстовый отчет
    report = []
    report.append("=" * 70)
    report.append("ОТЧЕТ ПО АНАЛИЗУ ЧУВСТВИТЕЛЬНОСТИ")
    report.append("=" * 70)
    report.append("")
    report.append("КОНФИГУРАЦИЯ:")
    report.append(f"  Количество сэмплов: {config['n_samples']}")
    report.append(f"  Метод сэмплирования: Latin Hypercube Sampling")
    report.append(f"  Random Forest: {config['rf_n_estimators']} деревьев")
    report.append(f"  Permutation repeats: {config['n_repeats']}")
    report.append("")
    report.append("РЕЗУЛЬТАТЫ МОДЕЛИ:")
    report.append(f"  OOB Score: {rf.oob_score_:.4f}")
    report.append(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    report.append(f"  Тета-ритм найден: {int(y.sum())} из {len(y)} ({100*y.sum()/len(y):.1f}%)")
    report.append("")
    report.append("РАНЖИРОВАНИЕ ПАРАМЕТРОВ (по важности):")
    report.append("-" * 50)
    report.append(f"{'Ранг':<6} {'Параметр':<12} {'Важность':>12} {'± Std':>12}")
    report.append("-" * 50)
    
    for rank, idx in enumerate(indices, 1):
        name = param_names[idx]
        imp = perm.importances_mean[idx]
        std = perm.importances_std[idx]
        report.append(f"{rank:<6} {name:<12} {imp:>12.4f} {std:>12.4f}")
    
    report.append("")
    report.append("ИНТЕРПРЕТАЦИЯ:")
    report.append("-" * 50)
    
    # Определение важных параметров
    mean_importance = perm.importances_mean.mean()
    top_params = [param_names[i] for i in indices[:3]]
    
    report.append(f"  Средняя важность: {mean_importance:.4f}")
    report.append(f"  Топ-3 параметра: {', '.join(top_params)}")
    
    # Проверка значимости
    significant = [param_names[i] for i in indices 
                   if perm.importances_mean[i] > 2 * perm.importances_std[i]]
    
    if significant:
        report.append(f"  Значимые параметры (importance > 2*std): {', '.join(significant)}")
    else:
        report.append("  Нет статистически значимых параметров")
    
    report.append("")
    report.append("ФАЙЛЫ:")
    report.append(f"  - simulation_results.csv: результаты всех симуляций")
    report.append(f"  - parameter_importance.csv: важность параметров")
    report.append(f"  - permutation_importance.png: bar chart важности")
    report.append(f"  - permutation_importance_boxplot.png: box plot")
    report.append(f"  - parameter_distributions.png: распределения топ-4")
    report.append(f"  - correlation_matrix.png: корреляции параметров")
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    with open(output_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nФайлы сохранены в: {output_dir}")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Главная функция запуска анализа.
    """
    print("\n" + "=" * 70)
    print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ: IZHIVEKEVICH + TSODYKS-MARKRAM")
    print("=" * 70)
    
    # 1. Генерация сэмплов
    print("\n[1/5] Генерация сэмплов методом LHS...")
    samples = latin_hypercube_sampling(
        CONFIG['n_samples'],
        CONFIG['param_bounds'],
        CONFIG['random_seed']
    )
    print(f"  Сгенерировано {len(samples)} сэмплов")
    print(f"  Параметры: {list(CONFIG['param_names'])}")
    
    # 2. Запуск симуляций
    print("\n[2/5] Запуск симуляций...")
    print("  ВНИМАНИЕ: Используется ЗАГЛУШКА! Замените simulate_theta_rhythm() на вашу модель!")
    y = run_simulations(samples)
    
    # Проверка достаточности данных
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        print("\n  ОШИБКА: Недостаточно примеров одного из классов!")
        print("  Увеличьте количество сэмплов или проверьте модель.")
        return
    
    # 3. Обучение Random Forest
    print("\n[3/5] Обучение Random Forest...")
    rf, cv_scores = train_random_forest(samples, y, CONFIG)
    
    # 4. Permutation Importance
    print("\n[4/5] Вычисление Permutation Importance...")
    perm, indices = compute_permutation_importance(rf, samples, y, CONFIG)
    
    # 5. Сохранение результатов
    print("\n[5/5] Сохранение результатов...")
    plot_results(samples, y, rf, perm, indices, CONFIG, CONFIG['output_dir'])
    save_results(samples, y, rf, perm, indices, cv_scores, CONFIG, CONFIG['output_dir'])
    
    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()

# Результаты
print("S₁  (главный эффект):", Si['S1'])
print("Sᵀᵢ (полный эффект):", Si['ST'])
print("S₂  (взаимодействия):", Si['S2'])  # матрица k×k
