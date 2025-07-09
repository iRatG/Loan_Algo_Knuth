"""
Модуль для визуализации результатов анализа кредитных данных
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_feature_distribution(data: np.ndarray, column: int, feature_name: str, stats: dict):
    """Визуализация распределения признака с границами выбросов"""
    plt.figure(figsize=(12, 6))
    
    # Гистограмма распределения
    sns.histplot(data[:, column], bins=50, kde=True)
    
    # Добавляем вертикальные линии для статистик
    plt.axvline(stats['mean'], color='r', linestyle='--', label='Среднее')
    plt.axvline(stats['median'], color='g', linestyle='--', label='Медиана')
    plt.axvline(stats['lower_bound'], color='orange', linestyle=':', label='Границы выбросов')
    plt.axvline(stats['upper_bound'], color='orange', linestyle=':')
    
    plt.title(f'Распределение признака {feature_name}')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.legend()
    
    # Создаем директорию для сохранения, если её нет
    save_dir = Path('visualization/figures')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{feature_name}_distribution.png')
    plt.close()

def plot_feature_boxplot(data: np.ndarray, column: int, feature_name: str):
    """Создание box plot для визуализации выбросов"""
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(y=data[:, column])
    plt.title(f'Box Plot признака {feature_name}')
    plt.ylabel('Значение')
    
    save_dir = Path('visualization/figures')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{feature_name}_boxplot.png')
    plt.close()

def plot_risk_score_distribution(scores: list, threshold_medium: float = 0.4, threshold_high: float = 0.7):
    """Визуализация распределения риск-скоров"""
    plt.figure(figsize=(12, 6))
    
    sns.histplot(scores, bins=50, kde=True)
    plt.axvline(threshold_medium, color='yellow', linestyle='--', label='Средний риск')
    plt.axvline(threshold_high, color='red', linestyle='--', label='Высокий риск')
    
    plt.title('Распределение риск-скоров')
    plt.xlabel('Риск-скор')
    plt.ylabel('Частота')
    plt.legend()
    
    save_dir = Path('visualization/figures')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'risk_scores_distribution.png')
    plt.close()

def plot_feature_importance(features: dict, weights: dict):
    """Визуализация важности признаков"""
    plt.figure(figsize=(10, 6))
    
    features_list = list(weights.keys())
    weights_list = [weights[f] for f in features_list]
    
    sns.barplot(x=features_list, y=weights_list)
    plt.title('Веса признаков в риск-скоре')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_dir = Path('visualization/figures')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'feature_weights.png')
    plt.close() 