"""
Модуль визуализации для анализа кредитных данных
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

def plot_feature_distribution(X: np.ndarray, feature_idx: int, feature_name: str, stats: dict) -> None:
    """Визуализация распределения признака"""
    plt.figure(figsize=(10, 6))
    
    # Строим гистограмму
    plt.hist(X[:, feature_idx], bins=50, density=True, alpha=0.7)
    
    # Добавляем вертикальные линии для статистик
    plt.axvline(stats['mean'], color='r', linestyle='--',
                label=f'Среднее = {stats["mean"]:.2f}')
    
    # Добавляем линию для среднего ± std
    plt.axvline(stats['mean'] - stats['std'], color='g', linestyle=':',
                label=f'Среднее ± STD')
    plt.axvline(stats['mean'] + stats['std'], color='g', linestyle=':')
    
    plt.title(f'Распределение признака: {feature_name}')
    plt.xlabel('Значение')
    plt.ylabel('Плотность')
    plt.legend()
    
    # Сохраняем график
    plt.savefig(f'visualization/figures/{feature_name}_distribution.png')
    plt.close()

def plot_feature_boxplot(data: np.ndarray, column: int, feature_name: str) -> None:
    """
    Построение boxplot для признака с разделением по классам
    
    Args:
        data: Данные
        column: Номер колонки
        feature_name: Название признака
    """
    plt.figure(figsize=(8, 6))
    
    # Подготавливаем данные
    values = data[:, column]
    groups = ['Не дефолт' if x == 0 else 'Дефолт' for x in data[:, -1]]
    
    # Строим boxplot
    sns.boxplot(x=groups, y=values)
    
    plt.title(f'Boxplot признака {feature_name}')
    plt.ylabel('Значение')
    
    # Сохраняем график
    plt.savefig(f'visualization/figures/{feature_name}_boxplot.png')
    plt.close()

def plot_correlation_matrix(correlations: np.ndarray, 
                          feature_names: List[str]) -> None:
    """
    Построение матрицы корреляций
    
    Args:
        correlations: Матрица корреляций
        feature_names: Названия признаков
    """
    plt.figure(figsize=(12, 10))
    
    # Строим тепловую карту
    sns.heatmap(correlations, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=feature_names, yticklabels=feature_names)
    
    plt.title('Матрица корреляций признаков')
    
    # Сохраняем график
    plt.savefig('visualization/figures/correlation_matrix.png')
    plt.close()

def plot_feature_importance(importance_dict: dict) -> None:
    """Визуализация важности признаков"""
    plt.figure(figsize=(12, 6))
    
    # Сортируем признаки по важности
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1]['weight'],
        reverse=True
    )
    
    features = [x[0] for x in sorted_features]
    weights = [x[1]['weight'] for x in sorted_features]
    ivs = [x[1]['iv'] for x in sorted_features]
    
    x = np.arange(len(features))
    width = 0.35
    
    # Строим сгруппированные столбцы
    plt.bar(x - width/2, weights, width, label='Weight')
    plt.bar(x + width/2, ivs, width, label='IV')
    
    plt.xlabel('Признаки')
    plt.ylabel('Значение')
    plt.title('Важность признаков (Weight) и Information Value (IV)')
    plt.xticks(x, features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Добавляем значения над столбцами
    for i, v in enumerate(weights):
        plt.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(ivs):
        plt.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Сохраняем график
    plt.savefig('visualization/figures/feature_importance.png')
    plt.close()

def plot_woe_analysis(model: object, feature_names: list) -> None:
    """Визуализация WoE анализа"""
    for col, name in enumerate(feature_names):
        if col not in model.feature_bins:
            continue
            
        plt.figure(figsize=(10, 6))
        bins = model.feature_bins[col]
        
        # Извлекаем данные для графика
        x = range(len(bins))
        woe_values = [bin_info['woe'] for bin_info in bins.values()]
        p_good = [bin_info['p_good'] for bin_info in bins.values()]
        p_bad = [bin_info['p_bad'] for bin_info in bins.values()]
        
        # Строим график WoE
        plt.subplot(2, 1, 1)
        plt.plot(x, woe_values, 'b-', marker='o')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'WoE анализ для признака {name}')
        plt.ylabel('Weight of Evidence')
        
        # Строим график распределения классов
        plt.subplot(2, 1, 2)
        plt.plot(x, p_good, 'g-', marker='o', label='Good')
        plt.plot(x, p_bad, 'r-', marker='o', label='Bad')
        plt.ylabel('Probability')
        plt.xlabel('Bin')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'visualization/figures/woe_analysis_{name}.png')
        plt.close()

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> None:
    """Построение ROC-кривой"""
    from sklearn.metrics import roc_curve, auc
    
    # Вычисляем ROC-кривую
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Случайный классификатор
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Сохраняем график
    plt.savefig('visualization/figures/roc_curve.png')
    plt.close()

def plot_learning_curves(train_scores: List[float], 
                        val_scores: List[float],
                        metric_name: str) -> None:
    """
    Построение кривых обучения
    
    Args:
        train_scores: Значения метрики на обучающей выборке
        val_scores: Значения метрики на валидационной выборке
        metric_name: Название метрики
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-', label='Обучающая выборка')
    plt.plot(epochs, val_scores, 'r-', label='Валидационная выборка')
    
    plt.title(f'Кривые обучения ({metric_name})')
    plt.xlabel('Эпоха')
    plt.ylabel(metric_name)
    plt.legend()
    
    plt.grid(True)
    
    # Сохраняем график
    plt.savefig(f'visualization/figures/learning_curves_{metric_name}.png')
    plt.close()

def plot_risk_score_distribution(scores: np.ndarray) -> None:
    """
    Построение распределения риск-скоров
    
    Args:
        scores: Массив риск-скоров
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(scores, bins=50, density=True)
    plt.axvline(np.mean(scores), color='r', linestyle='--', 
                label=f'Среднее: {np.mean(scores):.3f}')
    plt.axvline(np.median(scores), color='g', linestyle='--',
                label=f'Медиана: {np.median(scores):.3f}')
    
    plt.title('Распределение риск-скоров')
    plt.xlabel('Риск-скор')
    plt.ylabel('Плотность')
    plt.legend()
    
    # Сохраняем график
    plt.savefig('visualization/figures/risk_score_distribution.png')
    plt.close() 