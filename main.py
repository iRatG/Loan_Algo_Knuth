"""
Основной скрипт для запуска кредитного скоринга
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from algorithms.knuth_sort import KnuthSort
from algorithms.knuth_statistics import KnuthStatistics
from algorithms.knuth_search import KnuthSearch
from algorithms.knuth_metrics import KnuthMetrics
from models.credit_model import KnuthCreditModel
from visualization.plots import (
    plot_feature_distribution,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_risk_score_distribution,
    plot_woe_analysis
)

def load_data(path: str) -> pd.DataFrame:
    """Загрузка данных"""
    print("Загрузка данных...")
    df = pd.read_csv(path)
    print(f"Загружено {len(df)} записей с {len(df.columns)} признаками")
    return df

def preprocess_data(df: pd.DataFrame) -> tuple:
    """Предобработка данных"""
    print("\nПредобработка данных...")
    
    # Удаляем ID
    df = df.drop('ID', axis=1)
    
    # Разделяем признаки на категориальные и числовые
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    numeric_features = [col for col in df.columns if col not in categorical_features + ['default.payment.next.month']]
    
    # Нормализуем числовые признаки
    for col in numeric_features:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std
            
    # Преобразуем категориальные признаки
    for col in categorical_features:
        # Считаем средний target rate для каждой категории
        target_means = df.groupby(col)['default.payment.next.month'].mean()
        df[col] = df[col].map(target_means)
    
    # Разделяем признаки и целевую переменную
    X = df.drop('default.payment.next.month', axis=1).values
    y = df['default.payment.next.month'].values
    
    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def analyze_features(X: np.ndarray, feature_names: list) -> None:
    """Анализ признаков"""
    print("\nАнализ признаков...")
    stats = KnuthStatistics()
    
    # Создаем директорию для графиков
    Path('visualization/figures').mkdir(parents=True, exist_ok=True)
    
    # Анализируем каждый признак
    for i, name in enumerate(feature_names):
        if i >= X.shape[1]:
            break
            
        stats.update(X, [i])
        feature_stats = stats.get_stats(i)
        
        print(f"\nПризнак: {name}")
        print(f"Среднее: {feature_stats['mean']:.2f}")
        print(f"Стандартное отклонение: {feature_stats['std']:.2f}")
        print(f"Минимум: {feature_stats['min']:.2f}")
        print(f"Максимум: {feature_stats['max']:.2f}")
        
        # Строим распределение
        plot_feature_distribution(X, i, name, feature_stats)

def analyze_correlations(X: np.ndarray, feature_names: list) -> None:
    """Анализ корреляций"""
    print("\nАнализ корреляций...")
    stats = KnuthStatistics()
    n_features = X.shape[1]
    
    # Создаем матрицу корреляций
    correlations = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i, n_features):
            corr = stats.correlation_knuth(X, i, j)
            correlations[i, j] = corr
            correlations[j, i] = corr
    
    # Визуализируем корреляции
    plot_correlation_matrix(correlations, feature_names)
    print("Матрица корреляций сохранена в visualization/figures/correlation_matrix.png")

def train_and_evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list
) -> None:
    """Обучение и оценка модели"""
    print("\nОбучение модели...")
    model = KnuthCreditModel(n_bins=10)
    model.fit(X_train, y_train)
    
    # Получаем предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Оцениваем качество
    metrics = model.evaluate(X_test, y_test)
    
    print("\nМетрики качества:")
    print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"Коэффициент Джини: {metrics['gini']:.3f}")
    print(f"Статистика KS: {metrics['ks_stat']:.3f}")
    print(f"Precision: {metrics['precision_recall']['precision']:.3f}")
    print(f"Recall: {metrics['precision_recall']['recall']:.3f}")
    print(f"F1-score: {metrics['precision_recall']['f1']:.3f}")
    
    # Визуализируем результаты
    plot_roc_curve(y_test, y_proba)
    print("ROC-кривая сохранена в visualization/figures/roc_curve.png")
    
    # Анализируем важность признаков
    importance = model.get_feature_importance()
    feature_importance = {
        feature_names[i]: imp
        for i, imp in importance.items()
    }
    plot_feature_importance(feature_importance)
    print("График важности признаков сохранен в visualization/figures/feature_importance.png")
    
    # Анализируем WoE для каждого признака
    plot_woe_analysis(model, feature_names)
    print("WoE анализ сохранен в visualization/figures/woe_analysis_*.png")
    
    # Анализируем распределение риск-скоров
    plot_risk_score_distribution(y_proba)
    print("Распределение риск-скоров сохранено в visualization/figures/risk_score_distribution.png")

def main():
    """Основная функция"""
    # Загружаем данные
    df = load_data('source/UCI_Credit_Card.csv')
    
    # Получаем названия признаков до предобработки
    all_feature_names = df.drop('default.payment.next.month', axis=1).columns.tolist()
    
    # Предобрабатываем данные
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Получаем названия признаков после предобработки (без ID)
    feature_names = [name for name in all_feature_names if name != 'ID']
    
    # Анализируем признаки
    analyze_features(X_train, feature_names)
    
    # Анализируем корреляции
    analyze_correlations(X_train, feature_names)
    
    # Обучаем и оцениваем модель
    train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names)

if __name__ == '__main__':
    main() 