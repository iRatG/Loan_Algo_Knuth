"""
Скрипт для оценки работы градиентного бустинга на основе алгоритмов Кнута
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import gc
from collections import defaultdict

from algorithms.knuth_boosting import (
    KnuthGradientBoosting,
    KnuthPreprocessor,
    KnuthMetrics,
    KnuthOptimizer
)

# Настройка случайного состояния для воспроизводимости
np.random.seed(42)

def load_and_preprocess_data(sample_size=30000, batch_size=2000):
    """
    Загружаем и предобрабатываем данные из датасета кредитного скоринга
    """
    print("\nЗагрузка и предобработка данных кредитного скоринга...")
    start_time = time.time()
    
    data = pd.read_csv('source/UCI_Credit_Card.csv')
    print(f"Всего в датасете: {len(data)} записей с {len(data.columns)} признаками")
    
    # Удаляем ID из признаков
    if 'ID' in data.columns:
        data = data.drop('ID', axis=1)
    
    print("\nПредобработка данных:")
    print("1. Обработка пропущенных значений")
    print("2. Кодирование категориальных признаков")
    print("3. Нормализация числовых признаков")
    
    # Используем наш препроцессор
    preprocessor = KnuthPreprocessor(
        handle_missing=True,
        handle_categorical=True,
        normalize=True
    )
    
    X = data.drop(['default.payment.next.month'], axis=1).values
    y = data['default.payment.next.month'].values
    
    # Предобработка данных с выводом прогресса
    X_processed = preprocessor.fit_transform(X)
    
    elapsed_time = time.time() - start_time
    print(f"\nПредобработка завершена за {elapsed_time:.2f} секунд")
    
    # Разделяем на батчи для обучения
    n_batches = len(X_processed) // batch_size
    print(f"Данные разделены на {n_batches} батчей по {batch_size} записей")
    
    return X_processed, y, preprocessor, batch_size

def optimize_hyperparameters(X_train, y_train, cv_folds=5):
    """
    Оптимизация гиперпараметров с помощью кросс-валидации
    """
    print("\nОптимизация гиперпараметров...")
    start_time = time.time()
    
    param_grid = {
        'n_estimators': [20, 30],
        'learning_rate': [0.3],
        'max_depth': [6],
        'min_samples_split': [2, 4],
        'subsample': [0.8, 0.9],
        'batch_size': [2000],
        'n_iterations': [2],
        'early_stopping_rounds': [5],
        'class_weight': ['balanced']
    }
    
    print("\nПеребор параметров:")
    for param, values in param_grid.items():
        print(f"{param}: {values}")
    
    optimizer = KnuthOptimizer(
        param_grid=param_grid,
        n_folds=cv_folds,
        scoring='auc_roc',
        random_state=42
    )
    
    print("\nПоиск оптимальных параметров...")
    optimizer.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"\nОптимизация завершена за {elapsed_time:.2f} секунд")
    print("\nЛучшие параметры:", optimizer.best_params_)
    print("Лучший score:", optimizer.best_score_)
    
    return optimizer.get_best_model()

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, batch_size):
    """
    Полная оценка модели на всех выборках
    """
    print("\nОценка модели на всех выборках...")
    start_time = time.time()
    
    # Анализ баланса классов
    preprocessor = KnuthPreprocessor()
    train_balance = preprocessor.analyze_class_balance(y_train)
    print("\nБаланс классов в обучающей выборке:")
    print(f"Распределение: {train_balance['ratios']}")
    print(f"Коэффициент дисбаланса: {train_balance['imbalance_ratio']:.2f}")
    
    # Обучение модели с валидацией
    print("\nОбучение модели...")
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    datasets = {
        'train': (X_train, y_train),
        'validation': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    results = {}
    for name, (X, y) in datasets.items():
        print(f"\nОценка на {name} выборке:")
        metrics = model.evaluate(X, y)
        results[name] = metrics
        
        print(f"\nМетрики {name}:")
        for metric, value in metrics.items():
            if metric != 'feature_importance':
                print(f"{metric}: {value:.4f}")
    
    # Анализ важности признаков
    print("\nВажность признаков:")
    feature_importance = results['train']['feature_importance']
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:10]:
        print(f"Признак {feature}: {importance:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nОценка завершена за {elapsed_time:.2f} секунд")
    
    return results

def plot_metrics_history(results, save_path='visualization/figures/metrics_history.png'):
    """
    Визуализация метрик по выборкам
    """
    print("\nСоздание визуализации метрик...")
    
    plt.figure(figsize=(12, 8))
    
    metrics = ['accuracy', 'auc_roc', 'mse']
    datasets = ['train', 'validation', 'test']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        values = [results[ds][metric] for ds in datasets]
        plt.bar(datasets, values)
        plt.title(f'{metric.upper()} по выборкам')
        plt.ylabel('Значение')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Визуализация сохранена в {save_path}")

def main():
    print("=" * 80)
    print("ОЦЕНКА ГРАДИЕНТНОГО БУСТИНГА НА ОСНОВЕ АЛГОРИТМОВ КНУТА")
    print("=" * 80)
    
    # Загружаем и предобрабатываем данные
    X, y, preprocessor, batch_size = load_and_preprocess_data()
    
    # Разделяем данные на выборки
    print("\nРазделение данных на выборки...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    print(f"Размеры выборок:")
    print(f"Обучающая: {len(X_train)} записей")
    print(f"Валидационная: {len(X_val)} записей")
    print(f"Тестовая: {len(X_test)} записей")
    
    # Оптимизируем гиперпараметры
    best_model = optimize_hyperparameters(X_train, y_train)
    
    # Оцениваем модель
    results = evaluate_model(
        best_model,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        batch_size
    )
    
    # Визуализируем результаты
    plot_metrics_history(results)
    
    print("\nГотово! Результаты сохранены в директории visualization/figures/")
    print("=" * 80)

if __name__ == '__main__':
    main() 