"""
Тесты для реализации градиентного бустинга на основе алгоритмов Кнута
"""

import numpy as np
import pytest
from algorithms.knuth_boosting import KnuthGradientBoosting, KnuthDecisionTree

def generate_regression_data(n_samples=1000, n_features=10, noise=0.1):
    """
    Генерация тестовых данных для регрессии
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = np.dot(X, true_coefficients)
    y += noise * np.random.randn(n_samples)
    return X, y

def test_knuth_decision_tree():
    """
    Тест базового дерева решений
    """
    X, y = generate_regression_data(n_samples=100, n_features=3)
    
    tree = KnuthDecisionTree(max_depth=3)
    tree.fit(X, y)
    
    predictions = tree.predict(X)
    
    # Проверяем, что предсказания имеют правильную форму
    assert predictions.shape == y.shape
    
    # Проверяем, что MSE не слишком большое
    mse = np.mean((predictions - y) ** 2)
    assert mse < 10.0

def test_knuth_gradient_boosting():
    """
    Тест градиентного бустинга
    """
    X, y = generate_regression_data()
    
    # Разделяем данные на обучающую и тестовую выборки
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Обучаем модель
    model = KnuthGradientBoosting(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3
    )
    model.fit(X_train, y_train)
    
    # Получаем предсказания
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Проверяем размерности
    assert train_predictions.shape == y_train.shape
    assert test_predictions.shape == y_test.shape
    
    # Вычисляем метрики
    train_mse = np.mean((train_predictions - y_train) ** 2)
    test_mse = np.mean((test_predictions - y_test) ** 2)
    
    # Проверяем, что ошибки разумные
    assert train_mse < test_mse  # Не должно быть переобучения
    assert test_mse < 10.0  # Модель должна что-то выучить

def test_knuth_gradient_boosting_edge_cases():
    """
    Тест граничных случаев
    """
    # Тест на маленьком наборе данных
    X_small, y_small = generate_regression_data(n_samples=10, n_features=2)
    model_small = KnuthGradientBoosting(n_estimators=5)
    model_small.fit(X_small, y_small)
    predictions_small = model_small.predict(X_small)
    assert predictions_small.shape == y_small.shape
    
    # Тест на данных с одним признаком
    X_single, y_single = generate_regression_data(n_samples=100, n_features=1)
    model_single = KnuthGradientBoosting()
    model_single.fit(X_single, y_single)
    predictions_single = model_single.predict(X_single)
    assert predictions_single.shape == y_single.shape

def test_knuth_gradient_boosting_parameters():
    """
    Тест влияния параметров на производительность
    """
    X, y = generate_regression_data(n_samples=200)
    
    # Тест с разными learning_rate
    model_fast = KnuthGradientBoosting(learning_rate=0.5, n_estimators=20)
    model_slow = KnuthGradientBoosting(learning_rate=0.01, n_estimators=20)
    
    model_fast.fit(X, y)
    model_slow.fit(X, y)
    
    pred_fast = model_fast.predict(X)
    pred_slow = model_slow.predict(X)
    
    mse_fast = np.mean((pred_fast - y) ** 2)
    mse_slow = np.mean((pred_slow - y) ** 2)
    
    # Быстрое обучение должно давать более высокую ошибку
    assert mse_fast > mse_slow
    
    # Тест с разной глубиной деревьев
    model_shallow = KnuthGradientBoosting(max_depth=2, n_estimators=20)
    model_deep = KnuthGradientBoosting(max_depth=5, n_estimators=20)
    
    model_shallow.fit(X, y)
    model_deep.fit(X, y)
    
    pred_shallow = model_shallow.predict(X)
    pred_deep = model_deep.predict(X)
    
    mse_shallow = np.mean((pred_shallow - y) ** 2)
    mse_deep = np.mean((pred_deep - y) ** 2)
    
    # Более глубокие деревья должны давать меньшую ошибку на обучающей выборке
    assert mse_shallow > mse_deep 