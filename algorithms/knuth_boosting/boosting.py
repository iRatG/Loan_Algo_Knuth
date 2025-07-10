"""
Реализация градиентного бустинга на основе алгоритмов Кнута.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import time
from .trees import KnuthDecisionTree
from .preprocessing import KnuthPreprocessor
from .metrics import KnuthMetrics

class KnuthGradientBoosting:
    """
    Реализация градиентного бустинга на основе алгоритмов Кнута.
    """
    
    def __init__(
        self,
        n_estimators=30,
        learning_rate=0.3,
        max_depth=6,
        min_samples_split=2,
        subsample=0.8,
        batch_size=2000,
        n_iterations=2,
        early_stopping_rounds=5,
        class_weight='balanced',
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.trees = []
        self.feature_importances_ = None
        self.best_iteration_ = None
        self.preprocessor = KnuthPreprocessor()
        
    def _compute_gradients(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Вычисление градиентов"""
        return y_true - y_pred
        
    def _get_random_batch(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Получение случайного батча данных"""
        n_samples = len(X)
        indices = self.rng.choice(n_samples, size=min(self.batch_size, n_samples), replace=False)
        return X[indices], y[indices]
        
    def _subsample_indices(self, X: np.ndarray) -> np.ndarray:
        """Выбор подвыборки для обучения"""
        n_samples = X.shape[0]
        subsample_size = int(n_samples * self.subsample)
        return self.rng.choice(n_samples, size=subsample_size, replace=False)
        
    def _calculate_weights(self, y):
        """Расчет весов классов"""
        if self.class_weight == 'balanced':
            unique, counts = np.unique(y, return_counts=True)
            weights = {cls: len(y)/(len(unique) * count) 
                      for cls, count in zip(unique, counts)}
            return np.array([weights[cls] for cls in y])
        return np.ones(len(y))
        
    def _calculate_feature_importances(self):
        """Расчет важности признаков"""
        importances = np.zeros(self.n_features_)
        for tree in self.trees:
            importances += tree.feature_importances_
        self.feature_importances_ = importances / len(self.trees)
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Обучение модели с ранней остановкой"""
        self.n_features_ = X.shape[1]
        
        # Анализируем баланс классов
        balance_info = self.preprocessor.analyze_class_balance(y)
        print("\nАнализ баланса классов:")
        print(f"Распределение классов: {balance_info['ratios']}")
        print(f"Коэффициент дисбаланса: {balance_info['imbalance_ratio']:.2f}")
        
        # Рассчитываем веса
        sample_weights = self._calculate_weights(y)
        
        best_val_score = float('-inf')
        rounds_without_improve = 0
        
        for iteration in range(self.n_iterations):
            print(f"\nИтерация {iteration + 1}/{self.n_iterations}")
            
            # Получаем сбалансированные батчи
            batches = self.preprocessor.get_stratified_batches(X, y, self.batch_size)
            
            for batch_idx, (X_batch, y_batch) in enumerate(batches):
                # Обучаем деревья на текущем батче
                tree = KnuthDecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )
                
                # Используем веса при обучении
                batch_weights = sample_weights[batch_idx:batch_idx + self.batch_size]
                tree.fit(X_batch, y_batch, sample_weight=batch_weights)
                self.trees.append(tree)
                
                # Проверяем на валидационной выборке
                if X_val is not None and y_val is not None:
                    val_score = self.evaluate(X_val, y_val)['auc_roc']
                    print(f"Валидационный AUC-ROC: {val_score:.4f}")
                    
                    if val_score > best_val_score:
                        best_val_score = val_score
                        self.best_iteration_ = len(self.trees)
                        rounds_without_improve = 0
                    else:
                        rounds_without_improve += 1
                        
                    if rounds_without_improve >= self.early_stopping_rounds:
                        print("\nРанняя остановка!")
                        self.trees = self.trees[:self.best_iteration_]
                        break
                        
            # Обновляем важность признаков
            self._calculate_feature_importances()
            
        return self
        
    def predict_proba(self, X):
        """Вероятностные предсказания"""
        predictions = np.zeros(len(X))
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return 1 / (1 + np.exp(-predictions))
        
    def predict(self, X):
        """Предсказание классов"""
        return (self.predict_proba(X) >= 0.5).astype(int)
        
    def evaluate(self, X, y):
        """Оценка качества модели"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        return {
            'accuracy': np.mean(y_pred == y),
            'auc_roc': KnuthMetrics.auc_roc_knuth(y, y_proba),
            'mse': np.mean((y - y_proba) ** 2),
            'feature_importance': dict(enumerate(self.feature_importances_))
        } 