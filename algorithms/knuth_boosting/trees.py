"""
Реализация деревьев решений на основе алгоритмов из книги Кнута
том 1, раздел 2.3 "Trees and Binary Trees"
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .sorting import QuickSort3Way

@dataclass
class Node:
    """
    Узел дерева решений (реализация на основе Кнута, том 1, стр. 312)
    """
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[float] = None
    n_samples: int = 0
    impurity: float = 0.0
    impurity_decrease: float = 0.0
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

class KnuthDecisionTree:
    """
    Дерево решений, использующее алгоритмы из книги Кнута
    """
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.sorter = QuickSort3Way()
        self.n_samples = 0
    
    def _calculate_impurity(self, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Вычисляет примесь (для регрессии - взвешенное MSE)
        """
        if sample_weight is None:
            return np.mean((y - np.mean(y)) ** 2)
        
        weighted_mean = np.average(y, weights=sample_weight)
        return np.average((y - weighted_mean) ** 2, weights=sample_weight)
    
    def _calculate_split_impurity(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
        sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Вычисляет примесь для разделения с учетом весов
        """
        left_mask = y_pred <= threshold
        right_mask = ~left_mask
        
        left_impurity = (
            self._calculate_impurity(y[left_mask], sample_weight[left_mask] if sample_weight is not None else None)
            if np.any(left_mask) else float('inf')
        )
        right_impurity = (
            self._calculate_impurity(y[right_mask], sample_weight[right_mask] if sample_weight is not None else None)
            if np.any(right_mask) else float('inf')
        )
        
        return left_impurity + right_impurity, left_mask, right_mask
    
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_index: int,
        sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """
        Находит лучшее разделение используя алгоритмы Кнута с учетом весов
        """
        feature_values = X[:, feature_index]
        sorted_indices = self.sorter.sort(feature_values)
        
        feature_values = feature_values[sorted_indices]
        y_sorted = y[sorted_indices]
        weights_sorted = sample_weight[sorted_indices] if sample_weight is not None else None
        
        best_threshold = None
        best_score = float('inf')
        best_impurity_decrease = 0.0
        
        parent_impurity = self._calculate_impurity(y_sorted, weights_sorted)
        
        # Используем уникальные значения для порогов
        unique_values = np.unique(feature_values)
        
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            score, left_mask, right_mask = self._calculate_split_impurity(
                y_sorted, feature_values, threshold, weights_sorted
            )
            
            # Вычисляем уменьшение примеси с учетом весов
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            
            if weights_sorted is not None:
                w_left = np.sum(weights_sorted[left_mask])
                w_right = np.sum(weights_sorted[right_mask])
                total_weight = w_left + w_right
                
                impurity_decrease = parent_impurity - (
                    (w_left * self._calculate_impurity(y_sorted[left_mask], weights_sorted[left_mask]) +
                     w_right * self._calculate_impurity(y_sorted[right_mask], weights_sorted[right_mask])) / total_weight
                )
            else:
                impurity_decrease = parent_impurity - (
                    (n_left * self._calculate_impurity(y_sorted[left_mask]) +
                     n_right * self._calculate_impurity(y_sorted[right_mask])) / len(y_sorted)
                )
            
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_impurity_decrease = impurity_decrease
        
        return best_threshold, best_score, best_impurity_decrease
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0,
        sample_weight: Optional[np.ndarray] = None
    ) -> Node:
        """
        Рекурсивно строит дерево используя алгоритмы Кнута с учетом весов
        """
        n_samples, n_features = X.shape
        
        node = Node(
            n_samples=n_samples,
            impurity=self._calculate_impurity(y, sample_weight)
        )
        
        # Условия остановки
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            node.value = (
                np.average(y, weights=sample_weight)
                if sample_weight is not None
                else np.mean(y)
            )
            return node
        
        # Поиск лучшего разделения
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        best_impurity_decrease = 0.0
        
        for feature in range(n_features):
            threshold, score, impurity_decrease = self._find_best_split(
                X, y, feature, sample_weight
            )
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_feature = feature
                best_impurity_decrease = impurity_decrease
        
        if best_feature is None:
            node.value = (
                np.average(y, weights=sample_weight)
                if sample_weight is not None
                else np.mean(y)
            )
            return node
        
        # Обновляем информацию об узле
        node.feature_index = best_feature
        node.threshold = best_threshold
        node.impurity_decrease = best_impurity_decrease
        
        # Разделяем данные
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Строим поддеревья
        if np.sum(left_mask) >= self.min_samples_leaf:
            node.left = self._build_tree(
                X[left_mask],
                y[left_mask],
                depth + 1,
                sample_weight[left_mask] if sample_weight is not None else None
            )
        else:
            node.left = Node(
                value=(
                    np.average(y[left_mask], weights=sample_weight[left_mask])
                    if sample_weight is not None
                    else np.mean(y[left_mask])
                )
            )
            
        if np.sum(right_mask) >= self.min_samples_leaf:
            node.right = self._build_tree(
                X[right_mask],
                y[right_mask],
                depth + 1,
                sample_weight[right_mask] if sample_weight is not None else None
            )
        else:
            node.right = Node(
                value=(
                    np.average(y[right_mask], weights=sample_weight[right_mask])
                    if sample_weight is not None
                    else np.mean(y[right_mask])
                )
            )
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'KnuthDecisionTree':
        """
        Обучает дерево на данных с учетом весов
        """
        self.n_samples = len(y)
        self.root = self._build_tree(X, y, sample_weight=sample_weight)
        return self
    
    def _predict_single(self, x: np.ndarray, node: Node) -> float:
        """
        Предсказание для одного образца
        """
        if node.is_leaf():
            return node.value
            
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказания для набора данных
        """
        return np.array([self._predict_single(x, self.root) for x in X]) 