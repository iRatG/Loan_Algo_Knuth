import numpy as np
from typing import Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin

class KnuthPreprocessor(BaseEstimator, TransformerMixin):
    """
    Препроцессор данных для градиентного бустинга на основе алгоритмов Кнута.
    """
    
    def __init__(
        self,
        handle_missing: bool = True,
        handle_categorical: bool = True,
        normalize: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Инициализация препроцессора.
        
        Args:
            handle_missing: Обработка пропущенных значений
            handle_categorical: Обработка категориальных признаков
            normalize: Нормализация числовых признаков
            random_state: Seed для воспроизводимости
        """
        self.handle_missing = handle_missing
        self.handle_categorical = handle_categorical
        self.normalize = normalize
        self.random_state = random_state
        
        self.numerical_means_ = None
        self.numerical_stds_ = None
        self.categorical_maps_ = {}
        
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """
        Заполнение пропущенных значений.
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Матрица с заполненными пропусками
        """
        if not self.handle_missing:
            return X
            
        X = X.copy()
        if self.numerical_means_ is None:
            self.numerical_means_ = np.nanmean(X, axis=0)
            
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            X[mask, col] = self.numerical_means_[col]
            
        return X
        
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Нормализация числовых признаков.
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Нормализованная матрица
        """
        if not self.normalize:
            return X
            
        X = X.copy()
        if self.numerical_stds_ is None:
            self.numerical_means_ = np.mean(X, axis=0)
            self.numerical_stds_ = np.std(X, axis=0)
            self.numerical_stds_[self.numerical_stds_ == 0] = 1.0
            
        return (X - self.numerical_means_) / self.numerical_stds_
        
    def _encode_categorical(self, X: np.ndarray) -> np.ndarray:
        """
        Кодирование категориальных признаков.
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Матрица с закодированными категориальными признаками
        """
        if not self.handle_categorical:
            return X
            
        X = X.copy()
        for col in range(X.shape[1]):
            if X[:, col].dtype.name == 'object' or X[:, col].dtype.name == 'category':
                if col not in self.categorical_maps_:
                    unique_values = np.unique(X[:, col])
                    self.categorical_maps_[col] = {val: i for i, val in enumerate(unique_values)}
                    
                X[:, col] = np.array([self.categorical_maps_[col].get(x, -1) for x in X[:, col]])
                
        return X
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'KnuthPreprocessor':
        """
        Обучение препроцессора.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная (не используется)
            
        Returns:
            self: Обученный препроцессор
        """
        X = self._handle_missing_values(X)
        X = self._encode_categorical(X)
        X = self._normalize_features(X)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Преобразование данных.
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Преобразованная матрица
        """
        X = self._handle_missing_values(X)
        X = self._encode_categorical(X)
        X = self._normalize_features(X)
        return X
        
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Обучение и преобразование данных.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная (не используется)
            
        Returns:
            np.ndarray: Преобразованная матрица
        """
        return self.fit(X).transform(X) 

    def analyze_class_balance(self, y):
        """Анализ баланса классов"""
        unique, counts = np.unique(y, return_counts=True)
        balance = dict(zip(unique, counts))
        total = len(y)
        ratios = {cls: count/total for cls, count in balance.items()}
        return {
            'counts': balance,
            'ratios': ratios,
            'imbalance_ratio': max(ratios.values()) / min(ratios.values())
        }
    
    def get_balanced_batch_indices(self, y, batch_size):
        """Получение индексов для сбалансированного батча"""
        classes = np.unique(y)
        n_classes = len(classes)
        samples_per_class = batch_size // n_classes
        
        indices = []
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) > 0:
                selected = np.random.choice(
                    cls_indices,
                    size=min(samples_per_class, len(cls_indices)),
                    replace=False
                )
                indices.extend(selected)
        
        # Добираем до нужного размера батча
        if len(indices) < batch_size:
            remaining = batch_size - len(indices)
            all_indices = np.arange(len(y))
            available = np.setdiff1d(all_indices, indices)
            additional = np.random.choice(available, size=remaining, replace=False)
            indices.extend(additional)
            
        return np.array(indices)
    
    def get_stratified_batches(self, X, y, batch_size):
        """Генератор стратифицированных батчей"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = self.get_balanced_batch_indices(y, batch_size)
            yield X[batch_indices], y[batch_indices] 