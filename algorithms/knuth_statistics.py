"""
Реализация статистических алгоритмов из книги Кнута
The Art of Computer Programming, Volume 2: Seminumerical Algorithms

Основные алгоритмы:
1. Инкрементальное вычисление статистик (Vol 2, 4.2.2)
2. Поиск квантилей (Vol 3, 5.3.3)
3. Оценка корреляций (Vol 2, 4.2.4)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .knuth_sort import KnuthSort

class KnuthStatistics:
    def __init__(self):
        self.sorter = KnuthSort()
        self._reset_stats()
        
    def _reset_stats(self):
        """Сброс накопленных статистик"""
        self.n = 0
        self.means = {}
        self.m2 = {}  # Для вычисления дисперсии
        self.min_vals = {}
        self.max_vals = {}
        self.quantiles = {}
        
    def update(self, data: np.ndarray, columns: Optional[List[int]] = None) -> None:
        """
        Инкрементальное обновление статистик (Vol 2, 4.2.2)
        
        Args:
            data: Новые данные
            columns: Список колонок для обработки (если None, обрабатываем все)
        """
        if columns is None:
            columns = range(data.shape[1])
            
        for col in columns:
            if col not in self.means:
                self.means[col] = 0
                self.m2[col] = 0
                self.min_vals[col] = float('inf')
                self.max_vals[col] = float('-inf')
                
            values = data[:, col]
            
            # Обновляем min/max
            self.min_vals[col] = min(self.min_vals[col], np.min(values))
            self.max_vals[col] = max(self.max_vals[col], np.max(values))
            
            # Обновляем среднее и M2 для дисперсии
            for x in values:
                self.n += 1
                delta = x - self.means[col]
                self.means[col] += delta / self.n
                delta2 = x - self.means[col]
                self.m2[col] += delta * delta2
    
    def get_stats(self, column: int) -> Dict:
        """
        Получение статистик для колонки
        
        Args:
            column: Номер колонки
            
        Returns:
            Словарь со статистиками
        """
        if self.n < 2:
            return {
                'mean': self.means.get(column, 0),
                'variance': 0,
                'std': 0,
                'min': self.min_vals.get(column, 0),
                'max': self.max_vals.get(column, 0),
                'n': self.n
            }
            
        variance = self.m2[column] / (self.n - 1)
        
        return {
            'mean': self.means[column],
            'variance': variance,
            'std': np.sqrt(variance),
            'min': self.min_vals[column],
            'max': self.max_vals[column],
            'n': self.n
        }
    
    def find_quantiles(self, data: np.ndarray, column: int, q: List[float]) -> List[float]:
        """
        Поиск квантилей методом Кнута (Vol 3, 5.3.3)
        
        Args:
            data: Данные
            column: Номер колонки
            q: Список квантилей (0 <= q <= 1)
            
        Returns:
            Список значений квантилей
        """
        if len(data) == 0:
            return [0] * len(q)
            
        # Сортируем данные
        sorted_data = self.sorter.quicksort_3way(data, column)
        
        results = []
        n = len(sorted_data)
        
        for quantile in q:
            k = int(quantile * (n - 1))
            alpha = quantile * (n - 1) - k
            
            if k + 1 >= n:
                value = sorted_data[-1, column]
            else:
                value = (1 - alpha) * sorted_data[k, column] + alpha * sorted_data[k + 1, column]
                
            results.append(value)
            
        return results
    
    def correlation_knuth(self, data: np.ndarray, col1: int, col2: int) -> float:
        """
        Вычисление корреляции по методу Кнута (Vol 2, 4.2.4)
        
        Args:
            data: Данные
            col1: Первая колонка
            col2: Вторая колонка
            
        Returns:
            Коэффициент корреляции
        """
        n = len(data)
        if n < 2:
            return 0.0
            
        mean1 = mean2 = 0.0
        m2_1 = m2_2 = 0.0
        covar = 0.0
        
        for i in range(n):
            x = data[i, col1]
            y = data[i, col2]
            
            # Обновляем средние и M2 для x
            delta1 = x - mean1
            mean1 += delta1 / (i + 1)
            delta1_new = x - mean1
            m2_1 += delta1 * delta1_new
            
            # Обновляем средние и M2 для y
            delta2 = y - mean2
            mean2 += delta2 / (i + 1)
            delta2_new = y - mean2
            m2_2 += delta2 * delta2_new
            
            # Обновляем ковариацию
            covar += (delta1 * delta2_new * i) / (i + 1)
            
        # Вычисляем корреляцию
        var1 = m2_1 / (n - 1)
        var2 = m2_2 / (n - 1)
        
        if var1 == 0 or var2 == 0:
            return 0.0
            
        return covar / np.sqrt(var1 * var2) / (n - 1)
    
    def analyze_feature_importance(self, data: np.ndarray, target_col: int) -> Dict[int, float]:
        """
        Анализ важности признаков через корреляцию с целевой переменной
        
        Args:
            data: Данные
            target_col: Номер колонки с целевой переменной
            
        Returns:
            Словарь {номер_колонки: важность}
        """
        importance = {}
        n_features = data.shape[1]
        
        for col in range(n_features):
            if col != target_col:
                corr = abs(self.correlation_knuth(data, col, target_col))
                importance[col] = corr
                
        return importance 