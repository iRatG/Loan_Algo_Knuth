"""
Модуль для обработки выбросов на основе алгоритмов Кнута
The Art of Computer Programming, Volume 2: Seminumerical Algorithms
"""

import numpy as np
from typing import Tuple, List
from .knuth_statistics import KnuthStatistics
from .knuth_sort import KnuthSort

class KnuthOutlierDetector:
    def __init__(self):
        self.stats = KnuthStatistics()
        self.sorter = KnuthSort()
        
    def compute_robust_stats(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Вычисление робастных статистик по алгоритму Кнута (Vol 2, 4.2.3)
        
        Args:
            x: Одномерный массив данных
            
        Returns:
            Tuple[float, float]: (робастное среднее, робастное стандартное отклонение)
        """
        n = len(x)
        if n == 0:
            return 0.0, 0.0
            
        # Сортируем данные для вычисления квантилей
        sorted_x = self.sorter.quicksort(x.copy())
        
        # Находим медиану (P50)
        median_idx = n // 2
        median = sorted_x[median_idx] if n % 2 == 1 else \
                (sorted_x[median_idx - 1] + sorted_x[median_idx]) / 2
                
        # Находим квартили (P25 и P75)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_x[q1_idx]
        q3 = sorted_x[q3_idx]
        
        # Межквартильный размах (IQR)
        iqr = q3 - q1
        
        # Робастное стандартное отклонение (на основе IQR)
        robust_std = iqr / 1.349  # Константа из книги Кнута
        
        return median, robust_std
        
    def detect_outliers(self, x: np.ndarray, k: float = 3.0) -> np.ndarray:
        """
        Определение выбросов с использованием робастных статистик
        
        Args:
            x: Одномерный массив данных
            k: Количество робастных стандартных отклонений для определения выбросов
            
        Returns:
            np.ndarray: Булев массив, True для выбросов
        """
        median, robust_std = self.compute_robust_stats(x)
        
        # Определяем границы выбросов
        lower_bound = median - k * robust_std
        upper_bound = median + k * robust_std
        
        # Находим выбросы
        outliers = (x < lower_bound) | (x > upper_bound)
        
        return outliers
        
    def remove_outliers(self, x: np.ndarray, k: float = 3.0) -> np.ndarray:
        """
        Удаление выбросов из данных
        
        Args:
            x: Одномерный массив данных
            k: Количество робастных стандартных отклонений
            
        Returns:
            np.ndarray: Массив без выбросов
        """
        outliers = self.detect_outliers(x, k)
        return x[~outliers]
        
    def winsorize(self, x: np.ndarray, k: float = 3.0) -> np.ndarray:
        """
        Винзоризация данных (замена выбросов граничными значениями)
        
        Args:
            x: Одномерный массив данных
            k: Количество робастных стандартных отклонений
            
        Returns:
            np.ndarray: Винзоризованный массив
        """
        median, robust_std = self.compute_robust_stats(x)
        
        # Определяем границы
        lower_bound = median - k * robust_std
        upper_bound = median + k * robust_std
        
        # Создаем копию массива
        winsorized = x.copy()
        
        # Заменяем значения
        winsorized[winsorized < lower_bound] = lower_bound
        winsorized[winsorized > upper_bound] = upper_bound
        
        return winsorized 