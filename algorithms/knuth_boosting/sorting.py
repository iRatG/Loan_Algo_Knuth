"""
Реализация алгоритмов сортировки из книги Кнута том 3
"""

import numpy as np
from typing import List, Union, Tuple

class QuickSort3Way:
    """
    Трехпутевая быстрая сортировка из Кнута (том 3, раздел 5.2.2)
    """
    def __init__(self):
        self.array = None

    def _partition3(self, low: int, high: int) -> Tuple[int, int]:
        """
        Разделение массива на три части:
        - меньше опорного
        - равные опорному
        - больше опорного
        """
        if high <= low:
            return low, high
            
        # Выбираем опорный элемент как медиану из трёх
        mid = (low + high) // 2
        pivot_candidates = [
            (self.array[low], low),
            (self.array[mid], mid),
            (self.array[high], high)
        ]
        pivot_value, pivot_idx = sorted(pivot_candidates, key=lambda x: x[0])[1]
        
        # Перемещаем опорный элемент в начало
        self.array[low], self.array[pivot_idx] = self.array[pivot_idx], self.array[low]
        pivot = self.array[low]
        
        lt = low  # Индекс для элементов < pivot
        gt = high  # Индекс для элементов > pivot
        i = low + 1  # Текущий индекс
        
        while i <= gt:
            if self.array[i] < pivot:
                self.array[lt], self.array[i] = self.array[i], self.array[lt]
                lt += 1
                i += 1
            elif self.array[i] > pivot:
                self.array[i], self.array[gt] = self.array[gt], self.array[i]
                gt -= 1
            else:
                i += 1
                
        return lt, gt

    def _sort(self, low: int, high: int):
        """
        Рекурсивная часть алгоритма
        """
        if low >= high:
            return
            
        # Получаем границы равных элементов
        lt, gt = self._partition3(low, high)
        
        # Сортируем подмассивы
        self._sort(low, lt - 1)
        self._sort(gt + 1, high)

    def sort(self, arr: Union[List, np.ndarray]) -> np.ndarray:
        """
        Сортировка массива
        """
        # Преобразуем вход в numpy array
        self.array = np.array(arr)
        n = len(self.array)
        
        # Запускаем сортировку
        self._sort(0, n - 1)
        
        # Возвращаем индексы для сортировки
        return np.argsort(self.array) 