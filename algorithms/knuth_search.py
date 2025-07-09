"""
Реализация алгоритмов поиска из книги Кнута
The Art of Computer Programming, Volume 3: Sorting and Searching

Основные алгоритмы:
1. Поиск k-го элемента (Vol 3, 5.3.3)
2. Бинарный поиск с оптимизацией (Vol 3, 6.2.1)
3. Поиск в отсортированном массиве (Vol 3, 6.2)
"""

import numpy as np
from typing import List, Tuple, Optional
from .knuth_sort import KnuthSort

class KnuthSearch:
    def __init__(self):
        self.sorter = KnuthSort()
    
    def select_kth(self, arr: np.ndarray, k: int, column: Optional[int] = None) -> float:
        """
        Поиск k-го элемента по алгоритму Кнута (Vol 3, 5.3.3)
        
        Args:
            arr: Массив для поиска
            k: Индекс искомого элемента (0-based)
            column: Номер колонки (если None, ищем в одномерном массиве)
            
        Returns:
            k-й элемент
        """
        def partition(arr: np.ndarray, left: int, right: int) -> int:
            if column is not None:
                pivot = arr[right, column]
            else:
                pivot = arr[right]
                
            i = left - 1
            
            for j in range(left, right):
                if column is not None:
                    curr = arr[j, column]
                else:
                    curr = arr[j]
                    
                if curr <= pivot:
                    i += 1
                    if column is not None:
                        arr[[i, j]] = arr[[j, i]]
                    else:
                        arr[i], arr[j] = arr[j], arr[i]
                        
            if column is not None:
                arr[[i+1, right]] = arr[[right, i+1]]
            else:
                arr[i+1], arr[right] = arr[right], arr[i+1]
                
            return i + 1
        
        def select(arr: np.ndarray, left: int, right: int, k: int) -> float:
            if left == right:
                if column is not None:
                    return arr[left, column]
                return arr[left]
                
            pivot_idx = partition(arr, left, right)
            
            if k == pivot_idx:
                if column is not None:
                    return arr[k, column]
                return arr[k]
            elif k < pivot_idx:
                return select(arr, left, pivot_idx - 1, k)
            else:
                return select(arr, pivot_idx + 1, right, k)
        
        if k < 0 or k >= len(arr):
            raise ValueError("k должно быть в диапазоне [0, len(arr)-1]")
            
        arr_copy = arr.copy()
        return select(arr_copy, 0, len(arr_copy) - 1, k)
    
    def binary_search_knuth(self, arr: np.ndarray, target: float, 
                          column: Optional[int] = None) -> int:
        """
        Бинарный поиск с оптимизацией Кнута (Vol 3, 6.2.1)
        
        Args:
            arr: Отсортированный массив
            target: Искомое значение
            column: Номер колонки (если None, ищем в одномерном массиве)
            
        Returns:
            Индекс элемента или -1, если не найден
        """
        if len(arr) == 0:
            return -1
            
        left = 0
        right = len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if column is not None:
                curr = arr[mid, column]
            else:
                curr = arr[mid]
                
            if curr == target:
                return mid
            elif curr < target:
                left = mid + 1
            else:
                right = mid - 1
                
        return -1
    
    def interpolation_search(self, arr: np.ndarray, target: float,
                           column: Optional[int] = None) -> int:
        """
        Интерполяционный поиск Кнута (Vol 3, 6.2.1)
        
        Args:
            arr: Отсортированный массив
            target: Искомое значение
            column: Номер колонки (если None, ищем в одномерном массиве)
            
        Returns:
            Индекс элемента или -1, если не найден
        """
        if len(arr) == 0:
            return -1
            
        left = 0
        right = len(arr) - 1
        
        while left <= right:
            if column is not None:
                if arr[right, column] == arr[left, column]:
                    if arr[left, column] == target:
                        return left
                    return -1
                    
                pos = left + ((target - arr[left, column]) * (right - left)) // \
                      (arr[right, column] - arr[left, column])
            else:
                if arr[right] == arr[left]:
                    if arr[left] == target:
                        return left
                    return -1
                    
                pos = left + ((target - arr[left]) * (right - left)) // \
                      (arr[right] - arr[left])
            
            if pos < left or pos > right:
                return -1
                
            if column is not None:
                curr = arr[pos, column]
            else:
                curr = arr[pos]
                
            if curr == target:
                return pos
            elif curr < target:
                left = pos + 1
            else:
                right = pos - 1
                
        return -1
    
    def find_range(self, arr: np.ndarray, low: float, high: float,
                  column: Optional[int] = None) -> Tuple[int, int]:
        """
        Поиск диапазона значений в отсортированном массиве
        
        Args:
            arr: Отсортированный массив
            low: Нижняя граница
            high: Верхняя граница
            column: Номер колонки (если None, ищем в одномерном массиве)
            
        Returns:
            (начальный индекс, конечный индекс) или (-1, -1), если не найдено
        """
        # Находим левую границу
        left = 0
        right = len(arr) - 1
        left_bound = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            if column is not None:
                curr = arr[mid, column]
            else:
                curr = arr[mid]
                
            if curr >= low:
                left_bound = mid
                right = mid - 1
            else:
                left = mid + 1
                
        if left_bound == -1:
            return (-1, -1)
            
        # Находим правую границу
        left = left_bound
        right = len(arr) - 1
        right_bound = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            if column is not None:
                curr = arr[mid, column]
            else:
                curr = arr[mid]
                
            if curr <= high:
                right_bound = mid
                left = mid + 1
            else:
                right = mid - 1
                
        if right_bound == -1:
            return (-1, -1)
            
        return (left_bound, right_bound) 