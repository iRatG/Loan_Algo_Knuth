"""
Реализация алгоритмов сортировки из книги Кнута
The Art of Computer Programming, Volume 3: Sorting and Searching

Основные алгоритмы:
1. Трехпутевая быстрая сортировка (Vol 3, 5.2.2)
2. Сортировка слиянием с оптимизацией памяти (Vol 3, 5.2.4)
3. Сортировка подсчетом для целых чисел (Vol 3, 5.2)
"""

import numpy as np
from typing import List, Tuple, Union, Optional

class KnuthSort:
    @staticmethod
    def quicksort_3way(arr: np.ndarray, column: Optional[int] = None) -> np.ndarray:
        """
        Трехпутевая быстрая сортировка Кнута (Vol 3, 5.2.2)
        Особенно эффективна для данных с повторениями
        
        Args:
            arr: Массив для сортировки
            column: Номер столбца для сортировки (если None, сортируем весь массив)
            
        Returns:
            Отсортированный массив
        """
        def partition3(arr: np.ndarray, low: int, high: int) -> Tuple[int, int]:
            if column is not None:
                pivot = arr[high, column]
            else:
                pivot = arr[high]
                
            lt = low  # элементы < pivot
            gt = high # элементы > pivot
            i = low   # текущий элемент
            
            while i <= gt:
                if column is not None:
                    curr = arr[i, column]
                else:
                    curr = arr[i]
                    
                if curr < pivot:
                    if column is not None:
                        arr[[lt, i]] = arr[[i, lt]]
                    else:
                        arr[lt], arr[i] = arr[i], arr[lt]
                    lt += 1
                    i += 1
                elif curr > pivot:
                    if column is not None:
                        arr[[i, gt]] = arr[[gt, i]]
                    else:
                        arr[i], arr[gt] = arr[gt], arr[i]
                    gt -= 1
                else:
                    i += 1
            
            return lt, gt
        
        def quicksort3_recursive(arr: np.ndarray, low: int, high: int):
            if low >= high:
                return
                
            lt, gt = partition3(arr, low, high)
            quicksort3_recursive(arr, low, lt - 1)
            quicksort3_recursive(arr, gt + 1, high)
        
        if len(arr) <= 1:
            return arr
            
        arr_copy = arr.copy()
        quicksort3_recursive(arr_copy, 0, len(arr_copy) - 1)
        return arr_copy
    
    @staticmethod
    def mergesort_knuth(arr: np.ndarray, column: Optional[int] = None) -> np.ndarray:
        """
        Сортировка слиянием с оптимизацией памяти (Vol 3, 5.2.4)
        
        Args:
            arr: Массив для сортировки
            column: Номер столбца для сортировки (если None, сортируем весь массив)
            
        Returns:
            Отсортированный массив
        """
        def merge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            if column is not None:
                i, j = 0, 0
                result = np.empty_like(left)
                
                while i < len(left) and j < len(right):
                    if left[i, column] <= right[j, column]:
                        result[i+j] = left[i]
                        i += 1
                    else:
                        result[i+j] = right[j]
                        j += 1
                        
                result[i+j:] = left[i:] if i < len(left) else right[j:]
                return result
            else:
                return np.sort(np.concatenate([left, right]))
        
        if len(arr) <= 1:
            return arr
            
        mid = len(arr) // 2
        left = self.mergesort_knuth(arr[:mid], column)
        right = self.mergesort_knuth(arr[mid:], column)
        
        return merge(left, right)
    
    @staticmethod
    def counting_sort(arr: np.ndarray, column: Optional[int] = None, max_val: Optional[int] = None) -> np.ndarray:
        """
        Сортировка подсчетом для целых чисел (Vol 3, 5.2)
        
        Args:
            arr: Массив для сортировки
            column: Номер столбца для сортировки (если None, сортируем весь массив)
            max_val: Максимальное значение в массиве (если None, вычисляется)
            
        Returns:
            Отсортированный массив
        """
        if column is not None:
            values = arr[:, column]
        else:
            values = arr
            
        if max_val is None:
            max_val = int(np.max(values))
            
        # Создаем массив для подсчета
        count = np.zeros(max_val + 1, dtype=int)
        
        # Подсчитываем количество каждого элемента
        for val in values:
            count[int(val)] += 1
            
        # Вычисляем позиции элементов
        for i in range(1, len(count)):
            count[i] += count[i-1]
            
        # Создаем выходной массив
        output = np.empty_like(arr)
        
        # Расставляем элементы
        for i in range(len(arr)-1, -1, -1):
            if column is not None:
                val = int(arr[i, column])
                count[val] -= 1
                output[count[val]] = arr[i]
            else:
                val = int(arr[i])
                count[val] -= 1
                output[count[val]] = arr[i]
                
        return output 