"""
Реализация алгоритмов сортировки из книги Кнута
The Art of Computer Programming, Volume 3: Sorting and Searching

Основные алгоритмы:
1. Трехпутевая быстрая сортировка (Vol 3, 5.2.2)
2. Сортировка слиянием с оптимизацией памяти (Vol 3, 5.2.4)
3. Сортировка подсчетом для целых чисел (Vol 3, 5.2)
4. Оптимизированная быстрая сортировка с медианой из трёх (Vol 3, 5.2.2)
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
        
        def quicksort3_iterative(arr: np.ndarray, low: int, high: int):
            """Итеративная версия трехпутевой быстрой сортировки"""
            # Стек для хранения границ подмассивов
            stack = [(low, high)]
            
            while stack:
                low, high = stack.pop()
                
                if low >= high:
                    continue
                    
                # Разделяем массив
                lt, gt = partition3(arr, low, high)
                
                # Добавляем большую часть в стек первой
                if gt - lt > high - gt:
                    if lt - 1 > low:
                        stack.append((low, lt - 1))
                    if high > gt + 1:
                        stack.append((gt + 1, high))
                else:
                    if high > gt + 1:
                        stack.append((gt + 1, high))
                    if lt - 1 > low:
                        stack.append((low, lt - 1))
        
        if len(arr) <= 1:
            return arr
            
        arr_copy = arr.copy()
        quicksort3_iterative(arr_copy, 0, len(arr_copy) - 1)
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
        left = KnuthSort.mergesort_knuth(arr[:mid], column)
        right = KnuthSort.mergesort_knuth(arr[mid:], column)
        
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

    @staticmethod
    def quicksort_optimized(arr: np.ndarray, column: Optional[int] = None) -> np.ndarray:
        """
        Оптимизированная быстрая сортировка Кнута с медианой из трёх (Vol 3, 5.2.2)
        
        Оптимизации:
        1. Выбор опорного элемента как медианы из трёх
        2. Вставочная сортировка для малых подмассивов
        3. Оптимизация рекурсии для избежания переполнения стека
        
        Args:
            arr: Массив для сортировки
            column: Номер столбца для сортировки (если None, сортируем весь массив)
            
        Returns:
            Отсортированный массив
        """
        def get_value(arr: np.ndarray, index: int) -> Union[float, int]:
            """Получение значения с учетом столбца"""
            return arr[index, column] if column is not None else arr[index]
        
        def median_of_three(low: int, high: int) -> int:
            """Выбор медианы из трёх элементов"""
            mid = (low + high) // 2
            
            a = get_value(arr_copy, low)
            b = get_value(arr_copy, mid)
            c = get_value(arr_copy, high)
            
            if a <= b <= c:
                return mid
            if c <= b <= a:
                return mid
            if b <= a <= c:
                return low
            if c <= a <= b:
                return low
            return high
        
        def insertion_sort(low: int, high: int):
            """Вставочная сортировка для малых подмассивов"""
            for i in range(low + 1, high + 1):
                key = arr_copy[i].copy()
                key_val = get_value(arr_copy, i)
                j = i - 1
                
                while j >= low and get_value(arr_copy, j) > key_val:
                    arr_copy[j + 1] = arr_copy[j]
                    j -= 1
                    
                arr_copy[j + 1] = key
        
        def partition(low: int, high: int) -> int:
            """Разделение массива с использованием медианы из трёх"""
            pivot_idx = median_of_three(low, high)
            pivot_val = get_value(arr_copy, pivot_idx)
            
            # Перемещаем опорный элемент в конец
            arr_copy[pivot_idx], arr_copy[high] = arr_copy[high].copy(), arr_copy[pivot_idx].copy()
            
            i = low - 1
            
            for j in range(low, high):
                if get_value(arr_copy, j) <= pivot_val:
                    i += 1
                    arr_copy[i], arr_copy[j] = arr_copy[j].copy(), arr_copy[i].copy()
                    
            arr_copy[i + 1], arr_copy[high] = arr_copy[high].copy(), arr_copy[i + 1].copy()
            return i + 1
        
        def quicksort_internal(low: int, high: int):
            """Внутренняя функция быстрой сортировки"""
            while low < high:
                # Используем вставочную сортировку для малых подмассивов
                if high - low + 1 <= 10:
                    insertion_sort(low, high)
                    break
                
                # Разделяем массив
                pivot = partition(low, high)
                
                # Рекурсивно сортируем меньшую часть
                if pivot - low < high - pivot:
                    quicksort_internal(low, pivot - 1)
                    low = pivot + 1
                else:
                    quicksort_internal(pivot + 1, high)
                    high = pivot - 1
        
        if len(arr) <= 1:
            return arr
            
        arr_copy = arr.copy()
        quicksort_internal(0, len(arr_copy) - 1)
        return arr_copy 