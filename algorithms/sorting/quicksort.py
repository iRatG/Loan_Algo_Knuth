"""
Реализация быстрой сортировки на основе алгоритма Кнута
(Том 3, The Art of Computer Programming, раздел 5.2.2)
"""

import numpy as np
from collections import deque

def partition3(arr, low, high, key_func=None):
    """
    Трехпутевое разбиение массива (Knuth Vol.3, стр. 114)
    
    Args:
        arr: массив для разбиения
        low: нижняя граница
        high: верхняя граница
        key_func: функция для получения ключа сравнения
    
    Returns:
        tuple: (lt, gt) - границы равных элементов
    """
    if key_func is None:
        key_func = lambda x: x
        
    pivot = key_func(arr[high])
    lt = low      # индекс для элементов < pivot
    gt = high     # индекс для элементов > pivot
    i = low       # текущий индекс
    
    while i <= gt:
        current = key_func(arr[i])
        if current < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif current > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1
            
    return lt, gt

def quicksort3(arr, low=None, high=None, key_func=None):
    """
    Итеративная трехпутевая быстрая сортировка (Knuth Vol.3, стр. 115)
    
    Args:
        arr: массив для сортировки
        low: нижняя граница
        high: верхняя граница
        key_func: функция для получения ключа сравнения
    
    Returns:
        array: отсортированный массив
    """
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
        
    # Используем стек вместо рекурсии
    stack = deque([(low, high)])
    
    while stack:
        low, high = stack.pop()
        if low < high:
            lt, gt = partition3(arr, low, high, key_func)
            
            # Сначала обрабатываем меньшую часть
            if lt - low < high - gt:
                stack.append((gt + 1, high))
                if low < lt - 1:
                    stack.append((low, lt - 1))
            else:
                stack.append((low, lt - 1))
                if gt + 1 < high:
                    stack.append((gt + 1, high))
    
    return arr

def sort_by_column(matrix, col_idx):
    """
    Сортировка матрицы по указанному столбцу
    
    Args:
        matrix: numpy массив или список списков
        col_idx: индекс столбца для сортировки
    
    Returns:
        array: отсортированная матрица
    """
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist()
    
    return quicksort3(matrix, key_func=lambda x: x[col_idx]) 