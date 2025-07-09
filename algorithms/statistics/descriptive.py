"""
Реализация базовых статистических алгоритмов на основе методов Кнута
(Том 2, The Art of Computer Programming, раздел 4.2.2)
"""

import numpy as np
from ..sorting.quicksort import quicksort3

def knuth_mean_variance(data):
    """
    Однопроходный алгоритм Кнута для вычисления среднего и дисперсии
    (Knuth Vol.2, стр. 232)
    
    Args:
        data: одномерный массив данных
    
    Returns:
        tuple: (среднее, дисперсия)
    """
    n = 0
    mean = 0
    M2 = 0
    
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        M2 += delta * (x - mean)
    
    if n < 2:
        return mean, 0
    
    variance = M2 / (n - 1)
    return mean, variance

def median(data):
    """
    Вычисление медианы с использованием быстрой сортировки
    (Knuth Vol.3, стр. 199)
    
    Args:
        data: одномерный массив данных
    
    Returns:
        float: медиана
    """
    sorted_data = quicksort3(data.copy())
    n = len(sorted_data)
    
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        return sorted_data[n//2]

def quantiles(data, q):
    """
    Вычисление квантилей
    (Knuth Vol.3, стр. 200)
    
    Args:
        data: одномерный массив данных
        q: список квантилей (0-1)
    
    Returns:
        list: значения квантилей
    """
    sorted_data = quicksort3(data.copy())
    n = len(sorted_data)
    
    result = []
    for p in q:
        if p < 0 or p > 1:
            raise ValueError("Квантили должны быть в диапазоне [0,1]")
            
        h = (n - 1) * p
        i = int(h)
        if i == h:
            result.append(sorted_data[i])
        else:
            result.append(sorted_data[i] + (h - i) * (sorted_data[i + 1] - sorted_data[i]))
    
    return result

def correlation(x, y):
    """
    Вычисление корреляции Пирсона
    (Knuth Vol.2, стр. 233)
    
    Args:
        x, y: одномерные массивы данных одинаковой длины
    
    Returns:
        float: коэффициент корреляции
    """
    if len(x) != len(y):
        raise ValueError("Массивы должны быть одинаковой длины")
    
    n = len(x)
    mean_x, var_x = knuth_mean_variance(x)
    mean_y, var_y = knuth_mean_variance(y)
    
    covariance = 0
    for i in range(n):
        covariance += (x[i] - mean_x) * (y[i] - mean_y)
    
    covariance /= (n - 1)
    
    if var_x == 0 or var_y == 0:
        return 0
    
    return covariance / np.sqrt(var_x * var_y) 