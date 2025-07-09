"""
Модуль для визуализации процесса сортировки
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable
import time

def visualize_sorting_process(arr: np.ndarray, sort_func: Callable, title: str) -> None:
    """
    Визуализация процесса сортировки с промежуточными состояниями
    
    Args:
        arr: Исходный массив
        sort_func: Функция сортировки
        title: Название алгоритма
    """
    # Создаем копию массива для сортировки
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    # Создаем фигуру
    plt.figure(figsize=(12, 6))
    
    # Отображаем исходный массив
    plt.subplot(2, 2, 1)
    plt.bar(range(n), arr_copy)
    plt.title('Исходный массив')
    plt.ylim(0, max(arr_copy) * 1.1)
    
    # Сортируем 1/3 массива
    arr_copy[:n//3] = np.sort(arr_copy[:n//3])
    plt.subplot(2, 2, 2)
    plt.bar(range(n), arr_copy)
    plt.title('После сортировки 1/3')
    plt.ylim(0, max(arr_copy) * 1.1)
    
    # Сортируем 2/3 массива
    arr_copy[:2*n//3] = np.sort(arr_copy[:2*n//3])
    plt.subplot(2, 2, 3)
    plt.bar(range(n), arr_copy)
    plt.title('После сортировки 2/3')
    plt.ylim(0, max(arr_copy) * 1.1)
    
    # Сортируем весь массив
    arr_copy = sort_func(arr)
    plt.subplot(2, 2, 4)
    plt.bar(range(n), arr_copy)
    plt.title('Полностью отсортированный массив')
    plt.ylim(0, max(arr_copy) * 1.1)
    
    # Настраиваем общий заголовок
    plt.suptitle(f'Визуализация процесса сортировки\n{title}')
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig('visualization/figures/sorting_process.png')
    plt.close() 