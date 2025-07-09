"""
Модуль для сравнения производительности различных алгоритмов сортировки
"""

import numpy as np
import time
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
from algorithms.knuth_sort import KnuthSort

def generate_test_data(size: int, data_type: str = 'random') -> np.ndarray:
    """
    Генерация тестовых данных различных типов
    
    Args:
        size: Размер массива
        data_type: Тип данных ('random', 'sorted', 'reverse_sorted', 'many_duplicates')
        
    Returns:
        np.ndarray: Тестовый массив
    """
    if data_type == 'random':
        return np.random.rand(size)
    elif data_type == 'sorted':
        return np.sort(np.random.rand(size))
    elif data_type == 'reverse_sorted':
        return np.sort(np.random.rand(size))[::-1]
    elif data_type == 'many_duplicates':
        return np.random.randint(0, size//10, size=size).astype(float)
    else:
        raise ValueError(f"Неизвестный тип данных: {data_type}")

def measure_sorting_time(sort_func: Callable, arr: np.ndarray) -> float:
    """
    Измерение времени выполнения сортировки
    
    Args:
        sort_func: Функция сортировки
        arr: Массив для сортировки
        
    Returns:
        float: Время выполнения в секундах
    """
    start_time = time.time()
    sorted_arr = sort_func(arr)
    end_time = time.time()
    
    # Проверяем корректность сортировки
    assert np.all(np.diff(sorted_arr) >= 0), "Массив отсортирован неправильно!"
    
    return end_time - start_time

def compare_sorting_algorithms(sizes: List[int], data_types: List[str]) -> None:
    """
    Сравнение алгоритмов сортировки на разных размерах и типах данных
    
    Args:
        sizes: Список размеров массивов
        data_types: Список типов данных
    """
    sorter = KnuthSort()
    algorithms = {
        'Quicksort 3-way': sorter.quicksort_3way,
        'Quicksort optimized': sorter.quicksort_optimized,
        'Mergesort': sorter.mergesort_knuth
    }
    
    # Создаем директорию для графиков, если её нет
    import os
    os.makedirs('visualization/figures/performance', exist_ok=True)
    
    # Для каждого типа данных
    for data_type in data_types:
        print(f"\nТестирование на данных типа: {data_type}")
        
        # Словарь для хранения результатов
        results = {name: [] for name in algorithms.keys()}
        
        # Тестируем на разных размерах
        for size in sizes:
            print(f"Размер массива: {size}")
            arr = generate_test_data(size, data_type)
            
            # Тестируем каждый алгоритм
            for name, func in algorithms.items():
                time_taken = measure_sorting_time(func, arr)
                results[name].append(time_taken)
                print(f"{name}: {time_taken:.4f} сек")
        
        # Строим график
        plt.figure(figsize=(10, 6))
        for name, times in results.items():
            plt.plot(sizes, times, marker='o', label=name)
            
        plt.xlabel('Размер массива')
        plt.ylabel('Время выполнения (сек)')
        plt.title(f'Сравнение алгоритмов сортировки\nТип данных: {data_type}')
        plt.legend()
        plt.grid(True)
        
        # Сохраняем график
        plt.savefig(f'visualization/figures/performance/sorting_{data_type}.png')
        plt.close()

def main():
    """
    Основная функция для запуска тестов
    """
    # Тестируем на разных размерах массивов
    sizes = [1000, 5000, 10000, 50000, 100000]
    
    # Тестируем на разных типах данных
    data_types = ['random', 'sorted', 'reverse_sorted', 'many_duplicates']
    
    # Запускаем сравнение
    compare_sorting_algorithms(sizes, data_types)

if __name__ == '__main__':
    main() 