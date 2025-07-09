"""
Тестовый скрипт для визуализации процесса сортировки
"""

import numpy as np
import matplotlib.pyplot as plt
from algorithms.knuth_sort import KnuthSort
from visualization.sorting_visualization import visualize_sorting_process

def test_sorting_visualization():
    """Тест визуализации процесса сортировки"""
    # Генерируем случайный массив
    np.random.seed(42)  # для воспроизводимости
    arr = np.random.randint(1, 100, size=30)
    
    # Создаем экземпляр сортировщика
    sorter = KnuthSort()
    
    # Визуализируем процесс сортировки для трехпутевого быстрого алгоритма
    visualize_sorting_process(
        arr,
        sorter.quicksort_3way,
        'Трехпутевая быстрая сортировка Кнута'
    )

def test_sorting_performance():
    """Тест производительности алгоритмов сортировки"""
    # Тестируем на разных размерах массивов
    sizes = [100, 500, 1000, 2000, 5000]
    
    # Создаем экземпляр сортировщика
    sorter = KnuthSort()
    
    # Словарь алгоритмов
    algorithms = {
        'Quicksort 3-way': sorter.quicksort_3way,
        'Quicksort optimized': sorter.quicksort_optimized,
        'Mergesort': sorter.mergesort_knuth
    }
    
    # Создаем директорию для графиков
    import os
    os.makedirs('visualization/figures/performance', exist_ok=True)
    
    # Тестируем на разных типах данных
    data_types = {
        'random': lambda size: np.random.rand(size),
        'sorted': lambda size: np.sort(np.random.rand(size)),
        'reverse_sorted': lambda size: np.sort(np.random.rand(size))[::-1],
        'many_duplicates': lambda size: np.random.randint(0, size//10, size=size).astype(float)
    }
    
    for data_type, data_func in data_types.items():
        print(f"\nТестирование на данных типа: {data_type}")
        
        # Словарь для хранения результатов
        results = {name: [] for name in algorithms.keys()}
        
        # Тестируем на разных размерах
        for size in sizes:
            print(f"Размер массива: {size}")
            arr = data_func(size)
            
            # Тестируем каждый алгоритм
            for name, func in algorithms.items():
                # Замеряем время
                start_time = time.time()
                _ = func(arr.copy())
                end_time = time.time()
                
                time_taken = end_time - start_time
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
    """Основная функция для запуска тестов"""
    # Тестируем визуализацию
    test_sorting_visualization()
    
    # Тестируем производительность
    test_sorting_performance()

if __name__ == '__main__':
    import time
    main() 