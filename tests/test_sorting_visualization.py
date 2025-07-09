"""
Тестовый скрипт для визуализации процесса сортировки
"""

import numpy as np
from algorithms.knuth_sort import KnuthSort
from visualization.sorting_visualization import visualize_sorting_process

def main():
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

if __name__ == '__main__':
    main() 