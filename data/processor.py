"""
Модуль обработки финансовых данных с использованием алгоритмов Кнута
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from tqdm import tqdm

from algorithms import (
    quicksort3,
    sort_by_column,
    knuth_mean_variance,
    median,
    quantiles,
    correlation
)

class FinancialDataProcessor:
    """Класс для обработки финансовых данных"""
    
    def __init__(self, source_dir: str = "source", batch_size: int = 100):
        """
        Инициализация процессора данных
        
        Args:
            source_dir: директория с исходными данными
            batch_size: размер батча для обработки
        """
        self.source_dir = Path(source_dir)
        self.raw_data = None
        self.processed_data = None
        self.statistics = {}
        self.batch_size = batch_size
    
    def load_data(self, filename: str, sample_size: int = 1000, random_seed: int = 42) -> pd.DataFrame:
        """
        Загрузка данных из файла с выборкой
        
        Args:
            filename: имя файла с данными
            sample_size: размер выборки
            random_seed: зерно для генератора случайных чисел
            
        Returns:
            DataFrame: загруженные данные
        """
        print(f"Загрузка данных из {filename}...")
        file_path = self.source_dir / filename
        
        # Читаем только первую строку для получения заголовков
        headers = pd.read_csv(file_path, nrows=0)
        total_rows = sum(1 for _ in open(file_path)) - 1  # Вычитаем строку заголовка
        
        print(f"Всего записей в файле: {total_rows}")
        print(f"Выбираем случайную выборку размером {sample_size} записей...")
        
        # Устанавливаем seed для воспроизводимости
        np.random.seed(random_seed)
        
        # Генерируем случайные индексы для выборки
        skip_indices = sorted(np.random.choice(
            range(1, total_rows + 1), 
            total_rows - sample_size, 
            replace=False
        ))
        
        # Читаем данные, пропуская случайные строки
        self.raw_data = pd.read_csv(file_path, skiprows=skip_indices)
        
        print(f"Загружено {len(self.raw_data)} записей с {len(self.raw_data.columns)} признаками")
        return self.raw_data
    
    def compute_basic_statistics(self) -> Dict:
        """
        Расчет базовой статистики по числовым колонкам
        
        Returns:
            Dict: словарь со статистиками
        """
        if self.raw_data is None:
            raise ValueError("Сначала загрузите данные")
            
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        stats = {}
        
        # Прогресс-бар для колонок
        for col in tqdm(numeric_cols, desc="Расчет статистик по колонкам"):
            data = self.raw_data[col].values
            
            # Обработка данных батчами
            n_batches = len(data) // self.batch_size + (1 if len(data) % self.batch_size > 0 else 0)
            batch_stats = []
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(data))
                batch_data = data[start_idx:end_idx]
                
                mean, variance = knuth_mean_variance(batch_data)
                med = median(batch_data)
                quarts = quantiles(batch_data, [0.25, 0.75])
                
                batch_stats.append({
                    'mean': mean,
                    'variance': variance,
                    'median': med,
                    'q1': quarts[0],
                    'q3': quarts[1],
                    'size': len(batch_data)
                })
            
            # Объединение статистик батчей
            total_size = sum(s['size'] for s in batch_stats)
            mean = sum(s['mean'] * s['size'] for s in batch_stats) / total_size
            variance = sum((s['size'] - 1) * s['variance'] + s['size'] * (s['mean'] - mean)**2 
                         for s in batch_stats) / (total_size - 1)
            
            # Для медианы и квартилей используем полный набор данных
            med = median(data)
            quarts = quantiles(data, [0.25, 0.75])
            
            stats[col] = {
                'mean': mean,
                'std': np.sqrt(variance),
                'median': med,
                'q1': quarts[0],
                'q3': quarts[1]
            }
        
        self.statistics['basic'] = stats
        return stats
    
    def compute_correlations(self) -> pd.DataFrame:
        """
        Расчет корреляций между числовыми признаками
        
        Returns:
            DataFrame: матрица корреляций
        """
        if self.raw_data is None:
            raise ValueError("Сначала загрузите данные")
            
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        corr_matrix = np.zeros((n_cols, n_cols))
        
        # Прогресс-бар для пар признаков
        total_pairs = (n_cols * (n_cols + 1)) // 2
        with tqdm(total=total_pairs, desc="Расчет корреляций") as pbar:
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i <= j:
                        corr = correlation(
                            self.raw_data[col1].values,
                            self.raw_data[col2].values
                        )
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                        pbar.update(1)
        
        self.statistics['correlations'] = pd.DataFrame(
            corr_matrix,
            index=numeric_cols,
            columns=numeric_cols
        )
        return self.statistics['correlations']
    
    def sort_by_feature(self, feature: str) -> np.ndarray:
        """
        Сортировка данных по указанному признаку
        
        Args:
            feature: название признака
            
        Returns:
            array: отсортированные данные
        """
        if feature not in self.raw_data.columns:
            raise ValueError(f"Признак {feature} не найден в данных")
            
        print(f"\nСортировка по признаку {feature}...")
        data = self.raw_data.values
        col_idx = self.raw_data.columns.get_loc(feature)
        
        # Сортировка батчами
        n_batches = len(data) // self.batch_size + (1 if len(data) % self.batch_size > 0 else 0)
        sorted_batches = []
        
        for i in tqdm(range(n_batches), desc="Сортировка батчей"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(data))
            batch_data = data[start_idx:end_idx].copy()
            sorted_batches.append(sort_by_column(batch_data, col_idx))
        
        # Объединение отсортированных батчей
        print("Объединение отсортированных батчей...")
        result = np.vstack(sorted_batches)
        return sort_by_column(result, col_idx)
    
    def detect_outliers(self, feature: str, k: float = 1.5) -> Tuple[float, float]:
        """
        Определение выбросов методом межквартильного размаха
        
        Args:
            feature: название признака
            k: коэффициент для расчета границ (по умолчанию 1.5)
            
        Returns:
            Tuple[float, float]: нижняя и верхняя границы
        """
        if feature not in self.statistics['basic']:
            raise ValueError(f"Статистика для признака {feature} не рассчитана")
            
        stats = self.statistics['basic'][feature]
        iqr = stats['q3'] - stats['q1']
        lower_bound = stats['q1'] - k * iqr
        upper_bound = stats['q3'] + k * iqr
        
        return lower_bound, upper_bound
    
    def save_processed_data(self, filename: str):
        """
        Сохранение обработанных данных
        
        Args:
            filename: имя файла для сохранения
        """
        if self.processed_data is not None:
            save_path = Path('data/processed') / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"\nСохранение данных в {save_path}...")
            self.processed_data.to_csv(save_path, index=False) 