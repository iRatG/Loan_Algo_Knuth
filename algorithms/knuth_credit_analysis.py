"""
Реализация алгоритмов Кнута для анализа кредитных данных

Основные алгоритмы:
1. Трехпутевая быстрая сортировка (Vol 3, 5.2.2)
2. Поиск медианы и квантилей (Vol 3, 5.3.3)
3. Инкрементальные статистики (Vol 2, 4.2.2)
4. Алгоритм выбора k-го элемента (Vol 3, 5.3.3)
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
from visualization.knuth_plots import (
    plot_feature_distribution,
    plot_feature_boxplot,
    plot_risk_score_distribution,
    plot_feature_importance
)

class KnuthCreditAnalysis:
    def __init__(self):
        self.data = None
        self.sorted_indices = {}
        self.statistics = defaultdict(dict)
        
    def load_data(self, file_path: str) -> np.ndarray:
        """Загрузка данных с использованием numpy для эффективности"""
        self.data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        return self.data
    
    def quicksort3way(self, arr: np.ndarray, column: int) -> np.ndarray:
        """
        Трехпутевая быстрая сортировка Кнута (Vol 3, 5.2.2)
        Особенно эффективна для данных с повторениями
        """
        def partition3(arr: np.ndarray, low: int, high: int) -> Tuple[int, int]:
            pivot = arr[high, column]
            lt = low  # элементы < pivot
            gt = high # элементы > pivot
            i = low   # текущий элемент
            
            while i <= gt:
                if arr[i, column] < pivot:
                    arr[[lt, i]] = arr[[i, lt]]
                    lt += 1
                    i += 1
                elif arr[i, column] > pivot:
                    arr[[i, gt]] = arr[[gt, i]]
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
            
        quicksort3_recursive(arr, 0, len(arr) - 1)
        return arr
    
    def select_kth(self, arr: np.ndarray, column: int, k: int) -> float:
        """
        Итеративный алгоритм выбора k-го элемента (Vol 3, 5.3.3)
        Используется для поиска медианы и квантилей
        """
        arr = arr.copy()
        left = 0
        right = len(arr) - 1
        
        while True:
            if left == right:
                return arr[left, column]
            
            # Выбираем опорный элемент
            pivot_idx = (left + right) // 2
            pivot = arr[pivot_idx, column]
            
            # Разделяем массив
            arr[[pivot_idx, right]] = arr[[right, pivot_idx]]
            store_idx = left
            
            for i in range(left, right):
                if arr[i, column] < pivot:
                    arr[[store_idx, i]] = arr[[i, store_idx]]
                    store_idx += 1
            
            arr[[right, store_idx]] = arr[[store_idx, right]]
            
            # Определяем, в какой части искать k-й элемент
            if k < store_idx:
                right = store_idx - 1
            elif k > store_idx:
                left = store_idx + 1
            else:
                return arr[store_idx, column]
    
    def compute_incremental_stats(self, column: int) -> Dict:
        """
        Инкрементальные статистики Кнута (Vol 2, 4.2.2)
        Вычисляет среднее, дисперсию и другие статистики за один проход
        """
        n = 0
        mean = 0
        M2 = 0  # Для вычисления дисперсии
        min_val = float('inf')
        max_val = float('-inf')
        
        for x in self.data[:, column]:
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)
            min_val = min(min_val, x)
            max_val = max(max_val, x)
        
        variance = M2 / (n - 1) if n > 1 else 0
        std_dev = np.sqrt(variance)
        
        return {
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev,
            'min': min_val,
            'max': max_val,
            'n': n
        }
    
    def analyze_feature_target_relationship(self, feature_col: int, feature_name: str) -> dict:
        """
        Анализ связи признака с целевой переменной (дефолт) используя алгоритмы Кнута
        
        Целевая переменная находится в последней колонке (default.payment.next.month)
        """
        target_col = -1  # Последняя колонка - целевая переменная
        
        # Разделяем данные по классам целевой переменной
        default_mask = self.data[:, target_col] == 1
        non_default_mask = ~default_mask
        
        default_values = self.data[default_mask, feature_col]
        non_default_values = self.data[non_default_mask, feature_col]
        
        # Статистики для каждого класса
        stats = {
            'default': self.compute_incremental_stats_for_array(default_values),
            'non_default': self.compute_incremental_stats_for_array(non_default_values),
            'separation_power': {}
        }
        
        # Оценка разделяющей способности признака
        # Используем модифицированный алгоритм Кнута для подсчета пересечений распределений
        sorted_all = np.sort(self.data[:, feature_col])
        sorted_default = np.sort(default_values)
        sorted_non_default = np.sort(non_default_values)
        
        # Находим точки пересечения распределений (используя квантили)
        q_points = [0.25, 0.5, 0.75]
        default_quantiles = [sorted_default[int(q * len(sorted_default))] for q in q_points]
        non_default_quantiles = [sorted_non_default[int(q * len(sorted_non_default))] for q in q_points]
        
        # Оценка разделяющей способности через отношение средних и дисперсий
        mean_ratio = abs(stats['default']['mean'] - stats['non_default']['mean']) / \
                    max(stats['default']['std_dev'], stats['non_default']['std_dev'])
                    
        # Подсчет процента пересечения распределений
        overlap = 0
        total_points = len(sorted_all)
        window_size = total_points // 50  # 2% от данных
        
        for i in range(0, total_points - window_size, window_size):
            window = sorted_all[i:i + window_size]
            default_count = np.sum(np.isin(window, sorted_default))
            non_default_count = np.sum(np.isin(window, sorted_non_default))
            
            if default_count > 0 and non_default_count > 0:
                overlap += 1
        
        overlap_ratio = overlap / (total_points // window_size)
        
        stats['separation_power'] = {
            'mean_ratio': mean_ratio,
            'overlap_ratio': overlap_ratio,
            'separation_score': (1 - overlap_ratio) * mean_ratio
        }
        
        return stats
    
    def compute_incremental_stats_for_array(self, arr: np.ndarray) -> dict:
        """Вычисление статистик для одномерного массива"""
        n = 0
        mean = 0
        M2 = 0
        min_val = float('inf')
        max_val = float('-inf')
        
        for x in arr:
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)
            min_val = min(min_val, x)
            max_val = max(max_val, x)
        
        variance = M2 / (n - 1) if n > 1 else 0
        std_dev = np.sqrt(variance)
        
        return {
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev,
            'min': min_val,
            'max': max_val,
            'n': n
        }
    
    def analyze_feature(self, column: int, feature_name: str) -> Dict:
        """Полный анализ признака с использованием алгоритмов Кнута"""
        # Базовый анализ
        stats = super().analyze_feature(column, feature_name)
        
        # Добавляем анализ связи с целевой переменной
        target_stats = self.analyze_feature_target_relationship(column, feature_name)
        stats['target_relationship'] = target_stats
        
        return stats
    
    def analyze_credit_risk(self) -> Dict:
        """Анализ кредитного риска с использованием алгоритмов Кнута"""
        # Анализируем ключевые признаки
        key_features = {
            'LIMIT_BAL': 1,      # Кредитный лимит
            'PAY_0': 6,          # Текущий статус платежа
            'BILL_AMT1': 12,     # Текущий счет
            'PAY_AMT1': 18,      # Текущий платеж
            'AGE': 5             # Возраст
        }
        
        results = {}
        for feature_name, col_idx in key_features.items():
            print(f"\nАнализ признака {feature_name}...")
            stats = self.analyze_feature(col_idx, feature_name)
            results[feature_name] = stats
            
            print(f"Среднее: {stats['mean']:.2f}")
            print(f"Медиана: {stats['median']:.2f}")
            print(f"Стандартное отклонение: {stats['std_dev']:.2f}")
            print(f"Границы выбросов: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
            
            # Добавляем визуализацию
            plot_feature_distribution(self.data, col_idx, feature_name, stats)
            plot_feature_boxplot(self.data, col_idx, feature_name)
        
        # Визуализация весов признаков
        plot_feature_importance(results, self.get_feature_weights())
        
        return results
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Получение весов признаков для риск-скора"""
        return {
            'LIMIT_BAL': 0.3,    # Больший вес для кредитного лимита
            'PAY_0': 0.25,       # Важность текущего статуса платежа
            'BILL_AMT1': 0.2,    # Текущий счет
            'PAY_AMT1': 0.15,    # Сумма платежа
            'AGE': 0.1           # Возраст имеет меньший вес
        }
    
    def analyze_all_clients(self) -> List[float]:
        """Расчет риск-скоров для всех клиентов"""
        all_scores = []
        for client_data in self.data:
            score = self.get_risk_score(client_data)
            all_scores.append(score)
        return all_scores

    def get_risk_score(self, client_data: np.ndarray) -> float:
        """
        Расчет скоринга на основе статистик Кнута
        Использует относительное положение клиента в распределении признаков
        """
        if not self.statistics:
            raise ValueError("Сначала необходимо провести анализ данных")
        
        risk_score = 0
        weights = self.get_feature_weights() # Используем веса из get_feature_weights
        
        for feature_name, weight in weights.items():
            stats = self.statistics[feature_name]
            value = client_data[list(self.statistics.keys()).index(feature_name)]
            
            # Нормализация значения относительно статистик
            if stats['std_dev'] > 0:
                z_score = (value - stats['mean']) / stats['std_dev']
                # Преобразование z-score в риск-скор (0-1)
                feature_score = 1 / (1 + np.exp(-z_score))
                risk_score += weight * feature_score
        
        return risk_score

def main():
    # Инициализация анализатора
    analyzer = KnuthCreditAnalysis()
    
    # Загрузка данных
    print("Загрузка данных...")
    data = analyzer.load_data("source/UCI_Credit_Card.csv")
    
    # Анализ данных
    print("\nПроведение анализа с использованием алгоритмов Кнута...")
    results = analyzer.analyze_credit_risk()
    
    # Расчет и визуализация риск-скоров для всех клиентов
    print("\nРасчет риск-скоров для всех клиентов...")
    all_scores = analyzer.analyze_all_clients()
    plot_risk_score_distribution(all_scores)
    
    # Пример расчета риск-скора для клиента
    print("\nПример расчета риск-скора для случайного клиента:")
    sample_client = data[0]  # Берем первого клиента для примера
    risk_score = analyzer.get_risk_score(sample_client)
    print(f"Риск-скор: {risk_score:.2f}")
    print(f"Рекомендация: {'Высокий риск' if risk_score > 0.7 else 'Средний риск' if risk_score > 0.4 else 'Низкий риск'}")
    
    print("\nВизуализации сохранены в директории visualization/figures/")

if __name__ == "__main__":
    main() 