"""
Реализация модели кредитного скоринга на основе алгоритмов Кнута
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from algorithms.knuth_sort import KnuthSort
from algorithms.knuth_statistics import KnuthStatistics
from algorithms.knuth_search import KnuthSearch
from algorithms.knuth_metrics import KnuthMetrics

class KnuthCreditModel:
    def __init__(self, n_bins: int = 10):
        """
        Инициализация модели
        
        Args:
            n_bins: Количество бинов для дискретизации
        """
        self.n_bins = n_bins
        self.sorter = KnuthSort()
        self.stats = KnuthStatistics()
        self.search = KnuthSearch()
        self.metrics = KnuthMetrics()
        
        self.feature_bins = {}
        self.feature_weights = {}
        self.feature_woe = {}
        self.iv_scores = {}
        self.feature_stats = {}
        
    def _normalize_feature(self, data: np.ndarray, column: int) -> np.ndarray:
        """
        Нормализация признака
        
        Args:
            data: Данные
            column: Номер колонки
            
        Returns:
            Нормализованные данные
        """
        stats = self.stats.get_stats(column)
        if stats is None:
            self.stats.update(data, [column])
            stats = self.stats.get_stats(column)
            
        mean = stats['mean']
        std = stats['std']
        
        if std == 0:
            return data - mean
        return (data - mean) / std
        
    def _calculate_woe_iv(self, data: np.ndarray, column: int) -> Tuple[Dict, float]:
        """
        Расчет WoE (Weight of Evidence) и IV (Information Value)
        
        Args:
            data: Данные
            column: Номер колонки
            
        Returns:
            (словарь с WoE для бинов, значение IV)
        """
        # Нормализуем данные
        norm_data = self._normalize_feature(data[:, column], column)
        
        # Сортируем нормализованные данные
        sorted_indices = np.argsort(norm_data)
        sorted_data = norm_data[sorted_indices]
        sorted_target = data[sorted_indices, -1]
        
        # Находим границы бинов по квантилям
        n_samples = len(data)
        samples_per_bin = n_samples // self.n_bins
        
        # Считаем WoE и IV для каждого бина
        woe_dict = {}
        iv_total = 0
        
        total_good = np.sum(data[:, -1] == 0)
        total_bad = np.sum(data[:, -1] == 1)
        
        for i in range(self.n_bins):
            start_idx = i * samples_per_bin
            end_idx = (i + 1) * samples_per_bin if i < self.n_bins - 1 else n_samples
            
            bin_target = sorted_target[start_idx:end_idx]
            
            # Считаем доли хороших и плохих
            n_good = np.sum(bin_target == 0)
            n_bad = np.sum(bin_target == 1)
            
            # Добавляем сглаживание Лапласа
            n_good += 1
            n_bad += 1
            
            p_good = n_good / (total_good + self.n_bins)
            p_bad = n_bad / (total_bad + self.n_bins)
            
            # Считаем WoE
            woe = np.log(p_good / p_bad)
            woe_dict[i] = {
                'lower_bound': sorted_data[start_idx] if start_idx > 0 else float('-inf'),
                'upper_bound': sorted_data[end_idx-1] if end_idx < n_samples else float('inf'),
                'woe': woe,
                'n_samples': end_idx - start_idx,
                'p_good': p_good,
                'p_bad': p_bad
            }
            
            # Считаем IV
            iv = (p_good - p_bad) * woe
            iv_total += iv if not np.isnan(iv) else 0
            
        return woe_dict, abs(iv_total)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Обучение модели
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
        """
        # Балансируем классы
        n_samples = len(y)
        n_pos = np.sum(y == 1)
        n_neg = n_samples - n_pos
        
        # Вычисляем веса классов
        self.class_weight = {
            0: 1.0,
            1: n_neg / n_pos if n_pos > 0 else 1.0
        }
        
        # Добавляем целевую переменную к данным
        data = np.column_stack([X, y])
        
        # Для каждого признака считаем WoE и IV
        n_features = X.shape[1]
        
        for col in range(n_features):
            # Обновляем статистики
            self.stats.update(X, [col])
            self.feature_stats[col] = self.stats.get_stats(col)
            
            # Считаем WoE и IV
            woe_dict, iv = self._calculate_woe_iv(data, col)
            
            self.feature_bins[col] = woe_dict
            self.iv_scores[col] = iv
        
        # Отбираем признаки с высоким IV (> 0.02)
        selected_features = {
            col: iv for col, iv in self.iv_scores.items()
            if iv > 0.02
        }
        
        if len(selected_features) == 0:
            # Если нет признаков с высоким IV, берем топ-5
            selected_features = dict(
                sorted(self.iv_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            )
        
        # Обновляем веса только для выбранных признаков
        total_iv = sum(selected_features.values())
        self.feature_weights = {
            col: iv / total_iv
            for col, iv in selected_features.items()
        }
        
        # Для неиспользуемых признаков ставим вес 0
        for col in range(n_features):
            if col not in self.feature_weights:
                self.feature_weights[col] = 0.0
        
        # Находим оптимальный порог на обучающей выборке
        y_proba = self.predict_proba(X)
        self.threshold = self._find_optimal_threshold(y, y_proba)
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Поиск оптимального порога для максимизации F1-меры
        
        Args:
            y_true: Истинные метки
            y_proba: Предсказанные вероятности
            
        Returns:
            Оптимальный порог
        """
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            
            # Считаем метрики с учетом весов классов
            tp = np.sum((y_true == 1) & (y_pred == 1)) * self.class_weight[1]
            fp = np.sum((y_true == 0) & (y_pred == 1)) * self.class_weight[0]
            fn = np.sum((y_true == 1) & (y_pred == 0)) * self.class_weight[1]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        return best_threshold
    
    def _get_bin_woe(self, value: float, column: int) -> float:
        """
        Получение WoE для значения признака
        
        Args:
            value: Значение признака
            column: Номер колонки
            
        Returns:
            Значение WoE
        """
        # Нормализуем значение
        norm_value = self._normalize_feature(np.array([value]), column)[0]
        
        bins = self.feature_bins[column]
        
        for bin_info in bins.values():
            if norm_value > bin_info['lower_bound'] and norm_value <= bin_info['upper_bound']:
                return bin_info['woe']
                
        # Если значение не попало ни в один бин, используем ближайший
        if norm_value <= bins[0]['upper_bound']:
            return bins[0]['woe']
        return bins[self.n_bins - 1]['woe']
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятности дефолта
        
        Args:
            X: Матрица признаков
            
        Returns:
            Массив вероятностей
        """
        n_samples = len(X)
        scores = np.zeros(n_samples)
        
        # Для каждого примера
        for i in range(n_samples):
            score = 0
            
            # Для каждого признака
            for col in range(X.shape[1]):
                # Получаем WoE для значения
                woe = self._get_bin_woe(X[i, col], col)
                
                # Взвешиваем WoE по важности признака
                score += woe * self.feature_weights[col]
                
            # Преобразуем в вероятность через сигмоиду
            scores[i] = 1 / (1 + np.exp(-score))
            
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание класса
        
        Args:
            X: Матрица признаков
            
        Returns:
            Массив предсказанных классов
        """
        probas = self.predict_proba(X)
        return (probas > self.threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Оценка качества модели
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Словарь с метриками
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Считаем метрики
        tp = np.sum((y == 1) & (y_pred == 1))
        fp = np.sum((y == 0) & (y_pred == 1))
        tn = np.sum((y == 0) & (y_pred == 0))
        fn = np.sum((y == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC-AUC
        sorted_indices = np.argsort(-y_proba)
        sorted_y = y[sorted_indices]
        
        n_pos = np.sum(y == 1)
        n_neg = len(y) - n_pos
        
        tp_cum = np.cumsum(sorted_y == 1)
        fp_cum = np.cumsum(sorted_y == 0)
        
        tpr = tp_cum / n_pos
        fpr = fp_cum / n_neg
        
        auc_roc = np.trapz(tpr, fpr)
        
        # Коэффициент Джини
        gini = 2 * auc_roc - 1
        
        # Статистика Колмогорова-Смирнова
        ks_stat = np.max(np.abs(tpr - fpr))
        
        return {
            'precision_recall': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'auc_roc': auc_roc,
            'gini': gini,
            'ks_stat': ks_stat
        }
    
    def get_feature_importance(self) -> Dict[int, Dict]:
        """
        Получение важности признаков
        
        Returns:
            Словарь с важностью признаков
        """
        importance = {}
        for col, weight in self.feature_weights.items():
            importance[col] = {
                'weight': weight,
                'iv': self.iv_scores[col],
                'n_bins': len(self.feature_bins[col])
            }
        return importance 