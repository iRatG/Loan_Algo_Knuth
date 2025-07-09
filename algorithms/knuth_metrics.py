"""
Реализация метрик и функций потерь на основе алгоритмов Кнута
The Art of Computer Programming, Volume 2: Seminumerical Algorithms

Основные алгоритмы:
1. Инкрементальное вычисление метрик (Vol 2, 4.2.2)
2. Оценка точности и полноты (Vol 2, 4.5.1)
3. Расчет AUC-ROC (Vol 2, 4.5.3)
"""

import numpy as np
from typing import Dict, List, Tuple
from .knuth_sort import KnuthSort
from .knuth_statistics import KnuthStatistics

class KnuthMetrics:
    def __init__(self):
        self.sorter = KnuthSort()
        self.stats = KnuthStatistics()
        
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """
        Вычисление матрицы ошибок с использованием алгоритмов Кнута
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            
        Returns:
            Словарь с элементами матрицы ошибок
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Размеры массивов не совпадают")
            
        tp = fp = tn = fn = 0
        
        for true, pred in zip(y_true, y_pred):
            if true == 1:
                if pred == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == 1:
                    fp += 1
                else:
                    tn += 1
                    
        return {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
    
    def precision_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Вычисление точности и полноты
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            
        Returns:
            Словарь с метриками
        """
        cm = self.confusion_matrix(y_true, y_pred)
        
        precision = cm['tp'] / (cm['tp'] + cm['fp']) if (cm['tp'] + cm['fp']) > 0 else 0
        recall = cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def auc_roc_knuth(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Вычисление AUC-ROC по алгоритму Кнута
        
        Args:
            y_true: Истинные метки
            y_score: Предсказанные вероятности
            
        Returns:
            Значение AUC-ROC
        """
        if len(y_true) != len(y_score):
            raise ValueError("Размеры массивов не совпадают")
            
        # Сортируем по убыванию вероятностей
        indices = np.argsort(-y_score)
        y_true = y_true[indices]
        
        # Считаем ROC-кривую
        tp = 0
        fp = 0
        n_pos = np.sum(y_true == 1)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
            
        # Площадь под ROC-кривой
        auc = 0
        prev_score = float('inf')
        
        for i, label in enumerate(y_true):
            if label == 1:
                tp += 1
            else:
                fp += 1
                # Обновляем AUC
                auc += tp
                
        # Нормализуем
        auc = auc / (n_pos * n_neg)
        
        return auc
    
    def gini_coefficient(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Вычисление коэффициента Джини
        
        Args:
            y_true: Истинные метки
            y_score: Предсказанные вероятности
            
        Returns:
            Коэффициент Джини
        """
        # Сортируем по убыванию вероятностей
        indices = np.argsort(-y_score)
        y_true = y_true[indices]
        
        n = len(y_true)
        n_pos = np.sum(y_true == 1)
        
        if n_pos == 0 or n_pos == n:
            return 0
            
        # Вычисляем кривую Лоренца
        lorenz = np.cumsum(y_true) / n_pos
        
        # Площадь под кривой Лоренца
        area = np.sum(lorenz) / n
        
        # Коэффициент Джини
        gini = 2 * area - 1
        
        return gini
    
    def kolmogorov_smirnov(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Вычисление статистики Колмогорова-Смирнова
        
        Args:
            y_true: Истинные метки
            y_score: Предсказанные вероятности
            
        Returns:
            Значение статистики KS
        """
        # Сортируем по возрастанию вероятностей
        indices = np.argsort(y_score)
        y_true = y_true[indices]
        
        n = len(y_true)
        n_pos = np.sum(y_true == 1)
        n_neg = n - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0
            
        # Считаем кумулятивные распределения
        cum_pos = np.cumsum(y_true == 1) / n_pos
        cum_neg = np.cumsum(y_true == 0) / n_neg
        
        # Максимальная разница
        ks_stat = np.max(np.abs(cum_pos - cum_neg))
        
        return ks_stat
    
    def cross_validate_knuth(self, X: np.ndarray, y: np.ndarray, 
                           n_folds: int = 5) -> List[Dict[str, float]]:
        """
        Кросс-валидация с использованием алгоритмов Кнута
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            n_folds: Количество фолдов
            
        Returns:
            Список словарей с метриками для каждого фолда
        """
        if len(X) != len(y):
            raise ValueError("Размеры X и y не совпадают")
            
        # Перемешиваем данные
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Разбиваем на фолды
        fold_size = len(X) // n_folds
        results = []
        
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else len(X)
            
            # Формируем тестовую и обучающую выборки
            X_test = X[start:end]
            y_test = y[start:end]
            X_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])
            
            # Обучаем модель и получаем предсказания
            # Здесь должна быть ваша модель
            y_pred = np.zeros_like(y_test)  # Заглушка
            
            # Считаем метрики
            metrics = {
                'fold': i + 1,
                'precision_recall': self.precision_recall(y_test, y_pred),
                'auc_roc': self.auc_roc_knuth(y_test, y_pred),
                'gini': self.gini_coefficient(y_test, y_pred),
                'ks_stat': self.kolmogorov_smirnov(y_test, y_pred)
            }
            
            results.append(metrics)
            
        return results 