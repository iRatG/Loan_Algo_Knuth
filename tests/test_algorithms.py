"""
Тестирование алгоритмов кредитного скоринга
"""

import numpy as np
import unittest
from algorithms.knuth_sort import KnuthSort
from algorithms.knuth_statistics import KnuthStatistics
from algorithms.knuth_search import KnuthSearch
from algorithms.knuth_metrics import KnuthMetrics
from models.credit_model import KnuthCreditModel

class TestKnuthAlgorithms(unittest.TestCase):
    def setUp(self):
        """Подготовка тестовых данных"""
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 5
        
        # Генерируем синтетические данные
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.binomial(1, 0.3, self.n_samples)
        
        # Инициализируем классы
        self.sorter = KnuthSort()
        self.stats = KnuthStatistics()
        self.search = KnuthSearch()
        self.metrics = KnuthMetrics()
        self.model = KnuthCreditModel(n_bins=10)
        
    def test_quicksort_3way(self):
        """Тест трехпутевой быстрой сортировки"""
        # Тестируем на одном столбце
        column = 0
        sorted_data = self.sorter.quicksort_3way(self.X, column)
        
        # Проверяем, что данные отсортированы
        self.assertTrue(np.all(np.diff(sorted_data[:, column]) >= 0))
        
        # Проверяем, что все элементы сохранились
        self.assertEqual(len(sorted_data), len(self.X))
        
    def test_statistics(self):
        """Тест статистических алгоритмов"""
        # Обновляем статистики
        self.stats.update(self.X, [0])
        stats = self.stats.get_stats(0)
        
        # Проверяем основные статистики
        self.assertIn('mean', stats)
        self.assertIn('variance', stats)
        self.assertIn('std', stats)
        
        # Сравниваем с numpy
        np_mean = np.mean(self.X[:, 0])
        self.assertAlmostEqual(stats['mean'], np_mean, places=5)
        
    def test_search(self):
        """Тест алгоритмов поиска"""
        # Сортируем данные
        column = 0
        sorted_data = self.sorter.quicksort_3way(self.X, column)
        
        # Ищем медиану
        median_idx = len(self.X) // 2
        median = self.search.select_kth(self.X, median_idx, column)
        
        # Проверяем, что это действительно медиана
        n_less = np.sum(self.X[:, column] <= median)
        self.assertTrue(abs(n_less - median_idx) <= 1)
        
    def test_metrics(self):
        """Тест метрик качества"""
        # Генерируем предсказания
        y_pred = np.random.binomial(1, 0.3, self.n_samples)
        y_score = np.random.rand(self.n_samples)
        
        # Проверяем основные метрики
        metrics = self.metrics.precision_recall(self.y, y_pred)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # Проверяем AUC-ROC
        auc = self.metrics.auc_roc_knuth(self.y, y_score)
        self.assertTrue(0 <= auc <= 1)
        
    def test_credit_model(self):
        """Тест модели кредитного скоринга"""
        # Обучаем модель
        self.model.fit(self.X, self.y)
        
        # Получаем предсказания
        y_pred = self.model.predict(self.X)
        y_proba = self.model.predict_proba(self.X)
        
        # Проверяем формат предсказаний
        self.assertEqual(len(y_pred), len(self.X))
        self.assertTrue(np.all((y_pred == 0) | (y_pred == 1)))
        self.assertTrue(np.all((y_proba >= 0) & (y_proba <= 1)))
        
        # Проверяем метрики
        metrics = self.model.evaluate(self.X, self.y)
        self.assertIn('precision_recall', metrics)
        self.assertIn('auc_roc', metrics)
        self.assertIn('gini', metrics)
        self.assertIn('ks_stat', metrics)

if __name__ == '__main__':
    unittest.main() 