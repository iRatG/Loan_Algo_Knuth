import numpy as np
from typing import Optional, Union
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score

class KnuthMetrics:
    """
    Метрики для оценки качества модели градиентного бустинга на основе алгоритмов Кнута.
    """
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Среднеквадратичная ошибка.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            
        Returns:
            float: Значение метрики
        """
        return mean_squared_error(y_true, y_pred)
        
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Корень из среднеквадратичной ошибки.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            
        Returns:
            float: Значение метрики
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
        
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Средняя абсолютная ошибка.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            
        Returns:
            float: Значение метрики
        """
        return np.mean(np.abs(y_true - y_pred))
        
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Точность классификации.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            
        Returns:
            float: Значение метрики
        """
        return accuracy_score(y_true, y_pred.round())
        
    @staticmethod
    def auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Площадь под ROC-кривой.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            
        Returns:
            float: Значение метрики
        """
        return roc_auc_score(y_true, y_pred)
        
    @staticmethod
    def custom_metric(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Пользовательская метрика с возможностью взвешивания.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            weights: Веса для каждого наблюдения
            
        Returns:
            float: Значение метрики
        """
        if weights is None:
            weights = np.ones_like(y_true)
            
        errors = np.abs(y_true - y_pred)
        weighted_errors = errors * weights
        return np.mean(weighted_errors)
        
    @staticmethod
    def all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = 'regression'
    ) -> dict:
        """
        Расчет всех доступных метрик.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            task_type: Тип задачи ('regression' или 'classification')
            
        Returns:
            dict: Словарь со значениями всех метрик
        """
        metrics = {
            'mse': KnuthMetrics.mse(y_true, y_pred),
            'rmse': KnuthMetrics.rmse(y_true, y_pred),
            'mae': KnuthMetrics.mae(y_true, y_pred)
        }
        
        if task_type == 'classification':
            metrics.update({
                'accuracy': KnuthMetrics.accuracy(y_true, y_pred),
                'auc_roc': KnuthMetrics.auc_roc(y_true, y_pred)
            })
            
        return metrics 