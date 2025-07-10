"""
Оптимизатор гиперпараметров для градиентного бустинга на основе алгоритмов Кнута.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.model_selection import KFold
from tqdm import tqdm
from .boosting import KnuthGradientBoosting
from .metrics import KnuthMetrics

class KnuthOptimizer:
    """
    Оптимизатор гиперпараметров для градиентного бустинга на основе алгоритмов Кнута.
    """
    
    def __init__(
        self,
        param_grid: Dict[str, List[Union[int, float, str]]],
        n_folds: int = 5,
        scoring: str = 'rmse',
        random_state: Optional[int] = None
    ):
        """
        Инициализация оптимизатора.
        """
        self.param_grid = param_grid
        self.n_folds = n_folds
        self.scoring = scoring
        self.random_state = random_state
        
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []
        self.current_fold = 0
        self.total_combinations = self._count_combinations()
        
    def _count_combinations(self) -> int:
        """
        Подсчет общего количества комбинаций параметров
        """
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total
        
    def _evaluate_params(
        self,
        params: Dict[str, Union[int, float, str]],
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Оценка параметров с помощью кросс-валидации
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        scores = []
        metric_func = self._get_metric_function()
        
        print(f"\nОценка параметров (комбинация {len(self.cv_results_) + 1}/{self.total_combinations}):")
        for param, value in params.items():
            print(f"{param}: {value}")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"\nФолд {fold}/{self.n_folds}")
            
            model = KnuthGradientBoosting(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            score = metric_func(y_val, y_pred)
            scores.append(score)
            
            print(f"Метрика {self.scoring}: {score:.4f}")
            
            # Если это лучший результат, сохраняем
            if self.best_score_ is None or self._is_better_score(score, self.best_score_):
                print("★ Новый лучший результат!")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\nСредний {self.scoring}: {mean_score:.4f} ± {std_score:.4f}")
        return mean_score, std_score
        
    def _is_better_score(self, new_score: float, best_score: float) -> bool:
        """
        Проверка, является ли новый результат лучше текущего лучшего
        """
        if self.scoring in ['mse', 'rmse', 'mae']:
            return new_score < best_score
        return new_score > best_score
        
    def _get_metric_function(self) -> Callable:
        """
        Получение функции метрики по имени
        """
        metric_map = {
            'mse': KnuthMetrics.mse,
            'rmse': KnuthMetrics.rmse,
            'mae': KnuthMetrics.mae,
            'accuracy': KnuthMetrics.accuracy,
            'auc_roc': KnuthMetrics.auc_roc
        }
        return metric_map.get(self.scoring, KnuthMetrics.rmse)
        
    def _generate_param_combinations(self) -> List[Dict[str, Union[int, float, str]]]:
        """
        Генерация всех возможных комбинаций параметров
        """
        from itertools import product
        
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KnuthOptimizer':
        """
        Поиск оптимальных параметров
        """
        param_combinations = self._generate_param_combinations()
        best_score = float('inf') if self.scoring in ['mse', 'rmse', 'mae'] else float('-inf')
        
        print(f"\nВсего комбинаций параметров: {self.total_combinations}")
        print(f"Количество фолдов: {self.n_folds}")
        print(f"Метрика оптимизации: {self.scoring}")
        
        for params in tqdm(param_combinations, desc="Перебор параметров"):
            mean_score, std_score = self._evaluate_params(params, X, y)
            
            self.cv_results_.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score
            })
            
            is_better = self._is_better_score(mean_score, best_score)
            
            if is_better:
                best_score = mean_score
                self.best_params_ = params
                self.best_score_ = mean_score
                print("\n★ Обновлены лучшие параметры:")
                for param, value in params.items():
                    print(f"{param}: {value}")
                print(f"Новый лучший {self.scoring}: {mean_score:.4f}")
                
        return self
        
    def get_best_model(self) -> KnuthGradientBoosting:
        """
        Получение модели с оптимальными параметрами
        """
        if self.best_params_ is None:
            raise ValueError("Оптимизатор еще не обучен. Сначала вызовите метод fit()")
            
        return KnuthGradientBoosting(**self.best_params_) 