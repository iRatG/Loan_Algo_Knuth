"""
Реализация градиентного бустинга на основе алгоритмов Кнута.
Основано на принципах из книги "Искусство программирования".
"""

from .trees import KnuthDecisionTree, Node
from .boosting import KnuthGradientBoosting
from .preprocessing import KnuthPreprocessor
from .metrics import KnuthMetrics
from .optimization import KnuthOptimizer

__all__ = [
    'KnuthDecisionTree',
    'Node',
    'KnuthGradientBoosting',
    'KnuthPreprocessor',
    'KnuthMetrics',
    'KnuthOptimizer'
]

__version__ = '0.1.0' 