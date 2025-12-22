"""
Urgency Classificator Module
Provides fine-tuning, prediction, and explainability for urgency classification.
"""
from .model import UrgencyClassifier
from .predictor import UrgencyPredictor

__all__ = ['UrgencyClassifier', 'UrgencyPredictor']
