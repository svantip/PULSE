"""
Emotion Classificator Module
Provides fine-tuning, prediction, and explainability for emotion classification.
"""
from .model import EmotionClassifier
from .predictor import EmotionPredictor

__all__ = ['EmotionClassifier', 'EmotionPredictor']
