"""
Urgency Prediction Module
Handles predictions with explainability using gradient-based methods.
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class UrgencyPredictor:
    """
    Urgency predictor with explainability support.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to fine-tuned model
            model_name: Base model name if no fine-tuned model available
        """
        self.urgency_labels = ["low", "medium", "high", "critical"]
        
        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Use base model as fallback
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.urgency_labels)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model.eval()
        
    def predict(self, text: str, explain: bool = False) -> Dict[str, Any]:
        """
        Predict urgency level from text.
        
        Args:
            text: Input text to classify
            explain: Whether to include explainability information
            
        Returns:
            Dictionary with prediction and optional explanation
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = {
            "urgency": self.urgency_labels[predicted_class],
            "confidence": float(confidence),
            "all_scores": {
                label: float(prob) for label, prob in zip(self.urgency_labels, probabilities[0])
            }
        }
        
        if explain:
            result["explanation"] = self._get_explanation(text, inputs)
        
        return result
    
    def _get_explanation(self, text: str, inputs: Dict) -> Dict[str, Any]:
        """
        Generate explanation for prediction using gradient-based approach.
        
        Args:
            text: Original text
            inputs: Tokenized inputs
            
        Returns:
            Explanation dictionary with token importance
        """
        # Enable gradients temporarily for explanation
        self.model.train()  # Set to train mode for gradients
        
        try:
            # Simple gradient-based explanation
            inputs_with_grad = {k: v.clone().detach().requires_grad_(True) for k, v in inputs.items() if k == 'input_ids'}
            
            # Get model output
            outputs = self.model(**{**inputs, **inputs_with_grad})
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            
            # Compute gradients
            outputs.logits[0, predicted_class].backward()
            
            # Get token importance
            token_ids = inputs['input_ids'][0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Use gradient magnitude as importance score
            if 'input_ids' in inputs_with_grad:
                gradients = inputs_with_grad['input_ids'].grad
                if gradients is not None:
                    importance = gradients.abs().squeeze().tolist()
                    if not isinstance(importance, list):
                        importance = [importance]
                else:
                    importance = [0.0] * len(tokens)
            else:
                importance = [0.0] * len(tokens)
            
            # Filter out special tokens and create word-level importance
            token_importance = []
            for token, score in zip(tokens, importance):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    token_importance.append({
                        "token": token,
                        "importance": float(score)
                    })
            
            return {
                "method": "gradient_based",
                "token_importance": token_importance[:20],  # Top 20 tokens
                "description": "Token importance based on gradient magnitudes"
            }
        finally:
            # Clean up and set back to eval mode
            self.model.zero_grad()
            self.model.eval()
    
    def batch_predict(self, texts: List[str], explain: bool = False) -> List[Dict[str, Any]]:
        """
        Predict urgency levels for multiple texts using efficient batch processing.
        
        Args:
            texts: List of input texts
            explain: Whether to include explanations
            
        Returns:
            List of prediction dictionaries
        """
        if not texts:
            return []
        
        # For explanations, process individually since gradients are per-sample
        if explain:
            return [self.predict(text, explain) for text in texts]
        
        # Batch tokenization for efficiency
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Get predictions for all texts
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1).tolist()
            confidences = torch.max(probabilities, dim=-1).values.tolist()
        
        # Format results
        results = []
        for i, (pred_class, confidence) in enumerate(zip(predicted_classes, confidences)):
            result = {
                "urgency": self.urgency_labels[pred_class],
                "confidence": float(confidence),
                "all_scores": {
                    label: float(prob) for label, prob in zip(self.urgency_labels, probabilities[i])
                }
            }
            results.append(result)
        
        return results
