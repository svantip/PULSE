"""
Emotion Predictor with PEFT support
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class EmotionPredictor:
    """Emotion classification predictor with PEFT LoRA support"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize emotion predictor
        
        Args:
            model_path: Path to local model OR HuggingFace model name
                       If None, uses EMOTION_MODEL_PATH env var or default HF model
        """
        # Priority: function arg > env var > default HF model
        self.model_path = (
            model_path or 
            os.getenv("EMOTION_MODEL_PATH") or 
            "drPantagana/PULSE_emotion"
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_peft_model = False
        
        logger.info(f"Loading emotion model from {self.model_path}")
        
        try:
            # Try loading as PEFT model first
            self._load_peft_model()
        except Exception as e:
            logger.warning(f"Failed to load as PEFT model: {e}")
            logger.info("Attempting to load as standard model...")
            self._load_standard_model()
        
        logger.info(f"EmotionPredictor ready on device={self.device} labels={self.labels}")
    
    def _load_peft_model(self):
        """Load PEFT LoRA adapter model"""
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(self.model_path)
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=6
        )
        
        # Load PEFT adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer from base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path
        )
        
        self.is_peft_model = True
        self.labels = ["joy", "sadness", "anger", "love", "surprise", "neutral"]
        
        logger.info("✓ Loaded as PEFT LoRA model")
    
    def _load_standard_model(self):
        """Load standard transformers model (fallback)"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Try to get labels from config
        if hasattr(self.model.config, 'id2label'):
            self.labels = list(self.model.config.id2label.values())
        else:
            self.labels = ["joy", "sadness", "anger", "love", "surprise", "neutral"]
        
        logger.info("✓ Loaded as standard model")
    
    def predict(self, text: str, explain: bool = False) -> Dict:
        """
        Predict emotion for given text
        
        Args:
            text: Input text
            explain: Whether to include explanation
            
        Returns:
            Dict with emotion, confidence, all_scores, and optional explanation
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        
        # Get predicted class
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        predicted_emotion = self.labels[predicted_idx]
        
        # Create probability dictionary
        all_scores = {
            label: prob.item()
            for label, prob in zip(self.labels, probabilities)
        }
        
        result = {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "all_scores": all_scores
        }
        
        # Add explanation if requested
        if explain:
            important_tokens = self.extract_important_tokens(text, inputs)
            emotion_indicators = self.find_emotion_keywords(text, predicted_emotion)
            
            result["explanation"] = {
                "method": "attention_based",
                "important_tokens": important_tokens,
                "emotion_indicators": emotion_indicators,
                "reasoning": self._generate_reasoning(
                    text, predicted_emotion, confidence, 
                    important_tokens, emotion_indicators
                ),
                "model_type": "PEFT LoRA" if self.is_peft_model else "Standard",
                "text_length": len(text.split())
            }
        
        return result
    
    def extract_important_tokens(self, text: str, inputs: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract important tokens using model's attention weights"""
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                attentions = outputs.attentions
            
            # Get attention from last layer
            last_layer_attention = attentions[-1][0]
            avg_attention = last_layer_attention.mean(dim=0)
            cls_attention = avg_attention[0, :].cpu().numpy()
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Pair tokens with attention scores
            token_importance = []
            for token, score in zip(tokens, cls_attention):
                # Skip special tokens
                if token not in ['<s>', '</s>', '<pad>', '<mask>', '[CLS]', '[SEP]', '[PAD]']:
                    clean_token = token.replace('Ġ', '').replace('##', '').strip()
                    if clean_token:
                        token_importance.append((clean_token, float(score)))
            
            # Sort and return top k
            token_importance.sort(key=lambda x: x[1], reverse=True)
            return token_importance[:top_k]
            
        except Exception as e:
            logger.error(f"Error extracting tokens: {e}")
            return []
    
    def find_emotion_keywords(self, text: str, predicted_emotion: str) -> List[str]:
        """Find emotion-related keywords (supplementary)"""
        EMOTION_KEYWORDS = {
            "joy": ["happy", "great", "wonderful", "excellent", "love", "amazing", "fantastic"],
            "sadness": ["sad", "disappointed", "unhappy", "terrible", "awful", "bad"],
            "anger": ["angry", "furious", "mad", "frustrated", "annoyed", "outraged"],
            "love": ["love", "adore", "care", "cherish", "appreciate", "grateful"],
            "surprise": ["wow", "amazing", "unexpected", "surprised", "shocking", "incredible"],
            "neutral": ["ok", "fine", "alright", "normal", "regular"]
        }
        
        text_lower = text.lower()
        found_indicators = []
        
        for emotion, keywords in EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_indicators.append(f"{keyword} ({emotion})")
        
        return found_indicators
    
    def _generate_reasoning(self, text: str, emotion: str, confidence: float,
                           important_tokens: List[Tuple[str, float]],
                           emotion_indicators: List[str]) -> str:
        """Generate human-readable explanation"""
        reasoning_parts = []
        
        # Confidence
        if confidence > 0.8:
            reasoning_parts.append(f"High confidence ({confidence*100:.1f}%) in {emotion} classification.")
        elif confidence > 0.6:
            reasoning_parts.append(f"Moderate confidence ({confidence*100:.1f}%) in {emotion} classification.")
        else:
            reasoning_parts.append(f"Low confidence ({confidence*100:.1f}%). Text may be ambiguous.")
        
        # Model attention
        if important_tokens:
            top_tokens = [token for token, _ in important_tokens[:5]]
            reasoning_parts.append(f"Model focused on: {', '.join(top_tokens)}.")
        
        # Keyword hints
        if emotion_indicators:
            keywords = ", ".join([ind.split(" (")[0] for ind in emotion_indicators[:3]])
            reasoning_parts.append(f"Found emotion keywords: {keywords}.")
        
        return " ".join(reasoning_parts)
