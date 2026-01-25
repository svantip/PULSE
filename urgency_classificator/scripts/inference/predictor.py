from scipy import special
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from urgency_classificator.utils.config_loader import load_config, get_labels, get_model_name


class UrgencyPredictor:
    def __init__(self, model_path: Optional[str] = None, model_name: str = None, config_path: str = None):
        # config ostaje radi kompatibilnosti
        self.config = load_config(config_path)

        if model_name is None:
            model_name = get_model_name(self.config)

        # STRICT: uvijek učitaj iz model_name (HF) ako nije eksplicitno zadan model_path
        source = model_path if model_path else model_name

        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.tokenizer = AutoTokenizer.from_pretrained(source)

        # --- POPRAVLJENA LOGIKA ZA LABELE ---
        
        # 1. Pokušaj dohvatiti labele iz modela
        hf_labels = []
        if getattr(self.model.config, "id2label", None):
            hf_labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

        # 2. Provjeri jesu li te labele generičke (npr. "LABEL_0", "LABEL_1"...)
        # Pretvaramo u string da uhvatimo i slučajeve gdje su labele brojevi
        are_labels_generic = False
        if hf_labels:
            first_label = str(hf_labels[0]).upper()
            if first_label.startswith("LABEL") or first_label.isdigit():
                are_labels_generic = True

        # 3. Odluči koje labele koristiti
        # Koristimo HF labele samo ako postoje I ako NISU generičke
        if hf_labels and not are_labels_generic:
            self.urgency_labels = hf_labels
        else:
            # Inače koristi naše lijepe labele iz configa (low, medium, high, critical)
            # Ovo rješava problem s prikazom u Slacku
            self.urgency_labels = get_labels(self.config)

        self.model.eval()


    def predict(self, text: str, explain: bool = False) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)

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
        self.model.train()

        try:
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask', None)

            # Get the embedding layer
            embeddings = self.model.get_input_embeddings()(input_ids)
            embeddings = embeddings.detach().requires_grad_(True)

            # Forward pass with embeddings
            outputs = self.model(inputs_embeds=embeddings,
                                 attention_mask=attention_mask)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()

            # Compute gradients
            outputs.logits[0, predicted_class].backward()

            # Get token importance from embedding gradients
            token_ids = input_ids[0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

            # Use gradient magnitude as importance score
            if embeddings.grad is not None:
                # Sum gradients across embedding dimension for each token
                importance = embeddings.grad.abs().sum(dim=-1).squeeze().tolist()
                if not isinstance(importance, list):
                    importance = [importance]
            else:
                importance = [0.0] * len(tokens)

            # Filter out special tokens and create word-level importance
            token_importance = []
            for token, score in zip(tokens, importance):
                special = set(self.tokenizer.all_special_tokens)
                if token not in special:
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
