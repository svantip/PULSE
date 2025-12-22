"""
Urgency Classification Model
Handles fine-tuning of urgency classification models with versioning support.
"""
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import mlflow


class UrgencyClassifier:
    """
    Urgency classifier with fine-tuning capabilities and MLflow versioning.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", model_version: Optional[str] = None):
        """
        Initialize the urgency classifier.
        
        Args:
            model_name: Base model name from HuggingFace
            model_version: Specific version to load (if None, uses latest)
        """
        self.model_name = model_name
        self.model_version = model_version or "latest"
        self.model = None
        self.tokenizer = None
        self.urgency_labels = ["low", "medium", "high", "critical"]
        
    def load_model(self, model_path: Optional[str] = None):
        """
        Load model and tokenizer.
        
        Args:
            model_path: Path to saved model (if None, loads from HuggingFace)
        """
        if model_path and os.path.exists(model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.urgency_labels)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def fine_tune(self, train_dataset, eval_dataset, output_dir: str = "./urgency_models"):
        """
        Fine-tune the model on provided dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save fine-tuned model
        """
        # Start MLflow run for versioning
        with mlflow.start_run(run_name=f"urgency_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self._compute_metrics,
            )
            
            # Train the model
            trainer.train()
            
            # Log metrics to MLflow
            eval_results = trainer.evaluate()
            mlflow.log_metrics(eval_results)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("num_labels", len(self.urgency_labels))
            
            # Save model
            version_path = os.path.join(output_dir, f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            trainer.save_model(version_path)
            self.tokenizer.save_pretrained(version_path)
            
            # Save metadata
            metadata = {
                "model_name": self.model_name,
                "version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "labels": self.urgency_labels,
                "metrics": eval_results
            }
            
            with open(os.path.join(version_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Log model to MLflow
            mlflow.transformers.log_model(self.model, "urgency_model")
            
            return version_path
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted")
        }
