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

from .config_loader import load_config, get_labels, get_model_name, get_training_config, get_versioning_config


class UrgencyClassifier:
    """
    Urgency classifier with fine-tuning capabilities and MLflow versioning.
    """
    
    def __init__(self, model_name: str = None, model_version: Optional[str] = None, config_path: str = None):
        """
        Initialize the urgency classifier.
        
        Args:
            model_name: Base model name from HuggingFace (if None, loads from config)
            model_version: Specific version to load (if None, uses latest)
            config_path: Path to configuration file (if None, uses default)
        """
        # Load configuration
        self.config = load_config(config_path)
        
        self.model_name = model_name or get_model_name(self.config)
        self.model_version = model_version or "latest"
        self.model = None
        self.tokenizer = None
        self.urgency_labels = get_labels(self.config)
        
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
    
    def fine_tune(self, train_dataset, eval_dataset, output_dir: str = None):
        """
        Fine-tune the model on provided dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save fine-tuned model (if None, uses default from config or './urgency_models')
        """
        # Get training configuration
        training_config = get_training_config(self.config)
        versioning_config = get_versioning_config(self.config)
        
        if output_dir is None:
            output_dir = "./urgency_models"
        
        # Start MLflow run for versioning if enabled
        mlflow_enabled = versioning_config.get('ENABLED', True)
        
        if mlflow_enabled:
            mlflow_uri = versioning_config.get('MLFLOW_TRACKING_URI', './mlruns')
            experiment_name = versioning_config.get('EXPERIMENT_NAME', 'urgency_classification')
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
        
        run_context = mlflow.start_run(run_name=f"urgency_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}") if mlflow_enabled else None
        
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                learning_rate=training_config.get('LEARNING_RATE', 2e-5),
                per_device_train_batch_size=training_config.get('BATCH_SIZE', 16),
                per_device_eval_batch_size=training_config.get('BATCH_SIZE', 16),
                num_train_epochs=training_config.get('NUM_EPOCHS', 3),
                weight_decay=training_config.get('WEIGHT_DECAY', 0.01),
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
            
            # Log metrics to MLflow if enabled
            eval_results = trainer.evaluate()
            if mlflow_enabled:
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
            
            # Log model to MLflow if enabled
            if mlflow_enabled:
                mlflow.transformers.log_model(self.model, "urgency_model")
            
            return version_path
        finally:
            if run_context:
                mlflow.end_run()
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted")
        }
