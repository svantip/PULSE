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

from .config_loader import load_config, get_labels, get_model_name, get_models, get_training_config, get_versioning_config


class UrgencyClassifier:
    """
    Urgency classifier with fine-tuning capabilities and MLflow versioning.
    """
    
    def __init__(self, model_name: str = None, model_alias: str = None, model_version: Optional[str] = None, config_path: str = None):
        """
        Initialize the urgency classifier.
        
        Args:
            model_name: Base model name from HuggingFace (if None, loads from config)
            model_alias: Model alias to use from config (e.g., 'bert', 'distilbert')
            model_version: Specific version to load (if None, uses latest)
            config_path: Path to configuration file (if None, uses default)
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Determine model name
        if model_name:
            self.model_name = model_name
        elif model_alias:
            self.model_name = get_model_name(self.config, alias=model_alias)
        else:
            # Default to first model in config
            self.model_name = get_model_name(self.config)
        
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
            mlflow.start_run(run_name=f"urgency_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
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
            if mlflow_enabled and mlflow.active_run():
                mlflow.end_run()
    
    def compare_models(self, train_dataset, eval_dataset, output_dir: str = None) -> Dict[str, Any]:
        """
        Compare all models defined in config by training and evaluating each.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save models (if None, uses default './urgency_models')
            
        Returns:
            Dictionary with comparison results for each model
        """
        if output_dir is None:
            output_dir = "./urgency_models"
        
        models_config = get_models(self.config)
        results = {}
        
        print(f"\n{'='*60}")
        print(f"Comparing {len(models_config)} models for urgency classification")
        print(f"{'='*60}\n")
        
        for model_config in models_config:
            model_name = model_config['NAME']
            model_alias = model_config['ALIAS']
            
            print(f"\nTraining model: {model_alias} ({model_name})")
            print(f"{'-'*60}")
            
            # Initialize new model
            original_model_name = self.model_name
            self.model_name = model_name
            self.load_model()
            
            # Fine-tune
            model_output_dir = os.path.join(output_dir, model_alias)
            version_path = self.fine_tune(train_dataset, eval_dataset, model_output_dir)
            
            # Store results
            # Read metadata to get metrics
            metadata_path = os.path.join(version_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    results[model_alias] = {
                        'model_name': model_name,
                        'version_path': version_path,
                        'metrics': metadata.get('metrics', {})
                    }
            
            # Restore original model name
            self.model_name = original_model_name
        
        # Check if we have results
        if not results:
            raise ValueError("No models were successfully trained. Check your configuration and datasets.")
        
        # Determine best model
        best_model = max(results.items(), key=lambda x: x[1]['metrics'].get('eval_accuracy', 0))
        
        print(f"\n{'='*60}")
        print("Model Comparison Results:")
        print(f"{'='*60}")
        for alias, result in results.items():
            metrics = result['metrics']
            print(f"\n{alias} ({result['model_name']}):")
            print(f"  Accuracy: {metrics.get('eval_accuracy', 0):.4f}")
            print(f"  F1 Score: {metrics.get('eval_f1', 0):.4f}")
            if alias == best_model[0]:
                print(f"  ⭐ BEST MODEL")
        
        print(f"\n{'='*60}\n")
        
        results['best_model'] = best_model[0]
        return results
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted")
        }
