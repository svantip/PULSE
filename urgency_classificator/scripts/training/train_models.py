"""
Model Training Script for Urgency Classification
Streamlined training with configuration-based setup.

NOTE: Run from project root using: python train.py
      Or with PYTHONPATH: PYTHONPATH=. python scripts/training/train_models.py
"""
import sys
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import tempfile
import warnings
from utils.config_loader import load_config, get_labels, get_models
from utils.visualization import create_all_visualizations
from utils.reporting import generate_training_report
import os
# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Add project root to path (needed when running script directly)
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Project imports (after path is set up)

warnings.filterwarnings('ignore')


class UrgencyDataset(Dataset):
    """PyTorch Dataset for urgency classification."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModelTrainer:
    """Handles training of transformer models with configuration-based setup."""

    def __init__(self, config_path: str = None):
        """
        Initialize the trainer with configuration.

        Args:
            config_path: Path to config file (uses default if None)
        """
        self.config = load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load configuration
        self.labels = get_labels(self.config)
        self.models_config = get_models(self.config)
        self.training_config = self.config['TRAINING']
        self.class_weights_config = self.config['CLASS_WEIGHTS']
        self.paths_config = self.config['PATHS']
        self.data_split_config = self.config['DATA_SPLIT']

        # Setup paths - get project root (two levels up from this script)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../..'))
        self.data_path = os.path.join(project_root, self.paths_config['DATA'])
        self.output_dir = os.path.join(
            project_root, self.paths_config['MODELS'])
        self.reports_dir = os.path.join(
            project_root, self.paths_config['REPORTS'])

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, 'figures'), exist_ok=True)

        # Label mapping
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_label = {v: k for k, v in self.label_map.items()}

        # Initialize MLflow
        versioning_config = self.config['VERSIONING']
        mlflow.set_tracking_uri(versioning_config['MLFLOW_TRACKING_URI'])
        mlflow.set_experiment(versioning_config['EXPERIMENT_NAME'])

        print("="*70)
        print("URGENCY CLASSIFICATION MODEL TRAINING")
        print("="*70)
        print(f"\nTimestamp: {self.timestamp}")
        print(f"Data: {self.data_path}")
        print(f"Models: {[m['NAME'] for m in self.models_config]}")
        print(f"Labels: {self.labels}\n")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split dataset."""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        df['label_id'] = df['priority'].map(self.label_map)

        print(f"Total samples: {len(df)}")
        print(f"Label distribution:")
        for label, count in df['priority'].value_counts().sort_index().items():
            print(f"  {label}: {count} ({count/len(df)*100:.2f}%)")

        # Split data based on config
        val_test_size = 1.0 - self.data_split_config['TRAIN']
        test_ratio = self.data_split_config['TEST'] / val_test_size

        train_df, temp_df = train_test_split(
            df, test_size=val_test_size, random_state=self.training_config['SEED'],
            stratify=df['label_id']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_ratio, random_state=self.training_config['SEED'],
            stratify=temp_df['label_id']
        )

        print(f"\nData split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples\n")

        return train_df, val_df, test_df

    def _log_device_info(self):
        """Log device and environment information to MLflow."""
        # PyTorch and CUDA info
        mlflow.log_param("torch_version", torch.__version__)
        mlflow.log_param("cuda_available", torch.cuda.is_available())

        if torch.cuda.is_available():
            mlflow.log_param("cuda_version", torch.version.cuda)
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param("gpu_count", torch.cuda.device_count())
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mlflow.log_param("device_type", "MPS (Apple Silicon)")
        else:
            mlflow.log_param("device_type", "CPU")

        # Python version
        import platform
        mlflow.log_param("python_version", platform.python_version())

    def train_bert_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Train transformer model using configuration.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            model_name: HuggingFace model name

        Returns:
            Dictionary with model, metrics, and metadata
        """
        print("="*70)
        print(f"TRAINING MODEL: {model_name}")
        print("="*70)

        model_short_name = "XLM-RoBERTa" if "roberta" in model_name.lower() else "BERT"

        # Create unique run name to avoid conflicts
        run_name = f"{model_short_name}_{self.timestamp}"

        with mlflow.start_run(run_name=run_name):
            # Log device and environment info
            self._log_device_info()

            # Log config.yaml as artifact for reproducibility
            config_path = os.path.join(project_root, 'config.yaml')
            if os.path.exists(config_path):
                mlflow.log_artifact(config_path, "config")

            # Log basic model info (avoid duplicates with TrainingArguments auto-logging)
            mlflow.log_param("model_name", model_short_name)
            mlflow.log_param("base_model_hf", model_name)
            mlflow.log_param("train_samples", len(train_df))
            mlflow.log_param("val_samples", len(val_df))
            mlflow.log_param("test_samples", len(test_df))

            # Load tokenizer and model
            print("\nLoading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.label_map)
            )

            # Create datasets
            print("Creating datasets...")
            train_dataset = UrgencyDataset(
                train_df['body'].values,
                train_df['label_id'].values,
                tokenizer,
                max_length=self.training_config['MAX_LENGTH']
            )
            val_dataset = UrgencyDataset(
                val_df['body'].values,
                val_df['label_id'].values,
                tokenizer,
                max_length=self.training_config['MAX_LENGTH']
            )

            # Compute class weights from config - ensure proper label alignment
            # Sort to match label order
            unique_labels = np.sort(np.unique(train_df['label_id']))
            class_weights = compute_class_weight(
                'balanced',
                classes=unique_labels,
                y=train_df['label_id']
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float)

            # Apply multipliers from config - explicitly match label_id order
            print("\nClass weights (before multipliers):")
            for i, label_id in enumerate(unique_labels):
                label_name = self.id_to_label[label_id]
                print(
                    f"  {label_name} (id={label_id}): {class_weights[i]:.4f}")

            for i, label_id in enumerate(unique_labels):
                label_name = self.id_to_label[label_id]
                multiplier = self.class_weights_config[label_name.upper()]
                class_weights[i] = class_weights[i] * multiplier

            print("\nClass weights (after multipliers):")
            for i, label_id in enumerate(unique_labels):
                label_name = self.id_to_label[label_id]
                print(
                    f"  {label_name} (id={label_id}): {class_weights[i]:.4f}")
                mlflow.log_param(
                    f"class_weight_{label_name}", class_weights[i].item())

            # Define compute_metrics for evaluation
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                preds = np.argmax(predictions, axis=1)

                # Calculate metrics
                accuracy = accuracy_score(labels, preds)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, preds, average='weighted'
                )

                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # Calculate optimal eval_steps based on dataset size
            # Evaluate 3-5 times per epoch for efficiency
            total_train_steps = len(train_dataset) // (self.training_config['BATCH_SIZE'] *
                                                       self.training_config['GRADIENT_ACCUMULATION_STEPS'])
            # 4 evals per epoch, min 100
            optimal_eval_steps = max(100, total_train_steps // 4)
            eval_steps = self.training_config.get(
                'EVAL_STEPS', optimal_eval_steps)

            print(f"\nTraining info:")
            print(f"  Total training steps per epoch: ~{total_train_steps}")
            print(f"  Evaluation every {eval_steps} steps")
            print(f"  ~{total_train_steps // eval_steps} evaluations per epoch")

            # Training arguments from config
            training_args = TrainingArguments(
                output_dir=os.path.join(
                    self.output_dir, f"{model_short_name.lower()}_{self.timestamp}"),
                num_train_epochs=self.training_config['NUM_EPOCHS'],
                per_device_train_batch_size=self.training_config['BATCH_SIZE'],
                per_device_eval_batch_size=self.training_config['EVAL_BATCH_SIZE'],
                gradient_accumulation_steps=self.training_config['GRADIENT_ACCUMULATION_STEPS'],
                learning_rate=self.training_config['LEARNING_RATE'],
                weight_decay=self.training_config['WEIGHT_DECAY'],
                # Use only ratio, not steps
                warmup_ratio=self.training_config['WARMUP_RATIO'],
                logging_dir=os.path.join(self.output_dir, 'logs'),
                logging_steps=self.training_config['LOGGING_STEPS'],
                eval_strategy="steps",
                eval_steps=eval_steps,  # Use calculated optimal value
                save_strategy="steps",
                save_steps=eval_steps,  # Save at same frequency as eval
                save_total_limit=self.training_config['SAVE_TOTAL_LIMIT'],
                load_best_model_at_end=True,
                metric_for_best_model="f1",  # Use F1 instead of eval_loss
                greater_is_better=True,  # F1 is better when higher
                report_to=["mlflow"],
                seed=self.training_config['SEED'],
                lr_scheduler_type=self.training_config['LR_SCHEDULER_TYPE'],
                max_grad_norm=self.training_config.get('MAX_GRAD_NORM', 1.0),
                fp16=torch.cuda.is_available(),  # Enable only on CUDA GPUs
                dataloader_num_workers=self.training_config['DATALOADER_NUM_WORKERS'],
                dataloader_pin_memory=torch.cuda.is_available()  # Pin memory only with CUDA
            )

            # Note: TrainingArguments with report_to=["mlflow"] automatically logs:
            # - epochs, batch_size, learning_rate, weight_decay, etc.
            # We only log additional custom parameters here

            # Custom Trainer with class weights
            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fct = torch.nn.CrossEntropyLoss(
                        weight=class_weights.to(logits.device))
                    loss = loss_fct(logits, labels)
                    return (loss, outputs) if return_outputs else loss

                # Initialize trainer
            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=self.training_config['EARLY_STOPPING_PATIENCE'],
                    early_stopping_threshold=self.training_config['EARLY_STOPPING_THRESHOLD']
                )]
            )

            # Train
            print("\nStarting training...")
            train_result = trainer.train()

            print("\nTraining completed!")
            print(f"Training loss: {train_result.training_loss:.4f}")

            mlflow.log_metric("train_loss", train_result.training_loss)

            # Save model
            model_path = os.path.join(
                self.output_dir, f"{model_short_name.lower()}_final_{self.timestamp}")
            trainer.save_model(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"Model saved to: {model_path}")

            # Evaluate on test set
            print("\nEvaluating on test set...")
            test_dataset = UrgencyDataset(
                test_df['body'].values,
                test_df['label_id'].values,
                tokenizer,
                max_length=self.training_config['MAX_LENGTH']
            )

            predictions = trainer.predict(test_dataset)
            preds = np.argmax(predictions.predictions, axis=1)
            labels = predictions.label_ids

            # Calculate metrics
            metrics = self._calculate_metrics(
                labels, preds, model_short_name)

            # Log metrics to MLflow
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Log per-class metrics
            for i, label in enumerate(self.labels):
                mlflow.log_metric(f"{label}_precision",
                                  metrics['class_precision'][i])
                mlflow.log_metric(f"{label}_recall",
                                  metrics['class_recall'][i])
                mlflow.log_metric(f"{label}_f1", metrics['class_f1'][i])
                mlflow.log_metric(f"{label}_support",
                                  metrics['class_support'][i])

            # Log confusion matrix
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                np.savetxt(f.name, metrics['confusion_matrix'], fmt='%d')
                mlflow.log_artifact(f.name, "confusion_matrix.txt")
                os.unlink(f.name)

            # Log model
            mlflow.pytorch.log_model(model, "model")

            return {
                'model': model,
                'tokenizer': tokenizer,
                'model_path': model_path,
                'predictions': preds,
                'true_labels': labels,
                'metrics': metrics,
                'train_result': train_result
            }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        print(f"\n{model_name} Performance Metrics:")
        print("-"*70)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        # Per-class metrics
        print("\nPer-class metrics:")
        class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        for i, label in enumerate(self.labels):
            print(f"  {label:6} - Precision: {class_precision[i]:.4f}, "
                  f"Recall: {class_recall[i]:.4f}, F1: {class_f1[i]:.4f}, "
                  f"Support: {class_support[i]}")

        cm = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'class_precision': class_precision,
            'class_recall': class_recall,
            'class_f1': class_f1,
            'class_support': class_support
        }

    def explain_bert_predictions(
        self,
        model,
        tokenizer,
        test_df: pd.DataFrame,
        num_samples: int = 10,
        model_name: str = "BERT"
    ) -> List[Dict[str, Any]]:
        """
        Generate attention-based explanations.

        Args:
            model: Trained model
            tokenizer: Tokenizer
            test_df: Test dataframe
            num_samples: Number of samples to explain
            model_name: Name of model

        Returns:
            List of explanation dictionaries
        """
        print(f"\nGenerating {model_name} explanations...")

        model = model.cpu()
        model.eval()
        explanations = []

        sample_df = test_df.groupby('priority').head(num_samples // 3)

        for idx, row in sample_df.iterrows():
            text = row['body']
            true_label = row['priority']

            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.training_config['MAX_LENGTH'],
                padding=True
            )
            inputs = {k: v.cpu() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                logits = outputs.logits
                attentions = outputs.attentions

            pred_id = torch.argmax(logits, dim=1).item()
            pred_label = self.id_to_label[pred_id]
            confidence = torch.softmax(logits, dim=1)[0][pred_id].item()

            last_attention = attentions[-1][0].mean(dim=0)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            cls_attention = last_attention[0, :].cpu().numpy()

            top_indices = np.argsort(cls_attention)[-10:][::-1]
            important_tokens = [(tokens[i], cls_attention[i])
                                for i in top_indices if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]

            explanations.append({
                'text': text[:200] + '...' if len(text) > 200 else text,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence,
                'correct': true_label == pred_label,
                'important_tokens': important_tokens[:5]
            })

        return explanations

    def run_full_training_pipeline(self):
        """Execute complete training pipeline."""
        print("\n" + "="*70)
        print("STARTING TRAINING PIPELINE")
        print("="*70 + "\n")

        # Load data
        train_df, val_df, test_df = self.load_and_prepare_data()

        # Train models from config
        model1_name = self.models_config[0]['NAME']
        model2_name = self.models_config[1]['NAME']

        model1_alias = self.models_config[0]['ALIAS']
        model2_alias = self.models_config[1]['ALIAS']

        print(f"\n{'='*70}")
        print(f"TRAINING MODEL 1: {model1_name}")
        print(f"{'='*70}")
        model1_results = self.train_bert_model(
            train_df, val_df, test_df, model1_name)

        print(f"\n{'='*70}")
        print(f"TRAINING MODEL 2: {model2_name}")
        print(f"{'='*70}")
        model2_results = self.train_bert_model(
            train_df, val_df, test_df, model2_name)

        # Generate explanations
        model1_explanations = self.explain_bert_predictions(
            model1_results['model'],
            model1_results['tokenizer'],
            test_df,
            model_name=model1_alias
        )

        model2_explanations = self.explain_bert_predictions(
            model2_results['model'],
            model2_results['tokenizer'],
            test_df,
            model_name=model2_alias
        )

        # Create visualizations
        create_all_visualizations(
            model1_results,
            model2_results,
            model1_alias,
            model2_alias,
            self.labels,
            self.reports_dir,
            self.timestamp
        )

        # Generate report
        report_path = generate_training_report(
            model1_results,
            model2_results,
            model1_alias,
            model2_alias,
            model1_explanations,
            model2_explanations,
            self.data_path,
            self.reports_dir,
            self.timestamp
        )

        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETED!")
        print("="*70)
        print(f"\n✓ Models saved to: {self.output_dir}")
        print(f"✓ Reports saved to: {self.reports_dir}")
        print(
            f"✓ MLflow tracking: {self.config['VERSIONING']['MLFLOW_TRACKING_URI']}")
        print(f"\nTo view MLflow UI: mlflow ui")
        print("="*70 + "\n")

        return {
            'model1_results': model1_results,
            'model2_results': model2_results,
            'report_path': report_path
        }


def main():
    """Main execution function."""
    # Get project root (two levels up from this script)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..'))
    data_path = os.path.join(project_root, "data", "clean.csv")

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please run scripts/data_prep/clean_data.py first.")
        return

    trainer = ModelTrainer()
    results = trainer.run_full_training_pipeline()

    print("\n🎉 Training complete! Check the reports folder for analysis.")


if __name__ == "__main__":
    main()
