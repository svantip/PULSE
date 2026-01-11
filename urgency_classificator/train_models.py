"""
Model Training Script for Urgency Classification
Fine-tunes BERT and baseline models with gradient-based optimization,
includes explainability analysis, and generates comprehensive reports.
"""

import mlflow.pytorch
import mlflow.sklearn
import mlflow
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries

# MLflow for experiment tracking

# Note: We use attention-based explanations for BERT instead of SHAP
# Attention weights are more interpretable and efficient for transformer models


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
    """Handles training of BERT and baseline models with comprehensive evaluation."""

    def __init__(self, data_path: str, output_dir: str = "models", reports_dir: str = "reports"):
        """
        Initialize the trainer.

        Args:
            data_path: Path to cleaned CSV dataset
            output_dir: Directory to save trained models
            reports_dir: Directory to save reports and visualizations
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(os.path.join(reports_dir, 'figures'), exist_ok=True)

        # Label mapping
        self.label_map = {'low': 0, 'medium': 1, 'high': 2}
        self.id_to_label = {v: k for k, v in self.label_map.items()}

        # Initialize MLflow
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("urgency_classification")

        print("="*70)
        print("URGENCY CLASSIFICATION MODEL TRAINING")
        print("="*70)
        print(f"\nTimestamp: {self.timestamp}")
        print(f"Data: {data_path}")
        print(f"Output: {output_dir}")
        print(f"Reports: {reports_dir}\n")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split dataset."""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)

        # Map labels to integers
        df['label_id'] = df['priority'].map(self.label_map)

        print(f"Total samples: {len(df)}")
        print(f"Label distribution:")
        for label, count in df['priority'].value_counts().sort_index().items():
            print(f"  {label}: {count} ({count/len(df)*100:.2f}%)")

        # Split: 70% train, 15% validation, 15% test
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df['label_id']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_id']
        )

        print(f"\nData split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples\n")

        return train_df, val_df, test_df

    def train_bert_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_name: str = "bert-base-multilingual-cased"
    ) -> Dict[str, Any]:
        """
        Train BERT model with gradient-based optimization.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            model_name: HuggingFace model name

        Returns:
            Dictionary with model, metrics, and metadata
        """
        print("="*70)
        print(f"TRAINING BERT MODEL: {model_name}")
        print("="*70)

        with mlflow.start_run(run_name=f"BERT_{self.timestamp}"):
            # Log parameters
            mlflow.log_param("model_type", "BERT")
            mlflow.log_param("base_model", model_name)
            mlflow.log_param("train_samples", len(train_df))

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
                tokenizer
            )
            val_dataset = UrgencyDataset(
                val_df['body'].values,
                val_df['label_id'].values,
                tokenizer
            )

            # Compute class weights for imbalanced data with balanced emphasis
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_df['label_id']),
                y=train_df['label_id']
            )
            # Adjust weights: reduce LOW (was over-predicting), boost MEDIUM (was under-predicting)
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            # 20% boost for LOW (reduced from 50%)
            class_weights[0] = class_weights[0] * 1.2
            class_weights[1] = class_weights[1] * \
                1.5  # 50% boost for MEDIUM (new)
            print(f"\nClass weights (balanced for MEDIUM): {class_weights}")
            mlflow.log_params({f"class_weight_{i}": w.item()
                              for i, w in enumerate(class_weights)})

            # Training arguments - Optimized for M3 MacBook Pro (18GB RAM)
            training_args = TrainingArguments(
                output_dir=os.path.join(
                    self.output_dir, f"bert_{self.timestamp}"),
                num_train_epochs=10,  # Increased from 3 to 10 for better learning
                per_device_train_batch_size=8,  # Increased from 4 for M3 chip
                per_device_eval_batch_size=16,  # Increased from 8 for faster eval
                # Reduced from 4 (larger batch size compensates)
                gradient_accumulation_steps=2,
                learning_rate=3e-5,  # Slightly increased for faster learning
                weight_decay=0.01,
                warmup_steps=1000,  # Increased warmup for better convergence
                warmup_ratio=0.1,  # 10% of training for warmup
                logging_dir=os.path.join(self.output_dir, 'logs'),
                logging_steps=100,
                eval_strategy="steps",
                eval_steps=200,
                save_strategy="steps",
                save_steps=200,
                save_total_limit=3,  # Keep more checkpoints
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=["none"],
                seed=42,
                lr_scheduler_type="cosine",  # Cosine learning rate schedule
                fp16=False,  # Keep False for MPS compatibility
                dataloader_num_workers=2,  # Parallel data loading for M3
                dataloader_pin_memory=False  # Not needed for MPS
            )

            mlflow.log_params({
                "epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "weight_decay": training_args.weight_decay
            })

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
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # Train
            print("\nStarting training...")
            train_result = trainer.train()

            print("\nTraining completed!")
            print(f"Training loss: {train_result.training_loss:.4f}")

            # Save model
            model_path = os.path.join(
                self.output_dir, f"bert_final_{self.timestamp}")
            trainer.save_model(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"Model saved to: {model_path}")

            # Evaluate on test set
            print("\nEvaluating on test set...")
            test_dataset = UrgencyDataset(
                test_df['body'].values,
                test_df['label_id'].values,
                tokenizer
            )

            predictions = trainer.predict(test_dataset)
            preds = np.argmax(predictions.predictions, axis=1)
            labels = predictions.label_ids

            # Calculate metrics
            metrics = self._calculate_metrics(labels, preds, "BERT")

            # Log metrics to MLflow
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

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

    def train_baseline_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train TF-IDF + Logistic Regression baseline with gradient descent.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data

        Returns:
            Dictionary with model, metrics, and metadata
        """
        print("="*70)
        print("TRAINING BASELINE MODEL: TF-IDF + Logistic Regression")
        print("="*70)

        with mlflow.start_run(run_name=f"Baseline_{self.timestamp}"):
            # Log parameters
            mlflow.log_param("model_type", "Baseline")
            mlflow.log_param("vectorizer", "TF-IDF")
            mlflow.log_param("classifier", "LogisticRegression")
            mlflow.log_param("train_samples", len(train_df))

            # Create TF-IDF vectorizer
            print("\nCreating TF-IDF features...")
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                strip_accents='unicode'
            )

            X_train = vectorizer.fit_transform(train_df['body'])
            X_val = vectorizer.transform(val_df['body'])
            X_test = vectorizer.transform(test_df['body'])

            print(f"Feature dimensions: {X_train.shape[1]}")
            mlflow.log_param("n_features", X_train.shape[1])

            y_train = train_df['label_id'].values
            y_val = val_df['label_id'].values
            y_test = test_df['label_id'].values

            # Compute class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: w for i, w in enumerate(class_weights)}
            print(f"\nClass weights: {class_weight_dict}")

            # Train Logistic Regression with gradient descent (saga solver)
            print("\nTraining Logistic Regression (gradient-based optimization)...")
            model = LogisticRegression(
                max_iter=1000,
                solver='saga',  # Gradient-based solver
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            mlflow.log_params({
                "solver": "saga",
                "max_iter": 1000,
                "class_weight": "balanced"
            })

            model.fit(X_train, y_train)

            print("\nTraining completed!")

            # Predictions
            preds = model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, preds, "Baseline")

            # Log metrics to MLflow
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Save model
            model_path = os.path.join(
                self.output_dir, f"baseline_final_{self.timestamp}")
            os.makedirs(model_path, exist_ok=True)

            import joblib
            joblib.dump(model, os.path.join(model_path, 'model.pkl'))
            joblib.dump(vectorizer, os.path.join(model_path, 'vectorizer.pkl'))
            print(f"Model saved to: {model_path}")

            # Log model
            mlflow.sklearn.log_model(model, "model")

            return {
                'model': model,
                'vectorizer': vectorizer,
                'model_path': model_path,
                'predictions': preds,
                'true_labels': y_test,
                'metrics': metrics,
                'X_test': X_test
            }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        print(f"\n{model_name} Performance Metrics:")
        print("-"*70)

        # Basic metrics
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

        for i, label in enumerate(['low', 'medium', 'high']):
            print(f"  {label:6} - Precision: {class_precision[i]:.4f}, "
                  f"Recall: {class_recall[i]:.4f}, F1: {class_f1[i]:.4f}, "
                  f"Support: {class_support[i]}")

        # Confusion matrix
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
        Generate BERT explanations using attention weights.

        Attention-based explanations are superior to SHAP for BERT because:
        - Built into the model architecture (no additional computation)
        - Show exactly what the model attended to during prediction
        - Fast and interpretable
        - Gradient information already captured in attention mechanism

        Args:
            model: Trained BERT model
            tokenizer: Tokenizer
            test_df: Test dataframe
            num_samples: Number of samples to explain
            model_name: Name of model for display

        Returns:
            List of explanation dictionaries
        """
        print("\n" + "="*70)
        print(f"GENERATING {model_name} EXPLANATIONS (Attention-based)")
        print("="*70)

        # Move model to CPU to avoid MPS device issues with attention outputs
        model = model.cpu()
        model.eval()
        explanations = []

        # Sample diverse examples
        sample_df = test_df.groupby('priority').head(num_samples // 3)

        for idx, row in sample_df.iterrows():
            text = row['body']
            true_label = row['priority']

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )

            # Move inputs to CPU to match model device
            inputs = {k: v.cpu() for k, v in inputs.items()}

            # Get prediction and attention weights
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                logits = outputs.logits
                attentions = outputs.attentions  # Tuple of attention weights per layer

            pred_id = torch.argmax(logits, dim=1).item()
            pred_label = self.id_to_label[pred_id]
            confidence = torch.softmax(logits, dim=1)[0][pred_id].item()

            # Use last layer attention (average across heads)
            # Average across attention heads
            last_attention = attentions[-1][0].mean(dim=0)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            # Get top important tokens (based on attention to [CLS] token)
            cls_attention = last_attention[0, :].cpu().numpy()

            # Get top 10 tokens
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

    def explain_baseline_predictions(
        self,
        model,
        vectorizer,
        test_df: pd.DataFrame,
        num_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate baseline model explanations using feature importance.

        Args:
            model: Trained baseline model
            vectorizer: TF-IDF vectorizer
            test_df: Test dataframe
            num_samples: Number of samples to explain

        Returns:
            List of explanation dictionaries
        """
        print("\n" + "="*70)
        print("GENERATING BASELINE EXPLANATIONS (Feature Importance)")
        print("="*70)

        explanations = []
        feature_names = vectorizer.get_feature_names_out()

        # Sample diverse examples
        sample_df = test_df.groupby('priority').head(num_samples // 3)

        for idx, row in sample_df.iterrows():
            text = row['body']
            true_label = row['priority']

            # Transform and predict
            X = vectorizer.transform([text])
            pred_id = model.predict(X)[0]
            pred_label = self.id_to_label[pred_id]
            confidence = model.predict_proba(X)[0][pred_id]

            # Get feature importance (coefficients * feature values)
            coefficients = model.coef_[pred_id]
            feature_values = X.toarray()[0]
            importance = coefficients * feature_values

            # Get top features
            top_indices = np.argsort(np.abs(importance))[-5:][::-1]
            important_features = [(feature_names[i], importance[i])
                                  for i in top_indices]

            explanations.append({
                'text': text[:200] + '...' if len(text) > 200 else text,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence,
                'correct': true_label == pred_label,
                'important_features': important_features
            })

        return explanations

    def create_visualizations(
        self,
        model1_results: Dict[str, Any],
        model2_results: Dict[str, Any]
    ):
        """Create comprehensive visualization plots."""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)

        fig_dir = os.path.join(self.reports_dir, 'figures')

        # 1. Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (results, title) in enumerate([
            (model1_results, 'BERT Model'),
            (model2_results, 'DistilBERT Model')
        ]):
            cm = results['metrics']['confusion_matrix']
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'],
                ax=axes[idx]
            )
            axes[idx].set_title(f'{title}\nConfusion Matrix',
                                fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(os.path.join(
            fig_dir, f'confusion_matrices_{self.timestamp}.png'), dpi=300)
        plt.close()
        print(f"✓ Saved confusion matrices")

        # 2. Performance Comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        model1_metrics = [
            model1_results['metrics']['accuracy'],
            model1_results['metrics']['precision'],
            model1_results['metrics']['recall'],
            model1_results['metrics']['f1']
        ]
        model2_metrics = [
            model2_results['metrics']['accuracy'],
            model2_results['metrics']['precision'],
            model2_results['metrics']['recall'],
            model2_results['metrics']['f1']
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, model1_metrics, width,
                       label='BERT', color='#3498db')
        bars2 = ax.bar(x + width/2, model2_metrics, width,
                       label='DistilBERT', color='#e74c3c')

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(
            fig_dir, f'performance_comparison_{self.timestamp}.png'), dpi=300)
        plt.close()
        print(f"✓ Saved performance comparison")

        # 3. Per-class F1 Scores
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = ['Low', 'Medium', 'High']
        model1_f1 = model1_results['metrics']['class_f1']
        model2_f1 = model2_results['metrics']['class_f1']

        x = np.arange(len(labels))

        bars1 = ax.bar(x - width/2, model1_f1, width,
                       label='BERT', color='#3498db')
        bars2 = ax.bar(x + width/2, model2_f1, width,
                       label='DistilBERT', color='#e74c3c')

        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('Per-Class F1-Score Comparison',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(
            fig_dir, f'per_class_f1_{self.timestamp}.png'), dpi=300)
        plt.close()
        print(f"✓ Saved per-class F1 scores")

        print(f"\n✓ All visualizations saved to: {fig_dir}\n")

    def generate_report(
        self,
        model1_results: Dict[str, Any],
        model2_results: Dict[str, Any],
        model1_explanations: List[Dict],
        model2_explanations: List[Dict]
    ):
        """Generate comprehensive training report."""
        print("="*70)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*70)

        report_path = os.path.join(
            self.reports_dir, f'training_report_{self.timestamp}.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("URGENCY CLASSIFICATION MODEL TRAINING REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_path}\n\n")

            # Overview
            f.write("-"*70 + "\n")
            f.write("TRAINING OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write("Two transformer models were trained and compared:\n")
            f.write("1. BERT (bert-base-multilingual-cased)\n")
            f.write("   - 12 layers, 12 attention heads, 768 hidden size\n")
            f.write("   - 110M parameters\n")
            f.write("   - Full BERT architecture\n\n")
            f.write("2. DistilBERT (distilbert-base-multilingual-cased)\n")
            f.write("   - 6 layers, 12 attention heads, 768 hidden size\n")
            f.write("   - 66M parameters (40% smaller)\n")
            f.write("   - Distilled from BERT (faster, retains 97% performance)\n\n")

            f.write("Both models used gradient-based optimization:\n")
            f.write("- Optimizer: AdamW with learning rate 2e-5\n")
            f.write("- Training: Backpropagation through transformer layers\n")
            f.write("- Loss: Weighted Cross-Entropy (class-balanced)\n\n")

            # Performance Comparison
            f.write("-"*70 + "\n")
            f.write("PERFORMANCE COMPARISON\n")
            f.write("-"*70 + "\n\n")

            f.write("Overall Metrics:\n")
            f.write(
                f"{'Metric':<15} {'BERT':<12} {'DistilBERT':<12} {'Difference':<12}\n")
            f.write("-"*70 + "\n")

            metrics = ['accuracy', 'precision', 'recall', 'f1']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

            for metric, name in zip(metrics, metric_names):
                model1_val = model1_results['metrics'][metric]
                model2_val = model2_results['metrics'][metric]
                diff = model1_val - model2_val
                f.write(
                    f"{name:<15} {model1_val:<12.4f} {model2_val:<12.4f} {diff:+.4f}\n")

            f.write("\n\nPer-Class Performance (F1-Score):\n")
            f.write(
                f"{'Class':<15} {'BERT':<12} {'DistilBERT':<12} {'Difference':<12}\n")
            f.write("-"*70 + "\n")

            for i, label in enumerate(['Low', 'Medium', 'High']):
                model1_val = model1_results['metrics']['class_f1'][i]
                model2_val = model2_results['metrics']['class_f1'][i]
                diff = model1_val - model2_val
                f.write(
                    f"{label:<15} {model1_val:<12.4f} {model2_val:<12.4f} {diff:+.4f}\n")

            # Model Analysis
            f.write("\n\n" + "-"*70 + "\n")
            f.write("MODEL ANALYSIS & COMPARISON\n")
            f.write("-"*70 + "\n\n")

            # Determine winner
            model1_f1 = model1_results['metrics']['f1']
            model2_f1 = model2_results['metrics']['f1']

            if model1_f1 > model2_f1:
                winner = "BERT"
                diff_pct = ((model1_f1 - model2_f1) / model2_f1) * 100
                f.write(f"🏆 WINNER: BERT Model\n")
                f.write(f"Performance advantage: {diff_pct:.2f}%\n\n")

                f.write("Why BERT Outperforms DistilBERT:\n\n")
                f.write("1. MODEL CAPACITY:\n")
                f.write(
                    "   - BERT has 12 transformer layers vs DistilBERT's 6 layers\n")
                f.write("   - More layers allow deeper contextual understanding\n")
                f.write("   - Better at capturing complex urgency patterns\n\n")

                f.write("2. TRAINING DEPTH:\n")
                f.write("   - BERT: 110M parameters provide more expressive power\n")
                f.write(
                    "   - DistilBERT: 66M parameters (40% reduction) may limit capacity\n")
                f.write(
                    "   - Additional parameters help with multilingual understanding\n\n")

                f.write("3. FINE-TUNING FLEXIBILITY:\n")
                f.write(
                    "   - More layers in BERT enable better task-specific adaptation\n")
                f.write(
                    "   - Gradient flow through 12 layers captures nuanced patterns\n")
                f.write(
                    "   - DistilBERT's compression may lose some task-relevant features\n\n")

                f.write("Trade-offs:\n")
                f.write("   - BERT is 2x slower and uses 2x memory\n")
                f.write("   - Performance gain may not justify computational cost\n")
                f.write(
                    "   - Consider DistilBERT for production if speed matters\n\n")

            else:
                winner = "DistilBERT"
                diff_pct = ((model2_f1 - model1_f1) / model1_f1) * 100
                f.write(f"🏆 WINNER: DistilBERT Model\n")
                f.write(f"Performance advantage: {diff_pct:.2f}%\n\n")

                f.write("Why DistilBERT Matches or Exceeds BERT:\n\n")
                f.write("1. EFFICIENT ARCHITECTURE:\n")
                f.write(
                    "   - Knowledge distillation retained most of BERT's capabilities\n")
                f.write(
                    "   - 40% smaller, 60% faster, while maintaining 97% performance\n")
                f.write("   - Better regularization may prevent overfitting\n\n")

                f.write("2. DATASET SIZE:\n")
                f.write(
                    "   - With 20K samples, smaller model may generalize better\n")
                f.write("   - BERT's extra capacity not fully utilized\n")
                f.write("   - Less prone to overfitting on limited data\n\n")

                f.write("3. TRAINING EFFIergence due to fewer parameters\n")
                f.write("   - Better gradient flow through 6 layers\n")
                f.write("   - More stable training dynamics\n\n")

                f.write("Benefits:\n")
                f.write("   - 2x faster inference (important for production)\n")
                f.write("   - 2x less memory (can run on smaller GPUs)\n")
                f.write("   - Equal or better performance\n")
                f.write("   - ✅ Recommended for deployment\n\n")

            # Explainability
            f.write("-"*70 + "\n")
            f.write("MODEL EXPLAINABILITY (Attention-based)\n")
            f.write("-"*70 + "\n\n")

            f.write("Both models use attention weights to explain predictions.\n")
            f.write(
                "Attention shows which words the model focused on when making decisions.\n\n")

            f.write("BERT Explanations:\n")
            f.write("-"*70 + "\n")
            for i, exp in enumerate(model1_explanations[:5], 1):
                f.write(f"\nExample {i}:\n")
                f.write(f"Text: {exp['text']}\n")
                f.write(f"True: {exp['true_label']} | Predicted: {exp['predicted_label']} "
                        f"| Confidence: {exp['confidence']:.3f} | Correct: {exp['correct']}\n")
                f.write(f"Important tokens (attention-weighted):\n")
                for token, weight in exp['important_tokens']:
                    f.write(f"  - '{token}': {weight:.4f}\n")

            f.write("\n\nDistilBERT Explanations:\n")
            f.write("-"*70 + "\n")
            for i, exp in enumerate(model2_explanations[:5], 1):
                f.write(f"\nExample {i}:\n")
                f.write(f"Text: {exp['text']}\n")
                f.write(f"True: {exp['true_label']} | Predicted: {exp['predicted_label']} "
                        f"| Confidence: {exp['confidence']:.3f} | Correct: {exp['correct']}\n")
                f.write(f"Important tokens (attention-weighted):\n")
                for token, weight in exp['important_tokens']:
                    f.write(f"  - '{token}': {weight:.4f}\n")
            f.write("\n\n" + "-"*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-"*70 + "\n\n")

            f.write(f"1. DEPLOYMENT: Use {winner} model for production\n\n")

            f.write("2. PERFORMANCE vs EFFICIENCY TRADE-OFF:\n")
            if model1_f1 > model2_f1:
                perf_gain = ((model1_f1 - model2_f1) / model2_f1) * 100
                if perf_gain < 5:
                    f.write(
                        f"   - Performance gain is only {perf_gain:.2f}%\n")
                    f.write("   - Consider DistilBERT for 2x speed improvement\n")
                    f.write("   - BERT recommended only if accuracy is critical\n")
                else:
                    f.write(
                        f"   - Significant {perf_gain:.2f}% improvement justifies BERT\n")
                    f.write("   - Use BERT for production despite higher cost\n")
            else:
                f.write("   - DistilBERT is best choice: faster AND more accurate\n")
                f.write("   - No trade-off needed - clear winner\n")

            f.write("\n3. IMPROVEMENTS:\n")
            f.write("   - Try XLM-RoBERTa for even better multilingual performance\n")
            f.write("   - Implement ensemble of both models for robustness\n")
            f.write("   - Add data augmentation for minority classes\n")
            f.write(
                "   - Use active learning to collect labels for uncertain cases\n")

            f.write("\n4. MONITORING:\n")
            f.write("   - Track model performance on new data\n")
            f.write("   - Monitor for data drift (language distribution changes)\n")
            f.write("   - Retrain quarterly with new examples\n")
            f.write("   - Log attention patterns for debugging\n\n")

            # MLflow info
            f.write("-"*70 + "\n")
            f.write("MODEL VERSIONING (MLflow)\n")
            f.write("-"*70 + "\n\n")
            f.write(f"Experiment: urgency_classification\n")
            f.write(f"Tracking URI: ./mlruns\n\n")
            f.write("View experiments: mlflow ui\n")
            f.write("Compare runs in MLflow UI to see:\n")
            f.write("  - Training curves and loss\n")
            f.write("  - Hyperparameter comparison\n")
            f.write("  - Metric evolution over time\n")
            f.write("  - Model artifacts and checkpoints\n\n")

            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")

        print(f"\n✓ Report saved to: {report_path}\n")
        return report_path

    def run_full_training_pipeline(self):
        """Execute complete training pipeline."""
        print("\n" + "="*70)
        print("STARTING FULL TRAINING PIPELINE")
        print("="*70 + "\n")

        # Load data
        train_df, val_df, test_df = self.load_and_prepare_data()

        # Train Model 1: XLM-RoBERTa (Best for multilingual)
        print("\n" + "="*70)
        print("TRAINING MODEL 1: XLM-RoBERTa")
        print("="*70)
        model1_results = self.train_bert_model(
            train_df, val_df, test_df,
            model_name="xlm-roberta-base"
        )

        # Train Model 2: BERT Multilingual (for comparison)
        print("\n" + "="*70)
        print("TRAINING MODEL 2: BERT Multilingual")
        print("="*70)
        model2_results = self.train_bert_model(
            train_df, val_df, test_df,
            model_name="bert-base-multilingual-cased"
        )

        # Generate explanations for both models
        model1_explanations = self.explain_bert_predictions(
            model1_results['model'],
            model1_results['tokenizer'],
            test_df,
            model_name="XLM-RoBERTa"
        )

        model2_explanations = self.explain_bert_predictions(
            model2_results['model'],
            model2_results['tokenizer'],
            test_df,
            model_name="BERT"
        )

        # Create visualizations
        self.create_visualizations(model1_results, model2_results)

        # Generate report
        report_path = self.generate_report(
            model1_results,
            model2_results,
            model1_explanations,
            model2_explanations
        )

        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n✓ Models saved to: {self.output_dir}")
        print(f"✓ Reports saved to: {self.reports_dir}")
        print(f"✓ MLflow tracking: ./mlruns")
        print(f"\nTo view MLflow UI, run: mlflow ui")
        print("="*70 + "\n")

        return {
            'model1_results': model1_results,
            'model2_results': model2_results,
            'report_path': report_path
        }


def main():
    """Main execution function."""
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "clean.csv")
    output_dir = os.path.join(script_dir, "trained_models")
    reports_dir = os.path.join(script_dir, "reports")

    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please run clean_data.py first to generate the cleaned dataset.")
        return

    # Initialize trainer
    trainer = ModelTrainer(
        data_path=data_path,
        output_dir=output_dir,
        reports_dir=reports_dir
    )

    # Run training pipeline
    results = trainer.run_full_training_pipeline()

    print("\n🎉 Training complete! Check the reports folder for detailed analysis.")


if __name__ == "__main__":
    main()
