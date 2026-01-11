"""
Generate explanations for already-trained models without retraining.
"""

import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_models import ModelTrainer


def main():
    """Load trained model and generate explanations."""

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "clean.csv")
    model_path = os.path.join(
        script_dir, "trained_models", "bert_final_20260110_171619")
    reports_dir = os.path.join(script_dir, "reports")

    print("="*70)
    print("LOADING TRAINED MODEL AND GENERATING EXPLANATIONS")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")

    # Load the data
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    label_map = {'low': 0, 'medium': 1, 'high': 2}
    df['label_id'] = df['priority'].map(label_map)

    # Use same split as training (we just need test set)
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label_id']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_id']
    )

    print(f"Test samples: {len(test_df)}")

    # Load the trained model and tokenizer
    print("\nLoading trained model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print("Model loaded successfully!")

    # Create a trainer instance just to use the explanation method
    trainer = ModelTrainer(
        data_path=data_path,
        output_dir=os.path.join(script_dir, "trained_models"),
        reports_dir=reports_dir
    )

    # Generate explanations
    explanations = trainer.explain_bert_predictions(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        model_name="BERT-Final"
    )

    # Save explanations (convert numpy types to Python types for JSON)
    import json
    import numpy as np

    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        return obj

    explanations_serializable = convert_to_serializable(explanations)

    explanations_path = os.path.join(reports_dir, "bert_explanations.json")
    with open(explanations_path, 'w') as f:
        json.dump(explanations_serializable, f, indent=2)

    print(f"\n✅ Explanations saved to: {explanations_path}")
    print(f"Generated {len(explanations_serializable)} explanations")

    # Print a few examples
    print("\n" + "="*70)
    print("SAMPLE EXPLANATIONS")
    print("="*70)
    for i, exp in enumerate(explanations_serializable[:3]):
        print(f"\nExample {i+1}:")
        print(f"Text: {exp['text']}")
        print(
            f"True: {exp['true_label']} | Predicted: {exp['predicted_label']} | Confidence: {exp['confidence']:.3f}")
        print(f"Important tokens: {[(token, f'{score:.4f}')
              for token, score in exp['important_tokens'][:3]]}")


if __name__ == "__main__":
    main()
