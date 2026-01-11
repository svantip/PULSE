"""
Generate useful explanations for already-trained models without retraining.
Uses keyword extraction and error pattern analysis instead of attention weights.
"""

import os
import pandas as pd
import torch
import re
import numpy as np
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split


def extract_urgency_keywords(df, label, top_n=20):
    """Extract most common keywords for each urgency level."""
    texts = df[df['priority'] == label]['body']

    # Extract all words
    all_words = []
    for text in texts:
        words = re.findall(r'\b\w+\b', str(text).lower())
        all_words.extend(words)

    # Common stopwords (German + English)
    stopwords = {
        'der', 'die', 'das', 'und', 'den', 'dem', 'des', 'ein', 'eine', 'einer',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'mit', 'von', 'zu', 'um'
    }

    # Count and filter
    word_counts = Counter(all_words)
    filtered = [(w, c) for w, c in word_counts.items()
                if w not in stopwords and len(w) > 2]

    return sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]


def analyze_misclassifications(predictions_df):
    """Analyze error patterns in misclassified examples."""
    errors = predictions_df[predictions_df['true']
                            != predictions_df['predicted']]

    # Count error types
    error_patterns = errors.groupby(
        ['true', 'predicted']).size().sort_values(ascending=False)

    results = {
        'total_errors': len(errors),
        'error_rate': len(errors) / len(predictions_df),
        'patterns': {},
        'examples': {}
    }

    for (true_label, pred_label), count in error_patterns.head(5).items():
        pattern_key = f"{true_label}_to_{pred_label}"
        results['patterns'][pattern_key] = {
            'count': int(count),
            'percentage': float(count / len(errors) * 100)
        }

        # Get example
        subset = errors[(errors['true'] == true_label) &
                        (errors['predicted'] == pred_label)]
        if len(subset) > 0:
            example = subset.iloc[0]
            results['examples'][pattern_key] = {
                'text': example['text'][:200] + '...' if len(example['text']) > 200 else example['text'],
                'confidence': float(example['confidence']),
                'true': true_label,
                'predicted': pred_label
            }

    return results


def generate_predictions(model, tokenizer, test_df, label_map, max_length=256):
    """Generate predictions for test set."""
    model.eval()
    predictions = []

    id_to_label = {v: k for k, v in label_map.items()}

    print(f"Generating predictions for {len(test_df)} samples...")

    for idx, row in test_df.iterrows():
        text = str(row['body'])
        true_label = row['priority']

        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(logits, dim=1).item()
            confidence = probs[pred_id].item()

        predictions.append({
            'text': text,
            'true': true_label,
            'predicted': id_to_label[pred_id],
            'confidence': confidence,
            'correct': true_label == id_to_label[pred_id]
        })

    return pd.DataFrame(predictions)


def main():
    """Load trained model and generate useful explanations."""

    # Setup paths
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..'))
    data_path = os.path.join(project_root, "data", "clean.csv")
    # Use XLM-RoBERTa final model
    model_path = os.path.join(
        project_root, "trained_models", "xlm-roberta_final_20260111_184600")
    reports_dir = os.path.join(project_root, "reports")

    os.makedirs(reports_dir, exist_ok=True)

    print("="*70)
    print("GENERATING USEFUL MODEL EXPLANATIONS")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")

    # Load the data
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    label_map = {'low': 0, 'medium': 1, 'high': 2}
    df['label_id'] = df['priority'].map(label_map)

    # Use same split as training
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label_id']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_id']
    )

    print(f"Test samples: {len(test_df)}")

    # Load the trained model
    print("\nLoading trained model...")
    # Load tokenizer from base model to avoid compatibility issues
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("Model loaded successfully!")

    # Generate predictions on test set
    predictions_df = generate_predictions(model, tokenizer, test_df, label_map)

    # Calculate overall accuracy
    accuracy = predictions_df['correct'].mean()
    print(f"\nTest Accuracy: {accuracy:.1%}")

    # Analyze errors
    print("\n" + "="*70)
    print("ERROR PATTERN ANALYSIS")
    print("="*70)
    error_analysis = analyze_misclassifications(predictions_df)

    print(
        f"\nTotal errors: {error_analysis['total_errors']} ({error_analysis['error_rate']:.1%})")
    print("\nMost common misclassification patterns:")
    for pattern, data in error_analysis['patterns'].items():
        print(
            f"  {pattern}: {data['count']} cases ({data['percentage']:.1f}%)")

    # Extract urgency keywords
    print("\n" + "="*70)
    print("URGENCY INDICATOR KEYWORDS")
    print("="*70)

    keyword_analysis = {}
    for label in ['low', 'medium', 'high']:
        # Get texts from original df
        texts_df = test_df[test_df['priority'] == label]
        keywords = extract_urgency_keywords(texts_df, label, top_n=15)
        keyword_analysis[label] = keywords

        print(f"\n{label.upper()} urgency keywords:")
        for word, count in keywords[:10]:
            print(f"  {word}: {count} occurrences")

    # Create comprehensive explanation report
    import json

    explanations = {
        'model_path': model_path,
        'test_accuracy': float(accuracy),
        'total_test_samples': len(test_df),
        'error_analysis': error_analysis,
        'urgency_keywords': {
            label: [(word, int(count)) for word, count in keywords]
            for label, keywords in keyword_analysis.items()
        },
        'example_errors': []
    }

    # Add detailed error examples
    for pattern, example_data in error_analysis['examples'].items():
        explanations['example_errors'].append(example_data)

    # Save comprehensive explanation
    explanations_path = os.path.join(reports_dir, "model_explanations.json")
    with open(explanations_path, 'w', encoding='utf-8') as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Explanations saved to: {explanations_path}")

    # Print error examples
    print("\n" + "="*70)
    print("EXAMPLE MISCLASSIFICATIONS")
    print("="*70)
    for example in explanations['example_errors'][:3]:
        print(
            f"\n{example['true'].upper()} → {example['predicted'].upper()} (confidence: {example['confidence']:.1%})")
        print(f"Text: {example['text']}")


if __name__ == "__main__":
    main()
