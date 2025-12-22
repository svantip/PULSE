#!/usr/bin/env python3
"""
Example usage of the PULSE classification system.
This script demonstrates how to use both emotion and urgency classifiers.
"""

# Example 1: Using the Emotion Classifier for fine-tuning
print("=" * 60)
print("Example 1: Fine-tuning the Emotion Classifier")
print("=" * 60)
print("""
from emotion_classificator import EmotionClassifier

# Initialize classifier
classifier = EmotionClassifier(model_name="distilbert-base-uncased")
classifier.load_model()

# Prepare your datasets (not shown here)
# train_dataset = your_train_data
# eval_dataset = your_eval_data

# Fine-tune the model
model_path = classifier.fine_tune(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    output_dir="./emotion_models"
)

print(f"Model saved to: {model_path}")
""")

# Example 2: Using the Emotion Predictor
print("\n" + "=" * 60)
print("Example 2: Making Emotion Predictions")
print("=" * 60)
print("""
from emotion_classificator import EmotionPredictor

# Initialize predictor (uses base model if no fine-tuned model available)
predictor = EmotionPredictor()

# Make a prediction with explanation
result = predictor.predict(
    text="I am so happy today!",
    explain=True
)

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"All scores: {result['all_scores']}")

if 'explanation' in result:
    print("Token importance:")
    for token_info in result['explanation']['token_importance'][:5]:
        print(f"  {token_info['token']}: {token_info['importance']:.4f}")
""")

# Example 3: Using the API endpoints
print("\n" + "=" * 60)
print("Example 3: Using the REST API")
print("=" * 60)
print("""
# Start the server
python controller.py

# The server will be running at http://localhost:8000

# Example API request using curl:
curl -X POST "http://localhost:8000/emotion/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "I am so happy today!", "explain": true}'

# Example API request using Python requests:
import requests

response = requests.post(
    "http://localhost:8000/emotion/predict",
    json={"text": "I am so happy today!", "explain": True}
)

print(response.json())
# Output:
# {
#   "emotion": "joy",
#   "confidence": 0.92,
#   "all_scores": {
#     "joy": 0.92,
#     "sadness": 0.02,
#     ...
#   },
#   "explanation": {...}
# }
""")

# Example 4: Urgency Classification
print("\n" + "=" * 60)
print("Example 4: Making Urgency Predictions")
print("=" * 60)
print("""
from urgency_classificator import UrgencyPredictor

# Initialize predictor
predictor = UrgencyPredictor()

# Make a prediction
result = predictor.predict(
    text="URGENT: Server is down and users cannot access the system!",
    explain=False
)

print(f"Urgency: {result['urgency']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"All scores: {result['all_scores']}")
""")

# Example 5: Batch predictions
print("\n" + "=" * 60)
print("Example 5: Batch Predictions via API")
print("=" * 60)
print("""
# Batch prediction endpoint
curl -X POST "http://localhost:8000/emotion/predict/batch" \\
     -H "Content-Type: application/json" \\
     -d '{
       "texts": [
         "I love this product!",
         "This is terrible.",
         "I am feeling okay today."
       ],
       "explain": false
     }'

# Response:
# {
#   "predictions": [
#     {"emotion": "joy", "confidence": 0.95, ...},
#     {"emotion": "anger", "confidence": 0.88, ...},
#     {"emotion": "neutral", "confidence": 0.76, ...}
#   ]
# }
""")

# Example 6: Model Versioning with MLflow
print("\n" + "=" * 60)
print("Example 6: Model Versioning with MLflow")
print("=" * 60)
print("""
# Models are automatically versioned during fine-tuning
# Each training run creates a timestamped version

# To use a specific model version:
export EMOTION_MODEL_PATH="./emotion_models/v_20231222_120000"
python controller.py

# To view MLflow experiments:
mlflow ui --backend-store-uri ./mlruns

# Then open http://localhost:5000 in your browser
""")

print("\n" + "=" * 60)
print("For more information, see the README.md file")
print("=" * 60)
