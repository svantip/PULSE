# PULSE
PULSE - Priority Using Language, Sentiment & Escalation

An ML-powered classification system for emotion and urgency detection with explainability and model versioning.

## 🏗️ Project Structure

```
PULSE/
├── controller.py                 # Main FastAPI controller
├── emotion_classificator/        # Emotion classification module
│   ├── __init__.py
│   ├── model.py                 # Model fine-tuning with MLflow versioning
│   ├── predictor.py             # Prediction with explainability
│   └── config.yaml              # Configuration
├── urgency_classificator/        # Urgency classification module
│   ├── __init__.py
│   ├── model.py                 # Model fine-tuning with MLflow versioning
│   ├── predictor.py             # Prediction with explainability
│   └── config.yaml              # Configuration
└── requirements.txt             # Python dependencies
```

## 🚀 Features

- **Dual Classification**: Separate models for emotion and urgency detection
- **Fine-tuning Support**: Built on transformer models (DistilBERT) with easy fine-tuning
- **Explainability**: Gradient-based token importance for model interpretability
- **Model Versioning**: MLflow integration for experiment tracking and model versioning
- **REST API**: FastAPI-based endpoints for predictions
- **Batch Processing**: Support for single and batch predictions

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/svantip/PULSE.git
cd PULSE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Starting the API Server

```bash
python controller.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Emotion Prediction
```bash
POST /emotion/predict
Content-Type: application/json

{
  "text": "I am so happy today!",
  "explain": true
}
```

Response:
```json
{
  "emotion": "joy",
  "confidence": 0.92,
  "all_scores": {
    "joy": 0.92,
    "sadness": 0.02,
    "anger": 0.01,
    "fear": 0.02,
    "surprise": 0.02,
    "neutral": 0.01
  },
  "explanation": {
    "method": "gradient_based",
    "token_importance": [...],
    "description": "Token importance based on gradient magnitudes"
  }
}
```

#### 2. Urgency Prediction
```bash
POST /urgency/predict
Content-Type: application/json

{
  "text": "URGENT: Server is down and users cannot access the system!",
  "explain": false
}
```

Response:
```json
{
  "urgency": "critical",
  "confidence": 0.87,
  "all_scores": {
    "low": 0.02,
    "medium": 0.05,
    "high": 0.06,
    "critical": 0.87
  }
}
```

#### 3. Batch Predictions
```bash
POST /emotion/predict/batch
Content-Type: application/json

{
  "texts": ["I love this!", "This is terrible"],
  "explain": false
}
```

#### 4. Health Check
```bash
GET /health
```

## 🔧 Model Fine-tuning

### Emotion Classifier

```python
from emotion_classificator import EmotionClassifier

# Initialize classifier
classifier = EmotionClassifier(model_name="distilbert-base-uncased")
classifier.load_model()

# Fine-tune on your dataset
model_path = classifier.fine_tune(
    train_dataset=your_train_dataset,
    eval_dataset=your_eval_dataset,
    output_dir="./emotion_models"
)
```

### Urgency Classifier

```python
from urgency_classificator import UrgencyClassifier

# Initialize classifier
classifier = UrgencyClassifier(model_name="distilbert-base-uncased")
classifier.load_model()

# Fine-tune on your dataset
model_path = classifier.fine_tune(
    train_dataset=your_train_dataset,
    eval_dataset=your_eval_dataset,
    output_dir="./urgency_models"
)
```

## 🔍 Explainability

The system provides gradient-based explanations showing which tokens/words contributed most to the prediction:

```python
from emotion_classificator import EmotionPredictor

predictor = EmotionPredictor()
result = predictor.predict("I am very happy!", explain=True)

# Access token importance
for token_info in result['explanation']['token_importance']:
    print(f"{token_info['token']}: {token_info['importance']}")
```

## 📊 Model Versioning

Models are versioned using MLflow:

- Each fine-tuning run creates a new version with timestamp
- Metrics and parameters are logged automatically
- Models can be loaded by version or use the latest

```python
# Use specific model version
export EMOTION_MODEL_PATH="./emotion_models/v_20231222_120000"
python controller.py
```

## 🔬 Classification Labels

### Emotion Labels
- joy
- sadness
- anger
- fear
- surprise
- neutral

### Urgency Labels
- low
- medium
- high
- critical

## 🛠️ Technologies

- **FastAPI**: Web framework for API
- **Transformers**: HuggingFace transformers for NLP models
- **PyTorch**: Deep learning framework
- **MLflow**: Model versioning and experiment tracking
- **SHAP**: Model explainability (gradient-based approach)
- **scikit-learn**: Metrics and utilities

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
