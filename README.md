# PULSE - Priority Using Language, Sentiment & Escalation

AI-powered ticket classification system that automatically analyzes customer messages to determine urgency levels and detect emotions, helping support teams prioritize and respond effectively.

## 🎯 What Does PULSE Do?

PULSE provides three main services:

1. **Urgency Classification** - Classifies messages as HIGH, MEDIUM, or LOW urgency
2. **Emotion Detection** - Identifies customer emotions (happy, sad, angry, frustrated, satisfied, neutral)
3. **Slack Integration** - Automatically analyzes incoming Slack messages and posts formatted reports

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

## 🚀 Quick Start & Running Services

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) Slack workspace for integration

### Installation

1. **Navigate to the project directory**

   ```bash
   cd /Users/svantipuric/FIDIT_projects/PULSE
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (for Slack integration)**
   ```bash
   cp .env.example .env
   # Edit .env and add your Slack credentials
   ```

## 🎮 How to Start All Services

### Option 1: Start Everything Together (Recommended for Testing)

Start both the API server and web UI in one command:

```bash
cd scripts
./start_services.sh
```

This will launch:

- **API Server** at http://localhost:8000
- **Gradio Web UI** at http://localhost:7860

Press `Ctrl+C` to stop all services.

### Option 2: Start Services Individually

#### Start the API Server Only

```bash
# Using Python directly
python controller.py

# Or using uvicorn with auto-reload (recommended for development)
uvicorn controller:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at: http://localhost:8000

#### Start the Web UI Only (Gradio)

```bash
python app.py
```

Web interface will be available at: http://localhost:7860

### Option 3: Start with Slack Integration

For Slack bot integration with auto-analysis:

```bash
cd scripts
./quickstart_slack.sh
```

Follow the prompts to configure Slack credentials and ngrok for local development.

## 📖 How to Use PULSE

### 1. Using the Web Interface (Easiest Method)

The Gradio UI is the simplest way to test the classifiers:

1. **Start the services**: `python app.py` or use `./scripts/start_services.sh`
2. **Open your browser**: http://localhost:7860
3. **Enter a message**: Type or paste a customer message
4. **Get results instantly**:
   - Urgency level (HIGH/MEDIUM/LOW)
   - Emotion type (angry, happy, frustrated, etc.)
   - Confidence scores
   - Important keywords highlighted

**Example messages to try:**

- "URGENT: Production server is down!" → HIGH urgency, frustrated/angry
- "The app is running a bit slow today" → MEDIUM urgency, neutral
- "Thanks for the quick fix!" → LOW urgency, happy
- "I've been waiting for 2 hours!" → MEDIUM urgency, frustrated

### 2. Using the API (For Integration)

#### Combined Analysis (Urgency + Emotion)

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "URGENT: Production server is down!",
    "slack_channel": "support-tickets",
    "include_explanation": false
  }'
```

**Response:**

```json
{
  "timestamp": "2026-01-22T19:30:00",
  "text": "URGENT: Production server is down!",
  "urgency": {
    "level": "high",
    "confidence": 0.92,
    "all_scores": { "high": 0.92, "medium": 0.06, "low": 0.02 }
  },
  "emotion": {
    "type": "angry",
    "confidence": 0.88,
    "all_scores": { "angry": 0.88, "frustrated": 0.1, "neutral": 0.02 }
  },
  "priority_flag": {
    "level": "high",
    "escalate": true,
    "reason": "High urgency with angry emotion"
  }
}
```

#### Urgency Classification Only

```bash
curl -X POST http://localhost:8000/urgency/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "The server is not responding"}'
```

#### Emotion Detection Only

```bash
curl -X POST http://localhost:8000/emotion/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "I am so frustrated with this issue"}'
```

#### Health Check

```bash
curl http://localhost:8000/health
```

**Full API documentation available at:**

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Using Slack Integration

Once configured (see [Slack Integration Guide](docs/SLACK_INTEGRATION.md)), the bot will:

1. **Listen** for messages in channels it's invited to
2. **Analyze** each message for urgency and emotion
3. **Post** formatted reports to your support channel

**To use:**

1. Invite the bot: `/invite @Ticket Analyzer`
2. Send a message: `URGENT: Production server is down!`
3. Check your support channel for the analysis report

## 🧪 Testing

Run the test scripts to verify everything works:

```bash
# Test the API endpoints
python tests/test_api.py

# Test Slack integration
python tests/test_slack_integration.py
```

## 🚦 Check Service Status

```bash
# Check if API is running
curl http://localhost:8000/health

# Check processes
ps aux | grep controller
ps aux | grep app.py

# Check ports
lsof -i :8000  # API
lsof -i :7860  # Gradio UI
```

## 🐛 Troubleshooting

### Services Won't Start

```bash
# Check if ports are already in use
lsof -i :8000
lsof -i :7860

# Kill processes if needed
kill -9 <PID>

# Restart services
python controller.py
```

### Models Not Loading

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python version (3.8+ required)
python --version
```

### Slack Bot Not Responding

1. Verify bot is invited: `/invite @Ticket Analyzer`
2. Check `.env` file has correct credentials
3. Ensure server is publicly accessible (use ngrok for local dev)
4. Check server logs for errors

### Import Errors

Run from project root:

```bash
cd /Users/svantipuric/FIDIT_projects/PULSE
python controller.py
```

## 📚 Additional Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Detailed getting started
- **[Slack Integration](docs/SLACK_INTEGRATION.md)** - Complete Slack setup
- **[Project Structure](PROJECT_STRUCTURE.md)** - Organized file structure

## 🚀 Features

- **Dual Classification**: Separate models for emotion and urgency detection
- **Web Interface**: Gradio UI for easy testing
- **REST API**: FastAPI endpoints for integration
- **Slack Integration**: Automated ticket analysis in Slack
- **Explainability**: Gradient-based token importance
- **Model Versioning**: MLflow integration for experiment tracking
- **Batch Processing**: Support for single and batch predictions

All classifier settings are managed through YAML configuration files. No values are hardcoded.

### Multi-Model Comparison Support

Each classifier can compare multiple models to determine which performs best for your use case.

Edit `emotion_classificator/config.yaml` to customize:

```yaml
# Models to compare (define multiple models)
MODELS:
  - NAME: "distilbert-base-uncased"
    ALIAS: "distilbert"
  - NAME: "bert-base-uncased"
    ALIAS: "bert"

NUM_LABELS: 6
LABELS:
  - "joy"
  - "sadness"
  - "anger"
  - "fear"
  - "surprise"
  - "neutral"

# Training parameters
TRAINING:
  LEARNING_RATE: 2e-5
  BATCH_SIZE: 16
  NUM_EPOCHS: 3
  WEIGHT_DECAY: 0.01

# Model versioning with MLflow
VERSIONING:
  ENABLED: true
  MLFLOW_TRACKING_URI: "./mlruns"
  EXPERIMENT_NAME: "emotion_classification"
```

### Urgency Classifier Configuration

Edit `urgency_classificator/config.yaml` similarly for urgency classification.

## 🔧 Model Fine-tuning

### Single Model Training

```python
from emotion_classificator import EmotionClassifier

# Initialize with specific model
classifier = EmotionClassifier(model_alias="bert")
classifier.load_model()

# Fine-tune on your dataset
model_path = classifier.fine_tune(
    train_dataset=your_train_dataset,
    eval_dataset=your_eval_dataset
)
```

### Model Comparison (Automated)

Compare all models defined in config to find the best one:

```python
from emotion_classificator import EmotionClassifier

# Initialize classifier
classifier = EmotionClassifier()

# Compare all models and get results
results = classifier.compare_models(
    train_dataset=your_train_dataset,
    eval_dataset=your_eval_dataset
)

# Results include metrics for each model
print(f"Best model: {results['best_model']}")
for model_alias, metrics in results.items():
    if model_alias != 'best_model':
        print(f"{model_alias}: {metrics['metrics']}")
```

### Emotion Classifier

```python
from emotion_classificator import EmotionClassifier

# Initialize classifier (loads configuration from config.yaml)
classifier = EmotionClassifier()
classifier.load_model()

# Fine-tune on your dataset
model_path = classifier.fine_tune(
    train_dataset=your_train_dataset,
    eval_dataset=your_eval_dataset
)
```

### Urgency Classifier

```python
from urgency_classificator import UrgencyClassifier

# Initialize classifier (loads configuration from config.yaml)
classifier = UrgencyClassifier()
classifier.load_model()

# Fine-tune on your dataset
model_path = classifier.fine_tune(
    train_dataset=your_train_dataset,
    eval_dataset=your_eval_dataset
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
- **Gradient-based explainability**: Token importance using gradient magnitudes
- **scikit-learn**: Metrics and utilities

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
