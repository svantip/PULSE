# Urgency Classificator

ML-powered urgency classification system for customer support tickets.

## 📁 Project Structure

```
urgency_classificator/
├── config.yaml                   # 🎛️  Central configuration (single source of truth)
├── README.md                     # 📖 This file
│
├── scripts/                      # 📜 Executable scripts organized by purpose
│   ├── data_prep/               # 1️⃣  Data preparation pipeline
│   │   ├── download_dataset.py  #     ⬇️  Download from Kaggle
│   │   ├── merge_datasets.py    #     🔗 Merge multiple CSVs
│   │   ├── clean_data.py        #     🧹 Clean and prepare data
│   │   └── analyze_datasets.py  #     📊 Analyze statistics
│   │
│   ├── training/                # 2️⃣  Model training
│   │   ├── train_models.py      #     🚂 **MAIN TRAINING SCRIPT**
│   │   └── analyse.py           #     🔍 Training analysis utilities
│   │
│   └── inference/               # 3️⃣  Model inference & deployment
│       ├── predictor.py         #     🔮 Prediction interface (import this)
│       ├── model.py             #     🧠 Model class definitions
│       └── generate_explanations.py  # 💡 Attention-based explanations
│
├── utils/                        # 🛠️  Reusable utility modules
│   ├── config_loader.py         #     📋 Config loading
│   ├── visualization.py         #     📈 Plotting and charts
│   └── reporting.py             #     📝 Report generation
│
└── 📂 Data & Outputs
    ├── data/                     # 💾 Raw and cleaned datasets
    ├── trained_models/           # 🎯 Saved model checkpoints
    ├── reports/                  # 📄 Training reports and figures
    ├── mlruns/                   # 📊 MLflow experiment tracking
    └── visualizations/           # 📸 Analysis plots
```

## 🚀 Quick Start

### 1. Data Preparation (One-time setup)

```bash
# Download dataset from Kaggle
python scripts/data_prep/download_dataset.py

# Merge multiple CSV files (if needed)
python scripts/data_prep/merge_datasets.py

# Clean and prepare data
python scripts/data_prep/clean_data.py
```

### 2. Configure Training

Edit `config.yaml` to set:

- Models to train (XLM-RoBERTa, BERT, etc.)
- Hyperparameters (learning rate, batch size, epochs)
- Class weights for handling imbalance
- Paths and experiment name

**This is your single source of truth!** All scripts read from here.

### 3. Train Models

```bash
python scripts/training/train_models.py
```

This will:

- ✅ Train all models defined in config.yaml
- ✅ Log metrics to MLflow automatically
- ✅ Generate visualizations (confusion matrices, performance charts)
- ✅ Create comprehensive training report
- ✅ Save best model checkpoints

### 4. View Results

```bash
mlflow ui --port 5001
```

Open http://localhost:5001 to compare experiments.

Check `reports/training_report_<timestamp>.txt` for detailed analysis.

## 📊 What Gets Generated

After training, you'll find:

### Reports (`reports/`)

- `training_report_<timestamp>.txt` - Comprehensive analysis and recommendations
- `figures/confusion_matrices_<timestamp>.png` - Model comparison
- `figures/performance_comparison_<timestamp>.png` - Metrics bar charts
- `figures/per_class_f1_<timestamp>.png` - Per-class performance

### Models (`trained_models/`)

- `xlm-roberta_final_<timestamp>/` - Best XLM-RoBERTa checkpoint
- `bert_final_<timestamp>/` - Best BERT checkpoint

### MLflow (`mlruns/`)

- All hyperparameters logged
- Training and evaluation metrics
- Per-class precision, recall, F1 scores
- Confusion matrices as artifacts

## 🎯 Using Trained Models

```python
import sys
sys.path.append('scripts/inference')
from predictor import UrgencyPredictor

# Load trained model
predictor = UrgencyPredictor("trained_models/xlm-roberta_final_20260111_141514")

# Predict urgency
urgency = predictor.predict("Customer unable to login, losing revenue!")
print(urgency)  # "high"

# Get confidence scores
result = predictor.predict_with_confidence("Please update my address")
print(result)  # {"urgency": "low", "confidence": 0.92}
```

## ⚙️ Configuration Reference (config.yaml)

### Models

```yaml
MODELS:
  - NAME: "xlm-roberta-base"
    ALIAS: "xlm-roberta"
  - NAME: "bert-base-multilingual-cased"
    ALIAS: "bert-multilingual"
```

### Training Hyperparameters

```yaml
TRAINING:
  LEARNING_RATE: 3e-5
  BATCH_SIZE: 8
  NUM_EPOCHS: 5
  EARLY_STOPPING_PATIENCE: 7
  # ... see config.yaml for full list
```

### Class Weights (for imbalance)

```yaml
CLASS_WEIGHTS:
  LOW: 1.3 # Boost low urgency predictions
  MEDIUM: 1.3 # Boost medium urgency predictions
  HIGH: 0.9 # Reduce high urgency predictions
```

## 🔍 Model Explainability

Generate attention-based explanations showing which words influenced predictions:
scripts/inference/

```bash
python generate_explanations.py
```

This uses the model's attention weights (more interpretable than SHAP for transformers).

## 📈 Experiment Tracking

All training runs are automatically tracked:

- ✅ Hyperparameters (learning rate, batch size, epochs, etc.)
- ✅ Overall metrics (accuracy, precision, recall, F1)
- ✅ Per-class metrics (low/medium/high precision, recall, F1)
- ✅ Confusion matrices
- ✅ Training loss curves
- ✅ Model artifacts

Compare experiments side-by-side in MLflow UI.

## 🛠️ Advanced Usage

### Analyze Dataset

```bash
python scripts/data_prep/analyze_datasets.py
```

Shows:

- Class distribution
- Text length statistics
- Language breakdown
- Sample examples

### Custom Training

Modify `config.yaml` then run:

```bash
python train_models.py
```

No code changes needed - just update the config!

## 📚 Script Reference

| Script                     | Purpose                           | When to Run                         |
| -------------------------- | --------------------------------- | ----------------------------------- |
| `download_dataset.py`      | Download from Kaggle              | One-time setup                      |
| `merge_datasets.py`        | Merge multiple CSVs               | One-time setup (if needed)          |
| `clean_data.py`            | Prepare training data             | One-time setup or when data changes |
| `train_models.py`          | **Train models**                  | **Main workflow**                   |
| `generate_explanations.py` | Generate attention visualizations | After training                      |
| `predictor.py`             | Use model in production           | Import in your code                 |

## 🎓 Model Comparison

The system automatically compares models and recommends the best one based on:

- Overall F1 score
- Per-class performance (balanced recall across low/medium/high)
- Precision-recall trade-offs
- Training efficiency

Check the training report for the winner and deployment recommendation.

## 🐛 Troubleshooting

### MPS Device Error (Apple Silicon)

Already handled - models automatically fall back to CPU for attention extraction.

### Out of Memory

Reduce `BATCH_SIZE` in config.yaml (default: 8 for M3 MacBook Pro).

### MLflow UI Not Showing Runs

```bash
cd urgency_classificator
mlflow ui --backend-store-uri ./mlruns --port 5001
```

### Poor Performance on Specific Class

Adjust `CLASS_WEIGHTS` in config.yaml to boost underperforming classes.
python clean_data.py

# Analyze dataset (optional)

python analyze_datasets.py

````

### 2. Train Models

```bash
# Train XLM-RoBERTa and BERT models
python train_models.py

# Monitor training with MLflow
mlflow ui
# Then open http://localhost:5000
````

### 3. Make Predictions

```python
from predictor import UrgencyPredictor

predictor = UrgencyPredictor(model_path="trained_models/xlm-roberta_final_20260111_141514")
result = predictor.predict("Customer is very angry and demanding refund immediately!")
print(result)  # {'urgency': 'high', 'confidence': 0.94}
```

### 4. Generate Explanations

```bash
# Generate attention-based explanations for model predictions
python generate_explanations.py
```

## ⚙️ Configuration

All settings are in **`config.yaml`**:

- **Models**: Which transformer models to train
- **Labels**: Urgency levels (low/medium/high)
- **Training**: Hyperparameters (learning rate, batch size, epochs)
- **Class Weights**: Balance handling for imbalanced classes
- **Paths**: Data, models, and reports directories

Edit `config.yaml` to customize training.

## 📊 Workflow

```
download_dataset.py → merge_datasets.py → clean_data.py → train_models.py → predictor.py
                                              ↓
                                      analyze_datasets.py
                                              ↓
                                      generate_explanations.py
```

## 🎯 Scripts Guide

| Script                     | Purpose                     | When to Run                       |
| -------------------------- | --------------------------- | --------------------------------- |
| `download_dataset.py`      | Download data from Kaggle   | Once (initial setup)              |
| `merge_datasets.py`        | Combine multiple CSV files  | Once (if you have multiple files) |
| `clean_data.py`            | Prepare data for training   | Once per dataset                  |
| `analyze_datasets.py`      | Explore data statistics     | Anytime (for insights)            |
| `train_models.py`          | **Train urgency models**    | **Main training run**             |
| `generate_explanations.py` | Generate model explanations | After training                    |
| `predictor.py`             | Make predictions            | In production/API                 |

## 📈 MLflow Tracking

All training runs are logged to MLflow:

```bash
# View experiments in browser
mlflow ui

# Compare model performance
# Navigate to http://localhost:5000
# Compare runs, view metrics, download models
```

## 🧪 Current Models

1. **XLM-RoBERTa** (`xlm-roberta-base`)

   - Multilingual, optimized for cross-lingual tasks
   - 125M parameters
   - Primary model

2. **BERT Multilingual** (`bert-base-multilingual-cased`)
   - Strong multilingual baseline
   - 110M parameters
   - Comparison model

## 📝 Output Files

After training:

```
reports/
├── training_report_20260111_141514.txt  # Comprehensive report
└── figures/
    ├── confusion_matrices_*.png          # Confusion matrices
    ├── performance_comparison_*.png      # Metrics comparison
    └── per_class_f1_*.png                # Per-class F1 scores

trained_models/
├── xlm-roberta_final_20260111_141514/   # Best XLM-RoBERTa model
└── bert_final_20260111_141514/          # Best BERT model

mlruns/
└── [experiment_id]/                     # MLflow tracking data
```

## 🔧 Development

To modify training settings, edit `config.yaml`:

```yaml
TRAINING:
  LEARNING_RATE: 3e-5
  BATCH_SIZE: 8
  NUM_EPOCHS: 5
  # ... more settings
```

All scripts read from this single source of truth.

## 📦 Dependencies

See `requirements.txt` in project root.

## 🤝 Contributing

When adding new scripts:

1. Update this README with script purpose
2. Add configuration to `config.yaml` if needed
3. Use `config_loader.py` to read settings
4. Group scripts by purpose (1️⃣2️⃣3️⃣4️⃣)
