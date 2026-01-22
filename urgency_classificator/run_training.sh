#!/bin/bash
# Quick Start Script for Urgency Classificator
# Run from the urgency_classificator/ directory

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        URGENCY CLASSIFICATOR - QUICK START                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found. Please run this script from the urgency_classificator/ directory."
    exit 1
fi

echo "📋 Step 1: Data Preparation"
echo "────────────────────────────────────────────────────────────────"

# Check if clean data exists
if [ -f "data/clean.csv" ]; then
    echo "✅ Clean data already exists: data/clean.csv"
    read -p "   Re-run data preparation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping data preparation..."
    else
        echo "   Running clean_data.py..."
        python3 scripts/data_prep/clean_data.py
    fi
else
    echo "⚠️  Clean data not found. Running data preparation..."
    
    # Check if raw data exists
    if [ ! -f "data/raw.csv" ]; then
        echo "   📥 Downloading dataset..."
        python3 scripts/data_prep/download_dataset.py
        
        echo "   🔗 Merging datasets..."
        python3 scripts/data_prep/merge_datasets.py
    fi
    
    echo "   🧹 Cleaning data..."
    python3 scripts/data_prep/clean_data.py
fi

echo ""
echo "🚂 Step 2: Model Training"
echo "────────────────────────────────────────────────────────────────"
echo "   Configuration: config.yaml"
echo "   Models: XLM-RoBERTa + BERT Multilingual"
echo ""
read -p "   Start training? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "   🎯 Starting training (this may take several hours)..."
    python3 train.py
else
    echo "   Skipping training."
    echo ""
    echo "   To train manually, run:"
    echo "   $ python3 train.py"
fi

echo ""
echo "📊 Step 3: View Results"
echo "────────────────────────────────────────────────────────────────"
echo "   Reports: reports/training_report_*.txt"
echo "   Figures: reports/figures/"
echo ""
echo "   To view MLflow UI:"
echo "   $ mlflow ui --port 5001"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ QUICK START COMPLETE                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
