#!/usr/bin/env python3
"""
Analyze MLflow runs to find the best model and identify overfitting.
"""
import mlflow
import pandas as pd
import os

os.chdir('/Users/svantipuric/FIDIT_projects/PULSE/urgency_classificator')
mlflow.set_tracking_uri('./mlruns')

# Get all experiments
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

all_runs = []
for exp in experiments:
    runs = client.search_runs(exp.experiment_id, order_by=[
                              "metrics.test_f1 DESC"])
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params

        # Calculate train-test gap to detect overfitting
        train_f1 = metrics.get('train_f1', 0)
        test_f1 = metrics.get('test_f1', 0)
        overfit_gap = train_f1 - test_f1 if train_f1 > 0 and test_f1 > 0 else None

        all_runs.append({
            'run_id': run.info.run_id[:8],
            'model': params.get('model_name', 'unknown'),
            'test_f1': test_f1,
            'train_f1': train_f1,
            'overfit_gap': overfit_gap,
            'f1_low': metrics.get('test_f1_low', 0),
            'f1_medium': metrics.get('test_f1_medium', 0),
            'f1_high': metrics.get('test_f1_high', 0),
            'accuracy': metrics.get('test_accuracy', 0),
            'learning_rate': params.get('learning_rate', 'N/A'),
            'batch_size': params.get('batch_size', 'N/A'),
            'epochs': params.get('num_train_epochs', 'N/A'),
        })

df = pd.DataFrame(all_runs)

if len(df) == 0:
    print("No runs found in MLflow!")
else:
    df = df.sort_values('test_f1', ascending=False)

    print('\n' + '='*100)
    print('BEST MODELS BY TEST F1 SCORE (Top 15)')
    print('='*100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(df.head(15).to_string(index=False))

    # Find best model with balanced performance
    print('\n' + '='*100)
    print('ANALYSIS: BEST BALANCED MODEL (Good F1 + Low Overfitting)')
    print('='*100)

    # Filter models with good test F1 and low overfitting
    df_good = df[(df['test_f1'] > 0.85) & (df['overfit_gap'] < 0.05)]

    if len(df_good) > 0:
        best_model = df_good.iloc[0]
        print(f"\n🏆 BEST MODEL:")
        print(f"   Run ID: {best_model['run_id']}")
        print(f"   Model: {best_model['model']}")
        print(f"   Test F1: {best_model['test_f1']:.4f}")
        print(f"   Train F1: {best_model['train_f1']:.4f}")
        print(f"   Overfitting Gap: {best_model['overfit_gap']:.4f}")
        print(f"   F1 Low: {best_model['f1_low']:.4f}")
        print(f"   F1 Medium: {best_model['f1_medium']:.4f}")
        print(f"   F1 High: {best_model['f1_high']:.4f}")
        print(f"   Learning Rate: {best_model['learning_rate']}")
        print(f"   Batch Size: {best_model['batch_size']}")
        print(f"   Epochs: {best_model['epochs']}")
    else:
        print("\n⚠️ No models found with F1 > 0.85 and overfitting gap < 0.05")
        print("   Showing best by test F1 regardless of overfitting:")
        best_model = df.iloc[0]
        print(f"\n   Run ID: {best_model['run_id']}")
        print(f"   Test F1: {best_model['test_f1']:.4f}")
        print(f"   Overfitting Gap: {best_model['overfit_gap']:.4f}")

    # Overfitting analysis
    print('\n' + '='*100)
    print('OVERFITTING ANALYSIS')
    print('='*100)
    df_with_gap = df[df['overfit_gap'].notna()]
    if len(df_with_gap) > 0:
        avg_gap = df_with_gap['overfit_gap'].mean()
        print(f"Average Train-Test F1 Gap: {avg_gap:.4f}")
        print(
            f"Models with gap < 0.03 (good generalization): {len(df_with_gap[df_with_gap['overfit_gap'] < 0.03])}")
        print(
            f"Models with gap > 0.10 (overfitting): {len(df_with_gap[df_with_gap['overfit_gap'] > 0.10])}")
