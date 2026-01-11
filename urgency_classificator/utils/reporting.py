"""
Report generation utilities for training results.
"""
import os
from datetime import datetime
from typing import Dict, List, Any


def generate_training_report(
    model1_results: Dict[str, Any],
    model2_results: Dict[str, Any],
    model1_name: str,
    model2_name: str,
    model1_explanations: List[Dict],
    model2_explanations: List[Dict],
    data_path: str,
    output_dir: str,
    timestamp: str
) -> str:
    """Generate comprehensive training report."""
    print("="*70)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*70)

    report_path = os.path.join(output_dir, f'training_report_{timestamp}.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        _write_header(f, data_path)
        _write_overview(f, model1_results, model2_results,
                        model1_name, model2_name)
        _write_performance_comparison(
            f, model1_results, model2_results, model1_name, model2_name)
        _write_model_analysis(
            f, model1_results, model2_results, model1_name, model2_name)
        _write_explanations(f, model1_explanations,
                            model2_explanations, model1_name, model2_name)
        _write_recommendations(
            f, model1_results, model2_results, model1_name, model2_name)
        _write_mlflow_info(f)

    print(f"\n✓ Report saved to: {report_path}\n")
    return report_path


def _write_header(f, data_path: str):
    """Write report header."""
    f.write("="*70 + "\n")
    f.write("URGENCY CLASSIFICATION MODEL TRAINING REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {data_path}\n\n")


def _write_overview(f, model1_results: Dict, model2_results: Dict, model1_name: str, model2_name: str):
    """Write training overview section."""
    f.write("-"*70 + "\n")
    f.write("TRAINING OVERVIEW\n")
    f.write("-"*70 + "\n")
    f.write("Two transformer models were trained and compared:\n\n")

    # Model 1 info
    if "roberta" in model1_name.lower():
        f.write("1. XLM-RoBERTa (xlm-roberta-base)\n")
        f.write("   - 12 layers, 12 attention heads, 768 hidden size\n")
        f.write("   - 125M parameters\n")
        f.write("   - Optimized for multilingual tasks\n\n")
    else:
        f.write("1. BERT Multilingual (bert-base-multilingual-cased)\n")
        f.write("   - 12 layers, 12 attention heads, 768 hidden size\n")
        f.write("   - 110M parameters\n")
        f.write("   - Multilingual BERT baseline\n\n")

    # Model 2 info
    if "roberta" in model2_name.lower():
        f.write("2. XLM-RoBERTa (xlm-roberta-base)\n")
        f.write("   - 12 layers, 12 attention heads, 768 hidden size\n")
        f.write("   - 125M parameters\n")
        f.write("   - Optimized for multilingual tasks\n\n")
    else:
        f.write("2. BERT Multilingual (bert-base-multilingual-cased)\n")
        f.write("   - 12 layers, 12 attention heads, 768 hidden size\n")
        f.write("   - 110M parameters\n")
        f.write("   - Multilingual BERT baseline\n\n")

    f.write("Both models used gradient-based optimization:\n")
    f.write("- Optimizer: AdamW\n")
    f.write("- Training: Backpropagation through transformer layers\n")
    f.write("- Loss: Weighted Cross-Entropy (class-balanced)\n\n")


def _write_performance_comparison(f, model1_results: Dict, model2_results: Dict, model1_name: str, model2_name: str):
    """Write performance comparison section."""
    f.write("-"*70 + "\n")
    f.write("PERFORMANCE COMPARISON\n")
    f.write("-"*70 + "\n\n")

    f.write("Overall Metrics:\n")
    f.write(
        f"{'Metric':<15} {model1_name:<15} {model2_name:<15} {'Difference':<12}\n")
    f.write("-"*70 + "\n")

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for metric, name in zip(metrics, metric_names):
        model1_val = model1_results['metrics'][metric]
        model2_val = model2_results['metrics'][metric]
        diff = model1_val - model2_val
        f.write(f"{name:<15} {model1_val:<15.4f} {model2_val:<15.4f} {diff:+.4f}\n")

    f.write("\n\nPer-Class Performance (F1-Score):\n")
    f.write(
        f"{'Class':<15} {model1_name:<15} {model2_name:<15} {'Difference':<12}\n")
    f.write("-"*70 + "\n")

    for i, label in enumerate(['Low', 'Medium', 'High']):
        model1_val = model1_results['metrics']['class_f1'][i]
        model2_val = model2_results['metrics']['class_f1'][i]
        diff = model1_val - model2_val
        f.write(
            f"{label:<15} {model1_val:<15.4f} {model2_val:<15.4f} {diff:+.4f}\n")


def _write_model_analysis(f, model1_results: Dict, model2_results: Dict, model1_name: str, model2_name: str):
    """Write model analysis section."""
    f.write("\n\n" + "-"*70 + "\n")
    f.write("MODEL ANALYSIS\n")
    f.write("-"*70 + "\n\n")

    model1_f1 = model1_results['metrics']['f1']
    model2_f1 = model2_results['metrics']['f1']

    if model1_f1 > model2_f1:
        winner = model1_name
        diff_pct = ((model1_f1 - model2_f1) / model2_f1) * 100
    else:
        winner = model2_name
        diff_pct = ((model2_f1 - model1_f1) / model1_f1) * 100

    f.write(f"🏆 BEST MODEL: {winner}\n")
    f.write(f"Performance advantage: {diff_pct:.2f}%\n\n")

    if "roberta" in winner.lower():
        f.write("Why XLM-RoBERTa Performs Well:\n")
        f.write("- Optimized training objectives for cross-lingual transfer\n")
        f.write("- Better tokenization for multilingual text\n")
        f.write("- Improved handling of class imbalance\n\n")
    else:
        f.write("Why BERT Performs Well:\n")
        f.write("- Proven multilingual capabilities\n")
        f.write("- Effective for classification tasks\n")
        f.write("- Good balance of precision and recall\n\n")


def _write_explanations(f, model1_explanations: List[Dict], model2_explanations: List[Dict],
                        model1_name: str, model2_name: str):
    """Write explainability section."""
    f.write("-"*70 + "\n")
    f.write("MODEL EXPLAINABILITY (Attention-based)\n")
    f.write("-"*70 + "\n\n")

    f.write(f"{model1_name} Explanations:\n")
    f.write("-"*70 + "\n")
    for i, exp in enumerate(model1_explanations[:5], 1):
        f.write(f"\nExample {i}:\n")
        f.write(f"Text: {exp['text']}\n")
        f.write(f"True: {exp['true_label']} | Predicted: {exp['predicted_label']} "
                f"| Confidence: {exp['confidence']:.3f} | Correct: {exp['correct']}\n")
        f.write(f"Important tokens:\n")
        for token, weight in exp['important_tokens']:
            f.write(f"  - '{token}': {weight:.4f}\n")

    f.write(f"\n\n{model2_name} Explanations:\n")
    f.write("-"*70 + "\n")
    for i, exp in enumerate(model2_explanations[:5], 1):
        f.write(f"\nExample {i}:\n")
        f.write(f"Text: {exp['text']}\n")
        f.write(f"True: {exp['true_label']} | Predicted: {exp['predicted_label']} "
                f"| Confidence: {exp['confidence']:.3f} | Correct: {exp['correct']}\n")
        f.write(f"Important tokens:\n")
        for token, weight in exp['important_tokens']:
            f.write(f"  - '{token}': {weight:.4f}\n")


def _write_recommendations(f, model1_results: Dict, model2_results: Dict, model1_name: str, model2_name: str):
    """Write recommendations section."""
    f.write("\n\n" + "-"*70 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("-"*70 + "\n\n")

    model1_f1 = model1_results['metrics']['f1']
    model2_f1 = model2_results['metrics']['f1']
    winner = model1_name if model1_f1 > model2_f1 else model2_name

    f.write(f"1. DEPLOYMENT: Use {winner} model for production\n\n")
    f.write("2. MONITORING:\n")
    f.write("   - Track model performance on new data\n")
    f.write("   - Monitor for data drift\n")
    f.write("   - Retrain quarterly with new examples\n\n")
    f.write("3. IMPROVEMENTS:\n")
    f.write("   - Collect more training data for minority classes\n")
    f.write("   - Implement active learning for uncertain cases\n")
    f.write("   - Consider ensemble methods\n\n")


def _write_mlflow_info(f):
    """Write MLflow information."""
    f.write("-"*70 + "\n")
    f.write("MODEL VERSIONING (MLflow)\n")
    f.write("-"*70 + "\n\n")
    f.write("Experiment: urgency_classification\n")
    f.write("Tracking URI: ./mlruns\n\n")
    f.write("View experiments: mlflow ui\n")
    f.write("Compare runs in MLflow UI\n\n")
    f.write("="*70 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*70 + "\n")
