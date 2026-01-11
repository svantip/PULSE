"""
Visualization utilities for model comparison and analysis.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def create_confusion_matrices(
    model1_results: Dict[str, Any],
    model2_results: Dict[str, Any],
    model1_name: str,
    model2_name: str,
    labels: list,
    output_path: str
):
    """Create side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (results, title) in enumerate([
        (model1_results, f'{model1_name} Model'),
        (model2_results, f'{model2_name} Model')
    ]):
        cm = results['metrics']['confusion_matrix']
        label_names = [label.capitalize() for label in labels]
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
            ax=axes[idx]
        )
        axes[idx].set_title(f'{title}\nConfusion Matrix',
                            fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_performance_comparison(
    model1_results: Dict[str, Any],
    model2_results: Dict[str, Any],
    model1_name: str,
    model2_name: str,
    output_path: str
):
    """Create performance comparison bar chart."""
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
                   label=model1_name, color='#3498db')
    bars2 = ax.bar(x + width/2, model2_metrics, width,
                   label=model2_name, color='#e74c3c')

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
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_per_class_comparison(
    model1_results: Dict[str, Any],
    model2_results: Dict[str, Any],
    model1_name: str,
    model2_name: str,
    labels: list,
    output_path: str
):
    """Create per-class F1 score comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    label_names = [label.capitalize() for label in labels]
    model1_f1 = model1_results['metrics']['class_f1']
    model2_f1 = model2_results['metrics']['class_f1']

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, model1_f1, width,
                   label=model1_name, color='#3498db')
    bars2 = ax.bar(x + width/2, model2_f1, width,
                   label=model2_name, color='#e74c3c')

    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Class F1-Score Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
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
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_all_visualizations(
    model1_results: Dict[str, Any],
    model2_results: Dict[str, Any],
    model1_name: str,
    model2_name: str,
    labels: list,
    output_dir: str,
    timestamp: str
):
    """Create all visualization plots."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Confusion matrices
    create_confusion_matrices(
        model1_results, model2_results,
        model1_name, model2_name, labels,
        os.path.join(fig_dir, f'confusion_matrices_{timestamp}.png')
    )
    print(f"✓ Saved confusion matrices")

    # Performance comparison
    create_performance_comparison(
        model1_results, model2_results,
        model1_name, model2_name,
        os.path.join(fig_dir, f'performance_comparison_{timestamp}.png')
    )
    print(f"✓ Saved performance comparison")

    # Per-class comparison
    create_per_class_comparison(
        model1_results, model2_results,
        model1_name, model2_name, labels,
        os.path.join(fig_dir, f'per_class_f1_{timestamp}.png')
    )
    print(f"✓ Saved per-class F1 scores")

    print(f"\n✓ All visualizations saved to: {fig_dir}\n")
