#!/usr/bin/env python3
"""
Generate Dataset Analysis Visualizations

Analyzes clean.csv and generates distribution charts:
- Priority distribution (class balance)
- Text length distributions (characters & words)
- Sample statistics

Usage:
    python scripts/generate_dataset_viz.py
    python scripts/generate_dataset_viz.py --output visualizations/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def create_priority_distribution(df: pd.DataFrame, output_path: str):
    """Create priority class distribution chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count and percentage
    priority_counts = df['priority'].value_counts().sort_index()
    total = len(df)

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(
        priority_counts.index.str.capitalize(),
        priority_counts.values,
        color=colors,
        edgecolor='black',
        linewidth=1.2
    )

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, priority_counts.values)):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + total * 0.01,
            f'{count:,}\n({percentage:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_xlabel('Priority Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Tickets', fontsize=12, fontweight='bold')
    ax.set_title(f'Urgency Priority Distribution\nTotal: {total:,} tickets',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_character_distribution(df: pd.DataFrame, output_path: str):
    """Create character count distribution histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate character counts
    char_counts = df['body'].str.len()

    # Histogram
    ax1.hist(char_counts, bins=50, color='#3498db',
             edgecolor='black', alpha=0.7)
    ax1.axvline(char_counts.mean(), color='#e74c3c', linestyle='--',
                linewidth=2, label=f'Mean: {char_counts.mean():.0f}')
    ax1.axvline(char_counts.median(), color='#2ecc71', linestyle='--',
                linewidth=2, label=f'Median: {char_counts.median():.0f}')
    ax1.set_xlabel('Character Count', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Character Count Distribution',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Box plot by priority
    priority_order = ['low', 'medium', 'high']
    df_with_chars = df.copy()
    df_with_chars['char_count'] = char_counts

    box_data = [df_with_chars[df_with_chars['priority'] == p]['char_count'].values
                for p in priority_order]

    bp = ax2.boxplot(box_data, labels=[p.capitalize() for p in priority_order],
                     patch_artist=True, showmeans=True)

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xlabel('Priority Level', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Character Count', fontsize=11, fontweight='bold')
    ax2.set_title('Character Count by Priority',
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_word_distribution(df: pd.DataFrame, output_path: str):
    """Create word count distribution histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate word counts
    word_counts = df['body'].str.split().str.len()

    # Histogram
    ax1.hist(word_counts, bins=50, color='#9b59b6',
             edgecolor='black', alpha=0.7)
    ax1.axvline(word_counts.mean(), color='#e74c3c', linestyle='--',
                linewidth=2, label=f'Mean: {word_counts.mean():.0f}')
    ax1.axvline(word_counts.median(), color='#2ecc71', linestyle='--',
                linewidth=2, label=f'Median: {word_counts.median():.0f}')
    ax1.set_xlabel('Word Count', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Word Count Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Box plot by priority
    priority_order = ['low', 'medium', 'high']
    df_with_words = df.copy()
    df_with_words['word_count'] = word_counts

    box_data = [df_with_words[df_with_words['priority'] == p]['word_count'].values
                for p in priority_order]

    bp = ax2.boxplot(box_data, labels=[p.capitalize() for p in priority_order],
                     patch_artist=True, showmeans=True)

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xlabel('Priority Level', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Word Count', fontsize=11, fontweight='bold')
    ax2.set_title('Word Count by Priority', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_statistics_summary(df: pd.DataFrame, output_path: str):
    """Create comprehensive statistics visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate metrics
    char_counts = df['body'].str.len()
    word_counts = df['body'].str.split().str.len()

    # 1. Dataset Overview
    ax1.axis('off')
    overview_text = f"""
DATASET OVERVIEW
{'='*40}

Total Samples: {len(df):,}

Class Distribution:
  LOW:    {(df['priority'] == 'low').sum():,} ({(df['priority'] == 'low').sum()/len(df)*100:.1f}%)
  MEDIUM: {(df['priority'] == 'medium').sum():,} ({(df['priority'] == 'medium').sum()/len(df)*100:.1f}%)
  HIGH:   {(df['priority'] == 'high').sum():,} ({(df['priority'] == 'high').sum()/len(df)*100:.1f}%)

Text Length (Characters):
  Mean:   {char_counts.mean():.1f}
  Median: {char_counts.median():.1f}
  Min:    {char_counts.min()}
  Max:    {char_counts.max():,}
  95th:   {char_counts.quantile(0.95):.0f}

Text Length (Words):
  Mean:   {word_counts.mean():.1f}
  Median: {word_counts.median():.1f}
  Min:    {word_counts.min()}
  Max:    {word_counts.max():,}
  95th:   {word_counts.quantile(0.95):.0f}
    """
    ax1.text(0.1, 0.5, overview_text, fontsize=10, family='monospace',
             verticalalignment='center')
    ax1.set_title('Dataset Statistics', fontsize=14, fontweight='bold', pad=20)

    # 2. Percentile distribution (characters)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    char_percentiles = [char_counts.quantile(p/100) for p in percentiles]

    ax2.barh(percentiles, char_percentiles, color='#3498db', edgecolor='black')
    for i, (p, val) in enumerate(zip(percentiles, char_percentiles)):
        ax2.text(val + char_counts.max()*0.02, p, f'{val:.0f}',
                 va='center', fontweight='bold')
    ax2.set_xlabel('Character Count', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Percentile', fontsize=11, fontweight='bold')
    ax2.set_title('Character Count Percentiles',
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # 3. Priority vs Length correlation
    priority_stats = df.groupby('priority').agg({
        'body': lambda x: x.str.len().mean()
    }).reindex(['low', 'medium', 'high'])

    bars = ax3.bar(
        ['Low', 'Medium', 'High'],
        priority_stats['body'],
        color=['#3498db', '#e74c3c', '#2ecc71'],
        edgecolor='black'
    )

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.0f}', ha='center', fontweight='bold')

    ax3.set_ylabel('Avg Character Count', fontsize=11, fontweight='bold')
    ax3.set_title('Average Text Length by Priority',
                  fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Cumulative distribution
    sorted_chars = np.sort(char_counts)
    cumulative = np.arange(1, len(sorted_chars) + 1) / len(sorted_chars) * 100

    ax4.plot(sorted_chars, cumulative, color='#3498db', linewidth=2)
    ax4.axhline(95, color='#e74c3c', linestyle='--', label='95th percentile')
    ax4.axvline(char_counts.quantile(0.95), color='#e74c3c', linestyle='--')
    ax4.set_xlabel('Character Count', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative Percentage', fontsize=11, fontweight='bold')
    ax4.set_title('Cumulative Distribution of Text Length',
                  fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset analysis visualizations from clean.csv'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/clean.csv',
        help='Path to clean.csv (default: data/clean.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations',
        help='Output directory for visualizations (default: visualizations/)'
    )

    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GENERATING DATASET VISUALIZATIONS")
    print("="*70)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}/\n")

    # Load data
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} samples\n")

    # Generate visualizations
    print("Generating visualizations...")

    create_priority_distribution(
        df,
        str(output_dir / 'priority_distribution.png')
    )

    create_character_distribution(
        df,
        str(output_dir / 'character_count_distribution.png')
    )

    create_word_distribution(
        df,
        str(output_dir / 'word_count_distribution.png')
    )

    create_statistics_summary(
        df,
        str(output_dir / 'dataset_statistics.png')
    )

    print("\n" + "="*70)
    print(f"✅ All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - priority_distribution.png")
    print("  - character_count_distribution.png")
    print("  - word_count_distribution.png")
    print("  - dataset_statistics.png")
    print("="*70)


if __name__ == '__main__':
    main()
